#!/usr/bin/env python3
"""
CPU smoke: multi-scale field stacks → conditional multi-tower VAE → prior samples.

Scientific goal: encode snapshot → global ``z``; sample ``z ~ N(0,I) | θ`` to
generate non-equilibrium field ICs (barred vs quiet), preserving dens+moment
recon quality from the U-Net track.

    . .venv/bin/activate
    OMP_NUM_THREADS=6 python scripts/smoke_field_vae.py
    OMP_NUM_THREADS=6 python scripts/smoke_field_vae.py --with-evolve
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("OMP_NUM_THREADS", "8")
torch.set_num_threads(8)

from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_from_record
from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
from galacticsics.ml.fields.binning import MultiScaleSliceConfig, bin_multiscale_slice_stacks, scale_summary
from galacticsics.ml.fields.dataset import MultiScaleFieldDataset, collate_multiscale_batch
from galacticsics.ml.fields.frame import prepare_shared_frame
from galacticsics.ml.fields.normalize import denormalize_stack, normalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from galacticsics.ml.fields.vae import FieldVAEConfig, MultiTowerSliceVAE, field_vae_loss
from galacticsics.ml.morton.index import SnapshotRecord, resolve_snapshot_t_gyr
from galacticsics.ml.morton.tokenize import _component_ids
from ntropy.analysis.disk_density import bin_plane_density, disk_azimuthal_fourier


MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
WORK = Path("runs/mw_morton_corpus_v2")
DEVICE = "cpu"
SEED = 0
COUNT_FRACTIONS = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}

# Crisp AE reference (128² dens+moment, Fourier off) — morphology bar.
CRISP_AE = {
    "bar_a2_data": 0.37,
    "bar_a2_resampled": 0.33,
    "quiet_a2_resampled": 0.016,
    "note": "crisp_2026-07-24 U-Net AE 128²",
}

EXAMPLES = [
    {
        "name": "bar_54a8_late",
        "run": "54a8faf836a0",
        "dump": "evolution/particles/step_001700.npz",
        "label": "barred (A₂≈0.5)",
    },
    {
        "name": "quiet_ic_081e",
        "run": "081ed8af4b2b",
        "dump": "ic_state.npz",
        "label": "quiet IC (A₂≈0.01)",
    },
]


def _a2_median(pos, mass) -> float:
    out = disk_azimuthal_fourier(
        pos, mass, m=2, r_max=12.0, n_bins=12, z_max=0.5, min_count=10
    )
    return float(out["a_m_over_a0_median"])


def _load_shared_frame(path: Path):
    with np.load(path, allow_pickle=True) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        tags = data["tags"] if "tags" in data.files else None
        type_id = data["type_id"] if "type_id" in data.files else None
    cid = _component_ids(tags, type_id, pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    return pos, vel, mass, cid


def _theta_for_example(manifest: Path, ex: dict) -> np.ndarray:
    raw = json.loads(manifest.read_text())
    for r in raw["records"]:
        rec = SnapshotRecord(**r)
        if rec.run_hash == ex["run"] and ex["dump"] in rec.path:
            t_gyr = resolve_snapshot_t_gyr(rec.path, rec.t_gyr)
            return theta_from_record(rec.theta, t_gyr=t_gyr)
    for r in raw["records"]:
        rec = SnapshotRecord(**r)
        if rec.run_hash == ex["run"]:
            t_gyr = resolve_snapshot_t_gyr(rec.path, rec.t_gyr)
            return theta_from_record(rec.theta, t_gyr=t_gyr)
    raise SystemExit(f"no θ for example {ex['name']} run={ex['run']}")


def _disk_dens_collapse(stack: np.ndarray, n_z: int, n_mom: int) -> np.ndarray:
    dens = np.zeros(stack.shape[-2:], dtype=np.float64)
    for iz in range(n_z):
        dens += np.maximum(stack[iz * n_mom], 0.0)
    return dens


def _plot_panel(out_path, *, title, panels, labels):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, img, lab in zip(axes, panels, labels):
        pos = img[img > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        ax.imshow(
            np.log1p(np.maximum(img, 0.0)),
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=np.log1p(vmax),
        )
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _evolve_bh(parts, *, end_gyr, dt, omp_threads, timeout_s=600.0):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    os.environ["OMP_NUM_THREADS"] = str(max(1, int(omp_threads)))
    n_steps = max(1, int(np.ceil(float(end_gyr) / max(float(dt), 1e-9))))
    pos0 = np.asarray(parts["pos"], dtype=np.float64)
    vel0 = np.asarray(parts["vel"], dtype=np.float64)
    mass = np.asarray(parts["mass"], dtype=np.float64)
    eps = np.asarray(parts["eps"], dtype=np.float64)

    def _forces(p):
        try:
            from ntropy.forces.bhtree_c import compute_forces_bh_c, extension_available

            if extension_available():
                return compute_forces_bh_c(p, mass, eps, theta=0.8)
        except Exception:  # noqa: BLE001
            pass
        import importlib

        bhtree = importlib.import_module("ntropy.forces.bhtree")
        return bhtree.compute_forces_bh(p, mass, eps, theta=0.8)

    def _run():
        t0 = time.time()
        pos = pos0.copy()
        vel = vel0.copy()
        acc = _forces(pos)
        vel = vel + 0.5 * dt * acc
        for _ in range(n_steps):
            pos = pos + dt * vel
            acc = _forces(pos)
            vel = vel + dt * acc
        vel = vel - 0.5 * dt * acc
        return pos, vel, time.time() - t0

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            pos_f, _vel_f, wall = ex.submit(_run).result(timeout=float(timeout_s))
    except FuturesTimeout:
        return {"ok": False, "timed_out": True, "error": f"timeout {timeout_s}s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "timed_out": False, "error": f"{type(exc).__name__}: {exc}"}

    com0 = np.average(pos0, axis=0, weights=mass)
    com1 = np.average(pos_f, axis=0, weights=mass)
    return {
        "ok": True,
        "wall_s": wall,
        "com_drift_kpc": float(np.linalg.norm(com1 - com0)),
        "pos_final": np.asarray(pos_f, dtype=np.float64),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="artifact dir (default: runs/ml/field_maps/vae_latent_YYYY-MM-DD)",
    )
    p.add_argument(
        "--preset",
        choices=("smoke", "progressive", "baseline32"),
        default="progressive",
        help="grid preset (default progressive — match crisp AE disk res)",
    )
    p.add_argument("--disk-n-pix", type=int, default=128)
    p.add_argument("--moment-set", choices=("base", "disp", "full"), default="disp")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max-snap", type=int, default=28)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=3e-5, help="cosine floor (0 disables)")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=14)
    p.add_argument("--dens-weight", type=float, default=4.0)
    p.add_argument("--moment-weight", type=float, default=6.0)
    p.add_argument(
        "--a2-weight",
        type=float,
        default=0.0,
        help="radial A_m(R) weight (0=off; Fourier can invent quiet bars)",
    )
    p.add_argument("--beta", type=float, default=5e-4, help="KL weight")
    p.add_argument("--latent-dim", type=int, default=192)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--bottleneck-channels", type=int, default=64)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--skip-dropout", type=float, default=1.0)
    p.add_argument("--prior-decode-weight", type=float, default=0.0)
    p.add_argument("--skip-recon-weight", type=float, default=0.1)
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--n-evolve", type=int, default=30_000)
    p.add_argument("--evolve-gyr", type=float, default=0.02)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--omp-threads", type=int, default=6)
    p.add_argument("--with-evolve", action="store_true")
    p.add_argument("--n-prior-samples", type=int, default=2)
    p.add_argument("--note", type=str, default="", help="free-text note written to NOTES.md")
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="warm-start weights from a prior multitower_slice_vae.pt (matching arch)",
    )
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--work", type=Path, default=WORK)
    return p.parse_args()


def _rss_mb() -> float:
    import resource

    # Linux: ru_maxrss is kB
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _make_scheduler(opt, *, epochs: int, warmup: int, lr: float, min_lr: float):
    """Linear warmup then cosine decay to ``min_lr`` (identity if min_lr<=0)."""
    if min_lr <= 0.0 or epochs <= 1:
        return None
    warm = max(0, int(warmup))
    import math

    def _lr_lambda(epoch_idx: int) -> float:
        # epoch_idx is 0-based from scheduler.step() after each epoch
        e = epoch_idx + 1
        if warm > 0 and e <= warm:
            return max(e / float(warm), 1e-3)
        # cosine from lr → min_lr over remaining epochs
        remain = max(1, epochs - warm)
        t = min(max(e - warm, 0), remain) / float(remain)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        # scale so final multiplier = min_lr / lr
        floor = float(min_lr) / float(lr)
        return floor + (1.0 - floor) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)


def _build_cfg(args) -> MultiScaleSliceConfig:
    if args.preset == "baseline32":
        return MultiScaleSliceConfig.baseline_32_defaults(
            include_potential=False, moment_set=args.moment_set
        )
    if args.preset == "progressive":
        return MultiScaleSliceConfig.progressive_defaults(
            disk_n_pix=args.disk_n_pix,
            include_potential=False,
            moment_set=args.moment_set,
        )
    return MultiScaleSliceConfig.smoke_defaults(
        include_potential=False, moment_set=args.moment_set
    )


def main() -> None:
    args = parse_args()
    out = args.out or Path(f"runs/ml/field_maps/vae_latent_{date.today().isoformat()}")
    omp = max(1, int(args.omp_threads))
    os.environ["OMP_NUM_THREADS"] = str(omp)
    torch.set_num_threads(omp)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    print(f"artifacts → {out}  (OMP_NUM_THREADS={omp})", flush=True)

    cfg = _build_cfg(args)
    print("multi-scale grids:")
    for row in scale_summary(cfg):
        print(f"  {row}")
    if not args.manifest.is_file():
        raise SystemExit(f"missing manifest {args.manifest}")

    ds = MultiScaleFieldDataset(
        args.manifest,
        cfg=cfg,
        split=None,
        max_snapshots=args.max_snap,
        seed=SEED,
        augment=True,
        include_potential=False,
        theta_keys=DEFAULT_THETA_KEYS,
    )
    print(f"dataset size={len(ds)}; preloading…")
    t0 = time.time()
    ds.preload()
    print(f"  preload {time.time() - t0:.1f}s; fitting norm stats…")
    stats = ds.fit_norm_stats(n_samples=min(10, len(ds)))
    dens_idx = ds.dens_indices()

    def _collate(batch):
        b = collate_multiscale_batch(batch)
        stacks = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in b["stacks"].items()
        }
        theta = torch.as_tensor(b["theta"], dtype=torch.float32)
        return {"stacks": stacks, "theta": theta}

    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=_collate
    )
    vae_cfg = FieldVAEConfig(
        latent_dim=args.latent_dim,
        theta_dim=len(DEFAULT_THETA_KEYS),
        base_channels=args.base_channels,
        bottleneck_channels=args.bottleneck_channels,
        enc_grid=args.enc_grid,
        beta=args.beta,
        free_bits=0.05,
        skip_dropout=args.skip_dropout,
        prior_decode_weight=args.prior_decode_weight,
        skip_recon_weight=args.skip_recon_weight,
    )
    model = MultiTowerSliceVAE(cfg, vae_cfg=vae_cfg, include_potential=False).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(
            f"warm-start {args.init_from}: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    rss0 = _rss_mb()
    print(
        f"MultiTowerSliceVAE params={n_params:,} latent_dim={args.latent_dim} "
        f"β={args.beta} lr={args.lr} a2_w={args.a2_weight} rss={rss0:.0f}MB",
        flush=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = _make_scheduler(
        opt,
        epochs=args.epochs,
        warmup=args.warmup_epochs,
        lr=args.lr,
        min_lr=args.min_lr,
    )
    comp_w = {"disk": 2.5, "bulge": 1.0, "halo": 0.4}

    best = float("inf")
    best_state: dict | None = None
    stale = 0
    history: list[dict] = []
    t_train0 = time.time()

    for epoch in range(1, args.epochs + 1):
        ds.set_epoch(epoch)
        model.train()
        losses, dens_l, mom_l, kl_l, prior_l = [], [], [], [], []
        for batch in loader:
            stacks = {k: v.to(DEVICE) for k, v in batch["stacks"].items()}
            theta = batch["theta"].to(DEVICE)
            metrics = field_vae_loss(
                model,
                stacks,
                theta,
                dens_indices=dens_idx,
                dens_weight=args.dens_weight,
                moment_weight=args.moment_weight,
                a2_weight=args.a2_weight,
                component_weights=comp_w,
            )
            loss = metrics["loss"]
            if not torch.isfinite(loss):
                print("  skip non-finite batch", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            dens_l.append(float(metrics["mse_dens"].detach()))
            mom_l.append(float(metrics["mse_mom"].detach()))
            kl_l.append(float(metrics["kl"].detach()))
            if "prior_recon" in metrics:
                prior_l.append(float(metrics["prior_recon"].detach()))
        if not losses:
            print(f"  epoch {epoch}  no finite batches", flush=True)
            continue
        if sched is not None:
            sched.step()
        cur_lr = float(opt.param_groups[0]["lr"])
        mean_loss = float(np.mean(losses))
        mean_dens = float(np.mean(dens_l))
        mean_mom = float(np.mean(mom_l))
        mean_kl = float(np.mean(kl_l))
        mean_prior = float(np.mean(prior_l)) if prior_l else float("nan")
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "mse_dens": mean_dens,
                "mse_mom": mean_mom,
                "kl": mean_kl,
                "prior_recon": mean_prior,
                "lr": cur_lr,
                "rss_mb": _rss_mb(),
            }
        )
        print(
            f"  epoch {epoch}  loss={mean_loss:.4f}  dens={mean_dens:.4f}  "
            f"mom={mean_mom:.4f}  kl={mean_kl:.4f}  prior_rec={mean_prior:.4f}  "
            f"lr={cur_lr:.2e}  rss={_rss_mb():.0f}MB",
            flush=True,
        )
        if mean_loss + 1e-4 < best:
            best = mean_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "vae_cfg": vae_cfg.__dict__,
                    "cfg_grids": [g.__dict__ for g in cfg.grids],
                    "norm": {k: v.__dict__ for k, v in stats.items()},
                    "theta_keys": list(DEFAULT_THETA_KEYS),
                    "args": vars(args) | {"out": str(out), "manifest": str(args.manifest)},
                },
                out / "multitower_slice_vae.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch}")
                break
    train_wall_s = time.time() - t_train0
    peak_rss_mb = _rss_mb()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    disk_grid = cfg.grid_for("disk")
    n_mom = disk_grid.n_mom
    results = []

    def _resample_from_recon(recon: dict[str, np.ndarray], mass_true: dict[str, float], n: int):
        moment_maps = {
            g.name: recon[g.name][: g.n_moment_channels]
            for g in cfg.grids
            if g.name in recon
        }
        return resample_particles_from_multiscale(
            moment_maps,
            cfg=MultiScaleSliceConfig(grids=cfg.grids, include_potential=False),
            n_particles=n,
            mass_total_per_component=mass_true,
            count_fractions=COUNT_FRACTIONS,
            rng=rng,
            sample_dispersion=True,
        )

    # --- Posterior recon on barred / quiet examples ---
    for ex in EXAMPLES:
        path = args.work / ex["run"] / ex["dump"]
        pos, vel, mass, cid = _load_shared_frame(path)
        disk = cid == 0
        a2_data = _a2_median(pos[disk], mass[disk]) if np.any(disk) else float("nan")
        theta_np = _theta_for_example(args.manifest, ex)
        binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
        maps = {k: v[0] for k, v in binned.items()}
        normed = {k: normalize_stack(maps[k], stats[k]) for k in maps}
        with torch.no_grad():
            batch_t = {
                k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in normed.items()
            }
            theta_t = torch.as_tensor(theta_np[None], dtype=torch.float32)
            out_fwd = model(batch_t, theta_t, sample_posterior=False)
            # Evaluate generative path (matches sampling).
            gen = model.decode(
                out_fwd["mu"],
                theta_t,
                bottlenecks=None,
                skips=None,
                use_encoder_skips=False,
                target_shapes={k: (v.shape[-2], v.shape[-1]) for k, v in batch_t.items()},
            )
            pred_np = {k: v.cpu().numpy()[0] for k, v in gen.items()}
            z_mu = out_fwd["mu"].cpu().numpy()[0]
        recon = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}
        mse_dens = float(
            np.mean(
                (pred_np["disk"][dens_idx["disk"]] - normed["disk"][dens_idx["disk"]]) ** 2
            )
        )
        data_dens = _disk_dens_collapse(maps["disk"], disk_grid.n_z, n_mom)
        recon_dens = _disk_dens_collapse(recon["disk"], disk_grid.n_z, n_mom)
        mass_true = {
            "disk": float(mass[cid == 0].sum()) if np.any(cid == 0) else 0.0,
            "halo": float(mass[cid == 1].sum()) if np.any(cid == 1) else 0.0,
            "bulge": float(mass[cid == 2].sum()) if np.any(cid == 2) else 0.0,
        }
        parts = _resample_from_recon(recon, mass_true, args.n_resample)
        disk_rec = parts["component_id"] == 0
        a2_rec = (
            _a2_median(parts["pos"][disk_rec], parts["mass"][disk_rec])
            if np.any(disk_rec)
            else float("nan")
        )
        part_map = bin_plane_density(
            parts["pos"][disk_rec],
            parts["mass"][disk_rec],
            axes=(0, 1),
            n_bins=disk_grid.n_pix,
            half_extent=disk_grid.r_max,
        ).density
        _plot_panel(
            out / f"{ex['name']}_recon_panel.png",
            title=f"VAE recon · {ex['label']}",
            panels=(data_dens, recon_dens, part_map),
            labels=(
                f"data\nA₂≈{a2_data:.2f}",
                f"VAE recon\nmse_dens={mse_dens:.4f}",
                f"resampled\nA₂≈{a2_rec:.2f}",
            ),
        )
        results.append(
            {
                "name": ex["name"],
                "kind": "recon",
                "a2_data": a2_data,
                "a2_resampled": a2_rec,
                "mse_dens": mse_dens,
                "z_mu_norm": float(np.linalg.norm(z_mu)),
                "theta": theta_np.tolist(),
            }
        )
        print(
            f"recon {ex['name']}: A₂ data={a2_data:.3f} → resampled={a2_rec:.3f} "
            f"mse_dens={mse_dens:.4f}",
            flush=True,
        )

    # --- Prior samples + encode→z decode (no skips) + latent interpolation ---
    prior_results = []
    encoded_z: dict[str, torch.Tensor] = {}
    encoded_theta: dict[str, torch.Tensor] = {}
    for ex in EXAMPLES:
        path = args.work / ex["run"] / ex["dump"]
        pos, vel, mass, cid = _load_shared_frame(path)
        mass_true = {
            "disk": float(mass[cid == 0].sum()) if np.any(cid == 0) else 1.0,
            "halo": float(mass[cid == 1].sum()) if np.any(cid == 1) else 1.0,
            "bulge": float(mass[cid == 2].sum()) if np.any(cid == 2) else 1.0,
        }
        theta_np = _theta_for_example(args.manifest, ex)
        theta_t = torch.as_tensor(theta_np[None], dtype=torch.float32)
        print(f"θ[{ex['name']}] t_gyr={theta_np[-1]:.3f} disk.mass={theta_np[0]:.2f}", flush=True)

        binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
        maps = {k: v[0] for k, v in binned.items()}
        normed = {k: normalize_stack(maps[k], stats[k]) for k in maps}
        with torch.no_grad():
            batch_t = {
                k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in normed.items()
            }
            enc = model(batch_t, theta_t, sample_posterior=False)
            z_mu = enc["mu"]
            encoded_z[ex["name"]] = z_mu.detach().clone()
            encoded_theta[ex["name"]] = theta_t.detach().clone()
            # Decode μ through the *prior* path (no encoder skips) — tests that z
            # alone carries morphology for IC sampling.
            from_z = model.decode(
                z_mu,
                theta_t,
                bottlenecks=None,
                skips=None,
                use_encoder_skips=False,
                target_shapes={k: (v.shape[-2], v.shape[-1]) for k, v in batch_t.items()},
            )
            pred_np = {k: v.cpu().numpy()[0] for k, v in from_z.items()}
        recon = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}
        dens_z = _disk_dens_collapse(recon["disk"], disk_grid.n_z, n_mom)
        parts = _resample_from_recon(recon, mass_true, args.n_resample)
        disk_rec = parts["component_id"] == 0
        a2_z = (
            _a2_median(parts["pos"][disk_rec], parts["mass"][disk_rec])
            if np.any(disk_rec)
            else float("nan")
        )
        part_map = bin_plane_density(
            parts["pos"][disk_rec],
            parts["mass"][disk_rec],
            axes=(0, 1),
            n_bins=disk_grid.n_pix,
            half_extent=disk_grid.r_max,
        ).density

        panels = [dens_z, part_map]
        labels = [
            f"decode(μ) no-skip\n|μ|={float(z_mu.norm()):.2f}",
            f"particles←μ\nA₂≈{a2_z:.2f}",
        ]
        prior_results.append(
            {
                "name": ex["name"],
                "sample": "mu",
                "kind": "decode_mu",
                "a2_resampled": a2_z,
                "z_norm": float(z_mu.norm()),
                "theta_label": ex["label"],
                "t_gyr": float(theta_np[-1]),
                "evolve": None,
            }
        )

        for i in range(int(args.n_prior_samples)):
            # Mild prior: sample near the encoded μ so morphology is reachable,
            # plus one pure N(0,I) draw (i==0) for the unconditional prior.
            if i == 0:
                z = torch.randn(1, args.latent_dim)
                tag = "N(0,I)"
            else:
                z = z_mu + 0.35 * torch.randn_like(z_mu)
                tag = "μ+0.35ε"
            with torch.no_grad():
                samp = model.sample(theta_t, z=z)
                pred_np = {k: v.cpu().numpy()[0] for k, v in samp.items()}
            recon = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}
            dens = _disk_dens_collapse(recon["disk"], disk_grid.n_z, n_mom)
            parts = _resample_from_recon(recon, mass_true, args.n_resample)
            disk_rec = parts["component_id"] == 0
            a2 = (
                _a2_median(parts["pos"][disk_rec], parts["mass"][disk_rec])
                if np.any(disk_rec)
                else float("nan")
            )
            part_map = bin_plane_density(
                parts["pos"][disk_rec],
                parts["mass"][disk_rec],
                axes=(0, 1),
                n_bins=disk_grid.n_pix,
                half_extent=disk_grid.r_max,
            ).density
            panels.extend([dens, part_map])
            labels.extend(
                [
                    f"prior {tag}\n|z|={float(z.norm()):.2f}",
                    f"particles\nA₂≈{a2:.2f}",
                ]
            )
            evo_info = None
            if args.with_evolve and i == 0:
                evo_parts = _resample_from_recon(recon, mass_true, args.n_evolve)
                evo_info = _evolve_bh(
                    evo_parts,
                    end_gyr=args.evolve_gyr,
                    dt=args.dt,
                    omp_threads=omp,
                )
                if evo_info.get("ok"):
                    epos = evo_info["pos_final"]
                    edisk = evo_parts["component_id"] == 0
                    a2_evo = _a2_median(epos[edisk], evo_parts["mass"][edisk])
                    evo_map = bin_plane_density(
                        epos[edisk],
                        evo_parts["mass"][edisk],
                        axes=(0, 1),
                        n_bins=disk_grid.n_pix,
                        half_extent=disk_grid.r_max,
                    ).density
                    panels.append(evo_map)
                    labels.append(
                        f"evolved {args.evolve_gyr} Gyr\nA₂≈{a2_evo:.2f} "
                        f"COMΔ={evo_info['com_drift_kpc']:.3f}"
                    )
                    evo_info = {
                        "ok": True,
                        "a2_evolved": a2_evo,
                        "com_drift_kpc": evo_info["com_drift_kpc"],
                        "wall_s": evo_info["wall_s"],
                    }
            prior_results.append(
                {
                    "name": ex["name"],
                    "sample": i,
                    "kind": "prior",
                    "tag": tag,
                    "a2_resampled": a2,
                    "z_norm": float(z.norm()),
                    "theta_label": ex["label"],
                    "t_gyr": float(theta_np[-1]),
                    "evolve": evo_info,
                }
            )
        _plot_panel(
            out / f"{ex['name']}_prior_panel.png",
            title=f"Latent samples · θ from {ex['label']} (t={theta_np[-1]:.2f} Gyr)",
            panels=tuple(panels),
            labels=tuple(labels),
        )
        print(
            f"latent {ex['name']}: decode(μ) A₂={a2_z:.3f}; "
            f"priors={[round(r['a2_resampled'], 3) for r in prior_results if r['name']==ex['name'] and r['kind']=='prior']}",
            flush=True,
        )

    # Interpolate z between quiet μ and barred μ (same decode path).
    if "bar_54a8_late" in encoded_z and "quiet_ic_081e" in encoded_z:
        z0 = encoded_z["quiet_ic_081e"]
        z1 = encoded_z["bar_54a8_late"]
        # Use barred θ / mass mix as the non-eq target context.
        theta_bar = encoded_theta["bar_54a8_late"]
        path = args.work / EXAMPLES[0]["run"] / EXAMPLES[0]["dump"]
        pos, vel, mass, cid = _load_shared_frame(path)
        mass_true = {
            "disk": float(mass[cid == 0].sum()) if np.any(cid == 0) else 1.0,
            "halo": float(mass[cid == 1].sum()) if np.any(cid == 1) else 1.0,
            "bulge": float(mass[cid == 2].sum()) if np.any(cid == 2) else 1.0,
        }
        alphas = [0.0, 0.35, 0.65, 1.0]
        panels = []
        labels = []
        interp_a2 = []
        for a in alphas:
            z = (1.0 - a) * z0 + a * z1
            with torch.no_grad():
                samp = model.sample(theta_bar, z=z)
                pred_np = {k: v.cpu().numpy()[0] for k, v in samp.items()}
            recon = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}
            dens = _disk_dens_collapse(recon["disk"], disk_grid.n_z, n_mom)
            parts = _resample_from_recon(recon, mass_true, args.n_resample)
            disk_rec = parts["component_id"] == 0
            a2 = (
                _a2_median(parts["pos"][disk_rec], parts["mass"][disk_rec])
                if np.any(disk_rec)
                else float("nan")
            )
            interp_a2.append(a2)
            part_map = bin_plane_density(
                parts["pos"][disk_rec],
                parts["mass"][disk_rec],
                axes=(0, 1),
                n_bins=disk_grid.n_pix,
                half_extent=disk_grid.r_max,
            ).density
            panels.extend([dens, part_map])
            labels.extend([f"α={a:.2f} dens", f"A₂≈{a2:.2f}"])
        _plot_panel(
            out / "latent_interp_quiet_to_bar.png",
            title="Latent interp quiet μ → bar μ (barred θ)",
            panels=tuple(panels),
            labels=tuple(labels),
        )
        prior_results.append(
            {
                "name": "interp_quiet_to_bar",
                "kind": "interpolate",
                "alphas": alphas,
                "a2_resampled": interp_a2,
            }
        )
        print(f"interp quiet→bar A₂={ [round(a, 3) for a in interp_a2] }", flush=True)

    # Summary: recon / decode(μ) / prior for bar vs quiet.
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bar_recon = next(r["a2_resampled"] for r in results if r["name"].startswith("bar"))
    quiet_recon = next(r["a2_resampled"] for r in results if r["name"].startswith("quiet"))
    bar_mu = next(
        (r["a2_resampled"] for r in prior_results if r["name"].startswith("bar") and r["kind"] == "decode_mu"),
        float("nan"),
    )
    quiet_mu = next(
        (
            r["a2_resampled"]
            for r in prior_results
            if r["name"].startswith("quiet") and r["kind"] == "decode_mu"
        ),
        float("nan"),
    )
    bar_priors = [
        r["a2_resampled"]
        for r in prior_results
        if r["name"].startswith("bar") and r["kind"] == "prior"
    ]
    quiet_priors = [
        r["a2_resampled"]
        for r in prior_results
        if r["name"].startswith("quiet") and r["kind"] == "prior"
    ]
    labels_x = ["bar recon", "bar μ→dec", "bar prior", "quiet recon", "quiet μ→dec", "quiet prior"]
    vals = [
        bar_recon,
        bar_mu,
        float(np.nanmean(bar_priors)) if bar_priors else float("nan"),
        quiet_recon,
        quiet_mu,
        float(np.nanmean(quiet_priors)) if quiet_priors else float("nan"),
    ]
    ax.bar(np.arange(6), vals, color=["#8c8c8c", "#c45c26", "#e8a87c", "#8c8c8c", "#2a6f97", "#7fbadc"])
    ax.axhline(CRISP_AE["bar_a2_resampled"], color="#c45c26", ls="--", lw=1, label="crisp AE bar")
    ax.axhline(CRISP_AE["quiet_a2_resampled"], color="#2a6f97", ls="--", lw=1, label="crisp AE quiet")
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(labels_x, fontsize=8)
    ax.set_ylabel("A₂ median (resampled)")
    ax.set_title("Field VAE: recon vs latent decode / prior")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "vae_vs_ae_a2.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot([h["epoch"] for h in history], [h["loss"] for h in history], label="loss")
    ax.plot([h["epoch"] for h in history], [h["kl"] for h in history], label="KL")
    ax.set_xlabel("epoch")
    ax.legend(frameon=False)
    ax.set_title("Field VAE train curve")
    fig.tight_layout()
    fig.savefig(out / "train_curve.png", dpi=140)
    plt.close(fig)

    (out / "history.json").write_text(json.dumps(history, indent=2))
    bar_prior_mean = float(np.nanmean(bar_priors)) if bar_priors else float("nan")
    quiet_prior_mean = float(np.nanmean(quiet_priors)) if quiet_priors else float("nan")
    bar_recon_a2 = bar_recon
    quiet_recon_a2 = quiet_recon
    verdict = {
        "best_loss": best,
        "n_params": n_params,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "epochs_ran": len(history),
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "dens_weight": args.dens_weight,
        "moment_weight": args.moment_weight,
        "a2_weight": args.a2_weight,
        "preset": args.preset,
        "disk_n_pix": disk_grid.n_pix,
        "train_wall_s": train_wall_s,
        "peak_rss_mb": peak_rss_mb,
        "recon": results,
        "prior": prior_results,
        "crisp_ae_ref": CRISP_AE,
        "hires_ae_ref": {
            "bar_a2_resampled": 0.26,
            "quiet_a2_resampled": 0.013,
            "note": "64² U-Net AE (matched resolution)",
        },
        "morphology_mu_separated": bool(
            np.isfinite(bar_mu)
            and np.isfinite(quiet_mu)
            and bar_mu > quiet_mu + 0.08
            and quiet_mu < 0.12
        ),
        "morphology_prior_separated": bool(
            bar_prior_mean > quiet_prior_mean + 0.05 and quiet_prior_mean < 0.12
        ),
        "bar_recon_competitive": bool(
            np.isfinite(bar_recon_a2) and bar_recon_a2 >= 0.20
        ),
        "how_to_sample": (
            "API: z,θ → fields → particles. "
            "Encode snap → μ,logσ; sample z~N(μ,σ) or interpolate μ's; "
            "or z~N(0,I)|θ. fields=model.sample(theta,z=z); "
            "denorm → resample_particles_from_multiscale (count mix 4:2:1). "
            "Shared COM required (prepare_shared_frame)."
        ),
        "user_note": args.note,
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))

    notes = [
        f"# Field VAE run — {out.name}",
        "",
        f"- **latent_dim**: {args.latent_dim}",
        f"- **disk**: {disk_grid.n_pix}² (`{args.preset}`)",
        f"- **lr / min_lr / warmup**: {args.lr} / {args.min_lr} / {args.warmup_epochs}",
        f"- **epochs**: {len(history)}/{args.epochs} (patience={args.patience})",
        f"- **loss weights**: dens={args.dens_weight} moment={args.moment_weight} "
        f"a2={args.a2_weight} β={args.beta}",
        f"- **params / peak RSS**: {n_params:,} / {peak_rss_mb:.0f} MB",
        f"- **train wall**: {train_wall_s/60:.1f} min",
        f"- **best_loss**: {best:.4f}",
        "",
        "## Morphology",
        f"- bar recon A₂: {bar_recon_a2:.3f} (crisp AE ref {CRISP_AE['bar_a2_resampled']})",
        f"- quiet recon A₂: {quiet_recon_a2:.3f}",
        f"- decode(μ) A₂ bar/quiet: {bar_mu:.3f} / {quiet_mu:.3f} "
        f"(separated={verdict['morphology_mu_separated']})",
        f"- prior mean A₂ bar/quiet: {bar_prior_mean:.3f} / {quiet_prior_mean:.3f} "
        f"(separated={verdict['morphology_prior_separated']})",
        "",
        "## Sampling API",
        "```python",
        "z = torch.randn(1, latent_dim)   # or encode → μ, or interpolate μ's",
        "fields = model.sample(theta, z=z)",
        "# denormalize stacks → resample_particles_from_multiscale → evolve",
        "```",
        "",
    ]
    if args.note:
        notes.extend(["## Note", args.note, ""])
    (out / "NOTES.md").write_text("\n".join(notes))

    latest = Path("runs/ml/field_maps/LATEST")
    latest.write_text(str(out.resolve()) + "\n")
    print(f"verdict → {out / 'verdict.json'}", flush=True)
    print(f"notes → {out / 'NOTES.md'}", flush=True)
    print(
        f"decode(μ) A₂ bar≈{bar_mu:.3f} quiet≈{quiet_mu:.3f} "
        f"separated={verdict['morphology_mu_separated']}; "
        f"prior bar≈{bar_prior_mean:.3f} quiet≈{quiet_prior_mean:.3f}; "
        f"bar_recon={bar_recon_a2:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
