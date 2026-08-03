#!/usr/bin/env python3
"""
Creative overnight: frozen crisp AE + single-z skip distillation.

Separates recon quality (frozen U-Net AE) from prior learning (code VAE that
synthesizes bottleneck+skips from one latent z).  Also runs quiet↔bar
encode-interpolate demos to prove latent usefulness.

    . .venv/bin/activate
    OMP_NUM_THREADS=6 python scripts/train_latent_code_prior.py
    OMP_NUM_THREADS=6 python scripts/train_latent_code_prior.py --with-morph --barred-weight 3
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
from torch.utils.data import DataLoader, WeightedRandomSampler

os.environ.setdefault("OMP_NUM_THREADS", "8")
torch.set_num_threads(8)

from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_from_record
from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
from galacticsics.ml.fields.binning import MultiScaleSliceConfig, bin_multiscale_slice_stacks, scale_summary
from galacticsics.ml.fields.dataset import MultiScaleFieldDataset, collate_multiscale_batch
from galacticsics.ml.fields.frame import prepare_shared_frame
from galacticsics.ml.fields.latent_code import (
    FrozenAECodeVAE,
    LatentCodeConfig,
    disk_dens_collapse,
    latent_code_loss,
    load_frozen_teacher,
    morph_a2_summary,
)
from galacticsics.ml.fields.normalize import FieldNormStats, denormalize_stack, normalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from galacticsics.ml.morton.index import SnapshotRecord, resolve_snapshot_t_gyr
from galacticsics.ml.morton.tokenize import _component_ids
from ntropy.analysis.disk_density import disk_azimuthal_fourier


MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
WORK = Path("runs/mw_morton_corpus_v2")
CRISP_AE = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
DEVICE = "cpu"
SEED = 0
COUNT_FRACTIONS = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
OVERNIGHT_VAE_BAR_MU = 0.058  # best prior-path bar A₂ from a2ft

EXAMPLES = [
    {
        "name": "bar_54a8_late",
        "run": "54a8faf836a0",
        "dump": "evolution/particles/step_001700.npz",
        "label": "barred",
    },
    {
        "name": "quiet_ic_081e",
        "run": "081ed8af4b2b",
        "dump": "ic_state.npz",
        "label": "quiet",
    },
]


def _a2_median(pos, mass) -> float:
    out = disk_azimuthal_fourier(
        pos, mass, m=2, r_max=12.0, n_bins=12, z_max=0.5, min_count=10
    )
    return float(out["a_m_over_a0_median"])


def _a2_from_parts(parts: dict) -> float:
    """Median disk A₂; prefer component mask when present."""
    pos = np.asarray(parts["pos"], dtype=np.float64)
    mass = np.asarray(parts["mass"], dtype=np.float64)
    if "component" in parts:
        disk = np.asarray(parts["component"]) == "disk"
        if np.any(disk):
            return _a2_median(pos[disk], mass[disk])
    if "component_id" in parts:
        disk = np.asarray(parts["component_id"]) == 0
        if np.any(disk):
            return _a2_median(pos[disk], mass[disk])
    # Fallback: midplane cut (halo/bulge dilute A₂ less if z-thin).
    return _a2_median(pos, mass)


def _rss_mb() -> float:
    import resource

    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


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
    raise SystemExit(f"no θ for {ex['name']}")


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
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _disk_collapse_np(stack: np.ndarray, n_z: int, n_mom: int) -> np.ndarray:
    dens = np.zeros(stack.shape[-2:], dtype=np.float64)
    for iz in range(n_z):
        dens += np.maximum(stack[iz * n_mom], 0.0)
    return dens


def _example_path(work: Path, ex: dict) -> Path:
    return work / ex["run"] / ex["dump"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--teacher", type=Path, default=CRISP_AE)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max-snap", type=int, default=36)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--synth-grid", type=int, default=8)
    p.add_argument("--beta", type=float, default=5e-4)
    p.add_argument("--skip-weight", type=float, default=1.0)
    p.add_argument("--dens-weight", type=float, default=4.0)
    p.add_argument("--moment-weight", type=float, default=6.0)
    p.add_argument("--a2-weight", type=float, default=0.0, help="Fourier off by default")
    p.add_argument("--with-morph", action="store_true", help="append A₂ summary to θ")
    p.add_argument("--barred-weight", type=float, default=2.5, help="oversample high-A₂")
    p.add_argument("--use-flow", action="store_true", help="RealNVP prior on codes")
    p.add_argument("--flow-nll-weight", type=float, default=0.05)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="encode→μ only (no KL); use with --use-flow for prior sampling",
    )
    p.add_argument(
        "--select-by-bar-a2",
        action="store_true",
        help="every epoch eval barred example map A₂; keep best (avoids skip-L2 washout)",
    )
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--n-prior-samples", type=int, default=3)
    p.add_argument("--n-interp", type=int, default=5)
    p.add_argument("--omp-threads", type=int, default=6)
    p.add_argument("--eval-only", type=Path, default=None, help="load ckpt, skip train")
    p.add_argument("--note", type=str, default="")
    return p.parse_args()


def _make_scheduler(opt, *, epochs, warmup, lr, min_lr):
    if min_lr <= 0 or epochs <= 1:
        return None
    import math

    warm = max(0, int(warmup))

    def _lr_lambda(epoch_idx: int) -> float:
        e = epoch_idx + 1
        if warm > 0 and e <= warm:
            return max(e / float(warm), 1e-3)
        remain = max(1, epochs - warm)
        t = min(max(e - warm, 0), remain) / float(remain)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        floor = float(min_lr) / float(lr)
        return floor + (1.0 - floor) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)


def _estimate_bar_weights(ds: MultiScaleFieldDataset, *, barred_weight: float) -> np.ndarray:
    """Cheap proxy: late dumps / high step get higher weight."""
    w = np.ones(len(ds), dtype=np.float64)
    for i, rec in enumerate(ds.records):
        path = rec.path
        boost = 1.0
        if "step_" in path:
            # Extract step number when present.
            try:
                step = int(path.split("step_")[-1].split(".")[0])
                boost = 1.0 + min(step / 2000.0, 1.5) * (float(barred_weight) - 1.0)
            except ValueError:
                boost = float(barred_weight)
        elif "ic_state" in path:
            boost = 0.7
        # Prefer known barred campaign hashes lightly.
        if "54a8" in path or getattr(rec, "run_hash", "")[:4] == "54a8":
            boost *= float(barred_weight)
        w[i] = boost
    return w


def _as_norm_stats(raw) -> dict[str, FieldNormStats]:
    """Checkpoint may store ``FieldNormStats`` or plain dicts."""
    out: dict[str, FieldNormStats] = {}
    for k, v in raw.items():
        if isinstance(v, FieldNormStats):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = FieldNormStats(**v)
        else:
            out[k] = FieldNormStats(**v.__dict__)
    return out


def _prepare_example_stacks(cfg, path: Path, stats: dict[str, FieldNormStats]):
    pos, vel, mass, cid = _load_shared_frame(path)
    binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    raw = {k: v[0] for k, v in binned.items()}
    stacks = {
        k: torch.as_tensor(normalize_stack(raw[k], stats[k]), dtype=torch.float32).unsqueeze(0)
        for k in raw
    }
    return stacks, raw, pos, vel, mass, cid


def main() -> None:
    args = parse_args()
    tag = "morph" if args.with_morph else "base"
    if args.deterministic:
        tag = "det_" + tag
    if args.use_flow:
        tag += "_flow"
    out = args.out or Path(
        f"runs/ml/field_maps/creative_latent_{tag}_{date.today().isoformat()}"
    )
    omp = max(1, int(args.omp_threads))
    os.environ["OMP_NUM_THREADS"] = str(omp)
    torch.set_num_threads(omp)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    print(f"artifacts → {out}  OMP={omp}  rss0={_rss_mb():.0f}MB", flush=True)

    if not args.teacher.is_file():
        raise SystemExit(f"missing teacher AE {args.teacher}")
    if not args.manifest.is_file():
        raise SystemExit(f"missing manifest {args.manifest}")

    # Match crisp AE progressive grids from checkpoint args when possible.
    t_probe = torch.load(args.teacher, map_location="cpu", weights_only=False)
    t_args = t_probe.get("args", {})
    disk_n = int(t_args.get("disk_n_pix", 128))
    cfg = MultiScaleSliceConfig.progressive_defaults(
        disk_n_pix=disk_n,
        include_potential=False,
        moment_set=str(t_args.get("moment_set", "disp")),
    )
    print("grids:")
    for row in scale_summary(cfg):
        print(f"  {row}")

    teacher, t_ckpt = load_frozen_teacher(args.teacher, cfg, device=DEVICE)
    # Prefer teacher norm stats if present.
    stats_raw = t_ckpt.get("norm")
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"frozen teacher params={n_teacher:,}  base_c={t_args.get('base_channels')}", flush=True)

    morph_dim = 4 if args.with_morph else 0
    lcfg = LatentCodeConfig(
        latent_dim=args.latent_dim,
        theta_dim=len(DEFAULT_THETA_KEYS),
        morph_dim=morph_dim,
        enc_grid=args.enc_grid,
        synth_grid=args.synth_grid,
        beta=args.beta,
        skip_weight=args.skip_weight,
        recon_weight=1.0,
        bottleneck_weight=1.0,
        w_e1=1.5,
        w_e2=1.0,
        w_e3=0.75,
        deterministic=bool(args.deterministic),
    )
    model = FrozenAECodeVAE(
        teacher, cfg=lcfg, use_flow_prior=args.use_flow or args.deterministic
    ).to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"student trainable={n_train:,}  z={args.latent_dim}  morph_dim={morph_dim}  "
        f"flow={args.use_flow}  rss={_rss_mb():.0f}MB",
        flush=True,
    )

    ds = MultiScaleFieldDataset(
        args.manifest,
        cfg=cfg,
        split=None,
        max_snapshots=args.max_snap,
        seed=SEED,
        augment=True,
        include_potential=False,
        theta_keys=DEFAULT_THETA_KEYS,
        norm_stats=None,
    )
    print(f"dataset size={len(ds)}; preloading…", flush=True)
    t0 = time.time()
    ds.preload()
    print(f"  preload {time.time() - t0:.1f}s", flush=True)
    if stats_raw is None:
        stats = ds.fit_norm_stats(n_samples=min(10, len(ds)))
    else:
        stats = _as_norm_stats(stats_raw)
        ds.norm_stats = stats
    dens_idx = ds.dens_indices()
    disk_grid = cfg.grid_for("disk")

    def _morph_from_stacks(stacks_t: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if morph_dim <= 0:
            return None
        dens = disk_dens_collapse(
            stacks_t["disk"], n_z=disk_grid.n_z, n_mom=disk_grid.n_mom
        )
        # dens is still normalized log1p-ish — soft A₂ still ranks morphology.
        return morph_a2_summary(dens)

    def _collate(batch):
        b = collate_multiscale_batch(batch)
        stacks = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in b["stacks"].items()
        }
        theta = torch.as_tensor(b["theta"], dtype=torch.float32)
        return {"stacks": stacks, "theta": theta}

    weights = _estimate_bar_weights(ds, barred_weight=args.barred_weight)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(ds),
        replacement=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=0,
        collate_fn=_collate,
    )

    history: list[dict] = []
    best = float("inf")
    best_state = None
    stale = 0
    # Prefer skip+dens for ckpt selection — KL-heavy total loss washes bars.
    select_kl_blind = bool(args.deterministic) or float(args.beta) < 1e-4

    if args.eval_only is not None:
        ck = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        print(f"loaded {args.eval_only}", flush=True)
    else:
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=1e-4,
        )
        sched = _make_scheduler(
            opt,
            epochs=args.epochs,
            warmup=args.warmup_epochs,
            lr=args.lr,
            min_lr=args.min_lr,
        )
        comp_w = {"disk": 2.5, "bulge": 1.0, "halo": 0.4}
        t_train0 = time.time()
        for epoch in range(1, args.epochs + 1):
            ds.set_epoch(epoch)
            model.train()
            losses, dens_l, mom_l, skip_l, kl_l = [], [], [], [], []
            for batch in loader:
                stacks = {k: v.to(DEVICE) for k, v in batch["stacks"].items()}
                theta = batch["theta"].to(DEVICE)
                morph = _morph_from_stacks(stacks)
                if morph is not None:
                    morph = morph.to(DEVICE)
                metrics = latent_code_loss(
                    model,
                    stacks,
                    theta,
                    dens_indices=dens_idx,
                    dens_weight=args.dens_weight,
                    moment_weight=args.moment_weight,
                    a2_weight=args.a2_weight,
                    component_weights=comp_w,
                    morph=morph,
                    flow_nll_weight=(
                        args.flow_nll_weight
                        if (args.use_flow or args.deterministic)
                        else 0.0
                    ),
                )
                loss = metrics["loss"]
                if not torch.isfinite(loss):
                    print("  skip non-finite", flush=True)
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                opt.step()
                losses.append(float(loss.detach()))
                dens_l.append(float(metrics["mse_dens"].detach()))
                mom_l.append(float(metrics["mse_mom"].detach()))
                skip_l.append(float(metrics["skip_distill"].detach()))
                kl_l.append(float(metrics["kl"].detach()))
            if sched is not None:
                sched.step()
            if not losses:
                continue
            mean_loss = float(np.mean(losses))
            mean_dens = float(np.mean(dens_l))
            mean_mom = float(np.mean(mom_l))
            mean_skip = float(np.mean(skip_l))
            mean_kl = float(np.mean(kl_l))
            # Score without KL so posterior collapse cannot win the checkpoint race.
            score = float(mean_skip + mean_dens)
            bar_a2_epoch = None
            if args.select_by_bar_a2:
                try:
                    bar_ex = EXAMPLES[0]
                    bar_path = _example_path(args.work, bar_ex)
                    if not bar_path.is_file():
                        bar_path = (
                            Path("runs/mw_morton_corpus_v2") / bar_ex["run"] / bar_ex["dump"]
                        )
                    bst, _, _, _, _, _ = _prepare_example_stacks(cfg, bar_path, stats)
                    bst = {k: v.to(DEVICE) for k, v in bst.items()}
                    th = torch.as_tensor(
                        _theta_for_example(args.manifest, bar_ex)[None],
                        dtype=torch.float32,
                        device=DEVICE,
                    )
                    morph_b = _morph_from_stacks(bst)
                    if morph_b is not None:
                        morph_b = morph_b.to(DEVICE)
                    model.eval()
                    with torch.no_grad():
                        mu_b = model.encode_mu(bst)
                        ts = {k: (v.shape[-2], v.shape[-1]) for k, v in bst.items()}
                        nch = {k: int(v.shape[1]) for k, v in bst.items()}
                        syn = model.sample(
                            th, z=mu_b, morph=morph_b, target_shapes=ts, n_channels=nch
                        )
                    den = {
                        k: denormalize_stack(syn[k][0].cpu().numpy(), stats[k])
                        for k in syn
                    }
                    dens = _disk_collapse_np(den["disk"], disk_grid.n_z, disk_grid.n_mom)
                    bar_a2_epoch = float(
                        dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)[
                            "a_m_over_a0_median"
                        ]
                    )
                    score = -float(bar_a2_epoch)
                    model.train()
                except Exception as exc:  # noqa: BLE001
                    print(f"  bar-a2 select failed: {exc}", flush=True)
            row = {
                "epoch": epoch,
                "loss": mean_loss,
                "score": score,
                "bar_a2_map": bar_a2_epoch,
                "mse_dens": mean_dens,
                "mse_mom": mean_mom,
                "skip_distill": mean_skip,
                "kl": mean_kl,
                "lr": float(opt.param_groups[0]["lr"]),
                "rss_mb": _rss_mb(),
            }
            history.append(row)
            print(
                f"  epoch {epoch}  loss={mean_loss:.4f}  score={score:.4f}  "
                f"barA2={bar_a2_epoch}  dens={row['mse_dens']:.4f}  "
                f"mom={row['mse_mom']:.4f}  skip={row['skip_distill']:.4f}  "
                f"kl={row['kl']:.4f}  lr={row['lr']:.2e}  rss={row['rss_mb']:.0f}MB",
                flush=True,
            )
            if score + 1e-4 < best:
                best = score
                stale = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if not k.startswith("teacher.")
                }
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"  early stop @ epoch {epoch}", flush=True)
                    break
        print(f"train wall {time.time() - t_train0:.1f}s  best_score={best:.4f}", flush=True)
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)

        ckpt_path = out / "frozen_ae_code_vae.pt"
        torch.save(
            {
                "model": {
                    k: v
                    for k, v in model.state_dict().items()
                    if not k.startswith("teacher.")
                },
                "cfg": lcfg.__dict__,
                "norm": stats,
                "teacher_path": str(args.teacher),
                "args": vars(args),
                "history": history,
            },
            ckpt_path,
        )
        print(f"wrote {ckpt_path}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))

    # ---- Eval: teacher recon, synth recon (encode μ), prior, interp ----
    model.eval()
    verdict: dict = {
        "approach": "frozen crisp AE + skip-distill code VAE",
        "overnight_vae_bar_mu_ref": OVERNIGHT_VAE_BAR_MU,
        "crisp_ae_bar_a2_ref": 0.33,
        "latent_dim": args.latent_dim,
        "with_morph": args.with_morph,
        "use_flow": args.use_flow,
        "examples": [],
        "interp": [],
        "prior_samples": [],
    }

    # Cache μ for bar / quiet for interpolation.
    mu_cache: dict[str, torch.Tensor] = {}
    theta_cache: dict[str, torch.Tensor] = {}
    morph_cache: dict[str, torch.Tensor | None] = {}
    dens_true: dict[str, np.ndarray] = {}

    for ex in EXAMPLES:
        path = _example_path(args.work, ex)
        if not path.is_file():
            # try under runs/
            alt = Path("runs") / "mw_morton_corpus_v2" / ex["run"] / ex["dump"]
            path = alt if alt.is_file() else path
        if not path.is_file():
            print(f"  missing example {path}", flush=True)
            continue
        stacks, raw, pos, vel, mass, cid = _prepare_example_stacks(cfg, path, stats)
        stacks = {k: v.to(DEVICE) for k, v in stacks.items()}
        theta_np = _theta_for_example(args.manifest, ex)
        theta = torch.as_tensor(theta_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        morph = _morph_from_stacks(stacks)
        if morph is not None:
            morph = morph.to(DEVICE)

        with torch.no_grad():
            # Teacher recon (upper bound with skips).
            t_out = teacher(stacks)
            # Synth from encoded μ (no teacher skips at decode).
            mu = model.encode_mu(stacks)
            target_shapes = {k: (v.shape[-2], v.shape[-1]) for k, v in stacks.items()}
            n_ch = {k: int(v.shape[1]) for k, v in stacks.items()}
            synth_stacks = model.sample(
                theta, z=mu, morph=morph, target_shapes=target_shapes, n_channels=n_ch
            )
            # Prior samples for this θ.
            prior_list = []
            for si in range(args.n_prior_samples):
                z = torch.randn(1, args.latent_dim, device=DEVICE)
                # For morph conditioning at sample time: use example morph as
                # a controllable knob (bar vs quiet).  Structural θ alone for
                # morph_dim=0.
                ps = model.sample(
                    theta,
                    z=z,
                    morph=morph,
                    target_shapes=target_shapes,
                    n_channels=n_ch,
                )
                prior_list.append(ps)

        def _denorm_all(stk):
            return {
                k: denormalize_stack(stk[k][0].cpu().numpy(), stats[k]) for k in stk
            }

        true_dens = _disk_collapse_np(raw["disk"], disk_grid.n_z, disk_grid.n_mom)
        teacher_d = _denorm_all(t_out)
        synth_d = _denorm_all(synth_stacks)
        teacher_dens = _disk_collapse_np(teacher_d["disk"], disk_grid.n_z, disk_grid.n_mom)
        synth_dens = _disk_collapse_np(synth_d["disk"], disk_grid.n_z, disk_grid.n_mom)

        # Particle A₂ via resample from synth.
        parts = resample_particles_from_multiscale(
            synth_d,
            cfg=cfg,
            n_particles=args.n_resample,
            count_fractions=COUNT_FRACTIONS,
            rng=rng,
            sample_dispersion=True,
        )
        a2_part = _a2_from_parts(parts)
        a2_map_true = dens_map_azimuthal_fourier_numpy(true_dens, m=2, r_max=12.0)
        a2_map_teacher = dens_map_azimuthal_fourier_numpy(teacher_dens, m=2, r_max=12.0)
        a2_map_synth = dens_map_azimuthal_fourier_numpy(synth_dens, m=2, r_max=12.0)

        prior_a2 = []
        prior_a2_part = None
        prior_panels = [true_dens, teacher_dens, synth_dens]
        prior_labels = ["data", "teacher AE", "synth@μ"]
        for si, ps in enumerate(prior_list):
            pd = _denorm_all(ps)
            pdens = _disk_collapse_np(pd["disk"], disk_grid.n_z, disk_grid.n_mom)
            am = dens_map_azimuthal_fourier_numpy(pdens, m=2, r_max=12.0)
            prior_a2.append(float(am["a_m_over_a0_median"]))
            prior_panels.append(pdens)
            prior_labels.append(f"prior#{si}")
            # Particle A₂ for first prior only (costly).
            if si == 0:
                p_parts = resample_particles_from_multiscale(
                    pd,
                    cfg=cfg,
                    n_particles=args.n_resample,
                    count_fractions=COUNT_FRACTIONS,
                    rng=rng,
                    sample_dispersion=True,
                )
                prior_a2_part = _a2_from_parts(p_parts)

        _plot_panel(
            out / f"{ex['name']}_creative_panel.png",
            title=f"{ex['name']}  synth A₂(map)={a2_map_synth['a_m_over_a0_median']:.3f}",
            panels=prior_panels[:5],
            labels=prior_labels[:5],
        )

        row = {
            "name": ex["name"],
            "a2_map_data": float(a2_map_true["a_m_over_a0_median"]),
            "a2_map_teacher": float(a2_map_teacher["a_m_over_a0_median"]),
            "a2_map_synth_mu": float(a2_map_synth["a_m_over_a0_median"]),
            "a2_part_synth_mu": float(a2_part),
            "a2_map_prior_mean": float(np.mean(prior_a2)) if prior_a2 else None,
            "a2_map_prior_max": float(np.max(prior_a2)) if prior_a2 else None,
            "a2_part_prior0": prior_a2_part,
            "mu_norm": float(mu.norm().item()),
        }
        verdict["examples"].append(row)
        print(
            f"  {ex['name']}: map A₂ data={row['a2_map_data']:.3f}  "
            f"teacher={row['a2_map_teacher']:.3f}  synthμ={row['a2_map_synth_mu']:.3f}  "
            f"partμ={row['a2_part_synth_mu']:.3f}  prior_mean={row['a2_map_prior_mean']}  "
            f"prior_part0={row['a2_part_prior0']}",
            flush=True,
        )
        mu_cache[ex["name"]] = mu.cpu()
        theta_cache[ex["name"]] = theta.cpu()
        morph_cache[ex["name"]] = None if morph is None else morph.cpu()
        dens_true[ex["name"]] = true_dens

        for si, a2 in enumerate(prior_a2):
            verdict["prior_samples"].append(
                {"example": ex["name"], "i": si, "a2_map": a2}
            )

    # Quiet ↔ bar latent interpolation.
    if "bar_54a8_late" in mu_cache and "quiet_ic_081e" in mu_cache:
        z0 = mu_cache["quiet_ic_081e"]
        z1 = mu_cache["bar_54a8_late"]
        th = theta_cache["bar_54a8_late"].to(DEVICE)
        # Blend morph when present.
        m0 = morph_cache["quiet_ic_081e"]
        m1 = morph_cache["bar_54a8_late"]
        panels = [dens_true["quiet_ic_081e"]]
        labels = ["quiet data"]
        interp_rows = []
        target_shapes = {
            g.name: (g.n_pix, g.n_pix) for g in cfg.grids
        }
        n_ch = {g.name: g.n_moment_channels for g in cfg.grids}
        for i, alpha in enumerate(np.linspace(0.0, 1.0, args.n_interp)):
            z = ((1 - alpha) * z0 + alpha * z1).to(DEVICE)
            morph = None
            if m0 is not None and m1 is not None:
                morph = ((1 - alpha) * m0 + alpha * m1).to(DEVICE)
            with torch.no_grad():
                stk = model.sample(
                    th, z=z, morph=morph, target_shapes=target_shapes, n_channels=n_ch
                )
            den = {
                k: denormalize_stack(stk[k][0].cpu().numpy(), stats[k]) for k in stk
            }
            dens = _disk_collapse_np(den["disk"], disk_grid.n_z, disk_grid.n_mom)
            am = dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)
            a2 = float(am["a_m_over_a0_median"])
            # Particle A₂ at endpoints + midpoint.
            a2_part = None
            if i in (0, args.n_interp // 2, args.n_interp - 1):
                parts = resample_particles_from_multiscale(
                    den,
                    cfg=cfg,
                    n_particles=args.n_resample,
                    count_fractions=COUNT_FRACTIONS,
                    rng=rng,
                    sample_dispersion=True,
                )
                a2_part = _a2_from_parts(parts)
            interp_rows.append({"alpha": float(alpha), "a2_map": a2, "a2_part": a2_part})
            panels.append(dens)
            labels.append(f"α={alpha:.2f}\nA₂={a2:.2f}")
        panels.append(dens_true["bar_54a8_late"])
        labels.append("bar data")
        _plot_panel(
            out / "interp_quiet_to_bar.png",
            title="Encode–interpolate–decode (single z)",
            panels=panels,
            labels=labels,
        )
        verdict["interp"] = interp_rows
        print("  interp A₂(map):", [f"{r['a2_map']:.3f}" for r in interp_rows], flush=True)

    # Teacher-feature interpolation upper bound (skip blend, no z).
    if "bar_54a8_late" in mu_cache:
        path_b = _example_path(args.work, EXAMPLES[0])
        path_q = _example_path(args.work, EXAMPLES[1])
        for pth in (path_b, path_q):
            if not pth.is_file():
                pth = Path("runs/mw_morton_corpus_v2") / pth.parent.name / pth.name
        stacks_b, _, _, _, _, _ = _prepare_example_stacks(cfg, path_b if path_b.is_file() else Path("runs/mw_morton_corpus_v2") / EXAMPLES[0]["run"] / EXAMPLES[0]["dump"], stats)
        stacks_q, _, _, _, _, _ = _prepare_example_stacks(cfg, path_q if path_q.is_file() else Path("runs/mw_morton_corpus_v2") / EXAMPLES[1]["run"] / EXAMPLES[1]["dump"], stats)
        stacks_b = {k: v.to(DEVICE) for k, v in stacks_b.items()}
        stacks_q = {k: v.to(DEVICE) for k, v in stacks_q.items()}
        with torch.no_grad():
            fb = teacher.encode_features(stacks_b)
            fq = teacher.encode_features(stacks_q)
        feat_panels = []
        feat_labels = []
        feat_rows = []
        for alpha in np.linspace(0.0, 1.0, args.n_interp):
            mixed = {}
            for name in teacher._tower_order:
                bq, sq = fq[name]["bottleneck"], fq[name]["skips"]
                bb, sb = fb[name]["bottleneck"], fb[name]["skips"]
                mixed[name] = {
                    "bottleneck": (1 - alpha) * bq + alpha * bb,
                    "skips": tuple((1 - alpha) * a + alpha * b for a, b in zip(sq, sb)),
                }
            with torch.no_grad():
                out_s = teacher.decode_features(
                    mixed,
                    target_shapes={k: (v.shape[-2], v.shape[-1]) for k, v in stacks_b.items()},
                    n_channels={k: int(v.shape[1]) for k, v in stacks_b.items()},
                )
            den = {k: denormalize_stack(out_s[k][0].cpu().numpy(), stats[k]) for k in out_s}
            dens = _disk_collapse_np(den["disk"], disk_grid.n_z, disk_grid.n_mom)
            am = dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)
            a2 = float(am["a_m_over_a0_median"])
            feat_rows.append({"alpha": float(alpha), "a2_map": a2})
            feat_panels.append(dens)
            feat_labels.append(f"α={alpha:.2f}\nA₂={a2:.2f}")
        _plot_panel(
            out / "interp_teacher_features.png",
            title="Teacher feature interp (upper bound)",
            panels=feat_panels,
            labels=feat_labels,
        )
        verdict["teacher_feature_interp"] = feat_rows
        print(
            "  teacher-feat interp A₂:",
            [f"{r['a2_map']:.3f}" for r in feat_rows],
            flush=True,
        )

    # Success criteria vs overnight VAE.
    bar_row = next((e for e in verdict["examples"] if e["name"].startswith("bar")), None)
    quiet_row = next((e for e in verdict["examples"] if e["name"].startswith("quiet")), None)
    bar_mu_a2 = None
    if bar_row:
        bar_mu_a2 = bar_row.get("a2_part_synth_mu") or bar_row.get("a2_map_synth_mu")
        bar_prior = bar_row.get("a2_part_prior0") or bar_row.get("a2_map_prior_mean")
    else:
        bar_prior = None
    quiet_prior = None
    if quiet_row:
        quiet_prior = quiet_row.get("a2_part_prior0") or quiet_row.get("a2_map_prior_mean")

    better = False
    reasons = []
    if bar_mu_a2 is not None and bar_mu_a2 > OVERNIGHT_VAE_BAR_MU * 1.5:
        better = True
        reasons.append(f"bar_mu={bar_mu_a2:.3f}")
    if bar_prior is not None and bar_prior > OVERNIGHT_VAE_BAR_MU * 1.5:
        better = True
        reasons.append(f"bar_prior={bar_prior:.3f}")
    # Interp should morph (endpoint gap).
    if verdict.get("interp") and len(verdict["interp"]) >= 2:
        gap = abs(verdict["interp"][-1]["a2_map"] - verdict["interp"][0]["a2_map"])
        verdict["interp_a2_gap"] = gap
        if gap > 0.05:
            better = True
            reasons.append(f"interp_gap={gap:.3f}")
    # Quiet must not explode (Fourier-invented bars).
    if quiet_prior is not None and quiet_prior > 0.08:
        better = False
        reasons.append(f"quiet_too_loud={quiet_prior:.3f}")
    if quiet_row is not None:
        qmu = quiet_row.get("a2_part_synth_mu") or quiet_row.get("a2_map_synth_mu")
        if qmu is not None and qmu > 0.08 and (bar_mu_a2 is None or qmu > 0.7 * (bar_mu_a2 or 0)):
            better = False
            reasons.append(f"quiet_mu_loud={qmu:.3f}")

    verdict["better_than_overnight_vae"] = better
    verdict["better_reasons"] = reasons
    verdict["rss_mb"] = _rss_mb()
    verdict["note"] = args.note
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(
        json.dumps(
            {
                "better": better,
                "reasons": reasons,
                "bar_mu": bar_mu_a2,
                "bar_prior": bar_prior,
                "quiet_prior": quiet_prior,
            },
            indent=2,
        ),
        flush=True,
    )

    # Update LATEST pointer when clearly better.
    latest = Path("runs/ml/field_maps/LATEST")
    if better:
        latest.write_text(str(out.resolve()) + "\n")
        print(f"updated LATEST → {out}", flush=True)

    if args.note:
        (out / "NOTES.md").write_text(args.note + "\n")


if __name__ == "__main__":
    main()
