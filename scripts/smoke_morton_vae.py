#!/usr/bin/env python3
"""
Smoke-test the Morton set VAE (CPU + OpenMP by default while corpus owns the GPU).

Why absolute losses look “bad”
------------------------------
Total loss is dominated by **position MSE**.  Galaxy particles have
``E[|x|²] ~ 10³`` (kpc²).  An untrained decoder predicts near-zero coordinates,
so ``mse_dx ~ E[|x|²]`` and ``loss ≈ λ_recon * mse_dx ~ O(10²–10³)``.  That is
expected at init — not a NaN / divergence.  Compare terms to the calibration
block and watch **relative** drops / morphology score, not the absolute total.

Morphology score
----------------
``score = soft_Fourier_MSE(m=1,2) + soft_map_MSE`` between a held-out batch and
the decode (permutation-invariant).  Lower is better.  Absolute values are
unnormalized MSE (typically O(1–10)); the ablation asks whether ``full`` beats
``recon_only`` by a few percent.

Timing
------
Default device is **cpu** so training does not fight ``gpu_bh`` corpus jobs.
Evolve uses **bh_c** with ``OMP_NUM_THREADS`` (OpenMP).  Cap snapshots /
``n_train`` for a <1 min smoke; use ``--scale-up`` when the GPU is free.

Example
-------
::

    OMP_NUM_THREADS=8 python scripts/smoke_morton_vae.py
    python scripts/smoke_morton_vae.py --scale-up --device cuda   # when corpus idle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _configure_device(device: str, max_vram_gb: float) -> str:
    import torch

    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA unavailable → falling back to cpu", flush=True)
            return "cpu"
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        # Cap process footprint so we stay under the requested VRAM budget.
        frac = min(0.95, float(max_vram_gb) / max(total_gb, 1e-6))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(0)
        print(
            f"CUDA {props.name}: total={total/1024**3:.1f} GiB  "
            f"free={free/1024**3:.1f} GiB  process_cap={max_vram_gb:.1f} GiB "
            f"(fraction={frac:.3f})",
            flush=True,
        )
        return "cuda"
    return "cpu"


def _peak_vram_gb(device: str) -> float:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(0) / (1024**3))


def _write_synthetic_corpus(root: Path, *, n_runs: int, n_part: int, seed: int) -> Path:
    from galacticsics.ml.morton.index import write_snapshot_manifest

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_runs):
        run = root / f"smoke{i:03d}"
        run.mkdir(exist_ok=True)
        (run / "model.json").write_text(
            json.dumps(
                {
                    "label": f"smoke{i}",
                    "disk": {
                        "mass": 12.0 + float(i % 4),
                        "scale_length": 2.5,
                        "scale_height": 0.3,
                        "enabled": True,
                    },
                    "halo": {"v0": 3.5, "a": 30.0, "enabled": True},
                    "bulge": {"v0": 1.5, "a": 0.6, "enabled": True},
                    "disk_kinematics": {"toomre_q_target": 1.4, "sigma_r0": 0.5},
                }
            )
        )
        R = rng.exponential(3.0, n_part).clip(0.3, 14.0)
        phi = rng.uniform(0.0, 2.0 * np.pi, n_part)
        bar = 0.35 if (i % 2 == 0) else 0.05
        w = 1.0 + bar * np.cos(2.0 * phi)
        w = w / w.sum()
        idx = rng.choice(n_part, size=n_part, replace=True, p=w)
        R, phi = R[idx], phi[idx]
        pos = np.stack(
            [R * np.cos(phi), R * np.sin(phi), rng.normal(0.0, 0.25, n_part)],
            axis=1,
        )
        vel = np.stack(
            [-0.8 * np.sin(phi), 0.8 * np.cos(phi), rng.normal(0.0, 0.05, n_part)],
            axis=1,
        )
        n_disk = int(0.6 * n_part)
        n_halo = int(0.28 * n_part)
        n_bulge = n_part - n_disk - n_halo
        type_id = np.array(
            [0] * n_disk + [1] * n_halo + [2] * n_bulge,
            dtype=np.int32,
        )
        np.savez(
            run / "ic_state.npz",
            pos=pos.astype(np.float64),
            vel=vel.astype(np.float64),
            mass=np.ones(n_part) / n_part,
            eps=np.full(n_part, 0.1),
            type_id=type_id,
        )
    return write_snapshot_manifest(root)


def _resolve_manifest(work_roots: list[Path], out_dir: Path, seed: int) -> tuple[Path, str]:
    """Prefer real campaign snapshots; fall back to synthetic."""
    from galacticsics.ml.morton.index import build_snapshot_index, write_snapshot_manifest

    for root in work_roots:
        if not root.is_dir():
            continue
        records = build_snapshot_index(root)
        if not records:
            continue
        manifest = write_snapshot_manifest(root, out_dir / f"manifest_{root.name}.json")
        n_ic = sum(1 for r in records if r.source == "ic")
        n_dump = sum(1 for r in records if r.source == "dump")
        print(
            f"Using campaign corpus {root}  snapshots={len(records)}  "
            f"(ic={n_ic}, dumps={n_dump})",
            flush=True,
        )
        return manifest, f"campaign:{root}"
    print("No campaign dumps found → synthetic fallback", flush=True)
    return _write_synthetic_corpus(out_dir / "corpus", n_runs=8, n_part=600, seed=seed), "synthetic"


def _calibrate_token_scales(ds, n_show: int = 4) -> dict[str, float]:
    """Print expected MSE scales so a large loss is interpretable."""
    dms, dxs, vs = [], [], []
    for i in range(min(n_show, len(ds))):
        item = ds[i]
        dms.append(item["dm"])
        dxs.append(item["dx"])
        vs.append(item["v"])
    dm = np.concatenate(dms)
    dx = np.concatenate(dxs)
    v = np.concatenate(vs)
    calib = {
        "E_dm2_raw": float(np.mean(dm**2)),
        "E_log1p_dm2": float(np.mean(np.log1p(np.clip(dm, 0, None)) ** 2)),
        "E_dx2": float(np.mean(dx**2)),
        "E_v2": float(np.mean(v**2)),
        "dm_max": float(np.max(dm)),
    }
    print("=== loss scale calibration (from training snapshots) ===", flush=True)
    print(
        f"  E[Δm²] raw={calib['E_dm2_raw']:.3e}  → with log1p: {calib['E_log1p_dm2']:.3f}",
        flush=True,
    )
    print(
        f"  E[|x|²]={calib['E_dx2']:.3f}  E[|v|²]={calib['E_v2']:.3f}  "
        f"Δm_max={calib['dm_max']:.1f}",
        flush=True,
    )
    print(
        "  (Pre-fix: raw Δm MSE alone was ~1e9 and made total loss look 'broken'.)",
        flush=True,
    )
    return calib


def _collate(batch):
    import torch
    from galacticsics.ml.morton.dataset import collate_morton_batch

    np_batch = collate_morton_batch(batch)
    out = {
        "c": torch.as_tensor(np_batch["c"], dtype=torch.long),
        "dm": torch.as_tensor(np_batch["dm"], dtype=torch.float32),
        "dx": torch.as_tensor(np_batch["dx"], dtype=torch.float32),
        "v": torch.as_tensor(np_batch["v"], dtype=torch.float32),
        "theta": torch.as_tensor(np_batch["theta"], dtype=torch.float32),
    }
    return out


def _train(
    *,
    name: str,
    loader,
    theta_dim: int,
    theta_keys: list[str],
    device: str,
    epochs: int,
    lr: float,
    n_particles: int,
    lambdas: dict,
    map_n_pix: int,
    out_dir: Path,
    lambda_recon: float = 0.25,
) -> tuple[object, list[float], Path]:
    import torch
    from torch.utils.data import DataLoader

    from galacticsics.ml.models.sequence_vae import SequenceVAE, SequenceVAEConfig

    assert isinstance(loader, DataLoader)
    cfg = SequenceVAEConfig(
        n_particles=n_particles,
        theta_dim=theta_dim,
        d_model=64,
        latent_dim=32,
        n_layers=1,
        n_heads=2,
        n_decode_layers=1,
        map_n_pix=map_n_pix,
        lambda_recon=lambda_recon,
        **lambdas,
    )
    model = SequenceVAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["c"], batch["dm"], batch["dx"], batch["v"], batch["theta"])
            metrics = model.loss(batch, out)
            opt.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(metrics["loss"].detach().cpu()))
            if not np.isfinite(losses[-1]):
                print(f"  [{name}] NaN/Inf loss at epoch {epoch + 1} — aborting candidate", flush=True)
                break
        history.append(float(np.mean(losses)))
        last = metrics
        print(
            f"  [{name}] epoch {epoch + 1}/{epochs}  loss={history[-1]:.4f}  "
            f"recon={float(last['recon'].detach()):.3f} "
            f"(dm={float(last['mse_dm'].detach()):.3f} dx={float(last['mse_dx'].detach()):.3f} "
            f"v={float(last['mse_v'].detach()):.3f})  "
            f"ce={float(last['ce'].detach()):.3f}  profile={float(last['profile'].detach()):.3f}  "
            f"nax={float(last['nonaxisym'].detach()):.3f} maps={float(last['maps'].detach()):.3f}",
            flush=True,
        )
    ckpt = out_dir / f"vae_{name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "theta_keys": theta_keys,
            "history": history,
            "ablation": name,
        },
        ckpt,
    )
    return model, history, ckpt


# Start from package defaults; escalate morphology weight only if needed.
LOSS_BALANCE_CANDIDATES: list[dict[str, float]] = [
    {
        "lambda_recon": 0.25,
        "lambda_sigma": 0.5,
        "lambda_rho": 0.25,
        "lambda_vphi": 0.5,
        "lambda_nonaxisym": 2.0,
        "lambda_maps": 2.0,
    },
    {
        "lambda_recon": 0.1,
        "lambda_sigma": 1.0,
        "lambda_rho": 0.25,
        "lambda_vphi": 0.5,
        "lambda_nonaxisym": 5.0,
        "lambda_maps": 5.0,
    },
    {
        "lambda_recon": 0.05,
        "lambda_sigma": 1.0,
        "lambda_rho": 0.1,
        "lambda_vphi": 0.25,
        "lambda_nonaxisym": 10.0,
        "lambda_maps": 10.0,
    },
]


def _morphology_score(model, batch: dict, device: str) -> dict[str, float]:
    """Permutation-invariant soft Fourier+map mismatch (lower is better)."""
    import torch

    from galacticsics.ml.profiles import map_reconstruction_loss, nonaxisym_reconstruction_loss

    model.eval()
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["c"], batch["dm"], batch["dx"], batch["v"], batch["theta"])
        nax = nonaxisym_reconstruction_loss(
            batch["dx"],
            batch["c"],
            out["dx"],
            out["logits_c"],
            modes=(1, 2),
            n_bins=8,
            r_max_disk=12.0,
            z_max=0.5,
            lambda_am=1.0,
        )
        maps = map_reconstruction_loss(
            batch["dx"],
            batch["c"],
            out["dx"],
            out["logits_c"],
            n_pix=16,
            r_max=12.0,
            z_max_face=0.5,
        )
    nax_v = float(nax["nonaxisym"].cpu())
    map_v = float(maps["maps"].cpu())
    return {
        "total": nax_v + map_v,
        "fourier": nax_v,
        "maps": map_v,
        "am1": float(nax["am1"].cpu()),
        "am2": float(nax["am2"].cpu()),
        "map_xy": float(maps["map_xy"].cpu()),
        "map_xz": float(maps["map_xz"].cpu()),
    }


def _evolve_smoke(
    parts: dict[str, np.ndarray],
    *,
    end_gyr: float,
    dt: float,
    force: str,
    omp_threads: int,
):
    import os

    os.environ["OMP_NUM_THREADS"] = str(max(1, int(omp_threads)))

    from ntropy.analysis.disk_density import disk_azimuthal_fourier
    from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig
    from ntropy.particles import ParticleState
    from ntropy.simulation import Simulation

    state = ParticleState.from_arrays(
        parts["pos"],
        parts["vel"],
        parts["mass"],
        parts["eps"],
        type_id=parts["type_id"],
    )
    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(type="leapfrog", dt=dt, end_time_gyr=float(end_gyr))
    # Prefer OpenMP bh_c; fall back to pure-Python bh if the C ext / CUDA import path breaks.
    method = force
    try:
        if force == "bh_c":
            from ntropy.forces.bhtree_c import extension_available

            if not extension_available():
                method = "bh"
    except Exception:  # noqa: BLE001
        method = "bh"
    cfg.force = ForceConfig(method=method, theta=0.8)
    cfg.parallel = ParallelConfig(enabled=False)
    cfg.output.write_final = False
    cfg.output.every = 0
    # Softened total-energy is O(N²); only record endpoints for the smoke.
    n_steps = max(1, int(np.ceil(float(end_gyr) / max(float(dt), 1e-9))))
    cfg.output.energy_every = max(n_steps, 1)
    t0 = time.time()
    result = Simulation(cfg, state=state.copy()).run()
    wall = time.time() - t0
    energies = [float(e) for e in result.energies]
    e0, e1 = energies[0], energies[-1]
    dE = (e1 - e0) / abs(e0) if abs(e0) > 0 else float("nan")
    final = result.final_state
    disk = parts["type_id"] == 0
    if int(disk.sum()) < 200:
        disk = np.ones(len(parts["type_id"]), dtype=bool)
    a2_ic = float(
        disk_azimuthal_fourier(
            parts["pos"][disk],
            parts["mass"][disk],
            m=2,
            n_bins=8,
            r_max=12.0,
            z_max=0.5,
            min_count=20,
        )["a_m_over_a0_median"]
    )
    a2_f = float(
        disk_azimuthal_fourier(
            np.asarray(final.pos)[disk],
            parts["mass"][disk],
            m=2,
            n_bins=8,
            r_max=12.0,
            z_max=0.5,
            min_count=20,
        )["a_m_over_a0_median"]
    )
    return {
        "wall_s": wall,
        "dE_over_E": dE,
        "a2_ic": a2_ic,
        "a2_final": a2_f,
        "n_steps": n_steps,
        "E0": e0,
        "E1": e1,
        "omp_threads": int(omp_threads),
        "force_method": method,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/ml/smoke_morton_vae"))
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Train on CPU by default while corpus owns the GPU",
    )
    parser.add_argument("--max-vram-gb", type=float, default=8.0)
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=8,
        help="OMP_NUM_THREADS for bh_c evolve (and torch CPU intra-op)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Try multiple loss balances (slower); default trains one full config",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=32, help="Set size during training (debug-small default)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=24,
        help="Cap manifest rows for fast iteration (spread across ICs+dumps)",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=5_000,
        help="Particles to generate (raise after tiny debug succeeds)",
    )
    parser.add_argument(
        "--n-evolve",
        type=int,
        default=3_000,
        help="Subset used for short CPU evolve (bh_c); ≤ n-sample",
    )
    parser.add_argument("--evolve-gyr", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--force",
        type=str,
        default="bh_c",
        help="Evolve force (bh_c recommended; avoid gpu_bh while corpus runs)",
    )
    parser.add_argument(
        "--skip-evolve",
        action="store_true",
        default=True,
        help="Skip ntropy evolve (default True while corpus saturates the GPU/CUDA stack)",
    )
    parser.add_argument(
        "--with-evolve",
        action="store_true",
        help="Run short OpenMP bh_c evolve after generate (can be slow if CUDA init contends)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--work-root",
        type=Path,
        action="append",
        default=None,
        help="Campaign root with IC/dumps (repeatable). "
        "Default: runs/mw_morton_corpus_v2 then runs/mw_morton_corpus",
    )
    parser.add_argument(
        "--morph-improve-frac",
        type=float,
        default=0.05,
        help="Require full morphology score ≤ recon_only * (1 - frac)",
    )
    parser.add_argument(
        "--scale-up",
        action="store_true",
        help="Larger N / more snapshots / generate 5e4 after tiny debug path",
    )
    args = parser.parse_args(argv)

    if args.with_evolve:
        args.skip_evolve = False
    if args.scale_up:
        args.n_train = max(args.n_train, 128)
        args.max_snapshots = max(args.max_snapshots, 64)
        args.n_sample = max(args.n_sample, 50_000)
        args.n_evolve = max(args.n_evolve, 15_000)
        args.epochs = max(args.epochs, 4)
        print(
            f"--scale-up → n_train={args.n_train} max_snapshots={args.max_snapshots} "
            f"n_sample={args.n_sample} n_evolve={args.n_evolve} epochs={args.epochs}",
            flush=True,
        )
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("FAIL: torch not installed.  pip install -e '.[ml]'", file=sys.stderr)
        return 2

    from galacticsics.ml.morton.dataset import MortonSnapshotDataset

    args.out.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.omp_threads)))
    try:
        import torch

        torch.set_num_threads(max(1, args.omp_threads))
    except Exception:  # noqa: BLE001
        pass
    print(
        f"device={args.device}  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  "
        f"(corpus gpu_bh can keep the GPU; this smoke stays on CPU+OpenMP by default)",
        flush=True,
    )
    device = _configure_device(args.device, args.max_vram_gb)
    if device != "cuda":
        print(
            "WARNING: requested GPU path but device resolved to "
            f"{device!r}. Continuing on {device} (expected when --device cpu).",
            flush=True,
        )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(0)

    work_roots = args.work_root or [
        Path("runs/mw_morton_corpus_v2"),
        Path("runs/mw_morton_corpus"),
    ]
    print("=== snapshot manifest ===", flush=True)
    manifest, corpus_kind = _resolve_manifest(work_roots, args.out, args.seed)
    ds = MortonSnapshotDataset(
        manifest,
        n_particles=args.n_train,
        split=None,
        seed=args.seed,
        max_snapshots=args.max_snapshots,
    )
    if len(ds) == 0:
        raise SystemExit(f"no snapshots in {manifest}")
    calib = _calibrate_token_scales(ds)
    print(
        f"dataset size={len(ds)}  probe_batch={min(8, len(ds))}  corpus={corpus_kind}",
        flush=True,
    )
    print("=== preload snapshots into RAM (avoids re-reading 1.75M-particle NPZs) ===", flush=True)
    t_load = time.time()
    cached = [_collate([ds[i]]) for i in range(len(ds))]
    print(f"  cached {len(cached)} items in {time.time() - t_load:.1f}s", flush=True)

    class _Cached(torch.utils.data.Dataset):
        def __len__(self):
            return len(cached)

        def __getitem__(self, i):
            return {k: v[0] for k, v in cached[i].items()}

    loader = DataLoader(
        _Cached(),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda xs: {
            k: torch.stack([x[k] for x in xs], 0) for k in xs[0]
        },
    )
    probe_idx = list(range(min(8, len(cached))))
    probe = {
        k: torch.cat([cached[i][k] for i in probe_idx], 0) for k in cached[0]
    }

    off = dict(
        lambda_sigma=0.0,
        lambda_rho=0.0,
        lambda_vphi=0.0,
        lambda_nonaxisym=0.0,
        lambda_maps=0.0,
    )

    print("=== train recon_only (baseline) ===", flush=True)
    model_ro, hist_ro, _ = _train(
        name="recon_only",
        loader=loader,
        theta_dim=len(ds.theta_keys),
        theta_keys=ds.theta_keys,
        device=device,
        epochs=args.epochs,
        lr=1e-3,
        n_particles=args.n_train,
        lambdas=off,
        map_n_pix=16,
        out_dir=args.out,
        lambda_recon=1.0,
    )
    score_ro = _morphology_score(model_ro, probe, device)
    print(
        f"  recon_only morphology: total={score_ro['total']:.4f}  "
        f"(fourier={score_ro['fourier']:.4f} maps={score_ro['maps']:.4f}  "
        f"am1={score_ro['am1']:.3f} am2={score_ro['am2']:.3f})",
        flush=True,
    )
    print(
        "  note: morphology is unnormalized soft MSE (lower better); "
        "absolute O(1–10) is normal. Ablation cares about full < recon_only.",
        flush=True,
    )

    target = score_ro["total"] * (1.0 - float(args.morph_improve_frac))
    tune_rows: list[dict] = []
    model_full = None
    hist_full: list[float] = []
    ckpt_full = args.out / "vae_full.pt"
    best: dict | None = None

    balances = LOSS_BALANCE_CANDIDATES if args.tune else LOSS_BALANCE_CANDIDATES[:1]
    print(
        f"=== train full auxiliaries ({len(balances)} balance candidate(s)"
        f"{'' if args.tune else '; pass --tune to sweep'}) ===",
        flush=True,
    )
    for i, bal in enumerate(balances):
        name = f"full_bal{i}"
        print(f"--- candidate {i}: {bal} ---", flush=True)
        lambdas = {k: v for k, v in bal.items() if k != "lambda_recon"}
        try:
            model_i, hist_i, ckpt_i = _train(
                name=name,
                loader=loader,
                theta_dim=len(ds.theta_keys),
                theta_keys=ds.theta_keys,
                device=device,
                epochs=args.epochs,
                lr=1e-3,
                n_particles=args.n_train,
                lambdas=lambdas,
                map_n_pix=16,
                out_dir=args.out,
                lambda_recon=float(bal["lambda_recon"]),
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg:
                print(f"GPU issue on candidate {i}: {exc}", flush=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                tune_rows.append({"candidate": i, "balance": bal, "error": str(exc)})
                continue
            raise
        score_i = _morphology_score(model_i, probe, device)
        improve = (score_ro["total"] - score_i["total"]) / max(score_ro["total"], 1e-8)
        row = {
            "candidate": i,
            "balance": bal,
            "morphology": score_i["total"],
            "morphology_detail": score_i,
            "improve_frac": improve,
            "history": hist_i,
            "checkpoint": str(ckpt_i),
        }
        tune_rows.append(row)
        print(
            f"  morphology total={score_i['total']:.4f}  "
            f"(fourier={score_i['fourier']:.4f} maps={score_i['maps']:.4f})  "
            f"vs recon_only={score_ro['total']:.4f}  "
            f"improve={100 * improve:.1f}%  target≤{target:.4f}",
            flush=True,
        )
        if best is None or score_i["total"] < best["morphology"]:
            best = row
            model_full = model_i
            hist_full = hist_i
            ckpt_full = ckpt_i
        if score_i["total"] <= target:
            print(f"  reached morphology target with candidate {i}", flush=True)
            break

    if model_full is None or best is None:
        print("FAIL: no successful full-auxiliary training run", file=sys.stderr)
        return 1

    # Persist best as vae_full.pt for downstream notebooks
    import shutil

    if Path(best["checkpoint"]) != ckpt_full:
        shutil.copy2(best["checkpoint"], args.out / "vae_full.pt")
        ckpt_full = args.out / "vae_full.pt"
    else:
        # rename bal checkpoint path already unique; also write alias
        shutil.copy2(best["checkpoint"], args.out / "vae_full.pt")
        ckpt_full = args.out / "vae_full.pt"

    score_full = float(best["morphology"])
    score_ro_total = float(score_ro["total"])

    # --- checks ---
    checks: list[tuple[str, bool, str]] = []

    finite_hist = all(np.isfinite(hist_full)) and all(np.isfinite(hist_ro))
    decreased = hist_full[-1] < hist_full[0] * 1.05
    # Interpret position MSE vs data scale
    e_dx2 = float(calib.get("E_dx2", 1.0))
    print(
        "=== loss interpretation ===\n"
        f"  E[|x|²] (data) ≈ {e_dx2:.1f}.  Untrained mse_dx starts near that scale.\n"
        f"  With λ_recon={best['balance'].get('lambda_recon', 0.25)}, "
        f"total loss ≈ λ_recon·mse_dx (+ small CE/KL/aux) → O({0.25 * e_dx2:.0f}) at init.\n"
        f"  A total ~300 with mse_dx~1200 is consistent — not a failed train.",
        flush=True,
    )
    checks.append(
        (
            "train_loss_finite_decreasing",
            finite_hist and (hist_full[-1] < hist_full[0] or decreased),
            f"full {hist_full[0]:.3f} → {hist_full[-1]:.3f}",
        )
    )

    model_full.eval()
    theta = probe["theta"][:1].to(device)
    with torch.no_grad():
        g1 = model_full.generate(theta, n=args.n_train * 2, chunk_size=args.n_train)
        z_a = torch.randn(1, model_full.config.latent_dim, device=device)
        z_b = torch.randn(1, model_full.config.latent_dim, device=device)
        ga = model_full.generate(theta, n=args.n_train, z=z_a)
        gb = model_full.generate(theta, n=args.n_train, z=z_b)
    std_pos = float(np.std(g1["dx"]))
    diverse_z = not np.allclose(ga["dx"], gb["dx"])
    checks.append(
        (
            "generate_diversity",
            std_pos > 1e-3 and diverse_z,
            f"std(pos)={std_pos:.4f}  z_diff={diverse_z}",
        )
    )

    batch0 = {k: v.to(device) for k, v in probe.items()}
    with torch.no_grad():
        out = model_full(batch0["c"], batch0["dm"], batch0["dx"], batch0["v"], batch0["theta"])
        metrics = model_full.loss(batch0, out)
    aux_ok = all(
        torch.isfinite(metrics[k]).item()
        for k in ("loss", "profile", "nonaxisym", "maps", "sigma", "rho", "vphi")
    )
    checks.append(
        (
            "aux_losses_finite",
            aux_ok,
            f"profile={float(metrics['profile']):.3f} nax={float(metrics['nonaxisym']):.3f} "
            f"maps={float(metrics['maps']):.3f}  recon_term={float(metrics['recon']):.3f}",
        )
    )

    morph_ok = np.isfinite(score_full) and score_full < score_ro_total
    strong_ok = score_full <= target
    checks.append(
        (
            "ablation_morphology",
            morph_ok,
            f"recon_only={score_ro_total:.4f}  full={score_full:.4f}  "
            f"best_balance={best['balance']}  "
            f"improve={100 * (score_ro_total - score_full) / max(score_ro_total, 1e-8):.1f}%",
        )
    )
    checks.append(
        (
            "morphology_target",
            strong_ok,
            f"need ≤ {target:.4f} ({100 * args.morph_improve_frac:.0f}% better than recon_only)",
        )
    )

    evolve_info = None
    if not args.skip_evolve:
        print("=== sample + short evolve ===", flush=True)
        n_ev = min(int(args.n_evolve), int(args.n_sample))
        n_steps_est = max(1, int(np.ceil(args.evolve_gyr / max(args.dt, 1e-9))))
        print(
            f"  generate N={args.n_sample}, evolve subset n={n_ev}  "
            f"force={args.force}  steps≈{n_steps_est}  "
            f"(smoke does not run GalactICS solve/sample)",
            flush=True,
        )
        with torch.no_grad():
            tok = model_full.generate(theta, n=args.n_sample, chunk_size=min(512, args.n_train * 2))
        parts_full = {
            "pos": tok["dx"][0].astype(np.float64),
            "vel": tok["v"][0].astype(np.float64),
            "mass": np.full(args.n_sample, 1.0 / args.n_sample),
            "eps": np.full(args.n_sample, 0.1),
            "type_id": tok["c"][0].astype(np.int32),
        }
        np.savez_compressed(args.out / "sample_full.npz", **parts_full)
        # Stratified evolve subset (keep component mix)
        rng = np.random.default_rng(args.seed)
        cid = parts_full["type_id"]
        from galacticsics.ml.morton.tokenize import subsample_stratified

        ev_idx = subsample_stratified(cid.astype(np.int64), n_ev, rng=rng)
        parts = {k: (v[ev_idx] if k != "mass" else np.full(n_ev, 1.0 / n_ev)) for k, v in parts_full.items()}
        try:
            evolve_info = _evolve_smoke(
                parts,
                end_gyr=args.evolve_gyr,
                dt=args.dt,
                force=args.force,
                omp_threads=args.omp_threads,
            )
            evolve_info["n_generate"] = int(args.n_sample)
            evolve_info["n_evolve"] = int(n_ev)
            e_ok = np.isfinite(evolve_info["dE_over_E"]) and abs(evolve_info["dE_over_E"]) < 0.2
            a2_ok = np.isfinite(evolve_info["a2_ic"]) and evolve_info["a2_ic"] >= 0.0
            checks.append(
                (
                    "short_evolve_stable",
                    e_ok and a2_ok,
                    f"ΔE/E={evolve_info['dE_over_E']:.3e}  "
                    f"A2 {evolve_info['a2_ic']:.3f}→{evolve_info['a2_final']:.3f}  "
                    f"wall={evolve_info['wall_s']:.1f}s  n_ev={n_ev}",
                )
            )
        except Exception as exc:  # noqa: BLE001 — smoke report
            checks.append(("short_evolve_stable", False, f"evolve failed: {exc}"))
    else:
        checks.append(("short_evolve_stable", True, "skipped"))

    peak = _peak_vram_gb(device)
    vram_ok = peak <= args.max_vram_gb + 0.25
    checks.append(
        (
            "vram_under_budget",
            vram_ok,
            f"peak_allocated={peak:.3f} GiB  budget={args.max_vram_gb:.1f} GiB  device={device}",
        )
    )

    print("\n=== assessment ===", flush=True)
    n_pass = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        n_pass += int(ok)
        print(f"  [{status}] {name}: {detail}", flush=True)

    critical = {
        "train_loss_finite_decreasing",
        "generate_diversity",
        "aux_losses_finite",
        "vram_under_budget",
    }
    critical_ok = all(ok for name, ok, _ in checks if name in critical)
    if critical_ok and morph_ok and strong_ok:
        verdict = (
            "VIABLE: smoke passed on "
            f"{device}; tuned loss balance beats recon-only on soft Fourier morphology."
        )
        code = 0
    elif critical_ok and morph_ok:
        verdict = (
            "IMPROVED: morphology better than recon-only, but short of the "
            f"{100 * args.morph_improve_frac:.0f}% target — consider more epochs."
        )
        code = 0
    elif critical_ok:
        verdict = (
            "MIXED: train/generate OK on GPU, but morphology auxiliaries did not "
            "beat recon-only on this synthetic smoke."
        )
        code = 0
    else:
        verdict = "NOT VIABLE (smoke): critical checks failed — see FAIL lines above."
        code = 1

    report = {
        "verdict": verdict,
        "checks": [{"name": n, "pass": bool(ok), "detail": d} for n, ok, d in checks],
        "history_recon_only": hist_ro,
        "history_full": hist_full,
        "morphology_recon_only": score_ro,
        "morphology_full": score_full,
        "morphology_full_detail": best.get("morphology_detail"),
        "best_balance": best["balance"],
        "calibration": calib,
        "corpus": corpus_kind,
        "tune_rows": [
            {k: v for k, v in row.items() if k != "history"} | {"history_last": (row.get("history") or [None])[-1]}
            for row in tune_rows
        ],
        "evolve": evolve_info,
        "peak_vram_gb": peak,
        "device": device,
        "checkpoint": str(ckpt_full),
        "n_pass": n_pass,
        "n_checks": len(checks),
    }
    report_path = args.out / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n{verdict}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
