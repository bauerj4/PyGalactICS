#!/usr/bin/env python3
"""
CPU near-prod smoke: multi-scale slice fields → multi-tower U-Net AE → resample.

Default **crisp** track: progressive 128² disk + σ, dens+moment-weighted recon
(Fourier off — dens/⟨v⟩/σ at high res should carry non-axisym structure),
count-stratified resample, optional short OpenMP BH evolve.

    . .venv/bin/activate
    OMP_NUM_THREADS=6 python scripts/smoke_field_maps.py --with-evolve
    # ablate Fourier back on:
    OMP_NUM_THREADS=6 python scripts/smoke_field_maps.py --a2-weight 2.5 --with-evolve
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

from galacticsics.ml.fields.autoencoder import (
    MultiTowerSliceAE,
    dens_map_azimuthal_fourier_numpy,
    load_compatible_towers,
    multitower_reconstruction_loss,
)
from galacticsics.ml.fields.binning import (
    MultiScaleSliceConfig,
    MultiScaleVoxelConfig,
    bin_multiscale_slice_stacks,
    scale_summary,
)
from galacticsics.ml.fields.dataset import (
    MultiScaleFieldDataset,
    bin_multiscale_voxels_from_path,
    collate_multiscale_batch,
)
from galacticsics.ml.fields.normalize import denormalize_stack, normalize_stack
from galacticsics.ml.fields.potential import plummer_potential_multiscale
from galacticsics.ml.fields.frame import prepare_shared_frame
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from galacticsics.ml.morton.tokenize import _component_ids
from ntropy.analysis.disk_density import bin_plane_density, disk_azimuthal_fourier


MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
WORK = Path("runs/mw_morton_corpus_v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

HIRES_64 = {
    "best_loss": 0.313,
    "bar_a2_data": 0.375,
    "bar_a2_resampled": 0.264,
    "quiet_a2_resampled": 0.013,
    "note": "64² dens+σ U-Net, dens_weight=6, scalar A₂ loss, 30 epochs",
}

BASELINE_32 = {
    "best_loss": 0.698,
    "bar_a2_data": 0.375,
    "bar_a2_resampled": 0.119,
    "quiet_a2_resampled": 0.036,
    "note": "32² dens+⟨v⟩ shallow MultiTower CNN, dens_weight=3, 25 epochs",
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

FOURIER_MODES = (1, 2, 3)

# Corpus particle-count mix (disk:halo:bulge ≈ 4:2:1). Mass-weighted N splits
# starve the disk (~1%) and make A₂ / evolve panels meaningless.
COUNT_FRACTIONS = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}


def _a2_median(pos, mass) -> float:
    out = disk_azimuthal_fourier(
        pos, mass, m=2, r_max=12.0, n_bins=12, z_max=0.5, min_count=10
    )
    return float(out["a_m_over_a0_median"])


def _am_profiles(pos, mass, *, modes=FOURIER_MODES, r_max=12.0, n_bins=12, z_max=0.5):
    """Particle ``A_m(R)`` curves for m in ``modes`` (shared radial grid)."""
    if pos is None or len(pos) < 50:
        return {
            "r_mid": np.linspace(r_max / n_bins / 2, r_max - r_max / n_bins / 2, n_bins),
            "modes": {
                int(m): {"a_m_over_a0": np.full(n_bins, np.nan), "median": float("nan")}
                for m in modes
            },
        }
    profiles = {}
    r_mid = None
    for m in modes:
        out = disk_azimuthal_fourier(
            pos, mass, m=int(m), r_max=r_max, n_bins=n_bins, z_max=z_max, min_count=10
        )
        if r_mid is None:
            r_mid = np.asarray(out["r_mid"], dtype=np.float64)
        profiles[int(m)] = {
            "a_m_over_a0": np.asarray(out["a_m_over_a0"], dtype=np.float64),
            "median": float(out["a_m_over_a0_median"]),
        }
    return {"r_mid": r_mid, "modes": profiles}


def _profile_mse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.mean((a[mask] - b[mask]) ** 2))


def _profile_mse_radial(
    r_mid: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    r_split: float = 6.0,
) -> dict[str, float]:
    """Overall + inner/outer ``A_m(R)`` MSE (default split at 6 kpc)."""
    r = np.asarray(r_mid, dtype=np.float64)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return {
        "overall": _profile_mse(aa, bb),
        "inner": _profile_mse(aa[r < r_split], bb[r < r_split]),
        "outer": _profile_mse(aa[r >= r_split], bb[r >= r_split]),
    }


def _load_shared_frame(path: Path):
    """Load snapshot on one global COM frame (never per-component recenter)."""
    with np.load(path, allow_pickle=True) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        tags = data["tags"] if "tags" in data.files else None
        type_id = data["type_id"] if "type_id" in data.files else None
    cid = _component_ids(tags, type_id, pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    return pos, vel, mass, cid


def _disk_dens_collapse(stack: np.ndarray, n_z: int, n_mom: int) -> np.ndarray:
    dens = np.zeros(stack.shape[-2:], dtype=np.float64)
    for iz in range(n_z):
        dens += np.maximum(stack[iz * n_mom], 0.0)
    return dens


def _plot_panel(out_path, *, title, data_dens, recon_dens, particle_dens, a2_data, a2_rec, mse):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, img, lab in zip(
        axes,
        (data_dens, recon_dens, particle_dens),
        (
            f"data disk Σ\nA₂ med≈{a2_data:.2f}",
            f"AE recon dens\nmse_dens={mse:.4f}",
            f"particles←recon\nA₂ med≈{a2_rec:.2f}",
        ),
    ):
        pos = img[img > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        ax.imshow(
            np.log1p(np.maximum(img, 0.0)),
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=np.log1p(vmax),
        )
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_am_profiles(
    out_path,
    *,
    title: str,
    data_prof: dict,
    rec_prof: dict,
    evo_prof: dict | None = None,
    modes=FOURIER_MODES,
):
    """Plot ``A_m(R)`` curves — morphology lives in the radial shape, not medians."""
    n = len(modes)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), sharey=False)
    if n == 1:
        axes = [axes]
    r = data_prof["r_mid"]
    for ax, m in zip(axes, modes):
        d = data_prof["modes"][int(m)]["a_m_over_a0"]
        p = rec_prof["modes"][int(m)]["a_m_over_a0"]
        ax.plot(r, d, "k-", lw=1.8, label="data")
        ax.plot(r, p, color="#c45c26", lw=1.6, label="resampled")
        if evo_prof is not None:
            e = evo_prof["modes"][int(m)]["a_m_over_a0"]
            ax.plot(r, e, color="#2a6f97", lw=1.4, ls="--", label="evolved")
        ax.set_xlabel("R [kpc]")
        ax.set_ylabel(f"$A_{m}/A_0$")
        ax.set_title(f"m={m}")
        ax.set_xlim(0, float(r[-1]) if r is not None and len(r) else 12)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_evolve_panel(
    out_path,
    *,
    title,
    data_dens,
    recon_dens,
    resampled_dens,
    evolved_dens,
    labels,
):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, img, lab in zip(axes, (data_dens, recon_dens, resampled_dens, evolved_dens), labels):
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


def _plot_comparison(out_path, *, bar, quiet, baseline=HIRES_64):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    labels = ["bar A₂ med\n(data)", "bar A₂ med\n(resampled)", "quiet A₂ med\n(resampled)"]
    base_vals = [
        baseline["bar_a2_data"],
        baseline["bar_a2_resampled"],
        baseline["quiet_a2_resampled"],
    ]
    new_vals = [bar["a2_data"], bar["a2_resampled"], quiet["a2_resampled"]]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, base_vals, w, label="64² hires", color="#8c8c8c")
    ax.bar(x + w / 2, new_vals, w, label="this run", color="#c45c26")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("A₂ median (summary only)")
    ax.set_title("Field-map morphology vs 64² hires (see A_m(R) plots)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _evolve_bh(
    parts: dict[str, np.ndarray],
    *,
    end_gyr: float,
    dt: float,
    force: str,
    omp_threads: int,
    timeout_s: float = 600.0,
):
    """
    Short OpenMP BH evolve via ``bh_c`` leapfrog (no Simulation / CUDA import).

    Falls back to pure-Python ``bh`` only if the C extension is unavailable.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    os.environ["OMP_NUM_THREADS"] = str(max(1, int(omp_threads)))
    n_steps = max(1, int(np.ceil(float(end_gyr) / max(float(dt), 1e-9))))
    pos0 = np.asarray(parts["pos"], dtype=np.float64)
    vel0 = np.asarray(parts["vel"], dtype=np.float64)
    mass = np.asarray(parts["mass"], dtype=np.float64)
    eps = np.asarray(parts["eps"], dtype=np.float64)
    method = force

    def _forces(p):
        nonlocal method
        if method == "bh_c":
            try:
                from ntropy.forces.bhtree_c import compute_forces_bh_c, extension_available

                if extension_available():
                    return compute_forces_bh_c(p, mass, eps, theta=0.8)
            except Exception:  # noqa: BLE001
                method = "bh"
        # Import submodule directly to avoid CUDA init in forces.__init__.
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
        # Softened pairwise energy is O(N²); skip and report NaN ΔE.
        return pos, vel, time.time() - t0

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            pos_f, _vel_f, wall = ex.submit(_run).result(timeout=float(timeout_s))
    except FuturesTimeout:
        return {
            "ok": False,
            "timed_out": True,
            "timeout_s": float(timeout_s),
            "n_steps": n_steps,
            "omp_threads": int(omp_threads),
            "force_method": method,
            "error": f"evolve timed out after {timeout_s}s",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "timed_out": False,
            "n_steps": n_steps,
            "omp_threads": int(omp_threads),
            "force_method": method,
            "error": f"{type(exc).__name__}: {exc}",
        }

    com0 = np.average(pos0, axis=0, weights=mass)
    com1 = np.average(pos_f, axis=0, weights=mass)
    return {
        "ok": True,
        "timed_out": False,
        "wall_s": wall,
        "dE_over_E": float("nan"),
        "n_steps": n_steps,
        "E0": float("nan"),
        "E1": float("nan"),
        "omp_threads": int(omp_threads),
        "force_method": method,
        "com_drift_kpc": float(np.linalg.norm(com1 - com0)),
        "pos_final": np.asarray(pos_f, dtype=np.float64),
    }


def _build_cfg(args) -> MultiScaleSliceConfig:
    moment_set = args.moment_set
    if args.preset == "baseline32":
        return MultiScaleSliceConfig.baseline_32_defaults(
            include_potential=args.include_potential, moment_set=moment_set
        )
    if args.preset == "smoke":
        return MultiScaleSliceConfig.smoke_defaults(
            include_potential=args.include_potential, moment_set=moment_set
        )
    if args.preset == "cusp_bulge":
        return MultiScaleSliceConfig.cusp_bulge_defaults(
            disk_n_pix=args.disk_n_pix,
            include_potential=args.include_potential,
            moment_set=moment_set,
            bulge_n_pix=args.bulge_n_pix or 128,
            bulge_n_z=args.bulge_n_z or 32,
            bulge_z_max=4.0 if args.bulge_z_max is None else float(args.bulge_z_max),
            disk_n_z=args.disk_n_z,
            halo_n_z=args.halo_n_z,
        )
    return MultiScaleSliceConfig.progressive_defaults(
        disk_n_pix=args.disk_n_pix,
        include_potential=args.include_potential,
        moment_set=moment_set,
        disk_n_z=args.disk_n_z,
        bulge_n_z=args.bulge_n_z,
        halo_n_z=args.halo_n_z,
        bulge_z_max=args.bulge_z_max,
        bulge_z_spacing=args.bulge_z_spacing,
        bulge_n_pix=args.bulge_n_pix,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="artifact dir (default: runs/ml/field_maps/crisp_YYYY-MM-DD)",
    )
    p.add_argument(
        "--preset",
        choices=("smoke", "progressive", "baseline32", "cusp_bulge"),
        default="progressive",
        help="grid preset (default progressive; cusp_bulge = fine bulge tower)",
    )
    p.add_argument(
        "--disk-n-pix",
        type=int,
        default=128,
        help="for --preset progressive (default 128² crisp)",
    )
    p.add_argument(
        "--disk-n-z",
        type=int,
        default=None,
        help="override progressive disk n_z (vertical slabs; default tiered by n_pix)",
    )
    p.add_argument(
        "--bulge-n-z",
        type=int,
        default=None,
        help="override progressive bulge n_z",
    )
    p.add_argument(
        "--halo-n-z",
        type=int,
        default=None,
        help="override progressive halo n_z",
    )
    p.add_argument(
        "--bulge-z-max",
        type=float,
        default=None,
        help="override bulge |z| half-height (default 4 kpc); shrink to resolve cusp",
    )
    p.add_argument(
        "--bulge-z-spacing",
        choices=("uniform", "midplane"),
        default=None,
        help="override bulge z spacing (midplane packs Δz near z=0)",
    )
    p.add_argument(
        "--bulge-n-pix",
        type=int,
        default=None,
        help="override bulge n_pix (must match checkpoint if warm-starting)",
    )
    p.add_argument(
        "--moment-set",
        choices=("base", "disp", "full"),
        default="disp",
        help="channel layout: dens+⟨v⟩ / +σ / +β",
    )
    p.add_argument(
        "--deep-heads",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="3×3 dens/moment heads (default: False for crisp track; auto from --init-checkpoint)",
    )
    p.add_argument("--epochs", type=int, default=36)
    p.add_argument("--dens-finetune-epochs", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max-snap", type=int, default=28)
    p.add_argument(
        "--bar-frac",
        type=float,
        default=None,
        help="Oversample barred dumps when subsampling (e.g. 0.75); needs A2 rank JSON",
    )
    p.add_argument(
        "--bar-a2-floor",
        type=float,
        default=0.20,
        help="Particle A2 floor for --bar-frac barred pool",
    )
    p.add_argument(
        "--a2-rank",
        type=Path,
        default=Path("runs/ml/field_maps/corpus_particle_a2_rank.json"),
        help="corpus particle A2 rank JSON for barred-heavy sampling",
    )
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--dens-weight", type=float, default=4.0, help="loss weight on dens")
    p.add_argument(
        "--moment-weight",
        type=float,
        default=6.0,
        help="loss weight on ⟨v⟩/σ (≥ dens for phase-space fidelity)",
    )
    p.add_argument(
        "--dens-mass-weight",
        type=float,
        default=0.0,
        help="cusp emphasis: weight dens MSE by 1+α·target/mean (0=off)",
    )
    p.add_argument(
        "--bulge-dens-mass-weight",
        type=float,
        default=None,
        help="override dens-mass-weight for bulge tower only",
    )
    p.add_argument(
        "--comp-weight-disk",
        type=float,
        default=2.5,
        help="multitower loss weight for disk",
    )
    p.add_argument(
        "--comp-weight-bulge",
        type=float,
        default=1.0,
        help="multitower loss weight for bulge",
    )
    p.add_argument(
        "--comp-weight-halo",
        type=float,
        default=0.4,
        help="multitower loss weight for halo",
    )
    p.add_argument(
        "--dens-grad-weight",
        type=float,
        default=0.0,
        help="match ∇ dens (optional; 0=off — hurt bars in crisp_v3)",
    )
    p.add_argument(
        "--a2-weight",
        type=float,
        default=0.0,
        help="radial A_m(R) Fourier weight (0=off; dens+moments first)",
    )
    p.add_argument(
        "--dens-resid-weight",
        type=float,
        default=0.0,
        help="axisym-subtracted dens residual MSE (quiet-safe morphology match)",
    )
    p.add_argument(
        "--fourier-amp-weight",
        type=float,
        default=2.0,
        help="amp vs phase emphasis inside soft Fourier match",
    )
    p.add_argument(
        "--fourier-quiet-gate",
        type=float,
        default=0.05,
        help="soft-gate Fourier when target mean A_m below this (avoid quiet invention)",
    )
    p.add_argument(
        "--fft-weight",
        type=float,
        default=0.0,
        help="spatial FFT morphology on axisym-residual dens (0=off; disk-heavy)",
    )
    p.add_argument(
        "--fft-lambda-phase",
        type=float,
        default=0.15,
        help="light FFT phase cosine weight inside spatial FFT morphology",
    )
    p.add_argument(
        "--fft-quiet-gate",
        type=float,
        default=None,
        help="residual-RMS quiet gate for FFT loss (default: --fourier-quiet-gate)",
    )
    p.add_argument(
        "--fft-k-floor",
        type=float,
        default=0.08,
        help="FFT k high-pass floor (fraction of Nyquist; damp k≈0)",
    )
    p.add_argument(
        "--moment-phys-weight",
        type=float,
        default=0.0,
        help="√Σ-weighted moment field loss (DF option A; 0=off)",
    )
    p.add_argument(
        "--vphi-phys-weight",
        type=float,
        default=0.0,
        help="√Σ-weighted ⟨v_φ⟩ map loss from vx,vy (DF option A; 0=off)",
    )
    p.add_argument(
        "--r-focus-rd",
        action="store_true",
        help=(
            "Focus quiet-gated Am / dens-residual / FFT morphology on the ring "
            "near θ disk.scale_length R_d (and optional --r-focus-band). "
            "Normalised as R_d / disk r_max on the FOV map."
        ),
    )
    p.add_argument(
        "--r-focus-band",
        type=str,
        default="0.5,1.5",
        help="Radial band as lo,hi fractions of R_d (default 0.5,1.5)",
    )
    p.add_argument(
        "--r-focus-peak",
        type=float,
        default=4.0,
        help="Peak radial-focus weight at R_d (Gaussian bump)",
    )
    p.add_argument(
        "--r-focus-floor",
        type=float,
        default=0.25,
        help="Floor weight outside the R_d band (keeps global match alive)",
    )
    p.add_argument(
        "--a2-rd-weight",
        type=float,
        default=0.0,
        help=(
            "Dedicated soft A₂(R=R_d) match (+ undershoot hinge); requires "
            "--r-focus-rd. Disk tower only."
        ),
    )
    p.add_argument(
        "--evolve-gate-t-lo",
        type=float,
        default=0.25,
        help="Soft evolve-gate: upweight samples with t_gyr in [lo, hi] Gyr",
    )
    p.add_argument(
        "--evolve-gate-t-hi",
        type=float,
        default=0.5,
        help="Soft evolve-gate upper t_gyr [Gyr]",
    )
    p.add_argument(
        "--evolve-gate-t-boost",
        type=float,
        default=0.0,
        help=(
            "Multiply batch loss by 1+boost when t_gyr ∈ [lo,hi] (soft match of "
            "face-on Σ / A₂ at mid-evolve times vs data; 0=off)"
        ),
    )
    p.add_argument("--fourier-n-bins", type=int, default=12)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--latent-channels", type=int, default=128)
    p.add_argument("--arch", choices=("unet", "shallow"), default="unet")
    p.add_argument("--no-cross-tower", action="store_true")
    p.add_argument("--include-potential", action="store_true")
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="warm-start from multitower_slice_ae.pt (e.g. crisp teacher)",
    )
    p.add_argument(
        "--init-compatible-towers",
        type=str,
        default=None,
        help=(
            "comma list of towers to warm-start when shapes differ "
            "(e.g. disk,halo for cusp_bulge); default=strict full load"
        ),
    )
    p.add_argument("--n-resample", type=int, default=120_000)
    p.add_argument("--n-evolve", type=int, default=40_000)
    p.add_argument("--evolve-gyr", type=float, default=0.02)
    p.add_argument("--evolve-timeout", type=float, default=600.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--force", default="bh_c")
    p.add_argument("--omp-threads", type=int, default=6)
    p.add_argument(
        "--with-evolve",
        action="store_true",
        help="short OpenMP BH evolve of multi-component resample",
    )
    p.add_argument(
        "--mass-weighted-counts",
        action="store_true",
        help="split resample N by mass (starves disk); default=corpus count mix 4:2:1",
    )
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--work", type=Path, default=WORK)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or Path(f"runs/ml/field_maps/crisp_{date.today().isoformat()}")
    omp = max(1, int(args.omp_threads))
    os.environ["OMP_NUM_THREADS"] = str(omp)
    torch.set_num_threads(omp)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    print(f"artifacts → {out}  (OMP_NUM_THREADS={omp} DEVICE={DEVICE})", flush=True)

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
        include_potential=args.include_potential,
        potential_n_sub=512,
        bar_frac=args.bar_frac,
        bar_a2_floor=float(args.bar_a2_floor),
        a2_rank_path=args.a2_rank,
    )
    print(
        f"dataset size={len(ds)}; bar_frac={args.bar_frac} "
        f"bar_a2_floor={args.bar_a2_floor}; preloading…",
        flush=True,
    )
    t0 = time.time()
    ds.preload()
    print(f"  preload {time.time() - t0:.1f}s; fitting norm stats…")
    t0 = time.time()
    stats = ds.fit_norm_stats(n_samples=min(10, len(ds)))
    print(
        f"  norm done ({time.time() - t0:.1f}s) dens_scales="
        + str({k: f"{v.dens_scale:.3g}" for k, v in stats.items()})
    )

    ex0 = EXAMPLES[0]
    vox = bin_multiscale_voxels_from_path(
        args.work / ex0["run"] / ex0["dump"],
        cfg=MultiScaleVoxelConfig.smoke_defaults(moment_set=args.moment_set),
    )
    print("voxel shapes:", {k: tuple(v.shape) for k, v in vox.items()})

    dens_idx = ds.dens_indices()

    init_ckpt = None
    # Crisp track (best particle A₂) uses shallow 1×1 dens/moment heads.
    deep_heads = False if args.deep_heads is None else bool(args.deep_heads)
    if args.init_checkpoint is not None:
        ckpt_path = Path(args.init_checkpoint)
        if not ckpt_path.is_file():
            raise SystemExit(f"missing --init-checkpoint {ckpt_path}")
        init_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_deep = any(k.endswith("dens_head.0.weight") for k in init_ckpt["model"])
        if args.deep_heads is None:
            deep_heads = ckpt_deep
        elif bool(args.deep_heads) != ckpt_deep:
            raise SystemExit(
                f"--deep-heads={args.deep_heads} conflicts with checkpoint "
                f"(detected deep_heads={ckpt_deep})"
            )
        if "norm" in init_ckpt:
            from galacticsics.ml.fields.normalize import FieldNormStats

            stats = {k: FieldNormStats(**raw) for k, raw in init_ckpt["norm"].items()}
            ds.norm_stats = stats
            print(f"reusing norm stats from {ckpt_path}", flush=True)

    def _collate(batch):
        b = collate_multiscale_batch(batch)
        stacks = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in b["stacks"].items()
        }
        out: dict = {"stacks": stacks}
        if "theta" in b:
            out["theta"] = b["theta"]
        if "t_gyr" in b:
            out["t_gyr"] = b["t_gyr"]
        return out

    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=_collate
    )
    model = MultiTowerSliceAE(
        cfg,
        include_potential=args.include_potential,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        arch=args.arch,
        separate_heads=True,
        deep_heads=deep_heads,
        cross_tower_attention=not args.no_cross_tower,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"MultiTowerSliceAE arch={args.arch} deep_heads={deep_heads} params={n_params:,}"
    )
    dens_scales = {k: float(stats[k].dens_scale) for k in stats}
    if init_ckpt is not None:
        if args.init_compatible_towers:
            towers = tuple(
                t.strip() for t in str(args.init_compatible_towers).split(",") if t.strip()
            )
            info = load_compatible_towers(model, init_ckpt["model"], towers=towers)
            print(
                f"compatible warm-start from {args.init_checkpoint} "
                f"towers={towers} loaded={len(info['loaded'])} "
                f"skipped={len(info['skipped'])}",
                flush=True,
            )
        else:
            model.load_state_dict(init_ckpt["model"], strict=True)
            print(f"warm-start from {args.init_checkpoint}", flush=True)
    fft_quiet = (
        float(args.fourier_quiet_gate)
        if args.fft_quiet_gate is None
        else float(args.fft_quiet_gate)
    )
    _band = tuple(float(x) for x in str(args.r_focus_band).split(",") if x.strip())
    if len(_band) != 2:
        _band = (0.5, 1.5)
    args._r_focus_band = _band  # type: ignore[attr-defined]
    disk_r_max = float(next(g.r_max for g in cfg.grids if g.name == "disk"))
    args._disk_r_max = disk_r_max  # type: ignore[attr-defined]
    theta_keys = list(getattr(ds, "theta_keys", []))
    rd_key = "disk.scale_length"
    t_key = "t_gyr"
    args._rd_theta_idx = (  # type: ignore[attr-defined]
        theta_keys.index(rd_key) if rd_key in theta_keys else None
    )
    args._t_theta_idx = (  # type: ignore[attr-defined]
        theta_keys.index(t_key) if t_key in theta_keys else None
    )
    print(
        f"loss: dens_w={args.dens_weight} moment_w={args.moment_weight} "
        f"dens_grad_w={args.dens_grad_weight} dens_resid_w={args.dens_resid_weight} "
        f"fourier_w={args.a2_weight} amp_w={args.fourier_amp_weight} "
        f"quiet_gate={args.fourier_quiet_gate} "
        f"fft_w={args.fft_weight} fft_phase={args.fft_lambda_phase} "
        f"fft_quiet={fft_quiet} fft_k_floor={args.fft_k_floor} "
        f"moment_phys_w={args.moment_phys_weight} vphi_phys_w={args.vphi_phys_weight} "
        f"r_focus_rd={bool(args.r_focus_rd)} band={_band} "
        f"a2_rd_w={args.a2_rd_weight} "
        f"evolve_gate_t=[{args.evolve_gate_t_lo},{args.evolve_gate_t_hi}]"
        f"×{args.evolve_gate_t_boost} "
        f"(modes={FOURIER_MODES} n_bins={args.fourier_n_bins}) "
        f"latent={args.latent_channels} base={args.base_channels} "
        f"disk_r_max={disk_r_max}",
        flush=True,
    )
    comp_w = {
        "disk": float(args.comp_weight_disk),
        "bulge": float(args.comp_weight_bulge),
        "halo": float(args.comp_weight_halo),
    }
    dmw_by = {
        "disk": float(args.dens_mass_weight),
        "halo": float(args.dens_mass_weight),
        "bulge": float(
            args.dens_mass_weight
            if args.bulge_dens_mass_weight is None
            else args.bulge_dens_mass_weight
        ),
    }
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    best_state: dict | None = None
    stale = 0
    history: list[dict] = []

    def _one_epoch(
        *,
        dens_w: float,
        moment_w: float,
        dens_grad_w: float,
        dens_resid_w: float,
        fourier_w: float,
        fft_w: float,
        moment_phys_w: float,
        vphi_phys_w: float,
        epoch: int,
        tag: str,
        track_best_by: str = "loss",
    ) -> float:
        nonlocal best, best_state, stale
        ds.set_epoch(epoch)
        model.train()
        losses, dens_losses, mom_losses, dens_grad_losses, dens_resid_losses = (
            [],
            [],
            [],
            [],
            [],
        )
        a2_losses: list[float] = []
        fourier_losses: list[float] = []
        fft_losses: list[float] = []
        a2_rd_losses: list[float] = []
        moment_phys_losses: list[float] = []
        vphi_phys_losses: list[float] = []
        for batch in loader:
            stacks = {k: v.to(DEVICE) for k, v in batch["stacks"].items()}
            pred = model(stacks)
            r_focus_by = None
            if bool(args.r_focus_rd) and args._rd_theta_idx is not None:
                th = batch.get("theta")
                if th is not None:
                    th_t = torch.as_tensor(th, device=DEVICE, dtype=stacks["disk"].dtype)
                    rd = th_t[:, int(args._rd_theta_idx)].clamp_min(1e-3)
                    # Normalised FOV: map coords span [-1,1] ≡ [-r_max, r_max].
                    r_focus_norm = (rd / float(args._disk_r_max)).clamp(1e-3, 0.95)
                    r_focus_by = {"disk": r_focus_norm, "bulge": None, "halo": None}
            metrics = multitower_reconstruction_loss(
                pred,
                stacks,
                dens_indices=dens_idx,
                dens_weight=dens_w,
                moment_weight=moment_w,
                dens_grad_weight=dens_grad_w,
                dens_resid_weight=dens_resid_w,
                a2_weight=fourier_w,
                fourier_modes=FOURIER_MODES,
                fourier_n_bins=args.fourier_n_bins,
                fourier_amp_weight=float(args.fourier_amp_weight),
                fourier_quiet_gate_floor=float(args.fourier_quiet_gate),
                fft_weight=fft_w,
                fft_lambda_phase=float(args.fft_lambda_phase),
                fft_quiet_gate_floor=fft_quiet,
                fft_k_floor=float(args.fft_k_floor),
                dens_scales=dens_scales,
                component_weights=comp_w,
                dens_mass_weight=float(args.dens_mass_weight),
                dens_mass_weight_by_component=dmw_by,
                moment_phys_weight=moment_phys_w,
                vphi_phys_weight=vphi_phys_w,
                n_mom_by_component={g.name: int(g.n_mom) for g in cfg.grids},
                r_focus_by_component=r_focus_by,
                r_focus_band=tuple(args._r_focus_band),
                r_focus_peak=float(args.r_focus_peak),
                r_focus_floor=float(args.r_focus_floor),
                a2_rd_weight=float(args.a2_rd_weight),
            )
            loss = metrics["loss"]
            # Soft evolve-gate: upweight mid-time snaps (face-on Σ / A₂ vs data).
            eg_boost = float(args.evolve_gate_t_boost)
            if eg_boost > 0.0:
                t_gyr = None
                if batch.get("t_gyr") is not None:
                    t_gyr = torch.as_tensor(
                        batch["t_gyr"], device=DEVICE, dtype=loss.dtype
                    )
                elif (
                    args._t_theta_idx is not None and batch.get("theta") is not None
                ):
                    th_t = torch.as_tensor(
                        batch["theta"], device=DEVICE, dtype=loss.dtype
                    )
                    t_gyr = th_t[:, int(args._t_theta_idx)]
                if t_gyr is not None:
                    lo = float(args.evolve_gate_t_lo)
                    hi = float(args.evolve_gate_t_hi)
                    # Soft box: sigmoid edges ~0.05 Gyr wide.
                    inside = torch.sigmoid((t_gyr - lo) / 0.05) * torch.sigmoid(
                        (hi - t_gyr) / 0.05
                    )
                    loss = loss * (1.0 + eg_boost * inside.mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            dens_losses.append(float(metrics["mse_dens"].detach()))
            mom_losses.append(float(metrics["mse_mom"].detach()))
            if "dens_grad_mse" in metrics:
                dens_grad_losses.append(float(metrics["dens_grad_mse"].detach()))
            if "dens_resid_mse" in metrics:
                dens_resid_losses.append(float(metrics["dens_resid_mse"].detach()))
            if "a2_mse" in metrics:
                a2_losses.append(float(metrics["a2_mse"].detach()))
            if "fourier_mse" in metrics:
                fourier_losses.append(float(metrics["fourier_mse"].detach()))
            if "fft_mse" in metrics:
                fft_losses.append(float(metrics["fft_mse"].detach()))
            if "a2_rd_mse" in metrics:
                a2_rd_losses.append(float(metrics["a2_rd_mse"].detach()))
            if "moment_phys_mse" in metrics:
                moment_phys_losses.append(float(metrics["moment_phys_mse"].detach()))
            if "vphi_phys_mse" in metrics:
                vphi_phys_losses.append(float(metrics["vphi_phys_mse"].detach()))
        mean_loss = float(np.mean(losses))
        mean_dens = float(np.mean(dens_losses))
        mean_mom = float(np.mean(mom_losses))
        mean_g = float(np.mean(dens_grad_losses)) if dens_grad_losses else float("nan")
        mean_r = float(np.mean(dens_resid_losses)) if dens_resid_losses else float("nan")
        mean_a2 = float(np.mean(a2_losses)) if a2_losses else float("nan")
        mean_f = float(np.mean(fourier_losses)) if fourier_losses else float("nan")
        mean_fft = float(np.mean(fft_losses)) if fft_losses else float("nan")
        mean_a2rd = float(np.mean(a2_rd_losses)) if a2_rd_losses else float("nan")
        mean_mp = float(np.mean(moment_phys_losses)) if moment_phys_losses else float("nan")
        mean_vp = float(np.mean(vphi_phys_losses)) if vphi_phys_losses else float("nan")
        history.append(
            {
                "phase": tag,
                "epoch": epoch,
                "loss": mean_loss,
                "mse_dens": mean_dens,
                "mse_mom": mean_mom,
                "dens_grad_mse": mean_g,
                "dens_resid_mse": mean_r,
                "a2_mse": mean_a2,
                "fourier_mse": mean_f,
                "fft_mse": mean_fft,
                "a2_rd_mse": mean_a2rd,
                "moment_phys_mse": mean_mp,
                "vphi_phys_mse": mean_vp,
            }
        )
        print(
            f"  [{tag}] epoch {epoch}  loss={mean_loss:.4f}  "
            f"mse_dens={mean_dens:.4f}  mse_mom={mean_mom:.4f}  "
            f"dens_grad={mean_g:.4f}  dens_resid={mean_r:.4f}  "
            f"a2_R_mse={mean_a2:.4f}  fourier_R={mean_f:.4f}  "
            f"fft_morph={mean_fft:.4f}  a2_Rd={mean_a2rd:.4f}  "
            f"mom_phys={mean_mp:.4f}  vphi_phys={mean_vp:.4f}",
            flush=True,
        )
        score = mean_dens if track_best_by == "mse_dens" else mean_loss
        if score + 1e-4 < best:
            best = score
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "cfg_grids": [g.__dict__ for g in cfg.grids],
                    "norm": {k: v.__dict__ for k, v in stats.items()},
                    "include_potential": args.include_potential,
                    "arch": args.arch,
                    "fourier": {
                        "modes": list(FOURIER_MODES),
                        "n_bins": args.fourier_n_bins,
                        "radial": True,
                    },
                    "args": vars(args) | {"out": str(out), "manifest": str(args.manifest)},
                },
                out / "multitower_slice_ae.pt",
            )
        else:
            stale += 1
        return mean_loss

    for epoch in range(1, args.epochs + 1):
        _one_epoch(
            dens_w=float(args.dens_weight),
            moment_w=float(args.moment_weight),
            dens_grad_w=float(args.dens_grad_weight),
            dens_resid_w=float(args.dens_resid_weight),
            fourier_w=float(args.a2_weight),
            fft_w=float(args.fft_weight),
            moment_phys_w=float(args.moment_phys_weight),
            vphi_phys_w=float(args.vphi_phys_weight),
            epoch=epoch,
            tag="main",
            track_best_by="loss",
        )
        if stale >= args.patience:
            print(f"early stop at main epoch {epoch}")
            break

    if args.dens_finetune_epochs > 0 and best_state is not None:
        model.load_state_dict(best_state)
        for g in opt.param_groups:
            g["lr"] = float(args.lr) * 0.4
        stale = 0
        best = float("inf")  # re-track by mse_dens during dens fine-tune
        for epoch in range(1, int(args.dens_finetune_epochs) + 1):
            _one_epoch(
                dens_w=float(args.dens_weight) * 2.0,
                moment_w=float(args.moment_weight) * 0.75,
                dens_grad_w=float(args.dens_grad_weight) * 1.5,
                dens_resid_w=float(args.dens_resid_weight) * 1.5,
                fourier_w=float(args.a2_weight) * 0.5,
                fft_w=float(args.fft_weight) * 0.75,
                moment_phys_w=float(args.moment_phys_weight) * 0.75,
                vphi_phys_w=float(args.vphi_phys_weight) * 0.75,
                epoch=1000 + epoch,
                tag="dens_ft",
                track_best_by="mse_dens",
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    elif (out / "multitower_slice_ae.pt").is_file():
        ckpt = torch.load(out / "multitower_slice_ae.pt", map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
    model.eval()

    results = []
    disk_grid = cfg.grid_for("disk")
    n_mom = disk_grid.n_mom
    for ex in EXAMPLES:
        path = args.work / ex["run"] / ex["dump"]
        pos, vel, mass, cid = _load_shared_frame(path)
        disk = cid == 0
        disk_pos, disk_mass = pos[disk], mass[disk]
        a2_data = _a2_median(disk_pos, disk_mass) if np.any(disk) else float("nan")
        data_am = _am_profiles(disk_pos, disk_mass)

        binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
        maps = {k: v[0] for k, v in binned.items()}
        if args.include_potential:
            from galacticsics.ml.fields.normalize import append_phi_channels

            phis = plummer_potential_multiscale(
                pos, mass, cid, cfg=cfg, n_sub=512, rng=rng
            )
            maps = {k: append_phi_channels(maps[k], phis[k]) for k in maps}

        normed = {k: normalize_stack(maps[k], stats[k]) for k in maps}
        with torch.no_grad():
            pred = model(
                {
                    k: torch.as_tensor(v[None], dtype=torch.float32, device=DEVICE)
                    for k, v in normed.items()
                }
            )
            pred_np = {k: v.detach().cpu().numpy()[0] for k, v in pred.items()}
        recon = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}

        mse_dens = float(
            np.mean(
                (pred_np["disk"][dens_idx["disk"]] - normed["disk"][dens_idx["disk"]]) ** 2
            )
        )
        data_dens = _disk_dens_collapse(maps["disk"], disk_grid.n_z, n_mom)
        recon_dens = _disk_dens_collapse(recon["disk"], disk_grid.n_z, n_mom)
        recon_am_map = {
            int(m): dens_map_azimuthal_fourier_numpy(
                recon_dens, m=int(m), n_bins=12, r_max=disk_grid.r_max
            )
            for m in FOURIER_MODES
        }

        # All-component resample on shared COM frame.
        # True masses for particle mass values; count mix for N (mass-weighted
        # N starves disk ~1% and nukes A_m / evolve panels).
        moment_maps = {
            g.name: recon[g.name][: g.n_moment_channels] for g in cfg.grids if g.name in recon
        }
        mass_true = {
            "disk": float(mass[cid == 0].sum()) if np.any(cid == 0) else 0.0,
            "halo": float(mass[cid == 1].sum()) if np.any(cid == 1) else 0.0,
            "bulge": float(mass[cid == 2].sum()) if np.any(cid == 2) else 0.0,
        }
        parts = resample_particles_from_multiscale(
            moment_maps,
            cfg=MultiScaleSliceConfig(grids=cfg.grids, include_potential=False),
            n_particles=args.n_resample,
            mass_total_per_component=mass_true,
            count_fractions=None if args.mass_weighted_counts else COUNT_FRACTIONS,
            rng=rng,
            sample_dispersion=True,
        )
        disk_rec = parts["component_id"] == 0
        a2_rec = (
            _a2_median(parts["pos"][disk_rec], parts["mass"][disk_rec])
            if np.any(disk_rec)
            else float("nan")
        )
        rec_am = _am_profiles(parts["pos"][disk_rec], parts["mass"][disk_rec])
        part_map = bin_plane_density(
            parts["pos"][disk_rec],
            parts["mass"][disk_rec],
            axes=(0, 1),
            n_bins=disk_grid.n_pix,
            half_extent=disk_grid.r_max,
        ).density

        png = out / f"{ex['name']}_field_panel.png"
        _plot_panel(
            png,
            title=f"{ex['name']}: {ex['label']}",
            data_dens=data_dens,
            recon_dens=recon_dens,
            particle_dens=part_map,
            a2_data=a2_data,
            a2_rec=a2_rec,
            mse=mse_dens,
        )

        face = bin_plane_density(
            disk_pos,
            disk_mass,
            axes=(0, 1),
            n_bins=max(64, disk_grid.n_pix),
            half_extent=disk_grid.r_max,
            z_filter=np.abs(disk_pos[:, 2]) < 0.5,
        )
        fig, ax = plt.subplots(figsize=(4, 4))
        from galacticsics.campaign.analysis import dens_array_log10

        show, vmin_s, vmax_s, _ = dens_array_log10(face.density, vmax_pct=99.0)
        im = ax.imshow(
            show.T,
            origin="lower",
            cmap="inferno",
            vmin=vmin_s,
            vmax=vmax_s,
        )
        ax.set_title(f"{ex['name']} true face-on")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_{10}\Sigma$")
        fig.tight_layout()
        fig.savefig(out / f"{ex['name']}_true_faceon.png", dpi=120)
        plt.close(fig)

        # Midplane dens + σ_z / vz diagnostic
        mid = disk_grid.n_z // 2
        base = mid * n_mom
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        dens_img = recon["disk"][base]
        vz_img = recon["disk"][base + 3] if n_mom > 3 else dens_img
        sz_img = recon["disk"][base + 6] if n_mom >= 7 else np.abs(vz_img)
        for ax, img, lab, kind in (
            (axes[0], dens_img, "dens", "dens"),
            (axes[1], sz_img, "σ_z" if n_mom >= 7 else "|vz|", "disp"),
            (axes[2], vz_img, "vz", "vel"),
        ):
            if kind == "dens":
                vmax = float(np.percentile(img[img > 0], 99)) if np.any(img > 0) else 1.0
                ax.imshow(
                    np.log1p(np.maximum(img, 0)),
                    origin="lower",
                    cmap="inferno",
                    vmax=np.log1p(vmax),
                )
            elif kind == "disp":
                vmax = float(np.percentile(img[img > 0], 99)) if np.any(img > 0) else 1.0
                ax.imshow(np.maximum(img, 0), origin="lower", cmap="magma", vmax=vmax)
            else:
                vmax = float(np.percentile(np.abs(img), 98)) if img.size else 1.0
                ax.imshow(img, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"recon midplane {lab}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{ex['name']} midplane moments")
        fig.tight_layout()
        fig.savefig(out / f"{ex['name']}_midplane_moments.png", dpi=120)
        plt.close(fig)

        evolve_info = None
        evo_am = None
        evo_dens = None
        if args.with_evolve and "bar" in ex["name"]:
            print(f"=== evolve {ex['name']} N={args.n_evolve} ===", flush=True)
            from galacticsics.ml.morton.tokenize import subsample_stratified

            n_ev = min(int(args.n_evolve), int(parts["pos"].shape[0]))
            ev_idx = subsample_stratified(
                parts["component_id"].astype(np.int64), n_ev, rng=rng
            )
            ev_parts = {
                "pos": parts["pos"][ev_idx],
                "vel": parts["vel"][ev_idx],
                "mass": np.full(n_ev, 1.0 / n_ev),
                "eps": np.full(n_ev, 0.1),
                "type_id": parts["component_id"][ev_idx].astype(np.int32),
            }
            try:
                evolve_info = _evolve_bh(
                    ev_parts,
                    end_gyr=args.evolve_gyr,
                    dt=args.dt,
                    force=args.force,
                    omp_threads=args.omp_threads,
                    timeout_s=float(args.evolve_timeout),
                )
                if not evolve_info.get("ok", False):
                    print(f"  evolve fallback: {evolve_info}", flush=True)
                else:
                    pos_f = evolve_info.pop("pos_final")
                    disk_e = ev_parts["type_id"] == 0
                    evo_am = _am_profiles(pos_f[disk_e], ev_parts["mass"][disk_e])
                    evo_dens = bin_plane_density(
                        pos_f[disk_e],
                        ev_parts["mass"][disk_e],
                        axes=(0, 1),
                        n_bins=disk_grid.n_pix,
                        half_extent=disk_grid.r_max,
                    ).density
                    evolve_info["a2_ic"] = float(rec_am["modes"][2]["median"])
                    evolve_info["a2_final"] = float(evo_am["modes"][2]["median"])
                    evolve_info["n_evolve"] = n_ev
                    print(
                        f"  evolve wall={evolve_info['wall_s']:.1f}s  "
                        f"ΔE/E={evolve_info['dE_over_E']:.3e}  "
                        f"COM drift={evolve_info['com_drift_kpc']:.3f} kpc  "
                        f"A₂ med {evolve_info['a2_ic']:.3f}→{evolve_info['a2_final']:.3f}",
                        flush=True,
                    )
                    _plot_evolve_panel(
                        out / f"{ex['name']}_evolve_panel.png",
                        title=f"{ex['name']}: data | recon dens | resampled | evolved",
                        data_dens=data_dens,
                        recon_dens=recon_dens,
                        resampled_dens=part_map,
                        evolved_dens=evo_dens,
                        labels=(
                            f"data\nA₂≈{a2_data:.2f}",
                            "AE dens",
                            f"resampled\nA₂≈{a2_rec:.2f}",
                            f"evolved {args.evolve_gyr:.2f} Gyr\nA₂≈{evolve_info['a2_final']:.2f}",
                        ),
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  evolve failed: {exc}", flush=True)
                evolve_info = {"error": str(exc), "ok": False}

        am_png = out / f"{ex['name']}_Am_R.png"
        _plot_am_profiles(
            am_png,
            title=f"{ex['name']}: $A_m(R)$ (not a single scalar)",
            data_prof=data_am,
            rec_prof=rec_am,
            evo_prof=evo_am,
        )

        # Serialisable profile tables for verdict.json
        def _ser_prof(prof: dict) -> dict:
            return {
                "r_mid": [float(x) for x in prof["r_mid"]],
                "modes": {
                    str(m): {
                        "a_m_over_a0": [
                            None if not np.isfinite(v) else float(v)
                            for v in prof["modes"][m]["a_m_over_a0"]
                        ],
                        "median": prof["modes"][m]["median"],
                    }
                    for m in FOURIER_MODES
                },
            }

        am_mse = {
            f"m{m}": _profile_mse(
                data_am["modes"][m]["a_m_over_a0"],
                rec_am["modes"][m]["a_m_over_a0"],
            )
            for m in FOURIER_MODES
        }
        am_mse_radial = {
            f"m{m}": _profile_mse_radial(
                data_am["r_mid"],
                data_am["modes"][m]["a_m_over_a0"],
                rec_am["modes"][m]["a_m_over_a0"],
                r_split=0.5 * float(disk_grid.r_max),
            )
            for m in FOURIER_MODES
        }

        row = {
            "name": ex["name"],
            "label": ex["label"],
            "a2_data": a2_data,
            "a2_resampled": a2_rec,
            "a_m_R_mse": am_mse,
            "a_m_R_mse_radial": am_mse_radial,
            "a_m_R_data": _ser_prof(data_am),
            "a_m_R_resampled": _ser_prof(rec_am),
            "a_m_R_recon_dens_map": {
                str(m): {
                    "r_mid": [float(x) for x in recon_am_map[m]["r_mid"]],
                    "a_m_over_a0": [
                        None if not np.isfinite(v) else float(v)
                        for v in recon_am_map[m]["a_m_over_a0"]
                    ],
                    "median": float(recon_am_map[m]["a_m_over_a0_median"]),
                }
                for m in FOURIER_MODES
            },
            "mse_dens_norm": mse_dens,
            "panel": str(png),
            "am_profile_panel": str(am_png),
            "n_resample": int(parts["pos"].shape[0]),
            "n_per_component": {
                k: int(v) for k, v in (parts.get("n_per_component") or {}).items()
            },
        }
        if evo_am is not None:
            row["a_m_R_evolved"] = _ser_prof(evo_am)
        if evolve_info is not None:
            row["evolve"] = {
                k: v for k, v in evolve_info.items() if k != "pos_final"
            }
        results.append(row)
        a2_recon_map = float(recon_am_map[2]["a_m_over_a0_median"])
        m2_rad = am_mse_radial["m2"]
        print(
            f"{ex['name']}: A2 med data={a2_data:.3f} recon_map={a2_recon_map:.3f} "
            f"resampled={a2_rec:.3f}  "
            f"A_m(R) MSE m1/m2/m3="
            f"{am_mse['m1']:.4f}/{am_mse['m2']:.4f}/{am_mse['m3']:.4f}  "
            f"m2 inner/outer={m2_rad['inner']:.4f}/{m2_rad['outer']:.4f}  "
            f"mse_dens={mse_dens:.4f} n_disk={int(parts['n_per_component'].get('disk', 0))} "
            f"→ {png}",
            flush=True,
        )

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([h["epoch"] for h in history], [h["loss"] for h in history], label="loss")
    ax.plot(
        [h["epoch"] for h in history], [h["mse_dens"] for h in history], label="mse_dens"
    )
    ax.plot(
        [h["epoch"] for h in history],
        [h.get("mse_mom", float("nan")) for h in history],
        label="mse_mom",
    )
    if any(np.isfinite(h.get("fourier_mse", float("nan"))) for h in history):
        ax.plot(
            [h["epoch"] for h in history],
            [h["fourier_mse"] for h in history],
            label="fourier A_m(R)",
        )
    if any(np.isfinite(h.get("fft_mse", float("nan"))) for h in history):
        ax.plot(
            [h["epoch"] for h in history],
            [h["fft_mse"] for h in history],
            label="fft morph",
        )
    ax.set_xlabel("epoch")
    ax.legend()
    ax.set_title(f"Multi-tower {args.arch} slice AE")
    fig.tight_layout()
    fig.savefig(out / "train_curve.png", dpi=120)
    plt.close(fig)

    bar = next(r for r in results if "bar" in r["name"])
    quiet = next(r for r in results if "quiet" in r["name"])
    _plot_comparison(out / "comparison_vs_hires64.png", bar=bar, quiet=quiet)

    a2_bar_map = float(bar["a_m_R_recon_dens_map"]["2"]["median"])
    a2_quiet_map = float(quiet["a_m_R_recon_dens_map"]["2"]["median"])
    # Prefer recon dens-map morphology (particle A₂ depends on count mix).
    morph_ok = (a2_bar_map > 0.22 and bar["a2_data"] > 0.2) and (a2_quiet_map < 0.08)
    bar_improved = (
        a2_bar_map > 0.20
        or (np.isfinite(bar["a2_resampled"]) and bar["a2_resampled"] > HIRES_64["bar_a2_resampled"] + 0.02)
    )
    am_shape_ok = (
        np.isfinite(bar["a_m_R_mse"]["m2"]) and bar["a_m_R_mse"]["m2"] < 0.05
    )
    fourier_on = float(args.a2_weight) > 0.0
    fft_on = float(args.fft_weight) > 0.0

    verdict = {
        "approach": (
            f"multi-scale slices ({args.preset} disk={args.disk_n_pix}) + {args.arch} "
            f"multi-tower AE + dens/σ ({args.moment_set}) dens_w={args.dens_weight} "
            f"moment_w={args.moment_weight} latent={args.latent_channels} "
            + (f"Fourier A_m(R) w={args.a2_weight} " if fourier_on else "Fourier OFF ")
            + (f"FFT-morph w={args.fft_weight} " if fft_on else "")
            + "count-stratified dens resample"
            + (" + short BH evolve" if args.with_evolve else "")
        ),
        "shared_centering": {
            "rule": "one mass-weighted global COM for all particles; never per-component",
            "rotation": "one common in-plane rotation after shared centering (train augment)",
            "potential": "Φ grids use the same shared origin / FOV frame",
            "helper": "galacticsics.ml.fields.frame.prepare_shared_frame",
        },
        "fourier": {
            "note": "A_m is not constant in radius — match A_m(R) profiles + phase, not only median",
            "modes": list(FOURIER_MODES),
            "n_bins": args.fourier_n_bins,
            "loss": "soft_am_radial_from_dens_maps (amp + cos/sin) on dens collapse",
            "artifacts": "per-example *_Am_R.png and a_m_R_* tables in examples",
        },
        "fft_morphology": {
            "weight": float(args.fft_weight),
            "lambda_phase": float(args.fft_lambda_phase),
            "quiet_gate": fft_quiet,
            "k_floor": float(args.fft_k_floor),
            "loss": "spatial_fft_morphology_loss on axisym-residual linearized dens",
            "note": "k-band weighted |FFT(R)| match; soft-gated by residual RMS",
        },
        "grids": scale_summary(cfg),
        "include_potential": args.include_potential,
        "arch": args.arch,
        "params": n_params,
        "best_loss": best,
        "dens_weight": args.dens_weight,
        "moment_weight": args.moment_weight,
        "a2_weight": args.a2_weight,
        "fft_weight": args.fft_weight,
        "latent_channels": args.latent_channels,
        "base_channels": args.base_channels,
        "count_fractions": None if args.mass_weighted_counts else COUNT_FRACTIONS,
        "examples": results,
        "a2_bar_recon_dens_map": a2_bar_map,
        "a2_quiet_recon_dens_map": a2_quiet_map,
        "morphology_preserved": morph_ok,
        "am_R_shape_ok": am_shape_ok,
        "bar_a2_improved_vs_hires64": bar_improved,
        "baseline_32": BASELINE_32,
        "hires_64": HIRES_64,
        "delta_vs_hires64": {
            "bar_a2_resampled": bar["a2_resampled"] - HIRES_64["bar_a2_resampled"],
            "quiet_a2_resampled": quiet["a2_resampled"] - HIRES_64["quiet_a2_resampled"],
            "best_loss": best - HIRES_64["best_loss"],
        },
        "voxel_shapes": {k: list(v.shape) for k, v in vox.items()},
        "channel_layout": {
            "moment_set": args.moment_set,
            "keys": list(disk_grid.moment_keys),
            "per_slab": (
                "dens, vx, vy, vz [, sx, sy, sz] [, beta]; "
                "slices stack as n_z · n_mom channels"
            ),
        },
        "comparison": {
            "slice_images": (
                "Preferred: per-component FOVs on a shared global COM, U-Net towers "
                "with dens/moment heads, radial A_m(R) on dens collapse, optional "
                "cross-tower attention."
            ),
            "voxels": (
                "Complementary anisotropic 3-D grids (+σ); train 3-D CNN after slice "
                "AE + resample look good."
            ),
            "particle_set_vae": (
                "Field-per-component encoding avoids global mix-CE collapse; "
                "see docs/ml_findings.md."
            ),
        },
        "recommendation": (
            "Crisp track: 128²+ dens+moment-heavy recon (Fourier optional) + larger latent "
            "sharpens barred phase space; inspect field panels + A_m(R). Quiet should stay "
            "quiet without Fourier forcing. Count-stratified resample for evolve panels."
            if morph_ok or bar_improved or am_shape_ok
            else (
                "Still washed — raise latent (192+) / disk 160², keep moment_weight≥dens_weight, "
                "Fourier off; check recon dens map A₂ vs data (not only particle median)."
            )
        ),
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    pointer = Path("runs/ml/field_maps")
    pointer.mkdir(parents=True, exist_ok=True)
    (pointer / "LATEST").write_text(str(out.resolve()) + "\n")
    print("\n=== VERDICT ===")
    print(verdict["recommendation"])
    print(
        f"bar A₂ recon_map={a2_bar_map:.3f} (data={bar['a2_data']:.3f}); "
        f"resampled {HIRES_64['bar_a2_resampled']:.3f} → {bar['a2_resampled']:.3f}; "
        f"quiet recon_map={a2_quiet_map:.3f}; A₂(R) MSE={bar['a_m_R_mse']['m2']:.4f}"
    )
    print(f"artifacts → {out.resolve()}")


if __name__ == "__main__":
    main()
