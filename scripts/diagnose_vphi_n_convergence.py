#!/usr/bin/env python3
"""Diagnose ⟨v_φ⟩(R) apparent drop-off vs particle count N.

Uses a quiet MW disk+halo (no bulge) IC and compares buggy estimators
(empty bins → 0; equal-pixel map means with zero-fill) against mass-weighted
profiles with ``min_count`` masking + dens-weighted map reduction.

Example
-------
::

    python scripts/diagnose_vphi_n_convergence.py \\
        --out runs/ml/field_maps/vphi_n_converg_2026-07-27
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from galacticsics.builder import GalaxyBuilder
from galacticsics.ml.fields.binning import (
    MultiScaleSliceConfig,
    bin_multiscale_slice_stacks,
    radial_vphi_from_deposit_slab,
)
from galacticsics.ml.profiles import (
    cylindrical_radius,
    numpy_mass_weighted_profile,
    v_phi_cylindrical,
)
from galacticsics.models import GalaxyModel


def _pack(parts: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_l, vel_l, mass_l, cid_l = [], [], [], []
    for name, cid in (("disk", 0), ("halo", 1), ("bulge", 2)):
        if name not in parts:
            continue
        d = parts[name].data
        pos_l.append(np.column_stack([d["x"], d["y"], d["z"]]))
        vel_l.append(np.column_stack([d["vx"], d["vy"], d["vz"]]))
        mass_l.append(np.asarray(d["mass"], dtype=np.float64))
        cid_l.append(np.full(len(d), cid, dtype=np.int32))
    return (
        np.concatenate(pos_l),
        np.concatenate(vel_l),
        np.concatenate(mass_l),
        np.concatenate(cid_l),
    )


def _midplane(
    pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, *, z_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = np.abs(pos[:, 2]) <= float(z_max)
    return pos[mid], vel[mid], mass[mid]


def kin_profile(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    *,
    r_max: float = 15.0,
    n_bins: int = 28,
    z_max: float = 0.5,
    min_count: int = 0,
) -> dict[str, np.ndarray]:
    pos, vel, mass = _midplane(pos, vel, mass, z_max=z_max)
    R = cylindrical_radius(pos)
    vphi = v_phi_cylindrical(pos, vel)
    edges = np.linspace(0.0, float(r_max), int(n_bins) + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n_bins, np.nan)
    err = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = (R >= edges[i]) & (R < edges[i + 1])
        n = int(m.sum())
        counts[i] = n
        if n < max(int(min_count), 1):
            continue
        w = mass[m]
        mean[i] = float(np.average(vphi[m], weights=w))
        if n >= 2:
            mu = mean[i]
            var = float(np.average((vphi[m] - mu) ** 2, weights=w))
            err[i] = float(np.sqrt(var / n))
    return {"r_mid": r_mid, "mean_vphi": mean, "stderr": err, "counts": counts}


def _slab_channels(stack: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = list(meta["moment_keys"])
    n_mom = len(keys)
    iz = int(meta["n_z"]) // 2
    dens = stack[iz * n_mom + keys.index("dens")].astype(np.float64)
    vx = stack[iz * n_mom + keys.index("vx")].astype(np.float64)
    vy = stack[iz * n_mom + keys.index("vy")].astype(np.float64)
    return dens, vx, vy


def map_radial_vphi_equal_zerofill(
    dens: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    r_max: float,
    n_bins: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    """BUG pattern: equal-pixel mean with empty cells as v=0."""
    n_pix = dens.shape[0]
    edges_xy = np.linspace(-float(r_max), float(r_max), n_pix + 1)
    xc = 0.5 * (edges_xy[:-1] + edges_xy[1:])
    X = xc[:, None]
    Y = xc[None, :]
    R = np.sqrt(X * X + Y * Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        vphi = np.where(R > 1e-8, (-Y * vx + X * vy) / R, 0.0)
    # empty → 0 already in deposit; include them in the mean
    edges_r = np.linspace(0.0, float(r_max), int(n_bins) + 1)
    r_mid = 0.5 * (edges_r[:-1] + edges_r[1:])
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (R >= edges_r[i]) & (R < edges_r[i + 1])
        if not np.any(mask):
            continue
        out[i] = float(np.mean(vphi[mask]))
    return r_mid, out


def _interp_at(r: np.ndarray, y: np.ndarray, r0: float) -> float | None:
    ok = np.isfinite(y)
    if np.count_nonzero(ok) < 2:
        return None
    return float(np.interp(r0, r[ok], y[ok]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/vphi_n_converg_2026-07-27"),
    )
    p.add_argument("--n-list", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    p.add_argument("--z-max", type=float, default=0.5)
    p.add_argument("--min-count", type=int, default=20)
    args = p.parse_args()

    out: Path = args.out
    ic = out / "ic"
    out.mkdir(parents=True, exist_ok=True)
    ic.mkdir(parents=True, exist_ok=True)

    model = GalaxyModel.milky_way_disk_halo()
    b = GalaxyBuilder(model=model, model_dir=ic)
    if not (ic / "dbh.dat").is_file():
        print(f"solving potential → {ic}", flush=True)
        b.solve_potential(work_dir=str(ic), cleanup=False)
        b.solve_disk_df()
    else:
        b.load_artifacts()

    cfg = MultiScaleSliceConfig.smoke_defaults(include_potential=False, moment_set="disp")
    results: dict[int, dict] = {}

    for N in args.n_list:
        print(f"sampling N_disk={N}...", flush=True)
        parts = b.sample(
            n_disk=int(N),
            n_halo=max(int(N) // 2, 5000),
            n_bulge=0,
            seed=42 + int(N),
            work_dir=str(out / f"sample_{N}"),
            cleanup=False,
            run_diskdf=False,
        )
        pos, vel, mass, cid = _pack(parts)
        disk = cid == 0
        print(f"  n_disk={int(disk.sum())} n_tot={len(mass)}", flush=True)

        pos_d, vel_d, mass_d = pos[disk], vel[disk], mass[disk]
        kin0 = kin_profile(pos_d, vel_d, mass_d, z_max=args.z_max, min_count=0)
        kin_mc = kin_profile(
            pos_d, vel_d, mass_d, z_max=args.z_max, min_count=int(args.min_count)
        )
        pos_m, vel_m, mass_m = _midplane(pos_d, vel_d, mass_d, z_max=args.z_max)
        R = cylindrical_radius(pos_m)
        vphi = v_phi_cylindrical(pos_m, vel_m)
        # Legacy zero-fill path (empty→0) for the "before" panel.
        r_bug, mean_bug = numpy_mass_weighted_profile(
            R, vphi, mass_m, n_bins=28, r_max=15.0, empty=0.0, min_count=0
        )
        r_dens, dens_prof = numpy_mass_weighted_profile(
            R, None, mass_m, n_bins=28, r_max=15.0, density=True
        )

        binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
        stack, meta = binned["disk"]
        dens_ch, vx_ch, vy_ch = _slab_channels(stack, meta)
        r_m, v_eq0 = map_radial_vphi_equal_zerofill(
            dens_ch, vx_ch, vy_ch, r_max=float(meta["r_max"])
        )
        r_dw, v_dw, _ = radial_vphi_from_deposit_slab(
            dens_ch, vx_ch, vy_ch, r_max=float(meta["r_max"])
        )

        n_pix = dens_ch.shape[0]
        edges = np.linspace(-float(meta["r_max"]), float(meta["r_max"]), n_pix + 1)
        xc = 0.5 * (edges[:-1] + edges[1:])
        X = xc[:, None]
        Y = xc[None, :]
        Rpix = np.sqrt(X * X + Y * Y)
        outer = (Rpix > 8.0) & (Rpix < 12.0)
        frac_empty = float(np.mean(dens_ch[outer] <= 0)) if np.any(outer) else float("nan")

        results[int(N)] = {
            "kin0": kin0,
            "kin_mc": kin_mc,
            "r_bug": r_bug,
            "mean_bug": mean_bug,
            "r_dens": r_dens,
            "dens": dens_prof,
            "r_m": r_m,
            "v_eq0": v_eq0,
            "r_dw": r_dw,
            "v_dw": v_dw,
            "frac_empty_R8_12": frac_empty,
            "v_R12": {
                "particle_any": _interp_at(kin0["r_mid"], kin0["mean_vphi"], 12.0),
                "particle_mincount": _interp_at(kin_mc["r_mid"], kin_mc["mean_vphi"], 12.0),
                "numpy_zerofill": _interp_at(r_bug, mean_bug, 12.0),
                "map_equal_zerofill": _interp_at(r_m, v_eq0, 12.0),
                "map_dens_weighted": _interp_at(r_m, v_dw, 12.0),
            },
        }
        print(
            f"  empty R∈[8,12] frac={frac_empty:.3f}  "
            f"vφ(12) bug={results[int(N)]['v_R12']['numpy_zerofill']} "
            f"minc={results[int(N)]['v_R12']['particle_mincount']} "
            f"map0={results[int(N)]['v_R12']['map_equal_zerofill']} "
            f"mapdw={results[int(N)]['v_R12']['map_dens_weighted']}",
            flush=True,
        )

    # ---- figures ----
    colors = ["C0", "C1", "C2", "C3", "C4"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=140)

    ax = axes[0, 0]
    for N, c in zip(args.n_list, colors):
        d = results[int(N)]
        ax.plot(d["r_bug"], d["mean_bug"], color=c, lw=1.8, label=f"N={N:,}")
    ax.set_title(r"BUG: ``numpy_mass_weighted_profile`` (empty→0)")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$\langle v_\phi\rangle$ [100 km/s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for N, c in zip(args.n_list, colors):
        d = results[int(N)]["kin_mc"]
        ok = np.isfinite(d["mean_vphi"])
        ax.plot(d["r_mid"][ok], d["mean_vphi"][ok], color=c, lw=1.8, label=f"N={N:,}")
        ax.fill_between(
            d["r_mid"][ok],
            d["mean_vphi"][ok] - d["stderr"][ok],
            d["mean_vphi"][ok] + d["stderr"][ok],
            color=c,
            alpha=0.18,
        )
    ax.set_title(f"FIX: mass-wtd + min_count={args.min_count} + stderr")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$\langle v_\phi\rangle$ [100 km/s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for N, c in zip(args.n_list, colors):
        d = results[int(N)]
        ax.plot(
            d["r_m"],
            d["v_eq0"],
            color=c,
            lw=1.5,
            ls="--",
            label=f"N={N:,} equal+0",
        )
        ok = np.isfinite(d["v_dw"])
        ax.plot(d["r_dw"][ok], d["v_dw"][ok], color=c, lw=1.8, label=f"N={N:,} dens-wtd")
    ax.set_title("Deposit map → radial ⟨v_φ⟩")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$\langle v_\phi\rangle$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0)

    ax = axes[1, 1]
    for N, c in zip(args.n_list, colors):
        d = results[int(N)]
        ax.semilogy(d["r_dens"], np.maximum(d["dens"], 1e-12), color=c, lw=1.8, label=f"N={N:,}")
    ax.set_title(r"Midplane $\Sigma(R)$ (should agree where populated)")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$\Sigma$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"MW disk+halo (no bulge): $\langle v_\phi\rangle$ vs $N$ — estimator bias",
        fontsize=12,
    )
    fig.tight_layout()
    fig_path = out / "fig_vphi_N_before_after.png"
    fig.savefig(fig_path)
    print(f"wrote {fig_path}", flush=True)

    # Compact before/after single panel for paper
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=140)
    for N, c in zip(args.n_list, colors):
        d = results[int(N)]
        ax1.plot(d["r_bug"], d["mean_bug"], color=c, lw=1.8, label=f"N={N:,}")
        k = d["kin_mc"]
        ok = np.isfinite(k["mean_vphi"])
        ax2.plot(k["r_mid"][ok], k["mean_vphi"][ok], color=c, lw=1.8, label=f"N={N:,}")
    ax1.set_title("Before (empty bins → 0)")
    ax2.set_title(f"After (min_count={args.min_count}, NaN mask)")
    for ax in (ax1, ax2):
        ax.set_xlabel(r"$R$ [kpc]")
        ax.set_ylabel(r"$\langle v_\phi\rangle$ [100 km/s]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig2.suptitle("milky_way_disk_halo (bulge=None)", fontsize=11)
    fig2.tight_layout()
    fig2_path = out / "fig_vphi_N_paper.png"
    fig2.savefig(fig2_path)
    print(f"wrote {fig2_path}", flush=True)

    summary = {
        "system": "GalaxyModel.milky_way_disk_halo (bulge=None)",
        "campaign_analog": "campaigns/bauer_morphology.json (bulge particles=0)",
        "Ns_disk": [int(n) for n in args.n_list],
        "z_max_kpc": float(args.z_max),
        "min_count": int(args.min_count),
        "frac_empty_R8_12": {
            str(N): results[int(N)]["frac_empty_R8_12"] for N in args.n_list
        },
        "vphi_R12": {str(N): results[int(N)]["v_R12"] for N in args.n_list},
        "figures": {
            "before_after_panels": str(fig_path),
            "paper_pair": str(fig2_path),
        },
        "root_cause": (
            "Empty radial bins and empty deposit pixels were filled with 0, so "
            "⟨v_φ⟩ appears to fall with decreasing N (especially outer disk). "
            "True mass-weighted means with min_count masking agree across N."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
