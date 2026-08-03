#!/usr/bin/env python3
"""Decompose bulge recon failure: grid smear → AE → resample vs disk/halo.

Writes JSON + markdown under ``--out`` (default under papers results).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galacticsics.ml.fields.binning import (  # noqa: E402
    MultiScaleSliceConfig,
    bin_multiscale_slice_stacks,
    dens_channel_indices_component,
    z_edges_for_grid,
)
from galacticsics.ml.fields.feature_library import load_frozen_teacher_bundle  # noqa: E402
from galacticsics.ml.fields.frame import prepare_shared_frame  # noqa: E402
from galacticsics.ml.fields.normalize import denormalize_stack, normalize_stack  # noqa: E402
from galacticsics.ml.fields.resample import resample_particles_from_multiscale  # noqa: E402
from galacticsics.ml.morton.polygon import COMPONENT_IDS  # noqa: E402
from ntropy.analysis.density import bin_spherical_density  # noqa: E402

COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
CID = {"disk": 0, "halo": 1, "bulge": 2}


def _load(path: Path):
    with np.load(path, allow_pickle=True) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        if "component_id" in data:
            cid = np.asarray(data["component_id"], dtype=np.int64)
        elif "type_id" in data:
            # GalactICS: 1=halo, 2=bulge, 3=disk → ML 0/1/2
            tid = np.asarray(data["type_id"]).astype(np.int64).ravel()
            cid = np.full(pos.shape[0], -1, dtype=np.int64)
            cid[tid == 3] = 0
            cid[tid == 1] = 1
            cid[tid == 2] = 2
        else:
            raise SystemExit(f"no component tags in {path}")
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    return pos, vel, mass, cid


def _comp_com_frame(pos, mass, cid, name: str):
    mask = cid == CID[name]
    p = pos[mask]
    m = mass[mask]
    com = np.average(p, axis=0, weights=m)
    return p - com, m


def _rho_particles(pos, mass, cid, name: str, *, r_max: float, r_min: float, n_bins: int = 28):
    p, m = _comp_com_frame(pos, mass, cid, name)
    prof = bin_spherical_density(p, m, n_bins=n_bins, r_max=r_max, log_bins=True, r_min=r_min)
    r_mid = np.asarray(prof.r_mid, dtype=np.float64)
    rho = np.asarray(prof.rho, dtype=np.float64)
    counts = np.asarray(prof.counts, dtype=np.int64)
    # Recover shell masses from ρ·V (empty shells → 0).
    edges = np.geomspace(r_min, r_max, n_bins + 1)
    vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    shell_m = np.where(np.isfinite(rho), rho * vol, 0.0)
    return {
        "r_mid": r_mid,
        "rho": rho,
        "counts": counts,
        "m_enc": np.cumsum(shell_m),
    }


def _voxel_centers(grid):
    n_pix = int(grid.n_pix)
    xy = np.linspace(-float(grid.r_max), float(grid.r_max), n_pix, endpoint=False)
    xy = xy + 0.5 * (2.0 * float(grid.r_max) / n_pix)
    z_edges = z_edges_for_grid(grid)
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    zz, yy, xx = np.meshgrid(zc, xy, xy, indexing="ij")
    return xx, yy, zz, z_edges


def _rho_from_dens_stack(stack, grid, *, r_max: float, r_min: float, n_bins: int = 28):
    """Spherical ρ(r) from voxel dens map, recentered on dens-weighted COM.

    Maps are deposited on the shared global COM; the bulge (and sometimes disk)
    peak is often offset by ~1 kpc.  Measuring shells about the map origin then
    falsely reports a hollow core.  Use the dens-weighted voxel COM instead so
    map ρ(r) is comparable to particle profiles about the component COM.
    """
    keys = grid.moment_keys
    n_mom = len(keys)
    dens_i = keys.index("dens")
    xx, yy, zz, z_edges = _voxel_centers(grid)
    dens = np.zeros_like(xx, dtype=np.float64)
    for iz in range(grid.n_z):
        dens[iz] = np.maximum(stack[iz * n_mom + dens_i], 0.0)
    dx = 2.0 * float(grid.r_max) / int(grid.n_pix)
    # dens is slab Σ (mass/area), not ρ — integrate dens·dA (no Δz).
    mass3 = dens * dx * dx
    mtot = float(mass3.sum())
    if mtot > 0:
        com = np.array(
            [
                float((mass3 * xx).sum() / mtot),
                float((mass3 * yy).sum() / mtot),
                float((mass3 * zz).sum() / mtot),
            ]
        )
    else:
        com = np.zeros(3)
    r = np.sqrt((xx - com[0]) ** 2 + (yy - com[1]) ** 2 + (zz - com[2]) ** 2).ravel()
    d = dens.ravel()
    mass = mass3.ravel()
    edges = np.geomspace(r_min, r_max, n_bins + 1)
    rho = np.full(n_bins, np.nan)
    m_shell = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        sel = (r >= edges[i]) & (r < edges[i + 1]) & (d > 0)
        counts[i] = int(sel.sum())
        m_shell[i] = float(mass[sel].sum())
        vol = (4.0 / 3.0) * np.pi * (edges[i + 1] ** 3 - edges[i] ** 3)
        if counts[i] > 0 and vol > 0:
            rho[i] = m_shell[i] / vol
    return {
        "r_mid": 0.5 * (edges[:-1] + edges[1:]),
        "rho": rho,
        "counts": counts,
        "m_enc": np.cumsum(m_shell),
        "m_total_map": mtot,
        "dens_com_kpc": com.tolist(),
        "peak_dens": float(dens.max()),
    }


def _channel_mse(pred_n, tgt_n, dens_idx):
    err = (pred_n - tgt_n) ** 2
    dens_mse = float(err[dens_idx].mean()) if dens_idx else float(err.mean())
    mom_idx = [i for i in range(err.shape[0]) if i not in set(dens_idx)]
    mom_mse = float(err[mom_idx].mean()) if mom_idx else float("nan")
    # cusp: central 3×3× mid-z dens channels
    n_mom = 7
    n_z = err.shape[0] // n_mom
    h, w = err.shape[-2:]
    mid = n_z // 2
    c = dens_idx[mid] if mid < len(dens_idx) else dens_idx[0]
    cy, cx = h // 2, w // 2
    core = err[c, cy - 1 : cy + 2, cx - 1 : cx + 2]
    core_mse = float(core.mean()) if core.size else float("nan")
    return dens_mse, mom_mse, core_mse


def _rho_at(prof, r_target: float) -> tuple[float, float]:
    r = np.asarray(prof["r_mid"])
    y = np.asarray(prof["rho"])
    if r.size == 0:
        return float("nan"), float("nan")
    i = int(np.argmin(np.abs(r - r_target)))
    return float(y[i]), float(r[i])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--teacher",
        type=Path,
        default=Path("runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt"),
    )
    p.add_argument(
        "--dump",
        type=Path,
        default=Path(
            "runs/mw_morton_corpus_v2/54a8faf836a0/evolution/particles/step_001800.npz"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("papers/mnras_noneq_ics/results/bulge_poor_diag"),
    )
    p.add_argument("--n-resample", type=int, default=1_750_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
    teacher.eval()
    pos, vel, mass, cid = _load(args.dump)

    grids = {g.name: g for g in cfg.grids}
    report = {
        "teacher": str(args.teacher),
        "dump": str(args.dump),
        "grids": {},
        "loss_context": {
            "component_weights_train": {"disk": 2.5, "bulge": 1.0, "halo": 0.4},
            "fourier_component_scale": {"disk": 1.0, "bulge": 0.15, "halo": 0.05},
            "fft_component_scale": {"disk": 1.0, "bulge": 0.1, "halo": 0.05},
            "note": (
                "Uniform voxel MSE in log1p(dens/median) underweights the cusp: "
                "outer voxels dominate the mean."
            ),
        },
        "components": {},
    }

    for g in cfg.grids:
        dx = 2.0 * g.r_max / g.n_pix
        dz = 2.0 * g.z_max / g.n_z
        report["grids"][g.name] = {
            "n_pix": g.n_pix,
            "n_z": g.n_z,
            "r_max": g.r_max,
            "z_max": g.z_max,
            "z_spacing": g.z_spacing,
            "dx_kpc": dx,
            "mean_dz_kpc": dz,
            "dx_over_rmax": dx / g.r_max,
            "voxels": int(g.n_pix * g.n_pix * g.n_z),
        }

    binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    maps = {k: v[0] for k, v in binned.items()}
    normed = {k: normalize_stack(maps[k], stats[k]) for k in maps}
    with torch.no_grad():
        pred = teacher(
            {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in normed.items()}
        )
        pred_n = {k: v.cpu().numpy()[0] for k, v in pred.items()}
    recon = {k: denormalize_stack(pred_n[k], stats[k]) for k in pred_n}

    # Resample from teacher maps and from true binned maps (ablate AE vs grid).
    parts_recon = resample_particles_from_multiscale(
        recon, cfg=cfg, n_particles=args.n_resample, count_fractions=COUNT, rng=rng
    )
    parts_binned = resample_particles_from_multiscale(
        maps, cfg=cfg, n_particles=args.n_resample, count_fractions=COUNT, rng=rng
    )

    fov = {"bulge": (6.0, 0.05), "halo": (50.0, 0.5), "disk": (14.0, 0.2)}
    # disk uses spherical for fair cusp-style compare near center only
    for name in ("bulge", "disk", "halo"):
        g = grids[name]
        r_max, r_min = fov[name]
        dens_idx = dens_channel_indices_component(g)
        dens_mse, mom_mse, core_mse = _channel_mse(pred_n[name], normed[name], dens_idx)
        ds = float(stats[name].dens_scale)

        # peak dens true vs recon (physical)
        keys = g.moment_keys
        n_mom = len(keys)
        di = keys.index("dens")
        peak_true = float(max(maps[name][iz * n_mom + di].max() for iz in range(g.n_z)))
        peak_recon = float(max(recon[name][iz * n_mom + di].max() for iz in range(g.n_z)))
        # dynamic range vs dens_scale
        peak_over_scale = peak_true / max(ds, 1e-30)

        part = _rho_particles(pos, mass, cid, name, r_max=r_max, r_min=r_min)
        map_b = _rho_from_dens_stack(maps[name], g, r_max=r_max, r_min=r_min)
        map_r = _rho_from_dens_stack(recon[name], g, r_max=r_max, r_min=r_min)
        # resampled ρ about component COM
        pr = parts_recon
        pb = parts_binned
        cid_r = np.asarray(pr["component_id"], dtype=np.int64)
        cid_b = np.asarray(pb["component_id"], dtype=np.int64)
        # fake full cid arrays
        rho_resamp_recon = _rho_particles(
            pr["pos"], pr["mass"], cid_r, name, r_max=r_max, r_min=r_min
        )
        rho_resamp_binned = _rho_particles(
            pb["pos"], pb["mass"], cid_b, name, r_max=r_max, r_min=r_min
        )

        # pick diagnostic radius near first bin
        r_diag = float(part["r_mid"][0])
        rho_p, _ = _rho_at(part, r_diag)
        rho_mb, _ = _rho_at(map_b, r_diag)
        rho_mr, _ = _rho_at(map_r, r_diag)
        rho_sr, _ = _rho_at(rho_resamp_recon, r_diag)
        rho_sb, _ = _rho_at(rho_resamp_binned, r_diag)

        m_true = float(mass[cid == CID[name]].sum())
        row = {
            "n_particles_true": int(np.sum(cid == CID[name])),
            "mass_true": m_true,
            "dens_scale": ds,
            "peak_dens_true": peak_true,
            "peak_dens_recon": peak_recon,
            "peak_over_dens_scale": peak_over_scale,
            "mse_dens_norm": dens_mse,
            "mse_mom_norm": mom_mse,
            "mse_dens_core3x3_norm": core_mse,
            "rho_r_diag_kpc": r_diag,
            "rho_particles": rho_p,
            "rho_binned_map": rho_mb,
            "rho_recon_map": rho_mr,
            "rho_resample_from_binned": rho_sb,
            "rho_resample_from_recon": rho_sr,
            "ratio_binned_over_particles": rho_mb / rho_p if rho_p > 0 else None,
            "ratio_recon_over_particles": rho_mr / rho_p if rho_p > 0 else None,
            "ratio_resamp_recon_over_particles": rho_sr / rho_p if rho_p > 0 else None,
            "ratio_resamp_binned_over_particles": rho_sb / rho_p if rho_p > 0 else None,
            "m_enc_particles": part["m_enc"].tolist(),
            "m_enc_binned_map": map_b["m_enc"].tolist(),
            "m_enc_recon_map": map_r["m_enc"].tolist(),
            "r_mid": part["r_mid"].tolist(),
            "rho_curve_particles": part["rho"].tolist(),
            "rho_curve_binned": map_b["rho"].tolist(),
            "rho_curve_recon": map_r["rho"].tolist(),
            "rho_curve_resamp_recon": rho_resamp_recon["rho"].tolist(),
            "rho_curve_resamp_binned": rho_resamp_binned["rho"].tolist(),
            "m_total_binned_map": map_b["m_total_map"],
            "m_total_recon_map": map_r["m_total_map"],
        }
        report["components"][name] = row
        print(
            f"{name}: dens_mse={dens_mse:.4f} core_mse={core_mse:.4f} "
            f"ρ({r_diag:.3f}) part={rho_p:.4g} bin={rho_mb:.4g} "
            f"recon={rho_mr:.4g} resampR={rho_sr:.4g} "
            f"peak_true/scale={peak_over_scale:.1f}",
            flush=True,
        )

    # grid vs bulge physical scale
    bg = grids["bulge"]
    report["bulge_scale_vs_grid"] = {
        "typical_bulge_a_kpc": 0.4,
        "corpus_r_p10_kpc": 0.11,
        "corpus_r_med_kpc": 0.46,
        "dx_kpc": report["grids"]["bulge"]["dx_kpc"],
        "mean_dz_kpc": report["grids"]["bulge"]["mean_dz_kpc"],
        "dx_over_a": report["grids"]["bulge"]["dx_kpc"] / 0.4,
        "dz_over_a": report["grids"]["bulge"]["mean_dz_kpc"] / 0.4,
        "n_pix_across_2a": 2 * 0.4 / report["grids"]["bulge"]["dx_kpc"],
        "n_z_across_2a": 2 * 0.4 / report["grids"]["bulge"]["mean_dz_kpc"],
        "verdict": (
            "Bulge Δz≈0.67 kpc (uniform z∈[-4,4]/12) is comparable to the entire "
            "bulge scale length — vertical cusp is severely under-resolved; "
            "Δx≈0.095 kpc ≈ r_p10 also smears the innermost decade."
        ),
    }

    # cheap ablations on maps: finer synthetic binning of particles only
    fine = MultiScaleSliceConfig.progressive_defaults(disk_n_pix=128)
    # override bulge to finer by constructing new config via progressive 192
    fine192 = MultiScaleSliceConfig.progressive_defaults(disk_n_pix=192)
    for tag, c in (("teacher_grid", cfg), ("fine_disk192", fine192)):
        bb = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=c)
        mm = {k: v[0] for k, v in bb.items()}
        g = c.grid_for("bulge")
        pr = _rho_from_dens_stack(mm["bulge"], g, r_max=6.0, r_min=0.05)
        rp, rr = _rho_at(pr, report["components"]["bulge"]["rho_r_diag_kpc"])
        report.setdefault("grid_ablation", {})[tag] = {
            "n_pix": g.n_pix,
            "n_z": g.n_z,
            "dx": 2 * g.r_max / g.n_pix,
            "dz": 2 * g.z_max / g.n_z,
            "rho_inner": rp,
            "ratio_to_particles": rp
            / report["components"]["bulge"]["rho_particles"]
            if report["components"]["bulge"]["rho_particles"] > 0
            else None,
        }
        print(
            f"grid_ablation {tag}: n_pix={g.n_pix} n_z={g.n_z} "
            f"dx={2*g.r_max/g.n_pix:.4f} dz={2*g.z_max/g.n_z:.4f} "
            f"ρ_inner={rp:.4g}",
            flush=True,
        )

    # plot ρ(r) stages
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=False)
    for ax, name in zip(axes, ("bulge", "disk", "halo")):
        c = report["components"][name]
        r = np.asarray(c["r_mid"])
        ax.loglog(r, c["rho_curve_particles"], lw=2.0, label="particles")
        ax.loglog(r, c["rho_curve_binned"], lw=1.5, label="binned map")
        ax.loglog(r, c["rho_curve_recon"], lw=1.5, label="AE recon map")
        ax.loglog(r, c["rho_curve_resamp_recon"], lw=1.4, ls="--", label="resample AE")
        ax.set_title(name)
        ax.set_xlabel(r"$r$ [kpc]")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$\rho(r)$")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Teacher pipeline density stages (shared COM maps / component COM particles)")
    fig.tight_layout()
    png = args.out / "bulge_rho_pipeline.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    report["figure"] = str(png)

    # root-cause ranking
    b = report["components"]["bulge"]
    d = report["components"]["disk"]
    h = report["components"]["halo"]
    report["root_cause_ranking"] = [
        {
            "rank": 1,
            "factor": "bulge vertical grid too coarse (Δz ≳ a)",
            "evidence": report["bulge_scale_vs_grid"],
        },
        {
            "rank": 2,
            "factor": "in-plane Δx ≈ r_p10 smears cusp before AE",
            "evidence": {
                "ratio_binned_over_particles": b["ratio_binned_over_particles"],
                "dx_kpc": report["grids"]["bulge"]["dx_kpc"],
            },
        },
        {
            "rank": 3,
            "factor": "uniform log1p dens MSE + disk-heavy component weights",
            "evidence": {
                "bulge_mse_dens": b["mse_dens_norm"],
                "disk_mse_dens": d["mse_dens_norm"],
                "halo_mse_dens": h["mse_dens_norm"],
                "bulge_core_mse": b["mse_dens_core3x3_norm"],
                "peak_over_scale_bulge": b["peak_over_dens_scale"],
                "component_weights": report["loss_context"]["component_weights_train"],
            },
        },
        {
            "rank": 4,
            "factor": "AE + resample further softens already-smeared cusp",
            "evidence": {
                "ratio_recon": b["ratio_recon_over_particles"],
                "ratio_resamp_recon": b["ratio_resamp_recon_over_particles"],
                "ratio_resamp_binned": b["ratio_resamp_binned_over_particles"],
            },
        },
    ]

    (args.out / "verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.out / "verdict.json")
    print("figure", png)


if __name__ == "__main__":
    main()
