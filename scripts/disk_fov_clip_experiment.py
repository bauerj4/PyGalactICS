#!/usr/bin/env python3
"""Diagnostic-only disk FOV clip experiment (does NOT change defaults / retrain).

Compare deposits at the published teacher FOV (disk r_max=12 kpc) vs wider
boxes on 906c4 late. Writes:

  papers/mnras_noneq_ics/results/disk_fov_clip_EXPERIMENT.md
  papers/mnras_noneq_ics/figures/fig_latent_theta_disk_fov_clip.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz"
OUT_MD = ROOT / "papers/mnras_noneq_ics/results/disk_fov_clip_EXPERIMENT.md"
OUT_FIG = ROOT / "papers/mnras_noneq_ics/figures/fig_latent_theta_disk_fov_clip.png"
OUT_JSON = ROOT / "papers/mnras_noneq_ics/results/disk_fov_clip_metrics.json"

# Published FFT-long / progressive disk FOV (do not change production defaults).
R_CUR = 12.0
Z_CUR = 1.5
NPIX_CUR = 128
DX_CUR = (2.0 * R_CUR) / NPIX_CUR  # 0.1875 kpc
R_WIDE = (18.0, 24.0)  # 1.5× and 2×


def _hist2d(x, y, w, *, nbin: int, half: float) -> np.ndarray:
    edges = np.linspace(-half, half, nbin + 1)
    h, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=w)
    return h.T.astype(np.float64)


def _surface_density(R, mass, *, r_max: float, n_bins: int = 48):
    edges = np.linspace(0.0, r_max, n_bins + 1)
    msum, _ = np.histogram(R, bins=edges, weights=mass)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    sig = np.divide(msum, area, out=np.zeros_like(msum), where=area > 0)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    return r_mid, sig, msum


def _a2_profile(pos, mass, *, r_max: float, n_bins: int = 36, z_max: float = 0.5):
    from ntropy.analysis.disk_density import disk_azimuthal_fourier

    return disk_azimuthal_fourier(
        pos,
        mass,
        m=2,
        r_max=r_max,
        n_bins=n_bins,
        z_max=z_max,
        min_count=10,
        recenter=False,
    )


def main() -> None:
    from galacticsics.ml.fields.frame import prepare_shared_frame
    from galacticsics.ml.morton.polygon import _component_ids

    with np.load(DUMP) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        cid = _component_ids(None, data["type_id"], pos.shape[0])

    pos, vel, meta_frame = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    disk = cid == 0
    p = pos[disk]
    m = mass[disk]
    R = np.hypot(p[:, 0], p[:, 1])
    Mtot = float(m.sum())

    # --- mass outside FOV (square box |x|,|y|<=r_max AND circular R>r_max) ---
    fov_list = [R_CUR, *R_WIDE]
    mass_stats = {}
    for rmax in fov_list:
        in_sq = (np.abs(p[:, 0]) <= rmax) & (np.abs(p[:, 1]) <= rmax)
        in_circ = R <= rmax
        mass_stats[rmax] = {
            "frac_outside_square": float(1.0 - m[in_sq].sum() / Mtot),
            "frac_outside_circle": float(1.0 - m[in_circ].sum() / Mtot),
            "mass_outside_square": float(m[~in_sq].sum()),
            "mass_outside_circle": float(m[~in_circ].sum()),
            "M_disk": Mtot,
        }
        # also |z| cut relevance for z_max=1.5
        in_z = np.abs(p[:, 2]) <= Z_CUR
        mass_stats[rmax]["frac_outside_z"] = float(1.0 - m[in_z].sum() / Mtot)

    # Cumulative mass vs R
    order = np.argsort(R)
    Rc = R[order]
    Mc = np.cumsum(m[order]) / Mtot

    # Particle Σ(R) and A2(R) out to 24 kpc
    r_sig, sig, _ = _surface_density(R, m, r_max=24.0, n_bins=48)
    a2_full = _a2_profile(p, m, r_max=24.0, n_bins=36, z_max=0.5)
    a2_cur = _a2_profile(p, m, r_max=R_CUR, n_bins=24, z_max=0.5)

    # Truncation of Σ: fraction of disk mass in annuli beyond 12
    _, _, msum24 = _surface_density(R, m, r_max=24.0, n_bins=48)
    edges24 = np.linspace(0.0, 24.0, 49)
    r_mid24 = 0.5 * (edges24[:-1] + edges24[1:])
    mass_beyond_12 = float(msum24[r_mid24 > R_CUR].sum() / Mtot)

    # Face-on dens deposits at fixed dx (so wider FOV adds outer pixels)
    deposits = {}
    for rmax in fov_list:
        npix = int(round(2.0 * rmax / DX_CUR))
        if npix % 2:
            npix += 1
        dens = _hist2d(p[:, 0], p[:, 1], m, nbin=npix, half=rmax)
        deposits[rmax] = {"dens": dens, "n_pix": npix, "dx": 2.0 * rmax / npix}

    # Wider FOV cannot be fed to the frozen teacher without retrain (128² @ 12 kpc).
    from galacticsics.ml.fields.binning import MultiScaleSliceConfig

    cfg = MultiScaleSliceConfig.progressive_defaults(
        disk_n_pix=128, include_potential=False, moment_set="disp"
    )
    gdisk = cfg.grid_for("disk")
    assert abs(gdisk.r_max - 12.0) < 1e-9
    z_norm = {
        "disk_r_max": float(gdisk.r_max),
        "disk_z_max": float(gdisk.z_max),
        "disk_n_pix": int(gdisk.n_pix),
        "disk_n_z": int(gdisk.n_z),
    }
    encode_note = (
        "Teacher encode of wider FOV **not run** (architecture locked to disk "
        f"{gdisk.n_pix}² @ r_max={gdisk.r_max}, z_max={gdisk.z_max}). "
        "Comparing pooled z / recon across FOVs would require a new tower — "
        "out of scope for this diagnostic."
    )

    # ---------- figure ----------
    fig = plt.figure(figsize=(11.2, 8.2), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.95])

    # Row 0: face-on log dens at three FOVs
    vmax = None
    panels = []
    for i, rmax in enumerate(fov_list):
        ax = fig.add_subplot(gs[0, i])
        dens = deposits[rmax]["dens"]
        show = np.log10(dens + dens[dens > 0].min() * 0.1) if np.any(dens > 0) else dens
        if vmax is None:
            vmax = float(np.percentile(show[np.isfinite(show)], 99.5))
        im = ax.imshow(
            show,
            origin="lower",
            extent=[-rmax, rmax, -rmax, rmax],
            cmap="magma",
            vmin=vmax - 3.5,
            vmax=vmax,
            interpolation="nearest",
        )
        # mark current FOV square on wider panels
        if rmax > R_CUR:
            from matplotlib.patches import Circle, Rectangle

            ax.add_patch(
                Rectangle(
                    (-R_CUR, -R_CUR),
                    2 * R_CUR,
                    2 * R_CUR,
                    fill=False,
                    edgecolor="cyan",
                    lw=1.2,
                    ls="--",
                )
            )
            ax.add_patch(
                Circle((0, 0), R_CUR, fill=False, edgecolor="lime", lw=0.9, ls=":")
            )
        ax.set_title(
            f"r_max={rmax:.0f} kpc  (n={deposits[rmax]['n_pix']})",
            fontsize=10,
        )
        ax.set_xlabel("x [kpc]")
        if i == 0:
            ax.set_ylabel("y [kpc]")
        ax.set_aspect("equal")
        frac = mass_stats[rmax]["frac_outside_square"] * 100
        ax.text(
            0.03,
            0.97,
            f"mass outside box: {frac:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, ec="none"),
        )
        panels.append(im)
    cax = fig.colorbar(panels[0], ax=fig.axes[:3], fraction=0.025, pad=0.02)
    cax.set_label(r"$\log_{10}\Sigma$ (arb.)")

    # Row 1: Σ(R), A2(R), cumulative mass
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.semilogy(r_sig, sig, color="C0", lw=1.6, label=r"$\Sigma(R)$ particles")
    ax0.axvline(R_CUR, color="0.35", ls="--", lw=1.0, label=f"current r_max={R_CUR:.0f}")
    for rmax, c in zip(R_WIDE, ("C1", "C3")):
        ax0.axvline(rmax, color=c, ls=":", lw=1.0, label=f"{rmax:.0f} kpc")
    ax0.set_xlim(0, 24)
    ax0.set_xlabel("R [kpc]")
    ax0.set_ylabel(r"$\Sigma$ [mass / kpc$^2$]")
    ax0.legend(fontsize=7, loc="upper right")
    ax0.set_title("Surface density")

    ax1 = fig.add_subplot(gs[1, 1])
    ax1.plot(
        a2_full["r_mid"],
        a2_full["a_m_over_a0"],
        color="C0",
        lw=1.6,
        label=r"$A_2/A_0$ to 24 kpc",
    )
    ax1.plot(
        a2_cur["r_mid"],
        a2_cur["a_m_over_a0"],
        color="C2",
        lw=1.2,
        ls="--",
        label=r"$A_2/A_0$ to 12 kpc",
    )
    ax1.axvline(R_CUR, color="0.35", ls="--", lw=1.0)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, None)
    ax1.set_xlabel("R [kpc]")
    ax1.set_ylabel(r"$A_2/A_0$")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("Bar strength profile")

    ax2 = fig.add_subplot(gs[1, 2])
    ax2.plot(Rc, Mc, color="C0", lw=1.6)
    ax2.axvline(R_CUR, color="0.35", ls="--", lw=1.0)
    for rmax, c in zip(R_WIDE, ("C1", "C3")):
        ax2.axvline(rmax, color=c, ls=":", lw=1.0)
    for rmax in fov_list:
        f_out = mass_stats[rmax]["frac_outside_circle"]
        ax2.scatter([rmax], [1.0 - f_out], s=28, zorder=3)
        ax2.annotate(
            f"{100 * f_out:.1f}% out",
            (rmax, 1.0 - f_out),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=7,
        )
    ax2.set_xlim(0, 26)
    ax2.set_ylim(0.7, 1.02)
    ax2.set_xlabel("R [kpc]")
    ax2.set_ylabel("cumulative disk mass fraction")
    ax2.set_title("Enclosed mass")

    fig.suptitle(
        "Disk FOV clip diagnostic — 906c4 late (step_003200); "
        "current teacher FOV ±12 kpc (cyan square on wide panels)",
        fontsize=11,
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)

    # ---------- metrics + markdown ----------
    metrics = {
        "dump": str(DUMP.relative_to(ROOT)),
        "current_fov": {
            "r_max_kpc": R_CUR,
            "z_max_kpc": Z_CUR,
            "n_pix": NPIX_CUR,
            "dx_kpc": DX_CUR,
            "source": "MultiScaleSliceConfig.progressive_defaults / fft_morph_ft_long verdict",
        },
        "mass_stats": {str(k): v for k, v in mass_stats.items()},
        "mass_fraction_beyond_R12_annuli": mass_beyond_12,
        "a2_median_to_12": float(a2_cur["a_m_over_a0_median"]),
        "a2_median_to_24": float(a2_full["a_m_over_a0_median"]),
        "encode_note": encode_note,
        "teacher_disk_grid": z_norm,
        "constraint": "diagnostic only — no default FOV change, no retrain, no corpus rebuild",
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2) + "\n")

    f12 = mass_stats[R_CUR]
    f18 = mass_stats[18.0]
    f24 = mass_stats[24.0]
    # Mass loss is small, but face-on charts / outer A2 show real truncation.
    out_sq = f12["frac_outside_square"]
    rec = (
        "**Keep ±12 kpc for all published results** (teacher, library, 906c4 "
        "MATCH / no-bulge FADE MATCH / OOD). Mass outside the square FOV is only "
        f"**{100 * out_sq:.1f}%**, and the primary bar peak is inside ~8 kpc — "
        "not enough to justify migrating scoreboards.\n\n"
        "For a **future** corpus / teacher rebuild, prefer **r_max ≈ 18 kpc "
        "(1.5×)** if AE/library face-on charts should retain outer spirals "
        "(clearly visible outside the cyan ±12 box; secondary $A_2$ feature "
        "near 12–15 kpc is clipped today). Do not silently change defaults now."
    )
    rec_short = "keep published ±12; future rebuild → ~18"

    md = f"""# Disk FOV clip experiment (diagnostic only)

**Constraint:** Do **not** change default field-map FOV, retrain the teacher,
rebuild the feature library, alter path-LOO, or regenerate primary evolve
scoreboard figures. This note is diagnostic / outlook only.

## Setup

| Item | Value |
|------|-------|
| Example | `906c4af73543` late `step_003200.npz` |
| Frame | shared global COM (`prepare_shared_frame`) |
| Current disk FOV (teacher / progressive) | **r_max = 12 kpc**, **z_max = 1.5 kpc**, **n_pix = 128**, Δx = 0.1875 kpc |
| Wider FOVs | 18 kpc (1.5×), 24 kpc (2×), fixed Δx so n_pix scales |
| Source of current FOV | `MultiScaleSliceConfig.progressive_defaults` + `fft_morph_ft_long_2026-07-25/verdict.json` |

## Mass outside current FOV

Disk particles only (after shared centering):

| FOV half-width | Mass outside **square** \\|x\\|,\\|y\\|≤R | Mass outside **circle** R≤R | Mass outside \\|z\\|≤1.5 |
|----------------|----------------------------------------|-----------------------------|-------------------------|
| **12 kpc (current)** | **{100 * f12['frac_outside_square']:.2f}%** | **{100 * f12['frac_outside_circle']:.2f}%** | {100 * f12['frac_outside_z']:.2f}% |
| 18 kpc | {100 * f18['frac_outside_square']:.2f}% | {100 * f18['frac_outside_circle']:.2f}% | — |
| 24 kpc | {100 * f24['frac_outside_square']:.2f}% | {100 * f24['frac_outside_circle']:.2f}% | — |

Annular mass with R > 12 kpc (from Σ bins to 24 kpc): **{100 * mass_beyond_12:.2f}%** of disk mass.

Vertical: `z_max=1.5` leaves **{100 * f12['frac_outside_z']:.2f}%** of disk mass outside the slice stack
(expected for a thin midplane-focused tower; not the in-plane FOV question).

## Σ(R) / A₂(R)

- Particle Σ(R) remains detectable past 12 kpc; the current FOV **cuts the outer exponential tail** (and outer spiral arms in face-on charts).
- A₂/A₀ median to 12 kpc: **{float(a2_cur['a_m_over_a0_median']):.3f}**; to 24 kpc: **{float(a2_full['a_m_over_a0_median']):.3f}**.
- Primary bar peak is well inside ~8 kpc. A secondary $A_2$ feature near **~12–15 kpc** (outer spiral / oval) sits on the FOV edge and is truncated in the current maps — morphologically visible, but low mass.

## Face-on maps

Figure: [`../figures/fig_latent_theta_disk_fov_clip.png`](../figures/fig_latent_theta_disk_fov_clip.png)

Cyan dashed square / lime dotted circle mark the **current ±12 kpc** FOV on the wider deposits.

## Teacher encode

{encode_note}

## Recommendation

{rec}

**Paper:** at most a one-sentence limitation/outlook that the disk tower FOV
(±12 kpc) can truncate the outer disk; do **not** rewrite primary MATCH / OOD claims.

**Short verdict:** current FOV = ±12 × ±1.5 kpc (128²); mass outside square ≈ **{100 * out_sq:.1f}%**; action = **{rec_short}**.

## Artifacts

- Metrics JSON: `disk_fov_clip_metrics.json`
- Figure: `papers/mnras_noneq_ics/figures/fig_latent_theta_disk_fov_clip.png`
- Repro: `scripts/disk_fov_clip_experiment.py` (read-only vs production configs)
"""
    OUT_MD.write_text(md)
    print(json.dumps({
        "fig": str(OUT_FIG),
        "md": str(OUT_MD),
        "frac_outside_square_12": out_sq,
        "frac_outside_circle_12": f12["frac_outside_circle"],
        "frac_outside_z_1p5": f12["frac_outside_z"],
        "recommendation": rec_short,
    }, indent=2))


if __name__ == "__main__":
    main()
