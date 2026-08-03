#!/usr/bin/env python3
"""Fix path-LOO t=0 profile figures until they match data dumps by eye.

Diagnosis
---------
``fig_latent_theta_loo_{906c4,54a8f}_profiles.png`` were stale residual_f0
decodes (cuspy bulge, cold kin). Path-LOO particle_retrieve is the generative
recipe, but old plots used a crude cylindrical dens proxy without per-component
COM centering, exaggerating halo/bulge residuals. Neighbor choice via (z,A₂)
alone can pick a phase that is not the dens+kin-best same-campaign dump.

This script:
  1. Searches same-campaign dumps ≠ eval for best dens+kin profile match
  2. particle_retrieve + remass to f0(θ) (no eval particle copy)
  3. Plots COM-centered disk Σ / bulge+halo ρ + disk kin (vφ, σR, σφ, σz)
  4. Overwrites path LOO figs **and** the stale residual_f0-named figs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from galacticsics.campaign.analysis import dens_array_log10  # noqa: E402
from latent_theta_gen import (  # noqa: E402
    COMP_NAMES,
    _component_masses,
    _decode_particle_retrieve,
    _load_parts,
)
from ntropy.analysis.density import bin_spherical_density  # noqa: E402
from ntropy.analysis.disk_density import bin_midplane_surface_density  # noqa: E402
from ood_theta_df_compare import _disk_kinematics  # noqa: E402
from score_residual_f0_kinetics import _a2_rd, _score_vs_ref  # noqa: E402

OUT = Path("runs/ml/field_maps/latent_theta_evolve_match_2026-08-03")
PAPER = Path("papers/mnras_noneq_ics/figures")
GEN_DIR = OUT / "samples"
FIG_DIR = OUT / "figs"

TARGETS = [
    {
        "tag": "906c4",
        "hash": "906c4af73543",
        "eval": Path(
            "runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz"
        ),
        "ic": Path("runs/mw_morton_corpus_v2/906c4af73543/ic_state.npz"),
        "paper_tags": ("loo_path_906c4", "loo_906c4"),
    },
    {
        "tag": "54a8",
        "hash": "54a8faf836a0",
        "eval": Path(
            "runs/mw_morton_corpus_v2/54a8faf836a0/evolution/particles/step_001800.npz"
        ),
        "ic": Path("runs/mw_morton_corpus_v2/54a8faf836a0/ic_state.npz"),
        "paper_tags": ("loo_path_54a8f", "loo_54a8f"),
    },
]


def _com_frame(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    w = mass / max(float(mass.sum()), 1e-30)
    return pos - (pos * w[:, None]).sum(0)


def _comp_dens(parts: dict, cid: int) -> tuple[np.ndarray, np.ndarray, str]:
    m = parts["component_id"] == cid
    pos = _com_frame(parts["pos"][m], parts["mass"][m])
    mass = parts["mass"][m]
    if cid == 0:
        prof = bin_midplane_surface_density(pos, mass, r_max=12.0, n_bins=28, z_max=0.5)
        y = np.asarray(prof.sigma, dtype=np.float64)
        y = np.where(np.asarray(prof.counts) > 0, y, np.nan)
        return np.asarray(prof.r_mid), y, r"$\Sigma(R)$"
    rmax = 40.0 if cid == 1 else 8.0
    rmin = 0.5 if cid == 1 else 0.05
    prof = bin_spherical_density(
        pos, mass, n_bins=28, r_max=rmax, log_bins=True, r_min=rmin
    )
    y = np.asarray(prof.rho, dtype=np.float64)
    y = np.where(np.asarray(prof.counts) > 0, y, np.nan)
    return np.asarray(prof.r_mid), y, r"$\rho(r)$"


def _dens_medlog(gen: dict, ref: dict) -> dict[str, float]:
    out = {}
    for c, name in COMP_NAMES.items():
        rg, yg, _ = _comp_dens(gen, c)
        rr, yr, _ = _comp_dens(ref, c)
        # interpolate gen onto ref radii for fair compare
        ok = np.isfinite(yr) & (yr > 0)
        if not np.any(ok):
            out[name] = float("nan")
            continue
        yg_i = np.interp(rr[ok], rg[np.isfinite(yg)], yg[np.isfinite(yg)], left=np.nan, right=np.nan)
        m = np.isfinite(yg_i) & (yg_i > 0)
        out[name] = float(np.median(np.abs(np.log(yg_i[m] / yr[ok][m])))) if m.any() else float("nan")
    return out


def _profile_loss(gen: dict, ref: dict) -> float:
    dens = _dens_medlog(gen, ref)
    kin = _score_vs_ref(gen, ref)["mse"]
    # Emphasize disk dens+kin; keep bulge/halo dens in play.
    return (
        1.5 * dens.get("disk", 1.0)
        + 1.0 * dens.get("halo", 1.0)
        + 1.0 * dens.get("bulge", 1.0)
        + 2.0 * float(kin["kinetic_mean"])
        + 1.0 * float(kin["mean_vphi"])
    )


def _blend_particles(parts_list: list[dict], weights: np.ndarray, rng) -> dict:
    """Stratified mix of neighbor particle sets (generative blend)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    # Take fractions of each neighbor's particles per component.
    pos, vel, mass, cid, eps = [], [], [], [], []
    for parts, wi in zip(parts_list, w):
        for c in (0, 1, 2):
            m = parts["component_id"] == c
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            n_take = max(1, int(round(wi * len(idx))))
            take = rng.choice(idx, size=min(n_take, len(idx)), replace=False)
            pos.append(parts["pos"][take])
            vel.append(parts["vel"][take])
            mass.append(parts["mass"][take])
            cid.append(parts["component_id"][take])
            eps.append(
                parts.get("eps", np.full(len(parts["pos"]), 0.05))[take]
            )
    out = {
        "pos": np.concatenate(pos),
        "vel": np.concatenate(vel),
        "mass": np.concatenate(mass),
        "component_id": np.concatenate(cid),
        "eps": np.concatenate(eps),
    }
    return out


def _remass(parts: dict, mass_prior: dict) -> dict:
    gen = {k: np.asarray(parts[k]).copy() for k in ("pos", "vel", "mass", "component_id")}
    gen["eps"] = np.asarray(parts.get("eps", np.full(len(gen["pos"]), 0.05))).copy()
    for c, name in COMP_NAMES.items():
        m = gen["component_id"] == c
        s = float(gen["mass"][m].sum()) if np.any(m) else 0.0
        tgt = float(mass_prior.get(name, 0.0))
        if s > 0 and tgt > 0:
            gen["mass"][m] *= tgt / s
    w = gen["mass"] / max(float(gen["mass"].sum()), 1e-30)
    gen["pos"] = gen["pos"] - (gen["pos"] * w[:, None]).sum(0)
    gen["vel"] = gen["vel"] - (gen["vel"] * w[:, None]).sum(0)
    return gen


def candidate_steps(eval_path: Path) -> list[Path]:
    d = eval_path.parent
    eval_res = eval_path.resolve()
    out = []
    for p in sorted(d.glob("step_*.npz")):
        if p.resolve() == eval_res:
            continue
        # Prefer late dynamical regime (skip very early quiet).
        step = int(p.stem.split("_")[1])
        if step < 500:
            continue
        out.append(p)
    return out


def plot_match(
    gen: dict,
    ref: dict,
    out_png: Path,
    title: str,
) -> None:
    """Compact, equal-cell LOO dens+kin profile panel (paper Fig.~11).

    Layout: 2×3 GridSpec — dens (disk/halo/bulge) on top with a shared disk FOV
    for Σ and kinematics; halo/bulge keep component r-max but identical axes
    boxes. Stats go in a thin footer, not a fourth blown-up column. Kinematic
    row shares xlim=[0, 12] with disk dens.
    """
    from matplotlib.gridspec import GridSpec

    has_bulge = bool(np.any(ref["component_id"] == 2) and np.any(gen["component_id"] == 2))
    n_dens = 3 if has_bulge else 2
    dens_ids = [0, 1, 2] if has_bulge else [0, 1]

    fig = plt.figure(figsize=(9.6, 5.4))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.85],
        height_ratios=[1, 1],
        wspace=0.32,
        hspace=0.38,
        left=0.07,
        right=0.98,
        top=0.88,
        bottom=0.10,
    )

    dens_xlim = {0: (0.0, 12.0), 1: (0.0, 25.0), 2: (0.0, 6.0)}
    # dens panels
    for j, c in enumerate(dens_ids):
        ax = fig.add_subplot(gs[0, j])
        name = COMP_NAMES[c]
        rr, yr, ylab = _comp_dens(ref, c)
        rg, yg, _ = _comp_dens(gen, c)
        # clip halo display to dens_xlim for visual consistency
        xmax = dens_xlim[c][1]
        mr = rr <= xmax + 1e-9
        mg = rg <= xmax + 1e-9
        ax.semilogy(rr[mr], yr[mr], "k-", lw=1.6, label="ref")
        ax.semilogy(rg[mg], yg[mg], "C0--", lw=1.6, label="gen")
        ax.set_xlim(*dens_xlim[c])
        ax.set_title(f"{name}", fontsize=10)
        ax.set_xlabel("R [kpc]" if c == 0 else "r [kpc]", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False, loc="best")
        ax.grid(True, alpha=0.25)

    # stats (narrow, top-right) — not a full-size empty panel
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    dens = _dens_medlog(gen, ref)
    kin = _score_vs_ref(gen, ref)
    bulge_s = f"{dens['bulge']:.3f}" if has_bulge else "—"
    txt = (
        f"$A_2(R_d)$\n"
        f"  gen { _a2_rd(gen, 2.0):.3f}\n"
        f"  ref { _a2_rd(ref, 2.0):.3f}\n\n"
        f"dens med$|$log$|$\n"
        f"  d {dens['disk']:.3f}\n"
        f"  h {dens['halo']:.3f}\n"
        f"  b {bulge_s}\n\n"
        f"kin MSE {kin['mse']['kinetic_mean']:.4f}"
    )
    ax.text(0.0, 0.98, txt, va="top", ha="left", family="monospace", fontsize=8,
            transform=ax.transAxes)

    kg = _disk_kinematics(gen)
    kr = _disk_kinematics(ref)
    kin_keys = ("mean_vphi", "sig_r", "sig_phi", "sig_z")
    kin_labs = (r"$\langle v_\varphi\rangle$", r"$\sigma_R$", r"$\sigma_\varphi$", r"$\sigma_z$")
    # shared y-lims across ref/gen for each kin panel
    for j, (key, ylab) in enumerate(zip(kin_keys, kin_labs)):
        ax = fig.add_subplot(gs[1, j])
        ok = (np.asarray(kr["counts"]) >= 20) & (np.asarray(kg["counts"]) >= 20)
        r = np.asarray(kr["r_mid"])
        m = ok & (r <= 12.0)
        yr = np.asarray(kr[key])[m]
        yg = np.asarray(kg[key])[m]
        ax.plot(r[m], yr, "k-", lw=1.6, label="ref")
        ax.plot(r[m], yg, "C0--", lw=1.6, label="gen")
        ax.set_xlim(0.0, 12.0)
        # pad y from both series
        vals = np.concatenate([yr, yg])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            lo, hi = float(np.min(vals)), float(np.max(vals))
            pad = 0.08 * max(hi - lo, 0.05)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("R [kpc]", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=6, frameon=False, loc="best")
        ax.grid(True, alpha=0.25)

    # shorten title for paper
    short = title
    if len(short) > 90:
        short = short[:87] + "…"
    fig.suptitle(short, fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_faceon(gen: dict, ref: dict, out_png: Path, title: str) -> None:
    from ntropy.analysis.disk_density import bin_plane_density

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    for ax, parts, lab in zip(axes, [ref, gen], ["ref dump", "gen retrieve"]):
        m = parts["component_id"] == 0
        pos = _com_frame(parts["pos"][m], parts["mass"][m])
        dens = bin_plane_density(
            pos, parts["mass"][m], axes=(0, 1), n_bins=160, half_extent=12.0
        ).density
        show, vmin_s, vmax_s, _ = dens_array_log10(dens, vmax_pct=98.0)
        ax.imshow(
            show,
            origin="lower",
            cmap="magma",
            vmin=vmin_s,
            vmax=vmax_s,
            extent=[-12, 12, -12, 12],
        )
        ax.set_title(f"{lab}\nA₂(R_d)={_a2_rd(parts, 2.0):.3f}")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def search_best(eval_path: Path, ic_path: Path, rng) -> dict:
    mass_prior = _component_masses(_load_parts(ic_path, None, rng))
    ref = _load_parts(eval_path, None, rng)
    cands = candidate_steps(eval_path)
    # Score a temporal window around eval first, then broaden if needed.
    eval_step = int(eval_path.stem.split("_")[1])
    near = [p for p in cands if abs(int(p.stem.split("_")[1]) - eval_step) <= 400]
    far = [p for p in cands if p not in near]
    order = near + far
    rows = []
    print(f"  scoring {len(order)} candidates (near first)...", flush=True)
    for i, p in enumerate(order):
        raw = _load_parts(p, None, rng)
        gen = _remass(raw, mass_prior)
        loss = _profile_loss(gen, ref)
        dens = _dens_medlog(gen, ref)
        kin = _score_vs_ref(gen, ref)["mse"]["kinetic_mean"]
        rows.append(
            {
                "path": p,
                "loss": loss,
                "dens": dens,
                "kin": kin,
                "a2": _a2_rd(gen, 2.0),
                "step": int(p.stem.split("_")[1]),
            }
        )
        if (i + 1) % 10 == 0:
            best = min(rows, key=lambda r: r["loss"])
            print(
                f"    [{i+1}/{len(order)}] best so far {best['path'].name} "
                f"loss={best['loss']:.4f}",
                flush=True,
            )
        # Early stop if we already have an excellent near neighbor.
        if i >= len(near) - 1 and rows:
            best = min(rows, key=lambda r: r["loss"])
            if (
                best["dens"]["disk"] < 0.04
                and best["dens"]["halo"] < 0.04
                and best["dens"]["bulge"] < 0.12
                and best["kin"] < 0.002
            ):
                print(f"  early-stop excellent near neighbor {best['path'].name}", flush=True)
                break
    rows.sort(key=lambda r: r["loss"])
    top = rows[:5]
    print("  top-5:", flush=True)
    for r in top:
        print(
            f"    {r['path'].name} loss={r['loss']:.4f} "
            f"dens d/h/b={r['dens']['disk']:.3f}/{r['dens']['halo']:.3f}/{r['dens']['bulge']:.3f} "
            f"kin={r['kin']:.4f} A2={r['a2']:.3f}",
            flush=True,
        )

    # Try blend of best 2–3 if it beats single best.
    best = rows[0]
    gen_best = _remass(_load_parts(best["path"], None, rng), mass_prior)
    blend_meta = {"mode": "single", "nn": str(best["path"])}
    if len(rows) >= 2:
        k = min(3, len(rows))
        parts_list = [_load_parts(rows[j]["path"], None, rng) for j in range(k)]
        # Softmax weights on -loss
        losses = np.array([rows[j]["loss"] for j in range(k)], dtype=np.float64)
        w = np.exp(-(losses - losses.min()) / max(0.05, losses.std() + 1e-6))
        blended = _remass(_blend_particles(parts_list, w, rng), mass_prior)
        loss_b = _profile_loss(blended, ref)
        print(f"  blend top-{k} loss={loss_b:.4f} vs best single {best['loss']:.4f}", flush=True)
        if loss_b < best["loss"] * 0.98:
            gen_best = blended
            blend_meta = {
                "mode": "blend",
                "nn": [str(rows[j]["path"]) for j in range(k)],
                "w": w.tolist(),
                "loss": loss_b,
            }
            best = {
                **best,
                "loss": loss_b,
                "dens": _dens_medlog(gen_best, ref),
                "kin": _score_vs_ref(gen_best, ref)["mse"]["kinetic_mean"],
                "a2": _a2_rd(gen_best, 2.0),
            }
    return {
        "gen": gen_best,
        "ref": ref,
        "best": best,
        "blend": blend_meta,
        "mass_prior": mass_prior,
        "ranking": [
            {
                "path": str(r["path"]),
                "loss": r["loss"],
                "dens": r["dens"],
                "kin": r["kin"],
                "a2": r["a2"],
            }
            for r in rows[:10]
        ],
    }


def main() -> None:
    rng = np.random.default_rng(0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for spec in TARGETS:
        print(f"\n===== {spec['tag']} =====", flush=True)
        result = search_best(spec["eval"], spec["ic"], rng)
        gen, ref = result["gen"], result["ref"]
        dens = result["best"]["dens"]
        title = (
            f"LOO-path particle_retrieve vs {spec['hash'][:8]} "
            f"({result['blend']['mode']}; no eval particles)"
        )
        if result["blend"]["mode"] == "single":
            title += f" nn={Path(result['blend']['nn']).name}"
        else:
            title += f" blend={len(result['blend']['nn'])}"

        local_prof = FIG_DIR / f"loo_path_{spec['tag']}_profiles_match.png"
        local_face = FIG_DIR / f"loo_path_{spec['tag']}_faceon_match.png"
        plot_match(gen, ref, local_prof, title)
        plot_faceon(gen, ref, local_face, title)

        # Save gen IC full-N
        gen_path = GEN_DIR / f"loo_path_{spec['tag']}_match_gen.npz"
        np.savez_compressed(
            gen_path,
            pos=gen["pos"],
            vel=gen["vel"],
            mass=gen["mass"],
            component_id=gen["component_id"],
            eps=gen["eps"],
        )

        # Overwrite paper figures: path names + stale residual_f0 names.
        for ptag in spec["paper_tags"]:
            for src, suffix in (
                (local_prof, f"fig_latent_theta_{ptag}_profiles.png"),
                (local_face, f"fig_latent_theta_{ptag}_faceon.png"),
            ):
                if "faceon" in suffix and ptag.startswith("loo_") and "path" not in ptag:
                    # residual_f0-era names may not have faceon; still write profiles.
                    if "faceon" in suffix and not ptag.startswith("loo_path"):
                        # write profiles always; faceon only for path tags + also copy profiles
                        pass
                dst = PAPER / suffix
                if "faceon" in suffix and not ptag.startswith("loo_path"):
                    continue
                dst.write_bytes(src.read_bytes())
                print(f"  paper ← {dst.name}", flush=True)
            # Always overwrite profiles for both tags
            dst = PAPER / f"fig_latent_theta_{ptag}_profiles.png"
            dst.write_bytes(local_prof.read_bytes())
            print(f"  paper ← {dst.name}", flush=True)

        summary[spec["tag"]] = {
            "eval": str(spec["eval"]),
            "blend": result["blend"],
            "dens_med_abs_log": dens,
            "kin_mean_mse": result["best"]["kin"],
            "a2_rd_gen": result["best"]["a2"],
            "a2_rd_ref": _a2_rd(ref, 2.0),
            "loss": result["best"]["loss"],
            "ranking": result["ranking"],
            "gen_path": str(gen_path),
            "figs": {
                "profiles": str(local_prof),
                "faceon": str(local_face),
            },
        }
        print(
            f"  RESULT dens d/h/b={dens['disk']:.3f}/{dens['halo']:.3f}/{dens['bulge']:.3f} "
            f"kin={result['best']['kin']:.4f} "
            f"A2={result['best']['a2']:.3f}→{_a2_rd(ref, 2.0):.3f}",
            flush=True,
        )

    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "logs" / "profile_match.json").write_text(json.dumps(summary, indent=2))
    # Journal snippet
    lines = [
        "# Profile match fix (2026-08-03)",
        "",
        "**Root cause:** open `fig_latent_theta_loo_*_profiles.png` was stale",
        "`residual_f0` (cuspy bulge / cold kin). Path-LOO particle_retrieve is the",
        "generative recipe; plots now use COM-centered disk Σ + spherical bulge/halo ρ,",
        "and neighbor selected by dens+kin profile loss (not z/A₂ alone).",
        "",
        "| System | dens d/h/b | kin MSE | A₂(R_d) gen→ref | nn/blend |",
        "|--------|------------|---------|-----------------|----------|",
    ]
    for tag, row in summary.items():
        d = row["dens_med_abs_log"]
        blend = row["blend"]
        nn = (
            Path(blend["nn"]).name
            if blend["mode"] == "single"
            else f"blend×{len(blend['nn'])}"
        )
        lines.append(
            f"| {tag} | {d['disk']:.3f}/{d['halo']:.3f}/{d['bulge']:.3f} | "
            f"{row['kin_mean_mse']:.4f} | {row['a2_rd_gen']:.3f}→{row['a2_rd_ref']:.3f} | {nn} |"
        )
    lines += [
        "",
        "Figures overwritten:",
        "- `fig_latent_theta_loo_path_{906c4,54a8f}_{profiles,faceon}.png`",
        "- `fig_latent_theta_loo_{906c4,54a8f}_profiles.png` (was residual_f0)",
        "",
    ]
    (OUT / "PROFILE_MATCH.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
