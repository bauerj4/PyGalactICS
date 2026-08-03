#!/usr/bin/env python3
"""t=0 A₂(R_d) objective ablation: Rd-weight + sharpen + α dial."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ntropy.analysis.disk_density import disk_azimuthal_fourier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from galacticsics.ml.fields.feature_library import load_frozen_teacher_bundle  # noqa: E402
from galacticsics.ml.fields.frame import prepare_shared_frame  # noqa: E402
from galacticsics.ml.morton.polygon import _component_ids  # noqa: E402
from evolve_resample_compare import (  # noqa: E402
    _a2_disk,
    _residual_f0_particles,
    _stratified_down,
    _vcom_only,
)
from residual_galactics_ic import _ensure_eps  # noqa: E402


def health(parts, rd: float = 2.0):
    disk = parts["component_id"] == 0
    pos = parts["pos"][disk]
    mass = parts["mass"][disk]
    R = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    order = np.argsort(R)
    c = np.cumsum(mass[order])
    c /= c[-1]
    r50 = float(np.interp(0.5, c, R[order]))
    r90 = float(np.interp(0.9, c, R[order]))
    a2 = float(_a2_disk(parts))
    fout = disk_azimuthal_fourier(
        pos, mass, m=2, r_max=12.0, n_bins=24, z_max=0.5, min_count=10, r_eval=rd
    )
    a2rd = fout.get("a_m_over_a0_at_r")
    if a2rd is None or not np.isfinite(a2rd):
        r = np.asarray(fout["r_mid"])
        a = np.asarray(fout["a_m_over_a0"])
        ok = np.isfinite(a)
        a2rd = float(np.interp(rd, r[ok], a[ok]))
    else:
        a2rd = float(a2rd)
    return {"a2_med": a2, "a2_Rd": a2rd, "R50": r50, "R90": r90}


def main() -> int:
    teacher_path = Path(
        "runs/ml/field_maps/df_match_BA_2026-07-27/joint_df_fftlong_ft/multitower_slice_ae.pt"
    )
    ic = Path("runs/mw_morton_corpus_v2/906c4af73543/ic_state.npz")
    morph = Path(
        "runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz"
    )
    out = Path("runs/ml/field_maps/residual_f0_2026-07-28/a2rd_objective_t0")
    out.mkdir(parents=True, exist_ok=True)
    rd = 2.0
    target = 0.49
    n_tot = int(round(1_000_000 * 7 / 4))

    teacher, cfg, stats = load_frozen_teacher_bundle(teacher_path)
    rng = np.random.default_rng(0)

    with np.load(morph, allow_pickle=True) as d:
        pos = d["pos"].astype(np.float64)
        vel = d["vel"].astype(np.float64)
        mass = d["mass"].astype(np.float64)
        cid = _component_ids(d.get("tags"), d.get("type_id"), pos.shape[0])
        pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    morph_h = health({"pos": pos, "vel": vel, "mass": mass, "component_id": cid}, rd)

    # (name, morph_source, blend_w, alpha, alpha_mode, sharpen, rpeak, rsig)
    cfgs = [
        ("prior_teacher_a35", "teacher_recon", 0.0, 3.5, "fixed", 0.0, None, None),
        ("prior_blend90_a125", "blend", 0.9, 1.25, "fixed", 0.0, None, None),
        ("prior_deposit_a14", "deposit", 0.0, 1.4, "fixed", 0.0, None, None),
        ("tea_rdw_shp_a15", "teacher_recon", 0.0, 1.5, "fixed", 1.0, 2.0, 2.0),
        ("tea_rdw_shp_match", "teacher_recon", 0.0, 1.5, "match_a2_rd", 1.0, 2.0, 2.0),
        ("tea_rdw_shp15_match", "teacher_recon", 0.0, 1.5, "match_a2_rd", 1.5, 2.0, 2.0),
        ("blend90_rdw_shp_match", "blend", 0.9, 1.0, "match_a2_rd", 0.75, 2.0, 2.0),
        ("blend90_rdw_match", "blend", 0.9, 1.0, "match_a2_rd", 0.0, 2.0, 2.0),
        ("dep_rdw_match", "deposit", 0.0, 1.0, "match_a2_rd", 0.0, 2.0, 2.0),
        ("dep_rdw_shp_match", "deposit", 0.0, 1.0, "match_a2_rd", 0.5, 2.0, 2.0),
        ("dep_match_noshape", "deposit", 0.0, 1.0, "match_a2_rd", 0.0, None, None),
    ]

    rows = []
    for name, src, bw, alpha, amode, sharpen, rpeak, rsig in cfgs:
        print(f"=== {name} ===", flush=True)
        parts = _residual_f0_particles(
            ic,
            morph,
            teacher,
            cfg,
            stats,
            n_resample=n_tot,
            rng=rng,
            morph_source=src,
            alpha=alpha,
            dens_resid_kind="m2",
            blend_weight=bw,
            r_weight_peak=rpeak,
            r_weight_sigma=rsig,
            contrast_sharpen=sharpen,
            alpha_mode=amode,
            target_a2_rd=target,
            a2_r_eval=rd,
        )
        parts = _ensure_eps(_vcom_only(_stratified_down(parts, n_tot, rng)))
        h = health(parts, rd)
        h.update(
            name=name,
            src=src,
            alpha_req=alpha,
            alpha_used=float(parts.get("dens_resid_alpha_used", alpha)),
            alpha_mode=amode,
            sharpen=sharpen,
            r_peak=rpeak,
            a2_err=abs(h["a2_Rd"] - morph_h["a2_Rd"]),
        )
        h["R50_ratio"] = h["R50"] / 3.46
        h["R90_ratio"] = h["R90"] / 7.97
        h["ok"] = abs(h["R50_ratio"] - 1) < 0.08 and abs(h["R90_ratio"] - 1) < 0.08
        rows.append(h)
        print(
            f"  α={h['alpha_used']:.3f} A2_med={h['a2_med']:.3f} "
            f"A2(Rd)={h['a2_Rd']:.3f} err={h['a2_err']:.3f} health={h['ok']}",
            flush=True,
        )

    lines = [
        "# A₂(R_d) objective t=0 ablation (906c4, disk=1e6, fixed injector)",
        "",
        f"morph dump target: A2_med={morph_h['a2_med']:.3f} A2(Rd={rd})={morph_h['a2_Rd']:.3f}",
        f"match target A2(Rd)={target}",
        "",
        "| name | src | α_used | sharp | RdW | A₂ med | A₂(R_d) | |ΔRd| | health |",
        "|------|-----|--------|-------|-----|--------|---------|------|--------|",
    ]
    for r in rows:
        rdw = "Y" if r["r_peak"] is not None else "—"
        lines.append(
            f"| `{r['name']}` | {r['src']} | {r['alpha_used']:.3f} | "
            f"{r['sharpen']:.2f} | {rdw} | {r['a2_med']:.3f} | {r['a2_Rd']:.3f} | "
            f"{r['a2_err']:.3f} | {'OK' if r['ok'] else 'FAIL'} |"
        )
    # Prefer: health OK, small |ΔRd|, prefer deposit/blend over soft teacher for morph.
    ranked = sorted(
        [r for r in rows if r["ok"]],
        key=lambda r: (r["a2_err"], 0 if "dep" in r["name"] or "blend" in r["name"] else 1),
    )
    lines += ["", "## Suggested evolve set (health OK, nearest A₂(R_d))"]
    for r in ranked[:4]:
        lines.append(
            f"- `{r['name']}`: A₂(Rd)={r['a2_Rd']:.3f} (α={r['alpha_used']:.3f}, "
            f"src={r['src']}, sharp={r['sharpen']})"
        )
    summary = "\n".join(lines) + "\n"
    (out / "SUMMARY.md").write_text(summary)
    (out / "verdict.json").write_text(
        json.dumps({"morph": morph_h, "target": target, "rows": rows}, indent=2, default=float)
        + "\n"
    )
    paper = Path("papers/mnras_noneq_ics/results/residual_f0_a2rd_objective_t0_SUMMARY.md")
    paper.write_text(summary)
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
