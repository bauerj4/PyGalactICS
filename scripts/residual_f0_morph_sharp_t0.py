#!/usr/bin/env python3
"""t=0 morph sharpness: teacher m2 vs full, deposit m2, α ladder."""
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
    out = Path("runs/ml/field_maps/residual_f0_2026-07-28/morph_sharp")
    out.mkdir(parents=True, exist_ok=True)
    rd = 2.0
    n_tot = int(round(1_000_000 * 7 / 4))

    teacher, cfg, stats = load_frozen_teacher_bundle(teacher_path)
    rng = np.random.default_rng(0)

    with np.load(ic, allow_pickle=True) as d:
        pos = d["pos"].astype(np.float64)
        vel = d["vel"].astype(np.float64)
        mass = d["mass"].astype(np.float64)
        cid = _component_ids(d.get("tags"), d.get("type_id"), pos.shape[0])
        pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    f0h = health({"pos": pos, "vel": vel, "mass": mass, "component_id": cid}, rd)

    rows = []
    cfgs = [
        ("teacher_m2_a10", "teacher_recon", "m2", 1.0),
        ("teacher_m2_a15", "teacher_recon", "m2", 1.5),
        ("teacher_m2_a25", "teacher_recon", "m2", 2.5),
        ("teacher_full_a10", "teacher_recon", "full", 1.0),
        ("teacher_full_a15", "teacher_recon", "full", 1.5),
        ("deposit_m2_a10", "deposit", "m2", 1.0),
        ("deposit_m2_a15", "deposit", "m2", 1.5),
        ("deposit_m2_a25", "deposit", "m2", 2.5),
        ("deposit_full_a10", "deposit", "full", 1.0),
    ]
    for name, src, kind, alpha in cfgs:
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
            dens_resid_kind=kind,
        )
        parts = _ensure_eps(_vcom_only(_stratified_down(parts, n_tot, rng)))
        h = health(parts, rd)
        h.update(name=name, src=src, kind=kind, alpha=alpha)
        h["R50_ratio"] = h["R50"] / f0h["R50"]
        h["R90_ratio"] = h["R90"] / f0h["R90"]
        h["ok"] = abs(h["R50_ratio"] - 1) < 0.08 and abs(h["R90_ratio"] - 1) < 0.08
        rows.append(h)
        print(h, flush=True)

    with np.load(morph, allow_pickle=True) as d:
        pos = d["pos"].astype(np.float64)
        vel = d["vel"].astype(np.float64)
        mass = d["mass"].astype(np.float64)
        cid = _component_ids(d.get("tags"), d.get("type_id"), pos.shape[0])
        pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    morph_h = health({"pos": pos, "vel": vel, "mass": mass, "component_id": cid}, rd)
    morph_h["name"] = "morph_dump_data"

    lines = [
        "# Morph sharpness t=0 ablation (906c4, disk=1e6, fixed injector)",
        "",
        f"f0: A2_med={f0h['a2_med']:.3f} A2(Rd={rd})={f0h['a2_Rd']:.3f} "
        f"R50={f0h['R50']:.2f} R90={f0h['R90']:.2f}",
        f"morph dump: A2_med={morph_h['a2_med']:.3f} A2(Rd)={morph_h['a2_Rd']:.3f}",
        "",
        "| name | src | kind | α | A₂ med | A₂(R_d) | R50/f0 | health |",
        "|------|-----|------|---|--------|---------|--------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['src']} | {r['kind']} | {r['alpha']} | "
            f"{r['a2_med']:.3f} | {r['a2_Rd']:.3f} | {r['R50_ratio']:.3f} | "
            f"{'OK' if r['ok'] else 'FAIL'} |"
        )
    summary = "\n".join(lines) + "\n"
    (out / "SUMMARY.md").write_text(summary)
    (out / "verdict.json").write_text(
        json.dumps({"f0": f0h, "morph": morph_h, "rows": rows}, indent=2, default=float)
        + "\n"
    )
    Path("papers/mnras_noneq_ics/results/residual_f0_morph_sharp_t0_SUMMARY.md").write_text(
        summary
    )
    print(summary, flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
