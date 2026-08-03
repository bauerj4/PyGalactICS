#!/usr/bin/env python3
"""Evolve-gate best generative recipes (proper dt, many steps).

Compares amplify_knn_hybrid / z_amplify / disk_only_amplify / skip_residual_pca
(+ optional soft COM recenter) vs a barred data dump subsample.

  OMP_NUM_THREADS=6 python scripts/evolve_gate_recipes.py \\
    --evolve-gyr 0.50 --dt 0.01 --n-evolve 30000 --soft-com
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sample_latent_ic import (  # noqa: E402
    COUNT,
    CRISP,
    RANK,
    _a2_disk,
    build_library,
)
from sample_skip_residual_pca import (  # noqa: E402
    build_residual_pca,
    sample_skip_pca,
)
from smoke_field_maps import _evolve_bh  # noqa: E402

from galacticsics.ml.fields.frame import prepare_shared_frame  # noqa: E402
from galacticsics.ml.fields.normalize import denormalize_stack  # noqa: E402
from galacticsics.ml.fields.resample import resample_particles_from_multiscale  # noqa: E402
from galacticsics.ml.morton.polygon import _component_ids  # noqa: E402
from ntropy.analysis.disk_density import disk_azimuthal_fourier  # noqa: E402


def _subsample_dump(path: Path, n: int, rng: np.random.Generator) -> dict:
    with np.load(path, allow_pickle=True) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        eps = (
            np.asarray(data["eps"], dtype=np.float64)
            if "eps" in data.files
            else np.full(len(pos), 0.05)
        )
        cid = _component_ids(data.get("tags"), data.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    fr = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
    name_to_id = {"disk": 0, "halo": 1, "bulge": 2}
    parts = []
    for name, f in fr.items():
        nk = max(1, int(round(n * f)))
        mask = cid == name_to_id[name]
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        take = rng.choice(idx, size=min(nk, idx.size), replace=idx.size < nk)
        parts.append(take)
    sel = np.concatenate(parts)
    if sel.size > n:
        sel = rng.choice(sel, size=n, replace=False)
    return {
        "pos": pos[sel],
        "vel": vel[sel],
        "mass": mass[sel],
        "eps": eps[sel],
        "component_id": cid[sel],
    }


def _soft_com_recenter(parts: dict) -> dict:
    """Subtract mass-weighted COM / VCOM (shared global — never per-component)."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in parts.items()}
    m = out["mass"]
    com = np.average(out["pos"], axis=0, weights=m)
    vcom = np.average(out["vel"], axis=0, weights=m)
    out["pos"] = out["pos"] - com
    out["vel"] = out["vel"] - vcom
    return out


def _a2_post(parts: dict, *, r_eval: float | None = None) -> float:
    disk = parts["component_id"] == 0
    fout = disk_azimuthal_fourier(
        parts["pos"][disk],
        parts["mass"][disk],
        m=2,
        r_max=12.0,
        n_bins=12,
        z_max=0.5,
        min_count=10,
        r_eval=r_eval,
    )
    if r_eval is not None:
        return float(fout["a_m_over_a0_at_r"])
    return float(fout["a_m_over_a0_median"])


def _metrics(parts: dict, *, r_eval: float | None = None) -> dict:
    a2 = _a2_post(parts, r_eval=r_eval) if r_eval is not None else _a2_disk(parts)
    com = np.average(parts["pos"], axis=0, weights=parts["mass"])
    rng = np.random.default_rng(0)
    n_sub = min(256, parts["pos"].shape[0])
    idx = rng.choice(parts["pos"].shape[0], size=n_sub, replace=False)
    p = parts["pos"][idx]
    v = parts["vel"][idx]
    m = parts["mass"][idx]
    ke = 0.5 * np.sum(m * np.sum(v * v, axis=1))
    dr = p[:, None, :] - p[None, :, :]
    r2 = np.sum(dr * dr, axis=-1)
    np.fill_diagonal(r2, np.inf)
    pe = -0.5 * np.sum(m[:, None] * m[None, :] / np.sqrt(r2 + 0.1**2))
    q = float(2.0 * ke / max(abs(pe), 1e-8))
    return {"a2": float(a2), "com_norm": float(np.linalg.norm(com)), "Q_sub": q}


def _downsample(parts: dict, n: int, rng: np.random.Generator) -> dict:
    """Stratified downsample preserving ≈4:2:1 disk:halo:bulge counts."""
    n0 = parts["pos"].shape[0]
    n_ev = min(n, n0)
    cid = parts["component_id"]
    fr = {0: 4 / 7, 1: 2 / 7, 2: 1 / 7}
    takes = []
    for c, f in fr.items():
        idx = np.where(cid == c)[0]
        if idx.size == 0:
            continue
        nk = max(1, int(round(n_ev * f)))
        nk = min(nk, idx.size)
        takes.append(rng.choice(idx, size=nk, replace=False))
    take = np.concatenate(takes) if takes else rng.choice(n0, size=n_ev, replace=False)
    if take.size > n_ev:
        take = rng.choice(take, size=n_ev, replace=False)
    elif take.size < n_ev:
        rest = np.setdiff1d(np.arange(n0), take, assume_unique=False)
        if rest.size:
            need = min(n_ev - take.size, rest.size)
            take = np.concatenate([take, rng.choice(rest, size=need, replace=False)])
    out = {
        k: (v[take] if isinstance(v, np.ndarray) and getattr(v, "shape", (0,))[0] == n0 else v)
        for k, v in parts.items()
    }
    out["eps"] = np.full(out["pos"].shape[0], 0.1, dtype=np.float64)
    return out


def _fields_to_parts(lib, fields, n_resample, rng):
    phys = {
        k: denormalize_stack(v[0].detach().cpu().numpy(), lib.stats[k])
        for k, v in fields.items()
    }
    return resample_particles_from_multiscale(
        phys, cfg=lib.cfg, n_particles=n_resample, count_fractions=COUNT, rng=rng
    )


def _sample_recipe(lib, pca, method: str, rng):
    if method == "amplify_knn_hybrid":
        fields, meta = lib.sample_fields(
            kind="barred", method="amplify_knn_hybrid", rng=rng, alpha_lo=1.20, alpha_hi=1.45
        )
        return fields, meta
    if method == "disk_only_amplify":
        fields, meta = lib.sample_fields(
            kind="barred", method="disk_only_amplify", rng=rng, alpha_lo=1.20, alpha_hi=1.50
        )
        return fields, meta
    if method == "z_amplify":
        feat, meta = lib.sample_features_z_amplify(
            kind="barred", rng=rng, alpha_lo=1.25, alpha_hi=1.55
        )
        return lib.decode_features(feat), meta
    if method == "skip_residual_pca":
        feat, meta = sample_skip_pca(lib, pca, rng, beta_scale=1.55)
        # Mild amplify residual vs quiet to restore HF bar (pathway #3 hybrid)
        if lib._quiet_mean is not None:
            from galacticsics.ml.fields.feature_library import scale_residual

            alpha = float(rng.uniform(1.10, 1.35))
            feat = scale_residual(lib._quiet_mean, feat, alpha)
            meta = {**meta, "post_amplify_alpha": alpha}
        return lib.decode_features(feat), meta
    raise ValueError(method)


def _run_evolve(parts, args):
    r_eval = getattr(args, "a2_r_eval", None)
    pre = _metrics(parts, r_eval=r_eval)
    t0 = time.time()
    evo = _evolve_bh(
        parts,
        end_gyr=args.evolve_gyr,
        dt=args.dt,
        force="bh_c",
        omp_threads=args.omp,
        timeout_s=args.timeout_s,
    )
    wall = time.time() - t0
    if not evo.get("ok"):
        return {"pre": pre, "evolve": evo, "wall_s": wall, "ok": False}
    post_parts = {**parts, "pos": evo.pop("pos_final")}
    a2_post = _a2_post(post_parts, r_eval=r_eval)
    post = _metrics(post_parts, r_eval=r_eval)
    post["a2"] = a2_post
    return {
        "ok": True,
        "pre": pre,
        "post": post,
        "a2_pre": pre["a2"],
        "a2_post": a2_post,
        "da2": a2_post - pre["a2"],
        "com_drift_kpc": evo["com_drift_kpc"],
        "Q_pre": pre["Q_sub"],
        "Q_post": post["Q_sub"],
        "n_steps": evo["n_steps"],
        "force_method": evo["force_method"],
        "wall_s": evo.get("wall_s", wall),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ml/field_maps/evolve_gate_2026-07-25"))
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar", type=int, default=36)
    p.add_argument("--n-quiet", type=int, default=20)
    p.add_argument("--n-mid", type=int, default=12)
    p.add_argument("--n-rot-bar", type=int, default=6)
    p.add_argument("--bar-floor", type=float, default=0.25)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-comp", type=int, default=8)
    p.add_argument("--n-evolve", type=int, default=30_000)
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--evolve-gyr", type=float, default=0.50)
    p.add_argument("--omp", type=int, default=6)
    p.add_argument("--timeout-s", type=float, default=1800.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--soft-com", action="store_true", default=True)
    p.add_argument("--no-soft-com", action="store_false", dest="soft_com")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["amplify_knn_hybrid", "z_amplify", "disk_only_amplify", "skip_residual_pca"],
    )
    p.add_argument("--n-rep", type=int, default=3, help="replicates per method (barred)")
    p.add_argument(
        "--ic-a2-min",
        type=float,
        default=0.30,
        help="Reject samples with IC particle A2 below this (retry); 0 disables",
    )
    p.add_argument("--ic-a2-tries", type=int, default=8)
    p.add_argument(
        "--a2-r-eval",
        type=float,
        default=None,
        help=(
            "If set, score pre/post A₂ at this R [kpc] (interp of ring A₂; "
            "e.g. R_d) instead of median A₂. Default: median."
        ),
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(args.omp)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    data_path = Path(ranked[0]["path"])

    print("=== build stratified teacher library ===", flush=True)
    lib = build_library(args)
    print("=== disk skip-residual PCA ===", flush=True)
    pca = build_residual_pca(lib, strong_floor=0.28, n_comp=args.n_comp)

    report: dict = {
        "dt": args.dt,
        "evolve_gyr": args.evolve_gyr,
        "n_steps": int(np.ceil(args.evolve_gyr / args.dt)),
        "n_evolve": args.n_evolve,
        "omp": args.omp,
        "soft_com": bool(args.soft_com),
        "teacher": str(args.teacher),
        "data_path": str(data_path),
        "methods": {},
    }

    # Data reference once
    data_ev = _subsample_dump(data_path, args.n_evolve, rng)
    if args.soft_com:
        data_ev = _soft_com_recenter(data_ev)
    print("=== evolve data dump ===", flush=True)
    report["data"] = _run_evolve(data_ev, args)
    print(
        f"  data A2 {report['data'].get('a2_pre', float('nan')):.3f}→"
        f"{report['data'].get('a2_post', float('nan')):.3f}  "
        f"COM={report['data'].get('com_drift_kpc', float('nan')):.4f}",
        flush=True,
    )

    for method in args.methods:
        rows = []
        for rep in range(args.n_rep):
            print(f"=== sample+evolve {method} rep={rep} ===", flush=True)
            best = None
            n_try = max(1, int(args.ic_a2_tries)) if args.ic_a2_min > 0 else 1
            for attempt in range(n_try):
                fields, meta = _sample_recipe(lib, pca, method, rng)
                gen = _fields_to_parts(lib, fields, args.n_resample, rng)
                gen_ev = _downsample(gen, args.n_evolve, rng)
                if args.soft_com:
                    gen_ev = _soft_com_recenter(gen_ev)
                a2_ic = float(_a2_disk(gen_ev))
                cand = (a2_ic, fields, meta, gen_ev)
                if best is None or a2_ic > best[0]:
                    best = cand
                print(f"  try#{attempt} IC A2={a2_ic:.3f}", flush=True)
                if args.ic_a2_min <= 0 or a2_ic >= args.ic_a2_min:
                    break
            assert best is not None
            a2_ic, _fields, meta, gen_ev = best
            row = _run_evolve(gen_ev, args)
            row["ic_a2_selected"] = a2_ic
            row["sample_meta"] = {k: v for k, v in meta.items() if k != "z" and k != "beta"}
            if "beta" in meta:
                row["sample_meta"]["beta_norm"] = float(np.linalg.norm(meta["beta"]))
            rows.append(row)
            print(
                f"  {method}#{rep} A2 {row.get('a2_pre', float('nan')):.3f}→"
                f"{row.get('a2_post', float('nan')):.3f}  "
                f"dA2={row.get('da2', float('nan')):+.3f}  "
                f"COM={row.get('com_drift_kpc', float('nan')):.4f}  "
                f"wall={row.get('wall_s', float('nan')):.1f}s",
                flush=True,
            )
        ok = [r for r in rows if r.get("ok")]
        summary = {
            "n_ok": len(ok),
            "a2_pre_mean": float(np.mean([r["a2_pre"] for r in ok])) if ok else None,
            "a2_post_mean": float(np.mean([r["a2_post"] for r in ok])) if ok else None,
            "da2_mean": float(np.mean([r["da2"] for r in ok])) if ok else None,
            "com_drift_mean": float(np.mean([r["com_drift_kpc"] for r in ok])) if ok else None,
            "reps": rows,
        }
        # Gate: bar survives (post>=0.30) and |dA2| not much worse than data
        data_da2 = abs(report["data"].get("da2", 0.05) or 0.05)
        summary["passes_evolve_gate"] = bool(
            ok
            and summary["a2_post_mean"] is not None
            and summary["a2_post_mean"] >= 0.30
            and abs(summary["da2_mean"]) <= max(0.08, 2.0 * data_da2)
            and (summary["com_drift_mean"] or 99) <= 0.05
        )
        report["methods"][method] = summary

    out_json = args.out / "verdict.json"
    out_json.write_text(json.dumps(report, indent=2))
    # Compact summary markdown
    lines = [
        "# Evolve-gated generative recipes",
        "",
        f"Settings: `dt={args.dt}`, `end_gyr={args.evolve_gyr}` "
        f"→ **{report['n_steps']}** leapfrog steps; N={args.n_evolve}; "
        f"soft_com={args.soft_com}; OpenMP={args.omp}.",
        "",
        "| Recipe | A₂ pre→post | ΔA₂ | COM drift | Gate |",
        "|--------|-------------|-----|-----------|------|",
    ]
    if report["data"].get("ok"):
        d = report["data"]
        lines.append(
            f"| **data dump** | {d['a2_pre']:.3f}→{d['a2_post']:.3f} | "
            f"{d['da2']:+.3f} | {d['com_drift_kpc']:.4f} | ref |"
        )
    for method, s in report["methods"].items():
        if s["a2_pre_mean"] is None:
            lines.append(f"| `{method}` | FAIL | — | — | no |")
            continue
        gate = "PASS" if s["passes_evolve_gate"] else "fail"
        lines.append(
            f"| `{method}` | {s['a2_pre_mean']:.3f}→{s['a2_post_mean']:.3f} | "
            f"{s['da2_mean']:+.3f} | {s['com_drift_mean']:.4f} | **{gate}** |"
        )
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}", flush=True)
    print((args.out / "SUMMARY.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
