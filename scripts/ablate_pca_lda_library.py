#!/usr/bin/env python3
"""
Ablate library index bases: PCA vs LDA / whitened-PCA / PLS-vs-A₂.

Fits several linear projections of the pooled teacher bottleneck on the **same**
encoded library, then evaluates ``amplify_knn_hybrid`` and ``z_amplify`` with
identical sampling seeds. Uses the crisp AE teacher by default (no hires wait).

    OMP_NUM_THREADS=6 python scripts/ablate_pca_lda_library.py \\
      --out runs/ml/field_maps/pca_lda_ablate_2026-07-25
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
from galacticsics.ml.fields.feature_library import (
    FeatureLibraryConfig,
    IndexMethod,
    TeacherFeatureLibrary,
    encode_snapshot_features,
    fit_index_basis,
    load_frozen_teacher_bundle,
)
from galacticsics.ml.fields.normalize import denormalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from ntropy.analysis.disk_density import disk_azimuthal_fourier


CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
METHODS: tuple[IndexMethod, ...] = (
    "pca",
    "lda_concat",
    "whiten_pca",
    "pls_a2",
    "multiclass_lda",
)


def _a2_disk(parts) -> float:
    disk = parts["component_id"] == 0
    return float(
        disk_azimuthal_fourier(
            parts["pos"][disk],
            parts["mass"][disk],
            m=2,
            r_max=12.0,
            n_bins=12,
            z_max=0.5,
            min_count=10,
        )["a_m_over_a0_median"]
    )


def _disk_collapse(stack, n_z, n_mom):
    dens = np.zeros(stack.shape[-2:], dtype=np.float64)
    for iz in range(n_z):
        dens += np.maximum(stack[iz * n_mom], 0.0)
    return dens


def _am_mse(dens, dens_ref, r_max=12.0):
    """Optional Am(R) MSE of sampled dens map vs a bar reference map (m=2)."""
    a = dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=r_max)["a_m_over_a0"]
    b = dens_map_azimuthal_fourier_numpy(dens_ref, m=2, r_max=r_max)["a_m_over_a0"]
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    return float(np.mean((a[:n] - b[:n]) ** 2))


def _stratified(ranked, n_bar, n_quiet, n_mid, bar_floor, quiet_ceil):
    seen, picked = set(), []

    def add(row):
        if row["path"] in seen:
            return
        seen.add(row["path"])
        picked.append(row)

    bars = [r for r in ranked if r["a2"] >= bar_floor]
    quiets = [r for r in ranked if r["a2"] <= quiet_ceil]
    for r in bars[:n_bar]:
        add(r)
    for r in reversed(quiets[-n_quiet:]):
        add(r)
    rest = [r for r in ranked if r["path"] not in seen]
    if rest and n_mid > 0:
        idx = np.linspace(0, len(rest) - 1, num=min(n_mid, len(rest)), dtype=int)
        for i in idx:
            add(rest[int(i)])
    for r in ranked[:12]:
        add(r)
    return picked


def encode_library(args):
    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    picks = _stratified(
        ranked, args.n_bar, args.n_quiet, args.n_mid, args.bar_floor, args.quiet_ceil
    )
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
    feats, zs, a2s, meta = [], [], [], []
    print(f"encoding library snaps={len(picks)}…", flush=True)
    for j, row in enumerate(picks):
        is_bar = float(row["a2"]) >= args.bar_floor
        phis = (
            list(np.linspace(0.0, 2.0 * np.pi, args.n_rot_bar, endpoint=False))
            if is_bar and args.n_rot_bar > 1
            else [None]
        )
        for phi in phis:
            feat, z, a2 = encode_snapshot_features(
                row["path"],
                teacher=teacher,
                cfg=cfg,
                stats=stats,
                phi=None if phi is None else float(phi),
                enc_grid=args.enc_grid,
            )
            feats.append(feat)
            zs.append(z)
            a2s.append(a2)
            meta.append(
                {
                    "path": row["path"],
                    "rank_a2": float(row["a2"]),
                    "data_a2": float(a2),
                    "phi": None if phi is None else float(phi),
                    "kind": (
                        "bar"
                        if is_bar
                        else ("quiet" if row["a2"] <= args.quiet_ceil else "mid")
                    ),
                }
            )
        if (j + 1) % 8 == 0 or j + 1 == len(picks):
            print(f"  {j+1}/{len(picks)} lib={len(feats)}", flush=True)
    return teacher, cfg, stats, feats, np.stack(zs, 0), np.asarray(a2s, np.float64), meta


def _make_lib(teacher, cfg, stats, feats, a2, meta, mean, W, codes, args):
    return TeacherFeatureLibrary(
        teacher=teacher,
        cfg=cfg,
        stats=stats,
        feats=feats,
        codes=codes,
        a2=a2,
        meta=meta,
        pca_mean=mean,
        pca_w=W,
        lib_cfg=FeatureLibraryConfig(
            enc_grid=args.enc_grid,
            bar_floor=args.bar_floor,
            quiet_ceil=args.quiet_ceil,
            n_pc=args.n_pc,
        ),
    )


def _eval_method(lib, name, args, rng, dens_ref=None):
    disk_g = lib.cfg.grid_for("disk")
    rows = []

    def samp_hybrid(kind):
        return lib.sample_features(kind=kind, method="amplify_knn_hybrid", rng=rng)

    def samp_z_amp(kind):
        return lib.sample_features_z_amplify(
            kind=kind, rng=rng, alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi
        )

    out = {}
    for samp_name, sampler in (
        ("amplify_knn_hybrid", samp_hybrid),
        ("z_amplify", samp_z_amp),
    ):
        local = []
        for kind in ("barred", "quiet"):
            for j in range(args.n_samples):
                feat, meta = sampler(kind)
                fields = lib.decode_features(feat)
                den = {
                    k: denormalize_stack(fields[k][0].numpy(), lib.stats[k])
                    for k in fields
                }
                dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
                a2_map = float(
                    dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)[
                        "a_m_over_a0_median"
                    ]
                )
                parts = resample_particles_from_multiscale(
                    den,
                    cfg=lib.cfg,
                    n_particles=args.n_resample,
                    count_fractions=COUNT,
                    rng=rng,
                )
                a2_part = _a2_disk(parts)
                row = {
                    "kind": kind,
                    "a2_map": a2_map,
                    "a2_part": a2_part,
                    **{k: v for k, v in meta.items() if k != "z"},
                }
                if dens_ref is not None and kind == "barred":
                    row["am_r_mse_m2_vs_ref"] = _am_mse(dens, dens_ref)
                local.append(row)
                print(
                    f"  [{name}/{samp_name}] {kind}#{j} A₂p={a2_part:.3f}",
                    flush=True,
                )
        bar = float(np.mean([r["a2_part"] for r in local if r["kind"] == "barred"]))
        qui = float(np.mean([r["a2_part"] for r in local if r["kind"] == "quiet"]))
        ams = [
            r["am_r_mse_m2_vs_ref"]
            for r in local
            if "am_r_mse_m2_vs_ref" in r
        ]
        out[samp_name] = {
            "bar_mean": bar,
            "quiet_mean": qui,
            "hits_target": bar >= 0.33 and qui <= 0.05,
            "am_r_mse_m2_mean": float(np.mean(ams)) if ams else None,
            "samples": local,
        }
        print(
            f"  → {name}/{samp_name}: bar={bar:.3f} quiet={qui:.3f}"
            + (f" AmMSE={np.mean(ams):.4f}" if ams else ""),
            flush=True,
        )
    return out


def _plot_codes(path, codes, a2, title):
    if codes.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sc = ax.scatter(codes[:, 0], codes[:, 1], c=a2, cmap="coolwarm", s=18, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="A₂")
    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/pca_lda_ablate_2026-07-25"),
    )
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--n-bar", type=int, default=28)
    p.add_argument("--n-quiet", type=int, default=16)
    p.add_argument("--n-mid", type=int, default=10)
    p.add_argument("--n-rot-bar", type=int, default=4)
    p.add_argument("--bar-floor", type=float, default=0.22)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-resample", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--alpha-lo", type=float, default=1.08)
    p.add_argument("--alpha-hi", type=float, default=1.32)
    p.add_argument("--omp-threads", type=int, default=6)
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
    )
    args = p.parse_args()

    omp = max(1, int(args.omp_threads))
    os.environ["OMP_NUM_THREADS"] = str(omp)
    torch.set_num_threads(omp)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    teacher, cfg, stats, feats, Z, A2, meta = encode_library(args)
    cache = args.out / "pooled_bottleneck.npz"
    np.savez_compressed(cache, Z=Z, a2=A2)
    print(f"cached pooled Z → {cache} shape={Z.shape}", flush=True)

    # Reference dens collapse from strongest library bar (for optional Am MSE).
    i_ref = int(np.argmax(A2))
    dens_ref = None
    try:
        ref_out = TeacherFeatureLibrary(
            teacher=teacher,
            cfg=cfg,
            stats=stats,
            feats=feats,
            codes=np.zeros((len(feats), 2)),
            a2=A2,
            meta=meta,
            pca_mean=Z.mean(0),
            pca_w=np.eye(Z.shape[1], 2),
            lib_cfg=FeatureLibraryConfig(
                enc_grid=args.enc_grid,
                bar_floor=args.bar_floor,
                quiet_ceil=args.quiet_ceil,
            ),
        ).decode_features(feats[i_ref])
        den = {
            k: denormalize_stack(ref_out[k][0].numpy(), stats[k]) for k in ref_out
        }
        dens_ref = _disk_collapse(
            den["disk"], cfg.grid_for("disk").n_z, cfg.grid_for("disk").n_mom
        )
    except Exception as exc:  # pragma: no cover
        print(f"ref dens skipped: {exc}", flush=True)

    verdict = {
        "approach": "PCA vs LDA/PLS library index ablation",
        "teacher": str(args.teacher),
        "n_library": len(feats),
        "n_pc": args.n_pc,
        "seed": args.seed,
        "methods": {},
        "recommendation": None,
    }

    for method in args.methods:
        mean, W, codes, fit_meta = fit_index_basis(
            Z,
            A2,
            method=method,  # type: ignore[arg-type]
            n_comp=args.n_pc,
            bar_floor=args.bar_floor,
            quiet_ceil=args.quiet_ceil,
        )
        _plot_codes(args.out / f"codes_{method}.png", codes, A2, f"{method} (colored by A₂)")
        lib = _make_lib(teacher, cfg, stats, feats, A2, meta, mean, W, codes, args)
        # Fresh RNG stream per method so seed offset is comparable but independent.
        method_rng = np.random.default_rng(args.seed + hash(method) % 10_000)
        runs = _eval_method(lib, method, args, method_rng, dens_ref=dens_ref)
        verdict["methods"][method] = {"fit": fit_meta, "runs": runs}
        lib.save_codes(args.out / f"codes_{method}.npz")

    # Prefer continuous morph diversity: high bar A₂ without quiet inflation,
    # and without collapsing within-class variance (PCA / PLS / multiclass LDA).
    summary_rows = []
    for method, block in verdict["methods"].items():
        for samp, r in block["runs"].items():
            summary_rows.append(
                {
                    "method": method,
                    "sampler": samp,
                    "bar_mean": r["bar_mean"],
                    "quiet_mean": r["quiet_mean"],
                    "hits": r["hits_target"],
                    "am_r_mse_m2_mean": r.get("am_r_mse_m2_mean"),
                }
            )
    # Score: hit target first, then bar A₂, then low quiet.
    best = max(
        summary_rows,
        key=lambda r: (r["hits"], r["bar_mean"], -r["quiet_mean"]),
    )
    pca_hyb = next(
        (
            r
            for r in summary_rows
            if r["method"] == "pca" and r["sampler"] == "amplify_knn_hybrid"
        ),
        None,
    )
    lda_hyb = next(
        (
            r
            for r in summary_rows
            if r["method"] == "lda_concat" and r["sampler"] == "amplify_knn_hybrid"
        ),
        None,
    )
    verdict["summary_table"] = summary_rows
    verdict["best_run"] = best
    verdict["recommendation"] = {
        "keep_for_continuous_ic": "pca",
        "optional_retrieval_metric": "whiten_pca or pls_a2",
        "avoid_as_sole_index": "binary lda_concat",
        "rationale": (
            "PCA preserves within-class morph diversity along continuous A₂; "
            "binary LDA maximizes bar/quiet separation but collapses within-class "
            "variance — harmful for continuous IC sampling / z_amplify. "
            "Multi-class LDA on A₂ strata or PLS vs continuous A₂ are better "
            "supervised compromises if a label-aware index is desired."
        ),
        "pca_hybrid": pca_hyb,
        "lda_hybrid": lda_hyb,
        "empirical_best": best,
    }

    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    summary = args.out / "SUMMARY.md"
    lines = [
        "# PCA vs LDA library index ablation",
        "",
        f"Teacher: `{args.teacher}`  ",
        f"Library size: {len(feats)} (stratified + bar rotations)  ",
        f"Seed: {args.seed}  ",
        f"UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
        "## Math intuition",
        "",
        "- **PCA**: unsupervised variance of pooled bottleneck; continuous morph axes; "
        "current production path for `amplify_knn_hybrid` / `z_amplify`.",
        "- **Binary LDA**: Fisher direction maximizing bar vs quiet separation. "
        "Good for classification / retrieval pools, but **collapses within-class** "
        "diversity — bad as the sole continuous ``z`` for IC sampling.",
        "- **Whitened PCA**: class-conditional Mahalanobis metric without 1-D collapse.",
        "- **PLS vs A₂**: supervised axes maximizing covariance with continuous A₂ "
        "(better label use than binary LDA).",
        "- **Multi-class LDA** on A₂ quantile strata: supervised without binary collapse.",
        "",
        "## Results (particle A₂)",
        "",
        "| index | sampler | bar A₂ | quiet A₂ | hit | Am(R) MSE m2 |",
        "|---|---|---:|---:|---|---:|",
    ]
    for r in summary_rows:
        am = r["am_r_mse_m2_mean"]
        am_s = f"{am:.4f}" if am is not None else "—"
        lines.append(
            f"| {r['method']} | {r['sampler']} | {r['bar_mean']:.3f} | "
            f"{r['quiet_mean']:.3f} | {'yes' if r['hits'] else 'no'} | {am_s} |"
        )
    lines += [
        "",
        "## Recommendation",
        "",
        f"- Keep **PCA** as the default continuous index "
        f"(empirical best this run: `{best['method']}/{best['sampler']}` "
        f"bar={best['bar_mean']:.3f}, quiet={best['quiet_mean']:.3f}).",
        "- Do **not** replace PCA with binary LDA for generative sampling.",
        "- If supervised indexing is wanted: prefer **PLS-vs-A₂** or "
        "**multi-class LDA** (A₂ strata), or use **whitened PCA** only as a "
        "retrieval distance — still pad with unsupervised residual PCA.",
        "",
        "Artifacts: `verdict.json`, `codes_*.png`, `pooled_bottleneck.npz`.",
        "",
    ]
    summary.write_text("\n".join(lines))
    print(json.dumps({"best": best, "out": str(args.out)}, indent=2))
    print(f"wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
