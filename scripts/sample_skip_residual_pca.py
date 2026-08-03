#!/usr/bin/env python3
"""
Skip-residual PCA morph: quiet mean + Σ β_k * PC_k(skip residuals of strong bars).

Keeps frozen AE decode; continuous morph coeffs β control bar strength without
mean-collapsing skip synth.

    OMP_NUM_THREADS=6 python scripts/sample_skip_residual_pca.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
from galacticsics.ml.fields.feature_library import (
    FeatureLibraryConfig,
    TeacherFeatureLibrary,
    blend_features,
    encode_snapshot_features,
    load_frozen_teacher_bundle,
)
from galacticsics.ml.fields.normalize import denormalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from ntropy.analysis.disk_density import disk_azimuthal_fourier


CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
LOG = Path("runs/ml/field_maps/MARATHON_6H.md")
COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}


def _a2_disk(parts):
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


def build_library(args) -> TeacherFeatureLibrary:
    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    picks = _stratified(
        ranked, args.n_bar, args.n_quiet, args.n_mid, args.bar_floor, args.quiet_ceil
    )
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
    lib_cfg = FeatureLibraryConfig(
        enc_grid=args.enc_grid, bar_floor=args.bar_floor, quiet_ceil=args.quiet_ceil
    )
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
            meta.append({"path": row["path"], "rank_a2": float(row["a2"]), "data_a2": float(a2)})
        if (j + 1) % 8 == 0 or j + 1 == len(picks):
            print(f"  {j+1}/{len(picks)} lib={len(feats)}", flush=True)
    Z = np.stack(zs, axis=0)
    A2 = np.asarray(a2s, dtype=np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    n_pc = min(args.n_pc, Z.shape[1], max(Z.shape[0] - 1, 1))
    W = vt[:n_pc].T
    return TeacherFeatureLibrary(
        teacher=teacher,
        cfg=cfg,
        stats=stats,
        feats=feats,
        codes=Zc @ W,
        a2=A2,
        meta=meta,
        pca_mean=Z.mean(0),
        pca_w=W,
        lib_cfg=lib_cfg,
    )


def _flatten_feat(feat: dict) -> np.ndarray:
    parts = []
    for name in sorted(feat.keys()):
        parts.append(feat[name]["bottleneck"].detach().cpu().numpy().ravel())
        for s in feat[name]["skips"]:
            parts.append(s.detach().cpu().numpy().ravel())
    return np.concatenate(parts)


def _unflatten_residual(base: dict, vec: np.ndarray) -> dict:
    """Add residual vector onto base feature dict (same layout as flatten)."""
    out = {}
    offset = 0
    for name in sorted(base.keys()):
        bn = base[name]["bottleneck"]
        n_bn = bn.numel()
        bn_new = bn + torch.as_tensor(
            vec[offset : offset + n_bn], dtype=bn.dtype
        ).reshape_as(bn)
        offset += n_bn
        skips = []
        for s in base[name]["skips"]:
            n_s = s.numel()
            skips.append(
                s
                + torch.as_tensor(vec[offset : offset + n_s], dtype=s.dtype).reshape_as(s)
            )
            offset += n_s
        out[name] = {"bottleneck": bn_new, "skips": tuple(skips)}
    return out


def build_residual_pca(lib: TeacherFeatureLibrary, strong_floor: float = 0.28, n_comp: int = 8):
    base = lib._quiet_mean
    assert base is not None
    base_flat = _flatten_feat(base)
    strong = lib.bar_pool[lib.a2[lib.bar_pool] >= strong_floor]
    if strong.size < 4:
        strong = lib.bar_pool
    # subsample to keep PCA tractable in RAM (residuals are huge)
    if strong.size > 24:
        # pick highest-A2
        order = np.argsort(lib.a2[strong])[::-1][:24]
        strong = strong[order]
    X = []
    for i in strong:
        X.append(_flatten_feat(lib.feats[int(i)]) - base_flat)
    X = np.stack(X, axis=0)
    # PCA via SVD on (n × D) — D can be huge; use randomized on chunked... 
    # For tractability: only use disk tower residual (bars live in disk skips)
    return _disk_only_residual_pca(lib, strong, n_comp=n_comp)


def _disk_only_residual_pca(lib, strong, n_comp=8):
    base = lib._quiet_mean
    # flatten disk only
    def flat_disk(feat):
        d = feat["disk"]
        parts = [d["bottleneck"].detach().cpu().numpy().ravel()]
        for s in d["skips"]:
            parts.append(s.detach().cpu().numpy().ravel())
        return np.concatenate(parts)

    base_f = flat_disk(base)
    X = np.stack([flat_disk(lib.feats[int(i)]) - base_f for i in strong], axis=0)
    Xc = X - X.mean(0, keepdims=True)
    # economy SVD
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    n_c = min(n_comp, vt.shape[0], max(len(strong) - 1, 1))
    P = vt[:n_c]  # (n_c, D)
    scales = s[:n_c] / np.sqrt(max(len(strong), 1))
    return {
        "base": base,
        "disk_mean_resid": X.mean(0),
        "P": P,
        "scales": scales,
        "strong": strong,
        "base_flat_disk": base_f,
    }


def sample_skip_pca(lib, pca, rng, beta_scale=1.15, mix_bulge_halo=True, morph=None):
    """Sample: quiet + disk residual PCA reconstruction; keep BH from a strong bar.

    ``morph`` (optional float ≥0) scales the residual toward strong-bar strength;
    if None, draw morph ~ Uniform[0.8, 1.4] * beta_scale for continuous control.
    """
    n_c = pca["P"].shape[0]
    # Morph knob: continuous bar strength without library membership
    if morph is None:
        morph = float(rng.uniform(0.85, 1.45)) * float(beta_scale)
    else:
        morph = float(morph) * float(beta_scale)
    # Prefer residual along mean strong-bar direction + small PC jitter
    jitter = rng.normal(0.0, 0.35, size=n_c) * pca["scales"]
    # Project mean residual onto PCs for a morphable continuous code
    mean_coeff = pca["P"] @ pca["disk_mean_resid"]
    beta = morph * mean_coeff + jitter
    i_bar = int(rng.choice(pca["strong"]))
    disk_resid = beta @ pca["P"]
    # rebuild disk features
    base = pca["base"]
    d0 = base["disk"]
    offset = 0
    bn = d0["bottleneck"]
    n_bn = bn.numel()
    bn_new = bn + torch.as_tensor(disk_resid[offset : offset + n_bn], dtype=bn.dtype).reshape_as(bn)
    offset += n_bn
    skips = []
    for s in d0["skips"]:
        n_s = s.numel()
        skips.append(
            s + torch.as_tensor(disk_resid[offset : offset + n_s], dtype=s.dtype).reshape_as(s)
        )
        offset += n_s
    feat = {
        "disk": {"bottleneck": bn_new, "skips": tuple(skips)},
        "bulge": base["bulge"],
        "halo": base["halo"],
    }
    if mix_bulge_halo:
        bar = lib.feats[i_bar]
        alpha = float(rng.uniform(0.15, 0.45))
        for name in ("bulge", "halo"):
            feat[name] = {
                "bottleneck": (1 - alpha) * base[name]["bottleneck"] + alpha * bar[name]["bottleneck"],
                "skips": tuple(
                    (1 - alpha) * b + alpha * t
                    for b, t in zip(base[name]["skips"], bar[name]["skips"])
                ),
            }
    return feat, {
        "method": "skip_residual_pca",
        "beta": beta.tolist(),
        "morph": morph,
        "i_bar": i_bar,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ml/field_maps/marathon_skip_pca_2026-07-25"))
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar", type=int, default=28)
    p.add_argument("--n-quiet", type=int, default=16)
    p.add_argument("--n-mid", type=int, default=10)
    p.add_argument("--n-rot-bar", type=int, default=4)
    p.add_argument("--bar-floor", type=float, default=0.22)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-comp", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--beta-scale", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    lib = build_library(args)
    lib.save_codes(args.out / "feature_library_codes.npz")
    print("building disk skip-residual PCA…", flush=True)
    pca = build_residual_pca(lib, strong_floor=0.28, n_comp=args.n_comp)
    np.savez_compressed(
        args.out / "skip_residual_pca.npz",
        P=pca["P"],
        scales=pca["scales"],
        disk_mean_resid=pca["disk_mean_resid"],
        strong=pca["strong"],
    )

    disk_g = lib.cfg.grid_for("disk")
    rows = []
    for kind in ("barred", "quiet"):
        for j in range(args.n_samples):
            if kind == "quiet":
                feat, meta = lib.sample_features(kind="quiet", method="uniform_knn", rng=rng)
            else:
                feat, meta = sample_skip_pca(lib, pca, rng, beta_scale=args.beta_scale)
            out = lib.decode_features(feat)
            den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
            dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
            a2_map = float(
                dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
            )
            parts = resample_particles_from_multiscale(
                den, cfg=lib.cfg, n_particles=args.n_resample, count_fractions=COUNT, rng=rng
            )
            a2_part = _a2_disk(parts)
            rows.append({"kind": kind, "a2_map": a2_map, "a2_part": a2_part, **meta})
            print(f"  skip_pca {kind}#{j} A₂p={a2_part:.3f}", flush=True)

    # also amplify baseline on same lib
    amp_rows = []
    for kind in ("barred", "quiet"):
        for j in range(args.n_samples):
            feat, meta = lib.sample_features(kind=kind, method="amplify_residual", rng=rng)
            out = lib.decode_features(feat)
            den = {k: denormalize_stack(out[k][0].numpy(), lib.stats[k]) for k in out}
            dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
            a2_map = float(
                dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
            )
            parts = resample_particles_from_multiscale(
                den, cfg=lib.cfg, n_particles=args.n_resample, count_fractions=COUNT, rng=rng
            )
            a2_part = _a2_disk(parts)
            amp_rows.append({"kind": kind, "a2_map": a2_map, "a2_part": a2_part, **meta})
            print(f"  amplify {kind}#{j} A₂p={a2_part:.3f}", flush=True)

    def means(rs, k):
        return float(np.mean([r["a2_part"] for r in rs if r["kind"] == k]))

    verdict = {
        "skip_residual_pca": {
            "bar_mean": means(rows, "barred"),
            "quiet_mean": means(rows, "quiet"),
            "samples": rows,
        },
        "amplify_residual": {
            "bar_mean": means(amp_rows, "barred"),
            "quiet_mean": means(amp_rows, "quiet"),
            "samples": amp_rows,
        },
    }
    for name, r in verdict.items():
        r["hits_target"] = r["bar_mean"] >= 0.33 and r["quiet_mean"] <= 0.05
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with LOG.open("a") as f:
        for name, r in verdict.items():
            hits = "YES" if r["hits_target"] else ("near" if r["bar_mean"] >= 0.30 and r["quiet_mean"] <= 0.05 else "no")
            f.write(
                f"| {ts} | {name} (skip_pca script) | {r['bar_mean']:.3f} | {r['quiet_mean']:.3f} | "
                f"{hits} | {args.out.name} |\n"
            )
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in verdict.items()}, indent=2))
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
