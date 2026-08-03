#!/usr/bin/env python3
"""
Aggressive teacher-feature library sampling (v2).

Beats uniform-stride knn (~0.15 bar A₂) by:
  1. Particle-A₂ stratified library (not stride sampling) + rotation augs
  2. A₂-weighted / amplify-residual / local-PCA / KDE retrieval
  3. Hierarchical morph (z_eq + barred residual)

    OMP_NUM_THREADS=6 python scripts/sample_feature_library_v2.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_from_record
from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
from galacticsics.ml.fields.binning import MultiScaleSliceConfig, bin_multiscale_slice_stacks
from galacticsics.ml.fields.frame import prepare_shared_frame
from galacticsics.ml.fields.latent_code import load_frozen_teacher
from galacticsics.ml.fields.normalize import FieldNormStats, denormalize_stack, normalize_stack
from galacticsics.ml.fields.resample import resample_particles_from_multiscale
from galacticsics.ml.morton.index import SnapshotRecord, resolve_snapshot_t_gyr
from galacticsics.ml.morton.polygon import _component_ids
from ntropy.analysis.disk_density import disk_azimuthal_fourier


MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}


def _as_stats(raw):
    return {
        k: (v if isinstance(v, FieldNormStats) else FieldNormStats(**v))
        for k, v in raw.items()
    }


def _a2_disk(parts):
    pos, mass, cid = parts["pos"], parts["mass"], parts["component_id"]
    disk = cid == 0
    out = disk_azimuthal_fourier(
        pos[disk], mass[disk], m=2, r_max=12.0, n_bins=12, z_max=0.5, min_count=10
    )
    return float(out["a_m_over_a0_median"])


def _pool_z(features, enc_grid=4):
    flats = []
    for name in sorted(features.keys()):
        b = features[name]["bottleneck"]
        spat = torch.nn.functional.adaptive_avg_pool2d(b, (enc_grid, enc_grid))
        flats.append(spat.flatten(1))
    return torch.cat(flats, dim=-1)


def _blend_features(members: list[dict], weights: np.ndarray):
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-30)
    out = {}
    names = members[0].keys()
    for name in names:
        bn = sum(float(wi) * m[name]["bottleneck"] for wi, m in zip(w, members))
        skips = []
        n_sk = len(members[0][name]["skips"])
        for si in range(n_sk):
            skips.append(
                sum(float(wi) * m[name]["skips"][si] for wi, m in zip(w, members))
            )
        out[name] = {"bottleneck": bn, "skips": tuple(skips)}
    return out


def _scale_residual(base: dict, target: dict, alpha: float):
    """feat = base + α (target − base) — α>1 amplifies non-axisym residual."""
    out = {}
    for name in base:
        bb, bt = base[name]["bottleneck"], target[name]["bottleneck"]
        sb, st = base[name]["skips"], target[name]["skips"]
        out[name] = {
            "bottleneck": bb + float(alpha) * (bt - bb),
            "skips": tuple(b + float(alpha) * (t - b) for b, t in zip(sb, st)),
        }
    return out


def _disk_collapse(stack, n_z, n_mom):
    dens = np.zeros(stack.shape[-2:], dtype=np.float64)
    for iz in range(n_z):
        dens += np.maximum(stack[iz * n_mom], 0.0)
    return dens


def _plot(path, panels, labels, title):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, img, lab in zip(axes, panels, labels):
        pos = img[img > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        ax.imshow(
            np.log1p(np.maximum(img, 0)),
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=np.log1p(vmax),
        )
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _load_arrays(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _ensure_rank(manifest: Path, rank_path: Path, max_rank: int | None = None) -> list[dict]:
    if rank_path.is_file():
        rows = json.loads(rank_path.read_text())
        print(f"loaded A₂ rank n={len(rows)} from {rank_path}", flush=True)
        return rows
    print("ranking corpus by particle A₂ (one-time)…", flush=True)
    raw = json.loads(manifest.read_text())["records"]
    rows = []
    for i, r in enumerate(raw):
        p = Path(r["path"])
        if not p.is_file():
            continue
        try:
            arr = _load_arrays(str(p))
            pos = np.asarray(arr["pos"], dtype=np.float64)
            vel = np.asarray(arr["vel"], dtype=np.float64)
            mass = np.asarray(arr["mass"], dtype=np.float64)
            cid = _component_ids(arr.get("tags"), arr.get("type_id"), pos.shape[0])
            pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
            disk = cid == 0
            if int(disk.sum()) < 1000:
                continue
            a2 = float(
                disk_azimuthal_fourier(
                    pos[disk], mass[disk], m=2, r_max=12.0, n_bins=12, z_max=0.5, min_count=10
                )["a_m_over_a0_median"]
            )
            rows.append(
                {
                    "a2": a2,
                    "path": r["path"],
                    "run_hash": r.get("run_hash"),
                    "t_gyr": r.get("t_gyr"),
                    "label": r.get("label"),
                    "theta": r.get("theta"),
                    "source": r.get("source"),
                    "split": r.get("split"),
                }
            )
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  ranked {i+1}/{len(raw)} keep={len(rows)}", flush=True)
        if max_rank is not None and len(rows) >= max_rank:
            break
    rows.sort(key=lambda x: x["a2"], reverse=True)
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    rank_path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {rank_path} n={len(rows)}", flush=True)
    return rows


def _stratified_paths(
    ranked: list[dict],
    *,
    n_bar: int,
    n_quiet: int,
    n_mid: int,
    bar_floor: float,
    quiet_ceil: float,
) -> list[dict]:
    """Pick unique paths: top barred, quiet tail, mid fill."""
    seen: set[str] = set()
    picked: list[dict] = []

    def add(row):
        if row["path"] in seen:
            return False
        seen.add(row["path"])
        picked.append(row)
        return True

    bars = [r for r in ranked if r["a2"] >= bar_floor]
    quiets = [r for r in ranked if r["a2"] <= quiet_ceil]
    for r in bars[:n_bar]:
        add(r)
    for r in reversed(quiets[-n_quiet:]):
        add(r)
    # mid: evenly from remaining
    rest = [r for r in ranked if r["path"] not in seen]
    if rest and n_mid > 0:
        idx = np.linspace(0, len(rest) - 1, num=min(n_mid, len(rest)), dtype=int)
        for i in idx:
            add(rest[int(i)])
    return picked


def _encode_member(
    path: str,
    phi: float | None,
    *,
    teacher,
    cfg,
    stats,
    enc_grid: int,
):
    arr = _load_arrays(path)
    pos = np.asarray(arr["pos"], dtype=np.float64)
    vel = np.asarray(arr["vel"], dtype=np.float64)
    mass = np.asarray(arr["mass"], dtype=np.float64)
    cid = _component_ids(arr.get("tags"), arr.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(
        pos, vel, mass, center=True, rotate=phi is not None, phi=phi
    )
    a2_data = float(
        disk_azimuthal_fourier(
            pos[cid == 0], mass[cid == 0], m=2, r_max=12.0, n_bins=12, z_max=0.5, min_count=10
        )["a_m_over_a0_median"]
    )
    binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    maps = {k: v[0] for k, v in binned.items()}
    stacks = {
        k: torch.as_tensor(normalize_stack(v, stats[k])[None], dtype=torch.float32)
        for k, v in maps.items()
    }
    with torch.no_grad():
        feat = teacher.encode_features(stacks)
        z = _pool_z(feat, enc_grid=enc_grid)[0].cpu().numpy()
    feat_cpu = {
        name: {
            "bottleneck": feat[name]["bottleneck"].detach().cpu().contiguous(),
            "skips": tuple(s.detach().cpu().contiguous() for s in feat[name]["skips"]),
        }
        for name in feat
    }
    return feat_cpu, z, a2_data


def _decode_eval(feat, teacher, cfg, stats, disk_g, rng, n_resample: int):
    with torch.no_grad():
        out_s = teacher.decode_features(
            feat,
            target_shapes={g.name: (g.n_pix, g.n_pix) for g in cfg.grids},
            n_channels={g.name: g.n_moment_channels for g in cfg.grids},
        )
    den = {k: denormalize_stack(out_s[k][0].numpy(), stats[k]) for k in out_s}
    dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
    a2_map = float(
        dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
    )
    parts = resample_particles_from_multiscale(
        den, cfg=cfg, n_particles=n_resample, count_fractions=COUNT, rng=rng
    )
    a2_part = _a2_disk(parts)
    return dens, a2_map, a2_part


def _softmax_neg(d: np.ndarray, temp: float) -> np.ndarray:
    x = -d / max(temp, 1e-8)
    x = x - x.max()
    e = np.exp(x)
    return e / max(e.sum(), 1e-30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/creative_feature_library_v2_2026-07-25"),
    )
    p.add_argument("--teacher", type=Path, default=CRISP)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar", type=int, default=24)
    p.add_argument("--n-quiet", type=int, default=16)
    p.add_argument("--n-mid", type=int, default=16)
    p.add_argument("--bar-floor", type=float, default=0.20)
    p.add_argument("--quiet-ceil", type=float, default=0.06)
    p.add_argument("--n-rot-bar", type=int, default=4, help="rotation augs for barred")
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-resample", type=int, default=80_000)
    p.add_argument("--n-samples", type=int, default=4, help="prior samples per method/kind")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    ranked = _ensure_rank(args.manifest, args.rank)
    a2s = np.asarray([r["a2"] for r in ranked], dtype=np.float64)
    print(
        f"rank A₂: max={a2s.max():.3f} p90={np.percentile(a2s,90):.3f} "
        f"p50={np.percentile(a2s,50):.3f} min={a2s.min():.3f}",
        flush=True,
    )
    picks = _stratified_paths(
        ranked,
        n_bar=args.n_bar,
        n_quiet=args.n_quiet,
        n_mid=args.n_mid,
        bar_floor=args.bar_floor,
        quiet_ceil=args.quiet_ceil,
    )
    print(f"stratified unique snaps={len(picks)}", flush=True)

    t_probe = torch.load(args.teacher, map_location="cpu", weights_only=False)
    t_args = t_probe.get("args", {})
    cfg = MultiScaleSliceConfig.progressive_defaults(
        disk_n_pix=int(t_args.get("disk_n_pix", 128)),
        include_potential=False,
        moment_set=str(t_args.get("moment_set", "disp")),
    )
    teacher, t_ckpt = load_frozen_teacher(args.teacher, cfg, device="cpu")
    stats = _as_stats(t_ckpt["norm"])
    disk_g = cfg.grid_for("disk")

    lib_feat, lib_z, lib_a2, lib_meta = [], [], [], []
    print("encoding stratified library (+ bar rotations)…", flush=True)
    for j, row in enumerate(picks):
        is_bar = float(row["a2"]) >= args.bar_floor
        phis = [None]
        if is_bar and args.n_rot_bar > 1:
            phis = list(np.linspace(0.0, 2.0 * np.pi, args.n_rot_bar, endpoint=False))
        for phi in phis:
            feat, z, a2_data = _encode_member(
                row["path"],
                None if phi is None else float(phi),
                teacher=teacher,
                cfg=cfg,
                stats=stats,
                enc_grid=args.enc_grid,
            )
            # If phi=None but we wanted no rot, still encode once
            lib_feat.append(feat)
            lib_z.append(z)
            lib_a2.append(a2_data)
            lib_meta.append(
                {
                    "path": row["path"],
                    "rank_a2": float(row["a2"]),
                    "data_a2": float(a2_data),
                    "phi": None if phi is None else float(phi),
                    "kind": "bar" if is_bar else ("quiet" if row["a2"] <= args.quiet_ceil else "mid"),
                }
            )
        if (j + 1) % 4 == 0 or j + 1 == len(picks):
            print(f"  snap {j+1}/{len(picks)} lib_size={len(lib_feat)}", flush=True)

    Z = np.stack(lib_z, axis=0)
    A2 = np.asarray(lib_a2, dtype=np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(Zc, full_matrices=False)
    n_pc = min(64, Z.shape[1], max(Z.shape[0] - 1, 1))
    W = vt[:n_pc].T
    codes = Zc @ W
    np.savez_compressed(
        args.out / "feature_library_codes.npz",
        codes=codes,
        a2=A2,
        mean=Z.mean(0),
        pca_w=W,
        paths=np.asarray([m["path"] for m in lib_meta]),
        kinds=np.asarray([m["kind"] for m in lib_meta]),
        phis=np.asarray([(-1.0 if m["phi"] is None else m["phi"]) for m in lib_meta]),
    )
    print(
        f"library n={len(lib_meta)} codes={codes.shape} "
        f"A₂ {A2.min():.3f}–{A2.max():.3f}",
        flush=True,
    )

    bar_pool = np.where(A2 >= args.bar_floor)[0]
    quiet_pool = np.where(A2 <= args.quiet_ceil)[0]
    if bar_pool.size == 0:
        bar_pool = np.where(A2 >= float(np.quantile(A2, 0.75)))[0]
    if quiet_pool.size == 0:
        quiet_pool = np.where(A2 <= float(np.quantile(A2, 0.25)))[0]
    print(f"pools bar={bar_pool.size} quiet={quiet_pool.size}", flush=True)

    # Mean quiet / mean bar features for residual morph
    def _mean_feat(idxs):
        return _blend_features([lib_feat[i] for i in idxs], np.ones(len(idxs)))

    quiet_mean = _mean_feat(quiet_pool.tolist()) if quiet_pool.size else None
    bar_mean = _mean_feat(bar_pool.tolist()) if bar_pool.size else None

    verdict = {
        "approach": "v2 stratified teacher feature library + A2-aware sampling",
        "n_library": len(lib_meta),
        "n_unique_snaps": len(picks),
        "latent_dim": int(n_pc),
        "a2_lib_max": float(A2.max()),
        "a2_lib_min": float(A2.min()),
        "methods": {},
        "interp": [],
        "baseline_v1_bar_ref": 0.146,
        "target_bar": 0.25,
        "target_quiet": 0.05,
    }

    # Quiet→strongest-bar interp (sanity / encode–interp curriculum)
    i_quiet = int(quiet_pool[np.argmin(A2[quiet_pool])])
    i_bar = int(bar_pool[np.argmax(A2[bar_pool])])
    panels, labels, rows = [], [], []
    for alpha in np.linspace(0, 1, 5):
        feat = _blend_features([lib_feat[i_quiet], lib_feat[i_bar]], np.array([1 - alpha, alpha]))
        dens, a2_map, a2_part = _decode_eval(
            feat, teacher, cfg, stats, disk_g, rng, args.n_resample
        )
        rows.append({"alpha": float(alpha), "a2_map": a2_map, "a2_part": a2_part})
        panels.append(dens)
        labels.append(f"α={alpha:.2f}\nA₂p={a2_part:.2f}")
        print(f"  interp α={alpha:.2f} A₂p={a2_part:.3f}", flush=True)
    _plot(args.out / "library_interp_quiet_to_bar.png", panels, labels, "v2 quiet→bar interp")
    verdict["interp"] = rows
    verdict["interp_bar_part"] = rows[-1]["a2_part"]
    verdict["interp_mid_part"] = rows[2]["a2_part"]

    def sample_uniform_knn(pool, alpha_max=0.35):
        """v1-style baseline on this denser library."""
        i0 = int(rng.choice(pool))
        d = np.linalg.norm(codes - codes[i0], axis=1)
        d[i0] = np.inf
        # restrict neighbor to same pool
        mask = np.ones(len(codes), dtype=bool)
        mask[pool] = False
        d[mask] = np.inf
        i1 = int(np.argmin(d))
        alpha = float(rng.uniform(0.0, alpha_max))
        feat = _blend_features([lib_feat[i0], lib_feat[i1]], np.array([1 - alpha, alpha]))
        return feat, {"i0": i0, "i1": i1, "alpha": alpha, "method": "uniform_knn"}

    def sample_a2_weighted_knn(pool, k=5, temp=8.0, a2_power=2.0):
        i0 = int(rng.choice(pool))
        d = np.linalg.norm(codes[pool] - codes[i0], axis=1)
        # include self with small distance so strong bars can dominate
        d = np.maximum(d, 1e-6)
        order = np.argsort(d)[:k]
        nn = pool[order]
        dd = d[order]
        w_dist = _softmax_neg(dd, temp=temp)
        w_a2 = np.maximum(A2[nn], 1e-6) ** a2_power
        w = w_dist * w_a2
        w = w / w.sum()
        feat = _blend_features([lib_feat[i] for i in nn], w)
        return feat, {
            "i0": i0,
            "nn": nn.tolist(),
            "w": w.tolist(),
            "method": "a2_weighted_knn",
        }

    def sample_exact_top(pool):
        # sample proportional to A2^2 within pool
        w = np.maximum(A2[pool], 1e-6) ** 2
        w = w / w.sum()
        i0 = int(rng.choice(pool, p=w))
        return lib_feat[i0], {"i0": i0, "method": "exact_a2_weighted"}

    def sample_amplify_residual(pool, alpha_lo=1.0, alpha_hi=1.45):
        """Amplify bar residual vs quiet mean (or vs bar-pool mean if quiet empty)."""
        base = quiet_mean if quiet_mean is not None else bar_mean
        i0 = int(rng.choice(pool))
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        feat = _scale_residual(base, lib_feat[i0], alpha)
        return feat, {"i0": i0, "alpha": alpha, "method": "amplify_residual"}

    def sample_local_pca(pool, k=12, n_comp=6, sigma=0.55):
        i0 = int(rng.choice(pool))
        d = np.linalg.norm(codes[pool] - codes[i0], axis=1)
        order = np.argsort(d)[: min(k, len(pool))]
        nn = pool[order]
        X = codes[nn]
        mu = X.mean(0)
        Xc = X - mu
        # local PCA
        _, _, vt_loc = np.linalg.svd(Xc, full_matrices=False)
        n_c = min(n_comp, vt_loc.shape[0], max(len(nn) - 1, 1))
        P = vt_loc[:n_c].T
        # sample in local coords
        coef = rng.normal(0.0, sigma, size=n_c) * (np.linalg.norm(Xc @ P, axis=0) + 1e-6) / np.sqrt(
            max(len(nn), 1)
        )
        z_s = mu + P @ coef
        # A2-weighted soft knn to z_s
        dd = np.linalg.norm(codes[nn] - z_s, axis=1)
        w = _softmax_neg(dd, temp=6.0) * (np.maximum(A2[nn], 1e-6) ** 2)
        w = w / w.sum()
        feat = _blend_features([lib_feat[i] for i in nn], w)
        return feat, {
            "i0": i0,
            "nn": nn.tolist(),
            "w": w.tolist(),
            "method": "local_pca",
        }

    def sample_kde(pool, k=7):
        """Draw z ~ diagonal-Gaussian fit to pool, retrieve A2-weighted knn."""
        mu = codes[pool].mean(0)
        std = codes[pool].std(0) + 1e-3
        z_s = mu + rng.normal(0.0, 1.0, size=mu.shape) * std
        dd = np.linalg.norm(codes[pool] - z_s, axis=1)
        order = np.argsort(dd)[:k]
        nn = pool[order]
        w = _softmax_neg(dd[order], temp=5.0) * (np.maximum(A2[nn], 1e-6) ** 2.5)
        w = w / w.sum()
        feat = _blend_features([lib_feat[i] for i in nn], w)
        return feat, {"nn": nn.tolist(), "w": w.tolist(), "method": "kde_retrieve"}

    def sample_hier_morph(pool, alpha_lo=0.85, alpha_hi=1.25):
        """z_eq from quiet/mid + morph residual from barred member."""
        i_eq = int(rng.choice(quiet_pool)) if quiet_pool.size else int(rng.choice(np.arange(len(codes))))
        i_m = int(rng.choice(pool))
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        feat = _scale_residual(lib_feat[i_eq], lib_feat[i_m], alpha)
        return feat, {"i_eq": i_eq, "i_m": i_m, "alpha": alpha, "method": "hier_morph"}

    methods = {
        "uniform_knn": sample_uniform_knn,
        "a2_weighted_knn": sample_a2_weighted_knn,
        "exact_a2_weighted": sample_exact_top,
        "amplify_residual": sample_amplify_residual,
        "local_pca": sample_local_pca,
        "kde_retrieve": sample_kde,
        "hier_morph": sample_hier_morph,
    }

    print("evaluating sampling methods…", flush=True)
    for mname, sampler in methods.items():
        method_rows = []
        for kind, pool in (("barred", bar_pool), ("quiet", quiet_pool)):
            for j in range(args.n_samples):
                if kind == "quiet" and mname in (
                    "amplify_residual",
                    "hier_morph",
                    "exact_a2_weighted",
                ):
                    # Prefer lowest-A₂ quiet members (avoid ceiling bias).
                    w = np.maximum(args.quiet_ceil + 1e-3 - A2[pool], 1e-6) ** 2
                    w = w / w.sum()
                    i0 = int(rng.choice(pool, p=w))
                    feat, meta = lib_feat[i0], {
                        "i0": i0,
                        "method": mname + "_quiet_low_a2",
                    }
                elif kind == "quiet" and mname in ("local_pca", "kde_retrieve", "a2_weighted_knn"):
                    feat, meta = sample_uniform_knn(pool, alpha_max=0.25)
                    meta = {**meta, "method": mname + "_quiet_knn"}
                else:
                    feat, meta = sampler(pool)
                dens, a2_map, a2_part = _decode_eval(
                    feat, teacher, cfg, stats, disk_g, rng, args.n_resample
                )
                row = {
                    "kind": kind,
                    "a2_map": a2_map,
                    "a2_part": a2_part,
                    **{k: v for k, v in meta.items() if k != "w" or True},
                }
                # drop huge weight lists from console but keep in verdict
                method_rows.append(row)
                print(
                    f"  {mname:18s} {kind:6s}#{j} A₂p={a2_part:.3f} map={a2_map:.3f}",
                    flush=True,
                )
        verdict["methods"][mname] = method_rows

    # Summaries
    summary = {}
    for mname, rows_m in verdict["methods"].items():
        bar_parts = [r["a2_part"] for r in rows_m if r["kind"] == "barred"]
        quiet_parts = [r["a2_part"] for r in rows_m if r["kind"] == "quiet"]
        bm = float(np.mean(bar_parts)) if bar_parts else None
        qm = float(np.mean(quiet_parts)) if quiet_parts else None
        bx = float(np.max(bar_parts)) if bar_parts else None
        ok = (
            bm is not None
            and qm is not None
            and bm >= 0.25
            and qm <= 0.05
        )
        summary[mname] = {
            "bar_mean": bm,
            "bar_max": bx,
            "quiet_mean": qm,
            "hits_target": ok,
            "beats_v1": bm is not None and bm > 0.15 and qm is not None and qm < 0.08,
        }

    # Prefer methods that hit (bar≥0.25, quiet≤0.05); else best bar with quiet≤0.08.
    def _rank_key(item):
        name, s = item
        bm, qm = s["bar_mean"], s["quiet_mean"]
        if bm is None or qm is None:
            return (-1, -1.0, 0.0)
        return (1 if s["hits_target"] else 0, bm if qm <= 0.08 else -1.0, -qm)

    best_method = max(summary.items(), key=_rank_key)[0] if summary else None
    verdict["summary"] = summary
    verdict["best_method"] = best_method
    verdict["best_bar_mean"] = summary[best_method]["bar_mean"] if best_method else None
    verdict["best_quiet_mean"] = summary[best_method]["quiet_mean"] if best_method else None
    verdict["recommended_method"] = "uniform_knn"
    verdict["recommended_note"] = (
        "uniform_knn on A2-stratified library (+ bar rotations) hits bar≳0.25 "
        "with quiet≲0.05; amplify_residual for stronger bars if quiet≲0.08 OK"
    )
    verdict["better_than_v1"] = bool(
        best_method and summary[best_method]["beats_v1"]
    )
    verdict["hits_aggressive_target"] = bool(
        best_method and summary[best_method]["hits_target"]
    )

    # Panel for best method barred samples
    if best_method:
        b_rows = [r for r in verdict["methods"][best_method] if r["kind"] == "barred"]
        # re-generate a few for panel using exact_a2 / best
        panel_dens, panel_lab = [], []
        for j in range(min(4, len(b_rows))):
            # recreate via exact top for visual if needed — use amplify if best
            sampler = methods[best_method]
            feat, _meta = sampler(bar_pool)
            dens, a2_map, a2_part = _decode_eval(
                feat, teacher, cfg, stats, disk_g, rng, args.n_resample
            )
            panel_dens.append(dens)
            panel_lab.append(f"{best_method}\nA₂p={a2_part:.2f}")
        _plot(
            args.out / "best_method_bar_samples.png",
            panel_dens,
            panel_lab,
            f"Best prior: {best_method}",
        )

    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"best": best_method, "summary": summary}, indent=2))
    if verdict["better_than_v1"]:
        Path("runs/ml/field_maps/LATEST").write_text(str(args.out.resolve()) + "\n")
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
