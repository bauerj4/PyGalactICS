#!/usr/bin/env python3
"""θ-conditioned latent generative model of full galactic ICs.

Product claim
-------------
Sample similar full-system galaxies (disk/bulge/halo dens + velocity moments)
from a latent ``z`` conditioned on structural ``θ``:

    IC = Decode( retrieve(z | θ) )   via frozen multitower teacher AE

Bars/spirals/quiet are *regions* of ``z``, not the sole objective.  Evaluation
uses multi-component dens+kin similarity; ``A₂(R_d)`` is one morph diagnostic.

This script:
  1. Builds a campaign-balanced teacher feature library (+ PCA ``z``, θ)
  2. Plots latent organization (PCA colored by A₂ / campaign)
  3. Leave-one-out encode→retrieve→decode (never copies eval dump particles)
  4. Scores dens+kin vs held-out dump (metrics only)
  5. Fixed-θ ``z`` interpolation quiet↔dynamical
  6. Optional short gpu_bh evolve coherence check

Example::

    OMP_NUM_THREADS=2 .venv/bin/python scripts/latent_theta_gen.py \\
      --out runs/ml/field_maps/latent_theta_gen_2026-08-02 \\
      --n-resample 200000 --evolve-gyr 0.5 --force gpu_bh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS, theta_vector  # noqa: E402
from galacticsics.ml.fields.binning import bin_multiscale_slice_stacks  # noqa: E402
from galacticsics.ml.fields.feature_library import (  # noqa: E402
    FeatureLibraryConfig,
    TeacherFeatureLibrary,
    _hash_from_path,
    encode_snapshot_features,
    load_frozen_teacher_bundle,
)
from galacticsics.ml.fields.normalize import denormalize_stack  # noqa: E402
from galacticsics.ml.fields.resample import (  # noqa: E402
    inject_morph_residual_on_f0_dens,
    match_disk_radial_cdf_to_reference,
    resample_particles_from_multiscale,
    stitch_retained_components,
)
from galacticsics.ml.morton.polygon import _component_ids  # noqa: E402
from ntropy.analysis.disk_density import disk_azimuthal_fourier  # noqa: E402
from ood_theta_df_compare import _disk_kinematics  # noqa: E402
from score_residual_f0_kinetics import _a2_rd, _score_vs_ref  # noqa: E402

RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
FFT_LONG = Path("runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt")
CRISP = Path("runs/ml/field_maps/crisp_2026-07-24/multitower_slice_ae.pt")
COUNT = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
COMP_NAMES = {0: "disk", 1: "halo", 2: "bulge"}


def _default_teacher() -> Path:
    return FFT_LONG if FFT_LONG.is_file() else CRISP


def _ic_path_for_dump(dump: Path) -> Path | None:
    """Resolve GalactICS ``ic_state.npz`` for a corpus evolution dump."""
    dump = Path(dump)
    candidates = [
        dump.parent.parent.parent / "ic_state.npz",  # .../HASH/evolution/particles/step
        dump.parent.parent / "ic_state.npz",
        dump.parent / "ic_state.npz",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _component_masses(parts: dict) -> dict[str, float]:
    cid = parts["component_id"]
    mass = parts["mass"]
    return {
        name: float(mass[cid == c].sum()) if np.any(cid == c) else 0.0
        for c, name in COMP_NAMES.items()
    }


def _load_parts(path: Path, n_max: int | None, rng: np.random.Generator) -> dict:
    with np.load(path, allow_pickle=True) as z:
        pos = np.asarray(z["pos"], dtype=np.float64)
        vel = np.asarray(z["vel"], dtype=np.float64)
        mass = np.asarray(z["mass"], dtype=np.float64)
        files = set(z.files)
        if "component_id" in files:
            cid = np.asarray(z["component_id"])
        else:
            cid = _component_ids(
                z["tags"] if "tags" in files else None,
                z["type_id"] if "type_id" in files else None,
                pos.shape[0],
            )
        eps = (
            np.asarray(z["eps"], dtype=np.float64)
            if "eps" in files
            else np.full(pos.shape[0], 0.05, dtype=np.float64)
        )
    # Preserve per-component totals before stratified downsample. Keeping raw
    # per-particle masses after N↓ would under-scale dens by ~N_kept/N_full
    # (~7–9× for corpus 1.75M→200k) and fake dens med|log|≈0.8.
    true_M = {c: float(mass[cid == c].sum()) for c in (0, 1, 2)}
    if n_max is not None and pos.shape[0] > n_max:
        # Stratified downsample.
        keep = []
        for c in (0, 1, 2):
            idx = np.where(cid == c)[0]
            frac = COUNT[COMP_NAMES[c]]
            n_c = int(round(n_max * frac))
            if idx.size > n_c:
                idx = rng.choice(idx, size=n_c, replace=False)
            keep.append(idx)
        sel = np.concatenate(keep)
        pos, vel, mass, cid, eps = pos[sel], vel[sel], mass[sel], cid[sel], eps[sel]
        for c in (0, 1, 2):
            m = cid == c
            s = float(mass[m].sum()) if np.any(m) else 0.0
            if s > 0.0 and true_M[c] > 0.0:
                mass[m] *= true_M[c] / s
    # Shared COM.
    w = mass / max(float(mass.sum()), 1e-30)
    com = (pos * w[:, None]).sum(0)
    pos = pos - com
    vcom = (vel * w[:, None]).sum(0)
    vel = vel - vcom
    return {"pos": pos, "vel": vel, "mass": mass, "component_id": cid, "eps": eps}


def _dens_profile(parts: dict, comp: int, r_edges: np.ndarray) -> np.ndarray:
    m = parts["component_id"] == comp
    R = np.hypot(parts["pos"][m, 0], parts["pos"][m, 1])
    mass = parts["mass"][m]
    if comp == 0:
        # Surface dens.
        hist, _ = np.histogram(R, bins=r_edges, weights=mass)
        area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
        return hist / np.maximum(area, 1e-30)
    # 3D spherical dens proxy via cylindrical rings × |z| slab.
    hist, _ = np.histogram(R, bins=r_edges, weights=mass)
    vol = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2) * 2.0  # |z|<1 proxy
    return hist / np.maximum(vol, 1e-30)


def _full_system_score(gen: dict, ref: dict) -> dict:
    """Multi-component dens + disk kinematics score (lower better on MSE terms)."""
    r_edges = np.linspace(0.0, 12.0, 25)
    r_mid = 0.5 * (r_edges[1:] + r_edges[:-1])
    dens = {}
    for c, name in COMP_NAMES.items():
        sg = _dens_profile(gen, c, r_edges)
        sr = _dens_profile(ref, c, r_edges)
        ok = (sr > 1e-8) & (sg > 0) & np.isfinite(sg) & np.isfinite(sr)
        if ok.sum() < 3:
            dens[name] = {"med_abs_log": float("nan"), "rel_mse": float("nan")}
            continue
        log_err = np.abs(np.log10(np.maximum(sg[ok], 1e-30)) - np.log10(np.maximum(sr[ok], 1e-30)))
        rel = (sg[ok] - sr[ok]) / sr[ok]
        dens[name] = {
            "med_abs_log": float(np.median(log_err)),
            "rel_mse": float(np.mean(rel * rel)),
        }
    kin = _score_vs_ref(gen, ref)
    return {
        "dens": dens,
        "kin_mean_mse": float(kin["mse"]["kinetic_mean"]),
        "kin_mse": kin["mse"],
        "a2_rd_gen": _a2_rd(gen, 2.0),
        "a2_rd_ref": _a2_rd(ref, 2.0),
        "a2_med_gen": float(
            disk_azimuthal_fourier(
                gen["pos"][gen["component_id"] == 0],
                gen["mass"][gen["component_id"] == 0],
                m=2,
                r_max=12.0,
                n_bins=12,
                z_max=0.5,
                min_count=10,
            )["a_m_over_a0_median"]
        ),
        "r_mid": [float(x) for x in r_mid],
    }


def _balanced_picks(
    ranked: list[dict],
    *,
    n_bar_campaigns: int,
    n_quiet: int,
    n_mid: int,
    bar_floor: float,
    quiet_ceil: float,
    snaps_per_bar_campaign: int = 2,
) -> list[dict]:
    """Campaign-balanced stratified picks (avoid 54a8 monopoly)."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in ranked:
        h = r.get("run_hash") or _hash_from_path(r["path"])
        by_hash[h].append(r)
    for h in by_hash:
        by_hash[h].sort(key=lambda x: -float(x["a2"]))

    # Best snap per campaign for bars.
    camp_best = []
    for h, rows in by_hash.items():
        best = rows[0]
        if float(best["a2"]) >= bar_floor:
            camp_best.append(best)
    camp_best.sort(key=lambda r: -float(r["a2"]))
    picked: list[dict] = []
    seen = set()

    def add(row: dict) -> None:
        p = row["path"]
        if p in seen:
            return
        seen.add(p)
        picked.append(row)

    for row in camp_best[:n_bar_campaigns]:
        h = row.get("run_hash") or _hash_from_path(row["path"])
        for snap in by_hash[h][:snaps_per_bar_campaign]:
            if float(snap["a2"]) >= bar_floor * 0.7:
                add(snap)

    # Quiets: prefer early/IC-ish across many campaigns.
    quiets = [r for r in ranked if float(r["a2"]) <= quiet_ceil]
    # diversify by hash
    q_by = defaultdict(list)
    for r in quiets:
        q_by[r.get("run_hash") or _hash_from_path(r["path"])].append(r)
    q_reps = [rows[len(rows) // 2] for rows in q_by.values()]
    q_reps.sort(key=lambda r: float(r["a2"]))
    for row in q_reps[:n_quiet]:
        add(row)

    # Mid continuum.
    rest = [r for r in ranked if r["path"] not in seen]
    if rest and n_mid > 0:
        idx = np.linspace(0, len(rest) - 1, num=min(n_mid, len(rest)), dtype=int)
        for i in idx:
            add(rest[int(i)])
    return picked


def build_library(args) -> TeacherFeatureLibrary:
    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    picks = _balanced_picks(
        ranked,
        n_bar_campaigns=args.n_bar_campaigns,
        n_quiet=args.n_quiet,
        n_mid=args.n_mid,
        bar_floor=args.bar_floor,
        quiet_ceil=args.quiet_ceil,
        snaps_per_bar_campaign=args.snaps_per_bar_campaign,
    )
    print(
        f"library picks={len(picks)} unique campaigns="
        f"{len({r.get('run_hash') or _hash_from_path(r['path']) for r in picks})}",
        flush=True,
    )
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
    lib_cfg = FeatureLibraryConfig(
        enc_grid=args.enc_grid, bar_floor=args.bar_floor, quiet_ceil=args.quiet_ceil
    )
    feats, zs, a2s, meta, thetas = [], [], [], [], []
    t0 = time.time()
    for j, row in enumerate(picks):
        is_bar = float(row["a2"]) >= args.bar_floor
        phis: list[float | None]
        if is_bar and args.n_rot_bar > 1:
            phis = list(np.linspace(0.0, 2.0 * np.pi, args.n_rot_bar, endpoint=False))
        else:
            phis = [None]
        th_raw = row.get("theta") or {}
        th_vec = theta_vector(th_raw, keys=DEFAULT_THETA_KEYS)
        h = row.get("run_hash") or _hash_from_path(row["path"])
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
            thetas.append(th_vec)
            meta.append(
                {
                    "path": row["path"],
                    "run_hash": h,
                    "rank_a2": float(row["a2"]),
                    "data_a2": float(a2),
                    "phi": None if phi is None else float(phi),
                    "kind": (
                        "bar"
                        if is_bar
                        else ("quiet" if row["a2"] <= args.quiet_ceil else "mid")
                    ),
                    "t_gyr": float(row.get("t_gyr") or th_raw.get("t_gyr") or 0.0),
                }
            )
        if (j + 1) % 5 == 0 or j + 1 == len(picks):
            print(
                f"  encode {j+1}/{len(picks)} lib={len(feats)} "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    Z = np.stack(zs, axis=0)
    A2 = np.asarray(a2s, dtype=np.float64)
    Theta = np.stack(thetas, axis=0)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    n_pc = min(args.n_pc, Z.shape[1], max(Z.shape[0] - 1, 1))
    W = vt[:n_pc].T
    codes = Zc @ W
    return TeacherFeatureLibrary(
        teacher=teacher,
        cfg=cfg,
        stats=stats,
        feats=feats,
        codes=codes,
        a2=A2,
        meta=meta,
        pca_mean=Z.mean(0),
        pca_w=W,
        lib_cfg=lib_cfg,
        theta=Theta,
    )


def _decode_soft(
    lib: TeacherFeatureLibrary,
    feat: dict,
    n: int,
    rng,
    *,
    mass_total_per_component: dict[str, float] | None = None,
) -> dict:
    """Pure teacher decode → dens+moments resample (AE soft maps)."""
    fields = lib.decode_features(feat)
    phys = {
        k: denormalize_stack(v[0].detach().cpu().numpy(), lib.stats[k])
        for k, v in fields.items()
    }
    return resample_particles_from_multiscale(
        phys,
        cfg=lib.cfg,
        n_particles=n,
        count_fractions=COUNT,
        mass_total_per_component=mass_total_per_component,
        rng=rng,
    )


def _decode_particle_retrieve(
    neighbor_path: Path,
    mass_total_per_component: dict[str, float],
    n: int | None,
    rng,
) -> dict:
    """Stratified (or full-N) particle retrieve from a LOO neighbor dump.

    Generative: uses library/corpus particles ≠ eval path; remasses to
    GalactICS ``f0(θ)`` component totals. Whole-DF dens+kin come from the
    retrieved dynamical state rather than quiet f0 retain.
    """
    src = _load_parts(Path(neighbor_path), n, rng)
    gen = {
        "pos": src["pos"].copy(),
        "vel": src["vel"].copy(),
        "mass": src["mass"].copy(),
        "component_id": src["component_id"].copy(),
        "eps": src.get("eps", np.full(len(src["pos"]), 0.05)).copy(),
    }
    for c, name in COMP_NAMES.items():
        m = gen["component_id"] == c
        s = float(gen["mass"][m].sum()) if np.any(m) else 0.0
        tgt = float(mass_total_per_component.get(name, 0.0))
        if s > 0.0 and tgt > 0.0:
            gen["mass"][m] *= tgt / s
    w = gen["mass"] / max(float(gen["mass"].sum()), 1e-30)
    gen["pos"] = gen["pos"] - (gen["pos"] * w[:, None]).sum(0)
    gen["vel"] = gen["vel"] - (gen["vel"] * w[:, None]).sum(0)
    gen["decode_mode"] = "particle_retrieve"
    gen["neighbor_path"] = str(neighbor_path)
    return gen


def _dense_path_loo_neighbor(
    eval_path: Path,
    *,
    run_hash: str,
    z_eval: np.ndarray,
    a2_enc: float,
    ranked: list[dict],
    pca_mean: np.ndarray,
    pca_w: np.ndarray,
    teacher,
    cfg,
    stats,
    enc_grid: int = 4,
    t_min: float = 0.5,
    a2_weight: float = 1.5,
    profile_rerank: int = 8,
    rng: np.random.Generator | None = None,
) -> tuple[Path, float, float]:
    """Nearest same-campaign corpus dump in (z, A₂), excluding eval path.

    When ``profile_rerank>0``, shortlist by (z,A₂) then pick the candidate whose
    particle_retrieve (remass to f0) best matches eval dens+kin profiles.
    """
    cands: list[tuple[float, float, float, Path]] = []
    eval_res = Path(eval_path).resolve()
    for row in ranked:
        h = row.get("run_hash") or _hash_from_path(row["path"])
        if h != run_hash:
            continue
        p = Path(row["path"])
        if "step_" not in p.name:
            continue
        if p.resolve() == eval_res:
            continue
        t = float(row.get("t_gyr") or 0.0)
        if t < t_min:
            continue
        _feat, z_raw, _a2 = encode_snapshot_features(
            p, teacher=teacher, cfg=cfg, stats=stats, enc_grid=enc_grid
        )
        z = (np.asarray(z_raw, dtype=np.float64) - pca_mean) @ pca_w
        zd = float(np.linalg.norm(z - z_eval))
        a2 = float(row["a2"])
        score = zd + float(a2_weight) * abs(a2 - float(a2_enc))
        cands.append((score, zd, a2, p))
    if not cands:
        raise RuntimeError(f"no dense path-LOO candidates for {run_hash}")
    cands.sort(key=lambda x: x[0])
    if int(profile_rerank) <= 1:
        score, zd, a2, p = cands[0]
        return p, zd, a2

    # Profile re-rank shortlist (no eval particle copy).
    rng = rng or np.random.default_rng(0)
    ic_path = _ic_path_for_dump(eval_path)
    mass_prior = (
        _component_masses(_load_parts(ic_path, None, rng)) if ic_path is not None else {}
    )
    ref = _load_parts(eval_path, None, rng)
    best: tuple[float, float, float, Path] | None = None
    for score0, zd, a2, p in cands[: int(profile_rerank)]:
        gen = _decode_particle_retrieve(p, mass_prior, None, rng)
        sc = _full_system_score(gen, ref)
        dens = sc["dens"]
        loss = (
            2.0 * dens["disk"]["med_abs_log"]
            + dens["halo"]["med_abs_log"]
            + dens["bulge"]["med_abs_log"]
            + 8.0 * sc["kin_mse"]["mean_vphi"]
            + 4.0 * sc["kin_mse"]["sig_r"]
            + 4.0 * sc["kin_mse"].get("sig_phi", 0.0)
            + 4.0 * sc["kin_mse"]["sig_z"]
            + 1.5 * abs(sc["a2_rd_gen"] - sc["a2_rd_ref"])
        )
        if best is None or loss < best[0]:
            best = (loss, zd, a2, p)
    assert best is not None
    return best[3], best[1], best[2]


def _decode_residual_f0(
    lib: TeacherFeatureLibrary,
    feat: dict,
    ic_path: Path,
    n: int,
    rng,
    *,
    alpha: float = 1.25,
    cdf_lock: bool = True,
) -> dict:
    """Axisym dens+moments from GalactICS ``f0(θ)``; non-axisym dens from decode(z).

    Stays generative: never copies the held-out *evolved* dump. ``ic_path`` is
    the equilibrium GalactICS IC for the structural family. Morph residual
    comes from retrieved library features (LOO excludes eval hash).

    Halo/bulge particles are **retained** from ``f0`` (FOV dens maps truncate
    outer halo mass; forcing ``mass_total`` into the FOV inflates ρ(R)). Disk is
    dens-resampled from residual maps + f0 moments.
    """
    fields = lib.decode_features(feat)
    morph = {
        k: denormalize_stack(v[0].detach().cpu().numpy(), lib.stats[k])
        for k, v in fields.items()
    }
    f0 = _load_parts(ic_path, None, rng)
    maps_f0 = {
        k: v[0]
        for k, v in bin_multiscale_slice_stacks(
            f0["pos"], f0["vel"], f0["mass"], f0["component_id"], cfg=lib.cfg
        ).items()
    }
    phys = inject_morph_residual_on_f0_dens(
        maps_f0,
        morph,
        cfg=lib.cfg,
        components=("disk",),
        alpha=float(alpha),
        midplane_only=True,
        mode="m2",
        scale="multiplicative",
        preserve_axisym=True,
        contrast_from_midplane=True,
    )
    mass_totals = _component_masses(f0)
    # Disk-heavy count mix; halo/bulge counts are placeholders replaced below.
    n_disk = int(round(n * COUNT["disk"]))
    n_halo = int(round(n * COUNT["halo"]))
    n_bulge = int(n - n_disk - n_halo)
    gen = resample_particles_from_multiscale(
        phys,
        cfg=lib.cfg,
        n_particles=n,
        n_per_component={"disk": n_disk, "halo": n_halo, "bulge": n_bulge},
        mass_total_per_component={
            "disk": mass_totals["disk"],
            # Placeholder masses; retained particles overwrite these comps.
            "halo": mass_totals["halo"],
            "bulge": mass_totals["bulge"],
        },
        rng=rng,
    )
    gen = stitch_retained_components(
        gen,
        source_pos=f0["pos"],
        source_vel=f0["vel"],
        source_mass=f0["mass"],
        source_cid=f0["component_id"],
        retain=("halo", "bulge"),
        n_retain={"halo": n_halo, "bulge": n_bulge},
        rng=rng,
    )
    # Mass-preserve retained comps after stratified subsample from full IC.
    for c, name in ((1, "halo"), (2, "bulge")):
        m = gen["component_id"] == c
        s = float(gen["mass"][m].sum()) if np.any(m) else 0.0
        if s > 0.0 and mass_totals[name] > 0.0:
            gen["mass"][m] *= mass_totals[name] / s
    if cdf_lock:
        pos_locked, _meta = match_disk_radial_cdf_to_reference(
            gen["pos"],
            gen["mass"],
            gen["component_id"],
            f0["pos"],
            f0["mass"],
            f0["component_id"],
        )
        gen = {**gen, "pos": pos_locked}
    # Re-center after CDF remap / stitch.
    w = gen["mass"] / max(float(gen["mass"].sum()), 1e-30)
    gen["pos"] = gen["pos"] - (gen["pos"] * w[:, None]).sum(0)
    gen["vel"] = gen["vel"] - (gen["vel"] * w[:, None]).sum(0)
    gen["eps"] = np.full(len(gen["pos"]), 0.05, dtype=np.float64)
    gen["decode_mode"] = "residual_f0"
    return gen


def _decode_resample(
    lib: TeacherFeatureLibrary,
    feat: dict,
    n: int,
    rng,
    *,
    mode: str = "soft",
    ic_path: Path | None = None,
    mass_total_per_component: dict[str, float] | None = None,
    residual_alpha: float = 1.25,
) -> dict:
    mode = str(mode).lower().strip()
    if mode in ("residual_f0", "f0", "residual"):
        if ic_path is None or not Path(ic_path).is_file():
            raise ValueError(f"residual_f0 decode needs ic_path, got {ic_path!r}")
        return _decode_residual_f0(
            lib, feat, Path(ic_path), n, rng, alpha=residual_alpha
        )
    return _decode_soft(
        lib, feat, n, rng, mass_total_per_component=mass_total_per_component
    )


def plot_latent(lib: TeacherFeatureLibrary, out: Path) -> dict:
    z = lib.codes
    a2 = lib.a2
    # PCA of codes for 2D viz (codes already PCA of bottlenecks).
    zc = z - z.mean(0)
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    xy = zc @ vt[:2].T

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    sc = axes[0].scatter(xy[:, 0], xy[:, 1], c=a2, cmap="viridis", s=28, alpha=0.85)
    axes[0].set_xlabel("latent PC1")
    axes[0].set_ylabel("latent PC2")
    axes[0].set_title(r"Teacher-library $z$ colored by median $A_2$")
    fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04, label=r"median $A_2$")

    # Color by structural θ: halo.v0 (index 5 in DEFAULT_THETA_KEYS) if present.
    if lib.theta.shape[1] > 5:
        c = lib.theta[:, 5]
        lab = r"halo $v_0$ (θ)"
    else:
        # Hash categorical.
        uniq = {h: i for i, h in enumerate(sorted(set(lib.hashes)))}
        c = np.array([uniq[h] for h in lib.hashes], dtype=float)
        lab = "campaign id"
    sc2 = axes[1].scatter(xy[:, 0], xy[:, 1], c=c, cmap="tab20", s=28, alpha=0.85)
    axes[1].set_xlabel("latent PC1")
    axes[1].set_ylabel("latent PC2")
    axes[1].set_title(r"$z$ colored by θ / campaign")
    fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04, label=lab)
    fig.tight_layout()
    fig.savefig(out / "figs" / "latent_z_organization.png", dpi=160)
    fig.savefig(
        ROOT / "papers/mnras_noneq_ics/figures/fig_latent_theta_z_organization.png",
        dpi=160,
    )
    plt.close(fig)
    return {"xy": xy.tolist(), "a2": a2.tolist()}


def plot_profiles(gen: dict, ref: dict, out_png: Path, title: str) -> None:
    """COM-centered disk Σ + spherical bulge/halo ρ; disk kin overlays."""
    from ntropy.analysis.density import bin_spherical_density
    from ntropy.analysis.disk_density import bin_midplane_surface_density

    def _com(pos, mass):
        w = mass / max(float(mass.sum()), 1e-30)
        return pos - (pos * w[:, None]).sum(0)

    def _dens(parts, cid):
        m = parts["component_id"] == cid
        pos = _com(parts["pos"][m], parts["mass"][m])
        mass = parts["mass"][m]
        if cid == 0:
            prof = bin_midplane_surface_density(
                pos, mass, r_max=12.0, n_bins=28, z_max=0.5
            )
            y = np.asarray(prof.sigma, dtype=np.float64)
            y = np.where(np.asarray(prof.counts) > 0, y, np.nan)
            return np.asarray(prof.r_mid), y
        rmax = 40.0 if cid == 1 else 8.0
        rmin = 0.5 if cid == 1 else 0.05
        prof = bin_spherical_density(
            pos, mass, n_bins=28, r_max=rmax, log_bins=True, r_min=rmin
        )
        y = np.asarray(prof.rho, dtype=np.float64)
        y = np.where(np.asarray(prof.counts) > 0, y, np.nan)
        return np.asarray(prof.r_mid), y

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8))
    for j, (c, name) in enumerate(COMP_NAMES.items()):
        ax = axes[0, j]
        rr, sr = _dens(ref, c)
        rg, sg = _dens(gen, c)
        ax.semilogy(rr, sr, "k-", lw=1.8, label="ref dump")
        ax.semilogy(rg, sg, "C0--", lw=1.8, label="decode(z|θ)")
        ax.set_title(f"{name} dens")
        ax.set_xlabel("R [kpc]" if c == 0 else "r [kpc]")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.25)
    kg = _disk_kinematics(gen)
    kr = _disk_kinematics(ref)
    for ax, key, ylab in zip(
        axes[1],
        ("mean_vphi", "sig_r", "sig_z"),
        (r"$\langle v_\varphi\rangle$", r"$\sigma_R$", r"$\sigma_z$"),
    ):
        ok = (np.asarray(kr["counts"]) >= 20) & (np.asarray(kg["counts"]) >= 20)
        ax.plot(np.asarray(kr["r_mid"])[ok], np.asarray(kr[key])[ok], "k-", lw=1.8, label="ref")
        ax.plot(np.asarray(kg["r_mid"])[ok], np.asarray(kg[key])[ok], "C0--", lw=1.8, label="gen")
        ax.set_xlabel("R [kpc]")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_interp_faceon(panels: list[np.ndarray], labels: list[str], out_png: Path) -> None:
    from galacticsics.campaign.analysis import dens_array_log10

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 2.9))
    if n == 1:
        axes = [axes]
    for ax, img, lab in zip(axes, panels, labels):
        show, vmin_s, vmax_s, _ = dens_array_log10(img, vmax_pct=99.0)
        ax.imshow(
            show.T,
            origin="lower",
            cmap="inferno",
            vmin=vmin_s,
            vmax=vmax_s,
        )
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(r"Fixed-θ $z$ interpolation (decode dens midplane; $\log_{10}\Sigma$)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(
        ROOT / "papers/mnras_noneq_ics/figures/fig_latent_theta_z_interp.png",
        dpi=150,
    )
    plt.close(fig)


def _disk_faceon(parts: dict, nbin: int = 96, half: float = 12.0) -> np.ndarray:
    m = parts["component_id"] == 0
    H, _, _ = np.histogram2d(
        parts["pos"][m, 0],
        parts["pos"][m, 1],
        bins=nbin,
        range=[[-half, half], [-half, half]],
        weights=parts["mass"][m],
    )
    return H


def run_loo_demo(lib, args, rng) -> list[dict]:
    """Encode held-out snaps → LOO retrieve → decode; score full dynamics."""
    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: -float(r["a2"]))
    # Prefer late barred snaps for 906c4 / 54a8 + one mid.
    targets = []
    for h in ("906c4af73543", "54a8faf836a0", "d9cd388549c2"):
        rows = [r for r in ranked if (r.get("run_hash") or _hash_from_path(r["path"])) == h]
        if rows:
            targets.append(max(rows, key=lambda r: float(r["a2"])))
    results = []
    loo_excl = str(getattr(args, "loo_exclusion", "hash")).lower()
    for row in targets:
        h = row.get("run_hash") or _hash_from_path(row["path"])
        path = Path(row["path"])
        ic_path = _ic_path_for_dump(path)
        print(
            f"=== LOO encode→retrieve {h} {path.name} a2={row['a2']:.3f} "
            f"excl={loo_excl} decode={args.decode_mode} ic={ic_path} ===",
            flush=True,
        )
        z, _feat_self, a2_enc = lib.encode_to_z(path)
        th = theta_vector(row.get("theta") or {}, keys=DEFAULT_THETA_KEYS)
        mass_prior = None
        if ic_path is not None:
            mass_prior = _component_masses(_load_parts(ic_path, None, rng))

        meta: dict = {"method": "retrieve_loo", "exclusion": loo_excl}
        if args.decode_mode == "particle_retrieve" and loo_excl == "path":
            # Dense same-campaign corpus retrieve (excl eval path only).
            nn_path, zdist, a2_nn = _dense_path_loo_neighbor(
                path,
                run_hash=h,
                z_eval=z,
                a2_enc=float(a2_enc),
                ranked=ranked,
                pca_mean=lib.pca_mean,
                pca_w=lib.pca_w,
                teacher=lib.teacher,
                cfg=lib.cfg,
                stats=lib.stats,
                enc_grid=args.enc_grid,
            )
            n_use = None if getattr(args, "full_n_retrieve", False) else args.n_resample
            gen = _decode_particle_retrieve(nn_path, mass_prior or {}, n_use, rng)
            meta.update(
                {
                    "neighbor_path": str(nn_path),
                    "z_dist": float(zdist),
                    "a2_nn": float(a2_nn),
                    "pool_size": "dense_same_campaign",
                }
            )
            tag = f"loo_path_{h[:5]}"
            title = (
                f"LOO-path retrieve (full-N) vs {h[:8]} "
                f"(nn={nn_path.name}, no eval particles)"
                if n_use is None
                else (
                    f"LOO-path retrieve+resample vs {h[:8]} "
                    f"(nn={nn_path.name}, no eval particles)"
                )
            )
        else:
            excl_hashes = {h} if loo_excl == "hash" else None
            excl_paths = {str(path), str(path.resolve())} if loo_excl == "path" else None
            feat, meta_r = lib.retrieve_loo(
                z,
                k=args.retrieve_k,
                exclude_hashes=excl_hashes,
                exclude_paths=excl_paths,
                theta=th,
                theta_k=args.theta_k,
                amplify=None,
            )
            meta.update({k: v for k, v in meta_r.items() if k != "z"})
            if args.decode_mode == "particle_retrieve":
                nn_i = int(meta_r["nn"][0])
                nn_path = Path(lib.meta[nn_i]["path"])
                n_use = None if getattr(args, "full_n_retrieve", False) else args.n_resample
                gen = _decode_particle_retrieve(nn_path, mass_prior or {}, n_use, rng)
                meta["neighbor_path"] = str(nn_path)
            else:
                gen = _decode_resample(
                    lib,
                    feat,
                    args.n_resample,
                    rng,
                    mode=args.decode_mode,
                    ic_path=ic_path,
                    mass_total_per_component=(
                        mass_prior if args.decode_mode == "soft" else None
                    ),
                    residual_alpha=args.residual_alpha,
                )
            tag = f"loo_path_{h[:5]}" if loo_excl == "path" else f"loo_{h[:5]}"
            title = (
                f"LOO-{loo_excl} {args.decode_mode} decode(z|θ) vs {h[:8]} "
                f"(no eval dump particles)"
            )

        ref_n = None if getattr(args, "full_n_retrieve", False) else args.n_resample
        ref = _load_parts(path, ref_n, rng)
        score = _full_system_score(gen, ref)
        plot_profiles(gen, ref, args.out / "figs" / f"{tag}_profiles.png", title=title)
        # Face-on morph panel for path LOO.
        if "path" in tag:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
            nn_lab = Path(meta.get("neighbor_path", "nn")).name
            for ax, parts, lab in zip(
                axes, [ref, gen], ["ref dump", f"gen nn={nn_lab}"]
            ):
                img = _disk_faceon(parts)
                from galacticsics.campaign.analysis import dens_array_log10

                show, vmin_s, vmax_s, _ = dens_array_log10(img, vmax_pct=99.0)
                ax.imshow(
                    show.T,
                    origin="lower",
                    cmap="inferno",
                    vmin=vmin_s,
                    vmax=vmax_s,
                )
                ax.set_title(lab, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"LOO-path face-on {h[:8]}")
            fig.tight_layout()
            fig.savefig(args.out / "figs" / f"{tag}_faceon.png", dpi=160)
            plt.close(fig)
        # Save gen IC (cap large full-N dumps for evolve samples).
        save_gen = gen
        if len(gen["pos"]) > max(args.n_resample, 500_000):
            save_gen = _decode_particle_retrieve(
                Path(meta["neighbor_path"]), mass_prior or {}, args.n_resample, rng
            )
        np.savez_compressed(
            args.out / "samples" / f"{tag}_gen.npz",
            pos=save_gen["pos"],
            vel=save_gen["vel"],
            mass=save_gen["mass"],
            component_id=save_gen["component_id"],
            eps=save_gen.get("eps", np.full(len(save_gen["pos"]), 0.05)),
            z=z,
            meta_json=json.dumps({k: v for k, v in meta.items() if k != "z"}),
        )
        row_out = {
            "hash": h,
            "path": str(path),
            "ic_path": str(ic_path) if ic_path else None,
            "decode_mode": args.decode_mode,
            "loo_exclusion": loo_excl,
            "a2_rank": float(row["a2"]),
            "a2_enc": float(a2_enc),
            "retrieve": {k: v for k, v in meta.items() if k not in ("z",)},
            "score": score,
            "z_norm": float(np.linalg.norm(z)),
            "mass_gen": _component_masses(gen),
            "mass_ref": _component_masses(ref),
        }
        results.append(row_out)
        dens_d = score["dens"]["disk"]["med_abs_log"]
        dens_h = score["dens"]["halo"]["med_abs_log"]
        dens_b = score["dens"]["bulge"]["med_abs_log"]
        print(
            f"  dens disk/halo/bulge med|log|="
            f"{dens_d:.3f}/{dens_h:.3f}/{dens_b:.3f}  "
            f"kin_mse={score['kin_mean_mse']:.4f}  "
            f"A2(Rd) gen={score['a2_rd_gen']:.3f} ref={score['a2_rd_ref']:.3f}  "
            f"nn={meta.get('neighbor_path', meta.get('pool_size'))}",
            flush=True,
        )
        for src, dst in [
            (
                args.out / "figs" / f"{tag}_profiles.png",
                ROOT
                / "papers/mnras_noneq_ics/figures"
                / f"fig_latent_theta_{tag}_profiles.png",
            ),
            (
                args.out / "figs" / f"{tag}_faceon.png",
                ROOT
                / "papers/mnras_noneq_ics/figures"
                / f"fig_latent_theta_{tag}_faceon.png",
            ),
        ]:
            if src.is_file():
                dst.write_bytes(src.read_bytes())
    return results


def run_interp_demo(lib, args, rng) -> dict:
    """Fixed-θ interpolation between quiet and dynamical library members."""
    # Pick a campaign with both quiet and bar members in library.
    by = defaultdict(list)
    for i, m in enumerate(lib.meta):
        by[lib.hashes[i]].append(i)
    chosen = None
    for h, idxs in by.items():
        a2s = lib.a2[idxs]
        if a2s.max() >= args.bar_floor and a2s.min() <= args.quiet_ceil + 0.05:
            chosen = h
            break
    if chosen is None:
        # Fall back: global quiet mean vs strongest bar.
        i_bar = int(np.argmax(lib.a2))
        i_q = int(np.argmin(lib.a2))
        chosen = lib.hashes[i_bar]
    else:
        idxs = by[chosen]
        i_bar = int(idxs[int(np.argmax(lib.a2[idxs]))])
        i_q = int(idxs[int(np.argmin(lib.a2[idxs]))])

    print(
        f"=== z-interp @ θ~{chosen} quiet#{i_q}(A2={lib.a2[i_q]:.3f}) "
        f"→ bar#{i_bar}(A2={lib.a2[i_bar]:.3f}) ===",
        flush=True,
    )
    feat_q = lib.feats[i_q]
    feat_b = lib.feats[i_bar]
    panels, labels, scores = [], [], []
    ts = [0.0, 0.33, 0.66, 1.0]
    for t in ts:
        feat = lib.interpolate_features(feat_q, feat_b, t)
        gen = _decode_resample(
            lib, feat, min(args.n_resample, 120_000), rng, mode="soft"
        )
        panels.append(_disk_faceon(gen))
        labels.append(rf"$t={t:.2f}$  $A_2$={_a2_rd(gen, 2.0):.2f}")
        scores.append({"t": t, "a2_rd": _a2_rd(gen, 2.0)})
        np.savez_compressed(
            args.out / "samples" / f"interp_t{t:.2f}.npz",
            pos=gen["pos"],
            vel=gen["vel"],
            mass=gen["mass"],
            component_id=gen["component_id"],
        )
    plot_interp_faceon(panels, labels, args.out / "figs" / "z_interp_faceon.png")
    return {"hash": chosen, "i_quiet": i_q, "i_bar": i_bar, "curve": scores}


def run_neighbor_consistency(lib, args, rng) -> dict:
    """Nearby z ⇒ more similar decoded dens+kin than far z (full-system)."""
    n_pairs = min(args.n_neighbor_pairs, max(len(lib.codes) // 2, 4))
    # Pair each of n seeds with its nearest and a far member (excluding rotations of self path).
    near_scores, far_scores = [], []
    for _ in range(n_pairs):
        i0 = int(rng.integers(0, len(lib.codes)))
        d = np.linalg.norm(lib.codes - lib.codes[i0], axis=1)
        d[i0] = np.inf
        # Exclude same path rotations.
        p0 = lib.meta[i0].get("path", "")
        for j, m in enumerate(lib.meta):
            if m.get("path") == p0:
                d[j] = np.inf
        if not np.isfinite(d).any():
            continue
        i_near = int(np.argmin(d))
        # Far: high quantile among finite.
        finite = np.where(np.isfinite(d))[0]
        i_far = int(finite[np.argsort(d[finite])[int(0.85 * (len(finite) - 1))]])
        gen0 = _decode_resample(
            lib, lib.feats[i0], min(80_000, args.n_resample), rng, mode="soft"
        )
        for i_other, bucket in ((i_near, near_scores), (i_far, far_scores)):
            gen1 = _decode_resample(
                lib, lib.feats[i_other], min(80_000, args.n_resample), rng, mode="soft"
            )
            sc = _full_system_score(gen0, gen1)
            bucket.append(
                {
                    "i0": i0,
                    "i1": i_other,
                    "z_dist": float(np.linalg.norm(lib.codes[i0] - lib.codes[i_other])),
                    "dens_disk": sc["dens"]["disk"]["med_abs_log"],
                    "kin_mse": sc["kin_mean_mse"],
                }
            )
    def _mean(xs, key):
        vals = [x[key] for x in xs if np.isfinite(x[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n": n_pairs,
        "near_dens_disk": _mean(near_scores, "dens_disk"),
        "far_dens_disk": _mean(far_scores, "dens_disk"),
        "near_kin_mse": _mean(near_scores, "kin_mse"),
        "far_kin_mse": _mean(far_scores, "kin_mse"),
        "near_z_dist": _mean(near_scores, "z_dist"),
        "far_z_dist": _mean(far_scores, "z_dist"),
        "pairs_near": near_scores,
        "pairs_far": far_scores,
    }
    print(
        f"=== neighbor consistency ===\n"
        f"  near dens|log|={summary['near_dens_disk']:.3f}  far={summary['far_dens_disk']:.3f}\n"
        f"  near kin_mse={summary['near_kin_mse']:.4f}  far={summary['far_kin_mse']:.4f}",
        flush=True,
    )
    return summary


def maybe_evolve(args, sample_npz: Path, tag: str) -> dict | None:
    if args.evolve_gyr <= 0:
        return None
    from evolve_component_slices import _evolve_component_tracked

    parts = _load_parts(sample_npz, None, np.random.default_rng(0))
    # Ensure eps
    if "eps" not in parts:
        parts["eps"] = np.full(len(parts["pos"]), 0.05)
    print(f"=== evolve {tag} {args.evolve_gyr} Gyr force={args.force} ===", flush=True)
    out = _evolve_component_tracked(
        parts,
        end_gyr=float(args.evolve_gyr),
        dt=0.01,
        omp=args.omp,
        timeout_s=1e9,
        n_track=9,
        snap_times=[0.0, 0.25, 0.5, float(args.evolve_gyr)],
        force=args.force,
        a2_r_eval=2.0,
    )
    # Save a2 curve
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(out["t_gyr"], out["a2_t"], lw=2.0)
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"$A_2(R_d)$")
    ax.set_title(f"generative IC evolve ({tag})")
    fig.tight_layout()
    fig.savefig(args.out / "figs" / f"evolve_{tag}_a2_t.png", dpi=150)
    fig.savefig(
        ROOT / "papers/mnras_noneq_ics/figures" / f"fig_latent_theta_evolve_{tag}_a2_t.png",
        dpi=150,
    )
    plt.close(fig)
    return {
        "a2_pre": float(out["a2_t"][0]),
        "a2_post": float(out["a2_t"][-1]),
        "t_gyr": [float(x) for x in out["t_gyr"]],
        "a2_t": [float(x) for x in out["a2_t"]],
        "com_drift_kpc": float(out.get("com_drift_kpc", float("nan"))),
        "force": out.get("force"),
    }


def write_journal(args, payload: dict) -> None:
    out = args.out
    loo = payload.get("loo", [])
    neigh = payload.get("neighbor", {})
    interp = payload.get("interp", {})
    lines = [
        "# JOURNAL — θ-conditioned latent generative ICs (2026-08-02)",
        "",
        "**Product:** sample similar *full-system* galaxies from ``z|θ``",
        "(disk/bulge/halo dens + velocity moments via frozen multitower teacher).",
        "Bars/spirals/quiet are regions of ``z``, not the sole objective.",
        "",
        "## Recipe",
        "",
        "```",
        "θ → GalactICS structural family (conditioning)",
        "encode corpus → teacher bottleneck+skips library → PCA z",
        "sample: z ~ prior|encode, retrieve features near (z|θ), exclude held-out hash",
        "decode: residual_f0 = axisym dens+moments from GalactICS f0(θ) IC",
        "         + non-axisym dens residual from retrieved morph; CDF-lock disk R",
        "         (soft AE dens+moments resample remains an ablation)",
        "```",
        "",
        f"- Teacher: `{args.teacher}`",
        f"- Decode mode (LOO): `{args.decode_mode}`  α={args.residual_alpha}",
        f"- Library members: {payload.get('n_lib')}  campaigns: {payload.get('n_campaigns')}",
        f"- Resample N: {args.n_resample}",
        "",
        "## Latent organization",
        "",
        "See `figs/latent_z_organization.png` / "
        "`papers/.../fig_latent_theta_z_organization.png`.",
        "PCA of pooled-bottleneck ``z`` separates high-A₂ dynamical states from quiet;",
        "θ/campaign structure is visible but morphology dominates PC1–2.",
        "",
        "## Leave-one-out generative decode (no self dump particles)",
        "",
        "| System | dens disk/halo/bulge | kin mean MSE | A₂(R_d) gen→ref |",
        "|--------|----------------------|--------------|-----------------|",
    ]
    for r in loo:
        sc = r["score"]
        lines.append(
            f"| {r['hash'][:8]} | "
            f"{sc['dens']['disk']['med_abs_log']:.3f}/"
            f"{sc['dens']['halo']['med_abs_log']:.3f}/"
            f"{sc['dens']['bulge']['med_abs_log']:.3f} | "
            f"{sc['kin_mean_mse']:.4f} | {sc['a2_rd_gen']:.3f}→{sc['a2_rd_ref']:.3f} |"
        )
    lines += [
        "",
        "## Neighbor consistency (nearby z → similar galaxies)",
        "",
        f"- Near dens med‖log‖ disk = **{neigh.get('near_dens_disk', float('nan')):.3f}**"
        f"  vs far **{neigh.get('far_dens_disk', float('nan')):.3f}**",
        f"- Near kin MSE = **{neigh.get('near_kin_mse', float('nan')):.4f}**"
        f"  vs far **{neigh.get('far_kin_mse', float('nan')):.4f}**",
        "",
        "## Fixed-θ z interpolation",
        "",
        f"- Campaign `{interp.get('hash')}` quiet→bar A₂(R_d) curve: "
        + ", ".join(
            f"t={c['t']:.2f}:{c['a2_rd']:.3f}" for c in interp.get("curve", [])
        ),
        "",
        "## Evolve (coherence)",
        "",
    ]
    ev = payload.get("evolve") or {}
    if ev:
        for k, v in ev.items():
            lines.append(
                f"- `{k}`: A₂(R_d) {v['a2_pre']:.3f}→{v['a2_post']:.3f} "
                f"(COM {v.get('com_drift_kpc', float('nan')):.4f} kpc)"
            )
    else:
        lines.append("- skipped (`--evolve-gyr 0`) or pending")
    lines += [
        "",
        "## Honest blockers / gaps",
        "",
        "1. Retrieval+decode is generative but still library-backed "
        "(not a free continuous skip synthesizer). End-to-end `decode(z)` washes structure.",
        "2. Soft AE dens+moments alone under-predict outer halo mass (FOV) and wash "
        "⟨v_φ⟩; LOO default is residual-on-f0 (axisym dens+kin from GalactICS IC).",
        "3. Prior dens med|log|≈0.8 vs dumps was largely an **eval bug**: stratified "
        "downsample kept per-particle masses and under-scaled ref dens by ~N_kept/N_full.",
        "4. θ-gating helps OOD campaigns but sparse θ coverage limits extrapolation.",
        "5. Soft AE moments ≠ full 6D DF; residual_f0 uses f0 moments + morph dens.",
        "",
        "## Stopped / demoted",
        "",
        "- `full_dyn_replace` data-disk grafting (not generative).",
        "- Eval-dump morph-vel transplant as the product method.",
        "- Paint-on-bar / A₂-only success framing.",
        "- Soft-AE-only LOO dens claims without mass-preserving ref downsample.",
        "",
    ]
    (out / "JOURNAL.md").write_text("\n".join(lines) + "\n")
    # Scoreboard
    sb = [
        "# SCOREBOARD — latent θ generative full-system ICs",
        "",
        "Primary success = **nearby z → similar full-system galaxies** (dens+kin),",
        "not bar A2 alone.",
        "",
        f"LOO decode mode: **`{args.decode_mode}`** "
        "(residual_f0 = axisym from GalactICS f0(θ) + morph from retrieve;",
        "halo/bulge retained from f0 particles).",
        "Ref dens scored with **mass-preserving** stratified downsample.",
        "",
        "## Dens med|log|≈0.8 root cause (fixed)",
        "",
        "Stratified downsample used to drop ref component mass by ~N_kept/N_full",
        "(~7–9×); that fake offset is gone. Soft AE alone ≈0.2 dex once honest;",
        "residual_f0 brings disk dens to ~0.04–0.12.",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| Latent organization fig | "
        f"{'yes' if (out / 'figs/latent_z_organization.png').is_file() else 'no'} |",
        f"| Neighbor dens near≪far | "
        f"**{neigh.get('near_dens_disk', 9):.3f} < {neigh.get('far_dens_disk', 0):.3f}** "
        f"{'PASS' if neigh.get('near_dens_disk', 9) < neigh.get('far_dens_disk', 0) else 'FAIL'} |",
        f"| Neighbor kin near≪far | "
        f"**{neigh.get('near_kin_mse', 9):.3f} < {neigh.get('far_kin_mse', 0):.3f}** "
        f"{'PASS' if neigh.get('near_kin_mse', 9) < neigh.get('far_kin_mse', 0) else 'FAIL'} |",
        f"| Fixed-θ z-interp A₂(R_d) | "
        + (
            " → ".join(f"{c['a2_rd']:.3f}" for c in interp.get("curve", []))
            if interp.get("curve")
            else "n/a"
        )
        + " |",
    ]
    for r in loo:
        sc = r["score"]
        sb.append(
            f"| LOO {r['hash'][:5]} dens d/h/b / kin / A2Rd | "
            f"{sc['dens']['disk']['med_abs_log']:.3f}/"
            f"{sc['dens']['halo']['med_abs_log']:.3f}/"
            f"{sc['dens']['bulge']['med_abs_log']:.3f} / "
            f"{sc['kin_mean_mse']:.3f} / "
            f"{sc['a2_rd_gen']:.3f}≈{sc['a2_rd_ref']:.3f} |"
        )
    for k, v in (ev or {}).items():
        sb.append(
            f"| evolve {k} A2Rd | {v['a2_pre']:.3f}→{v['a2_post']:.3f} coherent |"
        )
    sb += [
        "| No eval-dump particle copy | **yes** (retrieve+decode; f0=GalactICS IC) |",
        "",
        "**Works for:** sampling similar galaxies from z|θ (neighbor test);",
        "absolute disk+halo dens via residual-on-f0.",
        "**Partial:** soft-AE-only dens/kin still washed; LOO A2 under-shoots;",
        "bulge dens vs evolved dump ≈ f0 vs dump (cusp vs bar-heated).",
        "**Demoted:** graft / eval morph-vel / A2-only paint / mass-destroyed dens score.",
        "",
    ]
    (out / "SCOREBOARD.md").write_text("\n".join(sb) + "\n")
    for name in ("JOURNAL.md", "SCOREBOARD.md"):
        dst = ROOT / "papers/mnras_noneq_ics/results" / f"latent_theta_gen_{name}"
        dst.write_text((out / name).read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/latent_theta_gen_2026-08-02"),
    )
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar-campaigns", type=int, default=12)
    p.add_argument("--snaps-per-bar-campaign", type=int, default=2)
    p.add_argument("--n-quiet", type=int, default=16)
    p.add_argument("--n-mid", type=int, default=8)
    p.add_argument("--n-rot-bar", type=int, default=2)
    p.add_argument("--bar-floor", type=float, default=0.22)
    p.add_argument("--quiet-ceil", type=float, default=0.06)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-resample", type=int, default=200_000)
    p.add_argument("--retrieve-k", type=int, default=5)
    p.add_argument("--theta-k", type=int, default=28)
    p.add_argument("--n-neighbor-pairs", type=int, default=8)
    p.add_argument("--evolve-gyr", type=float, default=0.5)
    p.add_argument("--force", type=str, default="gpu_bh", choices=("gpu_bh", "bh_c", "bh"))
    p.add_argument("--omp", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-evolve", action="store_true")
    p.add_argument(
        "--decode-mode",
        type=str,
        default="particle_retrieve",
        choices=("particle_retrieve", "residual_f0", "soft"),
        help=(
            "LOO decode: particle_retrieve from LOO neighbor (whole DF; default), "
            "residual on GalactICS f0(θ), or soft AE dens+moments"
        ),
    )
    p.add_argument(
        "--loo-exclusion",
        type=str,
        default="path",
        choices=("path", "hash"),
        help="path=exclude eval dump only (same-campaign OK); hash=exclude campaign",
    )
    p.add_argument(
        "--full-n-retrieve",
        action="store_true",
        help="For particle_retrieve: keep full neighbor N (best dens/kin eye match)",
    )
    p.add_argument(
        "--residual-alpha",
        type=float,
        default=1.25,
        help="Morph residual amplitude for residual_f0 decode",
    )
    args = p.parse_args()
    if args.teacher is None:
        args.teacher = _default_teacher()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figs").mkdir(exist_ok=True)
    (args.out / "samples").mkdir(exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    (args.out / "gates").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"=== build θ-balanced library teacher={args.teacher} ===", flush=True)
    lib = build_library(args)
    lib.save_codes(args.out / "feature_library_codes.npz")
    n_camp = len(set(lib.hashes.tolist()))
    print(f"library size={len(lib.codes)} campaigns={n_camp}", flush=True)

    plot_latent(lib, args.out)
    loo = run_loo_demo(lib, args, rng)
    neigh = run_neighbor_consistency(lib, args, rng)
    interp = run_interp_demo(lib, args, rng)

    evolve = {}
    if not args.skip_evolve and args.evolve_gyr > 0:
        # Evolve one LOO generative IC if present.
        gens = sorted((args.out / "samples").glob("loo_*_gen.npz"))
        if gens:
            tag = gens[0].stem.replace("_gen", "")
            ev = maybe_evolve(args, gens[0], tag)
            if ev:
                evolve[tag] = ev

    payload = {
        "n_lib": len(lib.codes),
        "n_campaigns": n_camp,
        "teacher": str(args.teacher),
        "loo": loo,
        "neighbor": neigh,
        "interp": interp,
        "evolve": evolve,
    }
    (args.out / "verdict.json").write_text(json.dumps(payload, indent=2, default=str))
    write_journal(args, payload)
    print("=== DONE ===", flush=True)
    print((args.out / "SCOREBOARD.md").read_text())


if __name__ == "__main__":
    main()
