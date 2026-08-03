#!/usr/bin/env python3
"""OOD structural-θ compare: GalactICS IC vs FFT recon vs θ-nearest library.

Training coverage is the MW Morton field-map manifest (19 unique models).
Default cases are held-out corpus runs whose θ are **not** among those models
(combinatorial OOD inside the axis box). Feature library has **no** native
θ-conditioning — we document that and fall back to θ-nearest quiet retrieve.

Example::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \\
      .venv/bin/python scripts/ood_theta_compare.py \\
        --teacher runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt \\
        --out runs/ml/field_maps/ood_theta_compare_2026-07-26 \\
        --n-disk 1000000 --evolve-gyr 0.50 --dt 0.01 --omp 2 \\
        --paper-figures papers/mnras_noneq_ics/figures
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sample_latent_ic import (  # noqa: E402
    COUNT,
    RANK,
    _a2_disk,
    _disk_collapse,
    build_library,
    default_teacher,
)
from evolve_resample_compare import (  # noqa: E402
    _evolve_tracked,
    _faceon,
    _full_com,
    _metrics,
    _plot_a2_t,
    _recon_particles,
    _stratified_down,
    _vcom_only,
)


def _ensure_eps(parts: dict, eps: float = 0.1) -> dict:
    """Resampled stacks omit softening; BH evolve requires ``eps``."""
    out = dict(parts)
    n = int(out["pos"].shape[0])
    if "eps" not in out or getattr(out["eps"], "shape", ()) != (n,):
        out["eps"] = np.full(n, float(eps), dtype=np.float64)
    return out
from smoke_field_maps import _am_profiles  # noqa: E402

from galacticsics.ml.conditioning import DEFAULT_THETA_KEYS  # noqa: E402
from galacticsics.ml.fields.normalize import denormalize_stack  # noqa: E402
from galacticsics.ml.fields.resample import resample_particles_from_multiscale  # noqa: E402
from galacticsics.ml.morton.polygon import _component_ids  # noqa: E402


MANIFEST = Path("runs/ml/smoke_morton_vae_cpu/manifest_mw_morton_corpus_v2.json")
CORPUS = Path("runs/mw_morton_corpus_v2")

# Structural keys used for OOD distance (exclude fixed sigma_r0 / t_gyr).
THETA_DIST_KEYS = [
    "disk.mass",
    "disk.scale_length",
    "disk.scale_height",
    "disk_kinematics.toomre_q_target",
    "halo.v0",
    "halo.a",
    "bulge.v0",
    "bulge.a",
]

# Held-out corpus runs: not in the 19-model field-map train set.
DEFAULT_OOD_CASES: list[dict] = [
    {
        "name": "heavy_ext_quiet",
        "run_hash": "b847fdf4852f",
        "note": "M=22 Rd=3.0 zd=0.25 Q=1.7 — heavy extended; stays quiet to 2 Gyr",
        "ood_how": "combinatorial OOD: θ absent from 19 train models (dmin≈1.06)",
    },
    {
        "name": "heavy_bar_forming",
        "run_hash": "b31995c39105",
        "note": "M=26 Rd=3.0 zd=0.25 Q=1.4 — extreme mass×Q; mild bar by 2 Gyr",
        "ood_how": "combinatorial OOD: θ absent from 19 train models (dmin≈0.93)",
    },
    {
        "name": "light_compact_bar",
        "run_hash": "695ec813a92f",
        "note": "M=10 Rd=2.0 zd=0.25 Q=1.7 Ha=40 — light compact + large halo",
        "ood_how": "combinatorial OOD: θ absent from 19 train models (dmin≈0.93)",
    },
    {
        "name": "thick_stable",
        "run_hash": "6272b66b640d",
        "note": "M=10 Rd=2.5 zd=0.55 Q=2.5 — thick high-Q stable disk",
        "ood_how": "combinatorial OOD: θ absent from 19 train models (dmin≈1.01)",
    },
]


def _theta_from_model(model: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in DEFAULT_THETA_KEYS:
        if k == "t_gyr":
            out[k] = 0.0
            continue
        comp, leaf = k.split(".", 1)
        out[k] = float(model[comp][leaf])
    return out


def _vec(theta: dict[str, float], keys: list[str]) -> np.ndarray:
    return np.asarray([float(theta[k]) for k in keys], dtype=np.float64)


def _train_coverage(manifest: Path) -> dict:
    raw = json.loads(manifest.read_text())
    by_run: dict[str, dict] = {}
    for r in raw["records"]:
        h = r["run_hash"]
        if h in by_run:
            continue
        by_run[h] = {k: float(r["theta"][k]) for k in THETA_DIST_KEYS}
    arr = np.stack([_vec(v, THETA_DIST_KEYS) for v in by_run.values()])
    lo, hi = arr.min(0), arr.max(0)
    return {
        "n_models": len(by_run),
        "n_snapshots": int(raw["n_snapshots"]),
        "keys": list(THETA_DIST_KEYS),
        "run_hashes": sorted(by_run.keys()),
        "theta_by_run": by_run,
        "box_lo": dict(zip(THETA_DIST_KEYS, lo.tolist())),
        "box_hi": dict(zip(THETA_DIST_KEYS, hi.tolist())),
        "span": np.maximum(hi - lo, 1e-6),
        "points": arr,
    }


def _dmin_to_train(theta: dict[str, float], cov: dict) -> float:
    v = (_vec(theta, THETA_DIST_KEYS) - np.asarray(list(cov["box_lo"].values()))) / cov["span"]
    pts = (cov["points"] - np.asarray(list(cov["box_lo"].values()))) / cov["span"]
    return float(np.min(np.linalg.norm(pts - v[None, :], axis=1)))


def _outside_box(theta: dict[str, float], cov: dict) -> bool:
    v = _vec(theta, THETA_DIST_KEYS)
    lo = np.asarray(list(cov["box_lo"].values()))
    hi = np.asarray(list(cov["box_hi"].values()))
    return bool(np.any(v < lo - 1e-9) or np.any(v > hi + 1e-9))


def _run_hash_from_path(path: str | Path) -> str | None:
    parts = Path(path).parts
    for i, x in enumerate(parts):
        if x.startswith("mw_morton_corpus") and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _load_ic(path: Path, n: int, rng: np.random.Generator) -> dict:
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
    parts = {
        "pos": pos,
        "vel": vel,
        "mass": mass,
        "eps": eps,
        "component_id": cid,
    }
    parts = _full_com(parts)
    if parts["pos"].shape[0] != n:
        parts = _stratified_down(parts, n, rng)
        parts["eps"] = np.full(parts["pos"].shape[0], 0.1, dtype=np.float64)
    return parts


def _library_theta_table(lib, cov: dict) -> tuple[np.ndarray, list[dict]]:
    """Map each library member → structural θ of its source run (train only)."""
    rows = []
    for i, m in enumerate(lib.meta):
        h = _run_hash_from_path(m["path"])
        if h is None or h not in cov["theta_by_run"]:
            continue
        th = cov["theta_by_run"][h]
        rows.append(
            {
                "idx": i,
                "run_hash": h,
                "kind": m.get("kind", "mid"),
                "a2": float(lib.a2[i]),
                "theta": th,
                "path": m["path"],
            }
        )
    if not rows:
        raise RuntimeError("no library members mapped to train θ")
    pts = np.stack([_vec(r["theta"], THETA_DIST_KEYS) for r in rows])
    return pts, rows


def _concat_particle_dicts(chunks: list[dict]) -> dict:
    """Concatenate particle dicts (pos/vel/mass/eps/component_id)."""
    if not chunks:
        raise ValueError("no particle chunks to concatenate")
    if len(chunks) == 1:
        return chunks[0]
    out = {}
    for key in ("pos", "vel", "mass", "eps", "component_id"):
        out[key] = np.concatenate([c[key] for c in chunks], axis=0)
    return out


def _theta_nearest_sample(
    lib,
    target_theta: dict[str, float],
    cov: dict,
    *,
    prefer_quiet: bool,
    quiet_ceil: float,
    k: int,
    rng: np.random.Generator,
    n_resample: int,
    velocity_frame: str = "cartesian",
    match_cell_moments: bool = False,
    via_ae: bool = False,
) -> tuple[dict, dict]:
    """θ-nearest quiet retrieve (no native θ-conditioning in TeacherFeatureLibrary).

    Default ``via_ae=False`` loads **raw GalactICS snapshot particles** from the
    nearest library member(s).  That preserves a trusted DF (healthy ⟨v_φ⟩).

    ``via_ae=True`` is the legacy path: blend teacher features → decode →
    resample.  AE moment channels systematically collapse outer-disk ⟨v_φ⟩ even
    when dens is good, so that path must not be used as the “library DF” arm.
    ``velocity_frame`` / ``match_cell_moments`` apply only when ``via_ae``.
    """
    pts, rows = _library_theta_table(lib, cov)
    lo = np.asarray(list(cov["box_lo"].values()))
    span = cov["span"]
    tn = (_vec(target_theta, THETA_DIST_KEYS) - lo) / span
    pn = (pts - lo) / span
    pool = list(range(len(rows)))
    if prefer_quiet:
        q = [j for j, r in enumerate(rows) if r["a2"] <= quiet_ceil or r["kind"] == "quiet"]
        if len(q) >= max(1, k):
            pool = q
    d = np.linalg.norm(pn[pool] - tn[None, :], axis=1)
    order = np.argsort(d)
    take = [pool[int(order[i])] for i in range(min(k, len(order)))]
    dists = [float(d[int(order[i])]) for i in range(len(take))]
    idxs = [rows[j]["idx"] for j in take]
    if len(idxs) == 1:
        w = np.array([1.0])
    else:
        # Softmax over −distance in θ-space.
        logits = -np.asarray(dists, dtype=np.float64)
        logits -= logits.max()
        w = np.exp(logits)
        w = w / w.sum()

    if via_ae:
        if len(idxs) == 1:
            feat = lib.feats[idxs[0]]
        else:
            from galacticsics.ml.fields.feature_library import blend_features

            feat = blend_features([lib.feats[i] for i in idxs], w)
        fields = lib.decode_features(feat)
        phys = {
            key: denormalize_stack(v[0].detach().cpu().numpy(), lib.stats[key])
            for key, v in fields.items()
        }
        parts = resample_particles_from_multiscale(
            phys,
            cfg=lib.cfg,
            n_particles=n_resample,
            count_fractions=COUNT,
            rng=rng,
            velocity_frame=velocity_frame,
            match_cell_moments=match_cell_moments,
        )
        method = "theta_nearest_ae_decode"
        limit = (
            "legacy AE feature decode→resample; outer ⟨v_φ⟩ often collapses"
        )
    else:
        # Raw snapshot particles (trusted DF). Weight-blend k neighbours by
        # stratified subsample counts.
        chunks: list[dict] = []
        rem = int(n_resample)
        for i, jrow in enumerate(take):
            n_i = rem if i == len(take) - 1 else int(round(float(w[i]) * n_resample))
            n_i = int(max(n_i, 0))
            rem -= n_i
            if n_i <= 0:
                continue
            chunks.append(_load_ic(Path(rows[jrow]["path"]), n_i, rng))
        parts = _concat_particle_dicts(chunks)
        if parts["pos"].shape[0] != int(n_resample):
            parts = _stratified_down(parts, int(n_resample), rng)
            parts["eps"] = np.full(parts["pos"].shape[0], 0.1, dtype=np.float64)
        method = "theta_nearest_raw_particles"
        limit = (
            "TeacherFeatureLibrary has no θ-conditioning; θ-nearest quiet "
            "retrieve uses raw GalactICS snapshot particles (not AE decode)"
        )

    parts = _vcom_only(parts)
    meta = {
        "method": method,
        "limit": limit,
        "via_ae": bool(via_ae),
        "prefer_quiet": prefer_quiet,
        "k": len(idxs),
        "weights": w.tolist(),
        "theta_dists": dists,
        "nn": [
            {
                "lib_idx": int(idxs[i]),
                "run_hash": rows[take[i]]["run_hash"],
                "kind": rows[take[i]]["kind"],
                "a2": rows[take[i]]["a2"],
                "theta": rows[take[i]]["theta"],
                "path": rows[take[i]]["path"],
            }
            for i in range(len(idxs))
        ],
    }
    return parts, meta


def _plot_case_faceon(
    path: Path,
    arms: dict[str, dict],
    *,
    title: str,
) -> None:
    keys = [k for k in ("galactics_ic", "fft_recon", "theta_nearest") if k in arms and arms[k].get("ok")]
    if not keys:
        return
    n = len(keys)
    # columns = arms; rows = pre / post
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.6), squeeze=False)
    labels = {
        "galactics_ic": "GalactICS IC",
        "fft_recon": "FFT recon",
        "theta_nearest": r"θ-nearest lib",
    }
    for j, tag in enumerate(keys):
        row = arms[tag]
        maps = row.get("faceon_maps") or []
        pre = maps[0] if maps else row.get("face_pre")
        post = maps[-1] if maps else row.get("face_post")
        for i, img in enumerate((pre, post)):
            ax = axes[i, j]
            if img is None:
                ax.axis("off")
                continue
            pos = img[img > 0]
            vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
            ax.imshow(
                np.log1p(np.maximum(img, 0)),
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=np.log1p(vmax),
            )
            a2s = row.get("a2_t") or []
            a2_lab = f"A₂={a2s[0]:.3f}" if i == 0 and a2s else (
                f"A₂={a2s[-1]:.3f}" if a2s else ""
            )
            ax.set_title(f"{labels.get(tag, tag)} {'t=0' if i == 0 else 'post'} {a2_lab}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_gallery(path: Path, cases: list[dict]) -> None:
    """One row per OOD case; columns = GalactICS / recon / θ-nearest at t=0."""
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.55 * n), squeeze=False)
    col_labs = ["GalactICS IC", "FFT recon", "θ-nearest lib"]
    tags = ["galactics_ic", "fft_recon", "theta_nearest"]
    for i, case in enumerate(cases):
        arms = case["arms"]
        for j, tag in enumerate(tags):
            ax = axes[i, j]
            row = arms.get(tag) or {}
            maps = row.get("faceon_maps") or []
            img = maps[0] if maps else None
            if img is None:
                ax.text(0.5, 0.5, "fail", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            pos = img[img > 0]
            vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
            ax.imshow(
                np.log1p(np.maximum(img, 0)),
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=np.log1p(vmax),
            )
            a0 = (row.get("a2_t") or [float("nan")])[0]
            ax.set_title(f"{col_labs[j]} A₂={a0:.3f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                th = case["theta"]
                ax.set_ylabel(
                    f"{case['name']}\nM={th['disk.mass']:.0f} Rd={th['disk.scale_length']}\n"
                    f"zd={th['disk.scale_height']} Q={th['disk_kinematics.toomre_q_target']}",
                    fontsize=7,
                )
    fig.suptitle(r"OOD $\theta$: face-on at $t=0$ (disk$=10^6$)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_a2_bands(path: Path, cases: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    tags = ["galactics_ic", "fft_recon", "theta_nearest"]
    titles = ["GalactICS IC", "FFT recon", r"θ-nearest library"]
    for ax, tag, title in zip(axes, tags, titles):
        for case in cases:
            row = case["arms"].get(tag) or {}
            if not row.get("ok"):
                continue
            ax.plot(row["t_gyr"], row["a2_t"], lw=1.6, label=case["name"])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$t$ [Gyr]")
        ax.axhline(0.10, color="0.6", ls=":", lw=1)
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel(r"disk median $A_2$")
    fig.suptitle(r"OOD $\theta$ evolve $A_2(t)$", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k not in ("faceon_maps", "pos_final", "face_pre", "face_post")}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ml/field_maps/ood_theta_compare_2026-07-26"))
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--n-bar", type=int, default=24)
    p.add_argument("--n-quiet", type=int, default=14)
    p.add_argument("--n-mid", type=int, default=8)
    p.add_argument("--n-rot-bar", type=int, default=4)
    p.add_argument("--bar-floor", type=float, default=0.20)
    p.add_argument("--quiet-ceil", type=float, default=0.06)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-disk", type=int, default=1_000_000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--evolve-gyr", type=float, default=0.50)
    p.add_argument("--omp", type=int, default=2)
    p.add_argument("--timeout-s", type=float, default=7200.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--theta-knn", type=int, default=3)
    p.add_argument("--faceon-bins", type=int, default=128)
    p.add_argument("--faceon-times", type=str, default="0,0.25,0.5")
    p.add_argument("--cases", type=str, default="", help="Comma list of DEFAULT case names")
    p.add_argument("--paper-figures", type=Path, default=None)
    p.add_argument("--skip-evolve", action="store_true", help="IC/recon/sample only (no BH evolve)")
    args = p.parse_args()

    if args.teacher is None:
        args.teacher = default_teacher()
    n_tot = int(round(args.n_disk * 7 / 4))
    print(
        f"n-disk={args.n_disk:,} → total N={n_tot:,} "
        f"(halo≈{args.n_disk // 2:,}, bulge≈{args.n_disk // 4:,})",
        flush=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.omp)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    cov = _train_coverage(args.manifest)
    print(
        f"train coverage: {cov['n_models']} models, {cov['n_snapshots']} snaps; "
        f"box M∈[{cov['box_lo']['disk.mass']},{cov['box_hi']['disk.mass']}] "
        f"Rd∈[{cov['box_lo']['disk.scale_length']},{cov['box_hi']['disk.scale_length']}]",
        flush=True,
    )
    (args.out / "train_theta_coverage.json").write_text(
        json.dumps(
            {
                "n_models": cov["n_models"],
                "n_snapshots": cov["n_snapshots"],
                "keys": cov["keys"],
                "box_lo": cov["box_lo"],
                "box_hi": cov["box_hi"],
                "run_hashes": cov["run_hashes"],
                "theta_by_run": cov["theta_by_run"],
                "note": "FFT teacher / library train on these 19 MW Morton models only",
            },
            indent=2,
        )
    )

    want = {x.strip() for x in args.cases.split(",") if x.strip()}
    cases_spec = [c for c in DEFAULT_OOD_CASES if not want or c["name"] in want]
    if not cases_spec:
        raise SystemExit(f"no cases selected from {want}")

    print(f"=== teacher {args.teacher} ===", flush=True)
    print("=== build stratified library ===", flush=True)
    lib = build_library(args)

    faceon_times = [float(x) for x in args.faceon_times.split(",") if x.strip()]
    if args.evolve_gyr not in faceon_times:
        faceon_times.append(float(args.evolve_gyr))
    faceon_times = sorted(set(faceon_times))

    case_reports: list[dict] = []
    for ci, spec in enumerate(cases_spec):
        h = spec["run_hash"]
        run_dir = args.corpus / h
        model = json.loads((run_dir / "model.json").read_text())
        theta = _theta_from_model(model)
        dmin = _dmin_to_train(theta, cov)
        outside = _outside_box(theta, cov)
        ic_path = run_dir / "ic_state.npz"
        if not ic_path.is_file():
            raise SystemExit(f"missing IC {ic_path}")
        if h in cov["run_hashes"]:
            raise SystemExit(f"{h} is in train set — not OOD")

        print(f"\n=== case {ci+1}/{len(cases_spec)} {spec['name']} ({h[:12]}) ===", flush=True)
        print(
            f"  θ: M={theta['disk.mass']} Rd={theta['disk.scale_length']} "
            f"zd={theta['disk.scale_height']} Q={theta['disk_kinematics.toomre_q_target']} "
            f"Hv0={theta['halo.v0']} Ha={theta['halo.a']} | dmin={dmin:.3f} outside_box={outside}",
            flush=True,
        )
        print(f"  OOD: {spec['ood_how']}", flush=True)

        arms: dict[str, dict] = {}

        # --- GalactICS IC ---
        print("  arm: galactics_ic", flush=True)
        ic = _load_ic(ic_path, n_tot, rng)
        ic = _ensure_eps(ic)
        ic_met = _metrics(ic)
        print(f"    IC A₂={ic_met['a2']:.4f} N={ic_met['n']}", flush=True)
        if args.skip_evolve:
            arms["galactics_ic"] = {
                "ok": True,
                "a2_t": [ic_met["a2"]],
                "t_gyr": [0.0],
                "com_norm_t": [0.0],
                "com_drift_kpc": 0.0,
                "faceon_maps": [_faceon(ic, n_bins=args.faceon_bins)],
                "faceon_t_gyr": [0.0],
                "ic_metrics": ic_met,
            }
        else:
            ev = _evolve_tracked(
                ic,
                end_gyr=args.evolve_gyr,
                dt=args.dt,
                omp=args.omp,
                timeout_s=args.timeout_s,
                faceon_times=faceon_times,
                faceon_bins=args.faceon_bins,
            )
            ev["ic_metrics"] = ic_met
            arms["galactics_ic"] = ev
            print(
                f"    evolve ok={ev.get('ok')} A₂ {ic_met['a2']:.3f}→"
                f"{(ev.get('a2_t') or [float('nan')])[-1]:.3f} "
                f"COM={ev.get('com_drift_kpc')} wall={ev.get('wall_s')}",
                flush=True,
            )

        # --- FFT teacher recon ---
        print("  arm: fft_recon", flush=True)
        recon = _recon_particles(
            ic_path, lib.teacher, lib.cfg, lib.stats, n_resample=n_tot, rng=rng
        )
        recon = _ensure_eps(_vcom_only(recon))
        recon_met = _metrics(recon)
        print(f"    recon A₂={recon_met['a2']:.4f}", flush=True)
        if args.skip_evolve:
            arms["fft_recon"] = {
                "ok": True,
                "a2_t": [recon_met["a2"]],
                "t_gyr": [0.0],
                "com_norm_t": [0.0],
                "com_drift_kpc": 0.0,
                "faceon_maps": [_faceon(recon, n_bins=args.faceon_bins)],
                "faceon_t_gyr": [0.0],
                "ic_metrics": recon_met,
            }
        else:
            ev = _evolve_tracked(
                recon,
                end_gyr=args.evolve_gyr,
                dt=args.dt,
                omp=args.omp,
                timeout_s=args.timeout_s,
                faceon_times=faceon_times,
                faceon_bins=args.faceon_bins,
            )
            ev["ic_metrics"] = recon_met
            arms["fft_recon"] = ev
            print(
                f"    evolve ok={ev.get('ok')} A₂ {recon_met['a2']:.3f}→"
                f"{(ev.get('a2_t') or [float('nan')])[-1]:.3f} "
                f"COM={ev.get('com_drift_kpc')} wall={ev.get('wall_s')}",
                flush=True,
            )

        # --- θ-nearest library (no native θ-cond) ---
        print("  arm: theta_nearest (quiet pool)", flush=True)
        gen, gen_meta = _theta_nearest_sample(
            lib,
            theta,
            cov,
            prefer_quiet=True,
            quiet_ceil=args.quiet_ceil,
            k=args.theta_knn,
            rng=rng,
            n_resample=n_tot,
        )
        gen = _ensure_eps(gen)
        gen_met = _metrics(gen)
        print(
            f"    gen A₂={gen_met['a2']:.4f} nn={[n['run_hash'][:8] for n in gen_meta['nn']]} "
            f"dθ={gen_meta['theta_dists']}",
            flush=True,
        )
        if args.skip_evolve:
            arms["theta_nearest"] = {
                "ok": True,
                "a2_t": [gen_met["a2"]],
                "t_gyr": [0.0],
                "com_norm_t": [0.0],
                "com_drift_kpc": 0.0,
                "faceon_maps": [_faceon(gen, n_bins=args.faceon_bins)],
                "faceon_t_gyr": [0.0],
                "ic_metrics": gen_met,
                "sample_meta": gen_meta,
            }
        else:
            ev = _evolve_tracked(
                gen,
                end_gyr=args.evolve_gyr,
                dt=args.dt,
                omp=args.omp,
                timeout_s=args.timeout_s,
                faceon_times=faceon_times,
                faceon_bins=args.faceon_bins,
            )
            ev["ic_metrics"] = gen_met
            ev["sample_meta"] = gen_meta
            arms["theta_nearest"] = ev
            print(
                f"    evolve ok={ev.get('ok')} A₂ {gen_met['a2']:.3f}→"
                f"{(ev.get('a2_t') or [float('nan')])[-1]:.3f} "
                f"COM={ev.get('com_drift_kpc')} wall={ev.get('wall_s')}",
                flush=True,
            )

        case_dir = args.out / spec["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "name": spec["name"],
            "run_hash": h,
            "note": spec["note"],
            "ood_how": spec["ood_how"],
            "theta": theta,
            "dmin_to_train": dmin,
            "outside_train_box": outside,
            "n_disk": args.n_disk,
            "n_total": n_tot,
            "evolve_gyr": args.evolve_gyr,
            "dt": args.dt,
            "teacher": str(args.teacher),
            "arms": arms,
            "limit_theta_cond": (
                "TeacherFeatureLibrary has no θ API; used θ-nearest quiet retrieve"
            ),
        }
        # per-case figures
        _plot_case_faceon(
            case_dir / "faceon_pre_post.png",
            arms,
            title=f"OOD {spec['name']} (dmin={dmin:.2f})",
        )
        _plot_a2_t(
            case_dir / "a2_t.png",
            {"dt": args.dt, "evolve_gyr": args.evolve_gyr, "arms": arms},
        )
        # COM panel
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        for tag, row in arms.items():
            if not row.get("ok"):
                continue
            ax.plot(row["t_gyr"], row["com_norm_t"], lw=1.6, label=tag)
        ax.set_xlabel(r"$t$ [Gyr]")
        ax.set_ylabel(r"$|\mathrm{COM}|$ [kpc]")
        ax.set_title(f"COM {spec['name']}")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(case_dir / "com_t.png", dpi=140)
        plt.close(fig)

        (case_dir / "verdict.json").write_text(json.dumps(_jsonable(report), indent=2))
        case_reports.append(report)

        # paper per-case face-on
        if args.paper_figures is not None:
            args.paper_figures.mkdir(parents=True, exist_ok=True)
            src = case_dir / "faceon_pre_post.png"
            dst = args.paper_figures / f"fig_ood_theta_{spec['name']}_faceon.png"
            shutil.copy2(src, dst)
            shutil.copy2(case_dir / "a2_t.png", args.paper_figures / f"fig_ood_theta_{spec['name']}_a2.png")

    # Gallery + band plots
    _plot_gallery(args.out / "faceon_gallery_t0.png", case_reports)
    _plot_a2_bands(args.out / "a2_t_all.png", case_reports)
    if args.paper_figures is not None:
        shutil.copy2(args.out / "faceon_gallery_t0.png", args.paper_figures / "fig_ood_theta_gallery.png")
        shutil.copy2(args.out / "a2_t_all.png", args.paper_figures / "fig_ood_theta_a2_t.png")

    # Verdict summary
    def _arm_sum(row):
        if not row.get("ok"):
            return {"ok": False, "error": row.get("error")}
        a2 = row.get("a2_t") or []
        return {
            "ok": True,
            "a2_pre": a2[0] if a2 else None,
            "a2_post": a2[-1] if a2 else None,
            "com_drift_kpc": row.get("com_drift_kpc"),
            "wall_s": row.get("wall_s"),
        }

    summaries = []
    recon_ok = True
    gen_weak = True
    for c in case_reports:
        s = {
            "name": c["name"],
            "run_hash": c["run_hash"],
            "dmin_to_train": c["dmin_to_train"],
            "outside_train_box": c["outside_train_box"],
            "ood_how": c["ood_how"],
            "theta": {k: c["theta"][k] for k in THETA_DIST_KEYS},
            "arms": {k: _arm_sum(v) for k, v in c["arms"].items()},
        }
        # failure notes
        notes = []
        g = c["arms"].get("theta_nearest") or {}
        if g.get("ok") and (g.get("a2_t") or [0])[0] > 0.15:
            notes.append("theta_nearest not quiet (library retrieved mid/bar morphology)")
        r = c["arms"].get("fft_recon") or {}
        ic = c["arms"].get("galactics_ic") or {}
        if r.get("ok") and ic.get("ok"):
            da2 = abs((r.get("a2_t") or [0])[0] - (ic.get("a2_t") or [0])[0])
            if da2 > 0.05:
                notes.append(f"recon A₂ mismatch vs IC Δ={da2:.3f}")
                recon_ok = False
            # COM
            if (r.get("com_drift_kpc") or 0) > 0.05 or (ic.get("com_drift_kpc") or 0) > 0.05:
                notes.append("COM drift > 0.05 kpc")
        if g.get("ok") and ic.get("ok"):
            # generative θ-cond should track quiet IC A2 if retrieve worked
            if abs((g.get("a2_t") or [0])[0] - (ic.get("a2_t") or [0])[0]) < 0.05:
                gen_weak = False
        s["notes"] = notes
        summaries.append(s)

    verdict = {
        "teacher": str(args.teacher),
        "train_n_models": cov["n_models"],
        "train_box": {"lo": cov["box_lo"], "hi": cov["box_hi"]},
        "n_disk": args.n_disk,
        "n_total": n_tot,
        "evolve_gyr": args.evolve_gyr,
        "library_theta_conditioning": False,
        "library_fallback": "θ-nearest quiet retrieve (k={})".format(args.theta_knn),
        "cases": summaries,
        "verdict": {
            "recon_ood": "OK" if recon_ok else "DEGRADED",
            "generative_theta_cond": (
                "WEAK — no native θ-cond; θ-nearest retrieve only"
                if gen_weak
                else "PARTIAL — θ-nearest quiet retrieve matches IC A₂"
            ),
            "summary": (
                "FFT teacher recon of held-out GalactICS ICs at combinatorial-OOD θ "
                "preserves quiet morphology when OK; generative path cannot invent "
                "θ-specific structure — library is morphology-indexed, not θ-conditioned."
            ),
        },
    }
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))

    lines = [
        "# OOD θ compare",
        "",
        f"Teacher: `{args.teacher}`",
        f"Train models: **{cov['n_models']}** (manifest `{args.manifest}`).",
        f"Settings: disk$={args.n_disk:,}$, total$={n_tot:,}$, "
        f"$t_{{\\rm end}}={args.evolve_gyr}$ Gyr, $dt={args.dt}$, OMP={args.omp}.",
        "",
        "## Limit",
        "",
        "`TeacherFeatureLibrary` has **no** θ-conditioning API. "
        "Generative arm = θ-nearest quiet retrieve (k="
        f"{args.theta_knn}).",
        "",
        "## Cases",
        "",
        "| Case | run | dmin | M | Rd | zd | Q | IC A₂→ | recon A₂→ | θ-nn A₂→ | notes |",
        "|------|-----|------|---|----|----|---|---------|-----------|----------|-------|",
    ]
    for s in summaries:
        th = s["theta"]
        def arrow(arm):
            a = s["arms"].get(arm) or {}
            if not a.get("ok"):
                return "fail"
            return f"{a['a2_pre']:.3f}→{a['a2_post']:.3f}"
        lines.append(
            f"| `{s['name']}` | `{s['run_hash'][:8]}` | {s['dmin_to_train']:.2f} | "
            f"{th['disk.mass']:.0f} | {th['disk.scale_length']} | {th['disk.scale_height']} | "
            f"{th['disk_kinematics.toomre_q_target']} | {arrow('galactics_ic')} | "
            f"{arrow('fft_recon')} | {arrow('theta_nearest')} | "
            f"{'; '.join(s['notes']) or '—'} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- Recon OOD: **{verdict['verdict']['recon_ood']}**",
        f"- Generative θ-cond: **{verdict['verdict']['generative_theta_cond']}**",
        "",
        verdict["verdict"]["summary"],
        "",
        f"Figures: `{args.out}/faceon_gallery_t0.png`, `{args.out}/a2_t_all.png`, "
        "per-case dirs; paper `fig_ood_theta_*`.",
    ]
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
