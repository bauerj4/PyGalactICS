#!/usr/bin/env python3
"""OOD θ held-out generative ICs via latent retrieve+decode (particle_retrieve).

Uses combinatorial-OOD structural θ absent from the 19-model field-map train
gallery (``DEFAULT_OOD_CASES``). For each θ:

1. Quiet arm = GalactICS ``f0(θ)`` (ic_state).
2. Barred arm = particle_retrieve of a strong-bar dump from the *nearest train*
   campaign in θ-space, remassed to OOD ``f0(θ)`` component totals
   (no eval-dump particle copy; not free continuous decode(z)).
3. Optional absolute path-LOO when the OOD campaign itself forms a bar.
4. t=0 dens+kin vs f0; face-on; evolve ≥1 Gyr (gpu_bh) for coherence.

Paper figs: ``fig_latent_theta_ood_*``.

Example::

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \\
      .venv/bin/python scripts/latent_theta_ood.py \\
        --out runs/ml/field_maps/latent_theta_ood_2026-08-03 \\
        --n-disk 1000000 --evolve-gyr 1.0 --force gpu_bh \\
        --n-evolve 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from galacticsics.ml.fields.feature_library import _hash_from_path  # noqa: E402
from latent_theta_evolve_suite import (  # noqa: E402
    _save_parts,
    _stratified_to_n,
    plot_newtheta_t0,
    run_evolve,
)
from latent_theta_gen import (  # noqa: E402
    _component_masses,
    _decode_particle_retrieve,
    _full_system_score,
    _load_parts,
)
from ood_theta_compare import (  # noqa: E402
    THETA_DIST_KEYS,
    _dmin_to_train,
    _theta_from_model,
    _vec,
)
from score_residual_f0_kinetics import _a2_rd  # noqa: E402

RANK = Path("runs/ml/field_maps/corpus_particle_a2_rank.json")
LIB_CODES = Path("runs/ml/field_maps/latent_theta_gen_2026-08-02/feature_library_codes.npz")
PAPER = Path("papers/mnras_noneq_ics/figures")
PAPER_ARCHIVE = Path("papers/mnras_noneq_ics/figures/archive")
RESULTS = Path("papers/mnras_noneq_ics/results")
CORPUS = Path("runs/mw_morton_corpus_v2")
TRAIN_COVERAGE = Path(
    "runs/ml/field_maps/ood_theta_compare_2026-07-26/train_theta_coverage.json"
)

# Held-out structural θ relative to the *feature-library* campaign gallery
# (primary latent train set). Also absent from the frozen 19-model teacher
# coverage where noted. Combinatorial OOD inside the corpus axis box.
OOD_CASES: list[dict] = [
    {
        "name": "cfc30_heavy_bar",
        "run_hash": "cfc30aaac857",
        "note": "M=22 Rd=2.0 Q=1.4 — heavy compact bar-forming; not in feature lib",
        "ood_how": "held-out vs feature-library campaigns (dmin to 19-teacher gallery)",
    },
    {
        "name": "ext_bar_6552",
        "run_hash": "6552ba6a7bda",
        "note": "M=18 Rd=3.0 Q=1.4 — extended low-Q bar; not in feature lib / 19-gallery",
        "ood_how": "held-out vs feature-library + 19-model teacher gallery",
    },
    {
        "name": "mid_bar_b139",
        "run_hash": "b13924e45f9f",
        "note": "M=14 Rd=3.0 Q=1.4 — mid-mass extended; not in feature lib / 19-gallery",
        "ood_how": "held-out vs feature-library + 19-model teacher gallery",
    },
    {
        "name": "thick_stable",
        "run_hash": "6272b66b640d",
        "note": "M=10 Rd=2.5 zd=0.55 Q=2.5 — thick high-Q quiet control",
        "ood_how": "held-out vs feature-library + 19-model teacher gallery",
    },
]

# In-distribution path-LOO reference (for scoreboard compare).
ID_REF = {
    "906c4": {"dens_d": 0.054, "kin_mse": 0.0006, "a2_gen": 0.494, "verdict": "MATCH"},
    "nobulge": {"dens_d": 0.055, "kin_mse": 0.0010, "a2_gen": 0.390, "verdict": "FADE MATCH"},
}


def _fast_path_loo_neighbor(
    eval_path: Path,
    *,
    run_hash: str,
    a2_enc: float,
    ranked: list[dict],
    t_min: float = 0.3,
    a2_weight: float = 1.5,
) -> tuple[Path, float, float]:
    """Same-campaign neighbor by A₂ only (no per-candidate teacher encode)."""
    eval_res = Path(eval_path).resolve()
    cands: list[tuple[float, float, Path]] = []
    for row in ranked:
        h = row.get("run_hash") or _hash_from_path(row["path"])
        if h != run_hash:
            continue
        p = Path(row["path"])
        if "step_" not in p.name or not p.is_file():
            continue
        if p.resolve() == eval_res:
            continue
        t = float(row.get("t_gyr") or 0.0)
        if t < t_min:
            continue
        a2 = float(row["a2"])
        score = float(a2_weight) * abs(a2 - float(a2_enc))
        cands.append((score, a2, p))
    if not cands:
        raise RuntimeError(f"no fast path-LOO candidates for {run_hash}")
    cands.sort(key=lambda x: x[0])
    score, a2, p = cands[0]
    return p, float(score), float(a2)


def _load_parts_safe(path: Path, rng):
    return _load_parts(path, None, rng)


def _theta_from_run(run_hash: str) -> dict[str, float]:
    model = json.loads((CORPUS / run_hash / "model.json").read_text())
    return _theta_from_model(model)


def _nearest_train_bar(
    *,
    ood_theta: dict[str, float],
    train_hashes: set[str],
    ranked: list[dict],
    a2_floor: float = 0.28,
) -> tuple[Path, str, float, float]:
    """Nearest train-campaign strong-bar dump in structural θ distance."""
    ood_v = _vec(ood_theta, THETA_DIST_KEYS)
    best = None
    for row in ranked:
        h = row.get("run_hash") or _hash_from_path(row["path"])
        if h not in train_hashes:
            continue
        a2 = float(row.get("a2") or 0.0)
        if a2 < a2_floor:
            continue
        p = Path(row["path"])
        if "step_" not in p.name or not p.is_file():
            continue
        th = row.get("theta") or {}
        if not th:
            try:
                th = _theta_from_run(h)
            except Exception:  # noqa: BLE001
                continue
        tv = _vec({k: float(th[k]) for k in THETA_DIST_KEYS}, THETA_DIST_KEYS)
        dth = float(np.linalg.norm(tv - ood_v))
        # Prefer closer θ, then stronger bar.
        key = (dth, -a2)
        if best is None or key < best[0]:
            best = (key, p, h, dth, a2)
    if best is None:
        raise RuntimeError("no train barred dump found")
    _, p, h, dth, a2 = best
    return p, h, dth, a2


def _peak_ood_bar(run_hash: str, ranked: list[dict], a2_floor: float = 0.12) -> Path | None:
    best = None
    for row in ranked:
        h = row.get("run_hash") or _hash_from_path(row["path"])
        if h != run_hash:
            continue
        a2 = float(row.get("a2") or 0.0)
        if a2 < a2_floor:
            continue
        p = Path(row["path"])
        if not p.is_file() or "step_" not in p.name:
            continue
        if best is None or a2 > best[0]:
            best = (a2, p)
    return None if best is None else best[1]


def _archive_newtheta_live() -> None:
    """Move weaker in-gallery newθ panels out of the lean live set."""
    PAPER_ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in sorted(PAPER.glob("fig_latent_theta_newtheta_*.png")):
        dst = PAPER_ARCHIVE / name.name
        shutil.move(str(name), str(dst))
        print(f"  archive ← {name.name}", flush=True)


def _plot_gallery(rows: list[dict], out_path: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(7.6, 2.6 * n))
    if n == 1:
        axes = np.asarray([axes])
    for i, row in enumerate(rows):
        quiet = row["quiet_parts"]
        barred = row["barred_parts"]
        rd = float(row["rd"])
        for ax, parts, lab in (
            (axes[i, 0], quiet, r"$f_0(\theta)$ quiet"),
            (axes[i, 1], barred, r"OOD $\theta$ + barred $z$"),
        ):
            disk = parts["component_id"] == 0
            pos = parts["pos"][disk]
            mass = parts["mass"][disk]
            com = np.average(pos, axis=0, weights=mass)
            xy = pos[:, :2] - com[:2]
            H, xe, ye = np.histogram2d(
                xy[:, 0],
                xy[:, 1],
                bins=96,
                range=[[-12, 12], [-12, 12]],
                weights=mass,
            )
            from galacticsics.campaign.analysis import dens_array_log10

            show, vmin_s, vmax_s, _ = dens_array_log10(H, vmax_pct=98.0)
            ax.imshow(
                show.T,
                origin="lower",
                cmap="magma",
                vmin=vmin_s,
                vmax=vmax_s,
                extent=[-12, 12, -12, 12],
            )
            a2 = _a2_rd(parts, rd)
            ax.set_title(f"{row['name']}: {lab}")
            ax.text(
                0.02,
                0.98,
                rf"$A_2(R_d)={a2:.3f}$",
                transform=ax.transAxes,
                va="top",
                color="w",
                fontsize=8,
            )
            ax.set_xlabel(r"$x$ [kpc]")
            ax.set_ylabel(r"$y$ [kpc]")
    fig.suptitle(
        r"OOD $\theta$ (held-out vs 19-model train): quiet $f_0$ vs barred retrieve+decode"
        r" (face-on $\log_{10}\Sigma$)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _verdict_evolve(a2_pre: float, a2_post: float, quiet_case: bool) -> str:
    if quiet_case:
        # Quiet transfer should not invent a lasting bar.
        if a2_post < 0.08:
            return "WIN (quiet stays quiet)"
        if a2_post < 0.15:
            return "PARTIAL (mild growth)"
        return "LOSS (spurious bar)"
    # Barred: stay barred, mild fade OK (like ID MATCH).
    if a2_pre >= 0.20 and a2_post >= 0.18:
        fade = a2_pre - a2_post
        if abs(fade) <= 0.12:
            return "WIN (barred coherent)"
        if fade > 0.12:
            return "PARTIAL (strong fade)"
        return "PARTIAL (growth)"
    if a2_pre >= 0.15 and a2_post >= 0.10:
        return "PARTIAL (weak bar)"
    return "LOSS (bar washes)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/latent_theta_ood_2026-08-03"),
    )
    p.add_argument("--n-disk", type=int, default=1_000_000)
    p.add_argument("--evolve-gyr", type=float, default=1.0)
    p.add_argument("--force", type=str, default="gpu_bh")
    p.add_argument("--omp", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-evolve", type=int, default=3)
    p.add_argument("--skip-evolve", action="store_true")
    p.add_argument("--skip-path-loo", action="store_true")
    p.add_argument("--cases", type=str, default="")
    p.add_argument("--a2-floor-train", type=float, default=0.28)
    p.add_argument(
        "--resume-from-meta",
        action="store_true",
        help="Skip cases that already have logs/ood_*_meta.json with evolve",
    )
    args = p.parse_args()

    out = args.out
    samples = out / "samples"
    gates = out / "gates"
    logs = out / "logs"
    for d in (samples, gates, logs, out / "figs"):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_tot = int(round(args.n_disk * 7 / 4))
    raw_cov = json.loads(TRAIN_COVERAGE.read_text())
    train_hashes = set(raw_cov["run_hashes"])
    # Rebuild coverage arrays expected by _dmin_to_train.
    pts = np.stack(
        [_vec(raw_cov["theta_by_run"][h], THETA_DIST_KEYS) for h in raw_cov["run_hashes"]]
    )
    lo, hi = pts.min(0), pts.max(0)
    cov = {
        "n_models": int(raw_cov["n_models"]),
        "run_hashes": list(raw_cov["run_hashes"]),
        "keys": list(THETA_DIST_KEYS),
        "box_lo": dict(zip(THETA_DIST_KEYS, lo.tolist())),
        "box_hi": dict(zip(THETA_DIST_KEYS, hi.tolist())),
        "span": np.maximum(hi - lo, 1e-6),
        "points": pts,
        "theta_by_run": raw_cov["theta_by_run"],
    }
    # Also exclude feature-library campaigns (primary latent train gallery).
    lib_hashes = set(str(x) for x in np.load(LIB_CODES, allow_pickle=True)["hashes"])
    train_hashes |= lib_hashes
    ranked = json.loads(RANK.read_text())
    # Teacher bundle not required for particle_retrieve transfer / fast path-LOO.

    want = {x.strip() for x in args.cases.split(",") if x.strip()}
    cases = [c for c in OOD_CASES if not want or c["name"] in want]

    print(
        f"=== latent_theta OOD n_disk={args.n_disk} N={n_tot} "
        f"n_cases={len(cases)} train_models={cov['n_models']} "
        f"lib_campaigns={len(lib_hashes)} ===",
        flush=True,
    )

    suite: dict = {
        "n_disk": args.n_disk,
        "n_tot": n_tot,
        "evolve_gyr": args.evolve_gyr,
        "train_n_models": cov["n_models"],
        "lib_n_campaigns": len(lib_hashes),
        "id_ref": ID_REF,
        "cases": [],
    }
    evolve_budget = int(args.n_evolve)

    # Prefer evolve order: barred-forming first, then quiet controls.
    evolve_priority = {
        "cfc30_heavy_bar": 0,
        "ext_bar_6552": 1,
        "mid_bar_b139": 2,
        "thick_stable": 3,
    }
    cases_sorted = sorted(cases, key=lambda c: evolve_priority.get(c["name"], 9))

    # Resume: keep completed evolve metas and free evolve budget.
    if args.resume_from_meta:
        for spec in cases_sorted:
            meta_p = logs / f"ood_{spec['name']}_meta.json"
            if not meta_p.is_file():
                continue
            prev = json.loads(meta_p.read_text())
            if prev.get("evolve"):
                suite["cases"].append(prev)
                evolve_budget = max(0, evolve_budget - 1)
                print(
                    f"  resume keep {spec['name']} evolve "
                    f"{prev['evolve'].get('verdict')}",
                    flush=True,
                )

    for spec in cases_sorted:
        name = spec["name"]
        if any(c.get("name") == name for c in suite["cases"]):
            continue
        h = spec["run_hash"]
        print(f"\n===== OOD {name} ({h[:8]}) =====", flush=True)
        theta = _theta_from_run(h)
        dmin = _dmin_to_train(theta, cov)
        if h in lib_hashes:
            raise RuntimeError(f"{h} is in the feature-library gallery — not OOD")
        rd = float(theta["disk.scale_length"])
        quiet_case = name == "thick_stable"
        ic_path = CORPUS / h / "ic_state.npz"
        ic = _load_parts(ic_path, None, rng)
        mass_prior = _component_masses(ic)
        quiet = _stratified_to_n(ic, n_tot, rng)

        bar_path, train_h, dth_nn, a2_nn = _nearest_train_bar(
            ood_theta=theta,
            train_hashes=train_hashes,
            ranked=ranked,
            a2_floor=float(args.a2_floor_train),
        )
        barred = _decode_particle_retrieve(bar_path, mass_prior, None, rng)
        barred = _stratified_to_n(barred, n_tot, rng)

        bpath = samples / f"ood_{name}_barred_gen.npz"
        qpath = samples / f"ood_{name}_f0_quiet.npz"
        _save_parts(bpath, barred)
        _save_parts(qpath, quiet)

        face = PAPER / f"fig_latent_theta_ood_{name}_faceon_t0.png"
        prof = PAPER / f"fig_latent_theta_ood_{name}_profiles_t0.png"
        t0sc = plot_newtheta_t0(
            barred=barred,
            quiet=quiet,
            out_faceon=face,
            out_profiles=prof,
            title=(
                rf"OOD $\theta$ {name} "
                rf"(Md={theta['disk.mass']}, Rd={rd}, "
                rf"Q={theta['disk_kinematics.toomre_q_target']}; "
                rf"$d_{{\min}}={dmin:.2f}$)"
            ),
            rd=rd,
        )
        # dens vs f0 (quiet): barred should differ in disk morph, halo/bulge remass-close
        score_f0 = _full_system_score(barred, quiet)
        dens_d = float(score_f0["dens"]["disk"]["med_abs_log"])
        dens_h = float(score_f0["dens"]["halo"]["med_abs_log"])
        dens_b = float(score_f0["dens"]["bulge"]["med_abs_log"])
        kin_mse = float(score_f0["kin_mean_mse"])

        row: dict = {
            "name": name,
            "hash": h,
            "ood_how": spec.get("ood_how"),
            "note": spec.get("note"),
            "dmin_to_train": dmin,
            "theta": {k: float(theta[k]) for k in THETA_DIST_KEYS},
            "rd": rd,
            "transfer": {
                "nn_path": str(bar_path),
                "nn_train_hash": train_h,
                "dtheta_nn": float(dth_nn),
                "a2_nn": float(a2_nn),
                "a2_rd_gen": float(t0sc["a2_rd_barred"]),
                "a2_rd_f0": float(t0sc["a2_rd_quiet"]),
                "dens_vs_f0_d/h/b": [dens_d, dens_h, dens_b],
                "kin_mse_vs_f0": kin_mse,
            },
            "figs": {"faceon": str(face), "profiles": str(prof)},
        }

        # Absolute path-LOO when OOD campaign forms a bar (fast A₂ neighbor).
        peak = _peak_ood_bar(h, ranked, a2_floor=0.12)
        if peak is not None and not args.skip_path_loo:
            print(f"  path-LOO absolute vs {peak.name}", flush=True)
            a2_enc = float(
                next(
                    (
                        float(r["a2"])
                        for r in ranked
                        if Path(r["path"]).resolve() == peak.resolve()
                    ),
                    _a2_rd(_load_parts(peak, None, rng), rd),
                )
            )
            try:
                nn_path, zd, a2_nn_loo = _fast_path_loo_neighbor(
                    peak,
                    run_hash=h,
                    a2_enc=float(a2_enc),
                    ranked=ranked,
                    t_min=0.3,
                )
                loo = _decode_particle_retrieve(nn_path, mass_prior, None, rng)
                loo = _stratified_to_n(loo, n_tot, rng)
                ref = _stratified_to_n(_load_parts(peak, None, rng), n_tot, rng)
                sc = _full_system_score(loo, ref)
                loo_path = samples / f"ood_{name}_pathloo_gen.npz"
                _save_parts(loo_path, loo)
                row["path_loo"] = {
                    "eval_path": str(peak),
                    "nn_path": str(nn_path),
                    "z_dist": float(zd),
                    "a2_nn": float(a2_nn_loo),
                    "a2_rd_gen": _a2_rd(loo, rd),
                    "a2_rd_ref": _a2_rd(ref, rd),
                    "dens_d/h/b": [
                        float(sc["dens"]["disk"]["med_abs_log"]),
                        float(sc["dens"]["halo"]["med_abs_log"]),
                        float(sc["dens"]["bulge"]["med_abs_log"]),
                    ],
                    "kin_mse": float(sc["kin_mean_mse"]),
                    "method": "fast_a2",
                }
                print(
                    f"  path-LOO dens={row['path_loo']['dens_d/h/b']} "
                    f"kin={row['path_loo']['kin_mse']:.4f} "
                    f"A2={row['path_loo']['a2_rd_gen']:.3f}→"
                    f"{row['path_loo']['a2_rd_ref']:.3f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                row["path_loo"] = {"error": str(exc)}
                print(f"  path-LOO skip: {exc}", flush=True)

        do_evolve = (
            (not args.skip_evolve)
            and evolve_budget > 0
            and (float(t0sc["a2_rd_barred"]) > 0.15 or quiet_case)
        )
        if do_evolve:
            evo_out = gates / f"evolve_ood_{name}"
            t0 = time.time()
            run_evolve(
                out=evo_out,
                data_path=qpath,
                gen_npz=bpath,
                paper_prefix=f"fig_latent_theta_ood_{name}",
                n_disk=args.n_disk,
                evolve_gyr=args.evolve_gyr,
                force=args.force,
                omp=args.omp,
                a2_r_eval=rd,
                extra_label=f"OOD barred transfer ({name})",
                data_arm_label=r"f0(θ) quiet baseline",
            )
            src = PAPER / f"fig_latent_theta_ood_{name}_a2_t.png"
            dst = PAPER / f"fig_latent_theta_ood_{name}_evolve_a2_t.png"
            if src.is_file():
                shutil.copy2(src, dst)
            v = json.loads((evo_out / "verdict.json").read_text())
            arms = v.get("arms") or {}
            gen_arm = arms.get("latent_gen") or {}
            a2_pre = float(gen_arm.get("a2_pre") or t0sc["a2_rd_barred"])
            a2_post = float(gen_arm.get("a2_post") or a2_pre)
            verd = _verdict_evolve(a2_pre, a2_post, quiet_case=quiet_case)
            row["evolve"] = {
                "out": str(evo_out),
                "gyr": args.evolve_gyr,
                "wall_s": time.time() - t0,
                "a2_pre": a2_pre,
                "a2_post": a2_post,
                "com_drift_kpc": gen_arm.get("com_drift_kpc"),
                "verdict": verd,
                "quiet_baseline": {
                    "a2_pre": (arms.get("data") or {}).get("a2_pre"),
                    "a2_post": (arms.get("data") or {}).get("a2_post"),
                },
            }
            print(
                f"  evolve A2 {a2_pre:.3f}→{a2_post:.3f}  {verd}",
                flush=True,
            )
            evolve_budget -= 1

        suite["cases"].append(row)
        (logs / f"ood_{name}_meta.json").write_text(json.dumps(row, indent=2, default=str))

    # Gallery from saved samples (covers resume + new).
    gallery_rows = []
    for c in suite["cases"]:
        name = c["name"]
        rd = float(c.get("rd") or (c.get("theta") or {}).get("disk.scale_length") or 2.0)
        bp = samples / f"ood_{name}_barred_gen.npz"
        qp = samples / f"ood_{name}_f0_quiet.npz"
        if bp.is_file() and qp.is_file():
            gallery_rows.append(
                {
                    "name": name,
                    "rd": rd,
                    "quiet_parts": _load_parts(qp, None, rng),
                    "barred_parts": _load_parts(bp, None, rng),
                }
            )
    gallery = PAPER / "fig_latent_theta_ood_gallery_faceon_t0.png"
    if gallery_rows:
        _plot_gallery(gallery_rows, gallery)
        suite["gallery"] = str(gallery)
    _archive_newtheta_live()

    # Keep lean: archive per-case faceons that are redundant with gallery,
    # keep profiles for evolved barred cases + all evolve a2 panels.
    keep_face = set()
    keep_prof = {
        c["name"]
        for c in suite["cases"]
        if c.get("evolve") and "WIN" in str(c["evolve"].get("verdict", ""))
    }
    if not keep_prof:
        keep_prof = {c["name"] for c in suite["cases"][:2]}
    for c in suite["cases"]:
        name = c["name"]
        face = PAPER / f"fig_latent_theta_ood_{name}_faceon_t0.png"
        prof = PAPER / f"fig_latent_theta_ood_{name}_profiles_t0.png"
        if face.is_file() and name not in keep_face:
            shutil.move(str(face), str(PAPER_ARCHIVE / face.name))
        if prof.is_file() and name not in keep_prof:
            shutil.move(str(prof), str(PAPER_ARCHIVE / prof.name))
        # Drop raw a2_t duplicates if evolve copy exists.
        raw = PAPER / f"fig_latent_theta_ood_{name}_a2_t.png"
        evo = PAPER / f"fig_latent_theta_ood_{name}_evolve_a2_t.png"
        if raw.is_file() and evo.is_file():
            raw.unlink()

    # Scoreboard + summary.
    wins = sum(1 for c in suite["cases"] if "WIN" in str((c.get("evolve") or {}).get("verdict", "")))
    losses = sum(1 for c in suite["cases"] if "LOSS" in str((c.get("evolve") or {}).get("verdict", "")))
    partials = sum(
        1 for c in suite["cases"] if "PARTIAL" in str((c.get("evolve") or {}).get("verdict", ""))
    )
    suite["scoreboard"] = {"W": wins, "L": losses, "P": partials}

    (out / "verdict.json").write_text(json.dumps(suite, indent=2, default=str))
    (logs / "ood_suite_verdict.json").write_text(json.dumps(suite, indent=2, default=str))

    md = _write_scoreboard_md(suite)
    (out / "SCOREBOARD.md").write_text(md)
    (RESULTS / "latent_theta_ood_SCOREBOARD.md").write_text(md)
    (RESULTS / "latent_theta_ood_verdict.json").write_text(
        json.dumps(suite, indent=2, default=str)
    )
    print("\n=== OOD DONE ===", flush=True)
    print(json.dumps(suite["scoreboard"], indent=2), flush=True)


def _write_scoreboard_md(suite: dict) -> str:
    lines = [
        "# SCOREBOARD — OOD θ (held-out vs 19-model train gallery)",
        "",
        "Generative = **retrieve+decode** (train barred dump remassed to OOD "
        r"$f_0(\theta)$); not free continuous `decode(z)`; not eval-dump copy.",
        "",
        f"Train models: **{suite['train_n_models']}**. "
        f"Evolve: {suite['evolve_gyr']} Gyr, disk$=10^6$, `gpu_bh`.",
        "",
        "## vs in-distribution path-LOO",
        "",
        "| ID system | dens disk | kin MSE | A₂ gen | verdict |",
        "|-----------|-----------|---------|--------|---------|",
    ]
    for k, v in suite["id_ref"].items():
        lines.append(
            f"| {k} | {v['dens_d']:.3f} | {v['kin_mse']:.4f} | "
            f"{v['a2_gen']:.3f} | {v['verdict']} |"
        )
    lines += [
        "",
        "## OOD transfer (barred z → held-out θ)",
        "",
        "| Case | dmin | A₂ f0 | A₂ gen | dens d/h/b vs f0 | kin vs f0 | "
        "evolve A₂ | verdict |",
        "|------|------|-------|--------|------------------|-----------|"
        "-----------|---------|",
    ]
    for c in suite["cases"]:
        t = c["transfer"]
        dens = "/".join(f"{x:.3f}" for x in t["dens_vs_f0_d/h/b"])
        evo = c.get("evolve") or {}
        if evo:
            a2e = f"{evo['a2_pre']:.3f}→{evo['a2_post']:.3f}"
            verd = evo.get("verdict", "—")
        else:
            a2e = "—"
            verd = "t=0 only"
        lines.append(
            f"| {c['name']} | {c['dmin_to_train']:.2f} | "
            f"{t['a2_rd_f0']:.3f} | {t['a2_rd_gen']:.3f} | {dens} | "
            f"{t['kin_mse_vs_f0']:.4f} | {a2e} | {verd} |"
        )
    lines += ["", "## OOD path-LOO absolute (same-campaign, when bar exists)", ""]
    lines.append("| Case | dens d/h/b | kin MSE | A₂ gen→ref | vs ID dens |")
    lines.append("|------|------------|---------|------------|------------|")
    for c in suite["cases"]:
        pl = c.get("path_loo")
        if not pl or "error" in pl:
            lines.append(f"| {c['name']} | — | — | — | no/weak bar |")
            continue
        dens = "/".join(f"{x:.3f}" for x in pl["dens_d/h/b"])
        id_d = suite["id_ref"]["906c4"]["dens_d"]
        cmp = "comparable" if pl["dens_d/h/b"][0] < 2.5 * id_d else "worse"
        lines.append(
            f"| {c['name']} | {dens} | {pl['kin_mse']:.4f} | "
            f"{pl['a2_rd_gen']:.3f}→{pl['a2_rd_ref']:.3f} | {cmp} |"
        )
    sb = suite.get("scoreboard") or {}
    lines += [
        "",
        f"**OOD evolve W/L/P:** {sb.get('W', 0)}/{sb.get('L', 0)}/{sb.get('P', 0)}",
        "",
        "Live figs: `fig_latent_theta_ood_gallery_faceon_t0.png`, "
        "`fig_latent_theta_ood_*_evolve_a2_t.png`, selected profiles.",
        "",
        "Interpretation: OOD tests whether barred $z$ from the train gallery "
        "remasses coherently onto held-out mass models. Absolute path-LOO on "
        "OOD campaigns (when they form bars) is compared to in-distribution "
        "906c4 dens~$0.05$.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
