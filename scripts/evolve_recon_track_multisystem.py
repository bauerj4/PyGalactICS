#!/usr/bin/env python3
"""Multi-system data vs FFT-teacher recon evolve tracking.

Empirical support for Proposition A in
``docs/dynamical_consistency_charts.md``: for several barred + quiet
snapshots, dens-resampled FFT recon tracks the data dump under the same
``bh_c`` leapfrog (shared COM conventions).

No feature-library build — data + recon arms only.

Example::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \\
      .venv/bin/python scripts/evolve_recon_track_multisystem.py \\
        --teacher runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt \\
        --out runs/ml/field_maps/recon_track_multisystem_2026-07-26 \\
        --n-disk 500000 --evolve-gyr 0.50 --dt 0.01 --omp 2 \\
        --paper-figures papers/mnras_noneq_ics/figures
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sample_latent_ic import COUNT, RANK, default_teacher  # noqa: E402
from smoke_field_maps import _am_profiles  # noqa: E402

from evolve_resample_compare import (  # noqa: E402
    _evolve_tracked,
    _faceon,
    _full_com,
    _metrics,
    _recon_particles,
    _stratified_down,
    _subsample_dump,
    _vcom_only,
)
from galacticsics.ml.fields.binning import bin_multiscale_slice_stacks  # noqa: E402
from galacticsics.ml.fields.feature_library import (  # noqa: E402
    load_frozen_teacher_bundle,
    pool_bottleneck_z,
)
from galacticsics.ml.fields.frame import prepare_shared_frame  # noqa: E402
from galacticsics.ml.fields.normalize import denormalize_stack, normalize_stack  # noqa: E402
from galacticsics.ml.morton.polygon import _component_ids  # noqa: E402

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Default multi-system slate: strong bar, mid bars, mild bar, quiet IC.
DEFAULT_SYSTEMS = [
    {
        "name": "bar_strong_54a8",
        "kind": "barred",
        "run_hash": "54a8faf836a0",
        "path": "runs/mw_morton_corpus_v2/54a8faf836a0/evolution/particles/step_001800.npz",
    },
    {
        "name": "bar_mid_17bf",
        "kind": "barred",
        "run_hash": "17bfd6b7a435",
        "path": "runs/mw_morton_corpus_v2/17bfd6b7a435/evolution/particles/step_001600.npz",
    },
    {
        "name": "bar_mid_07b0",
        "kind": "barred",
        "run_hash": "07b06d08981d",
        "path": "runs/mw_morton_corpus_v2/07b06d08981d/evolution/particles/step_003900.npz",
    },
    {
        "name": "bar_mild_3064",
        "kind": "barred",
        "run_hash": "3064dfb8d222",
        "path": "runs/mw_morton_corpus_v2/3064dfb8d222/evolution/particles/step_000800.npz",
    },
    {
        "name": "quiet_906c",
        "kind": "quiet",
        "run_hash": "906c4af73543",
        "path": "runs/mw_morton_corpus_v2/906c4af73543/ic_state.npz",
    },
]


def _disk_dens_collapse(stack: np.ndarray, n_z: int, n_mom: int) -> np.ndarray:
    dens = np.stack(
        [np.maximum(stack[iz * n_mom], 0.0) for iz in range(n_z)], axis=0
    )
    return dens.sum(axis=0)


def _field_recon_mse(
    path: Path, teacher, cfg, stats
) -> dict[str, float]:
    """Normed dens MSE (and midplane dens MSE) data vs AE recon at t=0."""
    with np.load(path, allow_pickle=True) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        cid = _component_ids(data.get("tags"), data.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    maps = {k: v[0] for k, v in binned.items()}
    stacks = {
        k: torch.as_tensor(normalize_stack(v, stats[k])[None], dtype=torch.float32)
        for k, v in maps.items()
    }
    with torch.no_grad():
        pred = teacher(stacks)
        pred_np = {k: v.cpu().numpy()[0] for k, v in pred.items()}
    recon = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}

    mse_all = []
    out: dict[str, float] = {}
    for g in cfg.grids:
        name = g.name
        if name not in maps or name not in recon:
            continue
        n_mom = int(g.n_mom)  # per-slab moments (not n_z·n_mom total channels)
        true_d = _disk_dens_collapse(maps[name], g.n_z, n_mom)
        rec_d = _disk_dens_collapse(recon[name], g.n_z, n_mom)
        # Normed dens MSE on positive-mass cells relative to data scale.
        scale = float(np.mean(true_d**2) + 1e-12)
        mse = float(np.mean((true_d - rec_d) ** 2) / scale)
        out[f"mse_dens_{name}"] = mse
        mse_all.append(mse)
        if name == "disk":
            # Midplane dens channel only (first z-slab dens).
            t0 = np.maximum(maps[name][0], 0.0)
            r0 = np.maximum(recon[name][0], 0.0)
            out["mse_dens_disk_midplane"] = float(
                np.mean((t0 - r0) ** 2) / (float(np.mean(t0**2)) + 1e-12)
            )
    out["mse_dens_mean"] = float(np.mean(mse_all)) if mse_all else float("nan")
    return out


def _clump_score(dens: np.ndarray, *, n_peaks: int = 8, frac: float = 0.35) -> dict:
    """Simple midplane clump / multiplicity score from face-on dens.

    Peaks = local maxima above ``frac * max``; score = sum of top peak
    masses / total mass (high ⇒ concentrated cores). Multiplicity = count
    of such peaks.
    """
    d = np.asarray(dens, dtype=np.float64)
    if d.size == 0 or not np.isfinite(d).any():
        return {"multiplicity": 0, "peak_mass_frac": 0.0, "peakiness": 0.0}
    thr = float(frac * np.nanmax(d))
    # 3×3 local max
    pad = np.pad(d, 1, mode="constant")
    peaks = []
    for i in range(1, pad.shape[0] - 1):
        for j in range(1, pad.shape[1] - 1):
            block = pad[i - 1 : i + 2, j - 1 : j + 2]
            v = pad[i, j]
            if v >= thr and v >= np.max(block) - 1e-15:
                peaks.append(v)
    peaks = sorted(peaks, reverse=True)[:n_peaks]
    tot = float(np.sum(d) + 1e-12)
    peak_sum = float(sum(peaks))
    return {
        "multiplicity": int(len(peaks)),
        "peak_mass_frac": peak_sum / tot,
        "peakiness": float(np.nanmax(d) / (np.mean(d) + 1e-12)),
    }


def _encode_z(path: Path, teacher, cfg, stats, enc_grid: int = 4) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        cid = _component_ids(data.get("tags"), data.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    maps = {k: v[0] for k, v in binned.items()}
    stacks = {
        k: torch.as_tensor(normalize_stack(v, stats[k])[None], dtype=torch.float32)
        for k, v in maps.items()
    }
    with torch.no_grad():
        feat = teacher.encode_features(stacks)
        z = pool_bottleneck_z(feat, enc_grid=enc_grid)[0].cpu().numpy()
    return np.asarray(z, dtype=np.float64)


def _feature_drift_along_run(
    run_hash: str,
    teacher,
    cfg,
    stats,
    *,
    max_dumps: int = 8,
    enc_grid: int = 4,
) -> dict | None:
    """Encode successive dumps for one run; report relative ||z(t)-z0||."""
    evo = ROOT / "runs" / "mw_morton_corpus_v2" / run_hash / "evolution" / "particles"
    if not evo.is_dir():
        return None
    dumps = sorted(evo.glob("step_*.npz"))
    if len(dumps) < 2:
        return None
    # Evenly subsample.
    idx = np.linspace(0, len(dumps) - 1, num=min(max_dumps, len(dumps)), dtype=int)
    paths = [dumps[i] for i in idx]
    zs = []
    labels = []
    for p in paths:
        try:
            zs.append(_encode_z(p, teacher, cfg, stats, enc_grid=enc_grid))
            labels.append(p.name)
        except Exception as exc:  # noqa: BLE001
            print(f"  feature-drift skip {p.name}: {exc}", flush=True)
    if len(zs) < 2:
        return None
    z0 = zs[0]
    n0 = float(np.linalg.norm(z0) + 1e-12)
    rel = [float(np.linalg.norm(z - z0) / n0) for z in zs]
    return {
        "run_hash": run_hash,
        "dumps": labels,
        "rel_drift": rel,
        "rel_drift_end": rel[-1],
        "n_dumps": len(labels),
    }


def _plot_system_a2(path: Path, case: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for tag, style in (("data", "-"), ("recon", "--")):
        row = case["arms"].get(tag) or {}
        if not row.get("ok"):
            continue
        ax.plot(row["t_gyr"], row["a2_t"], style, lw=1.8, label=tag)
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"disk median $A_2$")
    ax.set_title(f"{case['name']} ({case['kind']})")
    ax.axhline(0.30, color="0.5", ls=":", lw=1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_system_faceon(path: Path, case: dict) -> None:
    tags = [t for t in ("data", "recon") if (case["arms"].get(t) or {}).get("ok")]
    if not tags:
        return
    # rows = arms; cols = t=0 and t=end face-on
    fig, axes = plt.subplots(len(tags), 2, figsize=(5.6, 2.4 * len(tags)))
    if len(tags) == 1:
        axes = np.asarray([axes])
    for i, tag in enumerate(tags):
        row = case["arms"][tag]
        maps = row.get("faceon_maps") or []
        if len(maps) < 2:
            continue
        for j, (img, lab) in enumerate(
            ((maps[0], r"$t=0$"), (maps[-1], r"$t_{\rm end}$"))
        ):
            ax = axes[i, j]
            ax.imshow(
                np.log10(np.maximum(img, 1e-12)).T,
                origin="lower",
                cmap="inferno",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            a2s = row.get("a2_t") or [None, None]
            a2 = a2s[0] if j == 0 else a2s[-1]
            ax.set_title(f"{tag} {lab}" + (f"  $A_2$={a2:.3f}" if a2 is not None else ""))
    fig.suptitle(case["name"], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_a2_gallery(path: Path, cases: list[dict]) -> None:
    n = len(cases)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)
    for k, case in enumerate(cases):
        ax = axes[k // ncols][k % ncols]
        for tag, style in (("data", "-"), ("recon", "--")):
            row = case["arms"].get(tag) or {}
            if not row.get("ok"):
                continue
            ax.plot(row["t_gyr"], row["a2_t"], style, lw=1.6, label=tag)
        ax.set_title(case["name"], fontsize=9)
        ax.set_xlabel(r"$t$ [Gyr]")
        ax.set_ylabel(r"$A_2$")
        ax.axhline(0.30, color="0.5", ls=":", lw=0.8)
        ax.legend(frameon=False, fontsize=7)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(r"Data vs FFT recon $A_2(t)$ (multi-system)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_faceon_gallery(path: Path, cases: list[dict]) -> None:
    """One row per system: data t0, recon t0, data tend, recon tend."""
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(10.5, 2.35 * n))
    if n == 1:
        axes = np.asarray([axes])
    col_labs = [r"data $t{=}0$", r"recon $t{=}0$", r"data $t_{\rm end}$", r"recon $t_{\rm end}$"]
    for i, case in enumerate(cases):
        d = case["arms"].get("data") or {}
        r = case["arms"].get("recon") or {}
        dm = d.get("faceon_maps") or [None, None]
        rm = r.get("faceon_maps") or [None, None]
        imgs = [dm[0], rm[0], dm[-1] if dm[-1] is not None else None, rm[-1] if rm[-1] is not None else None]
        for j, img in enumerate(imgs):
            ax = axes[i, j]
            if img is None:
                ax.axis("off")
                continue
            ax.imshow(
                np.log10(np.maximum(img, 1e-12)).T,
                origin="lower",
                cmap="inferno",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_labs[j], fontsize=9)
        axes[i, 0].set_ylabel(case["name"], fontsize=8)
    fig.suptitle("Multi-system data vs FFT recon face-on", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _passes(case: dict) -> tuple[bool, list[str], dict]:
    """Score dynamical tracking (Prop A) separately from absolute A₂ fidelity.

    Returns (track_ok, notes, scores) where scores has
    ``dynamical_track`` / ``absolute_a2`` booleans.
    """
    notes = []
    d = case["arms"].get("data") or {}
    r = case["arms"].get("recon") or {}
    if not d.get("ok") or not r.get("ok"):
        return False, ["arm failed"], {"dynamical_track": False, "absolute_a2": False}
    kind = case["kind"]
    a2_d0, a2_d1 = float(d["a2_pre"]), float(d["a2_post"])
    a2_r0, a2_r1 = float(r["a2_pre"]), float(r["a2_post"])
    da2_d = a2_d1 - a2_d0
    da2_r = a2_r1 - a2_r0

    track_ok = True
    abs_ok = True

    if (d.get("com_drift_kpc") or 0) > 0.05 or (r.get("com_drift_kpc") or 0) > 0.05:
        notes.append("COM drift > 0.05 kpc")
        track_ok = False

    if kind == "barred":
        # Dynamical track: ΔA2 follows data; recon does not explode / fully wash.
        if abs(da2_r - da2_d) > 0.12:
            notes.append(f"ΔA2 track loose |ΔΔ|={abs(da2_r - da2_d):.3f}")
            track_ok = False
        if a2_r1 < 0.03 and a2_d0 >= 0.15:
            notes.append(f"recon fully washed A2_end={a2_r1:.3f}")
            track_ok = False
        # Absolute A₂ fidelity (FFT dens ↔ Am HOLD) — not required for Prop A.
        if a2_r0 < 0.55 * a2_d0 and a2_d0 >= 0.15:
            notes.append(
                f"absolute A2 soft recon={a2_r0:.3f} vs data={a2_d0:.3f} (FFT/Am HOLD)"
            )
            abs_ok = False
    else:
        if a2_r1 > 0.10:
            notes.append(f"quiet recon grew bar A2_end={a2_r1:.3f}")
            track_ok = False
            abs_ok = False
        if a2_r0 > 0.08:
            abs_ok = False

    # Clump: only fail track if recon multiplicity blows up *relative to data end*.
    cd = int((d.get("clump_post") or {}).get("multiplicity", 0))
    cr = int((r.get("clump_post") or {}).get("multiplicity", 0))
    if cr >= max(cd + 5, 10):
        notes.append(f"recon clump multiplicity {cr} vs data {cd}")
        track_ok = False

    return track_ok, notes, {"dynamical_track": track_ok, "absolute_a2": abs_ok}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/recon_track_multisystem_2026-07-26"),
    )
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument(
        "--n-disk",
        type=int,
        default=250_000,
        help="Disk alone; total = round(7/4·N). Default 2.5e5 under shared "
        "machine load; use 1e6 when free (corpus-scale single-system already "
        "at evolve_compare_disk1e6).",
    )
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--evolve-gyr", type=float, default=0.50)
    p.add_argument("--omp", type=int, default=2)
    p.add_argument("--timeout-s", type=float, default=7200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-track", type=int, default=11)
    p.add_argument("--faceon-times", type=str, default="0,0.25,0.5")
    p.add_argument("--faceon-bins", type=int, default=128)
    p.add_argument("--faceon-half", type=float, default=12.0)
    p.add_argument("--systems-json", type=Path, default=None)
    p.add_argument("--feature-drift-run", type=str, default="54a8faf836a0")
    p.add_argument("--skip-feature-drift", action="store_true")
    p.add_argument("--paper-figures", type=Path, default=None)
    p.add_argument("--paper-results", type=Path, default=None)
    args = p.parse_args()

    if args.teacher is None:
        args.teacher = default_teacher()
    n_tot = int(round(args.n_disk * 7 / 4))
    n_ev = n_tot
    n_resample = n_tot
    print(
        f"n-disk={args.n_disk:,} → total N={n_tot:,} "
        f"(halo≈{args.n_disk // 2:,}, bulge≈{args.n_disk // 4:,})",
        flush=True,
    )

    systems = DEFAULT_SYSTEMS
    if args.systems_json is not None:
        systems = json.loads(Path(args.systems_json).read_text())

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.omp)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    faceon_times = [float(x) for x in args.faceon_times.split(",") if x.strip()]
    if args.evolve_gyr not in faceon_times:
        faceon_times.append(float(args.evolve_gyr))
    faceon_times = sorted(set(faceon_times))

    print(f"=== teacher {args.teacher} ===", flush=True)
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)

    case_reports: list[dict] = []
    for spec in systems:
        name = spec["name"]
        path = Path(spec["path"])
        if not path.is_file():
            # try absolute under ROOT
            alt = ROOT / path
            if alt.is_file():
                path = alt
            else:
                print(f"SKIP missing {path}", flush=True)
                continue
        case_dir = args.out / name
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== system {name} ← {path} ===", flush=True)

        # Field dens MSE at t=0 (full dump → AE), independent of particle N.
        field_mse = _field_recon_mse(path, teacher, cfg, stats)
        print(
            f"  dens MSE mean={field_mse['mse_dens_mean']:.4f} "
            f"disk={field_mse.get('mse_dens_disk', float('nan')):.4f}",
            flush=True,
        )

        arms: dict[str, dict] = {}
        # data
        data_ev = _full_com(_subsample_dump(path, n_ev, rng))
        arms["data"] = {"parts": data_ev, "meta": {"path": str(path)}}
        # recon
        recon = _recon_particles(
            path, teacher, cfg, stats, n_resample=n_resample, rng=rng
        )
        recon_ev = _vcom_only(_stratified_down(recon, n_ev, rng))
        arms["recon"] = {"parts": recon_ev, "meta": {"teacher": str(args.teacher)}}

        case = {
            "name": name,
            "kind": spec.get("kind", "barred"),
            "run_hash": spec.get("run_hash"),
            "path": str(path),
            "field_mse": field_mse,
            "arms": {},
        }

        for tag, pack in arms.items():
            parts = pack["parts"]
            pre = _metrics(parts)
            face0 = _faceon(parts, n_bins=args.faceon_bins, half=args.faceon_half)
            clump0 = _clump_score(face0)
            print(
                f"  evolve {tag} A2={pre['a2']:.3f} clump_mult={clump0['multiplicity']}",
                flush=True,
            )
            evo = _evolve_tracked(
                parts,
                end_gyr=args.evolve_gyr,
                dt=args.dt,
                omp=args.omp,
                timeout_s=args.timeout_s,
                n_track=args.n_track,
                faceon_times=faceon_times,
                faceon_bins=args.faceon_bins,
                faceon_half=args.faceon_half,
            )
            if not evo.get("ok"):
                case["arms"][tag] = {
                    "ok": False,
                    "pre": pre,
                    "clump_pre": clump0,
                    "meta": pack["meta"],
                    "error": evo.get("error"),
                }
                print(f"    FAIL {evo.get('error')}", flush=True)
                continue
            pos_f = evo.pop("pos_final")
            face_maps = evo.pop("faceon_maps", [])
            face_t = evo.pop("faceon_t_gyr", [])
            post_parts = {**parts, "pos": pos_f}
            post = _metrics(post_parts)
            face1 = face_maps[-1] if face_maps else _faceon(
                post_parts, n_bins=args.faceon_bins, half=args.faceon_half
            )
            clump1 = _clump_score(face1)
            am_pre = _am_profiles(
                parts["pos"][parts["component_id"] == 0],
                parts["mass"][parts["component_id"] == 0],
            )
            am_post = _am_profiles(
                post_parts["pos"][post_parts["component_id"] == 0],
                post_parts["mass"][post_parts["component_id"] == 0],
            )
            row = {
                "ok": True,
                "pre": pre,
                "post": post,
                "a2_pre": pre["a2"],
                "a2_post": post["a2"],
                "da2": post["a2"] - pre["a2"],
                "com_drift_kpc": evo.get("com_drift_kpc"),
                "wall_s": evo.get("wall_s"),
                "t_gyr": evo.get("t_gyr"),
                "a2_t": evo.get("a2_t"),
                "com_norm_t": evo.get("com_norm_t"),
                "faceon_t_gyr": face_t,
                "faceon_maps": [np.asarray(m) for m in face_maps],
                "clump_pre": clump0,
                "clump_post": clump1,
                "am_pre": am_pre,
                "am_post": am_post,
                "meta": pack["meta"],
            }
            case["arms"][tag] = row
            print(
                f"    A2 {pre['a2']:.3f}→{post['a2']:.3f} "
                f"COM={row['com_drift_kpc']:.4f} "
                f"clump {clump0['multiplicity']}→{clump1['multiplicity']} "
                f"wall={row['wall_s']:.0f}s",
                flush=True,
            )

        ok, notes, scores = _passes(case)
        case["passes"] = ok  # dynamical track (Proposition A)
        case["notes"] = notes
        case["scores"] = scores

        # Per-system plots (drop heavy arrays from JSON later).
        _plot_system_a2(case_dir / "a2_t.png", case)
        _plot_system_faceon(case_dir / "faceon.png", case)
        # Save faceon maps npz
        np.savez_compressed(
            case_dir / "faceon_maps.npz",
            **{
                f"{tag}_{k}": np.asarray(m)
                for tag, row in case["arms"].items()
                if row.get("ok")
                for k, m in enumerate(row.get("faceon_maps") or [])
            },
        )
        # JSON-safe copy (drop face-on arrays + Am numpy profiles)
        slim = {
            **{k: v for k, v in case.items() if k != "arms"},
            "arms": {},
        }
        for tag, row in case["arms"].items():
            slim["arms"][tag] = {
                k: v
                for k, v in row.items()
                if k not in ("faceon_maps", "am_pre", "am_post")
            }
        (case_dir / "verdict.json").write_text(json.dumps(slim, indent=2, default=_json_default))
        case_reports.append(case)

    # Optional feature drift (Proposition B)
    drift = None
    if not args.skip_feature_drift and args.feature_drift_run:
        print(f"\n=== feature drift along {args.feature_drift_run} ===", flush=True)
        drift = _feature_drift_along_run(
            args.feature_drift_run, teacher, cfg, stats, max_dumps=8
        )
        if drift:
            print(
                f"  rel ||z(t)-z0|| end={drift['rel_drift_end']:.4f} "
                f"n={drift['n_dumps']}",
                flush=True,
            )
            (args.out / "feature_drift.json").write_text(
                json.dumps(drift, indent=2, default=_json_default)
            )
            fig, ax = plt.subplots(figsize=(5.0, 3.2))
            ax.plot(range(len(drift["rel_drift"])), drift["rel_drift"], "o-", lw=1.6)
            ax.set_xticks(range(len(drift["dumps"])))
            ax.set_xticklabels(
                [d.replace("step_", "").replace(".npz", "") for d in drift["dumps"]],
                rotation=45,
                ha="right",
                fontsize=7,
            )
            ax.set_ylabel(r"$\|z(t)-z(0)\|/\|z(0)\|$")
            ax.set_title(f"Bottleneck drift · {args.feature_drift_run}")
            fig.tight_layout()
            fig.savefig(args.out / "feature_drift.png", dpi=160)
            plt.close(fig)

    # Galleries
    _plot_a2_gallery(args.out / "a2_t_all.png", case_reports)
    _plot_faceon_gallery(args.out / "faceon_gallery.png", case_reports)

    n_pass = sum(1 for c in case_reports if c.get("passes"))
    n_abs = sum(
        1 for c in case_reports if (c.get("scores") or {}).get("absolute_a2")
    )
    track_verdict = (
        "CONSISTENT"
        if n_pass >= max(3, int(0.6 * len(case_reports)))
        else "MIXED / DEGRADED"
    )
    abs_verdict = (
        "OK"
        if n_abs >= max(3, int(0.6 * len(case_reports)))
        else "DEGRADED (FFT dens sharpness vs Am HOLD on mid/mild bars)"
    )
    verdict = {
        "teacher": str(args.teacher),
        "n_disk": args.n_disk,
        "n_total": n_tot,
        "evolve_gyr": args.evolve_gyr,
        "dt": args.dt,
        "omp": args.omp,
        "force": "bh_c",
        "shared_centering": (
            "data: full soft COM; dens-resampled recon: morphological origin + VCOM-only"
        ),
        "proposition": "A — approximate dynamical consistency (finite T)",
        "feature_drift": drift,
        "n_systems": len(case_reports),
        "n_pass_dynamical_track": n_pass,
        "n_pass_absolute_a2": n_abs,
        "n_pass": n_pass,
        "systems": [
            {
                "name": c["name"],
                "kind": c["kind"],
                "run_hash": c["run_hash"],
                "passes": c["passes"],
                "scores": c.get("scores"),
                "notes": c["notes"],
                "field_mse": c["field_mse"],
                "arms": {
                    tag: {
                        "ok": row.get("ok"),
                        "a2_pre": row.get("a2_pre"),
                        "a2_post": row.get("a2_post"),
                        "da2": row.get("da2"),
                        "com_drift_kpc": row.get("com_drift_kpc"),
                        "clump_pre": row.get("clump_pre"),
                        "clump_post": row.get("clump_post"),
                        "wall_s": row.get("wall_s"),
                    }
                    for tag, row in c["arms"].items()
                },
            }
            for c in case_reports
        ],
        "empirical_verdict": track_verdict,
        "dynamical_track_verdict": track_verdict,
        "absolute_a2_verdict": abs_verdict,
        "note": (
            "Primary claim (Prop A): FFT recon tracks data ΔA2/COM over T=0.5 Gyr. "
            "Absolute recon A2 may sit below data (FFT dens sharpness vs Am HOLD) — "
            "scored separately. "
            f"Also see disk=1e6 single-system evolve_compare_disk1e6 for 54a8."
        ),
    }
    (args.out / "verdict.json").write_text(
        json.dumps(verdict, indent=2, default=_json_default)
    )

    # SUMMARY.md
    lines = [
        "# Multi-system data vs FFT recon tracking",
        "",
        f"Teacher: `{args.teacher}`",
        f"Settings: disk$={args.n_disk:,}$, total$={n_tot:,}$, "
        f"$t_{{\\rm end}}={args.evolve_gyr}$ Gyr, $dt={args.dt}$, OMP={args.omp}.",
        "",
        "Proposition A (`docs/dynamical_consistency_charts.md`): "
        "recon tracks data under shared self-gravity for finite $T$.",
        "",
        f"**Dynamical-track verdict (Prop A): {verdict['dynamical_track_verdict']}** "
        f"({n_pass}/{len(case_reports)} systems).",
        f"**Absolute $A_2$ fidelity: {verdict['absolute_a2_verdict']}** "
        f"({n_abs}/{len(case_reports)}).",
        "",
        "| System | kind | dens MSE | data $A_2$→ | recon $A_2$→ | "
        "COM$_d$/COM$_r$ | track | abs $A_2$ | notes |",
        "|--------|------|----------|-------------|--------------|"
        "-----------------|-------|----------|-------|",
    ]
    for c in case_reports:
        d = c["arms"].get("data") or {}
        r = c["arms"].get("recon") or {}
        mse = c["field_mse"].get("mse_dens_mean", float("nan"))
        def arrow(row):
            if not row.get("ok"):
                return "fail"
            return f"{row['a2_pre']:.3f}→{row['a2_post']:.3f}"
        com_d = d.get("com_drift_kpc")
        com_r = r.get("com_drift_kpc")
        com_s = (
            f"{com_d:.1e}/{com_r:.1e}"
            if com_d is not None and com_r is not None
            else "—"
        )
        sc = c.get("scores") or {}
        notes = "; ".join(c.get("notes") or []) or "—"
        lines.append(
            f"| `{c['name']}` | {c['kind']} | {mse:.4f} | {arrow(d)} | {arrow(r)} | "
            f"{com_s} | "
            f"{'PASS' if sc.get('dynamical_track') else 'fail'} | "
            f"{'ok' if sc.get('absolute_a2') else 'soft'} | {notes} |"
        )
    if drift:
        lines += [
            "",
            "## Feature drift (Proposition B)",
            "",
            f"Run `{drift['run_hash']}`: relative bottleneck drift "
            f"end = **{drift['rel_drift_end']:.4f}** over {drift['n_dumps']} dumps "
            f"(`feature_drift.png`).",
        ]
    lines += [
        "",
        "## Notes",
        "",
        "- Absolute recon $A_2$ often trails data (FFT dens vs Am HOLD); "
        "tracking uses $\\Delta A_2$ and morph coherence.",
        "- Corpus-scale disk$=10^6$ single-system reference: "
        "`evolve_compare_disk1e6_2026-07-25` / `fig_evolve_disk1e6_*`.",
        f"- Figures: `{args.out}/a2_t_all.png`, `{args.out}/faceon_gallery.png`.",
    ]
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)

    if args.paper_figures is not None:
        args.paper_figures.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out / "a2_t_all.png", args.paper_figures / "fig_recon_track_a2_t.png")
        shutil.copy2(
            args.out / "faceon_gallery.png",
            args.paper_figures / "fig_recon_track_faceon.png",
        )
        if (args.out / "feature_drift.png").is_file():
            shutil.copy2(
                args.out / "feature_drift.png",
                args.paper_figures / "fig_recon_track_feature_drift.png",
            )
    if args.paper_results is not None:
        args.paper_results.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out / "SUMMARY.md", args.paper_results / "recon_track_SUMMARY.md")
        shutil.copy2(args.out / "verdict.json", args.paper_results / "recon_track_verdict.json")

    return 0 if verdict["empirical_verdict"] == "CONSISTENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
