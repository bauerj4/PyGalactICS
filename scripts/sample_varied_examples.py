#!/usr/bin/env python3
"""Generate a varied gallery of generative ICs (amplify_knn_hybrid + optional z_amplify).

Particle counts: **disk alone = 1e6** with 4:2:1 mix → halo=5e5, bulge=2.5e5,
total ≈ 1.75e6. Morph dens-map origin + VCOM-only (no soft position COM).

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \\
    .venv/bin/python scripts/sample_varied_examples.py \\
      --out runs/ml/field_maps/varied_examples_2026-07-25 \\
      --n-disk 1000000 \\
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
    RANK,
    _a2_disk,
    _disk_collapse,
    build_library,
    default_teacher,
)
from smoke_field_maps import _evolve_bh  # noqa: E402

from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy  # noqa: E402
from galacticsics.ml.fields.normalize import denormalize_stack  # noqa: E402
from galacticsics.ml.fields.resample import resample_particles_from_multiscale  # noqa: E402
from ntropy.analysis.disk_density import (  # noqa: E402
    bin_plane_density,
    disk_azimuthal_fourier,
)


PRESETS: list[dict] = [
    {
        "name": "hybrid_strong_01",
        "kind": "barred",
        "method": "amplify_knn_hybrid",
        "bar_kind": "strong",
        "seed": 11,
        "a2_min": 0.32,
        "kwargs": {"alpha_lo": 1.28, "alpha_hi": 1.48, "strong_floor": 0.28},
    },
    {
        "name": "hybrid_strong_02",
        "kind": "barred",
        "method": "amplify_knn_hybrid",
        "bar_kind": "strong",
        "seed": 29,
        "a2_min": 0.32,
        "kwargs": {"alpha_lo": 1.30, "alpha_hi": 1.50, "strong_floor": 0.30},
    },
    {
        "name": "hybrid_strong_03",
        "kind": "barred",
        "method": "amplify_knn_hybrid",
        "bar_kind": "strong",
        "seed": 47,
        "a2_min": 0.30,
        "kwargs": {
            "alpha_lo": 1.25,
            "alpha_hi": 1.45,
            "knn_alpha_max": 0.22,
            "strong_floor": 0.28,
        },
    },
    {
        "name": "hybrid_mild_01",
        "kind": "barred",
        "method": "amplify_knn_hybrid",
        "bar_kind": "mild",
        "seed": 13,
        "a2_min": 0.12,
        "a2_max": 0.28,
        "kwargs": {"alpha_lo": 1.05, "alpha_hi": 1.14, "strong_floor": 0.20},
    },
    {
        "name": "hybrid_mild_02",
        "kind": "barred",
        "method": "amplify_knn_hybrid",
        "bar_kind": "mild",
        "seed": 61,
        "a2_min": 0.12,
        "a2_max": 0.28,
        "kwargs": {
            "alpha_lo": 1.08,
            "alpha_hi": 1.18,
            "knn_alpha_max": 0.12,
            "strong_floor": 0.22,
        },
    },
    {
        "name": "hybrid_mid_01",
        "kind": "barred",
        "method": "amplify_knn_hybrid",
        "bar_kind": "mild",
        "seed": 83,
        "a2_min": 0.18,
        "a2_max": 0.35,
        "kwargs": {"alpha_lo": 1.15, "alpha_hi": 1.28, "strong_floor": 0.25},
    },
    {
        "name": "quiet_01",
        "kind": "quiet",
        "method": "amplify_knn_hybrid",
        "bar_kind": "quiet",
        "seed": 7,
        "a2_max": 0.08,
        "kwargs": {},
    },
    {
        "name": "quiet_02",
        "kind": "quiet",
        "method": "amplify_knn_hybrid",
        "bar_kind": "quiet",
        "seed": 101,
        "a2_max": 0.08,
        "kwargs": {},
    },
    {
        "name": "zamp_strong_01",
        "kind": "barred",
        "method": "z_amplify",
        "bar_kind": "strong",
        "seed": 19,
        "a2_min": 0.30,
        "kwargs": {"alpha_lo": 1.30, "alpha_hi": 1.55, "strong_floor": 0.28},
    },
    {
        "name": "zamp_mild_01",
        "kind": "barred",
        "method": "z_amplify",
        "bar_kind": "mild",
        "seed": 37,
        "a2_min": 0.12,
        "a2_max": 0.28,
        "kwargs": {"alpha_lo": 1.08, "alpha_hi": 1.20, "strong_floor": 0.22},
    },
]


def _vcom_only(parts: dict) -> dict:
    """Subtract shared global VCOM; keep morphological dens-map origin."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in parts.items()}
    m = out["mass"]
    out["vel"] = out["vel"] - np.average(out["vel"], axis=0, weights=m)
    return out


def _mix_from_disk(n_disk: int) -> dict[str, int]:
    """4:2:1 disk:halo:bulge with disk fixed at ``n_disk``."""
    n_disk = int(n_disk)
    return {"disk": n_disk, "halo": n_disk // 2, "bulge": n_disk // 4}


def _resample(lib, den, n_disk: int, rng: np.random.Generator) -> dict:
    n_per = _mix_from_disk(n_disk)
    n_tot = int(sum(n_per.values()))
    parts = resample_particles_from_multiscale(
        den,
        cfg=lib.cfg,
        n_particles=n_tot,
        n_per_component=n_per,
        rng=rng,
    )
    parts = _vcom_only(parts)
    if "eps" not in parts:
        parts["eps"] = np.full(parts["pos"].shape[0], 0.1, dtype=np.float64)
    return parts


def _faceon(parts: dict, n_bins: int = 160, half: float = 12.0) -> np.ndarray:
    disk = parts["component_id"] == 0
    return bin_plane_density(
        parts["pos"][disk],
        parts["mass"][disk],
        axes=(0, 1),
        n_bins=n_bins,
        half_extent=half,
    ).density


def _am_particle(parts: dict, n_bins: int = 16, r_max: float = 12.0) -> dict:
    disk = parts["component_id"] == 0
    return disk_azimuthal_fourier(
        parts["pos"][disk],
        parts["mass"][disk],
        m=2,
        r_max=r_max,
        n_bins=n_bins,
        z_max=0.5,
        min_count=8,
    )


def _imshow(ax, img: np.ndarray, title: str) -> None:
    from galacticsics.campaign.analysis import dens_array_log10

    show, vmin_s, vmax_s, _ = dens_array_log10(img, vmax_pct=99.2)
    ax.imshow(
        show,
        origin="lower",
        cmap="inferno",
        vmin=vmin_s,
        vmax=vmax_s,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_example(
    out: Path,
    *,
    name: str,
    face: np.ndarray,
    dens_map: np.ndarray,
    am: dict,
    a2_part: float,
    a2_map: float,
    method: str,
    alpha: float | None,
    bar_kind: str,
    n_disk: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    lab = f"{name}\n{method} · {bar_kind} · N_disk={n_disk:,}"
    if alpha is not None:
        lab += f" · α={alpha:.2f}"
    _imshow(axes[0], face, f"particle face-on\nA₂={a2_part:.3f}")
    _imshow(axes[1], dens_map, f"dens map\nA₂map={a2_map:.3f}")
    r = np.asarray(am["r_mid"], dtype=float)
    a2r = np.asarray(am["a_m_over_a0"], dtype=float)
    axes[2].plot(r, a2r, color="#c45c26", lw=1.8)
    axes[2].set_xlabel(r"$R$ [kpc]")
    axes[2].set_ylabel(r"$A_2(R)$")
    axes[2].set_ylim(0, None)
    axes[2].set_title(r"$A_2(R)$ (disk particles)", fontsize=9)
    fig.suptitle(lab, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{name}_panel.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    _imshow(ax, face, f"{name}\nA₂={a2_part:.3f} · {bar_kind}")
    fig.tight_layout()
    fig.savefig(out / f"{name}_faceon.png", dpi=150)
    plt.close(fig)


def _plot_summary_grid(out: Path, rows: list[dict], faces: list[np.ndarray]) -> Path:
    n = len(rows)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        r = rows[i]
        _imshow(
            ax,
            faces[i],
            f"{r['name']}\n{r['method']} · {r['bar_kind']}\nA₂={r['a2_part']:.3f}",
        )
    fig.suptitle(
        "Varied generative ICs (FFT long · amplify_knn_hybrid / z_amplify · N_disk=1e6)",
        fontsize=12,
    )
    fig.tight_layout()
    path = out / "summary_grid.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_a2_table(out: Path, rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(rows) + 1.2))
    names = [r["name"] for r in rows]
    a2s = [r["a2_part"] for r in rows]
    colors = []
    for r in rows:
        if r["bar_kind"] == "quiet":
            colors.append("#4a7c59")
        elif r["bar_kind"] == "strong":
            colors.append("#c45c26")
        else:
            colors.append("#2a6f97")
    y = np.arange(len(names))
    ax.barh(y, a2s, color=colors, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['name']} ({r['method']})" for r in rows], fontsize=8)
    ax.set_xlabel(r"particle $A_2$")
    ax.set_title("Varied generative IC gallery")
    ax.axvline(0.05, color="0.5", ls=":", lw=1)
    ax.axvline(0.25, color="0.5", ls=":", lw=1)
    fig.tight_layout()
    path = out / "a2_barh.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _sample_one(lib, preset: dict, rng: np.random.Generator):
    kind = preset["kind"]
    method = preset["method"]
    kw = dict(preset.get("kwargs") or {})
    if method == "z_amplify":
        feat, meta = lib.sample_features_z_amplify(kind=kind, rng=rng, **kw)
        fields = lib.decode_features(feat)
    else:
        fields, meta = lib.sample_fields(kind=kind, method=method, rng=rng, **kw)
    return fields, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/varied_examples_2026-07-25"),
    )
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar", type=int, default=36)
    p.add_argument("--n-quiet", type=int, default=20)
    p.add_argument("--n-mid", type=int, default=12)
    p.add_argument("--n-rot-bar", type=int, default=6)
    p.add_argument("--bar-floor", type=float, default=0.25)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument(
        "--n-disk",
        type=int,
        default=1_000_000,
        help="Disk particle count for IC panels (halo=n_disk/2, bulge=n_disk/4)",
    )
    p.add_argument(
        "--n-disk-gate",
        type=int,
        default=80_000,
        help="Cheaper disk count for IC A₂ gating retries only",
    )
    p.add_argument("--faceon-bins", type=int, default=160)
    p.add_argument("--omp", type=int, default=4)
    p.add_argument(
        "--n-disk-evolve",
        type=int,
        default=40_000,
        help="Disk count for optional short evolve (re-sampled from dens)",
    )
    p.add_argument("--evolve-gyr", type=float, default=0.10)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--n-evolve-pick", type=int, default=3)
    p.add_argument("--ic-a2-tries", type=int, default=8)
    p.add_argument("--skip-evolve", action="store_true")
    p.add_argument("--paper-figures", type=Path, default=None)
    p.add_argument("--seed", type=int, default=20260725)
    args = p.parse_args()
    if args.teacher is None:
        args.teacher = default_teacher()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.omp)))
    torch.set_num_threads(max(1, int(args.omp)))

    n_per_panel = _mix_from_disk(args.n_disk)
    n_tot_panel = int(sum(n_per_panel.values()))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"teacher={args.teacher}", flush=True)
    print(
        f"out={args.out}  n_disk={args.n_disk}  n_per={n_per_panel}  "
        f"N_tot≈{n_tot_panel}  OMP={args.omp}",
        flush=True,
    )

    lib_args = argparse.Namespace(
        teacher=args.teacher,
        rank=args.rank,
        n_bar=args.n_bar,
        n_quiet=args.n_quiet,
        n_mid=args.n_mid,
        n_rot_bar=args.n_rot_bar,
        bar_floor=args.bar_floor,
        quiet_ceil=args.quiet_ceil,
        enc_grid=args.enc_grid,
        n_pc=args.n_pc,
    )
    lib = build_library(lib_args)
    lib.save_codes(args.out / "feature_library_codes.npz")
    disk_g = lib.cfg.grid_for("disk")
    print(
        f"library n={len(lib.feats)} bar={lib.bar_pool.size} quiet={lib.quiet_pool.size} "
        f"A₂∈[{lib.a2.min():.3f},{lib.a2.max():.3f}]",
        flush=True,
    )

    rows: list[dict] = []
    faces: list[np.ndarray] = []
    den_by_name: dict[str, dict] = {}

    for preset in PRESETS:
        best = None
        base_seed = int(preset["seed"]) ^ int(args.seed)
        a2_min = float(preset.get("a2_min", 0.0))
        a2_max = float(preset.get("a2_max", 1.0))
        for attempt in range(max(1, int(args.ic_a2_tries))):
            rng = np.random.default_rng(base_seed + 10007 * attempt)
            fields, meta = _sample_one(lib, preset, rng)
            den = {k: denormalize_stack(fields[k][0].numpy(), lib.stats[k]) for k in fields}
            dens = _disk_collapse(den["disk"], disk_g.n_z, disk_g.n_mom)
            a2_map = float(
                dens_map_azimuthal_fourier_numpy(dens, m=2, r_max=12.0)["a_m_over_a0_median"]
            )
            parts_g = _resample(lib, den, args.n_disk_gate, rng)
            a2_part = _a2_disk(parts_g)
            alpha = meta.get("alpha")
            alpha_f = float(alpha) if alpha is not None else None

            if preset["kind"] == "quiet":
                ok = a2_part <= a2_max
                score = -a2_part
            else:
                ok = a2_part >= a2_min and a2_part <= a2_max
                mid = 0.5 * (a2_min + min(a2_max, 0.55))
                score = a2_part if ok else -abs(a2_part - mid)

            cand = (score, den, dens, meta, alpha_f, attempt, a2_part, a2_map, ok)
            if best is None or score > best[0]:
                best = cand
            print(
                f"  {preset['name']} try#{attempt} A₂p={a2_part:.3f} "
                f"A₂map={a2_map:.3f}"
                + (f" α={alpha_f:.2f}" if alpha_f is not None else "")
                + (" ✓" if ok else ""),
                flush=True,
            )
            del parts_g
            if ok:
                break

        assert best is not None
        _, den, dens, meta, alpha_f, attempt, a2_gate, a2_map, _ok = best
        rng = np.random.default_rng(base_seed + 90001 + attempt)
        n_disk_used = int(args.n_disk)
        try:
            parts = _resample(lib, den, n_disk_used, rng)
        except MemoryError:
            n_disk_used = max(250_000, int(args.n_disk) // 2)
            print(
                f"  ! OOM at n_disk={args.n_disk}; falling back to n_disk={n_disk_used}",
                flush=True,
            )
            parts = _resample(lib, den, n_disk_used, rng)

        a2_part = _a2_disk(parts)
        face = _faceon(parts, n_bins=args.faceon_bins)
        am = _am_particle(parts)
        n_per = parts.get("n_per_component") or _mix_from_disk(n_disk_used)

        bar_kind = preset["bar_kind"]
        if bar_kind != "quiet":
            if a2_part >= 0.35:
                bar_kind = "strong"
            elif a2_part >= 0.15:
                bar_kind = "mild"
            else:
                bar_kind = "weak"

        _plot_example(
            args.out,
            name=preset["name"],
            face=face,
            dens_map=dens,
            am=am,
            a2_part=a2_part,
            a2_map=a2_map,
            method=preset["method"],
            alpha=alpha_f,
            bar_kind=bar_kind,
            n_disk=n_disk_used,
        )
        row = {
            "name": preset["name"],
            "method": preset["method"],
            "kind": preset["kind"],
            "bar_kind": bar_kind,
            "seed": int(preset["seed"]),
            "attempt": int(attempt),
            "alpha": alpha_f,
            "a2_part": float(a2_part),
            "a2_gate": float(a2_gate),
            "a2_map": float(a2_map),
            "n_disk": int(n_disk_used),
            "n_per_component": {k: int(v) for k, v in dict(n_per).items()},
            "n_particles": int(parts["pos"].shape[0]),
            "meta": {
                k: (v if not isinstance(v, (list, np.ndarray)) else None)
                for k, v in meta.items()
                if k != "z"
            },
            "am_r": {
                "r_mid": np.asarray(am["r_mid"], dtype=float).tolist(),
                "a2": np.asarray(am["a_m_over_a0"], dtype=float).tolist(),
            },
            "panel": f"{preset['name']}_panel.png",
            "faceon": f"{preset['name']}_faceon.png",
        }
        rows.append(row)
        faces.append(face)
        den_by_name[preset["name"]] = den
        del parts
        print(
            f"→ {preset['name']:20s} {preset['method']:18s} "
            f"{bar_kind:6s} A₂p={a2_part:.3f} (gate={a2_gate:.3f}) A₂map={a2_map:.3f} "
            f"Ndisk={n_disk_used}"
            + (f" α={alpha_f:.2f}" if alpha_f is not None else ""),
            flush=True,
        )

    grid_path = _plot_summary_grid(args.out, rows, faces)
    barh_path = _plot_a2_table(args.out, rows)

    evolve_rows = []
    if not args.skip_evolve and args.n_evolve_pick > 0:
        barred = sorted(
            [r for r in rows if r["kind"] == "barred"],
            key=lambda r: r["a2_part"],
            reverse=True,
        )
        picks: list[dict] = []
        for r in barred:
            if len(picks) >= args.n_evolve_pick:
                break
            if not picks:
                picks.append(r)
                continue
            if min(abs(r["a2_part"] - p["a2_part"]) for p in picks) < 0.03:
                continue
            picks.append(r)
        while len(picks) < min(args.n_evolve_pick, len(barred)):
            for r in barred:
                if r not in picks:
                    picks.append(r)
                    break

        print(f"short evolve on {[p['name'] for p in picks]} …", flush=True)
        for r in picks:
            rng = np.random.default_rng(r["seed"] + 999)
            parts = _resample(lib, den_by_name[r["name"]], args.n_disk_evolve, rng)
            a2_pre = _a2_disk(parts)
            evo = _evolve_bh(
                parts,
                end_gyr=args.evolve_gyr,
                dt=args.dt,
                force="bh_c",
                omp_threads=args.omp,
                timeout_s=900.0,
            )
            face_post = None
            a2_post = float("nan")
            if evo.get("ok") and "pos_final" in evo:
                post_parts = {
                    "pos": evo["pos_final"],
                    "mass": parts["mass"],
                    "vel": parts["vel"],
                    "component_id": parts["component_id"],
                }
                a2_post = _a2_disk(post_parts)
                face_post = _faceon(post_parts, n_bins=96)

            fig, axes = plt.subplots(1, 2 if face_post is not None else 1, figsize=(6.4, 3.2))
            axes = [axes] if face_post is None else list(np.atleast_1d(axes))
            _imshow(axes[0], faces[rows.index(r)], f"IC A₂={a2_pre:.3f}")
            if face_post is not None:
                _imshow(axes[1], face_post, f"t={args.evolve_gyr:.2f} Gyr A₂={a2_post:.3f}")
            fig.suptitle(f"short evolve · {r['name']} · {r['method']}")
            fig.tight_layout()
            evo_png = args.out / f"{r['name']}_evolve_short.png"
            fig.savefig(evo_png, dpi=140)
            plt.close(fig)
            erow = {
                "name": r["name"],
                "method": r["method"],
                "a2_pre": float(a2_pre),
                "a2_post": float(a2_post),
                "da2": float(a2_post - a2_pre) if np.isfinite(a2_post) else None,
                "com_drift_kpc": float(evo.get("com_drift_kpc", float("nan"))),
                "n_disk_evolve": int(args.n_disk_evolve),
                "n_evolve": int(parts["pos"].shape[0]),
                "evolve_gyr": float(args.evolve_gyr),
                "ok": bool(evo.get("ok", False)),
                "error": evo.get("error"),
                "panel": evo_png.name,
            }
            evolve_rows.append(erow)
            del parts
            print(
                f"  evolve {r['name']}: A₂ {a2_pre:.3f}→{a2_post:.3f} "
                f"COM={erow['com_drift_kpc']:.4f}",
                flush=True,
            )

    a2_vals = [r["a2_part"] for r in rows]
    verdict = {
        "approach": "varied generative IC gallery",
        "teacher": str(args.teacher),
        "n_library": len(lib.feats),
        "n_disk": int(args.n_disk),
        "n_per_component_target": n_per_panel,
        "n_particles_target": int(n_tot_panel),
        "count_mix": "4:2:1 disk:halo:bulge with disk fixed at n_disk",
        "frame": "morphological dens-map origin + shared global VCOM (no soft position COM)",
        "n_examples": len(rows),
        "a2_part_min": float(min(a2_vals)),
        "a2_part_max": float(max(a2_vals)),
        "a2_part_range": [float(min(a2_vals)), float(max(a2_vals))],
        "examples": rows,
        "evolve_short": evolve_rows,
        "summary_grid": grid_path.name,
        "a2_barh": barh_path.name,
    }
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2))

    lines = [
        "# Varied generative IC examples",
        "",
        f"Teacher: `{args.teacher}`",
        f"n_disk={args.n_disk} → n_per={n_per_panel} (N_tot≈{n_tot_panel}); "
        "count mix 4:2:1; morph dens origin + VCOM-only.",
        "",
        "| name | method | kind | A₂ | α | N_disk |",
        "|------|--------|------|----|---|--------|",
    ]
    for r in rows:
        a = f"{r['alpha']:.2f}" if r["alpha"] is not None else "—"
        lines.append(
            f"| `{r['name']}` | `{r['method']}` | {r['bar_kind']} | "
            f"{r['a2_part']:.3f} | {a} | {r.get('n_disk', args.n_disk)} |"
        )
    lines.append("")
    lines.append(f"A₂ range: **{min(a2_vals):.3f}–{max(a2_vals):.3f}** ({len(rows)} examples).")
    if evolve_rows:
        lines += [
            "",
            "## Short evolve (selected)",
            "",
            "| name | A₂ pre→post | ΔA₂ | COM |",
            "|------|-------------|-----|-----|",
        ]
        for e in evolve_rows:
            da = e["da2"]
            da_s = f"{da:+.3f}" if da is not None else "—"
            lines.append(
                f"| `{e['name']}` | {e['a2_pre']:.3f}→{e['a2_post']:.3f} | {da_s} | "
                f"{e['com_drift_kpc']:.4f} |"
            )
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    if args.paper_figures is not None:
        fig_dir = args.paper_figures
        fig_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(grid_path, fig_dir / "fig_varied_examples.png")
        shutil.copy2(barh_path, fig_dir / "fig_varied_examples_a2.png")
        strong = sorted(
            [r for r in rows if r["bar_kind"] == "strong"],
            key=lambda r: r["a2_part"],
            reverse=True,
        )
        quiet = [r for r in rows if r["bar_kind"] == "quiet"]
        for r in strong[:2]:
            shutil.copy2(args.out / r["panel"], fig_dir / f"fig_varied_{r['name']}.png")
        if quiet:
            shutil.copy2(
                args.out / quiet[0]["panel"],
                fig_dir / f"fig_varied_{quiet[0]['name']}.png",
            )
        print(f"copied figures → {fig_dir}", flush=True)

    print(
        json.dumps(
            {
                "n_examples": len(rows),
                "a2_range": verdict["a2_part_range"],
                "n_disk": int(args.n_disk),
                "n_per": n_per_panel,
                "summary_grid": str(grid_path),
                "verdict": str(args.out / "verdict.json"),
            },
            indent=2,
        )
    )
    print(f"done → {args.out}", flush=True)


if __name__ == "__main__":
    main()
