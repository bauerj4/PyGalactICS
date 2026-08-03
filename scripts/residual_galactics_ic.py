#!/usr/bin/env python3
"""Phase-A residual around GalactICS ``f0`` — build IC and/or evolve-gate.

Recipe (hand / library residual; Phase B learnable δ not required)::

    base   = GalactICS equilibrium particles ``f0(θ)``  (--ic-path)
    morph  = teacher AE recon (or deposit) of a barred dump (--morph-path)
    dens'  = axisym(Σ_f0) + α · m2(Σ_morph − axisym)   (disk)
    merge  = dens-resample disk + retain GalactICS halo/bulge
             + kNN-transplant GalactICS disk velocities

This replaces pure AE dens+moments resample as the primary generative IC.

Examples::

    # Build IC only
    .venv/bin/python scripts/residual_galactics_ic.py \\
      --ic-path runs/mw_morton_corpus_v2/906c4af73543/ic_state.npz \\
      --morph-path runs/mw_morton_corpus_v2/906c4af73543/evolution/particles/step_003200.npz \\
      --teacher runs/ml/field_maps/df_match_BA_2026-07-27/joint_df_fftlong_ft/multitower_slice_ae.pt \\
      --out runs/ml/field_maps/residual_f0_2026-07-28/ic_906c4_a10 \\
      --alpha 1.0 --n-disk 1000000

    # Build + 2 Gyr evolve gate (wraps evolve_component_slices)
    .venv/bin/python scripts/residual_galactics_ic.py \\
      --ic-path .../ic_state.npz --morph-path .../step_003200.npz \\
      --teacher .../joint_df_fftlong_ft/multitower_slice_ae.pt \\
      --out runs/ml/field_maps/residual_f0_2026-07-28/evolve_2gyr_906c4_a10 \\
      --alpha 1.0 --evolve-gyr 2.0 --force gpu_bh --n-disk 1000000 \\
      --reuse-data-from runs/ml/field_maps/dyn_consistency_12h_2026-07-27/evolve_2gyr_dens_amp_906c4 \\
      --paper-figures papers/mnras_noneq_ics/figures \\
      --paper-prefix fig_residual_f0_906c4_a10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evolve_resample_compare import (  # noqa: E402
    _a2_disk,
    _full_dyn_residual_particles,
    _ic_has_bulge,
    _n_total_for_disk,
    _residual_f0_particles,
    _stratified_down,
    _vcom_only,
)
from sample_latent_ic import default_teacher  # noqa: E402

from galacticsics.ml.fields.feature_library import load_frozen_teacher_bundle  # noqa: E402
from ntropy.analysis.disk_density import disk_azimuthal_fourier  # noqa: E402


RECIPE_PAINT = {
    "name": "residual_f0_m2",
    "phase": "A_hand_library_residual_on_f0",
    "phase_b": "CondDelta morph-conditioned dens contrast on f0",
    "equation": "f = T_phi(f0(theta); z) ≈ dens'(f0, δ_φ(f0, morph(z))) + vel(morph|f0)",
    "dens": (
        "Phase A: Sigma' = ring-renorm[ axisym(Sigma_f0) * max(1 + alpha * W(R;Rd) * "
        "sharpen(m2(Sigma_morph-axisym)/axisym(Sigma_morph)), floor) ]. "
        "Phase B: Sigma' = ring-renorm[ axisym(Sigma_f0) * max(1 + alpha * "
        "δ_φ(f0, morph_hint), floor) ] with learned midplane contrast."
    ),
    "primary_metric": "A2(R_d) via --a2-r-eval (not median ring A2)",
    "kinematics": (
        "Disk: --vel-source f0|morph|blend|hybrid "
        "(kNN transplant; default f0=GalactICS). Halo/bulge retained from f0."
    ),
    "morph_chart": "teacher AE recon / deposit / blend; Phase B uses morph as CondDelta hint",
    "radial_lock": "preserve_axisym + match_disk_radial_cdf_to_f0 after resample",
    "ablation_baseline": "fft_recon_dens_amp_shell (AE dens+moments + m2 amplify)",
    "note": (
        "Paint-on-bar path. Prefer --recipe full_dyn_replace after closest-f0 "
        "search when the goal is nearest GalactICS DF + full dynamical residual."
    ),
}

RECIPE_FULL_DYN = {
    "name": "full_dyn_residual",
    "phase": "full_dyn_residual",
    "equation": (
        "f ≈ replace_disk(data) ⊕ retain_halo_bulge(f0(θ*))   OR   "
        "OT-lite transport f0(θ*) → data (disk) ⊕ retain_halo_bulge(f0)"
    ),
    "goal": (
        "Nearest GalactICS f0(θ*) to barred dump on axisym dens+kin, then "
        "reconstruct all disk dynamics from the dump (not m2 paint on quiet Σ)."
    ),
    "dens": "Disk dens from data dump (replace) or OT radial CDF match to data (ot_lite).",
    "kinematics": "Disk velocities from data dump (native) or OT ⟨vφ⟩/σ transport.",
    "radial_lock": "None on f0 axisym — data/morph sets axisym Σ(R).",
    "primary_metric": "A2(R_d) vs data dump; kinetic profiles vs data",
    "closest_f0": "scripts/closest_galactics_f0.py → --ic-path",
}

RECIPE = RECIPE_PAINT  # default export for older readers



def _ensure_eps(parts: dict, eps: float = 0.05) -> dict:
    n = int(parts["pos"].shape[0])
    if "eps" not in parts or parts["eps"] is None:
        parts["eps"] = np.full(n, float(eps), dtype=np.float64)
    return parts


def build_ic(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    has_bulge = _ic_has_bulge(args.ic_path)
    n_tot = _n_total_for_disk(args.n_disk, has_bulge=has_bulge)
    recipe_name = str(getattr(args, "recipe", "paint")).lower().strip()
    print(
        f"=== residual build recipe={recipe_name} "
        f"ic={args.ic_path} morph={args.morph_path} "
        f"α={args.alpha} source={args.morph_source} vel={args.velocity_mode} "
        f"N={n_tot} bulge={has_bulge} ===",
        flush=True,
    )
    if recipe_name in ("full_dyn_replace", "full_dyn_ot", "replace", "ot_lite"):
        mode = (
            "replace"
            if recipe_name in ("full_dyn_replace", "replace")
            else "ot_lite"
        )
        parts = _full_dyn_residual_particles(
            args.ic_path,
            args.morph_path,
            n_resample=n_tot,
            rng=rng,
            mode=mode,
        )
    else:
        teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)
        parts = _residual_f0_particles(
            args.ic_path,
            args.morph_path,
            teacher,
            cfg,
            stats,
            n_resample=n_tot,
            rng=rng,
            morph_source=str(args.morph_source),
            alpha=float(args.alpha),
            dens_resid_kind=str(args.dens_resid_kind),
            dens_resid_other_alpha=float(args.dens_resid_other_alpha),
            dens_resid_midplane_only=bool(args.dens_resid_midplane_only),
            residual_scale=str(args.residual_scale),
            velocity_mode=str(args.velocity_mode),
            velocity_frame=str(args.velocity_frame),
            match_cell_moments=bool(args.match_cell_moments),
            phase_b_ckpt=args.phase_b_ckpt,
            phase_b_hint=str(getattr(args, "phase_b_hint", "deposit")),
            blend_weight=float(args.blend_weight),
            morph_vel_blend_weight=float(args.morph_vel_blend_weight),
            vel_hybrid_r_max=float(args.vel_hybrid_r_max),
            r_weight_peak=args.r_weight_peak,
            r_weight_sigma=args.r_weight_sigma,
            contrast_sharpen=float(args.contrast_sharpen),
            contrast_smooth_kpc=float(args.contrast_smooth_kpc),
            r_weight_floor=float(args.r_weight_floor),
            alpha_mode=str(args.alpha_mode),
            target_a2_rd=args.target_a2_rd,
            a2_r_eval=args.a2_r_eval,
        )
    parts = _ensure_eps(_vcom_only(_stratified_down(parts, n_tot, rng)))
    a2_med = float(_a2_disk(parts))
    a2_rd = (
        float(_a2_disk(parts, r_eval=float(args.a2_r_eval)))
        if args.a2_r_eval is not None
        else None
    )
    disk = parts["component_id"] == 0
    fout = disk_azimuthal_fourier(
        parts["pos"][disk],
        parts["mass"][disk],
        m=2,
        r_max=12.0,
        n_bins=12,
        z_max=0.5,
        min_count=10,
        r_eval=float(args.a2_r_eval) if args.a2_r_eval is not None else None,
    )
    meta = {
        "recipe": (
            RECIPE_FULL_DYN
            if str(getattr(args, "recipe", "paint")).lower().startswith("full_dyn")
            or str(getattr(args, "recipe", "")).lower() in ("replace", "ot_lite")
            else RECIPE_PAINT
        ),
        "recipe_name": str(getattr(args, "recipe", "paint")),
        "alpha": float(parts.get("dens_resid_alpha_used", args.alpha)),
        "alpha_requested": float(args.alpha),
        "alpha_mode": str(args.alpha_mode),
        "target_a2_rd": args.target_a2_rd,
        "morph_source": str(parts.get("morph_source", args.morph_source)),
        "velocity_mode": str(
            parts.get("velocity_meta", {}).get("velocity_mode", args.velocity_mode)
        ),
        "dens_resid_kind": str(args.dens_resid_kind),
        "r_weight_peak": args.r_weight_peak,
        "r_weight_sigma": args.r_weight_sigma,
        "r_weight_floor": float(args.r_weight_floor),
        "contrast_sharpen": float(args.contrast_sharpen),
        "ic_path": str(args.ic_path),
        "morph_path": str(args.morph_path),
        "teacher": str(args.teacher) if args.teacher is not None else None,
        "n_disk": int(args.n_disk),
        "n_total": int(parts["pos"].shape[0]),
        "a2_median": a2_med,
        "a2_at_r": a2_rd if a2_rd is not None else (
            float(fout["a_m_over_a0_at_r"])
            if args.a2_r_eval is not None
            else None
        ),
        "a2_r_eval": args.a2_r_eval,
        "a2_primary": (
            "A2(R_d)" if args.a2_r_eval is not None else "A2_median"
        ),
        "retain_components": parts.get("retain_components"),
        "phase": parts.get("phase"),
    }
    primary = (
        f"A₂(R={args.a2_r_eval})={meta['a2_at_r']:.3f}"
        if meta["a2_at_r"] is not None
        else f"A₂(median)={a2_med:.3f}"
    )
    print(
        f"  IC {primary}  (median={a2_med:.3f}"
        + (
            f", α_used={meta['alpha']:.3f}"
            if abs(meta["alpha"] - float(args.alpha)) > 1e-6
            else ""
        )
        + ")",
        flush=True,
    )
    return {"parts": parts, "meta": meta}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ic-path", type=Path, required=True)
    p.add_argument("--morph-path", type=Path, required=True)
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--recipe",
        choices=("paint", "full_dyn_replace", "full_dyn_ot"),
        default="paint",
        help=(
            "paint=m2 residual on f0 axisym (legacy); "
            "full_dyn_replace=data disk ⊕ f0 halo/bulge; "
            "full_dyn_ot=OT-lite f0→data disk ⊕ f0 halo/bulge."
        ),
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--dens-resid-kind", choices=("m2", "full"), default="m2")
    p.add_argument("--dens-resid-other-alpha", type=float, default=0.0)
    p.add_argument("--dens-resid-midplane-only", action="store_true")
    p.add_argument(
        "--morph-source",
        choices=("teacher_recon", "deposit", "phase_b", "blend"),
        default="teacher_recon",
    )
    p.add_argument(
        "--residual-phase",
        choices=("A", "B"),
        default="A",
        help="A=hand morph residual; B=learned CondDelta (sets --morph-source phase_b).",
    )
    p.add_argument(
        "--phase-b-ckpt",
        type=Path,
        default=None,
        help="Phase-B delta_midplane.pt when --morph-source phase_b / --residual-phase B.",
    )
    p.add_argument(
        "--phase-b-hint",
        choices=("deposit", "teacher", "blend"),
        default="blend",
        help="Morph dens chart fed as CondDelta hint channel (default blend≈Phase A morph).",
    )
    p.add_argument(
        "--blend-weight",
        type=float,
        default=0.5,
        help="Deposit weight for --morph-source blend / phase_b hint=blend (0=teacher, 1=deposit).",
    )
    p.add_argument(
        "--velocity-mode",
        choices=(
            "transplant",
            "moments",
            "morph_transplant",
            "morph_blend",
            "morph_hybrid",
            "f0",
            "morph",
            "blend",
            "hybrid",
        ),
        default="transplant",
        help=(
            "Disk velocity assignment. Prefer --vel-source. "
            "f0/transplant=GalactICS kNN; morph/morph_transplant=barred dump kNN; "
            "blend=linear mix; hybrid=morph inside --vel-hybrid-r-max, f0 outside."
        ),
    )
    p.add_argument(
        "--vel-source",
        choices=("f0", "morph", "blend", "hybrid"),
        default=None,
        help=(
            "Alias for --velocity-mode: f0→transplant, morph→morph_transplant, "
            "blend→morph_blend, hybrid→morph_hybrid. Overrides --velocity-mode when set."
        ),
    )
    p.add_argument(
        "--morph-vel-blend-weight",
        type=float,
        default=0.5,
        help="Morph weight for --velocity-mode morph_blend / --vel-source blend (0=f0, 1=morph).",
    )
    p.add_argument(
        "--vel-hybrid-r-max",
        type=float,
        default=5.0,
        help="For --vel-source hybrid: morph velocities for R<=this [kpc], f0 outside.",
    )
    p.add_argument(
        "--residual-scale",
        choices=("multiplicative", "additive"),
        default="multiplicative",
    )
    p.add_argument("--velocity-frame", choices=("cartesian", "cylindrical"), default="cartesian")
    p.add_argument("--match-cell-moments", action="store_true")
    p.add_argument("--n-disk", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--a2-r-eval",
        type=float,
        default=2.0,
        help="Primary bar metric: A₂(R) at this radius [kpc] (default R_d=2).",
    )
    p.add_argument(
        "--r-weight-peak",
        type=float,
        default=-1.0,
        help="Radial Gaussian peak [kpc] for contrast weight (default -1=off; "
        "set e.g. 2 for R_d preference with --r-weight-floor).",
    )
    p.add_argument(
        "--r-weight-sigma",
        type=float,
        default=2.5,
        help="Radial Gaussian σ [kpc] for contrast weight (default 2.5).",
    )
    p.add_argument(
        "--r-weight-floor",
        type=float,
        default=0.4,
        help="Min radial weight outside peak (0=hard Gaussian; default 0.4).",
    )
    p.add_argument(
        "--contrast-sharpen",
        type=float,
        default=1.0,
        help="Unsharp-mask amount on morph contrast (0=off; default 1.0).",
    )
    p.add_argument(
        "--contrast-smooth-kpc",
        type=float,
        default=1.5,
        help="Unsharp smooth scale [kpc].",
    )
    p.add_argument(
        "--alpha-mode",
        choices=("fixed", "match_a2_rd"),
        default="match_a2_rd",
        help="fixed: --alpha; match_a2_rd: search α so map A₂(R_d)≈--target-a2-rd.",
    )
    p.add_argument(
        "--target-a2-rd",
        type=float,
        default=0.49,
        help="Target A₂(R_d) for --alpha-mode match_a2_rd (906c4 morph≈0.49).",
    )
    p.add_argument("--save-ic", action="store_true", help="Write residual_ic.npz")
    p.add_argument("--evolve-gyr", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--force", type=str, default="gpu_bh")
    p.add_argument("--omp", type=int, default=2)
    p.add_argument("--faceon-times", type=str, default="0,0.12,0.25,0.38,0.5,2.0")
    p.add_argument("--reuse-data-from", type=Path, default=None)
    p.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Evolve 'data' arm dump (default: --morph-path). Use quiet IC for no-bulge control.",
    )
    p.add_argument(
        "--data-arm-label",
        type=str,
        default=None,
        help=(
            "Legend/caption for evolve data arm. Default: 'quiet IC (control)' "
            "when --data-path is set and differs from --morph-path; else 'data dump'."
        ),
    )
    p.add_argument("--paper-figures", type=Path, default=None)
    p.add_argument("--paper-prefix", type=str, default=None)
    p.add_argument("--skip-data", action="store_true")
    args = p.parse_args()
    recipe_name = str(args.recipe).lower().strip()
    is_full_dyn = recipe_name.startswith("full_dyn")
    # Teacher required for paint recipe; also for evolve gate plumbing even on full_dyn.
    if args.teacher is None and (not is_full_dyn or float(args.evolve_gyr) > 0):
        args.teacher = default_teacher()
    active_recipe = RECIPE_FULL_DYN if is_full_dyn else RECIPE_PAINT
    # --vel-source overrides --velocity-mode with the user-facing alias.
    if args.vel_source is not None:
        _vel_map = {
            "f0": "transplant",
            "morph": "morph_transplant",
            "blend": "morph_blend",
            "hybrid": "morph_hybrid",
        }
        args.velocity_mode = _vel_map[str(args.vel_source)]
    # --residual-phase B is an alias that forces learned morph_source.
    if str(args.residual_phase).upper() == "B":
        args.morph_source = "phase_b"
        if args.phase_b_ckpt is None:
            raise SystemExit(
                "--residual-phase B requires --phase-b-ckpt path/to/delta_midplane.pt"
            )
    # Sentinel: negative peak disables radial weight.
    if args.r_weight_peak is not None and float(args.r_weight_peak) < 0:
        args.r_weight_peak = None
        args.r_weight_sigma = None
    if args.a2_r_eval is not None and float(args.a2_r_eval) < 0:
        args.a2_r_eval = None

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "RECIPE.json").write_text(json.dumps(active_recipe, indent=2) + "\n")

    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]

    built = build_ic(args)
    (args.out / "ic_meta.json").write_text(json.dumps(built["meta"], indent=2) + "\n")
    # Always persist residual_ic.npz (t=0 dens+vel scoring / restarts).
    parts = built["parts"]
    np.savez_compressed(
        args.out / "residual_ic.npz",
        pos=parts["pos"],
        vel=parts["vel"],
        mass=parts["mass"],
        eps=parts["eps"],
        component_id=parts["component_id"],
    )
    print(f"  wrote {args.out / 'residual_ic.npz'}", flush=True)

    if args.evolve_gyr <= 0:
        print("=== build-only done ===", flush=True)
        return

    # Delegate evolve gate to evolve_component_slices (same maps/profiles/paper figs).
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evolve_component_slices.py"),
    ]
    if args.teacher is not None:
        cmd.extend(["--teacher", str(args.teacher)])
    cmd.extend(
        [
            "--data-path",
            str(args.data_path if args.data_path is not None else args.morph_path),
            "--ic-path",
            str(args.ic_path),
            "--out",
            str(args.out),
            "--n-disk",
            str(args.n_disk),
            "--evolve-gyr",
            str(args.evolve_gyr),
            "--dt",
            str(args.dt),
            "--omp",
            str(args.omp),
            "--force",
            str(args.force),
            "--methods",
            "",
            "--with-residual-f0",
            "--residual-recipe",
            str(args.recipe),
            "--residual-morph-source",
            str(args.morph_source),
            "--residual-velocity-mode",
            str(args.velocity_mode),
            "--dens-resid-kind",
            str(args.dens_resid_kind),
            "--dens-resid-alpha",
            str(args.alpha),
            "--dens-resid-other-alpha",
            str(args.dens_resid_other_alpha),
            "--residual-scale",
            str(args.residual_scale),
            "--residual-morph-path",
            str(args.morph_path),
            "--faceon-times",
            str(args.faceon_times),
            "--skip-recon",
        ]
    )
    if args.dens_resid_midplane_only:
        cmd.append("--dens-resid-midplane-only")
    if args.reuse_data_from is not None:
        cmd.extend(["--reuse-data-from", str(args.reuse_data_from)])
    if args.skip_data:
        cmd.append("--skip-data")
    data_arm_label = args.data_arm_label
    if data_arm_label is None and args.data_path is not None:
        morph = Path(args.morph_path).resolve()
        data = Path(args.data_path).resolve()
        if data != morph:
            data_arm_label = "quiet IC (control)"
    if data_arm_label:
        cmd.extend(["--data-arm-label", str(data_arm_label)])
    if args.paper_figures is not None:
        cmd.extend(["--paper-figures", str(args.paper_figures)])
    if args.paper_prefix:
        cmd.extend(["--paper-prefix", str(args.paper_prefix)])
    if args.a2_r_eval is not None:
        cmd.extend(["--a2-r-eval", str(args.a2_r_eval)])
    if args.r_weight_peak is not None:
        cmd.extend(["--residual-r-weight-peak", str(args.r_weight_peak)])
    if args.r_weight_sigma is not None:
        cmd.extend(["--residual-r-weight-sigma", str(args.r_weight_sigma)])
    cmd.extend(["--residual-r-weight-floor", str(args.r_weight_floor)])
    if float(args.contrast_sharpen) > 0:
        cmd.extend(["--residual-contrast-sharpen", str(args.contrast_sharpen)])
        cmd.extend(["--residual-contrast-smooth-kpc", str(args.contrast_smooth_kpc)])
    cmd.extend(["--residual-alpha-mode", str(args.alpha_mode)])
    if args.target_a2_rd is not None:
        cmd.extend(["--residual-target-a2-rd", str(args.target_a2_rd)])
    if args.phase_b_ckpt is not None:
        cmd.extend(["--phase-b-ckpt", str(args.phase_b_ckpt)])
        cmd.extend(["--phase-b-hint", str(args.phase_b_hint)])
    if str(args.morph_source) in ("blend", "phase_b"):
        cmd.extend(["--residual-blend-weight", str(args.blend_weight)])
    cmd.extend(["--velocity-frame", str(args.velocity_frame)])
    if args.match_cell_moments:
        cmd.append("--match-cell-moments")
    if str(args.velocity_mode) in ("morph_blend", "blend"):
        cmd.extend(
            ["--residual-morph-vel-blend-weight", str(args.morph_vel_blend_weight)]
        )
    if str(args.velocity_mode) in ("morph_hybrid", "hybrid"):
        cmd.extend(["--residual-vel-hybrid-r-max", str(args.vel_hybrid_r_max)])
    cmd.extend(["--seed", str(args.seed)])
    # Always persist IC + request final particles for kinetic scoring.
    cmd.append("--save-final-particles")

    print("=== evolve gate ===", flush=True)
    print(" ", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    if rc != 0:
        raise SystemExit(rc)
    # Stamp recipe into verdict if present.
    verdict_path = args.out / "verdict.json"
    if verdict_path.is_file():
        verdict = json.loads(verdict_path.read_text())
        verdict["residual_recipe"] = active_recipe
        verdict["residual_ic_meta"] = built["meta"]
        verdict_path.write_text(json.dumps(verdict, indent=2) + "\n")
    print("=== residual evolve done ===", flush=True)


if __name__ == "__main__":
    main()
