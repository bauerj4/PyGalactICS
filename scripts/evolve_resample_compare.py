#!/usr/bin/env python3
"""Long evolve compare: data dump vs teacher recon vs library generative ICs.

Arms (shared global COM / VCOM conventions):
  * ``data`` — stratified subsample of a barred corpus dump
  * ``recon`` — encode→decode dens/moments with frozen teacher → resample
  * ``amplify_knn_hybrid`` / ``z_amplify`` — feature-library generative ICs

Proper integrator: ``bh_c`` leapfrog, ``dt≈0.01``, many steps (≳0.5 Gyr when
wall-time allows). Metrics: A₂(t), Am(R) snapshots, COM drift, face-on panels.

Example::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \\
      .venv/bin/python scripts/evolve_resample_compare.py \\
        --teacher runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt \\
        --evolve-gyr 0.50 --dt 0.01 --n-evolve 25000 --omp 2

Corpus-scale morphology (N≈1e6, multi-time face-on)::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 \\
      .venv/bin/python scripts/evolve_resample_compare.py \\
        --teacher runs/ml/field_maps/fft_morph_ft_long_2026-07-25/multitower_slice_ae.pt \\
        --out runs/ml/field_maps/evolve_compare_1e6_2026-07-25 \\
        --evolve-gyr 0.50 --dt 0.01 --n-evolve 1000000 --n-resample 1000000 \\
        --methods amplify_knn_hybrid,z_amplify --faceon-times 0,0.125,0.25,0.375,0.5 \\
        --faceon-bins 160 --omp 6 --timeout-s 7200 \\
        --paper-figures papers/mnras_noneq_ics/figures --paper-prefix fig_evolve_1e6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from smoke_field_maps import _am_profiles  # noqa: E402

from galacticsics.ml.fields.binning import bin_multiscale_slice_stacks  # noqa: E402
from galacticsics.ml.fields.feature_library import load_frozen_teacher_bundle  # noqa: E402
from galacticsics.ml.fields.frame import prepare_shared_frame  # noqa: E402
from galacticsics.ml.fields.normalize import denormalize_stack, normalize_stack  # noqa: E402
from galacticsics.ml.fields.resample import resample_particles_from_multiscale  # noqa: E402
from galacticsics.ml.morton.polygon import _component_ids  # noqa: E402
from ntropy.analysis.disk_density import bin_plane_density, disk_azimuthal_fourier  # noqa: E402


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
        # Prefer explicit component_id (generative ICs / remassed dumps).
        # Falling back to tags/type_id alone treats missing metadata as all-disk
        # and silently drops halo/bulge when stratifying — collapsing A₂.
        if "component_id" in data.files:
            cid = np.asarray(data["component_id"], dtype=np.int64)
        else:
            cid = _component_ids(data.get("tags"), data.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    fr = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
    name_to_id = {"disk": 0, "halo": 1, "bulge": 2}
    # Renormalize over present components so disk+halo (bulge=0) keeps 2:1
    # rather than starving the disk under a 4:2:1 schedule.
    present = {
        name: f for name, f in fr.items() if np.any(cid == name_to_id[name])
    }
    wsum = sum(present.values()) or 1.0
    parts = []
    for name, f in present.items():
        nk = max(1, int(round(n * (f / wsum))))
        mask = cid == name_to_id[name]
        idx = np.where(mask)[0]
        take = rng.choice(idx, size=min(nk, idx.size), replace=idx.size < nk)
        parts.append(take)
    sel = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if sel.size > n:
        sel = rng.choice(sel, size=n, replace=False)
    return {
        "pos": pos[sel],
        "vel": vel[sel],
        "mass": mass[sel],
        "eps": eps[sel],
        "component_id": cid[sel],
    }


def _ic_has_bulge(ic_path: Path) -> bool:
    """True if GalactICS / residual IC contains bulge particles."""
    with np.load(ic_path, allow_pickle=True) as data:
        if "tags" in data.files:
            tags = np.asarray(data["tags"]).astype(str)
            return bool(np.any(tags == "bulge"))
        if "component_id" in data.files:
            cid = np.asarray(data["component_id"])
            return bool(np.any(cid == 2))
        if "type_id" in data.files:
            # Legacy GalactICS encoding: type_id==2 is bulge when tags absent.
            tid = np.asarray(data["type_id"]).ravel()
            return bool(np.any(tid == 2))
    return True


def _n_total_for_disk(n_disk: int, *, has_bulge: bool) -> int:
    """disk:halo:bulge = 4:2:1 with bulge; disk:halo = 2:1 without."""
    n_disk = int(n_disk)
    if has_bulge:
        return int(round(n_disk * 7 / 4))
    return int(round(n_disk * 3 / 2))


def _stratified_down(parts: dict, n: int, rng: np.random.Generator) -> dict:
    cid = parts["component_id"]
    fr = {0: 4 / 7, 1: 2 / 7, 2: 1 / 7}
    present = {c: f for c, f in fr.items() if np.any(cid == c)}
    wsum = sum(present.values()) or 1.0
    takes = []
    for c, f in present.items():
        idx = np.where(cid == c)[0]
        nk = max(1, int(round(n * f / wsum)))
        nk = min(nk, idx.size)
        takes.append(rng.choice(idx, size=nk, replace=False))
    take = np.concatenate(takes) if takes else np.zeros(0, dtype=np.int64)
    if take.size > n:
        take = rng.choice(take, size=n, replace=False)
    out = {
        k: (
            v[take]
            if isinstance(v, np.ndarray) and getattr(v, "shape", (0,))[0] == cid.shape[0]
            else v
        )
        for k, v in parts.items()
    }
    out["eps"] = np.full(out["pos"].shape[0], 0.1, dtype=np.float64)
    return out


def _vcom_only(parts: dict) -> dict:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in parts.items()}
    m = out["mass"]
    out["vel"] = out["vel"] - np.average(out["vel"], axis=0, weights=m)
    return out


def _full_com(parts: dict) -> dict:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in parts.items()}
    m = out["mass"]
    out["pos"] = out["pos"] - np.average(out["pos"], axis=0, weights=m)
    out["vel"] = out["vel"] - np.average(out["vel"], axis=0, weights=m)
    return out


def _metrics(parts: dict) -> dict:
    a2 = _a2_disk(parts)
    com = np.average(parts["pos"], axis=0, weights=parts["mass"])
    return {"a2": float(a2), "com": com.tolist(), "n": int(parts["pos"].shape[0])}


def _faceon(
    parts: dict,
    n_bins: int = 96,
    half: float = 12.0,
    pos: np.ndarray | None = None,
) -> np.ndarray:
    disk = parts["component_id"] == 0
    p = parts["pos"] if pos is None else pos
    return bin_plane_density(
        p[disk],
        parts["mass"][disk],
        axes=(0, 1),
        n_bins=n_bins,
        half_extent=half,
    ).density


def _faceon_from_pos(
    pos: np.ndarray,
    mass: np.ndarray,
    cid: np.ndarray,
    *,
    n_bins: int,
    half: float,
) -> np.ndarray:
    disk = cid == 0
    return bin_plane_density(
        pos[disk],
        mass[disk],
        axes=(0, 1),
        n_bins=n_bins,
        half_extent=half,
    ).density


def _evolve_tracked(
    parts: dict,
    *,
    end_gyr: float,
    dt: float,
    omp: int,
    timeout_s: float,
    n_track: int = 11,
    faceon_times: list[float] | None = None,
    faceon_bins: int = 128,
    faceon_half: float = 12.0,
    force: str | None = None,
) -> dict:
    """Leapfrog evolve with A₂(t) / COM samples (wrapper around force loop)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    os.environ["OMP_NUM_THREADS"] = str(max(1, int(omp)))
    n_steps = max(1, int(np.ceil(float(end_gyr) / max(float(dt), 1e-9))))
    pos0 = np.asarray(parts["pos"], dtype=np.float64)
    vel0 = np.asarray(parts["vel"], dtype=np.float64)
    mass = np.asarray(parts["mass"], dtype=np.float64)
    eps = np.asarray(parts["eps"], dtype=np.float64)
    cid = np.asarray(parts["component_id"])
    try:
        from ntropy.forces.gpu_bh import gpu_bh_available as _gba

        default = "gpu_bh" if _gba() else "bh_c"
    except Exception:  # noqa: BLE001
        default = "bh_c"
    method = (force or default).strip().lower()
    if method not in ("gpu_bh", "bh_c", "bh"):
        raise ValueError(f"unknown force backend {method!r}")
    # Face-on snapshot steps (always include t=0 and end; plus requested times).
    want_t = [0.0, float(end_gyr)]
    if faceon_times:
        want_t.extend(float(t) for t in faceon_times)
    want_t = sorted({min(max(0.0, t), float(end_gyr)) for t in want_t})
    face_steps = sorted(
        {
            0 if t <= 0 else n_steps if t >= float(end_gyr) else int(round(t / max(dt, 1e-9)))
            for t in want_t
        }
    )

    def _forces(p):
        nonlocal method
        if method == "gpu_bh":
            try:
                from ntropy.forces.gpu_bh import compute_forces_gpu_bh, gpu_bh_available

                if gpu_bh_available():
                    return compute_forces_gpu_bh(p, mass, eps, theta=0.8)
                print("  gpu_bh_available=False → falling back to bh_c", flush=True)
                method = "bh_c"
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  gpu_bh FAILED ({type(exc).__name__}: {exc}) → bh_c",
                    flush=True,
                )
                method = "bh_c"
        if method == "bh_c":
            try:
                from ntropy.forces.bhtree_c import compute_forces_bh_c, extension_available

                if extension_available():
                    return compute_forces_bh_c(p, mass, eps, theta=0.8)
            except Exception:  # noqa: BLE001
                method = "bh"
        import importlib

        bhtree = importlib.import_module("ntropy.forces.bhtree")
        return bhtree.compute_forces_bh(p, mass, eps, theta=0.8)

    # Probe force backend once up front so logs/verdicts record the real method.
    t_force0 = time.time()
    _ = _forces(pos0)
    force_probe_s = time.time() - t_force0
    print(
        f"  force backend={method} N={len(pos0)} probe={force_probe_s:.3f}s "
        f"(~{force_probe_s:.2f} s/step)",
        flush=True,
    )

    track_steps = sorted(
        {0, n_steps}
        | {int(round(i * n_steps / max(n_track - 1, 1))) for i in range(n_track)}
        | set(face_steps)
    )

    def _a2_now(pos):
        disk = cid == 0
        return float(
            disk_azimuthal_fourier(
                pos[disk],
                mass[disk],
                m=2,
                r_max=12.0,
                n_bins=12,
                z_max=0.5,
                min_count=10,
            )["a_m_over_a0_median"]
        )

    def _run():
        t0 = time.time()
        pos = pos0.copy()
        vel = vel0.copy()
        times, a2s, coms = [], [], []
        face_t, face_maps = [], []
        face_set = set(face_steps)

        def _record(step: int):
            times.append(step * dt if step else 0.0)
            a2s.append(_a2_now(pos))
            coms.append(float(np.linalg.norm(np.average(pos, axis=0, weights=mass))))
            if step in face_set:
                face_t.append(step * dt if step else 0.0)
                face_maps.append(
                    _faceon_from_pos(
                        pos, mass, cid, n_bins=faceon_bins, half=faceon_half
                    )
                )

        _record(0)
        acc = _forces(pos)
        vel = vel + 0.5 * dt * acc
        next_i = 1
        for step in range(1, n_steps + 1):
            pos = pos + dt * vel
            acc = _forces(pos)
            vel = vel + dt * acc
            if next_i < len(track_steps) and step == track_steps[next_i]:
                _record(step)
                next_i += 1
        vel = vel - 0.5 * dt * acc
        return pos, vel, time.time() - t0, times, a2s, coms, face_t, face_maps

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            (
                pos_f,
                vel_f,
                wall,
                times,
                a2s,
                coms,
                face_t,
                face_maps,
            ) = ex.submit(_run).result(timeout=float(timeout_s))
    except FuturesTimeout:
        return {
            "ok": False,
            "timed_out": True,
            "timeout_s": float(timeout_s),
            "n_steps": n_steps,
            "error": f"evolve timed out after {timeout_s}s",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "timed_out": False,
            "n_steps": n_steps,
            "error": f"{type(exc).__name__}: {exc}",
        }

    com0 = np.average(pos0, axis=0, weights=mass)
    com1 = np.average(pos_f, axis=0, weights=mass)
    return {
        "ok": True,
        "wall_s": wall,
        "n_steps": n_steps,
        "force_method": method,
        "com_drift_kpc": float(np.linalg.norm(com1 - com0)),
        "pos_final": np.asarray(pos_f, dtype=np.float64),
        "vel_final": np.asarray(vel_f, dtype=np.float64),
        "t_gyr": [float(t) for t in times],
        "a2_t": [float(a) for a in a2s],
        "com_norm_t": [float(c) for c in coms],
        "faceon_t_gyr": [float(t) for t in face_t],
        "faceon_maps": face_maps,
    }


def _recon_particles(
    path: Path,
    teacher,
    cfg,
    stats,
    *,
    n_resample: int,
    rng: np.random.Generator,
    retain_components: tuple[str, ...] = (),
    bulge_method: str = "slab",
    dens_source: str = "recon",
    dens_components: tuple[str, ...] = ("disk", "halo", "bulge"),
    mass_calibrate_dens: bool = True,
    dens_resid_alpha: float = 1.0,
    dens_resid_components: tuple[str, ...] = ("disk",),
    dens_resid_mode: str = "fixed",
    dens_resid_midplane_only: bool = False,
    dens_resid_target_a2: float | None = None,
    dens_resid_kind: str = "full",
    dens_resid_other_alpha: float = 1.0,
    moments_source: str = "auto",
    velocity_frame: str = "cartesian",
    match_cell_moments: bool = False,
) -> dict:
    """
    FFT teacher encode→decode→resample.

    ``bulge_method``:
      - ``slab`` — all components from tower dens maps (default; soft cusp)
      - ``spherical_shells`` — disk/halo from towers; bulge from spherical
        shell dens/moments of the source dump (cusp-preserving)
      - combined with ``retain_components`` to keep exact GalactICS particles

    ``dens_source``:
      - ``recon`` — AE dens (default; soft bar morphology)
      - ``input`` — deposited dens from the dump + AE moments (closes IC A₂ gap)
      - ``deposit`` — deposited dens+moments (oracle upper bound for recon ICs)

    ``mass_calibrate_dens``: rescale dens channels so ∫ dens dA matches true
    component mass before resample (uniform gain; shape unchanged).

    ``dens_resid_alpha``: amplify non-axisym dens residual (α=1 off).

    ``dens_resid_mode``:
      - ``fixed`` — use dens_resid_alpha
      - ``residual_power`` — match midplane residual RMS to deposit (often α≈1)
      - ``map_a2`` — binary-search α so midplane dens map A₂ ≈ target
        (``dens_resid_target_a2`` or deposited map A₂)

    ``moments_source``: ``auto`` / ``deposit`` / ``recon`` moment channel swap.

    ``dens_resid_kind``: ``full`` (legacy residual amplify) or ``m2`` (amplify
    only m=2 Fourier projection; leave other residual at dens_resid_other_alpha).
    """
    from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
    from galacticsics.ml.fields.resample import (
        alpha_match_map_a2,
        alpha_match_residual_power,
        amplify_axisym_residual_dens,
        bin_spherical_shell_moments,
        copy_dens_channels,
        fuse_shell_bulge_with_multiscale,
        rescale_dens_channels_to_mass,
        stitch_retained_components,
    )

    with np.load(path, allow_pickle=True) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        vel = np.asarray(data["vel"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)
        cid = _component_ids(data.get("tags"), data.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
    binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    maps = {k: v[0] for k, v in binned.items()}
    device = next(teacher.parameters()).device
    stacks = {
        k: torch.as_tensor(
            normalize_stack(v, stats[k])[None], dtype=torch.float32, device=device
        )
        for k, v in maps.items()
    }
    with torch.no_grad():
        pred = teacher(stacks)
        pred_np = {k: v.detach().cpu().numpy()[0] for k, v in pred.items()}
    recon = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}
    if dens_source == "deposit":
        fields = maps
    elif dens_source == "input":
        fields = copy_dens_channels(
            maps, recon, cfg=cfg, components=tuple(dens_components)
        )
    elif dens_source == "recon":
        fields = recon
    else:
        raise ValueError(f"unknown dens_source={dens_source!r}")

    mass_totals = {
        "disk": float(mass[cid == 0].sum()) if np.any(cid == 0) else 0.0,
        "halo": float(mass[cid == 1].sum()) if np.any(cid == 1) else 0.0,
        "bulge": float(mass[cid == 2].sum()) if np.any(cid == 2) else 0.0,
    }
    alpha_used = float(dens_resid_alpha)
    if dens_source == "recon" and dens_resid_mode in ("residual_power", "map_a2"):
        g = cfg.grid_for("disk")
        di = g.moment_keys.index("dens")
        iz0 = g.n_z // 2
        dens_ae = fields["disk"][iz0 * g.n_mom + di]
        dens_ref = maps["disk"][iz0 * g.n_mom + di]
        if dens_resid_mode == "residual_power":
            alpha_used = alpha_match_residual_power(dens_ae, dens_ref)
        else:
            if dens_resid_target_a2 is not None:
                tgt = float(dens_resid_target_a2)
            else:
                tgt = float(
                    dens_map_azimuthal_fourier_numpy(
                        dens_ref, m=2, n_bins=12, r_max=g.r_max
                    )["a_m_over_a0_median"]
                )
            alpha_used = alpha_match_map_a2(
                dens_ae, target_a2=tgt, r_max=float(g.r_max)
            )
    if dens_source == "recon" and (
        abs(float(alpha_used) - 1.0) > 1e-12
        or abs(float(dens_resid_other_alpha) - 1.0) > 1e-12
    ):
        fields = amplify_axisym_residual_dens(
            fields,
            cfg=cfg,
            components=tuple(dens_resid_components),
            alpha=float(alpha_used),
            midplane_only=bool(dens_resid_midplane_only),
            mode=str(dens_resid_kind),
            other_alpha=float(dens_resid_other_alpha),
        )
    # Optional moment swap for ablations (AE dens + deposit moments, etc.).
    ms = str(moments_source)
    if ms == "deposit" and dens_source == "recon":
        fields = copy_dens_channels(
            fields, maps, cfg=cfg, components=tuple(dens_resid_components)
        )
    elif ms == "recon" and dens_source in ("input", "deposit"):
        fields = copy_dens_channels(
            fields if dens_source == "input" else maps,
            recon,
            cfg=cfg,
            components=tuple(dens_components),
        )
    if mass_calibrate_dens and dens_source in ("recon", "input"):
        fields = rescale_dens_channels_to_mass(
            fields, cfg=cfg, mass_total_per_component=mass_totals
        )

    if bulge_method == "spherical_shells":
        mask_b = cid == 2
        if not np.any(mask_b):
            raise ValueError("no bulge particles for spherical_shells recon")
        com_b = np.average(pos[mask_b], axis=0, weights=mass[mask_b])
        shells = bin_spherical_shell_moments(
            pos[mask_b] - com_b,
            vel[mask_b],
            mass[mask_b],
            n_shells=64,
            r_min=0.05,
            r_max=6.0,
            log_bins=True,
        )
        parts = fuse_shell_bulge_with_multiscale(
            fields,
            cfg=cfg,
            bulge_shells=shells,
            n_particles=n_resample,
            count_fractions=COUNT,
            mass_total_per_component=mass_totals,
            rng=rng,
            velocity_frame=velocity_frame,
            match_cell_moments=match_cell_moments,
        )
        bmask = parts["component_id"] == 2
        parts["pos"][bmask] = parts["pos"][bmask] + com_b
    else:
        # Prefer true snapshot masses so AE dens under-prediction does not
        # silently rescale dynamics; slab dens only sets spatial sampling.
        parts = resample_particles_from_multiscale(
            fields,
            cfg=cfg,
            n_particles=n_resample,
            count_fractions=COUNT,
            mass_total_per_component=mass_totals,
            rng=rng,
            velocity_frame=velocity_frame,
            match_cell_moments=match_cell_moments,
        )
    if retain_components:
        n_per = dict(parts.get("n_per_component") or {})
        parts = stitch_retained_components(
            parts,
            source_pos=pos,
            source_vel=vel,
            source_mass=mass,
            source_cid=cid,
            retain=tuple(retain_components),
            n_retain={c: int(n_per.get(c, 0)) for c in retain_components},
            rng=rng,
        )
    parts["dens_resid_alpha_used"] = float(alpha_used)
    parts["dens_resid_mode"] = str(dens_resid_mode)
    parts["dens_resid_midplane_only"] = bool(dens_resid_midplane_only)
    parts["moments_source"] = str(moments_source)
    parts["velocity_frame"] = str(velocity_frame)
    parts["match_cell_moments"] = bool(match_cell_moments)
    return parts


def _residual_f0_particles(
    ic_path: Path,
    morph_path: Path,
    teacher,
    cfg,
    stats,
    *,
    n_resample: int,
    rng: np.random.Generator,
    morph_source: str = "teacher_recon",
    alpha: float = 1.0,
    dens_resid_kind: str = "m2",
    dens_resid_other_alpha: float = 0.0,
    dens_resid_midplane_only: bool = False,
    keep_f0_residual: float = 1.0,
    residual_scale: str = "multiplicative",
    preserve_axisym: bool = True,
    factor_floor: float = 0.05,
    contrast_from_midplane: bool = True,
    match_f0_radial_cdf: bool = True,
    retain_components: tuple[str, ...] = ("halo", "bulge"),
    velocity_mode: str = "transplant",
    mass_calibrate_dens: bool = True,
    velocity_frame: str = "cartesian",
    match_cell_moments: bool = False,
    phase_b_ckpt: Path | None = None,
    phase_b_hint: str = "deposit",
    blend_weight: float = 0.5,
    morph_vel_blend_weight: float = 0.5,
    vel_hybrid_r_max: float = 5.0,
    r_weight_peak: float | None = None,
    r_weight_sigma: float | None = None,
    contrast_sharpen: float = 0.0,
    contrast_smooth_kpc: float = 1.5,
    r_weight_floor: float = 0.0,
    alpha_mode: str = "fixed",
    target_a2_rd: float | None = None,
    a2_r_eval: float | None = None,
) -> dict:
    """
    Phase-A residual around GalactICS ``f0``.

    1. Deposit GalactICS IC (``ic_path``) → ``maps_f0``.
    2. Build morph dens chart from ``morph_path`` (evolved dump / teacher source):
       ``teacher_recon`` encode→decode, or ``deposit`` of that dump.
    3. Inject m2/full morph residual onto ``f0`` dens
       (``inject_morph_residual_on_f0_dens``, default multiplicative contrast);
       keep ``f0`` moment channels.
    4. Resample disk (and any non-retained comps) from residual dens + f0 moments.
    5. Retain halo/bulge GalactICS particles; optionally kNN-transplant GalactICS
       disk velocities onto morph positions (``velocity_mode=transplant``).

    Honest Phase A: hand/library dens residual on ``f0`` (no trained δ network).
    """
    from galacticsics.ml.fields.resample import (
        alpha_match_predicted_contrast_a2_rd,
        alpha_match_residual_a2_rd,
        bin_spherical_shell_moments,
        fuse_shell_bulge_with_multiscale,
        inject_morph_residual_on_f0_dens,
        inject_predicted_contrast_on_f0_dens,
        match_disk_radial_cdf_to_reference,
        rescale_dens_channels_to_mass,
        resample_particles_from_multiscale,
        stitch_retained_components,
        transplant_velocities_knn,
    )

    def _load_frame(path: Path):
        with np.load(path, allow_pickle=True) as data:
            pos = np.asarray(data["pos"], dtype=np.float64)
            vel = np.asarray(data["vel"], dtype=np.float64)
            mass = np.asarray(data["mass"], dtype=np.float64)
            cid = _component_ids(data.get("tags"), data.get("type_id"), pos.shape[0])
        pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
        return pos, vel, mass, cid

    pos0, vel0, mass0, cid0 = _load_frame(ic_path)
    pos_morph, vel_morph, mass_morph, cid_morph = _load_frame(morph_path)

    maps_f0 = {
        k: v[0]
        for k, v in bin_multiscale_slice_stacks(
            pos0, vel0, mass0, cid0, cfg=cfg
        ).items()
    }
    maps_dep_m = {
        k: v[0]
        for k, v in bin_multiscale_slice_stacks(
            pos_morph, vel_morph, mass_morph, cid_morph, cfg=cfg
        ).items()
    }

    ms = str(morph_source).lower().strip()
    if ms in ("phase_b", "learned_delta", "delta"):
        if phase_b_ckpt is None:
            raise ValueError("morph_source=phase_b requires phase_b_ckpt")
        # Phase B: predict midplane contrast; CondDelta uses morph dump as hint.
        from galacticsics.ml.fields.residual_delta import (
            load_delta_checkpoint,
            midplane_contrast_from_dens,
            predict_contrast,
        )

        g_disk = cfg.grid_for("disk")
        dens_i = g_disk.moment_keys.index("dens")
        n_mom = len(g_disk.moment_keys)
        iz = g_disk.n_z // 2
        dens0 = np.asarray(maps_f0["disk"][iz * n_mom + dens_i], dtype=np.float32)
        dens_m = np.asarray(
            maps_dep_m["disk"][iz * n_mom + dens_i], dtype=np.float32
        )
        # Morph dens hint for CondDelta (deposit / teacher / blend).
        hint_mode = str(phase_b_hint).lower().strip()
        dens_hint = dens_m
        if hint_mode in ("teacher", "blend", "teacher_recon"):
            device = next(teacher.parameters()).device
            stacks = {
                k: torch.as_tensor(
                    normalize_stack(v, stats[k])[None],
                    dtype=torch.float32,
                    device=device,
                )
                for k, v in maps_dep_m.items()
            }
            with torch.no_grad():
                pred = teacher(stacks)
                pred_np = {k: v.detach().cpu().numpy()[0] for k, v in pred.items()}
            maps_tea = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}
            dens_tea = np.asarray(
                maps_tea["disk"][iz * n_mom + dens_i], dtype=np.float32
            )
            if hint_mode in ("teacher", "teacher_recon"):
                dens_hint = dens_tea
            else:
                bw = float(np.clip(blend_weight, 0.0, 1.0))
                dens_hint = (1.0 - bw) * dens_tea + bw * dens_m
        elif hint_mode not in ("deposit", "dep", "data"):
            raise ValueError(
                f"unknown phase_b_hint={phase_b_hint!r} "
                "(expected deposit|teacher|blend)"
            )
        hint = midplane_contrast_from_dens(
            dens_hint, contrast_clip=3.0, m2_only=True, r_max=float(g_disk.r_max)
        )
        net, ckpt_meta = load_delta_checkpoint(phase_b_ckpt, map_location="cpu")
        contrast = predict_contrast(
            net,
            dens0,
            morph_contrast=hint if int(getattr(net, "n_in", 2)) >= 3 else None,
        )
        alpha_used = float(alpha)
        amode = str(alpha_mode).lower().strip()
        if amode in ("match_a2_rd", "a2_rd", "match_rd"):
            tgt = float(target_a2_rd) if target_a2_rd is not None else 0.49
            rd = float(a2_r_eval) if a2_r_eval is not None else float(
                r_weight_peak if r_weight_peak is not None else 2.0
            )
            alpha_used = alpha_match_predicted_contrast_a2_rd(
                maps_f0,
                contrast,
                cfg=cfg,
                target_a2=tgt,
                r_eval=rd,
                factor_floor=float(factor_floor),
                preserve_axisym=bool(preserve_axisym),
            )
            print(
                f"  phase_b alpha_mode={amode}: α={alpha_used:.3f} "
                f"targeting A₂(R={rd:g})≈{tgt:.3f} "
                f"arch={ckpt_meta.get('arch', '?')} n_in={getattr(net, 'n_in', '?')}",
                flush=True,
            )
        fields = inject_predicted_contrast_on_f0_dens(
            maps_f0,
            contrast,
            cfg=cfg,
            components=("disk",),
            alpha=float(alpha_used),
            midplane_only=bool(dens_resid_midplane_only),
            keep_f0_residual=float(keep_f0_residual),
            preserve_axisym=bool(preserve_axisym),
            factor_floor=float(factor_floor),
        )
        alpha = float(alpha_used)
        maps_morph = None
    elif ms == "deposit":
        maps_morph = maps_dep_m
    elif ms in ("teacher_recon", "recon", "teacher"):
        device = next(teacher.parameters()).device
        stacks = {
            k: torch.as_tensor(
                normalize_stack(v, stats[k])[None], dtype=torch.float32, device=device
            )
            for k, v in maps_dep_m.items()
        }
        with torch.no_grad():
            pred = teacher(stacks)
            pred_np = {k: v.detach().cpu().numpy()[0] for k, v in pred.items()}
        maps_morph = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}
    elif ms in ("blend", "teacher_deposit_blend"):
        w = float(np.clip(blend_weight, 0.0, 1.0))
        device = next(teacher.parameters()).device
        stacks = {
            k: torch.as_tensor(
                normalize_stack(v, stats[k])[None], dtype=torch.float32, device=device
            )
            for k, v in maps_dep_m.items()
        }
        with torch.no_grad():
            pred = teacher(stacks)
            pred_np = {k: v.detach().cpu().numpy()[0] for k, v in pred.items()}
        maps_tea = {k: denormalize_stack(pred_np[k], stats[k]) for k in pred_np}
        maps_morph = {
            k: (1.0 - w) * maps_tea[k] + w * maps_dep_m[k] for k in maps_tea
        }
    else:
        raise ValueError(
            f"unknown morph_source={morph_source!r} "
            "(expected teacher_recon|deposit|phase_b|blend)"
        )

    if maps_morph is not None:
        alpha_used = float(alpha)
        amode = str(alpha_mode).lower().strip()
        if amode in ("match_a2_rd", "a2_rd", "match_rd"):
            tgt = float(target_a2_rd) if target_a2_rd is not None else 0.49
            rd = float(a2_r_eval) if a2_r_eval is not None else float(
                r_weight_peak if r_weight_peak is not None else 2.0
            )
            alpha_used = alpha_match_residual_a2_rd(
                maps_f0,
                maps_morph,
                cfg=cfg,
                target_a2=tgt,
                r_eval=rd,
                mode=str(dens_resid_kind),
                other_morph_alpha=float(dens_resid_other_alpha),
                residual_scale=str(residual_scale),
                factor_floor=float(factor_floor),
                preserve_axisym=bool(preserve_axisym),
                r_weight_peak=r_weight_peak,
                r_weight_sigma=r_weight_sigma,
                contrast_sharpen=float(contrast_sharpen),
                contrast_smooth_kpc=float(contrast_smooth_kpc),
                r_weight_floor=float(r_weight_floor),
            )
            print(
                f"  alpha_mode={amode}: α={alpha_used:.3f} "
                f"targeting A₂(R={rd:g})≈{tgt:.3f}",
                flush=True,
            )
        fields = inject_morph_residual_on_f0_dens(
            maps_f0,
            maps_morph,
            cfg=cfg,
            components=("disk",),
            alpha=float(alpha_used),
            midplane_only=bool(dens_resid_midplane_only),
            mode=str(dens_resid_kind),
            other_morph_alpha=float(dens_resid_other_alpha),
            keep_f0_residual=float(keep_f0_residual),
            scale=str(residual_scale),
            preserve_axisym=bool(preserve_axisym),
            factor_floor=float(factor_floor),
            contrast_from_midplane=bool(contrast_from_midplane),
            r_weight_peak=r_weight_peak,
            r_weight_sigma=r_weight_sigma,
            contrast_sharpen=float(contrast_sharpen),
            contrast_smooth_kpc=float(contrast_smooth_kpc),
            r_weight_floor=float(r_weight_floor),
        )
        alpha = float(alpha_used)

    mass_totals = {
        "disk": float(mass0[cid0 == 0].sum()) if np.any(cid0 == 0) else 0.0,
        "halo": float(mass0[cid0 == 1].sum()) if np.any(cid0 == 1) else 0.0,
        "bulge": float(mass0[cid0 == 2].sum()) if np.any(cid0 == 2) else 0.0,
    }
    if mass_calibrate_dens:
        fields = rescale_dens_channels_to_mass(
            fields, cfg=cfg, mass_total_per_component=mass_totals
        )

    # Disk/halo from residual maps; bulge from spherical shells of GalactICS f0
    # (cusp) when present; then stitch retain halo(+bulge) for exact f0 kinematics.
    mask_b = cid0 == 2
    has_bulge = bool(np.any(mask_b))
    if has_bulge:
        count_fr = dict(COUNT)
        retain = tuple(retain_components)
        com_b = np.average(pos0[mask_b], axis=0, weights=mass0[mask_b])
        shells = bin_spherical_shell_moments(
            pos0[mask_b] - com_b,
            vel0[mask_b],
            mass0[mask_b],
            n_shells=64,
            r_min=0.05,
            r_max=6.0,
            log_bins=True,
        )
        parts = fuse_shell_bulge_with_multiscale(
            fields,
            cfg=cfg,
            bulge_shells=shells,
            n_particles=n_resample,
            count_fractions=count_fr,
            mass_total_per_component=mass_totals,
            rng=rng,
            velocity_frame=velocity_frame,
            match_cell_moments=match_cell_moments,
        )
        bmask = parts["component_id"] == 2
        parts["pos"][bmask] = parts["pos"][bmask] + com_b
    else:
        # Disk+halo only (GalaxyModel.milky_way_disk_halo / bulge=0 campaigns).
        count_fr = {"disk": 2 / 3, "halo": 1 / 3, "bulge": 0.0}
        retain = tuple(c for c in retain_components if c != "bulge")
        maps_dh = {k: v for k, v in fields.items() if k != "bulge"}
        grids_dh = tuple(g for g in cfg.grids if g.name != "bulge")
        from galacticsics.ml.fields.binning import MultiScaleSliceConfig

        cfg_dh = MultiScaleSliceConfig(
            grids=grids_dh, include_potential=cfg.include_potential
        )
        parts = resample_particles_from_multiscale(
            maps_dh,
            cfg=cfg_dh,
            n_particles=n_resample,
            count_fractions={k: count_fr[k] for k in cfg_dh.components},
            mass_total_per_component=mass_totals,
            rng=rng,
            velocity_frame=velocity_frame,
            match_cell_moments=match_cell_moments,
        )
        parts["bulge_method"] = "none_no_bulge_ic"

    if retain:
        n_per = dict(parts.get("n_per_component") or {})
        parts = stitch_retained_components(
            parts,
            source_pos=pos0,
            source_vel=vel0,
            source_mass=mass0,
            source_cid=cid0,
            retain=retain,
            n_retain={c: int(n_per.get(c, 0)) for c in retain},
            rng=rng,
        )

    cdf_meta: dict = {"match_f0_radial_cdf": bool(match_f0_radial_cdf)}
    if match_f0_radial_cdf:
        # Lock particle Σ(R) to GalactICS f0 after dens-resample (φ preserved).
        # Must run before velocity transplant so kNN sees final positions.
        pos_locked, cdf_meta = match_disk_radial_cdf_to_reference(
            parts["pos"],
            parts["mass"],
            parts["component_id"],
            pos0,
            mass0,
            cid0,
            r_max=15.0,
            n_bins=64,
            match_z_scale=True,
        )
        parts["pos"] = pos_locked

    vmode = str(velocity_mode).lower().strip()
    # Aliases: --vel-source morph|f0|blend|hybrid
    if vmode in ("f0", "galactics"):
        vmode = "transplant"
    elif vmode in ("morph", "data", "dump"):
        vmode = "morph_transplant"
    elif vmode in ("blend", "mix"):
        vmode = "morph_blend"
    elif vmode in ("hybrid", "bar_hybrid", "morph_hybrid"):
        vmode = "morph_hybrid"
    vel_meta = {"velocity_mode": vmode}
    if vmode in (
        "transplant",
        "morph_transplant",
        "morph_blend",
        "morph_hybrid",
    ):
        # Morph components that were dens-resampled (not retained).
        morph_comps = tuple(
            c for c in ("disk", "halo", "bulge") if c not in set(retain)
        )
        if vmode == "transplant":
            pos_src, vel_src, cid_src = pos0, vel0, cid0
            src_label = "f0"
        else:
            pos_src, vel_src, cid_src = pos_morph, vel_morph, cid_morph
            src_label = "morph"
        vel_new, tmeta = transplant_velocities_knn(
            parts["pos"],
            parts["component_id"],
            pos_src,
            vel_src,
            cid_src,
            components=morph_comps,
            rotate_with_phi=True,
            rng=rng,
        )
        tmeta = dict(tmeta)
        tmeta["vel_source"] = src_label
        if vmode in ("morph_blend", "morph_hybrid"):
            # f0-transplant baseline for blend / radial hybrid.
            vel_f0, tmeta_f0 = transplant_velocities_knn(
                parts["pos"],
                parts["component_id"],
                pos0,
                vel0,
                cid0,
                components=morph_comps,
                rotate_with_phi=True,
                rng=rng,
            )
            tmeta["f0_transplant"] = tmeta_f0
            if vmode == "morph_blend":
                # Linear mix (weight=1 → pure morph dump kinematics).
                w = float(morph_vel_blend_weight)
                vel_new = (1.0 - w) * vel_f0 + w * vel_new
                tmeta["morph_blend_weight"] = w
            else:
                # Hybrid: morph velocities inside R<=r_max, f0 outside
                # (quiet outer disk; bar-supporting streaming in annulus).
                from galacticsics.ml.morton.tokenize import COMPONENT_IDS

                r_cut = float(vel_hybrid_r_max)
                for name in morph_comps:
                    cid = int(COMPONENT_IDS[name])
                    m = parts["component_id"] == cid
                    if not np.any(m):
                        continue
                    R = np.hypot(parts["pos"][m, 0], parts["pos"][m, 1])
                    use_morph = R <= r_cut
                    mixed = vel_f0[m].copy()
                    mixed[use_morph] = vel_new[m][use_morph]
                    vel_new[m] = mixed
                    tmeta.setdefault("hybrid", {})[name] = {
                        "r_max_kpc": r_cut,
                        "n_morph": int(np.count_nonzero(use_morph)),
                        "n_f0": int(np.count_nonzero(~use_morph)),
                    }
                tmeta["vel_source"] = f"morph_R<={r_cut:g}+f0_outer"
        # Only overwrite transplanted comps; retained already have f0 vel.
        for name in morph_comps:
            from galacticsics.ml.morton.tokenize import COMPONENT_IDS

            cid = int(COMPONENT_IDS[name])
            m = parts["component_id"] == cid
            parts["vel"][m] = vel_new[m]
        vel_meta["transplant"] = tmeta
    elif vmode not in ("moments", "f0_moments", "none"):
        raise ValueError(
            f"unknown velocity_mode={velocity_mode!r} "
            "(expected transplant|morph_transplant|morph_blend|morph_hybrid|"
            "moments|f0|morph|blend|hybrid)"
        )

    parts["dens_resid_alpha_used"] = float(alpha)
    parts["dens_resid_kind"] = str(dens_resid_kind)
    parts["residual_scale"] = str(residual_scale)
    parts["preserve_axisym"] = bool(preserve_axisym)
    parts["factor_floor"] = float(factor_floor)
    parts["contrast_from_midplane"] = bool(contrast_from_midplane)
    parts["match_f0_radial_cdf"] = bool(match_f0_radial_cdf)
    parts["radial_cdf_meta"] = cdf_meta
    parts["morph_source"] = ms
    parts["r_weight_peak"] = r_weight_peak
    parts["r_weight_sigma"] = r_weight_sigma
    parts["contrast_sharpen"] = float(contrast_sharpen)
    parts["r_weight_floor"] = float(r_weight_floor)
    parts["alpha_mode"] = str(alpha_mode)
    parts["target_a2_rd"] = target_a2_rd
    parts["recipe"] = "residual_f0"
    parts["ic_path"] = str(ic_path)
    parts["morph_path"] = str(morph_path)
    parts["retain_components"] = list(retain)
    parts["velocity_meta"] = vel_meta
    parts["phase"] = (
        "B_learned_delta_on_f0"
        if ms in ("phase_b", "learned_delta", "delta")
        else "A_hand_library_residual_on_f0"
    )
    if phase_b_ckpt is not None:
        parts["phase_b_ckpt"] = str(phase_b_ckpt)
    if ms in ("phase_b", "learned_delta", "delta"):
        parts["phase_b_hint"] = str(phase_b_hint)
    if ms in ("blend", "teacher_deposit_blend"):
        parts["blend_weight"] = float(np.clip(blend_weight, 0.0, 1.0))
    parts["has_bulge"] = bool(has_bulge)
    parts["count_fractions"] = dict(count_fr)
    return parts


def _full_dyn_residual_particles(
    ic_path: Path,
    morph_path: Path,
    *,
    n_resample: int,
    rng: np.random.Generator,
    mode: str = "replace",
    retain_components: tuple[str, ...] = ("halo", "bulge"),
    eps: float = 0.05,
) -> dict:
    """Full dynamical residual: match data dens+kin, not m2 paint on f0 axisym.

    Modes
    -----
    replace
        Disk phase-space taken from the barred dump (stratified); halo/bulge
        retained from GalactICS ``f0``. Closest to "reconstruct all dynamics"
        of the data disk while keeping an equilibrium scaffolding.
    ot_lite
        OT-lite transport of ``f0`` toward the dump (radial CDF + ⟨vφ⟩/σ), then
        re-stitch halo/bulge from ``f0`` so only the disk is transported.
    """
    from galacticsics.ml.fields.resample import (
        stitch_retained_components,
        transport_ot_lite,
    )
    from galacticsics.ml.morton.tokenize import COMPONENT_IDS

    def _load_frame(path: Path):
        with np.load(path, allow_pickle=True) as data:
            pos = np.asarray(data["pos"], dtype=np.float64)
            vel = np.asarray(data["vel"], dtype=np.float64)
            mass = np.asarray(data["mass"], dtype=np.float64)
            cid = _component_ids(data.get("tags"), data.get("type_id"), pos.shape[0])
            ep = (
                np.asarray(data["eps"], dtype=np.float64)
                if "eps" in data.files
                else np.full(pos.shape[0], float(eps), dtype=np.float64)
            )
        pos, vel, _ = prepare_shared_frame(pos, vel, mass, center=True, rotate=False)
        return {
            "pos": pos,
            "vel": vel,
            "mass": mass,
            "component_id": cid,
            "eps": ep,
        }

    f0 = _load_frame(ic_path)
    morph = _load_frame(morph_path)
    mode_l = str(mode).lower().strip()
    retain = tuple(retain_components)
    has_bulge_f0 = bool(np.any(f0["component_id"] == 2))
    if not has_bulge_f0:
        retain = tuple(c for c in retain if c != "bulge")

    # Target component counts from n_resample (disk-heavy GalactICS fractions).
    if has_bulge_f0:
        fr = {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
    else:
        fr = {"disk": 2 / 3, "halo": 1 / 3, "bulge": 0.0}
    n_per = {c: int(round(n_resample * f)) for c, f in fr.items() if f > 0}
    # Fix rounding so total == n_resample.
    deficit = int(n_resample) - int(sum(n_per.values()))
    if deficit != 0 and "disk" in n_per:
        n_per["disk"] = max(1, n_per["disk"] + deficit)

    if mode_l in ("replace", "data_disk", "disk_replace"):
        # Disk from morph dump; halo/bulge from f0.
        takes = []
        masses = []
        for name, n_want in n_per.items():
            cid = int(COMPONENT_IDS[name])
            if name == "disk":
                src = morph
            else:
                src = f0
            idx = np.flatnonzero(src["component_id"] == cid)
            if idx.size == 0:
                continue
            if idx.size >= n_want:
                sel = rng.choice(idx, size=n_want, replace=False)
            else:
                sel = rng.choice(idx, size=n_want, replace=True)
            takes.append(
                {
                    "pos": src["pos"][sel],
                    "vel": src["vel"][sel],
                    "mass": src["mass"][sel],
                    "component_id": np.full(n_want, cid, dtype=src["component_id"].dtype),
                    "eps": (
                        src["eps"][sel]
                        if src["eps"].shape[0] == src["pos"].shape[0]
                        else np.full(n_want, float(eps))
                    ),
                }
            )
            masses.append(float(src["mass"][idx].sum()))
        parts = {
            "pos": np.concatenate([t["pos"] for t in takes], axis=0),
            "vel": np.concatenate([t["vel"] for t in takes], axis=0),
            "mass": np.concatenate([t["mass"] for t in takes], axis=0),
            "component_id": np.concatenate([t["component_id"] for t in takes], axis=0),
            "eps": np.concatenate([t["eps"] for t in takes], axis=0),
        }
        # Preserve per-component mass totals from their sources.
        for name in n_per:
            cid = int(COMPONENT_IDS[name])
            m = parts["component_id"] == cid
            if not np.any(m):
                continue
            src = morph if name == "disk" else f0
            src_m = src["component_id"] == cid
            if not np.any(src_m):
                continue
            m_tot = float(src["mass"][src_m].sum())
            cur = float(parts["mass"][m].sum())
            if cur > 0:
                parts["mass"][m] *= m_tot / cur
        vel_meta = {"velocity_mode": "data_disk_native", "mode": "replace"}
        ot_meta: dict = {}
    elif mode_l in ("ot_lite", "ot", "transport"):
        transported, ot_meta = transport_ot_lite(f0, morph)
        # Start from transported, then force halo/bulge back to f0 samples.
        parts = {
            "pos": np.asarray(transported["pos"], dtype=np.float64),
            "vel": np.asarray(transported["vel"], dtype=np.float64),
            "mass": np.asarray(transported["mass"], dtype=np.float64),
            "component_id": np.asarray(transported["component_id"]),
        }
        parts["eps"] = np.full(parts["pos"].shape[0], float(eps), dtype=np.float64)
        # Downsample to n_resample first (stratified), then stitch retain.
        if parts["pos"].shape[0] != n_resample:
            parts = _stratified_down(parts, n_resample, rng)
        if retain:
            n_ret = {
                c: int(np.count_nonzero(parts["component_id"] == int(COMPONENT_IDS[c])))
                for c in retain
            }
            parts = stitch_retained_components(
                parts,
                source_pos=f0["pos"],
                source_vel=f0["vel"],
                source_mass=f0["mass"],
                source_cid=f0["component_id"],
                retain=retain,
                n_retain=n_ret,
                rng=rng,
            )
        vel_meta = {"velocity_mode": "ot_lite_disk", "mode": "ot_lite"}
    else:
        raise ValueError(
            f"unknown full_dyn mode={mode!r} (expected replace|ot_lite)"
        )

    parts["recipe"] = f"full_dyn_{mode_l}"
    parts["phase"] = "full_dyn_residual"
    parts["ic_path"] = str(ic_path)
    parts["morph_path"] = str(morph_path)
    parts["retain_components"] = list(retain)
    parts["velocity_meta"] = vel_meta
    parts["ot_meta"] = ot_meta
    parts["has_bulge"] = bool(np.any(parts["component_id"] == 2))
    parts["n_per_component"] = {
        c: int(np.count_nonzero(parts["component_id"] == int(COMPONENT_IDS[c])))
        for c in ("disk", "halo", "bulge")
    }
    parts["dens_resid_alpha_used"] = 1.0
    parts["morph_source"] = "data_dump"
    parts["alpha_mode"] = "n/a_full_dyn"
    return parts


def _sample_gen(lib, method: str, rng: np.random.Generator, n_resample: int) -> tuple[dict, dict]:
    if method == "z_amplify":
        feat, meta = lib.sample_features_z_amplify(kind="barred", rng=rng)
        fields = lib.decode_features(feat)
    else:
        fields, meta = lib.sample_fields(
            kind="barred", method=method, rng=rng, alpha_lo=1.15, alpha_hi=1.35
        )
    phys = {
        k: denormalize_stack(v[0].detach().cpu().numpy(), lib.stats[k])
        for k, v in fields.items()
    }
    gen = resample_particles_from_multiscale(
        phys, cfg=lib.cfg, n_particles=n_resample, count_fractions=COUNT, rng=rng
    )
    return gen, meta


def _plot_a2_t(path: Path, report: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for tag, row in report.get("arms", {}).items():
        if not row.get("ok"):
            continue
        ax.plot(row["t_gyr"], row["a2_t"], lw=1.8, label=tag)
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"disk median $A_2$")
    ax.set_title(
        rf"Evolve compare ($\mathrm{{d}}t={report['dt']}$, "
        rf"$t_{{\rm end}}={report['evolve_gyr']}\,\mathrm{{Gyr}}$)"
    )
    ax.axhline(0.30, color="0.5", ls=":", lw=1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_am_snap(path: Path, report: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for ax, which in zip(axes, ("am_pre", "am_post")):
        for tag, row in report.get("arms", {}).items():
            am = row.get(which)
            if not am:
                continue
            r = np.asarray(am["r_mid"], dtype=float)
            a2 = np.asarray(am["modes"]["2"]["a_m_over_a0"], dtype=float)
            ax.plot(r, a2, lw=1.6, label=tag)
        ax.set_xlabel(r"$R$ [kpc]")
        ax.set_ylabel(r"$A_2(R)$")
        ax.set_title("pre" if which == "am_pre" else "post")
        ax.set_ylim(0, None)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(r"$A_2(R)$ evolve compare")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_faceon(path: Path, panels: dict[str, tuple[np.ndarray, np.ndarray]], labels) -> None:
    keys = list(panels.keys())
    n = len(keys)
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 5.4))
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for j, tag in enumerate(keys):
        pre, post = panels[tag]
        for i, img in enumerate((pre, post)):
            ax = axes[i, j]
            pos = img[img > 0]
            vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
            # dens[i,j]=(x_i,y_j) from histogram2d; transpose for imshow (y,x).
            ax.imshow(
                np.log1p(np.maximum(img, 0)).T,
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=np.log1p(vmax),
            )
            lab = labels.get(tag, tag)
            ax.set_title(f"{lab} {'pre' if i == 0 else 'post'}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Face-on dens: pre / post evolve")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_faceon_evolution(
    path: Path,
    series: dict[str, dict],
    labels: dict[str, str],
    *,
    title: str | None = None,
) -> None:
    """Rows = methods, cols = time snapshots (detailed morphology panel)."""
    keys = [k for k in series if series[k].get("maps")]
    if not keys:
        return
    # Align on the longest common time grid (prefer first arm's times).
    t_ref = list(series[keys[0]]["t_gyr"])
    n_t = len(t_ref)
    n_m = len(keys)
    fig, axes = plt.subplots(n_m, n_t, figsize=(2.55 * n_t, 2.45 * n_m), squeeze=False)
    # Shared color scale per method (pre→post) so morphology changes are visible.
    for i, tag in enumerate(keys):
        maps = series[tag]["maps"]
        times = series[tag]["t_gyr"]
        stack = np.concatenate([m[m > 0] for m in maps if np.any(m > 0)], axis=0)
        vmax = float(np.percentile(stack, 99.5)) if stack.size else 1.0
        vmax_log = np.log1p(vmax)
        for j in range(n_t):
            ax = axes[i, j]
            # nearest time index if grids differ slightly
            if j < len(maps):
                img = maps[j]
                tlab = times[j]
            else:
                img = maps[-1]
                tlab = times[-1]
            # dens[i,j]=(x_i,y_j) from histogram2d; transpose for imshow (y,x).
            ax.imshow(
                np.log1p(np.maximum(img, 0)).T,
                origin="lower",
                cmap="inferno",
                vmin=0,
                vmax=vmax_log,
            )
            if i == 0:
                ax.set_title(rf"$t={tlab:.2f}\,\mathrm{{Gyr}}$", fontsize=9)
            if j == 0:
                ax.set_ylabel(labels.get(tag, tag), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        title or r"Disk face-on $\Sigma$ evolution (log dens)",
        y=1.01,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_com_t(path: Path, report: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for tag, row in report.get("arms", {}).items():
        if not row.get("ok"):
            continue
        ax.plot(row["t_gyr"], row["com_norm_t"], lw=1.6, label=tag)
    ax.set_xlabel(r"$t$ [Gyr]")
    ax.set_ylabel(r"$|\mathrm{COM}|$ [kpc]")
    ax.set_title("COM norm during evolve")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/ml/field_maps/evolve_compare_long_2026-07-25"),
    )
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--rank", type=Path, default=RANK)
    p.add_argument("--n-bar", type=int, default=12)
    p.add_argument("--n-quiet", type=int, default=8)
    p.add_argument("--n-mid", type=int, default=4)
    p.add_argument("--n-rot-bar", type=int, default=2)
    p.add_argument("--bar-floor", type=float, default=0.25)
    p.add_argument("--quiet-ceil", type=float, default=0.05)
    p.add_argument("--enc-grid", type=int, default=4)
    p.add_argument("--n-pc", type=int, default=64)
    p.add_argument("--n-evolve", type=int, default=25_000)
    p.add_argument("--n-resample", type=int, default=50_000)
    p.add_argument(
        "--n-disk",
        type=int,
        default=None,
        help="If set, disk alone = N_disk and total = round(7/4·N_disk) "
        "for both --n-evolve and --n-resample (mix 4:2:1). "
        "Preferred for corpus-scale runs (e.g. --n-disk 1000000).",
    )
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--evolve-gyr", type=float, default=0.50)
    p.add_argument("--omp", type=int, default=2)
    p.add_argument(
        "--force",
        type=str,
        default=None,
        choices=("gpu_bh", "bh_c", "bh"),
        help="N-body force backend (default: gpu_bh if available else bh_c).",
    )
    p.add_argument("--timeout-s", type=float, default=3600.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument(
        "--methods",
        type=str,
        default="amplify_knn_hybrid,z_amplify",
        help="Comma list of generative methods (+ always data + recon)",
    )
    p.add_argument("--ic-a2-min", type=float, default=0.30)
    p.add_argument("--ic-a2-tries", type=int, default=8)
    p.add_argument("--n-track", type=int, default=11)
    p.add_argument(
        "--faceon-times",
        type=str,
        default="0,0.25,0.5",
        help="Comma list of Gyr times for detailed face-on panels "
        "(0 and evolve-gyr always included).",
    )
    p.add_argument("--faceon-bins", type=int, default=128)
    p.add_argument("--faceon-half", type=float, default=12.0)
    p.add_argument(
        "--paper-figures",
        type=Path,
        default=None,
        help="Optional dir to also copy key panels (e.g. papers/mnras_noneq_ics/figures)",
    )
    p.add_argument(
        "--paper-prefix",
        type=str,
        default=None,
        help="If set, also write {prefix}_a2_t.png / _faceon.png / … "
        "(e.g. fig_evolve_1e6). Legacy fig5_* copies still written when "
        "--paper-figures is set.",
    )
    p.add_argument(
        "--skip-recon",
        action="store_true",
        help="Skip FFT recon arm (saves wall time at high N).",
    )
    p.add_argument(
        "--skip-z-amplify",
        action="store_true",
        help="Drop z_amplify even if listed in --methods.",
    )
    args = p.parse_args()
    if args.teacher is None:
        args.teacher = default_teacher()
    if args.n_disk is not None:
        n_tot = int(round(args.n_disk * 7 / 4))
        args.n_evolve = n_tot
        args.n_resample = n_tot
        print(
            f"n-disk={args.n_disk:,} → total N={n_tot:,} "
            f"(halo≈{args.n_disk // 2:,}, bulge≈{args.n_disk // 4:,})",
            flush=True,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["OMP_NUM_THREADS"] = str(args.omp)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ranked = json.loads(Path(args.rank).read_text())
    ranked.sort(key=lambda r: r["a2"], reverse=True)
    data_path = args.data_path or Path(ranked[0]["path"])
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.skip_z_amplify:
        methods = [m for m in methods if m != "z_amplify"]
    faceon_times = [
        float(x) for x in args.faceon_times.split(",") if x.strip()
    ]
    # Ensure mid-point when evolve-gyr differs from default faceon list.
    if args.evolve_gyr not in faceon_times:
        faceon_times.append(float(args.evolve_gyr))
    mid = 0.5 * float(args.evolve_gyr)
    if mid not in faceon_times and abs(mid) > 1e-9:
        faceon_times.append(mid)
    faceon_times = sorted(set(faceon_times))

    print(f"=== teacher {args.teacher} ===", flush=True)
    teacher, cfg, stats = load_frozen_teacher_bundle(args.teacher)

    print("=== build stratified library ===", flush=True)
    lib = build_library(args)

    n_ev = min(args.n_evolve, args.n_resample)
    arms: dict[str, dict] = {}
    face_panels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    face_series: dict[str, dict] = {}
    labels = {
        "data": "data",
        "recon": "FFT recon",
        "amplify_knn_hybrid": "hybrid",
        "z_amplify": "z_amplify",
    }

    # --- data ---
    print(f"=== arm data ← {data_path} (N={n_ev}) ===", flush=True)
    data_ev = _full_com(_subsample_dump(data_path, n_ev, rng))
    arms["data"] = {"parts": data_ev, "meta": {"path": str(data_path)}}

    # --- recon ---
    if not args.skip_recon:
        print("=== arm recon (teacher encode→decode→resample) ===", flush=True)
        recon = _recon_particles(
            data_path, teacher, cfg, stats, n_resample=args.n_resample, rng=rng
        )
        recon_ev = _vcom_only(_stratified_down(recon, n_ev, rng))
        arms["recon"] = {"parts": recon_ev, "meta": {"teacher": str(args.teacher)}}

    # --- generative ---
    for method in methods:
        print(f"=== arm {method} ===", flush=True)
        best = None
        for attempt in range(max(1, args.ic_a2_tries)):
            gen, meta = _sample_gen(lib, method, rng, args.n_resample)
            gen_ev = _vcom_only(_stratified_down(gen, n_ev, rng))
            a2 = float(_a2_disk(gen_ev))
            print(f"  try#{attempt} IC A2={a2:.3f}", flush=True)
            if best is None or a2 > best[0]:
                best = (a2, gen_ev, meta)
            if a2 >= args.ic_a2_min:
                break
        assert best is not None
        arms[method] = {"parts": best[1], "meta": {k: v for k, v in best[2].items() if k != "z"}}

    report = {
        "dt": args.dt,
        "evolve_gyr": args.evolve_gyr,
        "n_steps": int(np.ceil(args.evolve_gyr / args.dt)),
        "n_disk": int(round(n_ev * 4 / 7)),
        "n_halo_approx": int(round(n_ev * 2 / 7)),
        "n_bulge_approx": int(round(n_ev * 1 / 7)),
        "n_evolve": n_ev,
        "n_resample": args.n_resample,
        "omp": args.omp,
        "force": args.force or "auto",
        "teacher": str(args.teacher),
        "data_path": str(data_path),
        "faceon_times_gyr": faceon_times,
        "faceon_bins": args.faceon_bins,
        "count_mix": "disk:halo:bulge ≈ 4:2:1",
        "count_convention": (
            f"disk alone ≈ {int(round(n_ev * 4 / 7)):,}; total = {n_ev:,}"
        ),
        "shared_centering": (
            "data: full soft COM; dens-resampled recon/gen: morphological origin + VCOM-only"
        ),
        "note": (
            "Long evolve compare vs original data dump. "
            "Prefer FFT morph teacher for dens sharpness / sampling library. "
            "Corpus-scale claim uses disk alone = 1e6 (total ≈ 1.75e6)."
        ),
        "arms": {},
    }

    for tag, pack in arms.items():
        parts = pack["parts"]
        pre = _metrics(parts)
        am_pre = _am_profiles(
            parts["pos"][parts["component_id"] == 0],
            parts["mass"][parts["component_id"] == 0],
        )
        print(
            f"=== evolve {tag} A2={pre['a2']:.3f} N={parts['pos'].shape[0]} ===",
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
            force=args.force,
        )
        if not evo.get("ok"):
            report["arms"][tag] = {
                "ok": False,
                "pre": pre,
                "meta": pack["meta"],
                "evolve": evo,
            }
            print(f"  FAIL {evo.get('error')}", flush=True)
            continue
        pos_f = evo.pop("pos_final")
        face_maps = evo.pop("faceon_maps", [])
        face_t = evo.pop("faceon_t_gyr", [])
        post_parts = {**parts, "pos": pos_f}
        post = _metrics(post_parts)
        am_post = _am_profiles(
            post_parts["pos"][post_parts["component_id"] == 0],
            post_parts["mass"][post_parts["component_id"] == 0],
        )
        if face_maps:
            face_panels[tag] = (face_maps[0], face_maps[-1])
            face_series[tag] = {"t_gyr": face_t, "maps": face_maps}

        def _ser(prof):
            return {
                "r_mid": [float(x) for x in prof["r_mid"]],
                "modes": {
                    str(m): {
                        "a_m_over_a0": [
                            None if not np.isfinite(v) else float(v)
                            for v in prof["modes"][m]["a_m_over_a0"]
                        ],
                        "median": prof["modes"][m]["median"],
                    }
                    for m in prof["modes"]
                },
            }

        row = {
            "ok": True,
            "pre": pre,
            "post": post,
            "a2_pre": pre["a2"],
            "a2_post": post["a2"],
            "da2": post["a2"] - pre["a2"],
            "com_drift_kpc": evo["com_drift_kpc"],
            "n_steps": evo["n_steps"],
            "force_method": evo["force_method"],
            "wall_s": evo["wall_s"],
            "t_gyr": evo["t_gyr"],
            "a2_t": evo["a2_t"],
            "com_norm_t": evo["com_norm_t"],
            "faceon_t_gyr": face_t,
            "am_pre": _ser(am_pre),
            "am_post": _ser(am_post),
            "meta": pack["meta"],
        }
        report["arms"][tag] = row
        print(
            f"  A2 {pre['a2']:.3f}→{post['a2']:.3f}  COM={evo['com_drift_kpc']:.4f}  "
            f"wall={evo['wall_s']:.1f}s  steps={evo['n_steps']}",
            flush=True,
        )

    # figures
    a2_png = args.out / "a2_t.png"
    am_png = args.out / "am_r_pre_post.png"
    face_png = args.out / "faceon_pre_post.png"
    face_evo_png = args.out / "faceon_evolution.png"
    com_png = args.out / "com_t.png"
    _plot_a2_t(a2_png, report)
    _plot_am_snap(am_png, report)
    _plot_com_t(com_png, report)
    if face_panels:
        _plot_faceon(face_png, face_panels, labels)
    if face_series:
        _plot_faceon_evolution(
            face_evo_png,
            face_series,
            labels,
            title=(
                rf"Disk face-on $\Sigma$ (n_{{\rm disk}}\approx{int(round(n_ev * 4 / 7)):,}, "
                rf"$N_{{\rm tot}}={n_ev:,}$, "
                rf"$t_{{\rm end}}={args.evolve_gyr}\,\mathrm{{Gyr}}$)"
            ),
        )
    # Persist face-on maps for replot (npz; float32 to keep size modest).
    if face_series:
        np.savez_compressed(
            args.out / "faceon_maps.npz",
            **{
                f"{tag}_t": np.asarray(face_series[tag]["t_gyr"], dtype=np.float64)
                for tag in face_series
            },
            **{
                f"{tag}_map{i}": np.asarray(m, dtype=np.float32)
                for tag in face_series
                for i, m in enumerate(face_series[tag]["maps"])
            },
            tags=np.asarray(list(face_series.keys())),
        )
    report["figures"] = {
        "a2_t": str(a2_png),
        "am_r": str(am_png),
        "faceon": str(face_png),
        "faceon_evolution": str(face_evo_png),
        "com_t": str(com_png),
    }

    out_json = args.out / "verdict.json"
    out_json.write_text(json.dumps(report, indent=2))
    lines = [
        "# Long evolve compare",
        "",
        f"Teacher: `{args.teacher}`",
        f"Settings: dt={args.dt}, end_gyr={args.evolve_gyr} → **{report['n_steps']}** steps; "
        f"**disk ≈ {report['n_disk']:,}**, total N={n_ev:,} (mix 4:2:1); "
        f"OpenMP={args.omp}; face-on bins={args.faceon_bins}.",
        f"Face-on times [Gyr]: {faceon_times}",
        "",
        "| Arm | A₂ pre→post | ΔA₂ | COM drift | wall [s] |",
        "|-----|-------------|-----|-----------|----------|",
    ]
    for tag, row in report["arms"].items():
        if not row.get("ok"):
            lines.append(f"| `{tag}` | FAIL | — | — | — |")
            continue
        lines.append(
            f"| `{tag}` | {row['a2_pre']:.3f}→{row['a2_post']:.3f} | "
            f"{row['da2']:+.3f} | {row['com_drift_kpc']:.4f} | {row['wall_s']:.1f} |"
        )
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print((args.out / "SUMMARY.md").read_text(), flush=True)

    if args.paper_figures is not None:
        import shutil

        args.paper_figures.mkdir(parents=True, exist_ok=True)
        prefix = args.paper_prefix
        if prefix is None and report["n_disk"] >= 800_000:
            prefix = "fig_evolve_disk1e6"
        elif prefix is None and n_ev >= 400_000:
            # Legacy: older runs used total N=1e6 (disk≈5.7e5) under this name.
            prefix = "fig_evolve_1e6"
        # Legacy fig5_* only when no explicit high-N prefix (avoid clobber).
        if not prefix:
            for src, name in (
                (a2_png, "fig5_evolve_long_a2_t.png"),
                (am_png, "fig5b_evolve_long_am_r.png"),
                (face_png, "fig5c_evolve_long_faceon.png"),
            ):
                if src.is_file():
                    shutil.copy2(src, args.paper_figures / name)
        if prefix:
            named = (
                (a2_png, f"{prefix}_a2_t.png"),
                (am_png, f"{prefix}_am_r.png"),
                (com_png, f"{prefix}_com_t.png"),
                (face_png, f"{prefix}_faceon_pre_post.png"),
                (face_evo_png, f"{prefix}_faceon.png"),
            )
            for src, name in named:
                if src.is_file():
                    shutil.copy2(src, args.paper_figures / name)
                    print(f"  paper ← {name}", flush=True)
        print(f"copied panels → {args.paper_figures}", flush=True)

    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
