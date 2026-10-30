"""Campaign force benchmarks and density-profile sanity checks."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from galacticsics.campaign.analysis import (
    component_mask,
    density_drift_metrics,
    disk_surface_profile,
    halo_spherical_profile,
    load_evolved_state,
    load_ic_state,
)
from ntropy.benchmark.force_breakdown import BhCBreakdown, time_bh_c_components
from ntropy.config import BhOptimizationsConfig, ForceConfig
from ntropy.forces.bhtree_c import extension_available
from ntropy.forces.context import ForceContext
from ntropy.integrators.timestep import TimestepConfig, active_mask_for_step, update_timestep_bins
from ntropy.particle_types import TypeRegistry
from ntropy.particles import ParticleState


# Extremes measured across the 49 accepted one-million-particle disks in the
# corrected MW Morton v9 campaign.  The default limits below leave 10% headroom
# beyond every accepted run, rather than treating code-unit velocities as if
# all galaxy models shared the same circular speed and disk temperature.
CALIBRATED_HEALTHY_IC_EXTREMA: dict[str, float] = {
    "disk_v_median_over_vc_min": 0.7787401418757438,
    "disk_v_p9999_over_vscale_max": 2.2023889084797696,
    "disk_tail_scaled_max": 1.832180818726503,
    "disk_rotation_scaled_min": 0.8824127324072207,
    "disk_prograde_scaled_min": 0.6743928838858929,
}

DEFAULT_MODEL_SCALED_IC_THRESHOLDS: dict[str, float] = {
    "r_max_over_rtidal_max": 1.25,
    "disk_v_median_over_vc_min": 0.9
    * CALIBRATED_HEALTHY_IC_EXTREMA["disk_v_median_over_vc_min"],
    "disk_v_p9999_over_vscale_max": 1.1
    * CALIBRATED_HEALTHY_IC_EXTREMA["disk_v_p9999_over_vscale_max"],
    "disk_tail_scaled_max": 1.1
    * CALIBRATED_HEALTHY_IC_EXTREMA["disk_tail_scaled_max"],
    "disk_rotation_scaled_min": 0.9
    * CALIBRATED_HEALTHY_IC_EXTREMA["disk_rotation_scaled_min"],
    "disk_prograde_scaled_min": 0.9
    * CALIBRATED_HEALTHY_IC_EXTREMA["disk_prograde_scaled_min"],
    # The existing halo DF validator uses the same 99.5% convention.  The
    # worst accepted v9 disk measured 99.976% on the deterministic sample.
    "disk_bound_fraction_min": 0.995,
    # A correctly area-weighted exponential disk has R_median ~= 1.68 R_d;
    # sampling R as a simple exponential instead gives ~= 0.69 R_d.
    "disk_r_median_over_rd_min": 1.0,
}


@dataclass(frozen=True)
class ICStabilityScales:
    """Physical velocity and length scales used by the model-aware IC gate."""

    disk_scale_length: float
    reference_radius: float
    circular_speed_ref: float
    sigma_r0: float
    tidal_radius: float


def _ic_stability_context(
    work_dir: Path,
    *,
    model: Any | None = None,
) -> tuple[ICStabilityScales, Any]:
    """Load model scales and the solved potential without duplicating I/O."""
    from galacticsics.io import (
        read_disk_correction,
        read_frequency_table,
        read_harmonic_potential,
    )
    from galacticsics.io.formats import read_rtidal

    work_dir = Path(work_dir)
    pot = read_harmonic_potential(work_dir / "dbh.dat")
    model = model or pot.model
    disk = getattr(model, "disk", None)
    if disk is None or not disk.enabled:
        raise ValueError("model-aware disk IC checks require an enabled disk")
    corr = read_disk_correction(work_dir / "cordbh.dat")
    freq = read_frequency_table(work_dir / "freqdbh.dat")
    disk_scale_length = float(disk.scale_length)
    reference_radius = 2.5 * disk_scale_length
    circular_speed_ref = float(freq.omega(reference_radius) * reference_radius)
    if not math.isfinite(circular_speed_ref) or circular_speed_ref <= 0.0:
        raise ValueError(f"invalid circular speed at 2.5 R_d: {circular_speed_ref}")
    tidal_path = work_dir / "rtidal.dat"
    tidal_radius = (
        float(read_rtidal(tidal_path))
        if tidal_path.is_file()
        else float(pot.nr * pot.dr)
    )
    return (
        ICStabilityScales(
            disk_scale_length=disk_scale_length,
            reference_radius=reference_radius,
            circular_speed_ref=circular_speed_ref,
            sigma_r0=float(corr.sigma_r0),
            tidal_radius=tidal_radius,
        ),
        pot,
    )


def load_ic_stability_scales(
    work_dir: Path,
    *,
    model: Any | None = None,
) -> ICStabilityScales:
    """Load physical scales for model-aware pre-evolution IC checks."""
    scales, _ = _ic_stability_context(work_dir, model=model)
    return scales


def _bound_fraction(
    state: ParticleState,
    mask: np.ndarray,
    pot: Any,
    *,
    max_samples: int,
) -> tuple[float, int]:
    """Deterministic sampled fraction with relative energy above ``Psi_c``."""
    from galacticsics.potential.evaluate import evaluate_potential

    pos = state.pos[mask]
    vel = state.vel[mask]
    n = len(pos)
    if n == 0:
        return float("nan"), 0
    idx = np.linspace(0, n - 1, min(n, max_samples), dtype=int)
    pos = pos[idx]
    vel = vel[idx]
    r_cyl = np.hypot(pos[:, 0], pos[:, 1])
    psi = np.array(
        [
            evaluate_potential(pot, float(r_cyl[i]), float(pos[i, 2]))
            for i in range(len(pos))
        ]
    )
    energy = psi - 0.5 * np.sum(vel * vel, axis=1)
    return float(np.mean(energy > float(pot.psic))), int(len(idx))


def diagnose_ic_stability(
    state: ParticleState,
    *,
    work_dir: Path | None = None,
    model: Any | None = None,
    max_binding_samples: int = 8192,
) -> dict[str, float]:
    """
    Flag IC issues that tend to destabilize tiered N-body runs.

    Returns per-component ``v_max`` and global ``r_max``.  Hot disk outliers
    (``v_max`` well above the median) often indicate bad sampling or a softening
    mismatch vs ``ntropy_config.json``.  Disk ``v_phi_mean`` near zero usually
    means cylindrical velocities were written as Cartesian (gendisk bug).
    """
    out: dict[str, float] = {"r_max": float(np.max(np.linalg.norm(state.pos, axis=1)))}
    stability_context = (
        _ic_stability_context(Path(work_dir), model=model)
        if work_dir is not None
        else None
    )
    if state.tags is None:
        speeds = np.linalg.norm(state.vel, axis=1)
        out["v_max"] = float(np.max(speeds))
        out["v_median"] = float(np.median(speeds))
        return out
    for label in np.unique(state.tags):
        mask = state.tags == label
        speeds = np.linalg.norm(state.vel[mask], axis=1)
        out[f"{label}_v_max"] = float(np.max(speeds))
        out[f"{label}_v_median"] = float(np.median(speeds))
        if label == "disk":
            x = state.pos[mask, 0]
            y = state.pos[mask, 1]
            vx = state.vel[mask, 0]
            vy = state.vel[mask, 1]
            r = np.hypot(x, y)
            ok = r > 1e-8
            # Skip when positions are unset (unit tests) or all at R=0.
            if int(np.count_nonzero(ok)) >= 10:
                v_phi = (-y[ok] * vx[ok] + x[ok] * vy[ok]) / r[ok]
                out["disk_v_phi_mean"] = float(np.mean(v_phi))
                out["disk_v_phi_median"] = float(np.median(v_phi))
                out["disk_r_median"] = float(np.median(r[ok]))
                out["disk_v_phi_positive_frac"] = float(np.mean(v_phi > 0.0))
                out["disk_v_p9999"] = float(np.percentile(speeds, 99.99))
                if stability_context is not None:
                    scales, pot = stability_context
                    vc = scales.circular_speed_ref
                    hotness0 = scales.sigma_r0 / vc
                    velocity_scale = math.hypot(vc, scales.sigma_r0)
                    out.update(
                        {
                            "r_max_over_rtidal": out["r_max"]
                            / max(scales.tidal_radius, 1e-30),
                            "disk_v_median_over_vc": out["disk_v_median"] / vc,
                            "disk_v_p9999_over_vscale": out["disk_v_p9999"]
                            / velocity_scale,
                            "disk_tail_scaled": (
                                out["disk_v_p9999"]
                                / max(out["disk_v_median"], 1e-30)
                                / (1.0 + hotness0)
                            ),
                            "disk_rotation_scaled": out["disk_v_phi_mean"]
                            * (1.0 + hotness0 * hotness0)
                            / vc,
                            "disk_prograde_scaled": (
                                out["disk_v_phi_positive_frac"] - 0.5
                            )
                            * (1.0 + hotness0),
                            "disk_r_median_over_rd": out["disk_r_median"]
                            / scales.disk_scale_length,
                            "disk_circular_speed_ref": vc,
                            "disk_sigma_r0": scales.sigma_r0,
                            "disk_hotness0": hotness0,
                        }
                    )
                    bound_frac, bound_n = _bound_fraction(
                        state,
                        mask,
                        pot,
                        max_samples=max_binding_samples,
                    )
                    out["disk_bound_fraction"] = bound_frac
                    out["disk_bound_sample_size"] = float(bound_n)
    return out


def ic_looks_stable(
    state: ParticleState,
    *,
    diag: dict[str, float] | None = None,
    work_dir: Path | None = None,
    model: Any | None = None,
    model_scaled_thresholds: dict[str, float] | None = None,
    disk_v_median_min: float = 0.8,
    disk_v_max: float = 6.0,
    disk_v_max_ratio: float = 5.0,
    halo_v_max: float = 8.0,
    disk_v_phi_mean_min: float = 0.8,
    disk_v_phi_positive_frac_min: float = 0.9,
    disk_r_median_min: float = 2.5,
) -> bool:
    """Heuristic pre-evolve check (hot tails, collapsed disk rotation, extreme radii)."""
    diag = diag or diagnose_ic_stability(state, work_dir=work_dir, model=model)
    if "disk_v_median_over_vc" in diag:
        thr = {
            **DEFAULT_MODEL_SCALED_IC_THRESHOLDS,
            **(model_scaled_thresholds or {}),
        }
        checks = (
            diag.get("r_max_over_rtidal", 0.0) <= thr["r_max_over_rtidal_max"],
            diag["disk_v_median_over_vc"] >= thr["disk_v_median_over_vc_min"],
            diag.get("disk_v_p9999_over_vscale", math.inf)
            <= thr["disk_v_p9999_over_vscale_max"],
            diag.get("disk_tail_scaled", math.inf) <= thr["disk_tail_scaled_max"],
            diag.get("disk_rotation_scaled", -math.inf)
            >= thr["disk_rotation_scaled_min"],
            diag.get("disk_prograde_scaled", -math.inf)
            >= thr["disk_prograde_scaled_min"],
            diag.get("disk_bound_fraction", 0.0) >= thr["disk_bound_fraction_min"],
            diag.get("disk_r_median_over_rd", 0.0)
            >= thr["disk_r_median_over_rd_min"],
        )
        return all(checks)
    if diag["r_max"] > 250.0:
        return False
    if state.tags is not None:
        disk_med = diag.get("disk_v_median", 0.0)
        disk_max = diag.get("disk_v_max", 0.0)
        if disk_med < disk_v_median_min:
            return False
        if disk_max > disk_v_max:
            return False
        if disk_med > 0.0 and disk_max / disk_med > disk_v_max_ratio:
            return False
        if diag.get("halo_v_max", 0.0) > halo_v_max:
            return False
        if "disk_v_phi_mean" in diag and diag["disk_v_phi_mean"] < disk_v_phi_mean_min:
            return False
        if (
            "disk_v_phi_positive_frac" in diag
            and diag["disk_v_phi_positive_frac"] < disk_v_phi_positive_frac_min
        ):
            return False
        if "disk_r_median" in diag and diag["disk_r_median"] < disk_r_median_min:
            return False
    elif diag.get("v_max", 0.0) > disk_v_max:
        return False
    return True


def explain_ic_instability(
    state: ParticleState | None = None,
    *,
    diag: dict[str, float] | None = None,
    work_dir: Path | None = None,
    model: Any | None = None,
    model_scaled_thresholds: dict[str, float] | None = None,
    disk_v_median_min: float = 0.8,
    disk_v_max: float = 6.0,
    disk_v_max_ratio: float = 5.0,
    halo_v_max: float = 8.0,
    disk_v_phi_mean_min: float = 0.8,
    disk_v_phi_positive_frac_min: float = 0.9,
    disk_r_median_min: float = 2.5,
) -> str:
    """
    Human-readable summary of why :func:`ic_looks_stable` would fail.

    Typical causes: collapsed ``cordbh.dat`` (low disk median |v|), hot disk
    outliers (high disk ``v_max`` or ``v_max/v_median``), an incompatible
    DBH grid / Toomre-Q target for the coarse ``diskdf`` solve, cylindrical
    velocities written as Cartesian (near-zero mean ``v_phi``), or disk
    positions sampled without the cylindrical ``R`` Jacobian (``R`` median too
    small → hot inner disk + retrograde fraction).
    """
    if diag is None:
        if state is None:
            raise ValueError("provide state or diag")
        diag = diagnose_ic_stability(state, work_dir=work_dir, model=model)
    reasons: list[str] = []
    if "disk_v_median_over_vc" in diag:
        thr = {
            **DEFAULT_MODEL_SCALED_IC_THRESHOLDS,
            **(model_scaled_thresholds or {}),
        }
        scaled_checks = (
            (
                "r_max_over_rtidal",
                "r_max/r_tidal",
                "max",
                thr["r_max_over_rtidal_max"],
            ),
            (
                "disk_v_median_over_vc",
                "disk median |v|/v_c(2.5 R_d)",
                "min",
                thr["disk_v_median_over_vc_min"],
            ),
            (
                "disk_v_p9999_over_vscale",
                "disk v_99.99/[v_c^2+sigma_R,0^2]^1/2",
                "max",
                thr["disk_v_p9999_over_vscale_max"],
            ),
            (
                "disk_tail_scaled",
                "hot-tail ratio corrected for sigma_R,0/v_c",
                "max",
                thr["disk_tail_scaled_max"],
            ),
            (
                "disk_rotation_scaled",
                "asymmetric-drift-scaled mean rotation",
                "min",
                thr["disk_rotation_scaled_min"],
            ),
            (
                "disk_prograde_scaled",
                "temperature-scaled prograde excess",
                "min",
                thr["disk_prograde_scaled_min"],
            ),
            (
                "disk_bound_fraction",
                "disk bound fraction",
                "min",
                thr["disk_bound_fraction_min"],
            ),
            (
                "disk_r_median_over_rd",
                "disk R_median/R_d",
                "min",
                thr["disk_r_median_over_rd_min"],
            ),
        )
        for key, label, direction, limit in scaled_checks:
            value = diag.get(key)
            if value is None:
                reasons.append(f"missing model-scaled diagnostic {key}")
            elif direction == "min" and value < limit:
                reasons.append(f"{label}={value:.4f} < {limit:.4f}")
            elif direction == "max" and value > limit:
                reasons.append(f"{label}={value:.4f} > {limit:.4f}")
        if not reasons:
            return "IC velocity structure failed model-scaled checks (unknown)"
        return "; ".join(reasons)
    if diag["r_max"] > 250.0:
        reasons.append(f"r_max={diag['r_max']:.1f} kpc > 250 (escaped particles)")
    has_components = "disk_v_median" in diag or "disk_v_max" in diag or "halo_v_max" in diag
    if has_components or (state is not None and state.tags is not None):
        disk_med = diag.get("disk_v_median", 0.0)
        disk_max = diag.get("disk_v_max", 0.0)
        halo_max = diag.get("halo_v_max", 0.0)
        if disk_med < disk_v_median_min:
            reasons.append(
                f"disk median |v|={disk_med:.3f} < {disk_v_median_min} "
                "(collapsed rotation — invalid or stale cordbh.dat / diskdf failure)"
            )
        if disk_max > disk_v_max:
            reasons.append(
                f"disk v_max={disk_max:.2f} > {disk_v_max} "
                "(hot outliers — bad cordbh.dat or DBH grid too coarse for diskdf)"
            )
        if disk_med > 0.0 and disk_max / disk_med > disk_v_max_ratio:
            reasons.append(
                f"disk v_max/v_median={disk_max / disk_med:.1f} > {disk_v_max_ratio} "
                "(hot tail vs bulk rotation — bad cordbh.dat or coarse DBH grid)"
            )
        if halo_max > halo_v_max:
            reasons.append(f"halo v_max={halo_max:.2f} > {halo_v_max}")
        vphi = diag.get("disk_v_phi_mean")
        if vphi is not None and vphi < disk_v_phi_mean_min:
            reasons.append(
                f"disk mean v_phi={vphi:.3f} < {disk_v_phi_mean_min} "
                "(weak net rotation — check gendisk Cartesian conversion / invu R sampling)"
            )
        frac = diag.get("disk_v_phi_positive_frac")
        if frac is not None and frac < disk_v_phi_positive_frac_min:
            reasons.append(
                f"disk prograde frac={frac:.3f} < {disk_v_phi_positive_frac_min} "
                "(too many retrograde orbits — usually central oversampling from missing invu)"
            )
        rmed = diag.get("disk_r_median")
        if rmed is not None and rmed < disk_r_median_min:
            reasons.append(
                f"disk R median={rmed:.2f} kpc < {disk_r_median_min} "
                "(positions too central — gendisk must use invu, not -rd*ln(u))"
            )
    elif diag.get("v_max", 0.0) > disk_v_max:
        reasons.append(f"v_max={diag['v_max']:.2f} > {disk_v_max}")
    if not reasons:
        return "IC velocity structure failed heuristic checks (unknown)"
    return "; ".join(reasons)


def invalidate_ic_artifacts(
    work_dir: Path,
    *,
    include_evolve: bool = True,
    include_solve_marker: bool = True,
) -> list[str]:
    """
    Remove cached solve/sample/evolve outputs so IC generation re-runs fresh.

    Use when :func:`ic_looks_stable` fails on loaded particles — common when
    ``skip_done=True`` reused ``cordbh.dat`` or particle files from before a
    ``diskdf`` fix or config change.
    """
    work_dir = Path(work_dir)
    removed: list[str] = []
    names: list[str] = []
    if include_solve_marker:
        names.append(".done_solve")
    names.extend(
        [
            ".done_sample",
            ".done_evolve",
            "cordbh.dat",
            "toomre2.5",
            "disk",
            "halo",
            "bulge",
            "evolve_result.json",
        ]
    )
    if include_evolve:
        names.append("evolution/final.dat")
    for name in names:
        path = work_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def freq_table_sane(
    model,
    work_dir: Path,
    *,
    r_factor: float = 2.5,
    omega_min: float = 0.25,
) -> bool:
    """
    Return False when ``freqdbh.dat`` implies a collapsed or inconsistent disk potential.

    Python Poisson solves on coarse MW grids often under-predict ``Omega(R)`` at the
    disk scale; legacy ``diskdf`` then fails with NaN splines and an empty ``cordbh.dat``.
    """
    from galacticsics.io.formats import read_frequency_table

    freq_path = Path(work_dir) / "freqdbh.dat"
    if not freq_path.is_file():
        return False
    disk = getattr(model, "disk", None)
    if disk is None or not getattr(disk, "enabled", False):
        return True
    try:
        freq = read_frequency_table(freq_path)
        r = float(r_factor * disk.scale_length)
        omega = freq.omega(r)
        kappa = freq.kappa(r)
    except (ValueError, OSError, IndexError):
        return False
    return omega >= omega_min and kappa > 0.0 and omega < 5.0


DEFAULT_BENCHMARK_N = 10_000


def subsample_state(state: ParticleState, n: int, *, seed: int = 0) -> ParticleState:
    """Random subset for faster benchmarks (preserves tags/type_id)."""
    if state.n <= n:
        return state
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(state.n, size=n, replace=False))
    kwargs: dict[str, Any] = {
        "pos": state.pos[idx],
        "vel": state.vel[idx],
        "mass": state.mass[idx],
        "eps": state.eps[idx],
    }
    if state.type_id is not None:
        kwargs["type_id"] = state.type_id[idx]
    if state.timestep_bin is not None:
        kwargs["timestep_bin"] = state.timestep_bin[idx]
    if state.tags is not None:
        kwargs["tags"] = state.tags[idx]
    return ParticleState.from_arrays(**kwargs)


@dataclass
class ForceBenchRow:
    label: str
    ms_per_call: float


@dataclass
class ForceBenchResult:
    n_particles: int
    omp_threads: int
    rows: list[ForceBenchRow]
    bh_split_full: BhCBreakdown | None = None
    bh_split_active: BhCBreakdown | None = None
    n_active_step1: int | None = None

    def ms(self, label: str) -> float:
        for row in self.rows:
            if row.label == label:
                return row.ms_per_call
        raise KeyError(label)


def _tiered_bins(state: ParticleState) -> np.ndarray:
    registry = TypeRegistry.default_galaxy()
    ts = TimestepConfig(dt_base=0.05, eta=0.025, max_bin=7, update_every=2)
    bins = (
        state.timestep_bin.copy()
        if state.timestep_bin is not None
        else np.zeros(state.n, dtype=np.int32)
    )
    if state.timestep_bin is None:
        ctx = ForceContext(
            config=ForceConfig(
                method="bh_c",
                theta=0.6,
                rebuild_every=10,
                bh_optimizations=BhOptimizationsConfig(preset="optimized"),
            )
        )
        acc = ctx.accel_at_pos(state, state.pos)
        ctx.after_force_eval()
        bins = update_timestep_bins(acc, state.eps, state.type_id, registry, bins, ts)
    return bins


def bench_force_eval(
    state: ParticleState,
    *,
    preset: str = "optimized",
    theta: float = 0.6,
    rebuild_every: int = 1,
    target_indices: np.ndarray | None = None,
    n_repeat: int = 4,
) -> float:
    """Milliseconds per force evaluation via :class:`ForceContext`."""
    if not extension_available():
        raise ImportError("bh_c extension required")

    bh = BhOptimizationsConfig(preset=preset)  # type: ignore[arg-type]
    ctx = ForceContext(
        config=ForceConfig(
            method="bh_c",
            theta=theta,
            rebuild_every=rebuild_every,
            bh_optimizations=bh,
        )
    )
    pos = state.pos
    ctx.reset()
    for _ in range(rebuild_every):
        ctx.accel_at_pos(state, pos, target_indices=target_indices)
        ctx.after_force_eval()
    ctx.reset()
    for _ in range(rebuild_every):
        ctx.accel_at_pos(state, pos, target_indices=target_indices)
        ctx.after_force_eval()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        for _ in range(rebuild_every):
            ctx.accel_at_pos(state, pos, target_indices=target_indices)
            ctx.after_force_eval()
    return (time.perf_counter() - t0) / (n_repeat * rebuild_every) * 1000


def run_force_benchmark(
    state: ParticleState,
    *,
    n_repeat: int = 4,
) -> ForceBenchResult:
    """Standard bh_c comparison suite (legacy vs optimized, rebuild, active subset)."""
    bins = _tiered_bins(state)
    active = np.nonzero(active_mask_for_step(1, bins))[0]

    rows = [
        ForceBenchRow(
            "legacy_full_rebuild1",
            bench_force_eval(state, preset="legacy", n_repeat=n_repeat),
        ),
        ForceBenchRow(
            "optimized_full_rebuild1",
            bench_force_eval(state, preset="optimized", n_repeat=n_repeat),
        ),
        ForceBenchRow(
            "optimized_full_rebuild10",
            bench_force_eval(state, preset="optimized", rebuild_every=10, n_repeat=n_repeat),
        ),
        ForceBenchRow(
            "optimized_active_rebuild10",
            bench_force_eval(
                state,
                preset="optimized",
                rebuild_every=10,
                target_indices=active,
                n_repeat=n_repeat,
            ),
        ),
    ]
    bh_full = time_bh_c_components(state.pos, state.mass, state.eps, theta=0.6, n_repeat=3)
    bh_active = time_bh_c_components(
        state.pos,
        state.mass,
        state.eps,
        theta=0.6,
        target_indices=active,
        n_repeat=3,
    )
    return ForceBenchResult(
        n_particles=state.n,
        omp_threads=int(os.environ.get("OMP_NUM_THREADS", "1")),
        rows=rows,
        bh_split_full=bh_full,
        bh_split_active=bh_active,
        n_active_step1=int(active.size),
    )


def format_force_benchmark(result: ForceBenchResult) -> str:
    lines = [
        f"N={result.n_particles:,} OMP={result.omp_threads}",
        f"tiered active step1: {result.n_active_step1:,}",
        "--- ms per force eval ---",
    ]
    base = result.ms("optimized_full_rebuild1")
    for row in result.rows:
        speedup = base / row.ms_per_call if row.ms_per_call > 0 else float("nan")
        lines.append(f"  {row.ms_per_call:7.1f} ms ({speedup:.2f}x)  {row.label}")
    if result.bh_split_full is not None:
        bh = result.bh_split_full
        lines.append(
            f"bh_c split full: build={bh.ms_build:.1f} walk={bh.ms_walk:.1f} "
            f"total={bh.ms_total:.1f} ms (build {100*bh.build_fraction:.0f}%)"
        )
    if result.bh_split_active is not None and result.n_active_step1:
        bh = result.bh_split_active
        lines.append(
            f"bh_c split active ({result.n_active_step1:,}): walk={bh.ms_walk:.1f} "
            f"total={bh.ms_total:.1f} ms"
        )
    return "\n".join(lines)


@dataclass
class DensitySanityResult:
    work_dir: Path
    n_particles: int
    halo_rho_drift: float
    disk_sigma_drift: float
    disk_peak_r_ic: float
    disk_peak_r_final: float
    disk_z_rms_ic: float
    disk_z_rms_final: float
    halo_z_rms_ic: float
    halo_z_rms_final: float
    dE_over_E0: float | None
    mass_disk_rel_err: float
    mass_halo_rel_err: float


def _peak_radius(profile) -> float:
    values = profile.rho if hasattr(profile, "rho") else profile.sigma
    counts = profile.counts
    # Skip underpopulated inner bins (log-spaced shells often empty at small r).
    valid = counts >= max(20, int(0.01 * np.max(counts)))
    if not np.any(valid):
        return float(profile.r_mid[int(np.argmax(values))])
    idx = int(np.argmax(np.where(valid, values, -np.inf)))
    return float(profile.r_mid[idx])


def _z_rms(state: ParticleState, name: str) -> float:
    mask = component_mask(state, name)
    if not np.any(mask):
        return float("nan")
    z = state.pos[mask, 2]
    m = state.mass[mask]
    zcm = float(np.sum(m * z) / np.sum(m))
    return float(np.sqrt(np.sum(m * (z - zcm) ** 2) / np.sum(m)))


def _component_mass(state: ParticleState, name: str) -> float:
    mask = component_mask(state, name)
    return float(np.sum(state.mass[mask])) if np.any(mask) else 0.0


def check_density_evolution(work_dir: Path) -> DensitySanityResult:
    """Compare IC vs final density morphology for one campaign work directory."""
    ic = load_ic_state(work_dir / "ic_state.npz")
    final = load_evolved_state(work_dir, ic)
    if final is None:
        raise FileNotFoundError(f"missing evolution/final.dat in {work_dir}")

    drift = density_drift_metrics(ic, final)
    disk_ic = disk_surface_profile(ic)
    disk_f = disk_surface_profile(final)

    dE: float | None = None
    evo_path = work_dir / "evolve_result.json"
    if evo_path.is_file():
        import json

        dE = float(json.loads(evo_path.read_text()).get("dE_over_E0", 0.0))

    m_disk_ic = _component_mass(ic, "disk")
    m_disk_f = _component_mass(final, "disk")
    m_halo_ic = _component_mass(ic, "halo")
    m_halo_f = _component_mass(final, "halo")

    return DensitySanityResult(
        work_dir=work_dir,
        n_particles=ic.n,
        halo_rho_drift=drift["halo_rho_drift"],
        disk_sigma_drift=drift["disk_sigma_drift"],
        disk_peak_r_ic=_peak_radius(disk_ic),
        disk_peak_r_final=_peak_radius(disk_f),
        disk_z_rms_ic=_z_rms(ic, "disk"),
        disk_z_rms_final=_z_rms(final, "disk"),
        halo_z_rms_ic=_z_rms(ic, "halo"),
        halo_z_rms_final=_z_rms(final, "halo"),
        dE_over_E0=dE,
        mass_disk_rel_err=abs(m_disk_f - m_disk_ic) / max(m_disk_ic, 1e-30),
        mass_halo_rel_err=abs(m_halo_f - m_halo_ic) / max(m_halo_ic, 1e-30),
    )


def assert_density_sanity(result: DensitySanityResult) -> None:
    """Raise AssertionError when evolved profiles fail basic physical checks."""
    assert result.halo_rho_drift < 0.75, f"halo rho drift {result.halo_rho_drift:.2f}"
    assert result.disk_sigma_drift < 0.75, f"disk sigma drift {result.disk_sigma_drift:.2f}"
    assert 0.3 < result.disk_peak_r_ic < 10.0, f"IC disk peak R {result.disk_peak_r_ic:.2f} kpc"
    assert 0.3 < result.disk_peak_r_final < 12.0, f"final disk peak R {result.disk_peak_r_final:.2f} kpc"
    assert result.disk_z_rms_final < result.halo_z_rms_final, "disk should stay flatter than halo"
    assert result.mass_disk_rel_err < 0.02, "disk mass not conserved"
    assert result.mass_halo_rel_err < 0.02, "halo mass not conserved"
    if result.dE_over_E0 is not None:
        assert result.dE_over_E0 < 0.12, f"|dE/E0|={result.dE_over_E0:.3f}"


def is_stable_evolution(work_dir: Path, *, r_max_kpc: float = 280.0) -> bool:
    """Reject runs where particles have clearly escaped or ICs were already hot."""
    ic_path = work_dir / "ic_state.npz"
    if not ic_path.is_file() or not (work_dir / "evolution" / "final.dat").is_file():
        return False
    ic = load_ic_state(ic_path)
    if not ic_looks_stable(ic):
        return False
    final = load_evolved_state(work_dir, ic)
    if final is None:
        return False
    r = np.linalg.norm(final.pos, axis=1)
    if float(np.max(r)) > r_max_kpc:
        return False
    evo_path = work_dir / "evolve_result.json"
    if evo_path.is_file():
        import json

        dE = float(json.loads(evo_path.read_text()).get("dE_over_E0", 0.0))
        if dE > 0.12:
            return False
    return True


def find_evolved_campaign_dirs(root: Path, *, stable_only: bool = True) -> list[Path]:
    """Work dirs with both IC and final snapshots."""
    if not root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not (child / "ic_state.npz").is_file():
            continue
        if not (child / "evolution" / "final.dat").is_file():
            continue
        if stable_only and not is_stable_evolution(child):
            continue
        out.append(child)
    return out
