"""Human-readable summaries of tiered leapfrog timestep settings."""

from __future__ import annotations

from dataclasses import dataclass

from ntropy.units import code_time_to_gyr, code_time_to_myr, gyr_to_code_time


@dataclass(frozen=True)
class TieredTimestepSummary:
    """Derived quantities from tiered integrator timestep knobs."""

    dt_base_code: float
    dt_base_myr: float
    end_time_gyr: float
    n_fine_substeps: int
    max_bin: int
    eta: float
    dt_coarse_code: float
    dt_coarse_myr: float
    update_every: int
    integrator_order: int


def summarize_tiered_timestep(
    *,
    dt_base: float,
    end_time_gyr: float,
    max_bin: int = 6,
    eta: float = 0.025,
    update_every: int = 1,
    integrator_order: int = 2,
) -> TieredTimestepSummary:
    """Compute fine-substep count and coarse/fine timestep spans."""
    t_end_code = gyr_to_code_time(end_time_gyr)
    n_fine = max(1, int(round(t_end_code / dt_base)))
    dt_coarse = dt_base * (2 ** max_bin)
    return TieredTimestepSummary(
        dt_base_code=dt_base,
        dt_base_myr=code_time_to_myr(dt_base),
        end_time_gyr=end_time_gyr,
        n_fine_substeps=n_fine,
        max_bin=max_bin,
        eta=eta,
        dt_coarse_code=dt_coarse,
        dt_coarse_myr=code_time_to_myr(dt_coarse),
        update_every=update_every,
        integrator_order=integrator_order,
    )


def format_tiered_timestep_summary(summary: TieredTimestepSummary) -> str:
    """Multi-line string for notebook / campaign banners."""
    duration_code = summary.end_time_gyr / code_time_to_gyr(1.0)
    return "\n".join(
        [
            "Tiered timestep:",
            f"  dt_base     : {summary.dt_base_code:.5f} code "
            f"({summary.dt_base_myr:.3f} Myr, bin 0)",
            f"  dt_max      : {summary.dt_coarse_code:.4f} code "
            f"({summary.dt_coarse_myr:.2f} Myr, bin {summary.max_bin})",
            f"  eta         : {summary.eta:g}",
            f"  bin update  : every {summary.update_every} fine substep(s)",
            f"  integrator  : order-{summary.integrator_order} leapfrog",
            f"  duration    : {summary.end_time_gyr:g} Gyr "
            f"= {duration_code:.2f} code units in {summary.n_fine_substeps} fine substeps",
        ]
    )
