"""GalactICS code-unit conversions (G=1, length kpc, velocity 100 km/s)."""

from __future__ import annotations

G = 1.0

# 1 code time unit = 1 kpc / (100 km/s) ≈ 9.78 Myr
MYR_PER_CODE_TIME = 9.78
GYR_PER_CODE_TIME = MYR_PER_CODE_TIME / 1000.0
CODE_TIME_PER_GYR = 1.0 / GYR_PER_CODE_TIME

# Default long-run cadence: 1000 steps per Gyr for 5 Gyr
DEFAULT_STEPS_PER_GYR = 1000
DEFAULT_SIM_DURATION_GYR = 5.0
DEFAULT_SIM_N_STEPS = int(DEFAULT_SIM_DURATION_GYR * DEFAULT_STEPS_PER_GYR)
DEFAULT_SIM_DT = (1.0 / DEFAULT_STEPS_PER_GYR) / GYR_PER_CODE_TIME


def code_time_to_myr(t_code: float) -> float:
    """Convert code time to megayears."""
    return t_code * MYR_PER_CODE_TIME


def code_time_to_gyr(t_code: float) -> float:
    """Convert code time to gigayears."""
    return t_code * GYR_PER_CODE_TIME


def gyr_to_code_time(t_gyr: float) -> float:
    """Convert gigayears to code time."""
    return t_gyr / GYR_PER_CODE_TIME


def format_simulation_duration(dt: float, n_steps: int) -> str:
    """
    Human-readable simulation span for a (dt, n_steps) pair.

    Parameters
    ----------
    dt : float
        Timestep in code time units.
    n_steps : int
        Number of integration steps.

    Returns
    -------
    text : str
        One-line summary with code and physical units.
    """
    span_code = dt * n_steps
    span_gyr = code_time_to_gyr(span_code)
    span_myr = code_time_to_myr(span_code)
    dt_myr = code_time_to_myr(dt)
    return (
        f"{n_steps} steps × Δt={dt:.4f} code units "
        f"({dt_myr:.2f} Myr/step) ≈ {span_code:.1f} code units "
        f"≈ {span_gyr:.2f} Gyr ({span_myr:.0f} Myr)"
    )
