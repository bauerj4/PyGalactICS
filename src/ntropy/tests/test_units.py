"""Tests for GalactICS unit helpers."""

from __future__ import annotations

import pytest

from ntropy.units import (
    CODE_TIME_PER_GYR,
    DEFAULT_SIM_DT,
    DEFAULT_SIM_N_STEPS,
    code_time_to_gyr,
    code_time_to_myr,
    format_simulation_duration,
)


def test_default_simulation_span_is_five_gyr():
    span_gyr = code_time_to_gyr(DEFAULT_SIM_DT * DEFAULT_SIM_N_STEPS)
    assert span_gyr == pytest.approx(5.0, rel=0.02)


def test_dt_is_one_myr_per_step():
    assert code_time_to_myr(DEFAULT_SIM_DT) == pytest.approx(1.0, rel=0.02)


def test_code_time_per_gyr():
    assert CODE_TIME_PER_GYR == pytest.approx(102.25, rel=0.02)


def test_format_simulation_duration_mentions_gyr():
    text = format_simulation_duration(DEFAULT_SIM_DT, DEFAULT_SIM_N_STEPS)
    assert "5.00 Gyr" in text or "5.0 Gyr" in text
    assert "5000 steps" in text
