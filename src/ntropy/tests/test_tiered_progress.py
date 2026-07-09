"""Tests for ntropy tiered progress lines."""

from __future__ import annotations

from ntropy.analysis.tiered_diagnostics import StepDiagnostics
from ntropy.integrators.tiered_progress import (
    NtropyProgressReporter,
    format_ntropy_progress_from_dict,
    format_ntropy_progress_line,
)


def test_format_ntropy_progress_line():
    record = StepDiagnostics(
        step=12,
        t_code=1.2,
        t_gyr=0.0117,
        energy=-1.0,
        dE_over_E0=1e-4,
        n_active=50000,
        n_particles=200000,
        active_fraction=0.25,
        bin_counts=[100, 200, 300, 400, 0, 0, 0],
        by_type={
            "disk": {"mean_bin": 1.2, "n_particles": 100000},
            "halo": {"mean_bin": 2.8, "n_particles": 100000},
        },
        mean_accel=0.05,
    )
    line = format_ntropy_progress_line(record, n_steps_total=409)
    assert "step   12/409" in line
    assert "active  25.0%" in line


def test_format_ntropy_progress_from_dict():
    rec = {
        "step": 3,
        "t_gyr": 0.01,
        "n_active": 1000,
        "active_fraction": 0.5,
        "dE_over_E0": 2e-5,
        "bin_counts": [10, 20, 30],
        "by_type": {"disk": {"mean_bin": 1.0, "n_particles": 2000}},
    }
    line = format_ntropy_progress_from_dict(rec, n_steps_total=100)
    assert "step    3/100" in line
    assert "active  50.0%" in line
    assert "disk μb=1.0" in line


def test_ntropy_progress_reporter_prints(capsys):
    rep = NtropyProgressReporter(
        n_steps_total=10,
        n_particles=100,
        end_time_gyr=0.1,
        print_every=1,
    )
    rep.banner(dt_base=0.025)
    record = StepDiagnostics(
        step=1,
        t_code=0.025,
        t_gyr=0.00024,
        energy=-1.0,
        dE_over_E0=0.0,
        n_active=50,
        n_particles=100,
        active_fraction=0.5,
        bin_counts=[50, 50, 0, 0, 0, 0, 0],
        by_type={},
    )
    rep.update(record)
    out = capsys.readouterr().err
    assert "[ntropy]" in out
    assert "step    1/10" in out
