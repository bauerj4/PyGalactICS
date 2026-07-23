"""Tests for tiered timestep summary helpers."""

from __future__ import annotations

from galacticsics.campaign.timestep_summary import (
    format_tiered_timestep_summary,
    summarize_tiered_timestep,
)


def test_summarize_tiered_timestep_substep_count():
    summary = summarize_tiered_timestep(
        dt_base=0.025,
        end_time_gyr=0.1,
        max_bin=6,
    )
    assert summary.n_fine_substeps == 409
    assert "409" in format_tiered_timestep_summary(summary)
