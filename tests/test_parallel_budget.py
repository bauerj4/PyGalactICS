"""Tests for MPI/OpenMP core budgeting."""

from __future__ import annotations

from galacticsics.campaign.parallel_budget import (
    format_parallel_budget,
    parallel_env_dict,
    resolve_parallel_budget,
)


def test_resolve_parallel_budget_respects_fraction():
    budget = resolve_parallel_budget(4, core_fraction=0.75)
    assert budget.total_cores >= 1
    assert budget.budget_cores >= 1
    assert budget.budget_cores <= budget.total_cores
    assert budget.mpi_ranks >= 1
    assert budget.omp_threads >= 1
    assert budget.used_cores <= budget.budget_cores


def test_mpi_ranks_clamped_to_budget():
    budget = resolve_parallel_budget(999, core_fraction=0.5)
    assert budget.mpi_ranks <= budget.budget_cores


def test_parallel_env_dict_matches_omp_threads():
    budget = resolve_parallel_budget(2, core_fraction=0.75)
    env = parallel_env_dict(budget)
    assert env["OMP_NUM_THREADS"] == str(budget.omp_threads)
    assert env["OPENBLAS_NUM_THREADS"] == str(budget.omp_threads)


def test_format_parallel_budget():
    budget = resolve_parallel_budget(2, core_fraction=0.75)
    text = format_parallel_budget(budget)
    assert "MPI" in text
    assert "OMP" in text
