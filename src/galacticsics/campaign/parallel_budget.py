"""MPI + OpenMP core budgeting for campaign evolve runs."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelBudget:
    """Resolved MPI rank count and per-rank OpenMP thread count."""

    total_cores: int
    budget_cores: int
    mpi_ranks: int
    omp_threads: int

    @property
    def used_cores(self) -> int:
        return self.mpi_ranks * self.omp_threads


def resolve_parallel_budget(
    mpi_ranks: int = 2,
    *,
    core_fraction: float = 0.75,
) -> ParallelBudget:
    """
    Reserve ``core_fraction`` of logical CPUs for ntropy (MPI × OpenMP).

    Parameters
    ----------
    mpi_ranks : int
        Requested MPI rank count (clamped to ``budget_cores``).
    core_fraction : float
        Share of ``os.cpu_count()`` to use (default ``0.75`` leaves headroom
        for the OS, browser, and IDE).

    Returns
    -------
    ParallelBudget
        ``omp_threads = budget_cores // mpi_ranks`` (at least 1).
    """
    total = os.cpu_count() or 1
    fraction = min(max(float(core_fraction), 0.05), 1.0)
    budget = max(1, int(total * fraction))
    ranks = max(1, min(int(mpi_ranks), budget))
    omp_threads = max(1, budget // ranks)
    return ParallelBudget(
        total_cores=total,
        budget_cores=budget,
        mpi_ranks=ranks,
        omp_threads=omp_threads,
    )


def parallel_env_dict(budget: ParallelBudget) -> dict[str, str]:
    """Environment variables that cap BLAS/OpenMP threads per worker."""
    n = str(budget.omp_threads)
    return {
        "OMP_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "NUMEXPR_NUM_THREADS": n,
        "VECLIB_MAXIMUM_THREADS": n,
    }


def apply_parallel_env(budget: ParallelBudget) -> None:
    """Set process-wide thread limits (serial evolve or notebook kernel)."""
    os.environ.update(parallel_env_dict(budget))


def format_parallel_budget(budget: ParallelBudget) -> str:
    """One-line summary for campaign logs."""
    return (
        f"{budget.total_cores} cores → budget {budget.budget_cores} "
        f"(MPI {budget.mpi_ranks} × OMP {budget.omp_threads} = {budget.used_cores})"
    )
