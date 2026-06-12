"""MPI domain decomposition."""

from ntropy.parallel.domains import domain_slices, peano_keys, sort_by_peano
from ntropy.parallel.mpi import compute_forces_mpi, get_comm, mpi_available
from ntropy.parallel.pool import compute_forces_parallel

__all__ = [
    "peano_keys",
    "sort_by_peano",
    "domain_slices",
    "compute_forces_mpi",
    "compute_forces_parallel",
    "mpi_available",
    "get_comm",
]
