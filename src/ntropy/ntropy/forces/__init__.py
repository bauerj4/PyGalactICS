"""
Force computation backends.

Brute force
    :func:`compute_forces_brute` — vectorized $O(N^2)$ Plummer sum.

Barnes–Hut (Python)
    :class:`BarnesHutTree`, :func:`compute_forces_bh` — reference octree in pure Python.

Barnes–Hut (C)
    :class:`BarnesHutTreeC`, :func:`compute_forces_bh_c` — compiled tree in ``forces/c/``;
    use ``ForceConfig.method = "bh_c"`` or see ``forces/c/PARALLEL.md`` for MPI.
"""

from ntropy.forces.brute import compute_forces_brute
from ntropy.forces.bhtree import BarnesHutTree, compute_forces_bh
from ntropy.forces.bhtree_c import (
    BarnesHutTreeC,
    compute_forces_bh_c,
    extension_available,
)

__all__ = [
    "compute_forces_brute",
    "BarnesHutTree",
    "compute_forces_bh",
    "BarnesHutTreeC",
    "compute_forces_bh_c",
    "extension_available",
]
