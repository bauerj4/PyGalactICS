#!/usr/bin/env python3
"""Quick MPI force-eval timing on campaign ICs."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ntropy.config import load_config
from ntropy.forces.context import ForceContext
from ntropy.particles import ParticleState
from ntropy.parallel.mpi import mpi_rank0

work = Path("notebooks/artifacts/campaign_walkthrough/base_mw/ceaeb7ed7f9c")
with np.load(work / "ic_state.npz", allow_pickle=True) as d:
    state = ParticleState.from_arrays(
        d["pos"], d["vel"], d["mass"], d["eps"], type_id=d["type_id"]
    )
cfg = load_config(work / "ntropy_config.json")
ctx = ForceContext(
    config=cfg.force, parallel_enabled=True, n_workers=cfg.parallel.n_workers
)
pos = state.pos
for _ in range(1):
    ctx.accel_at_pos(state, pos)
    ctx.after_force_eval()
t0 = time.perf_counter()
n = 5
for _ in range(n):
    ctx.accel_at_pos(state, pos)
    ctx.after_force_eval()
ms = (time.perf_counter() - t0) / n * 1000
if mpi_rank0():
    omp = os.environ.get("OMP_NUM_THREADS", "?")
    print(f"OMP_NUM_THREADS={omp}")
    print(f"MPI force eval: {ms:.0f} ms")
    print(f"2 evals/substep: {2 * ms / 1000:.2f} s")
    print(f"511 substeps (force only): {2 * ms * 511 / 1000 / 60:.1f} min")
