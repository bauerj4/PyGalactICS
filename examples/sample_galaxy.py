#!/usr/bin/env python3
"""Sample disk+halo particles from the Milky Way reference model."""

from galacticsics.builder import GalaxyBuilder
from galacticsics.models import GalaxyModel

MODEL_DIR = "models/MilkyWay"

builder = GalaxyBuilder(
    model=GalaxyModel.milky_way_disk_halo(),
    model_dir=MODEL_DIR,
).load_artifacts()

particles = builder.sample(
    n_disk=500,
    n_halo=500,
    seed=-99,
    cleanup=False,
)

for name, ps in particles.items():
    print(f"{name}: N={len(ps)}, M={ps.total_mass:.4f}, sigma_v={ps.velocity_dispersion()}")
