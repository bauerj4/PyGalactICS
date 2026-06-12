#!/usr/bin/env python3
"""Milky Way reference model: load artifacts and evaluate potential."""

from galacticsics.builder import GalaxyBuilder
from galacticsics.models import GalaxyModel
from galacticsics.potential import evaluate_force, evaluate_potential

MODEL_DIR = "models/MilkyWay"

builder = GalaxyBuilder(model=GalaxyModel.milky_way_disk_halo(), model_dir=MODEL_DIR).load_artifacts()
pot = builder.potential

for r in (1.0, 5.0, 8.0, 12.0):
    psi = evaluate_potential(pot, r, 0.0)
    fr, fz, _ = evaluate_force(pot, r, 0.0)
    print(f"R={r:5.1f} kpc  Psi={psi:10.4f}  F_R={fr:10.4f}  F_z={fz:10.4f}")

if builder.disk_correction:
    r_test = 2.5 * pot.model.disk.scale_length
    print(f"Toomre Q proxy at 2.5 Rd: f_d={builder.disk_correction.f_d_at(r_test):.4f}")
