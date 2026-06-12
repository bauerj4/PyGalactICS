#!/usr/bin/env python3
"""
Solve a small NFW halo potential using the legacy dbh executable.

Requires: make legacy-build (or cd legacy/fortran && make all install)

This example uses a reduced radial grid for speed. For production Milky Way
models, use GalaxyModel.milky_way_disk_halo() with nr=20000.
"""

from galacticsics.models import GalaxyModel
from galacticsics.potential import evaluate_potential
from galacticsics.potential.solver import solve_potential

model = GalaxyModel.nfw_halo_only()
model.grid.nr = 2000
model.grid.dr = 0.05
model.grid.lmax = 0

print("Running legacy dbh solver (NFW halo only, nr=800)...")
result = solve_potential(model, cleanup=False)
print(f"Tidal radius: {result.diagnostics.tidal_radius:.2f} kpc")
print(f"Component masses: {result.diagnostics.component_masses}")
print(f"Work directory: {result.diagnostics.work_dir}")

r = 5.0
psi = evaluate_potential(result.potential, r, 0.0)
print(f"Psi(R={r}, z=0) = {psi:.4f}")
