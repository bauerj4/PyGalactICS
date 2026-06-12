#!/usr/bin/env python3
"""Two-step halo-first workflow: fit halo, then generate disk in fixed halo."""

from __future__ import annotations

from pathlib import Path

from galacticsics.builder import GalaxyBuilder
from galacticsics.fitting.halo_particles import apply_nfw_fit, estimate_nfw_from_particles
from galacticsics.io.formats import read_particles_ascii
from galacticsics.models import GalaxyModel
from galacticsics.potential import evaluate_potential

MW = Path(__file__).resolve().parents[1] / "models" / "MilkyWay"


def main() -> None:
    # Optional: estimate NFW parameters from pre-existing halo particles (Xhalo).
    xhalo = MW / "Xhalo"
    model = GalaxyModel.milky_way_disk_halo()
    if xhalo.is_file():
        data = read_particles_ascii(xhalo, max_particles=20_000)
        fit = estimate_nfw_from_particles(
            data["x"], data["y"], data["z"], data["mass"], center=True
        )
        model = apply_nfw_fit(model, fit)
        print(f"NFW fit from Xhalo: v0={fit.v0:.3f}, a={fit.a:.2f} kpc, rms={fit.rms_residual:.3f}")

    # Use a smaller grid for the demo (full MW uses nr=20000).
    model.grid.nr = 4000
    model.grid.lmax = 4

    builder = GalaxyBuilder(model=model)
    print("Step 1: halo-only dbh solve → h.dat")
    halo_step = builder.solve_halo_first(work_dir="/tmp/galacticsics_halo_demo", cleanup=False)
    print(f"  h.dat: {halo_step.work_dir / 'h.dat'}")

    print("Step 2: baryons in fixed halo → merged dbh.dat")
    baryon_step = builder.solve_baryons_in_fixed_halo(
        halo_work_dir=halo_step.work_dir,
        work_dir="/tmp/galacticsics_baryon_demo",
        cleanup=False,
    )
    psi = evaluate_potential(baryon_step.potential, s=8.0, z=0.0)
    print(f"  Psi(R=8, z=0) = {psi:.4f}")
    print(f"  merged dbh.dat: {baryon_step.work_dir / 'dbh.dat'}")

    # Sample disk only; use Xhalo for halo particles.
    print("Sampling disk (external halo from Xhalo)")
    builder.model_dir = str(baryon_step.work_dir)
    parts = builder.sample(
        n_disk=200,
        n_halo=0,
        work_dir=str(baryon_step.work_dir),
        cleanup=False,
        external_halo_path=xhalo if xhalo.is_file() else None,
    )
    print(f"  disk particles: {len(parts.get('disk', []))}")
    if "halo" in parts:
        print(f"  external halo particles: {len(parts['halo'])}")


if __name__ == "__main__":
    main()
