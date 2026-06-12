#!/usr/bin/env python3
"""End-to-end demo: GalactICS halo IC → ntropy stability check."""

from __future__ import annotations

from pathlib import Path

from ntropy.analysis.density import bin_spherical_density, compare_density_profiles
from ntropy.config import ForceConfig, IntegratorConfig, ParallelConfig, RunConfig
from ntropy.integrations.galacticsics import galacticsics_available, sample_galacticsics_halo
from ntropy.simulation import Simulation


def main() -> None:
    if not galacticsics_available():
        raise SystemExit(
            "galacticsics + legacy binaries required. Run: make install-dev"
        )

    out = Path("galacticsics_ntropy_run")
    out.mkdir(exist_ok=True)

    print("1. Solve NFW halo potential (dbh) and sample particles (genhalo)...")
    ic = sample_galacticsics_halo(
        n_particles=128,
        seed=-42,
        work_dir=out / "galacticsics_work",
        eps=0.04,
    )
    ic.state.write_ascii(out / "halo_galacticsics.dat")
    print(f"   N={ic.state.n}, work_dir={ic.work_dir}")

    print("2. Short ntropy evolution (brute force)...")
    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(dt=0.002, n_steps=25)
    cfg.force = ForceConfig(method="brute")
    cfg.parallel = ParallelConfig(enabled=False)
    cfg.output.write_final = False

    r_max = float((ic.state.pos**2).sum(axis=1) ** 0.5).max() * 1.2
    init_prof = bin_spherical_density(ic.state.pos, ic.state.mass, 15, r_max=r_max)
    result = Simulation(cfg, state=ic.state.copy()).run()
    final_prof = bin_spherical_density(
        result.final_state.pos, result.final_state.mass, 15, r_max=r_max
    )
    drift = compare_density_profiles(init_prof, final_prof, min_count=2)
    print(f"   Max density drift: {drift:.3f}")
    print(f"   Wrote {out / 'halo_galacticsics.dat'}")


if __name__ == "__main__":
    main()
