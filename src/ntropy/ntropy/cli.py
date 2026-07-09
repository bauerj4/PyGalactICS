"""Command-line interface for ntropy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ntropy.analysis.density import bin_spherical_density, compare_density_profiles
from ntropy.config import load_config
from ntropy.ics.composite import CompositeICSpec, sample_composite
from ntropy.ics.disk import ExponentialDiskParams, sample_exponential_disk
from ntropy.ics.nfw import NFWParams, sample_nfw
from ntropy.ics.plummer import PlummerParams, sample_plummer
from ntropy.ics.sersic import SersicParams, sample_sersic
from ntropy.simulation import run_simulation


def main(argv: list[str] | None = None) -> int:
    """Run an N-body simulation from a JSON config file."""
    parser = argparse.ArgumentParser(description="Run ntropy N-body simulation from JSON config")
    parser.add_argument("config", type=Path, help="Path to JSON configuration file")
    parser.add_argument(
        "--report-density",
        action="store_true",
        help="Print initial/final density profile comparison",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = run_simulation(config)

    print(f"Completed {config.integrator.n_steps} steps")
    print(f"Initial energy: {result.energies[0]:.6e}")
    print(f"Final energy:   {result.energies[-1]:.6e}")
    if result.output_dir:
        print(f"Output written to {result.output_dir}")

    if args.report_density:
        n_bins = config.analysis.density_bins
        r_max = config.analysis.r_max
        init_prof = bin_spherical_density(
            result.initial_state.pos,
            result.initial_state.mass,
            n_bins=n_bins,
            r_max=r_max,
        )
        final_prof = bin_spherical_density(
            result.final_state.pos,
            result.final_state.mass,
            n_bins=n_bins,
            r_max=r_max,
        )
        max_rel = compare_density_profiles(init_prof, final_prof)
        print(f"Max relative density change: {max_rel:.4f}")

    return 0


def make_ic_main(argv: list[str] | None = None) -> int:
    """Generate initial condition particle files."""
    parser = argparse.ArgumentParser(description="Generate ntropy equilibrium IC files")
    parser.add_argument(
        "model",
        choices=["nfw", "sersic", "plummer", "disk", "composite"],
        help="Equilibrium model to sample",
    )
    parser.add_argument("output", type=Path, help="Output particle file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-particles", type=int, default=None, help="Particle count")
    args = parser.parse_args(argv)

    if args.model == "nfw":
        params = NFWParams(n_particles=args.n_particles or 256)
        state = sample_nfw(params, seed=args.seed)
    elif args.model == "sersic":
        params = SersicParams(n_particles=args.n_particles or 128)
        state = sample_sersic(params, seed=args.seed)
    elif args.model == "disk":
        params = ExponentialDiskParams(n_particles=args.n_particles or 256)
        state = sample_exponential_disk(params, seed=args.seed)
    elif args.model == "composite":
        state = sample_composite(CompositeICSpec(), seed=args.seed)
    else:
        params = PlummerParams(n_particles=args.n_particles or 128)
        state = sample_plummer(params, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state.write_ascii(args.output)
    meta = {
        "model": args.model,
        "seed": args.seed,
        "n_particles": state.n,
        "eps": float(state.eps[0]),
    }
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {state.n} particles to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
