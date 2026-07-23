#!/usr/bin/env python3
"""Benchmark bh_c force kernels on campaign ICs."""

from __future__ import annotations

import argparse
from pathlib import Path

from galacticsics.campaign.benchmarks import (
    format_force_benchmark,
    load_ic_state,
    run_force_benchmark,
    subsample_state,
)
from ntropy.forces.bhtree_c import extension_available


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work_dir", type=Path, help="Campaign model dir with ic_state.npz")
    p.add_argument("--subsample", type=int, default=10_000, help="Particle count cap")
    p.add_argument("--repeat", type=int, default=4)
    args = p.parse_args()

    if not extension_available():
        raise SystemExit("bh_c extension not built; pip install -e src/ntropy")

    ic_path = args.work_dir / "ic_state.npz"
    if not ic_path.is_file():
        raise SystemExit(f"missing {ic_path}")

    state = load_ic_state(ic_path)
    if args.subsample and state.n > args.subsample:
        state = subsample_state(state, args.subsample)
    result = run_force_benchmark(state, n_repeat=args.repeat)
    print(format_force_benchmark(result))


if __name__ == "__main__":
    main()
