#!/usr/bin/env python3
"""Backward-compatible entry point; prefer: python -m ntropy.benchmark.mpi_force_bench."""

from __future__ import annotations

from ntropy.benchmark.mpi_force_bench import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
