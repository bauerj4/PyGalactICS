#!/usr/bin/env python3
"""Generate all reference IC files for ntropy examples."""

from __future__ import annotations

from pathlib import Path

from ntropy.ics.composite import CompositeICSpec, sample_composite
from ntropy.ics.disk import sample_exponential_disk
from ntropy.ics.nfw import sample_nfw
from ntropy.ics.plummer import sample_plummer
from ntropy.ics.sersic import sample_sersic

DATA = Path(__file__).resolve().parent / "data"
SEED = 42


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    sample_plummer(seed=SEED).write_ascii(DATA / "plummer_128.dat")
    sample_sersic(seed=SEED).write_ascii(DATA / "sersic_128.dat")
    sample_nfw(seed=SEED).write_ascii(DATA / "nfw_256.dat")
    sample_exponential_disk(seed=SEED).write_ascii(DATA / "disk_256.dat")
    sample_composite(CompositeICSpec(), seed=SEED).write_ascii(DATA / "composite.dat")
    print(f"Wrote IC files to {DATA}")


if __name__ == "__main__":
    main()
