"""CLI: build snapshot index for Morton generative training."""

from __future__ import annotations

import argparse
from pathlib import Path

from galacticsics.ml.morton.index import write_snapshot_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index campaign snapshots for on-the-fly Morton sequence training"
    )
    parser.add_argument("work_root", type=Path, help="Campaign work root with model dirs")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Manifest path (default: <work_root>/snapshot_manifest.json)",
    )
    args = parser.parse_args(argv)
    out = write_snapshot_manifest(args.work_root, args.output)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
