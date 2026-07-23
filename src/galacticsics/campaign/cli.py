"""CLI for DBH parameter grid campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

from galacticsics.campaign.manifest import load_manifest
from galacticsics.campaign.runner import run_campaign
from galacticsics.campaign.spec import load_grid_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GalactICS DBH parameter grid campaigns")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Execute a grid spec")
    run_p.add_argument("spec", type=Path, help="Grid spec JSON/YAML path")
    run_p.add_argument("--work-root", type=Path, default=Path("runs/mw_grid"))
    run_p.add_argument(
        "--stages",
        nargs="+",
        choices=["solve", "sample", "evolve"],
        default=["solve"],
    )
    run_p.add_argument("--end-time-gyr", type=float, default=1.0)

    status_p = sub.add_parser("status", help="Show campaign manifest summary")
    status_p.add_argument("work_root", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        spec = load_grid_spec(args.spec)
        manifest = run_campaign(
            spec,
            args.work_root,
            stages=args.stages,
            end_time_gyr=args.end_time_gyr,
        )
        print(f"Campaign {manifest.name}: {len(manifest.rows)} models → {manifest.work_root}")
        return 0

    if args.command == "status":
        manifest = load_manifest(args.work_root)
        print(f"Campaign {manifest.name}: {len(manifest.rows)} entries")
        for row in manifest.rows[-10:]:
            print(f"  {row.get('hash', '?')[:8]}  {row.get('label', '')}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
