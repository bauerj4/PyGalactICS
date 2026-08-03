"""CLI for DBH parameter grid campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

from galacticsics.campaign.manifest import load_manifest
from galacticsics.campaign.runner import run_campaign, run_campaign_from_config
from galacticsics.campaign.spec import load_grid_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GalactICS DBH parameter grid campaigns")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Execute a grid or walkthrough campaign JSON")
    run_p.add_argument("spec", type=Path, help="Grid spec or walkthrough JSON path")
    run_p.add_argument("--work-root", type=Path, default=None)
    run_p.add_argument(
        "--stages",
        nargs="+",
        choices=["solve", "sample", "evolve"],
        default=None,
    )
    run_p.add_argument("--end-time-gyr", type=float, default=None)
    run_p.add_argument(
        "--gpu-batch-size",
        type=int,
        default=None,
        help="Concurrent gpu_bh evolve processes (overrides config)",
    )

    status_p = sub.add_parser("status", help="Show campaign manifest summary")
    status_p.add_argument("work_root", type=Path)

    proj_p = sub.add_parser(
        "projections",
        help="Write face-on and side-on component projection PNGs for a campaign",
    )
    proj_p.add_argument(
        "work_root",
        type=Path,
        nargs="?",
        default=Path("runs/mw_morton_corpus"),
        help="Campaign work root (default: runs/mw_morton_corpus)",
    )
    proj_p.add_argument(
        "--snapshots",
        nargs="+",
        choices=["ic", "final", "latest", "steps"],
        default=None,
        help="States to plot (default: ic, steps when dumps exist, final when present)",
    )
    proj_p.add_argument(
        "--components",
        nargs="+",
        choices=["disk", "bulge", "halo"],
        default=None,
        help="Components to plot (default: all present)",
    )
    proj_p.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="When plotting steps, keep every Nth dump (default: 1 = all)",
    )
    proj_p.add_argument("--dpi", type=int, default=120)

    args = parser.parse_args(argv)
    if args.command == "run":
        raw_text = args.spec.read_text()
        use_walkthrough = '"particles"' in raw_text and '"base_model"' in raw_text
        if use_walkthrough:
            if args.end_time_gyr is not None or args.gpu_batch_size is not None or args.stages:
                from galacticsics.campaign.run_config import load_walkthrough_config

                cfg = load_walkthrough_config(args.spec)
                if args.end_time_gyr is not None:
                    cfg.raw.setdefault("evolve", {})["end_time_gyr"] = args.end_time_gyr
                if args.gpu_batch_size is not None:
                    cfg.raw.setdefault("run", {})["gpu_batch_size"] = args.gpu_batch_size
                if args.stages is not None:
                    cfg.raw.setdefault("run", {})["stages"] = list(args.stages)
                sweep = cfg.sweep_grid()
                manifest = run_campaign_from_config(
                    args.spec,
                    work_root=args.work_root,
                    grid_spec=sweep or cfg.base_grid,
                    raw=cfg.raw,
                )
            else:
                manifest = run_campaign_from_config(args.spec, work_root=args.work_root)
        else:
            spec = load_grid_spec(args.spec)
            kwargs: dict = {
                "stages": args.stages or ["solve"],
                "end_time_gyr": args.end_time_gyr if args.end_time_gyr is not None else 1.0,
            }
            if args.gpu_batch_size is not None:
                kwargs["gpu_batch_size"] = args.gpu_batch_size
            manifest = run_campaign(
                spec,
                args.work_root or Path("runs/mw_grid"),
                **kwargs,
            )
        print(f"Campaign {manifest.name}: {len(manifest.rows)} models → {manifest.work_root}")
        return 0

    if args.command == "status":
        manifest = load_manifest(args.work_root)
        print(f"Campaign {manifest.name}: {len(manifest.rows)} entries")
        for row in manifest.rows[-10:]:
            print(f"  {row.get('hash', '?')[:8]}  {row.get('label', '')}")
        return 0

    if args.command == "projections":
        from galacticsics.campaign.analysis import write_campaign_projection_pngs

        written = write_campaign_projection_pngs(
            args.work_root,
            snapshots=args.snapshots,
            components=args.components,
            dpi=args.dpi,
            step_stride=args.step_stride,
        )
        print(f"Wrote {len(written)} PNGs under {args.work_root}/*/projections/")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
