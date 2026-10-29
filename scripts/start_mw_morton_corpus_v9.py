#!/usr/bin/env python3
"""Regenerate priority or full-corpus models with all equilibrium fixes."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from galacticsics.campaign.run_config import load_walkthrough_config
from galacticsics.campaign.runner import run_campaign
from galacticsics.campaign.serialize import model_from_dict, model_to_dict
from galacticsics.campaign.spec import GridSpec, model_hash


DEFAULT_SOURCE_IDS = (
    "b31995c39105",
    "906c4af73543",
    "711af8dcbb02",
    "f6178506a67c",
)


def _surface_normalized_model(path: Path):
    model = model_from_dict(json.loads(path.read_text()))
    if model.disk is None:
        raise ValueError(f"source model has no disk: {path}")
    return replace(
        model,
        disk=replace(model.disk, vertical_density_mode="surface_normalized"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the four priority models or the complete v9 corpus"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("runs/mw_morton_corpus_v2"),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("runs/mw_morton_corpus_v9"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("campaigns/mw_morton_corpus_v9.json"),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--source-ids", nargs="+")
    selection.add_argument(
        "--all-source-models",
        action="store_true",
        help=(
            "regenerate every model.json in --source-root; completed v9 models "
            "are still skipped by the campaign runner"
        ),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("solve", "sample", "evolve"),
        default=("solve", "sample", "evolve"),
    )
    args = parser.parse_args()

    source_ids = list(DEFAULT_SOURCE_IDS if args.source_ids is None else args.source_ids)
    if args.all_source_models:
        source_ids = sorted(path.parent.name for path in args.source_root.glob("*/model.json"))
        if not source_ids:
            parser.error(f"no model.json files found below {args.source_root}")

    models = []
    mapping: dict[str, dict[str, str | float]] = {}
    for source_id in source_ids:
        source_path = args.source_root / source_id / "model.json"
        model = _surface_normalized_model(source_path)
        models.append(model)
        mapping[source_id] = {
            "source_model": str(source_path),
            "v9_id": model_hash(model),
            "vertical_density_mode": "surface_normalized",
            "spheroid_sampling": "final_potential_rho_psi",
            "disk_velocity_envelope_safety": 2.0,
            "disk_velocity_envelope_scan": "recentered_129_plus_zero_cusp",
        }

    args.work_root.mkdir(parents=True, exist_ok=True)
    mapping_path = args.work_root / "source_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")

    spec = GridSpec(
        name=(
            "mw_morton_corpus_v9_full"
            if args.all_source_models
            else "mw_morton_corpus_v9_priority"
        ),
        base="milky_way_disk_halo_bulge",
        grid_mode="list",
        models=[model_to_dict(model) for model in models],
        max_points=len(models),
        coarse_grid=False,
        physics_backend="python",
    )
    config = load_walkthrough_config(args.config)
    kwargs = config.run_campaign_kwargs()
    kwargs["stages"] = list(args.stages)
    run_campaign(spec, args.work_root, **kwargs)
    print(mapping_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
