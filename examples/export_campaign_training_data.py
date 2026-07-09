#!/usr/bin/env python3
"""Export campaign work trees as ML training bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from galacticsics.ml import export_campaign_training_bundle
from galacticsics.representations.learned import EncoderBackend, LearnedEncoderConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export campaign ML training bundle")
    parser.add_argument(
        "work_root",
        type=Path,
        help="Campaign root (contains manifest.jsonl and per-hash run dirs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: work_root/ml_training)",
    )
    parser.add_argument(
        "--backend",
        choices=[b.value for b in EncoderBackend],
        default=EncoderBackend.MEAN_POOL.value,
        help="Encoder backend for latent export",
    )
    parser.add_argument(
        "--no-encode",
        action="store_true",
        help="Only export features.npz, skip latent computation",
    )
    parser.add_argument(
        "--write-fields",
        action="store_true",
        help="Write binned field.npy (field backend)",
    )
    args = parser.parse_args(argv)

    cfg = LearnedEncoderConfig(backend=EncoderBackend(args.backend))
    bundle = export_campaign_training_bundle(
        args.work_root,
        output_dir=args.output,
        encoder_config=cfg,
        encode=not args.no_encode,
        write_fields=args.write_fields,
    )
    out = args.output or args.work_root / "ml_training"
    print(f"Exported {len(bundle.records)} records → {out / 'training_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
