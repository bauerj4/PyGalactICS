"""Command-line interface for artifact generation."""

from __future__ import annotations

import argparse
import sys

from galacticsics.artifacts.generate import generate_reference_artifacts
from galacticsics.artifacts.paths import default_artifact_dir
from galacticsics.artifacts.verify import verify_artifact_consistency


def main(argv: list[str] | None = None) -> int:
    """
    Generate or verify reference artifacts.

    Examples
    --------
    ::

        galacticsics-generate-artifacts generate
        galacticsics-generate-artifacts verify
    """
    parser = argparse.ArgumentParser(description="GalactICS reference artifact tools")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Run dbh + diskdf + sampling")
    gen.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output directory (default: {default_artifact_dir()})",
    )
    gen.add_argument("--force", action="store_true", help="Regenerate existing artifacts")
    gen.add_argument("--no-verify", action="store_true", help="Skip consistency checks")

    ver = sub.add_parser("verify", help="Check artifact consistency")
    ver.add_argument(
        "--dir",
        type=str,
        default=None,
        help=f"Artifact directory (default: {default_artifact_dir()})",
    )

    args = parser.parse_args(argv)

    if args.command == "generate":
        out = generate_reference_artifacts(
            args.output,
            verify=not args.no_verify,
            force=args.force,
        )
        print(f"Artifacts written to {out}")
        return 0

    if args.command == "verify":
        report = verify_artifact_consistency(args.dir or default_artifact_dir())
        print(report.summary())
        return 0 if report.ok else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
