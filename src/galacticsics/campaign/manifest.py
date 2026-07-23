"""Campaign manifest I/O."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CampaignManifest:
    """Aggregate campaign metadata."""

    name: str
    work_root: Path
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def csv_path(self) -> Path:
        return self.work_root / "campaign_index.csv"

    @property
    def jsonl_path(self) -> Path:
        return self.work_root / "manifest.jsonl"


def append_manifest_row(manifest: CampaignManifest, row: dict[str, Any]) -> None:
    """Append one row to manifest files."""
    manifest.work_root.mkdir(parents=True, exist_ok=True)
    manifest.rows.append(row)
    with open(manifest.jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")

    fieldnames = sorted(row.keys())
    write_header = not manifest.csv_path.exists()
    with open(manifest.csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_manifest(work_root: Path | str) -> CampaignManifest:
    work_root = Path(work_root)
    rows: list[dict[str, Any]] = []
    jsonl = work_root / "manifest.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    name = work_root.name
    return CampaignManifest(name=name, work_root=work_root, rows=rows)
