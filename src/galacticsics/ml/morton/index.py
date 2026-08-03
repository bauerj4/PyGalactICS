"""Build snapshot index manifests over campaign work roots (no sequence cache)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SnapshotRecord:
    """
    One indexed full-particle snapshot for on-the-fly training.

    Attributes
    ----------
    run_hash : str
        Campaign model directory name (content hash / id).
    label : str
        Human-readable label from ``model.json`` when present.
    path : str
        Absolute path to ``ic_state.npz`` or ``step_*.npz``.
    t_gyr : float
        Snapshot time [Gyr] (``0`` for ICs; NaN if unknown).
    split : str
        Deterministic ``train`` / ``val`` / ``test`` assignment from ``run_hash``.
    theta : dict of str to float
        Flattened structural parameters (e.g. ``disk.mass``) plus optional
        ``t_gyr`` for dumps.
    source : {'ic', 'dump'}
        Whether the file is an initial condition or an evolution dump.
    """

    run_hash: str
    label: str
    path: str
    t_gyr: float
    split: str
    theta: dict[str, float] = field(default_factory=dict)
    source: str = "dump"  # "ic" | "dump"


def _theta_from_model_json(model_path: Path) -> dict[str, float]:
    if not model_path.is_file():
        return {}
    raw = json.loads(model_path.read_text())
    theta: dict[str, float] = {}
    for comp in ("halo", "disk", "bulge", "disk_kinematics"):
        block = raw.get(comp) or {}
        if not isinstance(block, dict):
            continue
        for key, val in block.items():
            if isinstance(val, (int, float)) and key != "enabled":
                theta[f"{comp}.{key}"] = float(val)
    return theta


def _split_for_hash(run_hash: str, *, train: float = 0.8, val: float = 0.1) -> str:
    digest = hashlib.sha256(run_hash.encode()).hexdigest()
    u = int(digest[:8], 16) / 0xFFFFFFFF
    if u < train:
        return "train"
    if u < train + val:
        return "val"
    return "test"


def _t_gyr_for_step(run_dir: Path, step: int) -> float:
    csv_path = run_dir / "evolution" / "diagnostics.csv"
    if not csv_path.is_file():
        return float("nan")
    try:
        import csv

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if int(float(row.get("step", -1))) == step and "t_gyr" in row:
                return float(row["t_gyr"])
        if rows and "t_gyr" in rows[-1]:
            # Fallback: interpolate by step index
            steps = np.array([int(float(r["step"])) for r in rows if r.get("step")], dtype=float)
            times = np.array([float(r["t_gyr"]) for r in rows if r.get("t_gyr")], dtype=float)
            if steps.size:
                return float(np.interp(step, steps, times))
    except (OSError, ValueError, KeyError):
        pass
    return float("nan")


def resolve_snapshot_t_gyr(path: str | Path, t_gyr: float | None = None) -> float:
    """
    Resolve dump time in Gyr, recovering from ``diagnostics.csv`` when needed.

    Older manifests stored ``nan`` for dumps; this re-derives ``t_gyr`` from the
    step index and the run's diagnostics so ``θ`` conditioning stays time-aware.
    """
    if t_gyr is not None and np.isfinite(float(t_gyr)):
        return float(t_gyr)
    path = Path(path)
    stem = path.stem
    if stem.startswith("step_"):
        try:
            step = int(stem.split("_")[1])
        except (IndexError, ValueError):
            return 0.0
        # .../run_hash/evolution/particles/step_XXXX.npz
        run_dir = path.parent.parent.parent
        recovered = _t_gyr_for_step(run_dir, step)
        if np.isfinite(recovered):
            return float(recovered)
    if stem == "ic_state" or path.name == "ic_state.npz":
        return 0.0
    return 0.0


def build_snapshot_index(work_root: Path | str) -> list[SnapshotRecord]:
    """
    Index ``ic_state.npz`` and ``evolution/particles/step_*.npz`` under ``work_root``.

    Walks each model subdirectory, extracts ``θ`` from ``model.json``, assigns a
    hash-based train/val/test split, and records dump times from
    ``evolution/diagnostics.csv`` when available.

    Parameters
    ----------
    work_root : path-like
        Campaign work root containing per-model directories.

    Returns
    -------
    records : list of SnapshotRecord
        Flat list of IC + dump entries (no token arrays on disk).

    Notes
    -----
    Does **not** write Morton sequences — training draws subsets on the fly so
    the evolve campaign can keep writing dumps without a second corpus cache.
    """
    work_root = Path(work_root)
    records: list[SnapshotRecord] = []
    for model_dir in sorted(p for p in work_root.iterdir() if p.is_dir()):
        model_json = model_dir / "model.json"
        if not model_json.is_file() and not (model_dir / "ic_state.npz").is_file():
            continue
        run_hash = model_dir.name
        label = run_hash
        if model_json.is_file():
            try:
                label = json.loads(model_json.read_text()).get("label", run_hash)
            except json.JSONDecodeError:
                pass
        # Prefer label from campaign manifest if present later; keep hash as id.
        theta = _theta_from_model_json(model_json)
        split = _split_for_hash(run_hash)

        ic = model_dir / "ic_state.npz"
        if ic.is_file():
            records.append(
                SnapshotRecord(
                    run_hash=run_hash,
                    label=str(label),
                    path=str(ic.resolve()),
                    t_gyr=0.0,
                    split=split,
                    theta=dict(theta),
                    source="ic",
                )
            )

        part_dir = model_dir / "evolution" / "particles"
        if part_dir.is_dir():
            for dump in sorted(part_dir.glob("step_*.npz")):
                try:
                    step = int(dump.stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
                t_gyr = _t_gyr_for_step(model_dir, step)
                th = dict(theta)
                th["t_gyr"] = float(t_gyr) if np.isfinite(t_gyr) else float("nan")
                records.append(
                    SnapshotRecord(
                        run_hash=run_hash,
                        label=str(label),
                        path=str(dump.resolve()),
                        t_gyr=float(t_gyr) if np.isfinite(t_gyr) else float("nan"),
                        split=split,
                        theta=th,
                        source="dump",
                    )
                )
    return records


def write_snapshot_manifest(
    work_root: Path | str,
    output_path: Path | str | None = None,
) -> Path:
    """
    Build the snapshot index and write ``snapshot_manifest.json``.

    Parameters
    ----------
    work_root : path-like
        Campaign work root to index.
    output_path : path-like, optional
        Destination JSON path.  Defaults to
        ``work_root / snapshot_manifest.json``.

    Returns
    -------
    path : Path
        Written manifest path.
    """
    work_root = Path(work_root)
    records = build_snapshot_index(work_root)
    out = Path(output_path) if output_path else work_root / "snapshot_manifest.json"
    payload: dict[str, Any] = {
        "work_root": str(work_root.resolve()),
        "n_snapshots": len(records),
        "note": "Training draws random Morton subsets on the fly; no sequence cache.",
        "records": [asdict(r) for r in records],
    }
    out.write_text(json.dumps(payload, indent=2))
    return out
