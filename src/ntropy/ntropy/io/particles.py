"""GalactICS-compatible ASCII particle I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]

PARTICLE_DTYPE = np.dtype(
    [
        ("mass", "f8"),
        ("x", "f8"),
        ("y", "f8"),
        ("z", "f8"),
        ("vx", "f8"),
        ("vy", "f8"),
        ("vz", "f8"),
    ]
)

PARTICLE_DTYPE_WITH_TYPE = np.dtype(
    list(PARTICLE_DTYPE.descr) + [("type_id", "i4")]
)


def _parse_floats(line: str) -> list[float]:
    out: list[float] = []
    for tok in line.split():
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def read_particles_ascii(path: PathLike, *, max_particles: int | None = None) -> np.ndarray:
    """Read ASCII N-body particle file.

    Standard format: ``mass x y z vx vy vz`` (7 columns).
    Extended format: optional 8th column ``type_id`` (integer).
    Skips an optional first-line header ``nobj flag`` written by gendisk.
    """
    rows: list[list[float]] = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if max_particles is not None and len(rows) >= max_particles:
            break
        vals = _parse_floats(line.split("#", 1)[0])
        if len(vals) >= 8:
            rows.append(vals[:8])
        elif len(vals) >= 7:
            rows.append(vals[:7])
        elif i == 0 and len(vals) == 2:
            continue

    has_type = any(len(row) >= 8 for row in rows)
    dtype = PARTICLE_DTYPE_WITH_TYPE if has_type else PARTICLE_DTYPE
    arr = np.zeros(len(rows), dtype=dtype)
    for i, row in enumerate(rows):
        for j, name in enumerate(PARTICLE_DTYPE.names):
            arr[name][i] = row[j]
        if has_type and len(row) >= 8:
            arr["type_id"][i] = int(row[7])
    return arr


def write_particles_ascii(path: PathLike, data: np.ndarray) -> None:
    """Write particles in GalactICS ASCII format (with optional type_id column)."""
    has_type = "type_id" in data.dtype.names
    with open(path, "w") as f:
        for row in data:
            line = (
                f"{row['mass']:12.5E} {row['x']:12.5E} {row['y']:12.5E} "
                f"{row['z']:12.5E} {row['vx']:12.5E} {row['vy']:12.5E} "
                f"{row['vz']:12.5E}"
            )
            if has_type:
                line += f" {int(row['type_id'])}"
            f.write(line + "\n")


def read_type_ids(path: PathLike) -> np.ndarray:
    """Read one integer type id per line."""
    values: list[int] = []
    for line in Path(path).read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            values.append(int(stripped))
    return np.asarray(values, dtype=np.int32)


def write_type_ids(path: PathLike, type_ids: np.ndarray) -> None:
    """Write one integer type id per line."""
    with open(path, "w") as f:
        for tid in type_ids:
            f.write(f"{int(tid)}\n")
