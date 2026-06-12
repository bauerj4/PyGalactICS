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


def _parse_floats(line: str) -> list[float]:
    out: list[float] = []
    for tok in line.split():
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def read_particles_ascii(path: PathLike, *, max_particles: int | None = None) -> np.ndarray:
    """Read ASCII N-body particle file (mass, x, y, z, vx, vy, vz).

    Skips an optional first-line header ``nobj flag`` written by gendisk.
    """
    rows: list[list[float]] = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if max_particles is not None and len(rows) >= max_particles:
            break
        vals = _parse_floats(line.split("#", 1)[0])
        if len(vals) >= 7:
            rows.append(vals[:7])
        elif i == 0 and len(vals) == 2:
            continue
    arr = np.zeros(len(rows), dtype=PARTICLE_DTYPE)
    for i, row in enumerate(rows):
        for j, name in enumerate(PARTICLE_DTYPE.names):
            arr[name][i] = row[j]
    return arr


def write_particles_ascii(path: PathLike, data: np.ndarray) -> None:
    """Write particles in GalactICS ASCII format."""
    with open(path, "w") as f:
        for row in data:
            f.write(
                f"{row['mass']:12.5E} {row['x']:12.5E} {row['y']:12.5E} "
                f"{row['z']:12.5E} {row['vx']:12.5E} {row['vy']:12.5E} "
                f"{row['vz']:12.5E}\n"
            )
