"""Per-particle softening length file I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def read_softening_file(path: PathLike) -> np.ndarray:
    """Read one softening length per line (comments with # allowed)."""
    values: list[float] = []
    for line in Path(path).read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        values.append(float(stripped.split()[0]))
    return np.asarray(values, dtype=float)
