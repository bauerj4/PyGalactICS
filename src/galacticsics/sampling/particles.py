"""Particle data structures."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class ParticleSet:
    """Phase-space particles for one component."""

    data: np.ndarray
    component: str = "unknown"

    @classmethod
    def from_ascii(
        cls, path: PathLike, component: str = "unknown", *, max_particles: int | None = None
    ) -> ParticleSet:
        from galacticsics.io.formats import read_particles_ascii

        return cls(read_particles_ascii(path, max_particles=max_particles), component=component)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.data["mass"]))

    @property
    def center_of_mass(self) -> np.ndarray:
        m = self.data["mass"]
        pos = np.column_stack([self.data["x"], self.data["y"], self.data["z"]])
        return (m[:, None] * pos).sum(axis=0) / m.sum()

    def velocity_dispersion(self) -> np.ndarray:
        m = self.data["mass"]
        vel = np.column_stack([self.data["vx"], self.data["vy"], self.data["vz"]])
        vcm = (m[:, None] * vel).sum(axis=0) / m.sum()
        dv = vel - vcm
        return np.sqrt((m[:, None] * dv**2).sum(axis=0) / m.sum())

    def write_ascii(self, path: PathLike) -> None:
        with open(path, "w") as f:
            for row in self.data:
                f.write(
                    f"{row['mass']:12.5E} {row['x']:12.5E} {row['y']:12.5E} "
                    f"{row['z']:12.5E} {row['vx']:12.5E} {row['vy']:12.5E} "
                    f"{row['vz']:12.5E}\n"
                )
