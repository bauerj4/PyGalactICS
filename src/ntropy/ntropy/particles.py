"""Particle state container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from ntropy.config import RunConfig
from ntropy.io.particles import read_particles_ascii, write_particles_ascii
from ntropy.io.softening import read_softening_file

PathLike = Union[str, Path]


@dataclass
class ParticleState:
    """
    N-body particle state with positions, velocities, masses, and softening.

    Attributes
    ----------
    pos : ndarray, shape (N, 3)
        Cartesian positions [kpc].
    vel : ndarray, shape (N, 3)
        Cartesian velocities [100 km/s].
    mass : ndarray, shape (N,)
        Particle masses [GalactICS mass units].
    eps : ndarray, shape (N,)
        Per-particle gravitational softening lengths [kpc].
    tags : ndarray of str, optional
        Component labels (e.g. ``'halo'``, ``'bulge'``, ``'disk'``).
    """

    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    eps: np.ndarray
    tags: np.ndarray | None = None

    @property
    def n(self) -> int:
        """Number of particles."""
        return len(self.mass)

    @classmethod
    def from_config(cls, config: RunConfig) -> ParticleState:
        """
        Load particles and softening from a JSON run configuration.

        Parameters
        ----------
        config : RunConfig
            Parsed run configuration with particle file path and softening options.

        Returns
        -------
        ParticleState
            Loaded particle state.

        Raises
        ------
        ValueError
            If per-particle softening file length does not match particle count.
        """
        path = config.resolve_path(config.particles.file)
        data = read_particles_ascii(path)
        n = len(data)
        pos = np.column_stack([data["x"], data["y"], data["z"]])
        vel = np.column_stack([data["vx"], data["vy"], data["vz"]])
        mass = data["mass"].copy()

        eps = np.full(n, config.softening.default, dtype=float)
        if config.softening.per_particle and config.softening.file:
            eps_path = config.resolve_path(config.softening.file)
            eps_file = read_softening_file(eps_path)
            if len(eps_file) != n:
                raise ValueError(
                    f"Softening file has {len(eps_file)} entries but {n} particles"
                )
            eps = eps_file

        return cls(pos=pos, vel=vel, mass=mass, eps=eps, tags=None)

    @classmethod
    def from_arrays(
        cls,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        eps: float | np.ndarray,
    ) -> ParticleState:
        """
        Construct a particle state from NumPy arrays.

        Parameters
        ----------
        pos : array_like, shape (N, 3)
            Positions [kpc].
        vel : array_like, shape (N, 3)
            Velocities [100 km/s].
        mass : array_like, shape (N,)
            Masses.
        eps : float or array_like, shape (N,)
            Softening length(s).

        Returns
        -------
        ParticleState
        """
        n = len(mass)
        if isinstance(eps, (int, float)):
            eps_arr = np.full(n, float(eps))
        else:
            eps_arr = np.asarray(eps, dtype=float)
        return cls(
            pos=np.asarray(pos, dtype=float),
            vel=np.asarray(vel, dtype=float),
            mass=np.asarray(mass, dtype=float),
            eps=eps_arr,
            tags=None,
        )

    def remove_center_of_mass(self) -> None:
        """
        Shift to the center-of-mass frame in position and velocity.

        Modifies ``pos`` and ``vel`` in place.
        """
        total_mass = self.mass.sum()
        com = (self.mass[:, None] * self.pos).sum(axis=0) / total_mass
        vcom = (self.mass[:, None] * self.vel).sum(axis=0) / total_mass
        self.pos -= com
        self.vel -= vcom

    def copy(self) -> ParticleState:
        """Return a deep copy of this state."""
        return ParticleState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            mass=self.mass.copy(),
            eps=self.eps.copy(),
            tags=None if self.tags is None else self.tags.copy(),
        )

    def to_structured_array(self) -> np.ndarray:
        """
        Convert to GalactICS ASCII structured dtype.

        Returns
        -------
        ndarray
            Structured array with fields ``mass, x, y, z, vx, vy, vz``.
        """
        from ntropy.io.particles import PARTICLE_DTYPE

        arr = np.zeros(self.n, dtype=PARTICLE_DTYPE)
        arr["mass"] = self.mass
        arr["x"], arr["y"], arr["z"] = self.pos.T
        arr["vx"], arr["vy"], arr["vz"] = self.vel.T
        return arr

    def write_ascii(self, path: PathLike) -> None:
        """
        Write particles in GalactICS ASCII format.

        Parameters
        ----------
        path : path-like
            Output file path.
        """
        write_particles_ascii(path, self.to_structured_array())

    def reorder(self, indices: np.ndarray) -> None:
        """
        Reorder particles in place.

        Parameters
        ----------
        indices : ndarray, shape (N,)
            Permutation indices.
        """
        self.pos = self.pos[indices]
        self.vel = self.vel[indices]
        self.mass = self.mass[indices]
        self.eps = self.eps[indices]
        if self.tags is not None:
            self.tags = self.tags[indices]

    def mask(self, tag: str) -> ParticleState:
        """
        Return particles belonging to one component tag.

        Parameters
        ----------
        tag : str
            Component label to select.

        Returns
        -------
        ParticleState
            Filtered subset (tags preserved).
        """
        if self.tags is None:
            raise ValueError("ParticleState has no component tags")
        sel = self.tags == tag
        return ParticleState(
            pos=self.pos[sel],
            vel=self.vel[sel],
            mass=self.mass[sel],
            eps=self.eps[sel],
            tags=self.tags[sel],
        )
