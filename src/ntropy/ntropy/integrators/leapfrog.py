"""Leapfrog (kick-drift-kick) integrator."""

from __future__ import annotations

import numpy as np


def leapfrog_step(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance one leapfrog (velocity Verlet) step.

    The caller supplies the acceleration at the beginning of the step; a
    second force evaluation at the end of the step completes the symmetric
    integrator (see :class:`~ntropy.simulation.Simulation.step`).

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Positions at the start of the step.
    vel : ndarray, shape (N, 3)
        Velocities at the start of the step.
    acc : ndarray, shape (N, 3)
        Accelerations at the start of the step.
    dt : float
        Timestep [GalactICS time units].

    Returns
    -------
    pos_new : ndarray, shape (N, 3)
        Positions after the drift.
    vel_half : ndarray, shape (N, 3)
        Velocities after the first half-kick (before the final kick).
    """
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    return pos_new, vel_half
