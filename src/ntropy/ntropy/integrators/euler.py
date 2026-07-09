"""Explicit forward Euler integrator (not symplectic)."""

from __future__ import annotations

import numpy as np


def euler_step(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance one forward-Euler step for :math:`\\ddot{\\mathbf{x}} = \\mathbf{a}(\\mathbf{x})`.

    .. math::

        \\mathbf{x}_{n+1} = \\mathbf{x}_n + \\Delta t\\,\\mathbf{v}_n, \\quad
        \\mathbf{v}_{n+1} = \\mathbf{v}_n + \\Delta t\\,\\mathbf{a}(\\mathbf{x}_n)

    This explicit scheme is **not symplectic** and generally does not conserve
    the Hamiltonian of a gravitating N-body system.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Positions at the start of the step.
    vel : ndarray, shape (N, 3)
        Velocities at the start of the step.
    acc : ndarray, shape (N, 3)
        Accelerations at the start of the step.
    dt : float
        Timestep.

    Returns
    -------
    pos_new, vel_new : ndarray, shape (N, 3)
        Updated positions and velocities.
    """
    pos_new = pos + dt * vel
    vel_new = vel + dt * acc
    return pos_new, vel_new
