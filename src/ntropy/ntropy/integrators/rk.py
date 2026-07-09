"""Explicit Runge–Kutta integrators (not symplectic)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

AccelFn = Callable[[np.ndarray], np.ndarray]


def rk2_step(
    pos: np.ndarray,
    vel: np.ndarray,
    accel_fn: AccelFn,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Second-order explicit midpoint (RK2) step.

    Parameters
    ----------
    pos, vel : ndarray, shape (N, 3)
        Phase-space coordinates at the start of the step.
    accel_fn : callable
        ``accel_fn(pos) -> acc`` for the current particle masses/softening.
    dt : float
        Timestep.

    Returns
    -------
    pos_new, vel_new : ndarray, shape (N, 3)
    """
    a1 = accel_fn(pos)
    x_mid = pos + 0.5 * dt * vel
    v_mid = vel + 0.5 * dt * a1
    a_mid = accel_fn(x_mid)
    pos_new = pos + dt * v_mid
    vel_new = vel + dt * a_mid
    return pos_new, vel_new


def rk3_step(
    pos: np.ndarray,
    vel: np.ndarray,
    accel_fn: AccelFn,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Third-order explicit RK step (Kutta's 3-stage method).

    Parameters
    ----------
    pos, vel : ndarray, shape (N, 3)
        Phase-space coordinates at the start of the step.
    accel_fn : callable
        ``accel_fn(pos) -> acc``.
    dt : float
        Timestep.

    Returns
    -------
    pos_new, vel_new : ndarray, shape (N, 3)
    """
    a1 = accel_fn(pos)
    k1_x, k1_v = vel, a1

    x2 = pos + 0.5 * dt * k1_x
    v2 = vel + 0.5 * dt * k1_v
    a2 = accel_fn(x2)
    k2_x, k2_v = v2, a2

    x3 = pos + dt * (-k1_x + 2.0 * k2_x)
    v3 = vel + dt * (-k1_v + 2.0 * k2_v)
    a3 = accel_fn(x3)
    k3_x, k3_v = v3, a3

    pos_new = pos + dt / 6.0 * (k1_x + 4.0 * k2_x + k3_x)
    vel_new = vel + dt / 6.0 * (k1_v + 4.0 * k2_v + k3_v)
    return pos_new, vel_new


def rk4_step(
    pos: np.ndarray,
    vel: np.ndarray,
    accel_fn: AccelFn,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical fourth-order explicit RK step.

    Parameters
    ----------
    pos, vel : ndarray, shape (N, 3)
        Phase-space coordinates at the start of the step.
    accel_fn : callable
        ``accel_fn(pos) -> acc``.
    dt : float
        Timestep.

    Returns
    -------
    pos_new, vel_new : ndarray, shape (N, 3)
    """
    a1 = accel_fn(pos)
    k1_x, k1_v = vel, a1

    x2 = pos + 0.5 * dt * k1_x
    v2 = vel + 0.5 * dt * k1_v
    a2 = accel_fn(x2)
    k2_x, k2_v = v2, a2

    x3 = pos + 0.5 * dt * k2_x
    v3 = vel + 0.5 * dt * k2_v
    a3 = accel_fn(x3)
    k3_x, k3_v = v3, a3

    x4 = pos + dt * k3_x
    v4 = vel + dt * k3_v
    a4 = accel_fn(x4)
    k4_x, k4_v = v4, a4

    pos_new = pos + dt / 6.0 * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    vel_new = vel + dt / 6.0 * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return pos_new, vel_new
