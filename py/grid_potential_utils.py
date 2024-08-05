import numpy as np
from scipy.special import lpmn

import numpy as np


def pot_from_grid(
    s: float = 0.0,
    z: float = 0.0,
    pot_grid: np.ndarray = None,
    radial_step: float = 1.0,
    lmax: int = 10,
) -> float:
    """
    Calculate the potential from the grid at a specific point (s, z).

    Parameters
    ----------
    s : float, optional
        The s-coordinate. Default is 0.0.
    z : float, optional
        The z-coordinate. Default is 0.0.
    pot_grid : np.ndarray, optional
        The grid of potential values. Shape should be (lmax//2 + 1, nr). Default is None.
    radial_step : float, optional
        The radial step size between grid points. Default is 1.0.
    lmax : int, optional
        The maximum Legendre polynomial order to use in interpolation. Default is 10.

    Returns
    -------
    float
        The calculated potential at (s, z).

    Notes
    -----
    If `radial_distance` is zero, the potential is taken from the grid's origin value.
    """
    if pot_grid is None:
        raise ValueError("pot_grid must be provided")

    radial_distance = np.sqrt(s**2 + z**2)

    if radial_distance == 0:
        return pot_grid[0, 0] / np.sqrt(4 * np.pi)

    index_high = int(radial_distance / radial_step) + 1
    index_high = min(max(index_high, 1), pot_grid.shape[1])

    radial_low = radial_step * (index_high - 1)
    radial_high = radial_step * index_high
    t = (radial_distance - radial_low) / (radial_high - radial_low)
    t_minus_1 = 1.0 - t

    costheta = z / radial_distance if radial_distance != 0 else 0
    potential = 0.0

    for l in range(lmax, -1, -2):
        legendre_value = lpmn(l // 2, l // 2, costheta)[0][-1, l // 2]
        potential += legendre_value * (
            t * pot_grid[l // 2 + 1, index_high - 1]
            + t_minus_1 * pot_grid[l // 2 + 1, index_high - 2]
        )

    return potential


def fr2_from_grid(
    s: float = 0.0,
    z: float = 0.0,
    fr2_grid: np.ndarray = None,
    radial_step: float = 1.0,
    lmax: int = 10,
) -> float:
    """
    Calculate the second radial derivative of the potential from the grid at a specific point (s, z).

    Parameters
    ----------
    s : float, optional
        The s-coordinate. Default is 0.0.
    z : float, optional
        The z-coordinate. Default is 0.0.
    fr2_grid : np.ndarray, optional
        The grid of second radial derivative values. Shape should be (lmax//2 + 1, nr). Default is None.
    radial_step : float, optional
        The radial step size between grid points. Default is 1.0.
    lmax : int, optional
        The maximum Legendre polynomial order to use in interpolation. Default is 10.

    Returns
    -------
    float
        The calculated second radial derivative at (s, z).

    Notes
    -----
    If `radial_distance` is zero, the second radial derivative is taken from the grid's origin value.
    """
    if fr2_grid is None:
        raise ValueError("fr2_grid must be provided")

    radial_distance = np.sqrt(s**2 + z**2)

    if radial_distance == 0:
        return fr2_grid[0, 0] / np.sqrt(4 * np.pi)

    index_high = int(radial_distance / radial_step) + 1
    index_high = min(max(index_high, 1), fr2_grid.shape[1])

    radial_low = radial_step * (index_high - 1)
    radial_high = radial_step * index_high
    t = (radial_distance - radial_low) / (radial_high - radial_low)
    t_minus_1 = 1.0 - t

    costheta = z / radial_distance if radial_distance != 0 else 0
    fr2_value = 0.0

    for l in range(lmax, -1, -2):
        legendre_value = lpmn(l // 2, l // 2, costheta)[0][-1, l // 2]
        fr2_value += legendre_value * (
            t * fr2_grid[l // 2 + 1, index_high - 1]
            + t_minus_1 * fr2_grid[l // 2 + 1, index_high - 2]
        )

    return fr2_value
