import numpy as np
from numba import njit


@njit
def simpson_harm_int(nx, dx, density_array):
    """
    Compute the gravitational potential and force using Simpson's rule for a given density array.

    Parameters
    ----------
    nx : int
        Number of grid points.
    dx : float
        Grid spacing.
    density_array : np.ndarray
        Density array of shape (nx + 1).

    Returns
    -------
    potential_array : np.ndarray
        Potential array of shape (nx + 1).
    force_array : np.ndarray
        Force array of shape (nx + 1).
    """
    potential_array = np.zeros(nx + 1)
    force_array = np.zeros(nx + 1)

    integral_s1 = np.zeros(nx + 1)
    integral_s2 = np.zeros(nx + 1)

    integral_s1[0] = 0.0
    radial_distance = 2.0 * dx
    integral_s1[2] = (radial_distance * dx / 3.0) * (
        4 * density_array[1] * (1.0 - dx / radial_distance) ** 2.0 + density_array[2]
    )
    previous_radial_distance = radial_distance

    for i in range(4, nx + 1, 2):
        radial_distance = i * dx
        integral_s1_part = (radial_distance * dx / 3.0) * (
            density_array[i - 2] * (1.0 - 2 * dx / radial_distance) ** 2.0
            + 4 * density_array[i - 1] * (1.0 - dx / radial_distance) ** 2.0
            + density_array[i]
        )
        integral_s1[i] = (
            integral_s1_part
            + integral_s1[i - 2] * previous_radial_distance / radial_distance
        )
        previous_radial_distance = radial_distance

    integral_s2[nx] = 0.0
    previous_radial_distance = nx * dx

    for i in range(nx - 2, 1, -2):
        radial_distance = i * dx
        integral_s2_part = (radial_distance * dx / 3.0) * (
            density_array[i + 2] * (1.0 + 2 * dx / radial_distance)
            + 4 * density_array[i + 1] * (1.0 + dx / radial_distance)
            + density_array[i]
        )
        integral_s2[i] = integral_s2_part + integral_s2[i + 2]
        previous_radial_distance = radial_distance

    for i in range(2, nx + 1, 2):
        radial_distance = i * dx
        potential_array[i] = (4.0 * np.pi) * (integral_s1[i] + integral_s2[i])
        force_array[i] = -(4.0 * np.pi) * integral_s1[i] / radial_distance

    potential_array[0] = (
        3 * (potential_array[2] - potential_array[4]) + potential_array[6]
    )
    force_array[0] = 0.0

    for i in range(1, nx, 2):
        potential_array[i] = (potential_array[i - 1] + potential_array[i + 1]) / 2.0
        force_array[i] = (force_array[i - 1] + force_array[i + 1]) / 2.0

    return potential_array, force_array


@njit
def simpson_harm_int_high_l(nx, nl, dx, previous_l_max, fraction, density_array, max_l):
    """
    Compute the gravitational potential and forces for higher harmonics using Simpson's rule.

    Parameters
    ----------
    nx : int
        Number of grid points.
    nl : int
        Number of harmonic levels.
    dx : float
        Grid spacing.
    previous_l_max : int
        Highest harmonic level from the previous iteration.
    fraction : float
        Fraction for damping oscillations between iterations.
    density_array : np.ndarray
        Density array of shape (max_l + 1, nx + 1).
    max_l : int
        Maximum harmonic level.

    Returns
    -------
    potential_array : np.ndarray
        Potential array of shape (max_l + 1, nx + 1).
    force_array : np.ndarray
        Force array of shape (max_l + 1, nx + 1).
    second_order_force_array : np.ndarray
        Second-order force array of shape (max_l + 1, nx + 1).
    """
    potential_array = np.zeros((max_l + 1, nx + 1))
    force_array = np.zeros((max_l + 1, nx + 1))
    second_order_force_array = np.zeros((max_l + 1, nx + 1))

    for l in range(0, nl + 1, 2):
        integral_s1 = np.zeros(nx + 1)
        integral_s2 = np.zeros(nx + 1)

        integral_s1[0] = 0.0
        radial_distance = 2.0 * dx
        integral_s1[2] = (radial_distance * dx / 3.0) * (
            4 * density_array[l // 2 + 1, 1] * (1.0 - dx / radial_distance) ** 2.0
            + density_array[l // 2 + 1, 2]
        )
        previous_radial_distance = radial_distance

        for i in range(4, nx + 1, 2):
            radial_distance = i * dx
            integral_s1_part = (radial_distance * dx / 3.0) * (
                density_array[l // 2 + 1, i - 2]
                * (1.0 - 2 * dx / radial_distance) ** (l + 2)
                + 4
                * density_array[l // 2 + 1, i - 1]
                * (1.0 - dx / radial_distance) ** (l + 2)
                + density_array[l // 2 + 1, i]
            )
            integral_s1[i] = integral_s1_part + integral_s1[i - 2] * (
                previous_radial_distance / radial_distance
            ) ** (l + 1)
            previous_radial_distance = radial_distance

        integral_s2[nx] = 0.0
        previous_radial_distance = nx * dx

        for i in range(nx - 2, 1, -2):
            radial_distance = i * dx
            integral_s2_part = (radial_distance * dx / 3.0) * (
                density_array[l // 2 + 1, i + 2]
                * (1.0 + 2 * dx / radial_distance) ** (1 - l)
                + 4
                * density_array[l // 2 + 1, i + 1]
                * (1.0 + dx / radial_distance) ** (1 - l)
                + density_array[l // 2 + 1, i]
            )
            integral_s2[i] = (
                integral_s2_part
                + integral_s2[i + 2] * (radial_distance / previous_radial_distance) ** l
            )
            previous_radial_distance = radial_distance

        for i in range(2, nx + 1, 2):
            if l <= previous_l_max:
                potential_array[l // 2 + 1, i] = fraction * potential_array[
                    l // 2 + 1, i
                ] + (1.0 - fraction) * (4.0 * np.pi) / (2.0 * l + 1.0) * (
                    integral_s1[i] + integral_s2[i]
                )
            else:
                potential_array[l // 2 + 1, i] = (
                    (4.0 * np.pi) / (2.0 * l + 1.0) * (integral_s1[i] + integral_s2[i])
                )

        for i in range(2, nx + 1, 2):
            radial_distance = i * dx
            force_array[l // 2 + 1, i] = (
                -4
                * np.pi
                / (2.0 * l + 1.0)
                * (-(l + 1) * integral_s1[i] + l * integral_s2[i])
                / radial_distance
            )
            second_order_force_array[l // 2 + 1, i] = (
                -4
                * np.pi
                / (2.0 * l + 1.0)
                * (
                    (l + 1) * (l + 2) * integral_s1[i] / radial_distance**2
                    + l * (l - 1) * integral_s2[i] / radial_distance**2
                    - (2 * l + 1) * density_array[l // 2 + 1, i]
                )
            )

    potential_array[1, 0] = (
        3 * (potential_array[1, 2] - potential_array[1, 4]) + potential_array[1, 6]
    )
    force_array[1, 0] = 0.0
    second_order_force_array[1, 0] = (
        2 * second_order_force_array[1, 2] - second_order_force_array[1, 4]
    )

    for l in range(2, nl + 1, 2):
        potential_array[l // 2 + 1, 0] = 0.0
        force_array[l // 2 + 1, 0] = 0.0
        second_order_force_array[l // 2 + 1, 0] = 0.0

    for i in range(1, nx, 2):
        for l in range(0, nl + 1, 2):
            potential_array[l // 2 + 1, i] = (
                potential_array[l // 2 + 1, i - 1] + potential_array[l // 2 + 1, i + 1]
            ) / 2.0
            force_array[l // 2 + 1, i] = (
                force_array[l // 2 + 1, i - 1] + force_array[l // 2 + 1, i + 1]
            ) / 2.0
            second_order_force_array[l // 2 + 1, i] = (
                second_order_force_array[l // 2 + 1, i - 1]
                + second_order_force_array[l // 2 + 1, i + 1]
            ) / 2.0

    return potential_array, force_array, second_order_force_array
