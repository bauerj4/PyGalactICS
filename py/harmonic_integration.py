import numpy as np
from numba import njit


@njit
def simpson_harm_int(nx, dx, densArr):
    """
    Compute the gravitational potential and force using Simpson's rule for a given density array.

    Parameters
    ----------
    nx : int
        Number of grid points.
    dx : float
        Grid spacing.
    densArr : np.ndarray
        Density array of shape (nx + 1).

    Returns
    -------
    PotArr : np.ndarray
        Potential array of shape (nx + 1).
    ForArr : np.ndarray
        Force array of shape (nx + 1).
    """
    PotArr = np.zeros(nx + 1)
    ForArr = np.zeros(nx + 1)

    s1 = np.zeros(nx + 1)
    s2 = np.zeros(nx + 1)

    s1[0] = 0.0
    x = 2.0 * dx
    s1[2] = (x * dx / 3.0) * (4 * densArr[1] * (1.0 - dx / x) ** 2.0 + densArr[2])
    xold = x

    for i in range(4, nx + 1, 2):
        x = i * dx
        s1a = (x * dx / 3.0) * (
            densArr[i - 2] * (1.0 - 2 * dx / x) ** 2.0
            + 4 * densArr[i - 1] * (1.0 - dx / x) ** 2.0
            + densArr[i]
        )
        s1[i] = s1a + s1[i - 2] * xold / x
        xold = x

    s2[nx] = 0.0
    xold = nx * dx

    for i in range(nx - 2, 1, -2):
        x = i * dx
        s2a = (x * dx / 3.0) * (
            densArr[i + 2] * (1.0 + 2 * dx / x)
            + 4 * densArr[i + 1] * (1.0 + dx / x)
            + densArr[i]
        )
        s2[i] = s2a + s2[i + 2]
        xold = x

    for i in range(2, nx + 1, 2):
        x = i * dx
        PotArr[i] = (4.0 * np.pi) * (s1[i] + s2[i])
        ForArr[i] = -(4.0 * np.pi) * s1[i] / x

    PotArr[0] = 3 * (PotArr[2] - PotArr[4]) + PotArr[6]
    ForArr[0] = 0.0

    for i in range(1, nx, 2):
        PotArr[i] = (PotArr[i - 1] + PotArr[i + 1]) / 2.0
        ForArr[i] = (ForArr[i - 1] + ForArr[i + 1]) / 2.0

    print(PotArr)
    return PotArr, ForArr


@njit
def simpson_harm_int_highL(nx, nl, dx, lmold, frac, densArr, lmx):
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
    lmold : int
        Highest harmonic level from the previous iteration.
    frac : float
        Fraction for damping oscillations between iterations.
    densArr : np.ndarray
        Density array of shape (lmx + 1, nx + 1).
    lmx : int
        Maximum harmonic level.

    Returns
    -------
    PotArr : np.ndarray
        Potential array of shape (lmx + 1, nx + 1).
    ForArr : np.ndarray
        Force array of shape (lmx + 1, nx + 1).
    For2Arr : np.ndarray
        Second-order force array of shape (lmx + 1, nx + 1).
    """
    PotArr = np.zeros((lmx + 1, nx + 1))
    ForArr = np.zeros((lmx + 1, nx + 1))
    For2Arr = np.zeros((lmx + 1, nx + 1))

    for l in range(0, nl + 1, 2):
        s1 = np.zeros(nx + 1)
        s2 = np.zeros(nx + 1)

        s1[0] = 0.0
        x = 2.0 * dx
        s1[2] = (x * dx / 3.0) * (
            4 * densArr[l // 2 + 1, 1] * (1.0 - dx / x) ** 2.0 + densArr[l // 2 + 1, 2]
        )
        xold = x

        for i in range(4, nx + 1, 2):
            x = i * dx
            s1a = (x * dx / 3.0) * (
                densArr[l // 2 + 1, i - 2] * (1.0 - 2 * dx / x) ** (l + 2)
                + 4 * densArr[l // 2 + 1, i - 1] * (1.0 - dx / x) ** (l + 2)
                + densArr[l // 2 + 1, i]
            )
            s1[i] = s1a + s1[i - 2] * (xold / x) ** (l + 1)
            xold = x

        s2[nx] = 0.0
        xold = nx * dx

        for i in range(nx - 2, 1, -2):
            x = i * dx
            s2a = (x * dx / 3.0) * (
                densArr[l // 2 + 1, i + 2] * (1.0 + 2 * dx / x) ** (1 - l)
                + 4 * densArr[l // 2 + 1, i + 1] * (1.0 + dx / x) ** (1 - l)
                + densArr[l // 2 + 1, i]
            )
            s2[i] = s2a + s2[i + 2] * (x / xold) ** l
            xold = x

        for i in range(2, nx + 1, 2):
            if l <= lmold:
                PotArr[l // 2 + 1, i] = frac * PotArr[l // 2 + 1, i] + (1.0 - frac) * (
                    4.0 * np.pi
                ) / (2.0 * l + 1.0) * (s1[i] + s2[i])
            else:
                PotArr[l // 2 + 1, i] = (
                    (4.0 * np.pi) / (2.0 * l + 1.0) * (s1[i] + s2[i])
                )

        for i in range(2, nx + 1, 2):
            x = i * dx
            ForArr[l // 2 + 1, i] = (
                -4 * np.pi / (2.0 * l + 1.0) * (-(l + 1) * s1[i] + l * s2[i]) / x
            )
            For2Arr[l // 2 + 1, i] = (
                -4
                * np.pi
                / (2.0 * l + 1.0)
                * (
                    (l + 1) * (l + 2) * s1[i] / x**2
                    + l * (l - 1) * s2[i] / x**2
                    - (2 * l + 1) * densArr[l // 2 + 1, i]
                )
            )

    PotArr[1, 0] = 3 * (PotArr[1, 2] - PotArr[1, 4]) + PotArr[1, 6]
    ForArr[1, 0] = 0.0
    For2Arr[1, 0] = 2 * For2Arr[1, 2] - For2Arr[1, 4]

    for l in range(2, nl + 1, 2):
        PotArr[l // 2 + 1, 0] = 0.0
        ForArr[l // 2 + 1, 0] = 0.0
        For2Arr[l // 2 + 1, 0] = 0.0

    for i in range(1, nx, 2):
        for l in range(0, nl + 1, 2):
            PotArr[l // 2 + 1, i] = (
                PotArr[l // 2 + 1, i - 1] + PotArr[l // 2 + 1, i + 1]
            ) / 2.0
            ForArr[l // 2 + 1, i] = (
                ForArr[l // 2 + 1, i - 1] + ForArr[l // 2 + 1, i + 1]
            ) / 2.0
            For2Arr[l // 2 + 1, i] = (
                For2Arr[l // 2 + 1, i - 1] + For2Arr[l // 2 + 1, i + 1]
            ) / 2.0

    return PotArr, ForArr, For2Arr
