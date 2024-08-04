"""
Truncation functions
"""

import numpy as np
from scipy.special import erfc


def eerfc(r: float, out: float, dr_trunc: float, scale: float = 4.0) -> float:
    """
    Computes the complementary error function with truncation for a given radius.

    Parameters
    ----------
    r : float
        The radius at which to compute the complementary error function.
    out : float
        The center point for the truncation region.
    dr_trunc : float
        The truncation scale or width.
    scale : float
        The truncation region width. Outside of this, the value
        is either 1 or 0. Measured in units of dr_trunc.

    Returns
    -------
    float
        The complementary error function value.
    """
    t = np.sqrt(0.5) * (r - out) / dr_trunc
    if t < -scale * np.sqrt(0.5):
        return 1.0
    elif t < scale * np.sqrt(0.5):
        return 0.5 * erfc(t)
    else:
        return 0.0


def eerfcprime(r: float, out: float, dr_trunc: float, scale: float = 4.0) -> float:
    """
    Computes the first derivative of the complementary error function.

    Parameters
    ----------
    r : float
        The radius at which to compute the first derivative of the complementary error function.
    out : float
        The center point for the truncation region.
    dr_trunc : float
        The truncation scale or width.
    scale : float
        The truncation region width. Outside of this, the value
        is 0. Measured in units of dr_trunc.

    Returns
    -------
    float
        The first derivative of the complementary error function.
    """
    t = np.sqrt(0.5) * (r - out) / dr_trunc
    t2 = t * t
    if t2 > scale**2 * 0.5:
        return 0.0
    else:
        return -0.5 * np.sqrt(2.0 / np.pi) / dr_trunc * np.exp(-t2)


def eerfc2prime(r: float, out: float, dr_trunc: float, scale: float) -> float:
    """
    Computes the second derivative of the complementary error function.

    Parameters
    ----------
    r : float
        The radius at which to compute the second derivative of the complementary error function.
    out : float
        The center point for the truncation region.
    dr_trunc : float
        The truncation scale or width.
    scale : float
        The truncation region width. Outside of this, the value
        is 0. Measured in units of dr_trunc.

    Returns
    -------
    float
        The second derivative of the complementary error function.
    """
    t = np.sqrt(0.5) * (r - out) / dr_trunc
    t2 = t * t
    if t2 > scale**2 * 0.5:
        return 0.0
    else:
        return 1.0 / np.sqrt(np.pi) / dr_trunc**2 * t * np.exp(-t2)


def eexp(r: float, out: float, dr_trunc: float, scale: float) -> float:
    """
    Computes a truncated Gaussian function. Intended for use with cutting
    a hole in the center of the disk.

    Parameters
    ----------
    r : float
        The radius at which to compute the second derivative of the complementary error function.
    out : float
        The center point for the truncation region.
    dr_trunc : float
        The truncation scale or width.
    scale : float
        The truncation region width. Outside of this, the value
        is 0. Measured in units of dr_trunc.

    Returns
    -------
    float
        The second derivative of the complementary error function.
    """
    t = np.sqrt(0.5) * (r - out) / dr_trunc
    t2 = t * t
    if t < -scale * np.sqrt(0.5):
        return 0.0
    elif t < scale * np.sqrt(0.5):
        return np.exp(-t2) / np.sqrt(2 * np.pi) / dr_trunc
    else:
        return 0.0
