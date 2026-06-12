"""Modern numerics layer replacing legacy splined/splintd/simpson/golden routines."""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.special import eval_legendre


def natural_cubic_spline(x: np.ndarray, y: np.ndarray) -> CubicSpline:
    """
    Natural cubic spline replacing ``legacy/fortran/splined.f``.

    Parameters
    ----------
    x, y : ndarray
        1-D knot arrays of equal length (``len(x) >= 2``).

    Returns
    -------
    CubicSpline
        SciPy spline with natural boundary conditions (second derivative zero
        at endpoints), matching ``yp1 = ypn = 1e32`` in the Fortran code.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if len(x) < 2:
        raise ValueError("need at least two points for a spline")
    return CubicSpline(x, y, bc_type="natural")


def legendre_even_l(costheta: float, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Legendre polynomials P_l(cos theta) and dP_l/dtheta for even l only.

    Returns arrays indexed 0..lmax//2 for l = 0, 2, 4, ...
    """
    if lmax % 2 != 0:
        raise ValueError("lmax must be even")
    n = lmax // 2 + 1
    p = np.zeros(n, dtype=float)
    dp = np.zeros(n, dtype=float)
    sintheta = math.sqrt(max(0.0, 1.0 - costheta * costheta))
    for i, ell in enumerate(range(0, lmax + 1, 2)):
        p[i] = float(eval_legendre(ell, costheta))
        if abs(sintheta) < 1e-14 or ell == 0:
            dp[i] = 0.0
        else:
            # dP_l/dtheta = (l/(sin theta)) * (cos theta * P_l - P_{l-1})
            p_lm1 = float(eval_legendre(ell - 1, costheta)) if ell > 0 else 0.0
            dp[i] = ell * (costheta * p[i] - p_lm1) / sintheta
    return p, dp


def simpson_integrate(func, a: float, b: float, n: int = 128) -> float:
    """Simpson integration replacing simpson.c."""
    if n % 2 != 0:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = np.asarray([func(xi) for xi in x], dtype=float)
    return float(integrate.simpson(y, x=x))


def bounded_maximize(func, a: float, b: float, *, xtol: float = 1e-6) -> float:
    """Find argmax on [a,b] replacing golden.c (for unimodal functions)."""
    result = minimize_scalar(lambda x: -func(x), bounds=(a, b), method="bounded", options={"xatol": xtol})
    return float(result.x)
