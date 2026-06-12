"""Optimization drivers replacing simplex.c (scipy.optimize)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass
class FitResult:
    """Result of a chi-squared minimization."""

    parameters: np.ndarray
    chi2: float
    success: bool
    message: str


def nelder_mead_fit(
    chi2: Callable[[np.ndarray], float],
    x0: Sequence[float],
    *,
    bounds: list[tuple[float, float]] | None = None,
    maxiter: int = 500,
) -> FitResult:
    """Downhill simplex minimization via scipy.optimize.minimize."""
    x0_arr = np.asarray(x0, dtype=float)
    if bounds is not None:
        result = minimize(
            chi2,
            x0_arr,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
    else:
        result = minimize(
            chi2,
            x0_arr,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
        )
    return FitResult(
        parameters=result.x,
        chi2=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
    )
