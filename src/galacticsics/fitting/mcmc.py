"""MCMC parameter exploration (replaces mcmc_courteau.F orchestration)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class MCMCResult:
    """Stored MCMC chain samples."""

    samples: np.ndarray
    log_probability: np.ndarray
    acceptance_fraction: float


def metropolis_hastings(
    log_prob: Callable[[np.ndarray], float],
    initial: np.ndarray,
    *,
    n_steps: int,
    step_size: float,
    seed: int = 42,
) -> MCMCResult:
    """Simple Metropolis MCMC (single walker).

    For production use, prefer emcee with multiple walkers. This provides
    a zero-dependency fallback matching the legacy mcmc_courteau.F workflow.
    """
    rng = np.random.default_rng(seed)
    dim = initial.size
    samples = np.zeros((n_steps, dim))
    log_p = np.zeros(n_steps)
    current = initial.astype(float).copy()
    current_lp = log_prob(current)
    n_accept = 0

    for i in range(n_steps):
        proposal = current + step_size * rng.normal(size=dim)
        lp_prop = log_prob(proposal)
        if np.log(rng.uniform()) < lp_prop - current_lp:
            current = proposal
            current_lp = lp_prop
            n_accept += 1
        samples[i] = current
        log_p[i] = current_lp

    return MCMCResult(
        samples=samples,
        log_probability=log_p,
        acceptance_fraction=n_accept / n_steps,
    )
