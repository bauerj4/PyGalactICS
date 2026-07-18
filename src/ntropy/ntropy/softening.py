"""Plummer softened gravity kernels."""

from __future__ import annotations

import numpy as np

from ntropy.units import G

# Above this particle count, skip O(N²) potential energy (use KE-only drift proxy).
LARGE_N_ENERGY_THRESHOLD = 16_384


def pairwise_softening(eps_i: np.ndarray, eps_j: np.ndarray) -> np.ndarray:
    """
    Symmetric pairwise softening length (Gadget-style mean).

    Parameters
    ----------
    eps_i : ndarray, shape (N,)
        Softening lengths for target particles.
    eps_j : ndarray, shape (N,)
        Softening lengths for source particles.

    Returns
    -------
    h_ij : ndarray, shape (N, N)
        Pairwise softening with ``h_ij = 0.5 * (eps_i + eps_j)``.
    """
    return 0.5 * (eps_i[:, None] + eps_j[None, :])


def softened_acceleration_vectorized(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Plummer-softened gravitational acceleration.

    .. math::

       \\mathbf{a}_i = \\sum_j G m_j
       \\frac{\\mathbf{r}_j - \\mathbf{r}_i}
       {(|\\mathbf{r}_j-\\mathbf{r}_i|^2 + h_{ij}^2)^{3/2}}

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions [kpc].
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Per-particle softening lengths [kpc].

    Returns
    -------
    acc : ndarray, shape (N, 3)
        Accelerations [kpc / (100 km/s)²] in GalactICS units.

    Notes
    -----
    Complexity is O(N²) in memory and time.  Self-interactions are excluded.
    """
    dr = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(dr * dr, axis=2)
    h_ij = pairwise_softening(eps, eps)
    h2 = h_ij**2
    denom = (r2 + h2) ** 1.5
    np.fill_diagonal(denom, np.inf)
    factor = G * mass[None, :] / denom
    acc = -(factor[..., None] * dr).sum(axis=1)
    return acc


def softened_acceleration_targets(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """
    Vectorized softened acceleration for a subset of target particles.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Softening lengths.
    target_indices : ndarray, shape (N_t,)
        Indices of particles to evaluate.

    Returns
    -------
    acc : ndarray, shape (N_t, 3)
        Accelerations on the requested targets only.

    Notes
    -----
    Complexity is O(N_t × N).  Self-interactions are excluded.
    """
    targets = np.asarray(target_indices, dtype=int)
    t_pos = pos[targets]
    t_eps = eps[targets]
    dr = t_pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(dr * dr, axis=2)
    h_ij = 0.5 * (t_eps[:, None] + eps[None, :])
    h2 = h_ij**2
    denom = (r2 + h2) ** 1.5
    rows = np.arange(len(targets))
    denom[rows, targets] = np.inf
    factor = G * mass[None, :] / denom
    return -(factor[..., None] * dr).sum(axis=1)


def softened_potential_energy(
    pos: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
) -> float:
    """
    Total pairwise softened potential energy.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Particle positions.
    mass : ndarray, shape (N,)
        Particle masses.
    eps : ndarray, shape (N,)
        Softening lengths.

    Returns
    -------
    energy : float
        Softened potential energy (negative for bound pairs).
    """
    n = len(mass)
    energy = 0.0
    for i in range(n - 1):
        mi = mass[i]
        if mi == 0.0:
            continue
        dr = pos[i + 1 :] - pos[i]
        r2 = np.sum(dr * dr, axis=1)
        h = 0.5 * (eps[i] + eps[i + 1 :])
        mj = mass[i + 1 :]
        energy -= G * mi * np.sum(mj / np.sqrt(r2 + h * h))
    return float(energy)


def kinetic_energy(vel: np.ndarray, mass: np.ndarray) -> float:
    """
    Total kinetic energy.

    Parameters
    ----------
    vel : ndarray, shape (N, 3)
        Velocities.
    mass : ndarray, shape (N,)
        Masses.

    Returns
    -------
    energy : float
    """
    return float(0.5 * np.sum(mass * np.sum(vel * vel, axis=1)))


def virial_diagnostic(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    rtol: float = 0.3,
    max_particles: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float | bool | int]:
    """
    Virial-theorem check for a self-gravitating N-body state.

    For equilibrium, ``2T + W ≈ 0`` (equivalently ``T/|W| ≈ 0.5`` or
    ``2T/|W| ≈ 1``).  Uses the same Plummer-softened pairwise potential as
    :func:`softened_potential_energy`.

    When ``N`` exceeds ``max_particles`` (default
    :data:`LARGE_N_ENERGY_THRESHOLD`), a random subset is used for the
    potential term so the check stays O(N_sub²).

    Parameters
    ----------
    pos, vel, mass, eps
        Particle state arrays.
    rtol
        Relative tolerance on ``|2T + W| / |W|`` for the equilibrium flag.
    max_particles
        Cap on particles used for the potential sum; ``None`` uses
        :data:`LARGE_N_ENERGY_THRESHOLD`.
    rng
        Random generator for subsampling (default: unseeded).

    Returns
    -------
    dict
        ``kinetic_energy``, ``potential_energy``, ``virial_sum`` (2T+W),
        ``virial_ratio`` (2T/|W|), ``ke_over_abs_pe`` (T/|W|),
        ``virial_residual_rel`` (|2T+W|/|W|), ``is_virial_equilibrium``,
        ``n_particles``, ``n_used``, ``subsampled``.
    """
    n = len(mass)
    cap = LARGE_N_ENERGY_THRESHOLD if max_particles is None else max_particles
    subsampled = n > cap
    if subsampled:
        # T and W must use the same particles.  Prefer mass-weighted sampling so a
        # disk+halo mix (many light disk particles) still represents the mass that
        # dominates the potential; uniform particle picks understate |W|.
        rng = rng or np.random.default_rng()
        u = np.clip(rng.random(n), 1e-300, 1.0)
        keys = u ** (1.0 / np.maximum(mass.astype(float), 1e-300))
        idx = np.argpartition(keys, -cap)[-cap:]
        ke = kinetic_energy(vel[idx], mass[idx])
        pe = softened_potential_energy(pos[idx], mass[idx], eps[idx])
        n_used = cap
    else:
        ke = kinetic_energy(vel, mass)
        pe = softened_potential_energy(pos, mass, eps)
        n_used = n
    abs_pe = abs(pe)
    virial_sum = 2.0 * ke + pe
    virial_ratio = (2.0 * ke / abs_pe) if abs_pe > 0 else 0.0
    ke_over_abs_pe = (ke / abs_pe) if abs_pe > 0 else 0.0
    residual_rel = abs(virial_sum) / max(abs_pe, 1e-30)
    return {
        "kinetic_energy": ke,
        "potential_energy": pe,
        "virial_sum": virial_sum,
        "virial_ratio": virial_ratio,
        "ke_over_abs_pe": ke_over_abs_pe,
        "virial_residual_rel": residual_rel,
        "is_virial_equilibrium": residual_rel <= rtol,
        "n_particles": n,
        "n_used": n_used,
        "subsampled": subsampled,
    }


def total_energy(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    *,
    allow_kinetic_only: bool = True,
) -> float:
    """
    Total energy (kinetic + softened potential).

    For ``N > LARGE_N_ENERGY_THRESHOLD`` (default 16384), returns kinetic
    energy only to avoid O(N²) memory and time.  Tiered diagnostics then
    report kinetic-energy drift, which is still a useful stability proxy.

    Parameters
    ----------
    pos, vel, mass, eps
        Particle state arrays.
    allow_kinetic_only : bool
        When ``True`` (default), large ``N`` uses the KE-only fast path.

    Returns
    -------
    energy : float
    """
    ke = kinetic_energy(vel, mass)
    if allow_kinetic_only and len(mass) > LARGE_N_ENERGY_THRESHOLD:
        return ke
    return ke + softened_potential_energy(pos, mass, eps)
