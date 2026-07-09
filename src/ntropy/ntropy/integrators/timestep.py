"""Per-particle timestep bin computation (GADGET-style)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ntropy.particle_types import TypeRegistry


@dataclass
class TimestepConfig:
    """
    Configuration for adaptive, quantized per-particle timesteps.

    Timesteps follow the GADGET-2 prescription: an ideal step is estimated from
    the local acceleration and softening length, then quantized to a power-of-two
    multiple of ``dt_base``.  Each particle stores an integer **bin** ``b`` such
    that its timestep is ``dt_base * 2**b``.

    Attributes
    ----------
    eta : float
        Dimensionless accuracy parameter (GADGET ``eta``).  Typical values are
        ``0.01``–``0.05`` for collisionless systems.
    dt_base : float
        Finest (base) timestep in code units.  Bin ``0`` uses this value.
    max_bin : int
        Coarsest allowed bin index.  Maximum timestep is
        ``dt_base * 2**max_bin``.
    update_every : int
        Recompute timestep bins every this many fine substeps.
    accel_floor : float
        Lower bound on ``|a|`` when evaluating the criterion [code units],
        preventing divergent steps in very weak fields.
    """

    eta: float = 0.025
    dt_base: float = 0.025
    max_bin: int = 6
    update_every: int = 1
    accel_floor: float = 1e-6


def ideal_timestep(
    acc: np.ndarray,
    eps: np.ndarray,
    *,
    eta: float,
    accel_floor: float = 1e-6,
) -> np.ndarray:
    """
    Compute the ideal collisionless timestep for each particle.

    Uses the GADGET-style criterion (Springel 2005, eq. 9 for gravity-only):

    .. math::

       \\Delta t_i = \\eta \\sqrt{\\frac{\\varepsilon_i}{|\\mathbf{a}_i|}}

    where :math:`\\varepsilon_i` is the Plummer softening length and
    :math:`\\mathbf{a}_i` is the gravitational acceleration.

    Parameters
    ----------
    acc : ndarray, shape (N, 3)
        Gravitational accelerations [code units].
    eps : ndarray, shape (N,)
        Per-particle softening lengths [kpc].
    eta : float
        Accuracy parameter (see :class:`TimestepConfig`).
    accel_floor : float, optional
        Minimum acceleration magnitude used in the denominator.

    Returns
    -------
    dt_ideal : ndarray, shape (N,)
        Ideal timestep per particle [code units].

    Notes
    -----
    Particles with very small accelerations are assigned large ideal timesteps;
    callers should clamp via :func:`bin_from_timestep` and type-specific limits.
    """
    acc_mag = np.linalg.norm(acc, axis=1)
    acc_mag = np.maximum(acc_mag, accel_floor)
    return eta * np.sqrt(eps / acc_mag)


def bin_from_timestep(
    dt_ideal: np.ndarray,
    dt_base: float,
    *,
    max_bin: int,
) -> np.ndarray:
    """
    Quantize ideal timesteps to power-of-two bins.

    Returns the smallest integer bin ``b`` such that
    ``dt_base * 2**b >= dt_ideal``, capped at ``max_bin``.

    Parameters
    ----------
    dt_ideal : ndarray, shape (N,)
        Ideal timesteps [code units].
    dt_base : float
        Base (finest) timestep [code units].
    max_bin : int
        Maximum allowed bin index.

    Returns
    -------
    bins : ndarray, shape (N,), dtype int32
        Integer bin index per particle.

    Examples
    --------
    >>> import numpy as np
    >>> bin_from_timestep(np.array([0.02, 0.05, 0.2]), 0.025, max_bin=4)
    array([0, 1, 3], dtype=int32)
    """
    dt_base = float(dt_base)
    if dt_base <= 0:
        raise ValueError(f"dt_base must be > 0, got {dt_base}")
    ratio = np.maximum(dt_ideal / dt_base, 1.0)
    bins = np.ceil(np.log2(ratio)).astype(np.int32)
    return np.clip(bins, 0, int(max_bin))


def timestep_from_bin(bins: np.ndarray, dt_base: float) -> np.ndarray:
    """
    Convert bin indices to physical timesteps.

    Parameters
    ----------
    bins : ndarray, shape (N,), dtype int32
        Integer bin per particle.
    dt_base : float
        Base timestep [code units].

    Returns
    -------
    dt : ndarray, shape (N,)
        Timestep ``dt_base * 2**bins[i]`` for each particle.
    """
    return dt_base * np.power(2.0, bins.astype(float))


def factor_from_bin(bins: np.ndarray) -> np.ndarray:
    """
    Return the integer sync factor ``2**b`` for each bin.

    Parameters
    ----------
    bins : ndarray, shape (N,), dtype int32
        Bin indices.

    Returns
    -------
    factors : ndarray, shape (N,), dtype int32
        Sync factors used in the tiered integrator active test
        ``step % factors[i] == 0``.
    """
    return np.power(2, bins.astype(np.int64)).astype(np.int64)


def clamp_bins_for_types(
    bins: np.ndarray,
    type_id: np.ndarray,
    registry: TypeRegistry,
    *,
    global_max_bin: int,
) -> np.ndarray:
    """
    Clamp bin indices to per-type ``min_timestep_bin`` / ``max_timestep_bin``.

    Parameters
    ----------
    bins : ndarray, shape (N,), dtype int32
        Proposed bin indices.
    type_id : ndarray, shape (N,), dtype int32
        Particle type ids.
    registry : TypeRegistry
        Type metadata with per-type bin limits.
    global_max_bin : int
        Fallback maximum when a type has no explicit ``max_timestep_bin``.

    Returns
    -------
    clipped : ndarray, shape (N,), dtype int32
        Type-clamped bin indices.
    """
    out = bins.copy()
    for label, spec in registry.types.items():
        mask = type_id == spec.id
        if not np.any(mask):
            continue
        tmax = spec.max_timestep_bin if spec.max_timestep_bin is not None else global_max_bin
        out[mask] = np.clip(out[mask], spec.min_timestep_bin, tmax)
    return out


def update_timestep_bins(
    acc: np.ndarray,
    eps: np.ndarray,
    type_id: np.ndarray,
    registry: TypeRegistry,
    current_bins: np.ndarray | None,
    config: TimestepConfig,
) -> np.ndarray:
    """
    Update per-particle timestep bins from accelerations (GADGET-style).

    Computes ideal timesteps, quantizes to power-of-two bins, clamps to
    type-specific limits, then applies the GADGET damping rule: a particle's
    bin may change by at most **one** level per update to avoid oscillations.

    Parameters
    ----------
    acc : ndarray, shape (N, 3)
        Current accelerations [code units].
    eps : ndarray, shape (N,)
        Per-particle softening [kpc].
    type_id : ndarray, shape (N,), dtype int32
        Particle type ids.
    registry : TypeRegistry
        Type registry with per-type bin limits.
    current_bins : ndarray, shape (N,), dtype int32 or None
        Existing bins; ``None`` for initial assignment (no damping).
    config : TimestepConfig
        Timestep parameters.

    Returns
    -------
    new_bins : ndarray, shape (N,), dtype int32
        Updated integer bin per particle.

    Notes
    -----
    This matches the spirit of GADGET-2's ``get_timestep`` / integer timestep
    hierarchy: timesteps are determined locally from dynamics, not fixed by type.
    Types only set allowed bin **ranges** (e.g. gas forced finer than halo).
    """
    dt_ideal = ideal_timestep(
        acc, eps, eta=config.eta, accel_floor=config.accel_floor
    )
    proposed = bin_from_timestep(
        dt_ideal, config.dt_base, max_bin=config.max_bin
    )
    proposed = clamp_bins_for_types(
        proposed, type_id, registry, global_max_bin=config.max_bin
    )

    if current_bins is None:
        return proposed

    current_bins = np.asarray(current_bins, dtype=np.int32)
    delta = proposed - current_bins
    # GADGET: limit change to one bin per update
    delta = np.clip(delta, -1, 1)
    return np.clip(current_bins + delta, 0, config.max_bin)


def active_mask_for_step(step: int, bins: np.ndarray) -> np.ndarray:
    """
    Return which particles are active on fine substep ``step``.

    Particle ``i`` is active when ``step % 2**bins[i] == 0``.

    Parameters
    ----------
    step : int
        Fine substep counter (1-based in the integrator loop).
    bins : ndarray, shape (N,), dtype int32
        Per-particle bin indices.

    Returns
    -------
    active : ndarray, shape (N,), dtype bool
        True for particles to be kicked/drifted on this substep.
    """
    factors = factor_from_bin(bins)
    return (step % factors) == 0
