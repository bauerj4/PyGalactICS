"""Export particle states as fixed feature tensors for ML / transformer models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Column order for :func:`particles_to_feature_matrix`
FEATURE_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "log_mass",
    "log_eps",
    "type_id",
)


@dataclass
class ParticleFeatureBatch:
    """
    Token sequence derived from an N-body particle snapshot.

    Designed as input to permutation-equivariant or transformer encoders that
    learn compressed representations of particle data (alternative to harmonic
    or parametric galaxy models).

    Attributes
    ----------
    features : ndarray, shape (N, F)
        Per-particle feature matrix; columns match :data:`FEATURE_NAMES`.
    type_id : ndarray, shape (N,), dtype int32
        Integer particle type per row.
    tags : ndarray of object, shape (N,) or None
        Optional string component labels.
    n_particles : int
        Number of tokens ``N``.
    n_features : int
        Feature dimension ``F``.
    mask : ndarray, shape (N,), dtype bool
        Valid-token mask (all ``True`` for dense snapshots; supports padding later).
    """

    features: np.ndarray
    type_id: np.ndarray
    tags: np.ndarray | None
    n_particles: int
    n_features: int
    mask: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """
        Serialize arrays for ``np.savez`` or PyTorch ``Dataset`` loaders.

        Returns
        -------
        data : dict
            Keys ``features``, ``type_id``, ``mask``, and optionally ``tags``.
        """
        out: dict[str, np.ndarray] = {
            "features": self.features,
            "type_id": self.type_id,
            "mask": self.mask,
        }
        if self.tags is not None:
            out["tags"] = self.tags
        return out


def particles_to_feature_matrix(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    eps: np.ndarray,
    type_id: np.ndarray | None = None,
    *,
    mass_floor: float = 1e-30,
    eps_floor: float = 1e-12,
) -> np.ndarray:
    """
    Build a per-particle feature matrix for neural encoders.

    Uses log-scaled mass and softening for numerical stability.  Positions and
    velocities are in GalactICS code units (kpc, 100 km/s).

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Cartesian positions [kpc].
    vel : ndarray, shape (N, 3)
        Cartesian velocities [100 km/s].
    mass : ndarray, shape (N,)
        Particle masses [GalactICS mass units].
    eps : ndarray, shape (N,)
        Softening lengths [kpc].
    type_id : ndarray, shape (N,), optional
        Integer type ids; zeros when absent.
    mass_floor, eps_floor : float
        Floors for logarithms.

    Returns
    -------
    features : ndarray, shape (N, 9)
        Columns: ``x,y,z,vx,vy,vz,log_mass,log_eps,type_id``.

    Notes
    -----
    For transformer models, treat each row as a token.  Neural encoders use
    :data:`~galacticsics.representations.preprocess.ENCODER_FEATURE_NAMES`
    (8 columns); ``type_id`` is passed to ``nn.Embedding`` separately.
    """
    n = len(mass)
    if type_id is None:
        type_id = np.zeros(n, dtype=np.float64)
    else:
        type_id = np.asarray(type_id, dtype=np.float64)

    return np.column_stack(
        [
            pos,
            vel,
            np.log10(np.maximum(mass, mass_floor)),
            np.log10(np.maximum(eps, eps_floor)),
            type_id,
        ]
    )


def particle_state_to_batch(state, *, registry=None) -> ParticleFeatureBatch:
    """
    Convert an ntropy :class:`~ntropy.particles.ParticleState` to a feature batch.

    Parameters
    ----------
    state : ParticleState
        Particle snapshot (pos, vel, mass, eps, optional type_id/tags).
    registry : TypeRegistry, optional
        Unused today; reserved for type embedding metadata.

    Returns
    -------
    batch : ParticleFeatureBatch
        Token sequence ready for ML encoders.

    Raises
    ------
    ImportError
        If ntropy is not installed (lazy import).
    """
    type_id = state.type_id
    if type_id is None:
        type_id = np.zeros(state.n, dtype=np.int32)

    features = particles_to_feature_matrix(
        state.pos, state.vel, state.mass, state.eps, type_id
    )
    mask = np.ones(state.n, dtype=bool)
    return ParticleFeatureBatch(
        features=features,
        type_id=np.asarray(type_id, dtype=np.int32),
        tags=state.tags,
        n_particles=state.n,
        n_features=features.shape[1],
        mask=mask,
    )
