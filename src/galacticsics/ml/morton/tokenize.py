"""Morton tokenization of particle snapshots."""

from __future__ import annotations

from typing import Literal

import numpy as np

from galacticsics.ml.morton.morton_keys import morton_keys

COMPONENT_IDS = {"disk": 0, "halo": 1, "bulge": 2}
ID_TO_COMPONENT = {v: k for k, v in COMPONENT_IDS.items()}
# ntropy TypeRegistry.default_galaxy: halo=1, bulge=2, disk=3
TYPE_ID_TO_COMPONENT = {1: 1, 2: 2, 3: 0}


def center_phase_space(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subtract the (mass-weighted) center of mass from positions and velocities.

    Parameters
    ----------
    pos, vel : ndarray, shape (N, 3)
        Particle phase-space coordinates.
    mass : ndarray, shape (N,), optional
        Masses for COM weighting.  Uniform (unweighted) mean when omitted.

    Returns
    -------
    pos_c, vel_c : ndarray, shape (N, 3)
        Centered copies (inputs are not modified).
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    if mass is None:
        com = pos.mean(axis=0)
        vcom = vel.mean(axis=0)
    else:
        m = np.asarray(mass, dtype=np.float64).reshape(-1)
        w = m / max(float(m.sum()), 1e-30)
        com = (pos * w[:, None]).sum(axis=0)
        vcom = (vel * w[:, None]).sum(axis=0)
    return pos - com, vel - vcom


def rotate_about_z(
    pos: np.ndarray,
    vel: np.ndarray,
    phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate positions and velocities by angle ``phi`` about the ``z`` axis.

    Midplane (``z``) is preserved so face-on Fourier / map losses stay valid.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    c = float(np.cos(phi))
    s = float(np.sin(phi))
    # Row-vector convention: (x, y) → (c x - s y, s x + c y)
    rp = pos.copy()
    rv = vel.copy()
    rp[:, 0] = c * pos[:, 0] - s * pos[:, 1]
    rp[:, 1] = s * pos[:, 0] + c * pos[:, 1]
    rv[:, 0] = c * vel[:, 0] - s * vel[:, 1]
    rv[:, 1] = s * vel[:, 0] + c * vel[:, 1]
    return rp, rv


def random_rotate_z(
    pos: np.ndarray,
    vel: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a uniform random in-plane rotation ``φ ∼ U[0, 2π)``."""
    return rotate_about_z(pos, vel, float(rng.uniform(0.0, 2.0 * np.pi)))


def _component_ids(tags: np.ndarray | None, type_id: np.ndarray | None, n: int) -> np.ndarray:
    """
    Map snapshot labels to ML component ids ``{0=disk, 1=halo, 2=bulge}``.

    Prefers string ``tags`` when present.  Otherwise maps ntropy
    ``TypeRegistry.default_galaxy`` integer ids (halo=1, bulge=2, disk=3).
    Unknown ids fall back to ``0`` (disk) after a clip into ``{0,1,2}`` only
    when already in that range.
    """
    out = np.zeros(n, dtype=np.int64)
    if tags is not None:
        for name, cid in COMPONENT_IDS.items():
            out[np.asarray(tags) == name] = cid
        return out
    if type_id is not None:
        tid = np.asarray(type_id, dtype=np.int64)
        mapped = np.full(n, -1, dtype=np.int64)
        for src, dst in TYPE_ID_TO_COMPONENT.items():
            mapped[tid == src] = dst
        # Already-compact {0,1,2} labels (synthetic / unit tests)
        compact = (tid >= 0) & (tid <= 2) & (mapped < 0)
        mapped[compact] = tid[compact]
        mapped[mapped < 0] = 0
        return mapped
    return out


def subsample_stratified(
    component_id: np.ndarray,
    n_out: int,
    *,
    rng: np.random.Generator,
    fractions: dict[int, float] | None = None,
) -> np.ndarray:
    """
    Draw ``n_out`` indices stratified by component.

    Parameters
    ----------
    component_id : ndarray, shape (N,)
        Integer labels ``{0=disk, 1=halo, 2=bulge}``.
    n_out : int
        Desired subsample size.  If ``n_out >= N``, returns ``arange(N)``.
    rng : numpy.random.Generator
        Source of randomness for within-component draws.
    fractions : dict of int to float, optional
        Target mass fractions per component.  Default matches the corpus mix
        disk:halo:bulge ≈ 4:2:1.

    Returns
    -------
    indices : ndarray, shape (n_out,), dtype int64
        Indices into the original particle array (no replacement within a
        component when possible).
    """
    n = int(component_id.shape[0])
    if n_out >= n:
        return np.arange(n, dtype=np.int64)
    if fractions is None:
        fractions = {0: 4 / 7, 1: 2 / 7, 2: 1 / 7}
    present = sorted(set(int(c) for c in np.unique(component_id)))
    weights = np.array([fractions.get(c, 1.0) for c in present], dtype=float)
    weights = weights / weights.sum()
    counts = np.maximum(1, np.round(weights * n_out).astype(int))
    # Adjust to exact n_out
    while counts.sum() > n_out:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < n_out:
        counts[np.argmax(weights)] += 1
    chosen: list[np.ndarray] = []
    for c, k in zip(present, counts):
        idx = np.flatnonzero(component_id == c)
        if idx.size == 0:
            continue
        take = min(int(k), idx.size)
        chosen.append(rng.choice(idx, size=take, replace=False))
    if not chosen:
        return rng.choice(n, size=n_out, replace=False).astype(np.int64)
    out = np.concatenate(chosen)
    if out.size < n_out:
        rest = np.setdiff1d(np.arange(n), out, assume_unique=False)
        need = n_out - out.size
        out = np.concatenate([out, rng.choice(rest, size=need, replace=False)])
    elif out.size > n_out:
        out = rng.choice(out, size=n_out, replace=False)
    return out.astype(np.int64)


def tokenize_morton(
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    component_id: np.ndarray | None = None,
    tags: np.ndarray | None = None,
    type_id: np.ndarray | None = None,
    bits: int = 10,
    order: Literal["morton", "random"] = "morton",
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Convert particles to Morton-ordered tokens ``(c, Δm, x, v)``.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Cartesian positions [kpc].
    vel : ndarray, shape (N, 3)
        Cartesian velocities [code units].
    component_id : ndarray, shape (N,), optional
        Integer labels ``{0=disk, 1=halo, 2=bulge}``.  Built from ``tags`` or
        ``type_id`` when omitted.
    tags : ndarray of str, optional
        Component name per particle.
    type_id : ndarray of int, optional
        Fallback integer type ids.
    bits : int
        Bits per axis for the Morton key (key space ``(2^bits)^3``).
    order : {'morton', 'random'}
        Sort by Morton key or apply a random permutation (AR ablation).
    rng : numpy.random.Generator, optional
        Required flavour for ``order='random'`` (a default generator is used
        when ``None``).

    Returns
    -------
    dict of ndarray
        ``c`` (N,), ``dm`` (N,), ``dx`` (N,3) absolute positions in the chosen
        order (legacy key name), ``v`` (N,3), ``keys`` (N,), ``box_min`` (3,),
        ``box_size`` (,), ``bits``, ``order``.

    Notes
    -----
    ``dm[0]`` is the first Morton key; ``dm[1:] = diff(keys)``.  Absolute
    positions are stored rather than cell residuals because the 3-D Morton
    expand used here is not a clean bijection for decode.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    n = pos.shape[0]
    if component_id is None:
        component_id = _component_ids(tags, type_id, n)
    else:
        component_id = np.asarray(component_id, dtype=np.int64)

    box_min = pos.min(axis=0)
    box_size = float((pos.max(axis=0) - box_min).max())
    if box_size <= 0:
        box_size = 1.0
    keys = morton_keys(pos, box_min, box_size, bits=bits)

    if order == "random":
        rng = rng or np.random.default_rng(0)
        perm = rng.permutation(n)
    else:
        perm = np.argsort(keys, kind="stable")

    pos_s = pos[perm]
    vel_s = vel[perm]
    keys_s = keys[perm]
    c_s = component_id[perm]

    max_val = (1 << bits) - 1
    # Absolute positions in Morton order (v1). Cell residuals are attractive for
    # generative Δm models, but ntropy's 3D Morton expand is not a clean bijection
    # for decode; keep absolute x and use Δm as an ordering feature.
    x = pos_s

    dm = np.empty(n, dtype=np.float64)
    dm[0] = float(keys_s[0])
    if n > 1:
        dm[1:] = np.diff(keys_s.astype(np.float64))

    return {
        "c": c_s.astype(np.int64),
        "dm": dm,
        "dx": x.astype(np.float64),  # absolute x,y,z in Morton order (legacy name)
        "v": vel_s.astype(np.float64),
        "keys": keys_s.astype(np.uint64),
        "box_min": box_min.astype(np.float64),
        "box_size": np.asarray(box_size, dtype=np.float64),
        "bits": np.asarray(bits, dtype=np.int64),
        "order": np.asarray(order),
    }


def particles_from_tokens(
    tokens: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct ``(pos, vel, component_id)`` from Morton tokens.

    Parameters
    ----------
    tokens : dict
        Output of :func:`tokenize_morton` (or a model decode with the same
        keys).  Uses absolute positions stored under ``dx``.

    Returns
    -------
    pos : ndarray, shape (N, 3)
    vel : ndarray, shape (N, 3)
    component_id : ndarray, shape (N,)
    """
    pos = np.asarray(tokens["dx"], dtype=np.float64)
    v = np.asarray(tokens["v"], dtype=np.float64)
    c = np.asarray(tokens["c"], dtype=np.int64)
    return pos, v, c
