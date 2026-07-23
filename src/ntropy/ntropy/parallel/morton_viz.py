"""Visualize Morton (Z-order) curve construction for domain decomposition."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ntropy.parallel.domains import _expand_bits


def morton_keys_2d(
    xy: np.ndarray,
    *,
    box_min: np.ndarray | None = None,
    box_size: float | None = None,
    bits: int = 10,
) -> np.ndarray:
    """
    2D Morton (Z-order) keys for points in the plane.

    Same bit-interleave idea as :func:`~ntropy.parallel.domains.peano_keys`,
    but only ``x`` and ``y`` (useful for face-on construction demos).

    Parameters
    ----------
    xy : ndarray, shape (N, 2)
        Plane coordinates.
    box_min : ndarray, shape (2,), optional
        Lower corner of the bounding square (default: data min).
    box_size : float, optional
        Square side length (default: max extent of the data).
    bits : int
        Bits per axis (key has ``2 * bits`` significant bits).

    Returns
    -------
    keys : ndarray, shape (N,), dtype uint64
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {xy.shape}")
    if box_min is None:
        box_min = xy.min(axis=0)
    else:
        box_min = np.asarray(box_min, dtype=float)
    if box_size is None:
        box_size = float((xy.max(axis=0) - box_min).max())
    if box_size <= 0:
        box_size = 1.0
    scaled = (xy - box_min) / box_size
    scaled = np.clip(scaled, 0.0, 1.0 - 1e-12)
    max_val = (1 << bits) - 1
    ix = (scaled[:, 0] * max_val).astype(np.uint64)
    iy = (scaled[:, 1] * max_val).astype(np.uint64)
    return _expand_bits(ix) | (_expand_bits(iy) << 1)


def morton_curve_cell_centers(bits: int) -> np.ndarray:
    """
    Cell-center polyline of the unit-square Morton curve at resolution ``2**bits``.

    Parameters
    ----------
    bits : int
        Bits per axis (``1`` → 2×2 Z, ``2`` → 4×4, …).

    Returns
    -------
    centers : ndarray, shape (4**bits, 2)
        Centers in ``[0, 1]²``, ordered by increasing Morton key.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    n = 1 << bits
    coords = np.arange(n, dtype=np.uint64)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    keys = _expand_bits(xx.ravel()) | (_expand_bits(yy.ravel()) << 1)
    order = np.argsort(keys, kind="stable")
    centers = np.column_stack(
        [
            (xx.ravel()[order].astype(float) + 0.5) / n,
            (yy.ravel()[order].astype(float) + 0.5) / n,
        ]
    )
    return centers


def _subsample_xy(
    pos: np.ndarray,
    n_show: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take an ``(n_show, 2)`` face-on subsample from ``(N, 2|3)`` positions."""
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError(f"pos must have shape (N, 2) or (N, 3), got {pos.shape}")
    xy = pos[:, :2]
    if xy.shape[0] <= n_show:
        return xy.copy()
    idx = np.sort(rng.choice(xy.shape[0], size=n_show, replace=False))
    return xy[idx]


def write_morton_construction_gif(
    pos: np.ndarray,
    out_path: str | Path,
    *,
    n_particles: int = 96,
    curve_bits: Sequence[int] = (1, 2, 3, 4),
    n_path_frames: int = 48,
    hold_frames: int = 8,
    fps: int = 8,
    dpi: int = 120,
    seed: int = 0,
    title: str | None = None,
) -> Path:
    """
    Build a GIF showing Morton Z-order construction on an example distribution.

    Animation narrative
    -------------------
    1. **Curve refinement** — draw the classic Z-curve on the unit square at
       increasing bit depth (``2×2``, ``4×4``, …).
    2. **Particle visit order** — overlay a face-on subsample of ``pos``, colour
       by 2D Morton rank, and grow a polyline that visits particles in key order.

    Production MPI domains use 3D Morton keys (:func:`~ntropy.parallel.domains.peano_keys`);
    this demo uses the same bit-interleave construction in the *x–y* plane so the
    path matches what you see.

    Parameters
    ----------
    pos : ndarray, shape (N, 2) or (N, 3)
        Example particle positions (kpc). Only ``x, y`` are used.
    out_path : path-like
        Destination ``.gif`` path.
    n_particles : int
        How many particles to show in the visit-order phase.
    curve_bits : sequence of int
        Bit depths for the construction phase.
    n_path_frames : int
        Number of frames while growing the particle polyline.
    hold_frames : int
        Extra frames to hold each completed curve level.
    fps, dpi : int
        GIF frame rate and render DPI.
    seed : int
        RNG seed for particle subsampling.
    title : str, optional
        Figure title override.

    Returns
    -------
    path : Path
        Written GIF path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    xy = _subsample_xy(pos, n_particles, rng)
    box_min = xy.min(axis=0)
    box_size = float((xy.max(axis=0) - box_min).max())
    if box_size <= 0:
        box_size = 1.0
    # Pad slightly so edge particles are not clipped.
    pad = 0.04 * box_size
    box_min = box_min - pad
    box_size = box_size + 2.0 * pad

    keys = morton_keys_2d(xy, box_min=box_min, box_size=box_size, bits=10)
    order = np.argsort(keys, kind="stable")
    xy_sorted = xy[order]
    rank = np.empty(len(xy), dtype=float)
    rank[order] = np.linspace(0.0, 1.0, len(xy))

    # Unit-square curves mapped into the particle bounding square.
    curves = {
        b: box_min + box_size * morton_curve_cell_centers(b) for b in curve_bits
    }

    # Frame plan: each bit level gets (draw + hold), then path growth frames.
    level_frames: list[tuple[str, int, int]] = []
    for b in curve_bits:
        n_draw = max(4, 2 ** (b + 1))  # more segments as the curve lengthens
        level_frames.append(("curve", b, n_draw))
        level_frames.append(("hold", b, hold_frames))
    level_frames.append(("path", 0, n_path_frames))
    level_frames.append(("hold_path", 0, hold_frames))

    frame_plan: list[tuple[str, int, float]] = []
    for kind, bits, n in level_frames:
        if kind == "curve":
            for k in range(1, n + 1):
                frame_plan.append((kind, bits, k / n))
        elif kind == "hold":
            for _ in range(n):
                frame_plan.append((kind, bits, 1.0))
        elif kind == "path":
            for k in range(1, n + 1):
                frame_plan.append((kind, bits, k / n))
        else:  # hold_path
            for _ in range(n):
                frame_plan.append((kind, bits, 1.0))

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(box_min[0], box_min[0] + box_size)
    ax.set_ylim(box_min[1], box_min[1] + box_size)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    if title is None:
        title = "Morton Z-order construction"
    title_artist = ax.set_title(title)

    # Background particles (dim) always visible in path phase.
    scat = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c="0.75",
        s=18,
        linewidths=0,
        zorder=2,
        alpha=0.0,
    )
    scat_ranked = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=rank,
        s=22,
        cmap="viridis",
        linewidths=0,
        zorder=3,
        alpha=0.0,
    )
    curve_line, = ax.plot([], [], "-", color="C3", lw=1.8, alpha=0.9, zorder=4)
    path_line, = ax.plot([], [], "-", color="C0", lw=1.4, alpha=0.95, zorder=5)
    cursor, = ax.plot([], [], "o", color="C1", ms=7, zorder=6)

    # Bit-interleave caption (static).
    ax.text(
        0.02,
        0.98,
        "key = interleave(x_bits, y_bits)\n"
        "same idea as 3D Morton (x|y|z) used for MPI domains",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="0.25",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85),
    )

    def _set_curve(bits: int, frac: float) -> None:
        poly = curves[bits]
        n = max(1, int(round(frac * (len(poly) - 1))) + 1)
        curve_line.set_data(poly[:n, 0], poly[:n, 1])
        cursor.set_data([poly[n - 1, 0]], [poly[n - 1, 1]])

    def _set_path(frac: float) -> None:
        n = max(1, int(round(frac * (len(xy_sorted) - 1))) + 1)
        path_line.set_data(xy_sorted[:n, 0], xy_sorted[:n, 1])
        cursor.set_data([xy_sorted[n - 1, 0]], [xy_sorted[n - 1, 1]])

    def update(frame_idx: int):
        kind, bits, frac = frame_plan[frame_idx]
        if kind in ("curve", "hold"):
            scat.set_alpha(0.15)
            scat_ranked.set_alpha(0.0)
            path_line.set_data([], [])
            curve_line.set_alpha(0.9)
            _set_curve(bits, frac)
            nside = 1 << bits
            title_artist.set_text(
                f"Morton Z-curve refinement — {nside}×{nside} (bits={bits})"
            )
        else:
            scat.set_alpha(0.25)
            scat_ranked.set_alpha(0.9)
            # Show finest construction curve faintly underneath.
            fine = curves[max(curve_bits)]
            curve_line.set_data(fine[:, 0], fine[:, 1])
            curve_line.set_alpha(0.25)
            _set_path(frac)
            title_artist.set_text(
                f"Visit particles in Morton order — {int(round(frac * 100))}%"
            )
        return curve_line, path_line, cursor, scat, scat_ranked, title_artist

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_plan),
        interval=1000 / max(fps, 1),
        blit=False,
    )
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


def write_morton_order_static(
    pos: np.ndarray,
    out_path: str | Path,
    *,
    n_particles: int = 200,
    bits_overlay: int = 4,
    seed: int = 0,
    dpi: int = 160,
) -> Path:
    """
    Static companion figure: particles coloured by Morton rank + Z-curve overlay.

    Parameters
    ----------
    pos : ndarray, shape (N, 2) or (N, 3)
        Particle positions.
    out_path : path-like
        Destination PNG path.
    n_particles : int
        Subsample size.
    bits_overlay : int
        Resolution of the background Z-curve.
    seed, dpi : int
        RNG seed and PNG DPI.

    Returns
    -------
    path : Path
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    xy = _subsample_xy(pos, n_particles, rng)
    box_min = xy.min(axis=0)
    box_size = float((xy.max(axis=0) - box_min).max()) or 1.0
    pad = 0.04 * box_size
    box_min = box_min - pad
    box_size = box_size + 2.0 * pad

    keys = morton_keys_2d(xy, box_min=box_min, box_size=box_size)
    order = np.argsort(keys, kind="stable")
    rank = np.empty(len(xy), dtype=float)
    rank[order] = np.linspace(0.0, 1.0, len(xy))
    curve = box_min + box_size * morton_curve_cell_centers(bits_overlay)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.plot(curve[:, 0], curve[:, 1], "-", color="0.7", lw=1.0, zorder=1)
    ax.plot(xy[order, 0], xy[order, 1], "-", color="C0", lw=0.9, alpha=0.7, zorder=2)
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=rank, s=28, cmap="viridis", linewidths=0, zorder=3
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Morton visit rank")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"Morton order on particles + {1 << bits_overlay}×{1 << bits_overlay} Z-curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
