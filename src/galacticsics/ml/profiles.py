"""Radial / cylindrical / azimuthal profile helpers for generative losses.

These utilities turn particle phase-space samples into coarse summaries
(surface density, spherical density, mean azimuthal velocity, and soft
azimuthal Fourier modes) that can be matched between data and model
reconstructions. Soft binning keeps gradients flowing into decoded
positions and velocities.

Axisymmetric auxiliaries (``Σ``, ``ρ``, ``⟨v_φ⟩``) do not constrain bars or
spirals; the Fourier terms ``A_m / A_0`` (and their cos/sin parts) reinforce
non-axisymmetric feature reconstruction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


def cylindrical_radius(pos: np.ndarray) -> np.ndarray:
    """
    Cylindrical radius ``R = sqrt(x^2 + y^2)``.

    Parameters
    ----------
    pos : ndarray, shape (..., 3)
        Cartesian positions [kpc].

    Returns
    -------
    R : ndarray, shape (...,)
        Midplane cylindrical radii [kpc].
    """
    pos = np.asarray(pos, dtype=np.float64)
    return np.hypot(pos[..., 0], pos[..., 1])


def spherical_radius(pos: np.ndarray) -> np.ndarray:
    """
    Spherical radius ``r = ||x||``.

    Parameters
    ----------
    pos : ndarray, shape (..., 3)
        Cartesian positions [kpc].

    Returns
    -------
    r : ndarray, shape (...,)
        3-D radii [kpc].
    """
    pos = np.asarray(pos, dtype=np.float64)
    return np.linalg.norm(pos, axis=-1)


def v_phi_cylindrical(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """
    Azimuthal velocity ``v_φ = (-y v_x + x v_y) / R`` in the midplane sense.

    Parameters
    ----------
    pos, vel : ndarray, shape (..., 3)
        Positions [kpc] and velocities [code units].

    Returns
    -------
    v_phi : ndarray, shape (...,)
        Azimuthal speed; ``0`` where ``R`` is below ``1e-8`` kpc.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    x, y = pos[..., 0], pos[..., 1]
    R = np.hypot(x, y)
    out = np.zeros_like(R)
    np.divide(-y * vel[..., 0] + x * vel[..., 1], R, out=out, where=R > 1e-8)
    return out


def numpy_mass_weighted_profile(
    radius: np.ndarray,
    values: np.ndarray | None,
    mass: np.ndarray,
    *,
    n_bins: int = 12,
    r_max: float = 15.0,
    density: bool = False,
    min_count: int = 0,
    empty: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hard-binned mass-weighted radial profile (numpy; for diagnostics / targets).

    Parameters
    ----------
    radius : ndarray, shape (N,)
        Per-particle radii (cylindrical or spherical) [kpc].
    values : ndarray, shape (N,), optional
        Quantity to average (e.g. ``v_φ``). When ``None``, returns mass density
        (surface or volume depending on ``density``).
    mass : ndarray, shape (N,)
        Particle masses.
    n_bins : int
        Number of equal-width bins on ``[0, r_max]``.
    r_max : float
        Outer bin edge [kpc].
    density : bool
        If ``True`` and ``values is None``, divide mass by annular area
        ``π(R_{i+1}^2 - R_i^2)`` (cylindrical surface density). Spherical
        volume density should be computed by the caller with a different
        normalisation if needed.
    min_count : int
        Require at least this many particles for a finite mean of ``values``.
        Density / mass histograms ignore this (empty → 0).
    empty : float, optional
        Fill value for empty / under-populated **mean** bins. Default ``NaN``
        when averaging ``values`` (do **not** use 0 — that makes ⟨v⟩ appear to
        drop in the outer disk as ``N`` decreases). Density bins still use 0.

    Returns
    -------
    r_mid : ndarray, shape (n_bins,)
        Bin centers [kpc].
    profile : ndarray, shape (n_bins,)
        Mass density or mass-weighted mean of ``values``.
    """
    radius = np.asarray(radius, dtype=np.float64).ravel()
    mass = np.asarray(mass, dtype=np.float64).ravel()
    edges = np.linspace(0.0, float(r_max), int(n_bins) + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    which = np.digitize(radius, edges) - 1
    averaging = values is not None
    fill = float("nan") if empty is None else float(empty)
    if averaging:
        profile = np.full(int(n_bins), fill, dtype=np.float64)
    else:
        profile = np.zeros(int(n_bins), dtype=np.float64)
    for i in range(int(n_bins)):
        m = which == i
        n_bin = int(np.count_nonzero(m))
        if n_bin == 0:
            continue
        w = mass[m]
        wsum = float(w.sum())
        if wsum <= 0:
            continue
        if values is None:
            if density:
                area = np.pi * (edges[i + 1] ** 2 - edges[i] ** 2)
                profile[i] = wsum / max(area, 1e-30)
            else:
                profile[i] = wsum
        else:
            if n_bin < max(int(min_count), 1):
                continue
            profile[i] = float(
                np.average(np.asarray(values, dtype=np.float64).ravel()[m], weights=w)
            )
    return r_mid, profile


def soft_radial_histogram(
    radius: "torch.Tensor",
    weights: "torch.Tensor",
    *,
    n_bins: int = 12,
    r_max: float = 15.0,
    soft_width: float | None = None,
) -> "torch.Tensor":
    """
    Differentiable soft mass histogram along a 1-D radius.

    Each particle contributes a Gaussian (in radius) of width ``soft_width``
    centered on its radius; contributions are normalised across bins so the
    total weight is conserved to first order.

    Parameters
    ----------
    radius : Tensor, shape (B, N)
        Per-particle radii [kpc].
    weights : Tensor, shape (B, N)
        Non-negative masses or component soft-weights × mass.
    n_bins : int
        Number of bins on ``[0, r_max]``.
    r_max : float
        Outer radius [kpc].
    soft_width : float, optional
        Gaussian σ [kpc]. Default ``0.5 * bin_width``.

    Returns
    -------
    hist : Tensor, shape (B, n_bins)
        Soft-binned weight per bin (same units as ``weights``).
    """
    import torch

    b, n = radius.shape
    edges = torch.linspace(0.0, float(r_max), int(n_bins) + 1, device=radius.device, dtype=radius.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(r_max) / max(int(n_bins), 1)
    sigma = float(soft_width) if soft_width is not None else 0.5 * width
    # (B, N, 1) vs (1, 1, n_bins)
    r = radius.unsqueeze(-1)
    c = centers.view(1, 1, -1)
    logits = -0.5 * ((r - c) / max(sigma, 1e-6)) ** 2
    # Zero weight outside the radial window. Do NOT fill logits with -1e4:
    # particles with all bins masked yield softmax(−∞,…)=NaN.
    inside = (radius >= 0) & (radius <= float(r_max) * 1.05)
    w = weights * inside.to(weights.dtype)
    attn = torch.softmax(logits, dim=-1)  # (B, N, n_bins)
    hist = (attn * w.unsqueeze(-1)).sum(dim=1)
    return hist


def soft_surface_density(
    pos: "torch.Tensor",
    mass: "torch.Tensor",
    *,
    n_bins: int = 12,
    r_max: float = 15.0,
    component_weight: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """
    Soft cylindrical surface density ``Σ(R)`` [mass / kpc²].

    Parameters
    ----------
    pos : Tensor, shape (B, N, 3)
        Positions [kpc].
    mass : Tensor, shape (B, N)
        Particle masses.
    n_bins, r_max
        Radial grid.
    component_weight : Tensor, shape (B, N), optional
        Soft component membership in ``[0, 1]`` (e.g. softmax probability).

    Returns
    -------
    sigma : Tensor, shape (B, n_bins)
        Soft ``Σ`` in each annulus.
    """
    import torch

    R = torch.sqrt(pos[..., 0] ** 2 + pos[..., 1] ** 2 + 1e-16)
    w = mass if component_weight is None else mass * component_weight
    hist = soft_radial_histogram(R, w, n_bins=n_bins, r_max=r_max)
    edges = torch.linspace(0.0, float(r_max), int(n_bins) + 1, device=pos.device, dtype=pos.dtype)
    area_t = (edges[1:] ** 2 - edges[:-1] ** 2) * np.pi
    return hist / area_t.clamp_min(1e-30)


def soft_spherical_density(
    pos: "torch.Tensor",
    mass: "torch.Tensor",
    *,
    n_bins: int = 12,
    r_max: float = 20.0,
    component_weight: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """
    Soft spherical density ``ρ(r)`` [mass / kpc³].

    Parameters
    ----------
    pos : Tensor, shape (B, N, 3)
    mass : Tensor, shape (B, N)
    n_bins, r_max
        Radial grid.
    component_weight : Tensor, shape (B, N), optional
        Soft component weights.

    Returns
    -------
    rho : Tensor, shape (B, n_bins)
    """
    import torch

    r = torch.linalg.norm(pos, dim=-1).clamp_min(1e-8)
    w = mass if component_weight is None else mass * component_weight
    hist = soft_radial_histogram(r, w, n_bins=n_bins, r_max=r_max)
    edges = torch.linspace(0.0, float(r_max), int(n_bins) + 1, device=pos.device, dtype=pos.dtype)
    vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return hist / vol.clamp_min(1e-30)


def soft_mean_vphi(
    pos: "torch.Tensor",
    vel: "torch.Tensor",
    mass: "torch.Tensor",
    *,
    n_bins: int = 12,
    r_max: float = 15.0,
    component_weight: "torch.Tensor | None" = None,
    min_mass: float | None = None,
) -> "torch.Tensor":
    """
    Soft mass-weighted mean ``⟨v_φ⟩(R)`` in cylindrical annuli.

    Parameters
    ----------
    pos, vel : Tensor, shape (B, N, 3)
    mass : Tensor, shape (B, N)
    n_bins, r_max
        Radial grid.
    component_weight : Tensor, shape (B, N), optional
    min_mass : float, optional
        Soft-bin mass floor. Bins below this are set to NaN so empty outer
        annuli do not report ``⟨v_φ⟩→0`` (which biases low-``N`` profiles).
        Default: ``1e-6 * mass.sum() / n_bins`` per batch row.

    Returns
    -------
    vphi_profile : Tensor, shape (B, n_bins)
    """
    import torch

    x, y = pos[..., 0], pos[..., 1]
    R = torch.sqrt(x * x + y * y + 1e-16)
    vphi = (-y * vel[..., 0] + x * vel[..., 1]) / R
    w = mass if component_weight is None else mass * component_weight
    num = soft_radial_histogram(R, w * vphi, n_bins=n_bins, r_max=r_max)
    den = soft_radial_histogram(R, w, n_bins=n_bins, r_max=r_max)
    out = num / den.clamp_min(1e-30)
    if min_mass is None:
        # Fraction of mean mass per bin — scales with subsample size.
        floor = (w.sum(dim=-1, keepdim=True) / max(int(n_bins), 1)) * 1e-6
    else:
        floor = den.new_full((), float(min_mass))
    return torch.where(den > floor, out, out.new_full((), float("nan")))


def soft_midplane_weight(
    pos: "torch.Tensor",
    *,
    z_max: float = 0.5,
    soft_z: float | None = None,
) -> "torch.Tensor":
    """
    Smooth midplane membership for disk Fourier diagnostics.

    Parameters
    ----------
    pos : Tensor, shape (B, N, 3)
        Positions [kpc].
    z_max : float
        Soft half-thickness [kpc]; weight → 1 for ``|z| ≪ z_max``.
    soft_z : float, optional
        Sigmoid width [kpc]. Default ``0.1 * z_max``.

    Returns
    -------
    weight : Tensor, shape (B, N)
        Values in ``(0, 1)``.
    """
    import torch

    width = float(soft_z) if soft_z is not None else 0.1 * max(float(z_max), 1e-3)
    return torch.sigmoid((float(z_max) - pos[..., 2].abs()) / max(width, 1e-6))


def soft_azimuthal_fourier(
    pos: "torch.Tensor",
    mass: "torch.Tensor",
    *,
    m: int = 2,
    n_bins: int = 12,
    r_max: float = 15.0,
    component_weight: "torch.Tensor | None" = None,
    z_max: float | None = 0.5,
) -> dict[str, "torch.Tensor"]:
    """
    Soft radial profiles of azimuthal Fourier moments ``a_m(R)``.

    For each annulus the complex moment is

    .. math::

        a_m = \\sum_i w_i e^{i m \\phi_i}, \\qquad
        A_m/A_0 = |a_m| / a_0

    with soft radial (and optional midplane) weights. Matching both
    ``Re(a_m)/a_0`` and ``Im(a_m)/a_0`` is phase-aware (bar angle), not only
    the amplitude used in campaign ``A2/A0`` diagnostics.

    Parameters
    ----------
    pos : Tensor, shape (B, N, 3)
        Positions [kpc].
    mass : Tensor, shape (B, N)
        Particle masses.
    m : int
        Azimuthal mode (``1`` lopsidedness, ``2`` bar/spiral).
    n_bins, r_max
        Soft cylindrical radial grid.
    component_weight : Tensor, shape (B, N), optional
        Soft component membership.
    z_max : float or None
        Midplane soft cut [kpc]; ``None`` keeps all ``z``.

    Returns
    -------
    dict of Tensor
        ``a0`` (B, n_bins), ``cos`` / ``sin`` = ``Re/Im(a_m)/a_0``,
        ``amp`` = ``|a_m|/a_0``.
    """
    import torch

    x, y = pos[..., 0], pos[..., 1]
    R = torch.sqrt(x * x + y * y + 1e-16)
    phi = torch.atan2(y, x)
    w = mass if component_weight is None else mass * component_weight
    if z_max is not None:
        w = w * soft_midplane_weight(pos, z_max=float(z_max))
    a0 = soft_radial_histogram(R, w, n_bins=n_bins, r_max=r_max).clamp_min(1e-30)
    m_f = float(m)
    cos_m = soft_radial_histogram(R, w * torch.cos(m_f * phi), n_bins=n_bins, r_max=r_max)
    sin_m = soft_radial_histogram(R, w * torch.sin(m_f * phi), n_bins=n_bins, r_max=r_max)
    cos_n = cos_m / a0
    sin_n = sin_m / a0
    amp = torch.sqrt(cos_n * cos_n + sin_n * sin_n + 1e-16)
    return {"a0": a0, "cos": cos_n, "sin": sin_n, "amp": amp}


def nonaxisym_reconstruction_loss(
    pos_data: "torch.Tensor",
    c_data: "torch.Tensor",
    pos_pred: "torch.Tensor",
    logits_c: "torch.Tensor",
    *,
    modes: tuple[int, ...] = (1, 2),
    n_bins: int = 12,
    r_max_disk: float = 15.0,
    z_max: float = 0.5,
    lambda_am: float = 0.1,
    lambda_phase: float = 1.0,
) -> dict[str, "torch.Tensor"]:
    """
    Soft Fourier losses that reinforce non-axisymmetric disk structure.

    For each mode ``m`` matches normalised ``(cos, sin)`` profiles (phase-aware)
    and optionally the amplitude ``|a_m|/a_0`` between data and decode. Disk
    particles only (hard labels on data, soft membership on predictions).

    Parameters
    ----------
    pos_data : Tensor, shape (B, N, 3)
    c_data : Tensor, shape (B, N)
    pos_pred : Tensor, shape (B, N, 3)
    logits_c : Tensor, shape (B, N, 3)
    modes : tuple of int
        Azimuthal modes to match (default ``(1, 2)``).
    n_bins, r_max_disk, z_max
        Soft radial / midplane grids for the disk Fourier estimate.
    lambda_am : float
        Overall weight applied outside this function; here scales the sum of
        per-mode terms when returned as ``nonaxisym``.
    lambda_phase : float
        Relative weight of cos/sin MSE vs amplitude MSE within each mode
        (``1`` → equal emphasis on complex parts and ``|a_m|/a_0``).

    Returns
    -------
    dict
        ``nonaxisym`` (weighted sum over modes), plus ``am{m}`` per mode.
    """
    import torch
    import torch.nn.functional as F

    b, n, _ = pos_data.shape
    mass = torch.full((b, n), 1.0 / max(n, 1), device=pos_data.device, dtype=pos_data.dtype)
    soft_c = F.softmax(logits_c, dim=-1)
    w_disk_d = (c_data.long() == 0).to(pos_data.dtype)
    w_disk_p = soft_c[..., 0]

    total = pos_data.new_zeros(())
    out: dict[str, "torch.Tensor"] = {}
    for m in modes:
        m = int(m)
        d = soft_azimuthal_fourier(
            pos_data,
            mass,
            m=m,
            n_bins=n_bins,
            r_max=r_max_disk,
            component_weight=w_disk_d,
            z_max=z_max,
        )
        p = soft_azimuthal_fourier(
            pos_pred,
            mass,
            m=m,
            n_bins=n_bins,
            r_max=r_max_disk,
            component_weight=w_disk_p,
            z_max=z_max,
        )
        loss_phase = F.mse_loss(p["cos"], d["cos"]) + F.mse_loss(p["sin"], d["sin"])
        loss_amp = F.mse_loss(p["amp"], d["amp"])
        loss_m = float(lambda_phase) * loss_phase + loss_amp
        out[f"am{m}"] = loss_m
        total = total + loss_m
    out["nonaxisym"] = float(lambda_am) * total
    return out


def soft_plane_density_map(
    pos: "torch.Tensor",
    mass: "torch.Tensor",
    *,
    axes: tuple[int, int] = (0, 1),
    n_pix: int = 32,
    r_max: float = 15.0,
    soft_width: float | None = None,
    component_weight: "torch.Tensor | None" = None,
    midplane_z_max: float | None = None,
) -> "torch.Tensor":
    """
    Soft 2-D mass map by Gaussian splat onto a Cartesian pixel grid.

    Each particle distributes its weight across pixels with a normalised
    Gaussian kernel (mass-conserving softmax over the ``n_pix²`` grid).  Use
    ``axes=(0, 1)`` for face-on ``Σ(x, y)`` and ``axes=(0, 2)`` for edge-on
    ``Σ(x, z)``.

    Parameters
    ----------
    pos : Tensor, shape (B, N, 3)
        Positions [kpc].
    mass : Tensor, shape (B, N)
        Particle masses.
    axes : tuple of int
        Coordinate axes for the map plane.
    n_pix : int
        Pixels per side (keep ≤32 under low-VRAM training).
    r_max : float
        Half-width of the square map [kpc]; domain ``[-r_max, r_max]²``.
    soft_width : float, optional
        Gaussian σ [kpc]. Default ``0.75 * pixel_width``.
    component_weight : Tensor, shape (B, N), optional
        Soft component membership.
    midplane_z_max : float, optional
        If set, multiply by :func:`soft_midplane_weight` (useful for face-on).

    Returns
    -------
    dens : Tensor, shape (B, n_pix, n_pix)
        Soft deposited mass per pixel (not divided by area; ``log1p`` loss
        absorbs the constant pixel area).
    """
    import torch

    ax0, ax1 = int(axes[0]), int(axes[1])
    w = mass if component_weight is None else mass * component_weight
    if midplane_z_max is not None:
        w = w * soft_midplane_weight(pos, z_max=float(midplane_z_max))
    u = pos[..., ax0]
    v = pos[..., ax1]
    coords = torch.linspace(
        -float(r_max), float(r_max), int(n_pix), device=pos.device, dtype=pos.dtype
    )
    pixel = (2.0 * float(r_max)) / max(int(n_pix), 1)
    sigma = float(soft_width) if soft_width is not None else 0.75 * pixel
    # (B, N, 1, 1) vs (1, 1, H, 1) / (1, 1, 1, W)
    du = u.unsqueeze(-1).unsqueeze(-1) - coords.view(1, 1, -1, 1)
    dv = v.unsqueeze(-1).unsqueeze(-1) - coords.view(1, 1, 1, -1)
    logits = -0.5 * ((du / max(sigma, 1e-6)) ** 2 + (dv / max(sigma, 1e-6)) ** 2)
    # Avoid masked_fill(-1e4): fully-outside particles → softmax NaN.
    inside = (u.abs() <= float(r_max) * 1.05) & (v.abs() <= float(r_max) * 1.05)
    w = w * inside.to(w.dtype)
    flat = logits.reshape(logits.shape[0], logits.shape[1], -1)
    attn = torch.softmax(flat, dim=-1).reshape_as(logits)
    return (attn * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)


def map_reconstruction_loss(
    pos_data: "torch.Tensor",
    c_data: "torch.Tensor",
    pos_pred: "torch.Tensor",
    logits_c: "torch.Tensor",
    *,
    n_pix: int = 32,
    r_max: float = 15.0,
    z_max_face: float = 0.5,
    lambda_xy: float = 0.1,
    lambda_xz: float = 0.05,
) -> dict[str, "torch.Tensor"]:
    """
    Soft face-on / edge-on map MSE for non-axisymmetric disk structure.

    Matches ``log1p`` of soft ``Σ(x,y)`` and ``Σ(x,z)`` between data and
    decode (disk component). Complementary to radial Fourier: maps see
    localised overdensities without projecting onto a single ``m``.

    Parameters
    ----------
    pos_data, pos_pred : Tensor, shape (B, N, 3)
    c_data : Tensor, shape (B, N)
    logits_c : Tensor, shape (B, N, 3)
    n_pix, r_max
        Map grid.
    z_max_face : float
        Midplane soft cut for the face-on map.
    lambda_xy, lambda_xz : float
        Relative weights inside the returned ``maps`` total (outer
        ``lambda_maps`` in :func:`profile_reconstruction_loss` still applies).

    Returns
    -------
    dict
        ``maps``, ``map_xy``, ``map_xz`` scalar losses.
    """
    import torch
    import torch.nn.functional as F

    b, n, _ = pos_data.shape
    mass = torch.full((b, n), 1.0 / max(n, 1), device=pos_data.device, dtype=pos_data.dtype)
    soft_c = F.softmax(logits_c, dim=-1)
    w_d = (c_data.long() == 0).to(pos_data.dtype)
    w_p = soft_c[..., 0]

    xy_d = soft_plane_density_map(
        pos_data,
        mass,
        axes=(0, 1),
        n_pix=n_pix,
        r_max=r_max,
        component_weight=w_d,
        midplane_z_max=z_max_face,
    )
    xy_p = soft_plane_density_map(
        pos_pred,
        mass,
        axes=(0, 1),
        n_pix=n_pix,
        r_max=r_max,
        component_weight=w_p,
        midplane_z_max=z_max_face,
    )
    xz_d = soft_plane_density_map(
        pos_data, mass, axes=(0, 2), n_pix=n_pix, r_max=r_max, component_weight=w_d
    )
    xz_p = soft_plane_density_map(
        pos_pred, mass, axes=(0, 2), n_pix=n_pix, r_max=r_max, component_weight=w_p
    )
    loss_xy = F.mse_loss(torch.log1p(xy_p), torch.log1p(xy_d))
    loss_xz = F.mse_loss(torch.log1p(xz_p), torch.log1p(xz_d))
    return {
        "maps": float(lambda_xy) * loss_xy + float(lambda_xz) * loss_xz,
        "map_xy": loss_xy,
        "map_xz": loss_xz,
    }


def _stratified_subsample_indices(
    c: "torch.Tensor",
    n_sub: int,
) -> "torch.Tensor":
    """
    Stratified particle indices for virial / force diagnostics.

    Parameters
    ----------
    c : Tensor, shape (N,)
        Integer component labels ``{0,1,2}``.
    n_sub : int
        Target subsample size (capped at ``N``).

    Returns
    -------
    idx : Tensor, shape (M,)
        Indices with roughly equal draws per present component.
    """
    import torch

    n = int(c.shape[0])
    m = min(int(n_sub), n)
    if m >= n:
        return torch.arange(n, device=c.device)
    parts: list[torch.Tensor] = []
    present = []
    for k in range(3):
        idx_k = torch.nonzero(c.long() == k, as_tuple=False).view(-1)
        if idx_k.numel() > 0:
            present.append(idx_k)
    if not present:
        return torch.randperm(n, device=c.device)[:m]
    quota = max(1, m // len(present))
    for idx_k in present:
        take = min(quota, int(idx_k.numel()))
        if take >= int(idx_k.numel()):
            parts.append(idx_k)
        else:
            sel = torch.randperm(idx_k.numel(), device=c.device)[:take]
            parts.append(idx_k[sel])
    idx = torch.cat(parts, dim=0)
    if idx.numel() < m:
        # Fill remainder uniformly from leftovers
        mask = torch.ones(n, dtype=torch.bool, device=c.device)
        mask[idx] = False
        rest = torch.nonzero(mask, as_tuple=False).view(-1)
        need = m - int(idx.numel())
        if rest.numel() > 0 and need > 0:
            extra = rest[torch.randperm(rest.numel(), device=c.device)[:need]]
            idx = torch.cat([idx, extra], dim=0)
    if idx.numel() > m:
        idx = idx[torch.randperm(idx.numel(), device=c.device)[:m]]
    return idx


def plummer_virial_stats(
    pos: "torch.Tensor",
    vel: "torch.Tensor",
    *,
    eps: float = 0.1,
    g: float = 1.0,
) -> dict[str, "torch.Tensor"]:
    """
    Differentiable Plummer-softened kinetic / potential / virial ratio.

    Uses equal mass ``1/M`` per particle in the (sub)set.  Matches the
    pairwise convention in :func:`ntropy.softening.softened_potential_energy`
    (``G=1`` GalactICS units) on the diagonal-excluded pair sum.

    Parameters
    ----------
    pos, vel : Tensor, shape (B, M, 3)
        Particle positions [kpc] and velocities [code units].
    eps : float
        Constant Plummer softening [kpc].
    g : float
        Gravitational constant (``1`` in GalactICS / ntropy units).

    Returns
    -------
    dict
        ``ke``, ``pe``, ``virial_ratio`` (``2K/|W|``), each shape ``(B,)``.
    """
    import torch

    b, m, _ = pos.shape
    if m < 2:
        z = pos.new_zeros(b)
        return {"ke": z, "pe": z, "virial_ratio": z + 1.0}
    mass = 1.0 / float(m)
    ke = 0.5 * mass * (vel * vel).sum(dim=-1).sum(dim=-1)
    # Pairwise r² with inf on the diagonal (exclude self)
    dr = pos.unsqueeze(2) - pos.unsqueeze(1)
    r2 = (dr * dr).sum(dim=-1)
    eye = torch.eye(m, device=pos.device, dtype=torch.bool)
    r2 = r2.masked_fill(eye.unsqueeze(0), float("inf"))
    inv_r = torch.rsqrt(r2 + float(eps) ** 2)
    # sum_{i≠j} → factor 1/2 recovers sum_{i<j}
    pe = -0.5 * float(g) * (mass * mass) * inv_r.sum(dim=(-1, -2))
    abs_pe = pe.abs().clamp_min(1e-8)
    ratio = (2.0 * ke) / abs_pe
    return {"ke": ke, "pe": pe, "virial_ratio": ratio}


def virial_consistency_loss(
    pos_data: "torch.Tensor",
    vel_data: "torch.Tensor",
    c_data: "torch.Tensor",
    pos_pred: "torch.Tensor",
    vel_pred: "torch.Tensor",
    *,
    n_sub: int = 256,
    eps: float = 0.1,
    lambda_ratio: float = 1.0,
    lambda_ke: float = 1.0,
    lambda_com: float = 0.5,
    lambda_target: float = 0.0,
    target_ratio: float = 1.0,
) -> dict[str, "torch.Tensor"]:
    """
    Soft dynamical-consistency loss via stratified-subsample virial stats.

    GalactICS ICs are equilibrium in the fixed ``dbh`` potential, so a hard
    self-gravity target ``2K/|W|≈1`` is only approximate.  Equal-mass pairwise
    ratios on real dumps are typically *hot* (≫1); matching that ratio blindly
    teaches unbound BH clouds.  Prefer a soft pull of the **predicted** ratio
    toward ``target_ratio≈1``, plus kinetic-scale and COM matching to data.

    Complexity is ``O(B · n_sub²)``; keep ``n_sub`` in ``128–512`` for train.

    Parameters
    ----------
    pos_data, vel_data, c_data
        Target cloud and component labels ``(B, N, …)``.
    pos_pred, vel_pred
        Reconstructed cloud (same indexing; teacher-forced ``c``).
    n_sub : int
        Stratified subsample size for the pairwise Plummer sum.
    eps : float
        Softening [kpc].
    lambda_ratio, lambda_ke, lambda_com : float
        Weights for log-ratio match, log-KE match, and COM/momentum.
    lambda_target, target_ratio : float
        Optional soft pull of the *predicted* ratio toward ``target_ratio``
        (useful when samples will be evolved under self-gravity).

    Returns
    -------
    dict
        ``virial`` (weighted sum), ``virial_ratio``, ``ke``, ``com``,
        ``ratio_data``, ``ratio_pred`` (batch-mean diagnostics).
    """
    import torch
    import torch.nn.functional as F

    b, n, _ = pos_data.shape
    m = min(int(n_sub), n)
    pos_d_parts: list[torch.Tensor] = []
    vel_d_parts: list[torch.Tensor] = []
    pos_p_parts: list[torch.Tensor] = []
    vel_p_parts: list[torch.Tensor] = []
    for bi in range(b):
        idx = _stratified_subsample_indices(c_data[bi], m)
        pos_d_parts.append(pos_data[bi, idx])
        vel_d_parts.append(vel_data[bi, idx])
        pos_p_parts.append(pos_pred[bi, idx])
        vel_p_parts.append(vel_pred[bi, idx])
    pos_d = torch.stack(pos_d_parts, dim=0)
    vel_d = torch.stack(vel_d_parts, dim=0)
    pos_p = torch.stack(pos_p_parts, dim=0)
    vel_p = torch.stack(vel_p_parts, dim=0)

    st_d = plummer_virial_stats(pos_d, vel_d, eps=eps)
    st_p = plummer_virial_stats(pos_p, vel_p, eps=eps)
    # Log-space ratio / KE matching is scale-stable across snapshot epochs
    loss_ratio = F.mse_loss(
        torch.log(st_p["virial_ratio"].clamp_min(1e-3)),
        torch.log(st_d["virial_ratio"].clamp_min(1e-3)),
    )
    loss_ke = F.mse_loss(
        torch.log1p(st_p["ke"].clamp_min(0.0)),
        torch.log1p(st_d["ke"].clamp_min(0.0)),
    )
    # COM / bulk velocity of the full predicted set (data assumed centered)
    com_x = pos_pred.mean(dim=1)
    com_v = vel_pred.mean(dim=1)
    loss_com = (com_x * com_x).mean() + (com_v * com_v).mean()
    loss_tgt = pos_pred.new_zeros(())
    if float(lambda_target) > 0.0:
        loss_tgt = F.mse_loss(
            torch.log(st_p["virial_ratio"].clamp_min(1e-3)),
            pos_pred.new_full((b,), float(np.log(max(target_ratio, 1e-3)))),
        )
    total = (
        float(lambda_ratio) * loss_ratio
        + float(lambda_ke) * loss_ke
        + float(lambda_com) * loss_com
        + float(lambda_target) * loss_tgt
    )
    return {
        "virial": total,
        "virial_ratio": loss_ratio,
        "ke": loss_ke,
        "com": loss_com,
        "virial_target": loss_tgt,
        "ratio_data": st_d["virial_ratio"].detach().mean(),
        "ratio_pred": st_p["virial_ratio"].detach().mean(),
    }


def profile_reconstruction_loss(
    pos_data: "torch.Tensor",
    vel_data: "torch.Tensor",
    c_data: "torch.Tensor",
    pos_pred: "torch.Tensor",
    vel_pred: "torch.Tensor",
    logits_c: "torch.Tensor",
    *,
    n_bins: int = 12,
    r_max_disk: float = 15.0,
    r_max_sph: float = 20.0,
    lambda_sigma: float = 0.1,
    lambda_rho: float = 0.1,
    lambda_vphi: float = 0.1,
    lambda_nonaxisym: float = 0.1,
    lambda_maps: float = 0.1,
    fourier_modes: tuple[int, ...] = (1, 2),
    fourier_z_max: float = 0.5,
    lambda_fourier_phase: float = 1.0,
    map_n_pix: int = 32,
    map_r_max: float | None = None,
) -> dict[str, "torch.Tensor"]:
    """
    Auxiliary MSE between soft density / kinematics / Fourier / map fields.

    Component ids: ``0=disk``, ``1=halo``, ``2=bulge``. Masses are taken as
    uniform ``1/N`` per particle (subsample tokens are unweighted beyond that).

    Axisymmetric terms (``Σ``, ``ρ``, ``⟨v_φ⟩``) are complemented by soft
    azimuthal Fourier matching and soft face-on / edge-on maps so bars and
    spirals are not washed out by radial-only objectives.

    Parameters
    ----------
    pos_data, vel_data : Tensor, shape (B, N, 3)
        Target particle state (absolute positions in ``dx`` tokens).
    c_data : Tensor, shape (B, N)
        Integer component labels.
    pos_pred, vel_pred : Tensor, shape (B, N, 3)
        Reconstructed state.
    logits_c : Tensor, shape (B, N, 3)
        Predicted component logits (softmax → soft weights).
    n_bins, r_max_disk, r_max_sph
        Profile grids.
    lambda_sigma, lambda_rho, lambda_vphi : float
        Weights for disk ``Σ``, halo+bulge ``ρ``, and disk ``⟨v_φ⟩``.
    lambda_nonaxisym : float
        Weight for the summed soft Fourier (non-axisymmetric) loss.
    lambda_maps : float
        Weight for soft face-on / edge-on map loss.
    fourier_modes : tuple of int
        Azimuthal modes to match (default ``(1, 2)``).
    fourier_z_max : float
        Soft midplane cut [kpc] for Fourier estimates.
    lambda_fourier_phase : float
        Relative weight of cos/sin vs amplitude within each mode.
    map_n_pix : int
        Pixels per side for map losses.
    map_r_max : float, optional
        Map half-width [kpc]; defaults to ``r_max_disk``.

    Returns
    -------
    dict
        ``profile`` (weighted sum), ``sigma``, ``rho``, ``vphi``,
        ``nonaxisym``, ``maps``, ``map_xy``, ``map_xz``, and per-mode ``am{m}``.
    """
    import torch
    import torch.nn.functional as F

    b, n, _ = pos_data.shape
    mass = torch.full((b, n), 1.0 / max(n, 1), device=pos_data.device, dtype=pos_data.dtype)
    soft_c = F.softmax(logits_c, dim=-1)
    # Disk surface density
    w_disk_d = (c_data.long() == 0).to(pos_data.dtype)
    sigma_d = soft_surface_density(
        pos_data, mass, n_bins=n_bins, r_max=r_max_disk, component_weight=w_disk_d
    )
    sigma_p = soft_surface_density(
        pos_pred, mass, n_bins=n_bins, r_max=r_max_disk, component_weight=soft_c[..., 0]
    )
    loss_sigma = F.mse_loss(torch.log1p(sigma_p), torch.log1p(sigma_d))

    # Halo + bulge spherical density (sum components 1 and 2)
    w_sph_d = ((c_data.long() == 1) | (c_data.long() == 2)).to(pos_data.dtype)
    rho_d = soft_spherical_density(
        pos_data, mass, n_bins=n_bins, r_max=r_max_sph, component_weight=w_sph_d
    )
    rho_p = soft_spherical_density(
        pos_pred,
        mass,
        n_bins=n_bins,
        r_max=r_max_sph,
        component_weight=soft_c[..., 1] + soft_c[..., 2],
    )
    loss_rho = F.mse_loss(torch.log1p(rho_p), torch.log1p(rho_d))

    vphi_d = soft_mean_vphi(
        pos_data, vel_data, mass, n_bins=n_bins, r_max=r_max_disk, component_weight=w_disk_d
    )
    vphi_p = soft_mean_vphi(
        pos_pred,
        vel_pred,
        mass,
        n_bins=n_bins,
        r_max=r_max_disk,
        component_weight=soft_c[..., 0],
    )
    # Mask empty soft bins (NaN) so outer-disk zeros do not dominate.
    vphi_ok = torch.isfinite(vphi_d) & torch.isfinite(vphi_p)
    if bool(vphi_ok.any()):
        loss_vphi = F.mse_loss(vphi_p[vphi_ok], vphi_d[vphi_ok])
    else:
        loss_vphi = vphi_p.new_zeros(())

    nax = nonaxisym_reconstruction_loss(
        pos_data,
        c_data,
        pos_pred,
        logits_c,
        modes=fourier_modes,
        n_bins=n_bins,
        r_max_disk=r_max_disk,
        z_max=fourier_z_max,
        lambda_am=1.0,
        lambda_phase=lambda_fourier_phase,
    )
    maps = map_reconstruction_loss(
        pos_data,
        c_data,
        pos_pred,
        logits_c,
        n_pix=map_n_pix,
        r_max=float(r_max_disk if map_r_max is None else map_r_max),
        z_max_face=fourier_z_max,
    )

    total = (
        float(lambda_sigma) * loss_sigma
        + float(lambda_rho) * loss_rho
        + float(lambda_vphi) * loss_vphi
        + float(lambda_nonaxisym) * nax["nonaxisym"]
        + float(lambda_maps) * maps["maps"]
    )
    out = {
        "profile": total,
        "sigma": loss_sigma,
        "rho": loss_rho,
        "vphi": loss_vphi,
        "nonaxisym": nax["nonaxisym"],
        "maps": maps["maps"],
        "map_xy": maps["map_xy"],
        "map_xz": maps["map_xz"],
    }
    for key, val in nax.items():
        if key.startswith("am"):
            out[key] = val
    return out
