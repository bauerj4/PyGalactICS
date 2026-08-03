"""Resample particles from reconstructed slice density / moment stacks."""

from __future__ import annotations

import numpy as np

from galacticsics.ml.fields.binning import (
    ComponentSliceGrid,
    MultiScaleSliceConfig,
    SliceMapConfig,
    z_edges_for_grid,
)
from galacticsics.ml.morton.tokenize import COMPONENT_IDS


def integrated_slab_mass(stack: np.ndarray, grid: ComponentSliceGrid) -> float:
    """Integrate dens·dA over slabs (no Δz); dens is slab Σ = mass/area."""
    stack = np.asarray(stack, dtype=np.float64)
    keys = grid.moment_keys
    dens_i = keys.index("dens")
    n_mom = len(keys)
    xy_edges = np.linspace(-float(grid.r_max), float(grid.r_max), int(grid.n_pix) + 1)
    dx = float(xy_edges[1] - xy_edges[0])
    total = 0.0
    for iz in range(int(grid.n_z)):
        dens = np.maximum(stack[iz * n_mom + dens_i], 0.0)
        total += float(dens.sum() * dx * dx)
    return total


def rescale_dens_channels_to_mass(
    maps: dict[str, np.ndarray],
    *,
    cfg: MultiScaleSliceConfig,
    mass_total_per_component: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    Scale each component's dens channels so ∫ dens dA matches ``mass_total``.

    Does **not** change spatial morphology (uniform gain). Prefer
    ``mass_total_per_component`` at resample time for particle masses; use this
    when dens *maps* must be mass-calibrated for viz / loss diagnostics.
    """
    out: dict[str, np.ndarray] = {}
    for g in cfg.grids:
        stack = np.asarray(maps[g.name], dtype=np.float64).copy()
        target = float(mass_total_per_component.get(g.name, 0.0))
        if target <= 0:
            out[g.name] = stack
            continue
        cur = integrated_slab_mass(stack, g)
        if cur <= 0:
            out[g.name] = stack
            continue
        scale = target / cur
        dens_i = g.moment_keys.index("dens")
        n_mom = len(g.moment_keys)
        for iz in range(int(g.n_z)):
            stack[iz * n_mom + dens_i] *= scale
        out[g.name] = stack
    return out


def dens_axisym_residual_rms(
    dens: np.ndarray,
) -> float:
    """RMS of dens − soft radial mean (pixel-radius bins)."""
    dens = np.asarray(dens, dtype=np.float64)
    n_pix = dens.shape[-1]
    yy, xx = np.mgrid[0:n_pix, 0:n_pix]
    cx = 0.5 * (n_pix - 1)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cx) ** 2)
    r_bin = np.clip(np.floor(rr).astype(np.int64), 0, n_pix - 1)
    dens = np.maximum(dens, 0.0)
    sums = np.bincount(r_bin.ravel(), weights=dens.ravel(), minlength=n_pix)
    counts = np.bincount(r_bin.ravel(), minlength=n_pix).astype(np.float64)
    means = sums / np.maximum(counts, 1.0)
    resid = dens - means[r_bin]
    return float(np.sqrt(np.mean(resid**2)))


def alpha_match_residual_power(
    dens_ae: np.ndarray,
    dens_ref: np.ndarray,
    *,
    alpha_min: float = 1.0,
    alpha_max: float = 4.0,
) -> float:
    """
    Choose α so RMS(α·resid_ae) ≈ RMS(resid_ref).

    Dump-aware (needs a reference dens, typically deposited). Caps to
    [alpha_min, alpha_max]. Returns 1.0 if AE residual is negligible.

    Note: AE dens often already has comparable residual *power* but wrong
    morphology — prefer ``alpha_match_map_a2`` when the goal is bar A₂.
    """
    rms_ae = dens_axisym_residual_rms(dens_ae)
    rms_ref = dens_axisym_residual_rms(dens_ref)
    if rms_ae <= 1e-30:
        return float(alpha_min)
    alpha = float(rms_ref / rms_ae)
    return float(np.clip(alpha, alpha_min, alpha_max))


def alpha_match_map_a2(
    dens_ae: np.ndarray,
    *,
    target_a2: float,
    r_max: float,
    alpha_min: float = 1.0,
    alpha_max: float = 3.5,
    n_bins: int = 12,
    n_search: int = 10,
    r_eval: float | None = None,
) -> float:
    """
    Binary-search α so midplane dens map A₂ ≈ ``target_a2``.

    Default metric is median ring A₂. When ``r_eval`` is set (e.g. disk scale
    length ``R_d``), match ``A₂(R=r_eval)`` instead.

    Operates on a single dens map (no particles). Caps to [alpha_min, alpha_max].
    """
    from galacticsics.ml.fields.autoencoder import dens_map_azimuthal_fourier_numpy
    from ntropy.analysis.disk_density import plane_density_azimuthal_fourier

    target = float(target_a2)
    lo, hi = float(alpha_min), float(alpha_max)

    def _a2(alpha: float) -> float:
        # Inline amplify on one map (avoid full multitower overhead).
        dens = np.asarray(dens_ae, dtype=np.float64)
        n_pix = dens.shape[-1]
        yy, xx = np.mgrid[0:n_pix, 0:n_pix]
        cx = 0.5 * (n_pix - 1)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cx) ** 2)
        r_bin = np.clip(np.floor(rr).astype(np.int64), 0, n_pix - 1)
        dens = np.maximum(dens, 0.0)
        sums = np.bincount(r_bin.ravel(), weights=dens.ravel(), minlength=n_pix)
        counts = np.bincount(r_bin.ravel(), minlength=n_pix).astype(np.float64)
        means = sums / np.maximum(counts, 1.0)
        boosted = np.maximum(means[r_bin] + float(alpha) * (dens - means[r_bin]), 0.0)
        if r_eval is not None:
            fout = plane_density_azimuthal_fourier(
                boosted,
                half_extent=float(r_max),
                m=2,
                n_bins=n_bins,
                r_max=min(float(r_max), 12.0),
                r_eval=float(r_eval),
            )
            return float(fout["a_m_over_a0_at_r"])
        return float(
            dens_map_azimuthal_fourier_numpy(
                boosted, m=2, n_bins=n_bins, r_max=float(r_max)
            )["a_m_over_a0_median"]
        )

    a_lo = _a2(lo)
    if a_lo >= target:
        return lo
    a_hi = _a2(hi)
    if a_hi <= target:
        return hi
    best = hi
    best_err = abs(a_hi - target)
    for _ in range(int(n_search)):
        mid = 0.5 * (lo + hi)
        a_mid = _a2(mid)
        err = abs(a_mid - target)
        if err < best_err:
            best_err = err
            best = mid
        if a_mid < target:
            lo = mid
        else:
            hi = mid
    return float(best)


def _xy_radius_kpc(n: int, r_max: float) -> np.ndarray:
    """Pixel cylindrical radius map [kpc] for an ``n×n`` face-on dens chart."""
    yy, xx = np.mgrid[0:n, 0:n]
    cx = 0.5 * (n - 1)
    scale = (2.0 * float(r_max)) / float(n)
    x = (xx - cx) * scale
    y = (yy - cx) * scale
    return np.sqrt(x * x + y * y)


def radial_gaussian_weight(
    dens_shape: tuple[int, ...],
    *,
    r_max: float,
    peak: float,
    sigma: float,
    floor: float = 0.0,
) -> np.ndarray:
    """
    Soft radial window peaked at ``peak`` (e.g. ``R_d``).

    Used to concentrate morph contrast near one disk scale length so the
    residual objective tracks ``A₂(R_d)`` rather than a washed global median.

    ``floor`` keeps a minimum weight outside the peak (0 = hard Gaussian;
    ~0.35–0.5 preserves extended bar while still preferring ``R_d``).
    """
    n = int(dens_shape[-1])
    R = _xy_radius_kpc(n, float(r_max))
    sig = max(float(sigma), 1e-3)
    g = np.exp(-0.5 * ((R - float(peak)) / sig) ** 2)
    fl = float(np.clip(floor, 0.0, 1.0))
    w = fl + (1.0 - fl) * g
    return w.astype(np.float64)


def sharpen_contrast_map(
    contrast: np.ndarray,
    *,
    r_max: float,
    amount: float,
    smooth_kpc: float = 1.5,
) -> np.ndarray:
    """
    Unsharp-mask a multiplicative contrast map for sharper bar edges.

    ``out = contrast + amount · (contrast − smooth_R(contrast))`` with a
    soft radial-bin smoother (same pixel-radius bins as axisym). Keeps the
    large-scale m=2 while restoring midplane edge contrast washed out by
    soft teacher AE morph charts. ``amount=0`` is a no-op.
    """
    amount = float(amount)
    if amount <= 0.0:
        return np.asarray(contrast, dtype=np.float64)
    contrast = np.asarray(contrast, dtype=np.float64)
    n = contrast.shape[-1]
    yy, xx = np.mgrid[0:n, 0:n]
    cx = 0.5 * (n - 1)
    # Physical smoothing width → pixel bins.
    pix_kpc = (2.0 * float(r_max)) / float(n)
    soft = max(1, int(round(float(smooth_kpc) / max(pix_kpc, 1e-6))))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cx) ** 2)
    r_bin = np.clip(np.floor(rr).astype(np.int64), 0, n - 1)
    # Box-smooth along radius by rebinning with a widened kernel: average
    # contrast in annuli of width ``soft`` pixels.
    n_ann = int(np.ceil(n / soft))
    ann = np.clip(r_bin // soft, 0, n_ann - 1)
    sums = np.bincount(ann.ravel(), weights=contrast.ravel(), minlength=n_ann)
    counts = np.bincount(ann.ravel(), minlength=n_ann).astype(np.float64)
    means = sums / np.maximum(counts, 1.0)
    smooth = means[ann]
    return contrast + amount * (contrast - smooth)


def alpha_match_residual_a2_rd(
    maps_f0: dict[str, np.ndarray],
    maps_morph: dict[str, np.ndarray],
    *,
    cfg: MultiScaleSliceConfig,
    target_a2: float,
    r_eval: float = 2.0,
    alpha_min: float = 0.5,
    alpha_max: float = 4.0,
    n_search: int = 10,
    mode: str = "m2",
    other_morph_alpha: float = 0.0,
    residual_scale: str = "multiplicative",
    factor_floor: float = 0.05,
    preserve_axisym: bool = True,
    r_weight_peak: float | None = None,
    r_weight_sigma: float | None = None,
    contrast_sharpen: float = 0.0,
    contrast_smooth_kpc: float = 1.5,
    r_weight_floor: float = 0.0,
) -> float:
    """
    Binary-search residual α so injected midplane map ``A₂(R=r_eval) ≈ target``.

    Uses the same ``inject_morph_residual_on_f0_dens`` path (axisym lock,
    Rd weight, sharpen) so the dial matches the particle IC objective.
    """
    from ntropy.analysis.disk_density import plane_density_azimuthal_fourier

    target = float(target_a2)
    lo, hi = float(alpha_min), float(alpha_max)
    g = cfg.grid_for("disk") if hasattr(cfg, "grid_for") else None
    if g is None:
        g = next(x for x in cfg.grids if x.name == "disk")
    dens_i = g.moment_keys.index("dens")
    n_mom = len(g.moment_keys)
    iz = int(g.n_z) // 2
    half = float(g.r_max)

    def _a2(alpha: float) -> float:
        fields = inject_morph_residual_on_f0_dens(
            maps_f0,
            maps_morph,
            cfg=cfg,
            components=("disk",),
            alpha=float(alpha),
            mode=str(mode),
            other_morph_alpha=float(other_morph_alpha),
            scale=str(residual_scale),
            preserve_axisym=bool(preserve_axisym),
            factor_floor=float(factor_floor),
            contrast_from_midplane=True,
            r_weight_peak=r_weight_peak,
            r_weight_sigma=r_weight_sigma,
            contrast_sharpen=float(contrast_sharpen),
            contrast_smooth_kpc=float(contrast_smooth_kpc),
            r_weight_floor=float(r_weight_floor),
        )
        dens = np.asarray(fields["disk"][iz * n_mom + dens_i], dtype=np.float64)
        fout = plane_density_azimuthal_fourier(
            dens,
            half_extent=half,
            m=2,
            n_bins=24,
            r_max=min(half, 12.0),
            r_eval=float(r_eval),
        )
        return float(fout["a_m_over_a0_at_r"])

    a_lo = _a2(lo)
    if a_lo >= target:
        return lo
    a_hi = _a2(hi)
    if a_hi <= target:
        return hi
    best = hi
    best_err = abs(a_hi - target)
    for _ in range(int(n_search)):
        mid = 0.5 * (lo + hi)
        a_mid = _a2(mid)
        err = abs(a_mid - target)
        if err < best_err:
            best_err = err
            best = mid
        if a_mid < target:
            lo = mid
        else:
            hi = mid
    return float(best)


def alpha_match_predicted_contrast_a2_rd(
    maps_f0: dict[str, np.ndarray],
    contrast_midplane: np.ndarray,
    *,
    cfg: MultiScaleSliceConfig,
    target_a2: float,
    r_eval: float = 2.0,
    alpha_min: float = 0.25,
    alpha_max: float = 4.0,
    n_search: int = 10,
    factor_floor: float = 0.05,
    preserve_axisym: bool = True,
) -> float:
    """Binary-search α so Phase-B injected map ``A₂(R=r_eval) ≈ target``."""
    from ntropy.analysis.disk_density import plane_density_azimuthal_fourier

    target = float(target_a2)
    lo, hi = float(alpha_min), float(alpha_max)
    g = cfg.grid_for("disk") if hasattr(cfg, "grid_for") else None
    if g is None:
        g = next(x for x in cfg.grids if x.name == "disk")
    dens_i = g.moment_keys.index("dens")
    n_mom = len(g.moment_keys)
    iz = int(g.n_z) // 2
    half = float(g.r_max)

    def _a2(alpha: float) -> float:
        fields = inject_predicted_contrast_on_f0_dens(
            maps_f0,
            contrast_midplane,
            cfg=cfg,
            components=("disk",),
            alpha=float(alpha),
            preserve_axisym=bool(preserve_axisym),
            factor_floor=float(factor_floor),
        )
        dens = np.asarray(fields["disk"][iz * n_mom + dens_i], dtype=np.float64)
        fout = plane_density_azimuthal_fourier(
            dens,
            half_extent=half,
            m=2,
            n_bins=24,
            r_max=min(half, 12.0),
            r_eval=float(r_eval),
        )
        return float(fout["a_m_over_a0_at_r"])

    a_lo = _a2(lo)
    if a_lo >= target:
        return lo
    a_hi = _a2(hi)
    if a_hi <= target:
        return hi
    best = hi
    best_err = abs(a_hi - target)
    for _ in range(int(n_search)):
        mid = 0.5 * (lo + hi)
        a_mid = _a2(mid)
        err = abs(a_mid - target)
        if err < best_err:
            best_err = err
            best = mid
        if a_mid < target:
            lo = mid
        else:
            hi = mid
    return float(best)


def _axisym_and_rbin(dens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (axisym, r_bin) for a dens map using soft pixel-radius bins."""
    dens = np.maximum(np.asarray(dens, dtype=np.float64), 0.0)
    n_pix = dens.shape[-1]
    yy, xx = np.mgrid[0:n_pix, 0:n_pix]
    cx = 0.5 * (n_pix - 1)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cx) ** 2)
    r_bin = np.clip(np.floor(rr).astype(np.int64), 0, n_pix - 1)
    sums = np.bincount(r_bin.ravel(), weights=dens.ravel(), minlength=n_pix)
    counts = np.bincount(r_bin.ravel(), minlength=n_pix).astype(np.float64)
    axisym = (sums / np.maximum(counts, 1.0))[r_bin]
    return axisym, r_bin


def m2_residual_field(
    resid: np.ndarray,
    *,
    r_max: float,
    n_bins: int = 16,
) -> np.ndarray:
    """
    Project a dens residual onto the m=2 azimuthal Fourier field.

    Reconstructs ``2·(c₂(R) cos 2φ + s₂(R) sin 2φ)`` in radial bins so
    non-bar residual (m≠2 spirals / noise) can be left unamplified.
    """
    resid = np.asarray(resid, dtype=np.float64)
    n = resid.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    cx = 0.5 * (n - 1)
    scale = (2.0 * float(r_max)) / float(n)
    x = (xx - cx) * scale
    y = (yy - cx) * scale
    R = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    edges = np.linspace(0.0, float(r_max), int(n_bins) + 1)
    out = np.zeros_like(resid)
    c2p = np.cos(2.0 * phi)
    s2p = np.sin(2.0 * phi)
    for i in range(int(n_bins)):
        mask = (R >= edges[i]) & (R < edges[i + 1])
        if int(mask.sum()) < 8:
            continue
        c = float(np.mean(resid[mask] * c2p[mask]))
        s = float(np.mean(resid[mask] * s2p[mask]))
        # factor 2 recovers real-space m=2 from single-sided cos/sin coeffs
        out[mask] = 2.0 * (c * c2p[mask] + s * s2p[mask])
    return out


def amplify_axisym_residual_dens(
    maps: dict[str, np.ndarray],
    *,
    cfg: MultiScaleSliceConfig,
    components: tuple[str, ...] = ("disk",),
    alpha: float = 2.0,
    clip_neg: bool = True,
    midplane_only: bool = False,
    midplane_n_slabs: int = 3,
    mode: str = "full",
    m2_n_bins: int = 16,
    other_alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Amplify non-axisymmetric dens residual: dens' = axisym + α·(dens − axisym).

    Generative-compatible morphology boost (no deposit oracle). α=1 is identity;
    α>1 strengthens bars/spirals already present in AE dens. Does not invent
    structure absent from the AE map.

    ``midplane_only``: amplify only the central ``midplane_n_slabs`` z-slabs
    (reduces vertical noise amplification on fragile systems).

    ``mode``:
      - ``full`` — amplify entire residual (legacy; fragile when AE residual
        cosine vs deposit is low, e.g. 906c4 ~0.60)
      - ``m2`` — amplify only the m=2 Fourier projection of the residual;
        leave other residual at ``other_alpha`` (default 1 = unchanged).
        Prefer this when full amplify invents unstable non-bar morph.
    """
    alpha = float(alpha)
    other_alpha = float(other_alpha)
    mode = str(mode).lower().strip()
    if mode not in ("full", "m2"):
        raise ValueError(f"unknown amplify mode={mode!r} (expected full|m2)")
    keep = set(components)
    out: dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float64).copy() for k, v in maps.items()}
    for g in cfg.grids:
        if g.name not in keep or g.name not in out:
            continue
        if abs(alpha - 1.0) < 1e-12 and abs(other_alpha - 1.0) < 1e-12:
            continue
        stack = out[g.name]
        dens_i = g.moment_keys.index("dens")
        n_mom = len(g.moment_keys)
        n_z = int(g.n_z)
        if midplane_only:
            half = max(1, int(midplane_n_slabs) // 2)
            iz0 = n_z // 2
            iz_list = range(max(0, iz0 - half), min(n_z, iz0 + half + 1))
        else:
            iz_list = range(n_z)
        for iz in iz_list:
            dens = np.maximum(stack[iz * n_mom + dens_i], 0.0)
            axisym, _r_bin = _axisym_and_rbin(dens)
            resid = dens - axisym
            if mode == "full":
                boosted = axisym + alpha * resid
            else:
                m2 = m2_residual_field(resid, r_max=float(g.r_max), n_bins=int(m2_n_bins))
                other = resid - m2
                boosted = axisym + alpha * m2 + other_alpha * other
            if clip_neg:
                boosted = np.maximum(boosted, 0.0)
            stack[iz * n_mom + dens_i] = boosted
        out[g.name] = stack
    return out


def copy_dens_channels(
    dens_from: dict[str, np.ndarray],
    moments_from: dict[str, np.ndarray],
    *,
    cfg: MultiScaleSliceConfig,
    components: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """
    Replace dens channels in ``moments_from`` with those from ``dens_from``.

    Practical recon recipe when a dump is available: deposited dens (true
    morphology / A₂) + teacher moments (phase space), optionally with shell /
    retained bulge. Closes the IC A₂ gap that pure AE dens under-predicts.
    """
    keep = set(components) if components is not None else None
    out: dict[str, np.ndarray] = {}
    for g in cfg.grids:
        base = np.asarray(moments_from[g.name], dtype=np.float64).copy()
        if keep is not None and g.name not in keep:
            out[g.name] = base
            continue
        src = np.asarray(dens_from[g.name], dtype=np.float64)
        dens_i = g.moment_keys.index("dens")
        n_mom = len(g.moment_keys)
        for iz in range(int(g.n_z)):
            sl = iz * n_mom + dens_i
            base[sl] = src[sl]
        out[g.name] = base
    return out


def _morph_contrast_delta(
    dens_m: np.ndarray,
    *,
    r_max: float,
    mode: str,
    m2_n_bins: int,
    other_morph_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(axisym_m, delta)`` for morph dens (full residual or m=2)."""
    dens_m = np.maximum(np.asarray(dens_m, dtype=np.float64), 0.0)
    axisym_m, _ = _axisym_and_rbin(dens_m)
    resid_m = dens_m - axisym_m
    if mode == "full":
        delta = resid_m
    else:
        m2 = m2_residual_field(
            resid_m, r_max=float(r_max), n_bins=int(m2_n_bins)
        )
        delta = m2 + float(other_morph_alpha) * (resid_m - m2)
    return axisym_m, delta


def _renormalize_axisym_rings(
    dens: np.ndarray,
    axisym_target: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """
    Scale each soft radial ring so azimuthal mean matches ``axisym_target``.

    Preserves relative azimuthal contrast while locking Σ_axisym(R) to f0.
    """
    dens = np.asarray(dens, dtype=np.float64)
    axisym_target = np.asarray(axisym_target, dtype=np.float64)
    axisym_cur, _ = _axisym_and_rbin(np.maximum(dens, 0.0))
    scale = np.ones_like(dens)
    ok = axisym_cur > eps
    scale[ok] = axisym_target[ok] / axisym_cur[ok]
    # Empty rings with target mass: fall back to pure axisym floor.
    empty = (~ok) & (axisym_target > eps)
    out = dens * scale
    out[empty] = axisym_target[empty]
    out[axisym_target <= eps] = 0.0
    return out


def inject_morph_residual_on_f0_dens(
    maps_f0: dict[str, np.ndarray],
    maps_morph: dict[str, np.ndarray],
    *,
    cfg: MultiScaleSliceConfig,
    components: tuple[str, ...] = ("disk",),
    alpha: float = 1.0,
    clip_neg: bool = True,
    midplane_only: bool = False,
    midplane_n_slabs: int = 3,
    mode: str = "m2",
    m2_n_bins: int = 16,
    keep_f0_residual: float = 1.0,
    other_morph_alpha: float = 0.0,
    scale: str = "multiplicative",
    preserve_axisym: bool = True,
    factor_floor: float = 0.05,
    contrast_from_midplane: bool = True,
    r_weight_peak: float | None = None,
    r_weight_sigma: float | None = None,
    contrast_sharpen: float = 0.0,
    contrast_smooth_kpc: float = 1.5,
    r_weight_floor: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Phase-A residual around GalactICS dens: inject morph residual onto ``f0``.

    ``scale="multiplicative"`` (default; required when dens unit scales differ)::

        dens' = axisym(Σ_f0) · max(1 + α · δ_morph / max(axisym(Σ_morph), ε), floor)
              + keep_f0_residual · (Σ_f0 − axisym)

    then (default) **ring-renormalize** so azimuthal mean(Σ') = axisym(Σ_f0).
    That transfers *relative* m=2 contrast without collapsing / truncating the
    GalactICS radial mass profile when clipping would otherwise zero anti-bar
    pixels and pile mass into the bar.

    ``scale="additive"`` uses absolute dens units::

        dens' = axisym(Σ_f0) + keep_f0_residual · (Σ_f0 − axisym) + α · δ_morph

    Prefer multiplicative when teacher/deposit dens is renormalized differently
    from the GalactICS IC deposit (common after AE denorm).

    With ``contrast_from_midplane=True`` (default), the morph contrast map is
    taken from the midplane slab and applied to every injected z-slab so the
    f0 vertical dens ladder is not independently warped by noisy AE slabs.

    Optional sharpness / Rd targeting (post-median→A₂(R_d) objective):
    - ``r_weight_peak`` / ``r_weight_sigma``: Gaussian radial window on the
      contrast / δ so bar power concentrates near one disk scale length.
    - ``contrast_sharpen``: unsharp-mask amount on the contrast map (teacher
      AE morph is soft; deposit needs little/none).

    Moments channels are copied from ``maps_f0`` unchanged.
    """
    alpha = float(alpha)
    keep_f0 = float(keep_f0_residual)
    other_a = float(other_morph_alpha)
    floor = float(factor_floor)
    mode = str(mode).lower().strip()
    scale = str(scale).lower().strip()
    if mode not in ("full", "m2"):
        raise ValueError(f"unknown inject mode={mode!r} (expected full|m2)")
    if scale not in ("multiplicative", "additive"):
        raise ValueError(f"unknown scale={scale!r} (expected multiplicative|additive)")
    if not (0.0 <= floor < 1.0):
        raise ValueError(f"factor_floor={floor!r} must be in [0, 1)")
    keep = set(components)
    out: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float64).copy() for k, v in maps_f0.items()
    }

    def _apply_contrast_shaping(
        contrast_or_delta: np.ndarray, *, r_max: float, is_contrast: bool
    ) -> np.ndarray:
        x = np.asarray(contrast_or_delta, dtype=np.float64)
        if is_contrast and float(contrast_sharpen) > 0.0:
            x = sharpen_contrast_map(
                x,
                r_max=float(r_max),
                amount=float(contrast_sharpen),
                smooth_kpc=float(contrast_smooth_kpc),
            )
        if r_weight_peak is not None and r_weight_sigma is not None:
            w = radial_gaussian_weight(
                x.shape,
                r_max=float(r_max),
                peak=float(r_weight_peak),
                sigma=float(r_weight_sigma),
                floor=float(r_weight_floor),
            )
            x = x * w
        return x

    for g in cfg.grids:
        if g.name not in keep or g.name not in out or g.name not in maps_morph:
            continue
        stack = out[g.name]
        morph = np.asarray(maps_morph[g.name], dtype=np.float64)
        dens_i = g.moment_keys.index("dens")
        n_mom = len(g.moment_keys)
        n_z = int(g.n_z)
        iz_mid = n_z // 2
        if midplane_only:
            half = max(1, int(midplane_n_slabs) // 2)
            iz_list = range(max(0, iz_mid - half), min(n_z, iz_mid + half + 1))
        else:
            iz_list = range(n_z)

        mid_factor = None
        mid_delta = None
        if contrast_from_midplane:
            sl_m = iz_mid * n_mom + dens_i
            axisym_m, delta_m = _morph_contrast_delta(
                morph[sl_m],
                r_max=float(g.r_max),
                mode=mode,
                m2_n_bins=int(m2_n_bins),
                other_morph_alpha=other_a,
            )
            if scale == "multiplicative":
                eps_m = 1e-12 * max(float(np.max(axisym_m)), 1e-30)
                contrast = delta_m / np.maximum(axisym_m, eps_m)
                contrast = _apply_contrast_shaping(
                    contrast, r_max=float(g.r_max), is_contrast=True
                )
                mid_factor = np.maximum(1.0 + alpha * contrast, floor)
            else:
                mid_delta = _apply_contrast_shaping(
                    delta_m, r_max=float(g.r_max), is_contrast=False
                )

        for iz in iz_list:
            sl = iz * n_mom + dens_i
            dens0 = np.maximum(stack[sl], 0.0)
            axisym0, _ = _axisym_and_rbin(dens0)
            resid0 = dens0 - axisym0
            eps0 = 1e-12 * max(float(np.max(axisym0)), 1e-30)
            if scale == "multiplicative":
                if mid_factor is not None:
                    factor = mid_factor
                else:
                    dens_m = np.maximum(morph[sl], 0.0)
                    axisym_m, delta = _morph_contrast_delta(
                        dens_m,
                        r_max=float(g.r_max),
                        mode=mode,
                        m2_n_bins=int(m2_n_bins),
                        other_morph_alpha=other_a,
                    )
                    eps_m = 1e-12 * max(float(np.max(axisym_m)), 1e-30)
                    contrast = delta / np.maximum(axisym_m, eps_m)
                    contrast = _apply_contrast_shaping(
                        contrast, r_max=float(g.r_max), is_contrast=True
                    )
                    factor = np.maximum(1.0 + alpha * contrast, floor)
                boosted = axisym0 * factor + keep_f0 * resid0
            else:
                if mid_delta is not None:
                    delta = mid_delta
                else:
                    dens_m = np.maximum(morph[sl], 0.0)
                    _, delta = _morph_contrast_delta(
                        dens_m,
                        r_max=float(g.r_max),
                        mode=mode,
                        m2_n_bins=int(m2_n_bins),
                        other_morph_alpha=other_a,
                    )
                    delta = _apply_contrast_shaping(
                        delta, r_max=float(g.r_max), is_contrast=False
                    )
                boosted = axisym0 + keep_f0 * resid0 + alpha * delta
            if clip_neg:
                boosted = np.maximum(boosted, 0.0)
            if preserve_axisym:
                # Lock Σ_axisym(R) to f0 after clip so anti-bar wipeout cannot
                # redistribute radial mass into the bar.
                boosted = _renormalize_axisym_rings(boosted, axisym0, eps=eps0)
            stack[sl] = boosted
        out[g.name] = stack
    return out


def inject_predicted_contrast_on_f0_dens(
    maps_f0: dict[str, np.ndarray],
    contrast_midplane: np.ndarray,
    *,
    cfg: MultiScaleSliceConfig,
    components: tuple[str, ...] = ("disk",),
    alpha: float = 1.0,
    clip_neg: bool = True,
    midplane_only: bool = False,
    midplane_n_slabs: int = 3,
    keep_f0_residual: float = 1.0,
    preserve_axisym: bool = True,
    factor_floor: float = 0.05,
) -> dict[str, np.ndarray]:
    """
    Phase-B: apply a predicted midplane multiplicative contrast onto ``f0`` dens.

    ``factor = max(1 + α · contrast, floor)``, then ring-renormalize to
    ``axisym(Σ_f0)`` (same collapse guard as ``inject_morph_residual_on_f0_dens``).
    Moments channels are copied from ``maps_f0`` unchanged.
    """
    alpha = float(alpha)
    keep_f0 = float(keep_f0_residual)
    floor = float(factor_floor)
    if not (0.0 <= floor < 1.0):
        raise ValueError(f"factor_floor={floor!r} must be in [0, 1)")
    contrast = np.asarray(contrast_midplane, dtype=np.float64)
    if contrast.ndim != 2:
        raise ValueError(f"contrast_midplane must be 2D, got {contrast.shape}")
    mid_factor = np.maximum(1.0 + alpha * contrast, floor)
    keep = set(components)
    out: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float64).copy() for k, v in maps_f0.items()
    }
    for g in cfg.grids:
        if g.name not in keep or g.name not in out:
            continue
        stack = out[g.name]
        dens_i = g.moment_keys.index("dens")
        n_mom = len(g.moment_keys)
        n_z = int(g.n_z)
        iz_mid = n_z // 2
        if midplane_only:
            half = max(1, int(midplane_n_slabs) // 2)
            iz_list = range(max(0, iz_mid - half), min(n_z, iz_mid + half + 1))
        else:
            iz_list = range(n_z)
        n_pix = int(stack.shape[-1])
        if mid_factor.shape != (n_pix, n_pix):
            raise ValueError(
                f"contrast shape {mid_factor.shape} != dens ({n_pix}, {n_pix})"
            )
        for iz in iz_list:
            sl = iz * n_mom + dens_i
            dens0 = np.maximum(stack[sl], 0.0)
            axisym0, _ = _axisym_and_rbin(dens0)
            resid0 = dens0 - axisym0
            eps0 = 1e-12 * max(float(np.max(axisym0)), 1e-30)
            boosted = axisym0 * mid_factor + keep_f0 * resid0
            if clip_neg:
                boosted = np.maximum(boosted, 0.0)
            if preserve_axisym:
                boosted = _renormalize_axisym_rings(boosted, axisym0, eps=eps0)
            stack[sl] = boosted
        out[g.name] = stack
    return out


def transplant_velocities_knn(
    pos_new: np.ndarray,
    cid_new: np.ndarray,
    pos_src: np.ndarray,
    vel_src: np.ndarray,
    cid_src: np.ndarray,
    *,
    components: tuple[str, ...] | None = None,
    rotate_with_phi: bool = True,
    n_ref: int = 64_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Assign GalactICS (source) velocities to morph-resampled positions.

    Per component, each new particle inherits the cylindrical velocity of its
    nearest source particle in midplane ``(x, y, z)`` (subsampled to ``n_ref``
    for speed). When ``rotate_with_phi`` is True, ``(v_R, v_φ, v_z)`` are
    measured at the source azimuth and rebuilt at the new azimuth so streaming
    rotation follows the remapped bar angle.
    """
    rng = rng or np.random.default_rng(0)
    pos_new = np.asarray(pos_new, dtype=np.float64)
    vel_out = np.zeros_like(pos_new)
    cid_new = np.asarray(cid_new)
    pos_src = np.asarray(pos_src, dtype=np.float64)
    vel_src = np.asarray(vel_src, dtype=np.float64)
    cid_src = np.asarray(cid_src)
    names = components if components is not None else tuple(COMPONENT_IDS.keys())
    meta: dict = {"method": "knn_cyl_vel_transplant", "components": {}}

    for name in names:
        if name not in COMPONENT_IDS:
            continue
        cid = int(COMPONENT_IDS[name])
        i_new = np.flatnonzero(cid_new == cid)
        i_src = np.flatnonzero(cid_src == cid)
        if i_new.size == 0 or i_src.size == 0:
            continue
        src_idx = i_src
        if src_idx.size > int(n_ref):
            src_idx = rng.choice(src_idx, size=int(n_ref), replace=False)
        p_s = pos_src[src_idx]
        v_s = vel_src[src_idx]
        p_n = pos_new[i_new]
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(p_s)
            _, nn = tree.query(p_n, k=1, workers=-1)
            nn = np.asarray(nn, dtype=np.int64)
        except Exception:  # noqa: BLE001
            # Chunked brute-force fallback (keep chunks small to limit RAM).
            nn = np.empty(i_new.size, dtype=np.int64)
            chunk = 2048
            for a in range(0, i_new.size, chunk):
                b = min(a + chunk, i_new.size)
                d2 = (
                    (p_n[a:b, None, 0] - p_s[None, :, 0]) ** 2
                    + (p_n[a:b, None, 1] - p_s[None, :, 1]) ** 2
                    + (p_n[a:b, None, 2] - p_s[None, :, 2]) ** 2
                )
                nn[a:b] = np.argmin(d2, axis=1)
        if not rotate_with_phi:
            vel_out[i_new] = v_s[nn]
        else:
            xs, ys = p_s[nn, 0], p_s[nn, 1]
            xn, yn = p_n[:, 0], p_n[:, 1]
            cs, ss = _cyl_basis(xs, ys)
            cn, sn = _cyl_basis(xn, yn)
            vR, vphi, vz = _cart_vel_to_cyl(
                v_s[nn, 0], v_s[nn, 1], v_s[nn, 2], cs, ss
            )
            vx, vy, vz2 = _cyl_vel_to_cart(vR, vphi, vz, cn, sn)
            vel_out[i_new, 0] = vx
            vel_out[i_new, 1] = vy
            vel_out[i_new, 2] = vz2
        meta["components"][name] = {
            "n_new": int(i_new.size),
            "n_src_ref": int(src_idx.size),
            "rotate_with_phi": bool(rotate_with_phi),
        }
    # Components not requested: leave zero (caller should retain those particles).
    return vel_out, meta


def _cyl_basis(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cos φ, sin φ)`` for midplane azimuth; ``φ=0`` on the axis."""
    r = np.hypot(x, y)
    c = np.ones_like(r)
    s = np.zeros_like(r)
    ok = r > 1e-8
    c[ok] = x[ok] / r[ok]
    s[ok] = y[ok] / r[ok]
    return c, s


def _cart_vel_to_cyl(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, c: np.ndarray, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian → ``(v_R, v_φ, v_z)`` with basis ``(c,s)=(cos φ, sin φ)``."""
    return vx * c + vy * s, -vx * s + vy * c, vz


def _cyl_vel_to_cart(
    v_r: np.ndarray, v_phi: np.ndarray, v_z: np.ndarray, c: np.ndarray, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(v_R, v_φ, v_z)`` → Cartesian with basis ``(c,s)=(cos φ, sin φ)``."""
    return v_r * c - v_phi * s, v_r * s + v_phi * c, v_z


def _cart_diag_sigma_to_cyl(
    sx: np.ndarray, sy: np.ndarray, sz: np.ndarray, c: np.ndarray, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project diagonal Cartesian σ onto cylindrical axes (ignore cyl off-diag).

    For ``C = diag(sx², sy², sz²)`` and rotation into ``(ê_R, ê_φ, ê_z)``:
    ``σ_R² = sx² c² + sy² s²``, ``σ_φ² = sx² s² + sy² c²``, ``σ_z = sz``.
    """
    sx2 = np.maximum(sx, 0.0) ** 2
    sy2 = np.maximum(sy, 0.0) ** 2
    sig_r = np.sqrt(np.maximum(sx2 * c * c + sy2 * s * s, 0.0))
    sig_phi = np.sqrt(np.maximum(sx2 * s * s + sy2 * c * c, 0.0))
    return sig_r, sig_phi, np.maximum(sz, 0.0)


def _match_cell_moments_affine(
    vel: np.ndarray,
    flat_idx: np.ndarray,
    *,
    target_mean: np.ndarray,
    target_sigma: np.ndarray | None,
    min_count_mean: int = 1,
    min_count_sigma: int = 2,
) -> np.ndarray:
    """
    Per-cell affine correct so sample Cartesian mean / σ match deposited targets.

    Low-``n`` cells: mean-only when ``n >= min_count_mean``; σ scale when
    ``n >= min_count_sigma``. Operates in Cartesian so redeposited ``⟨v⟩,σ``
    maps regenerate the targets (works with either velocity draw frame).
    """
    vel = np.asarray(vel, dtype=np.float64).copy()
    flat_idx = np.asarray(flat_idx, dtype=np.int64)
    n = vel.shape[0]
    if n == 0:
        return vel
    target_mean = np.asarray(target_mean, dtype=np.float64)
    target_sigma_f = (
        None if target_sigma is None else np.asarray(target_sigma, dtype=np.float64)
    )

    order = np.argsort(flat_idx, kind="mergesort")
    sorted_idx = flat_idx[order]
    breaks = np.flatnonzero(np.diff(sorted_idx)) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [n]])
    for a, b in zip(starts, ends):
        sel = order[a:b]
        n_cell = int(sel.size)
        if n_cell < int(min_count_mean):
            continue
        sample = vel[sel]
        mean_s = sample.mean(axis=0)
        centered = sample - mean_s
        # All particles in a cell share the same deposited target.
        tgt_mean = target_mean[sel[0]]
        out = tgt_mean + centered
        if target_sigma_f is not None and n_cell >= int(min_count_sigma):
            std_s = sample.std(axis=0, ddof=0)
            tgt0 = target_sigma_f[sel[0]]
            scale = np.ones(3, dtype=np.float64)
            for k in range(3):
                if std_s[k] > 1e-12 and tgt0[k] > 0.0:
                    scale[k] = float(tgt0[k] / std_s[k])
            out = tgt_mean + centered * scale
        vel[sel] = out
    return vel


def _resample_one_component(
    stack: np.ndarray,
    grid: ComponentSliceGrid,
    *,
    n_particles: int,
    mass_total: float | None,
    rng: np.random.Generator,
    sample_dispersion: bool = True,
    velocity_frame: str = "cartesian",
    match_cell_moments: bool = False,
) -> dict[str, np.ndarray]:
    """Draw particles from one component's native ``(C, H, W)`` moment stack.

    ``velocity_frame``:
      - ``cartesian`` — legacy ``⟨v⟩ + σ⊙N(0,1)`` in ``(vx,vy,vz)``
      - ``cylindrical`` — draw in ``(v_R, v_φ, v_z)`` from Cartesian deposit
        means / diagonal-σ projection, then rotate to Cartesian (reduces
        ``⟨v_φ⟩`` leakage from anisotropic Cartesian σ)

    ``match_cell_moments``: after sampling, affine-correct per slab cell so
    sample mean / σ match the deposited targets (moment-consistent map→particle).
    """
    stack = np.asarray(stack, dtype=np.float64)
    keys = grid.moment_keys
    n_mom = len(keys)
    n_z = int(grid.n_z)
    n_pix = int(grid.n_pix)
    frame = str(velocity_frame).lower().strip()
    if frame not in ("cartesian", "cylindrical"):
        raise ValueError(
            f"unknown velocity_frame={velocity_frame!r} (expected cartesian|cylindrical)"
        )
    if stack.shape[-2:] != (n_pix, n_pix):
        raise ValueError(
            f"{grid.name}: stack spatial {stack.shape[-2:]} != grid ({n_pix}, {n_pix})"
        )

    xy_edges = np.linspace(-float(grid.r_max), float(grid.r_max), n_pix + 1)
    z_edges = z_edges_for_grid(grid)
    dx = float(xy_edges[1] - xy_edges[0])
    dy = dx

    dens_i = keys.index("dens")
    has_vel = all(k in keys for k in ("vx", "vy", "vz"))
    has_sigma = all(k in keys for k in ("sx", "sy", "sz"))

    # Slice dens is slab surface density Σ = mass / (dx·dy) (see
    # ``_deposit_moments_2d``), not volume density.  Recover slab mass with
    # dens·dx·dy — do **not** multiply by Δz (that under-masses thin midplane
    # slabs and over-masses thick halo slabs by factors of ~Δz).
    mass_slab = np.zeros((n_z, n_pix, n_pix), dtype=np.float64)
    vel = np.zeros((n_z, n_pix, n_pix, 3), dtype=np.float64)
    sigma = np.zeros((n_z, n_pix, n_pix, 3), dtype=np.float64)
    for iz in range(n_z):
        base = iz * n_mom
        dens = np.maximum(stack[base + dens_i], 0.0)
        mass_slab[iz] = dens * dx * dy
        if has_vel:
            vel[iz, ..., 0] = stack[base + keys.index("vx")]
            vel[iz, ..., 1] = stack[base + keys.index("vy")]
            vel[iz, ..., 2] = stack[base + keys.index("vz")]
        if has_sigma:
            sigma[iz, ..., 0] = np.maximum(stack[base + keys.index("sx")], 0.0)
            sigma[iz, ..., 1] = np.maximum(stack[base + keys.index("sy")], 0.0)
            sigma[iz, ..., 2] = np.maximum(stack[base + keys.index("sz")], 0.0)

    total = float(mass_total) if mass_total is not None else float(mass_slab.sum())
    n_c = int(n_particles)
    if n_c <= 0:
        return {
            "pos": np.zeros((0, 3)),
            "vel": np.zeros((0, 3)),
            "mass": np.zeros(0),
            "component_id": np.zeros(0, dtype=np.int64),
        }

    w = np.maximum(mass_slab.ravel(), 0.0)
    if w.sum() <= 0:
        p = np.ones_like(w) / w.size
    else:
        p = w / w.sum()
    flat_idx = rng.choice(w.size, size=n_c, replace=True, p=p)
    iz = flat_idx // (n_pix * n_pix)
    rem = flat_idx % (n_pix * n_pix)
    ix = rem // n_pix
    iy = rem % n_pix

    u = rng.random(n_c)
    v = rng.random(n_c)
    wj = rng.random(n_c)
    x = xy_edges[ix] + u * dx
    y = xy_edges[iy] + v * dy
    z = z_edges[iz] + wj * (z_edges[iz + 1] - z_edges[iz])
    pos_c = np.stack([x, y, z], axis=1)
    mean_c = vel[iz, ix, iy].copy()
    sig_c = sigma[iz, ix, iy] if has_sigma else None

    if not has_vel:
        vel_c = np.zeros((n_c, 3), dtype=np.float64)
    elif frame == "cylindrical":
        c_b, s_b = _cyl_basis(x, y)
        v_r, v_phi, v_z = _cart_vel_to_cyl(
            mean_c[:, 0], mean_c[:, 1], mean_c[:, 2], c_b, s_b
        )
        if sample_dispersion and sig_c is not None:
            sig_r, sig_phi, sig_z = _cart_diag_sigma_to_cyl(
                sig_c[:, 0], sig_c[:, 1], sig_c[:, 2], c_b, s_b
            )
            eps = rng.normal(0.0, 1.0, size=(n_c, 3))
            v_r = v_r + eps[:, 0] * sig_r
            v_phi = v_phi + eps[:, 1] * sig_phi
            v_z = v_z + eps[:, 2] * sig_z
        vx, vy, vz = _cyl_vel_to_cart(v_r, v_phi, v_z, c_b, s_b)
        vel_c = np.stack([vx, vy, vz], axis=1)
    else:
        vel_c = mean_c
        if sample_dispersion and sig_c is not None:
            vel_c = vel_c + rng.normal(0.0, 1.0, size=vel_c.shape) * sig_c

    if match_cell_moments and has_vel:
        vel_c = _match_cell_moments_affine(
            vel_c,
            flat_idx,
            target_mean=mean_c,
            target_sigma=sig_c if (sample_dispersion and has_sigma) else None,
        )

    m_each = total / n_c if total > 0 else 1.0 / n_c
    cid = np.full(n_c, int(COMPONENT_IDS[grid.name]), dtype=np.int64)
    return {
        "pos": pos_c,
        "vel": vel_c,
        "mass": np.full(n_c, m_each, dtype=np.float64),
        "component_id": cid,
    }


def resample_particles_from_component_stack(
    stack: np.ndarray,
    grid: ComponentSliceGrid,
    *,
    n_particles: int,
    mass_total: float | None = None,
    rng: np.random.Generator | None = None,
    sample_dispersion: bool = True,
    velocity_frame: str = "cartesian",
    match_cell_moments: bool = False,
) -> dict[str, np.ndarray]:
    """Resample one component from its native multi-scale slice stack."""
    rng = rng or np.random.default_rng(0)
    return _resample_one_component(
        stack,
        grid,
        n_particles=n_particles,
        mass_total=mass_total,
        rng=rng,
        sample_dispersion=sample_dispersion,
        velocity_frame=velocity_frame,
        match_cell_moments=match_cell_moments,
    )


def resample_particles_from_multiscale(
    maps: dict[str, np.ndarray],
    *,
    cfg: MultiScaleSliceConfig,
    n_particles: int = 50_000,
    mass_total_per_component: dict[str, float] | None = None,
    count_fractions: dict[str, float] | None = None,
    n_per_component: dict[str, int] | None = None,
    rng: np.random.Generator | None = None,
    sample_dispersion: bool = True,
    velocity_frame: str = "cartesian",
    match_cell_moments: bool = False,
) -> dict[str, np.ndarray]:
    """
    Draw particles from per-component native-resolution reconstructed stacks.

    Particle **masses** default to integrated dens (or ``mass_total_per_component``
    overrides — prefer true snapshot masses).  Particle **counts** default to a
    mass-proportional split, which starves the disk in MW-like mass ratios
    (halo ≫ disk).  Prefer ``count_fractions`` (e.g. corpus mix disk:halo:bulge
    ≈ 4:2:1) or an explicit ``n_per_component`` for morphology / evolve panels.
    Each component is sampled inside **its own** FOV / Δz grid.  When σ
    channels are present, velocities are drawn as ``⟨v⟩ + σ · N(0,1)``
    (``velocity_frame="cartesian"``) or in the cylindrical local frame
    (``velocity_frame="cylindrical"``).  ``match_cell_moments`` affine-corrects
    per cell so redeposited moments regenerate the map targets.
    """
    rng = rng or np.random.default_rng(0)
    totals: dict[str, float] = {}
    for g in cfg.grids:
        stack = np.asarray(maps[g.name], dtype=np.float64)
        if mass_total_per_component and g.name in mass_total_per_component:
            totals[g.name] = float(mass_total_per_component[g.name])
        else:
            keys = g.moment_keys
            n_mom = len(keys)
            dens_i = keys.index("dens")
            xy_edges = np.linspace(-float(g.r_max), float(g.r_max), int(g.n_pix) + 1)
            dx = float(xy_edges[1] - xy_edges[0])
            msum = 0.0
            # dens is slab Σ (mass/area); integrate dens·dA over slabs (no Δz).
            for iz in range(g.n_z):
                dens = np.maximum(stack[iz * n_mom + dens_i], 0.0)
                msum += float(dens.sum() * dx * dx)
            totals[g.name] = msum

    total_mass = sum(max(v, 0.0) for v in totals.values())
    if total_mass <= 0:
        raise ValueError("multiscale stacks have zero mass; cannot resample")

    comps = list(cfg.components)
    n_per: dict[str, int] = {}
    if n_per_component is not None:
        for c in comps:
            n_per[c] = int(max(0, n_per_component.get(c, 0)))
        # Top up / trim so sum matches n_particles when possible.
        s = sum(n_per.values())
        if s > 0 and s != int(n_particles):
            scale = int(n_particles) / s
            rem = int(n_particles)
            for i, c in enumerate(comps):
                if i == len(comps) - 1:
                    n_per[c] = rem
                else:
                    n_c = int(round(n_per[c] * scale))
                    n_per[c] = n_c
                    rem -= n_c
    elif count_fractions is not None:
        fr = {c: float(count_fractions.get(c, 0.0)) for c in comps}
        wsum = sum(max(v, 0.0) for v in fr.values())
        if wsum <= 0:
            raise ValueError("count_fractions sum to zero")
        rem = int(n_particles)
        for i, c in enumerate(comps):
            if i == len(comps) - 1:
                n_per[c] = rem
            else:
                n_c = int(round(n_particles * max(fr[c], 0.0) / wsum))
                n_per[c] = n_c
                rem -= n_c
    else:
        rem = int(n_particles)
        for i, c in enumerate(comps):
            if i == len(comps) - 1:
                n_per[c] = rem
            else:
                n_c = int(round(n_particles * totals[c] / total_mass))
                n_per[c] = n_c
                rem -= n_c

    parts = [
        _resample_one_component(
            maps[g.name],
            g,
            n_particles=n_per[g.name],
            mass_total=totals[g.name],
            rng=rng,
            sample_dispersion=sample_dispersion,
            velocity_frame=velocity_frame,
            match_cell_moments=match_cell_moments,
        )
        for g in cfg.grids
        if n_per[g.name] > 0
    ]
    if not parts:
        raise ValueError("no particles requested")
    return {
        "pos": np.concatenate([p["pos"] for p in parts], axis=0),
        "vel": np.concatenate([p["vel"] for p in parts], axis=0),
        "mass": np.concatenate([p["mass"] for p in parts], axis=0),
        "component_id": np.concatenate([p["component_id"] for p in parts], axis=0),
        "n_per_component": n_per,
    }


def bin_spherical_shell_moments(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    *,
    n_shells: int = 48,
    r_min: float = 0.05,
    r_max: float = 4.0,
    log_bins: bool = True,
) -> dict[str, np.ndarray]:
    """
    Azimuthally averaged spherical-shell dens + ⟨v⟩ + σ for a spheroid.

    Returns arrays of length ``n_shells`` (empty shells → dens=0, vel/σ=0).
    Coordinates are assumed already centred on the component of interest.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).ravel()
    n_shells = int(n_shells)
    if log_bins:
        edges = np.geomspace(float(r_min), float(r_max), n_shells + 1)
    else:
        edges = np.linspace(float(r_min), float(r_max), n_shells + 1)
    r = np.linalg.norm(pos, axis=1)
    shell = np.clip(np.searchsorted(edges, r, side="right") - 1, -1, n_shells - 1)
    vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    dens = np.zeros(n_shells, dtype=np.float64)
    mean_v = np.zeros((n_shells, 3), dtype=np.float64)
    mean_v2 = np.zeros((n_shells, 3), dtype=np.float64)
    m_shell = np.zeros(n_shells, dtype=np.float64)
    for i in range(n_shells):
        mask = shell == i
        if not np.any(mask):
            continue
        m = mass[mask]
        m_shell[i] = float(m.sum())
        dens[i] = m_shell[i] / max(vol[i], 1e-30)
        for k in range(3):
            mean_v[i, k] = float(np.average(vel[mask, k], weights=m))
            mean_v2[i, k] = float(np.average(vel[mask, k] ** 2, weights=m))
    var = np.maximum(mean_v2 - mean_v**2, 0.0)
    sigma = np.sqrt(var)
    return {
        "r_edges": edges.astype(np.float64),
        "r_mid": (0.5 * (edges[:-1] + edges[1:])).astype(np.float64),
        "dens": dens,
        "mass": m_shell,
        "vx": mean_v[:, 0],
        "vy": mean_v[:, 1],
        "vz": mean_v[:, 2],
        "sx": sigma[:, 0],
        "sy": sigma[:, 1],
        "sz": sigma[:, 2],
        "vol": vol.astype(np.float64),
    }


def resample_bulge_from_spherical_shells(
    shells: dict[str, np.ndarray],
    *,
    n_particles: int,
    mass_total: float | None = None,
    rng: np.random.Generator | None = None,
    sample_dispersion: bool = True,
) -> dict[str, np.ndarray]:
    """
    Draw isotropic bulge particles from spherical-shell dens + moment profiles.

    Prefer this over cylindrical slab voxels when the cusp is spheroidal: shell
    masses follow ρ(r)·4πr²Δr without Δz smear.  Azimuthal structure is not
    preserved (acceptable for near-spherical GalactICS bulges).
    """
    rng = rng or np.random.default_rng(0)
    n_c = int(n_particles)
    edges = np.asarray(shells["r_edges"], dtype=np.float64)
    m_shell = np.asarray(shells["mass"], dtype=np.float64)
    if mass_total is not None:
        # Rescale shell masses to match true component mass.
        s = float(m_shell.sum())
        if s > 0:
            m_shell = m_shell * (float(mass_total) / s)
        else:
            m_shell = np.full_like(m_shell, float(mass_total) / max(len(m_shell), 1))
    total = float(m_shell.sum()) if mass_total is None else float(mass_total)
    if n_c <= 0:
        return {
            "pos": np.zeros((0, 3)),
            "vel": np.zeros((0, 3)),
            "mass": np.zeros(0),
            "component_id": np.zeros(0, dtype=np.int64),
        }
    w = np.maximum(m_shell, 0.0)
    if w.sum() <= 0:
        p = np.ones_like(w) / w.size
    else:
        p = w / w.sum()
    iz = rng.choice(w.size, size=n_c, replace=True, p=p)
    # Uniform in volume within each shell: r³ lerp.
    u = rng.random(n_c)
    r_lo = edges[iz]
    r_hi = edges[iz + 1]
    r = (r_lo**3 + u * (r_hi**3 - r_lo**3)) ** (1.0 / 3.0)
    # Isotropic directions.
    mu = rng.uniform(-1.0, 1.0, size=n_c)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_c)
    sphi = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    x = r * sphi * np.cos(phi)
    y = r * sphi * np.sin(phi)
    z = r * mu
    pos_c = np.stack([x, y, z], axis=1)
    vel_c = np.stack(
        [
            np.asarray(shells["vx"], dtype=np.float64)[iz],
            np.asarray(shells["vy"], dtype=np.float64)[iz],
            np.asarray(shells["vz"], dtype=np.float64)[iz],
        ],
        axis=1,
    )
    if sample_dispersion:
        sig = np.stack(
            [
                np.asarray(shells["sx"], dtype=np.float64)[iz],
                np.asarray(shells["sy"], dtype=np.float64)[iz],
                np.asarray(shells["sz"], dtype=np.float64)[iz],
            ],
            axis=1,
        )
        vel_c = vel_c + rng.normal(0.0, 1.0, size=vel_c.shape) * sig
    m_each = total / n_c if total > 0 else 1.0 / n_c
    return {
        "pos": pos_c,
        "vel": vel_c,
        "mass": np.full(n_c, m_each, dtype=np.float64),
        "component_id": np.full(n_c, int(COMPONENT_IDS["bulge"]), dtype=np.int64),
    }


def stitch_retained_components(
    resampled: dict[str, np.ndarray],
    *,
    source_pos: np.ndarray,
    source_vel: np.ndarray,
    source_mass: np.ndarray,
    source_cid: np.ndarray,
    retain: tuple[str, ...] = ("bulge",),
    n_retain: dict[str, int] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Hybrid IC: keep GalactICS (or dump) particles for selected components.

    Replaces resampled particles whose ``component_id`` matches each retained
    component with (subsampled) source particles.  Use when the field tower
    cannot resolve a spheroidal cusp (cylindrical slab bulge) but disk/halo
    morph fields are still useful.

    Source arrays must already share the same morphological frame as the
    resampled particles (typically after ``prepare_shared_frame``).
    """
    rng = rng or np.random.default_rng(0)
    retain_names = tuple(str(c) for c in retain)
    if not retain_names:
        return dict(resampled)

    pos_r = np.asarray(resampled["pos"], dtype=np.float64)
    vel_r = np.asarray(resampled["vel"], dtype=np.float64)
    mass_r = np.asarray(resampled["mass"], dtype=np.float64)
    cid_r = np.asarray(resampled["component_id"], dtype=np.int64)

    keep = np.ones(cid_r.shape[0], dtype=bool)
    chunks_pos: list[np.ndarray] = []
    chunks_vel: list[np.ndarray] = []
    chunks_mass: list[np.ndarray] = []
    chunks_cid: list[np.ndarray] = []
    n_per = dict(resampled.get("n_per_component") or {})

    for name in retain_names:
        if name not in COMPONENT_IDS:
            raise KeyError(f"unknown component {name!r}")
        cid = int(COMPONENT_IDS[name])
        keep &= cid_r != cid
        src = np.asarray(source_cid, dtype=np.int64) == cid
        if not np.any(src):
            raise ValueError(f"source has no particles for component {name!r}")
        sp = np.asarray(source_pos[src], dtype=np.float64)
        sv = np.asarray(source_vel[src], dtype=np.float64)
        sm = np.asarray(source_mass[src], dtype=np.float64)
        n_want = int(n_retain[name]) if n_retain and name in n_retain else int(sp.shape[0])
        if n_want <= 0:
            n_per[name] = 0
            continue
        if n_want < sp.shape[0]:
            sel = rng.choice(sp.shape[0], size=n_want, replace=False)
            sp, sv, sm = sp[sel], sv[sel], sm[sel]
        elif n_want > sp.shape[0]:
            # Upsample with replacement (rare; usually we subsample to mix).
            sel = rng.choice(sp.shape[0], size=n_want, replace=True)
            sp, sv, sm = sp[sel], sv[sel], sm[sel]
        chunks_pos.append(sp)
        chunks_vel.append(sv)
        chunks_mass.append(sm)
        chunks_cid.append(np.full(sp.shape[0], cid, dtype=np.int64))
        n_per[name] = int(sp.shape[0])

    out_pos = [pos_r[keep]]
    out_vel = [vel_r[keep]]
    out_mass = [mass_r[keep]]
    out_cid = [cid_r[keep]]
    out_pos.extend(chunks_pos)
    out_vel.extend(chunks_vel)
    out_mass.extend(chunks_mass)
    out_cid.extend(chunks_cid)
    return {
        "pos": np.concatenate(out_pos, axis=0),
        "vel": np.concatenate(out_vel, axis=0),
        "mass": np.concatenate(out_mass, axis=0),
        "component_id": np.concatenate(out_cid, axis=0),
        "n_per_component": n_per,
        "retained_components": retain_names,
    }


def fuse_shell_bulge_with_multiscale(
    maps: dict[str, np.ndarray],
    *,
    cfg: MultiScaleSliceConfig,
    bulge_shells: dict[str, np.ndarray],
    n_particles: int = 50_000,
    count_fractions: dict[str, float] | None = None,
    mass_total_per_component: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
    sample_dispersion: bool = True,
    velocity_frame: str = "cartesian",
    match_cell_moments: bool = False,
) -> dict[str, np.ndarray]:
    """
    Disk/halo from slice towers; bulge from spherical-shell profiles.

    Morph fields stay multi-scale for the disk (bars); the bulge cusp is
    sampled spherically so cylindrical Δz smear does not erase ρ(0).
    """
    rng = rng or np.random.default_rng(0)
    fr = count_fractions or {"disk": 4 / 7, "halo": 2 / 7, "bulge": 1 / 7}
    # Resample non-bulge components only, then stitch shell bulge.
    maps_no_b = {k: v for k, v in maps.items() if k != "bulge"}
    grids_no_b = tuple(g for g in cfg.grids if g.name != "bulge")
    if not grids_no_b:
        raise ValueError("fuse_shell_bulge_with_multiscale needs disk/halo grids")
    cfg_no_b = MultiScaleSliceConfig(
        grids=grids_no_b, include_potential=cfg.include_potential
    )
    n_bulge = int(round(n_particles * float(fr.get("bulge", 0.0))))
    n_rest = int(n_particles) - n_bulge
    rest = resample_particles_from_multiscale(
        maps_no_b,
        cfg=cfg_no_b,
        n_particles=max(n_rest, 1),
        count_fractions={k: fr.get(k, 0.0) for k in cfg_no_b.components},
        mass_total_per_component=mass_total_per_component,
        rng=rng,
        sample_dispersion=sample_dispersion,
        velocity_frame=velocity_frame,
        match_cell_moments=match_cell_moments,
    )
    m_bulge = None
    if mass_total_per_component and "bulge" in mass_total_per_component:
        m_bulge = float(mass_total_per_component["bulge"])
    bulge = resample_bulge_from_spherical_shells(
        bulge_shells,
        n_particles=max(n_bulge, 0),
        mass_total=m_bulge,
        rng=rng,
        sample_dispersion=sample_dispersion,
    )
    n_per = dict(rest.get("n_per_component") or {})
    n_per["bulge"] = int(bulge["pos"].shape[0])
    return {
        "pos": np.concatenate([rest["pos"], bulge["pos"]], axis=0),
        "vel": np.concatenate([rest["vel"], bulge["vel"]], axis=0),
        "mass": np.concatenate([rest["mass"], bulge["mass"]], axis=0),
        "component_id": np.concatenate(
            [rest["component_id"], bulge["component_id"]], axis=0
        ),
        "n_per_component": n_per,
        "bulge_method": "spherical_shells",
    }


def _mass_weighted_cdf_edges(
    radius: np.ndarray,
    mass: np.ndarray,
    *,
    n_bins: int = 64,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(edges, cdf_at_edges)`` for a mass-weighted radial CDF."""
    radius = np.asarray(radius, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if radius.size == 0:
        r_hi = 1.0 if r_max is None else float(r_max)
        edges = np.linspace(0.0, r_hi, int(n_bins) + 1)
        return edges, np.linspace(0.0, 1.0, int(n_bins) + 1)
    r_hi = float(r_max) if r_max is not None else float(np.percentile(radius, 99.5))
    r_hi = max(r_hi, float(np.max(radius)) * 1.01, 1e-3)
    edges = np.linspace(0.0, r_hi, int(n_bins) + 1)
    hist, _ = np.histogram(radius, bins=edges, weights=np.maximum(mass, 0.0))
    cdf = np.concatenate([[0.0], np.cumsum(hist)])
    total = float(cdf[-1])
    if total <= 0.0:
        return edges, np.linspace(0.0, 1.0, edges.size)
    cdf = cdf / total
    # Enforce strictly non-decreasing for interp (flat bins OK).
    cdf = np.maximum.accumulate(cdf)
    cdf[-1] = 1.0
    return edges, cdf


def _invert_cdf(u: np.ndarray, edges: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Map CDF values ``u∈[0,1]`` to radii via linear interp of ``(cdf, edges)``."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    # Deduplicate flat CDF plateaus so searchsorted is well-defined.
    keep = np.concatenate([[True], np.diff(cdf) > 1e-15])
    c = cdf[keep]
    e = edges[keep]
    if c.size < 2:
        return np.full_like(u, float(e[0]) if e.size else 0.0)
    return np.interp(u, c, e)


def match_disk_radial_cdf_to_reference(
    pos: np.ndarray,
    mass: np.ndarray,
    cid: np.ndarray,
    pos_ref: np.ndarray,
    mass_ref: np.ndarray,
    cid_ref: np.ndarray,
    *,
    r_max: float = 15.0,
    n_bins: int = 64,
    match_z_scale: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Remap disk cylindrical ``R`` so the mass-weighted CDF matches a reference.

    Preserves azimuth ``φ`` and (optionally) rescales ``z`` by the median-R
    ratio. Used after morph dens-resample so particle Σ(R) cannot drift away
    from GalactICS ``f0`` even if the dens field was imperfectly normalized.
    """
    pos_out = np.asarray(pos, dtype=np.float64).copy()
    mass = np.asarray(mass, dtype=np.float64)
    cid = np.asarray(cid)
    pos_ref = np.asarray(pos_ref, dtype=np.float64)
    mass_ref = np.asarray(mass_ref, dtype=np.float64)
    cid_ref = np.asarray(cid_ref)
    disk = int(COMPONENT_IDS["disk"])
    i = np.flatnonzero(cid == disk)
    i_ref = np.flatnonzero(cid_ref == disk)
    meta: dict = {"method": "match_disk_radial_cdf", "applied": False}
    if i.size == 0 or i_ref.size == 0:
        return pos_out, meta
    Rs = np.hypot(pos_out[i, 0], pos_out[i, 1])
    Rt = np.hypot(pos_ref[i_ref, 0], pos_ref[i_ref, 1])
    e_s, c_s = _mass_weighted_cdf_edges(Rs, mass[i], n_bins=n_bins, r_max=r_max)
    e_t, c_t = _mass_weighted_cdf_edges(
        Rt, mass_ref[i_ref], n_bins=n_bins, r_max=r_max
    )
    u = np.interp(Rs, e_s, c_s)
    R_new = _invert_cdf(u, e_t, c_t)
    scale_R = np.ones_like(Rs)
    ok = Rs > 1e-8
    scale_R[ok] = R_new[ok] / Rs[ok]
    pos_out[i, 0] = pos_out[i, 0] * scale_R
    pos_out[i, 1] = pos_out[i, 1] * scale_R
    z_fac = 1.0
    if match_z_scale:
        med_s = float(np.median(Rs[Rs > 0.1])) if np.any(Rs > 0.1) else 1.0
        med_t = float(np.median(Rt[Rt > 0.1])) if np.any(Rt > 0.1) else med_s
        z_fac = float(np.clip(med_t / max(med_s, 1e-6), 0.5, 2.0))
        pos_out[i, 2] = pos_out[i, 2] * z_fac
    meta.update(
        {
            "applied": True,
            "n_disk": int(i.size),
            "R50_before": float(np.median(Rs)),
            "R50_after": float(np.median(np.hypot(pos_out[i, 0], pos_out[i, 1]))),
            "z_scale": z_fac,
        }
    )
    return pos_out, meta


def _radial_mean_profile(
    radius: np.ndarray,
    values: np.ndarray,
    mass: np.ndarray,
    *,
    n_bins: int = 28,
    r_max: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mass-weighted mean of ``values`` vs R; NaN bins filled by interp."""
    edges = np.linspace(0.0, float(r_max), int(n_bins) + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(n_bins, np.nan, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    for i in range(n_bins):
        m = (radius >= edges[i]) & (radius < edges[i + 1])
        if not np.any(m):
            continue
        w = mass[m]
        if float(np.sum(w)) <= 0.0:
            continue
        out[i] = float(np.average(values[m], weights=w))
    good = np.isfinite(out)
    if not np.any(good):
        return r_mid, np.zeros_like(r_mid)
    if not np.all(good):
        out[~good] = np.interp(r_mid[~good], r_mid[good], out[good])
    return r_mid, out


def transport_ot_lite(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    n_bins_r: int = 64,
    n_bins_vphi: int = 28,
    r_max_disk: float = 15.0,
    r_max_sphere: float = 40.0,
    match_vphi: bool = True,
    match_dispersion: bool = True,
    z_scale_disk: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    OT-lite (option F): radial mass remap + ⟨v_φ⟩(R) / σ rescale from source→target.

    Starts from a trusted DF (``source``, e.g. θ-nearest GalactICS / library) and
    transports positions so each component's mass-weighted radial CDF matches
    ``target`` (OOD GalactICS IC). Disk uses cylindrical ``R``; bulge/halo use
    spherical ``r``. Azimuth / angular structure of the source is preserved
    (φ kept; unit-vector direction kept for sphere).

    Velocities: for the disk, replace the streaming ⟨v_φ⟩ with the target
    profile at the new ``R`` while keeping source residuals; optionally rescale
    ``σ_R, σ_z`` residuals to the target radial profiles. Bulge/halo keep
    source velocities (spatial remap only) — enough to cut dens offset.

    Returns transported particle dict + meta diagnostics.
    """
    pos_s = np.asarray(source["pos"], dtype=np.float64).copy()
    vel_s = np.asarray(source["vel"], dtype=np.float64).copy()
    mass_s = np.asarray(source["mass"], dtype=np.float64).copy()
    cid_s = np.asarray(source["component_id"])
    pos_t = np.asarray(target["pos"], dtype=np.float64)
    vel_t = np.asarray(target["vel"], dtype=np.float64)
    mass_t = np.asarray(target["mass"], dtype=np.float64)
    cid_t = np.asarray(target["component_id"])

    pos_out = pos_s.copy()
    vel_out = vel_s.copy()
    meta: dict = {"method": "ot_lite_radial_vphi", "components": {}}

    # --- disk: cylindrical R CDF + v_φ ---
    disk_s = cid_s == COMPONENT_IDS["disk"]
    disk_t = cid_t == COMPONENT_IDS["disk"]
    if np.any(disk_s) and np.any(disk_t):
        Rs = np.hypot(pos_s[disk_s, 0], pos_s[disk_s, 1])
        Rt = np.hypot(pos_t[disk_t, 0], pos_t[disk_t, 1])
        e_s, c_s = _mass_weighted_cdf_edges(
            Rs, mass_s[disk_s], n_bins=n_bins_r, r_max=r_max_disk
        )
        e_t, c_t = _mass_weighted_cdf_edges(
            Rt, mass_t[disk_t], n_bins=n_bins_r, r_max=r_max_disk
        )
        # u = CDF_src(R_s); R_new = inv_CDF_tgt(u)
        u = np.interp(Rs, e_s, c_s)
        R_new = _invert_cdf(u, e_t, c_t)
        scale_R = np.ones_like(Rs)
        ok = Rs > 1e-8
        scale_R[ok] = R_new[ok] / Rs[ok]
        idx = np.flatnonzero(disk_s)
        pos_out[idx, 0] = pos_s[idx, 0] * scale_R
        pos_out[idx, 1] = pos_s[idx, 1] * scale_R
        med_s = float(np.median(Rs[Rs > 0.1])) if np.any(Rs > 0.1) else 1.0
        med_t = float(np.median(Rt[Rt > 0.1])) if np.any(Rt > 0.1) else med_s
        z_fac = 1.0
        if z_scale_disk:
            # Mild vertical stretch by median R ratio (preserves disk thinness order).
            z_fac = float(np.clip(med_t / max(med_s, 1e-6), 0.5, 2.0))
            pos_out[idx, 2] = pos_s[idx, 2] * z_fac

        vphi_meta: dict = {
            "z_scale": z_fac,
            "r_med_src": med_s,
            "r_med_tgt": med_t,
        }
        if match_vphi:
            vphi_s = (
                -pos_s[idx, 1] * vel_s[idx, 0] + pos_s[idx, 0] * vel_s[idx, 1]
            ) / np.maximum(Rs, 1e-8)
            vphi_t_all = (
                -pos_t[disk_t, 1] * vel_t[disk_t, 0]
                + pos_t[disk_t, 0] * vel_t[disk_t, 1]
            ) / np.maximum(Rt, 1e-8)
            # cylindrical v_R, v_z
            vr_s = (
                pos_s[idx, 0] * vel_s[idx, 0] + pos_s[idx, 1] * vel_s[idx, 1]
            ) / np.maximum(Rs, 1e-8)
            vz_s = vel_s[idx, 2]

            r_mid_s, mean_s = _radial_mean_profile(
                Rs, vphi_s, mass_s[idx], n_bins=n_bins_vphi, r_max=r_max_disk
            )
            _, mean_t = _radial_mean_profile(
                Rt, vphi_t_all, mass_t[disk_t], n_bins=n_bins_vphi, r_max=r_max_disk
            )
            mean_s_at_old = np.interp(Rs, r_mid_s, mean_s)
            mean_t_at_new = np.interp(R_new, r_mid_s, mean_t)
            resid = vphi_s - mean_s_at_old

            if match_dispersion:
                # mass-weighted σ profiles of residuals
                def _sig_prof(rad, val, mass, mean_at):
                    resid_ = val - mean_at
                    edges = np.linspace(0.0, float(r_max_disk), int(n_bins_vphi) + 1)
                    r_mid = 0.5 * (edges[:-1] + edges[1:])
                    sig = np.full(n_bins_vphi, np.nan)
                    for i in range(n_bins_vphi):
                        m = (rad >= edges[i]) & (rad < edges[i + 1])
                        if not np.any(m):
                            continue
                        w = mass[m]
                        if float(np.sum(w)) <= 0:
                            continue
                        sig[i] = float(np.sqrt(np.average(resid_[m] ** 2, weights=w)))
                    good = np.isfinite(sig) & (sig > 1e-8)
                    if np.any(good):
                        sig[~good] = np.interp(r_mid[~good], r_mid[good], sig[good])
                    else:
                        sig[:] = 1.0
                    return r_mid, sig

                # For source use old R + old mean; for target use Rt + target mean
                mean_t_at_Rt = np.interp(Rt, r_mid_s, mean_t)
                _, sig_s = _sig_prof(Rs, vphi_s, mass_s[idx], mean_s_at_old)
                _, sig_t = _sig_prof(Rt, vphi_t_all, mass_t[disk_t], mean_t_at_Rt)
                sig_s_p = np.interp(Rs, r_mid_s, sig_s)
                sig_t_p = np.interp(R_new, r_mid_s, sig_t)
                scale_sig = np.ones_like(resid)
                ok_s = sig_s_p > 1e-8
                scale_sig[ok_s] = sig_t_p[ok_s] / sig_s_p[ok_s]
                scale_sig = np.clip(scale_sig, 0.25, 4.0)
                resid = resid * scale_sig

                # Also lightly match σ_R / σ_z amplitudes (global scale per R)
                _, sigR_s = _sig_prof(Rs, vr_s, mass_s[idx], np.zeros_like(vr_s))
                vr_t = (
                    pos_t[disk_t, 0] * vel_t[disk_t, 0]
                    + pos_t[disk_t, 1] * vel_t[disk_t, 1]
                ) / np.maximum(Rt, 1e-8)
                _, sigR_t = _sig_prof(Rt, vr_t, mass_t[disk_t], np.zeros_like(vr_t))
                sR_s = np.interp(Rs, r_mid_s, sigR_s)
                sR_t = np.interp(R_new, r_mid_s, sigR_t)
                sc_r = np.ones_like(vr_s)
                ok_r = sR_s > 1e-8
                sc_r[ok_r] = np.clip(sR_t[ok_r] / sR_s[ok_r], 0.25, 4.0)
                vr_new = vr_s * sc_r

                _, sigz_s = _sig_prof(Rs, vz_s, mass_s[idx], np.zeros_like(vz_s))
                vz_t = vel_t[disk_t, 2]
                _, sigz_t = _sig_prof(Rt, vz_t, mass_t[disk_t], np.zeros_like(vz_t))
                sz_s = np.interp(Rs, r_mid_s, sigz_s)
                sz_t = np.interp(R_new, r_mid_s, sigz_t)
                sc_z = np.ones_like(vz_s)
                ok_z = sz_s > 1e-8
                sc_z[ok_z] = np.clip(sz_t[ok_z] / sz_s[ok_z], 0.25, 4.0)
                vz_new = vz_s * sc_z
            else:
                vr_new = vr_s
                vz_new = vz_s

            vphi_new = mean_t_at_new + resid
            # rebuild Cartesian at new (x,y)
            x_n, y_n = pos_out[idx, 0], pos_out[idx, 1]
            Rn = np.maximum(np.hypot(x_n, y_n), 1e-8)
            c = x_n / Rn
            s = y_n / Rn
            vel_out[idx, 0] = vr_new * c - vphi_new * s
            vel_out[idx, 1] = vr_new * s + vphi_new * c
            vel_out[idx, 2] = vz_new
            vphi_meta["match_vphi"] = True
            vphi_meta["match_dispersion"] = bool(match_dispersion)
            vphi_meta["mean_vphi_src_med"] = float(np.nanmedian(mean_s))
            vphi_meta["mean_vphi_tgt_med"] = float(np.nanmedian(mean_t))
        meta["components"]["disk"] = {
            "n": int(idx.size),
            "R_med_src": float(np.median(Rs)),
            "R_med_new": float(np.median(R_new)),
            **vphi_meta,
        }

    # --- bulge / halo: spherical r CDF (positions only) ---
    for name in ("bulge", "halo"):
        cid = COMPONENT_IDS[name]
        m_s = cid_s == cid
        m_t = cid_t == cid
        if not (np.any(m_s) and np.any(m_t)):
            continue
        rs = np.linalg.norm(pos_s[m_s], axis=1)
        rt = np.linalg.norm(pos_t[m_t], axis=1)
        r_max = r_max_sphere if name == "halo" else min(r_max_sphere, 12.0)
        e_s, c_s = _mass_weighted_cdf_edges(rs, mass_s[m_s], n_bins=n_bins_r, r_max=r_max)
        e_t, c_t = _mass_weighted_cdf_edges(rt, mass_t[m_t], n_bins=n_bins_r, r_max=r_max)
        u = np.interp(rs, e_s, c_s)
        r_new = _invert_cdf(u, e_t, c_t)
        scale = np.ones_like(rs)
        ok = rs > 1e-8
        scale[ok] = r_new[ok] / rs[ok]
        idx = np.flatnonzero(m_s)
        pos_out[idx] = pos_s[idx] * scale[:, None]
        meta["components"][name] = {
            "n": int(idx.size),
            "r_med_src": float(np.median(rs)),
            "r_med_new": float(np.median(r_new)),
        }

    out = {
        "pos": pos_out,
        "vel": vel_out,
        "mass": mass_s,
        "component_id": cid_s,
    }
    if "eps" in source:
        out["eps"] = np.asarray(source["eps"]).copy()
    return out, meta


def resample_particles_from_slice_stack(
    stack: np.ndarray,
    *,
    cfg: SliceMapConfig,
    n_particles: int = 50_000,
    mass_total_per_component: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
    components: tuple[str, ...] | None = None,
    sample_dispersion: bool = True,
    velocity_frame: str = "cartesian",
    match_cell_moments: bool = False,
) -> dict[str, np.ndarray]:
    """
    Draw particles from a **shared-geometry** slice density (+ moment) stack.

    Prefer :func:`resample_particles_from_multiscale` when FOVs differ per
    component.
    """
    rng = rng or np.random.default_rng(0)
    stack = np.asarray(stack, dtype=np.float64)
    comps = tuple(components or cfg.components)
    n_mom = cfg.n_mom
    n_z = int(cfg.n_z)
    n_pix = int(cfg.n_pix)
    assert stack.shape[-2:] == (n_pix, n_pix)

    grids = []
    maps: dict[str, np.ndarray] = {}
    for ic, comp in enumerate(cfg.components):
        if comp not in comps:
            continue
        g = ComponentSliceGrid(
            name=comp,
            n_pix=n_pix,
            n_z=n_z,
            r_max=cfg.r_max,
            z_max=cfg.z_max,
            z_spacing="uniform",
            moment_set=cfg.moment_set,
        )
        grids.append(g)
        block = n_z * n_mom
        maps[comp] = stack[ic * block : (ic + 1) * block]
    mcfg = MultiScaleSliceConfig(grids=tuple(grids))
    return resample_particles_from_multiscale(
        maps,
        cfg=mcfg,
        n_particles=n_particles,
        mass_total_per_component=mass_total_per_component,
        rng=rng,
        sample_dispersion=sample_dispersion,
        velocity_frame=velocity_frame,
        match_cell_moments=match_cell_moments,
    )
