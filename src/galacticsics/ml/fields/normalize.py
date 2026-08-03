"""Normalize field stacks for CNN training and invert for diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from galacticsics.ml.fields.binning import (
    MOMENT_KEYS,
    ComponentSliceGrid,
    SliceMapConfig,
    dens_channel_indices,
    dens_channel_indices_component,
    moment_kind,
    resolve_moment_keys,
)


@dataclass
class FieldNormStats:
    """Per-channel affine stats for dens / vel / disp / optional potential."""

    dens_scale: float = 1.0
    vel_scale: float = 2.0
    disp_scale: float = 1.0
    phi_scale: float = 1.0
    phi_offset: float = 0.0
    n_moment_channels: int = 0
    n_phi_channels: int = 0
    n_mom: int = 7
    moment_keys: tuple[str, ...] = MOMENT_KEYS


def _channel_kind(ch: int, moment_keys: tuple[str, ...]) -> str:
    n_mom = len(moment_keys)
    if n_mom <= 0:
        return "vel"
    key = moment_keys[ch % n_mom]
    return moment_kind(key)


def estimate_norm_stats(
    stacks: list[np.ndarray],
    *,
    cfg: SliceMapConfig | None = None,
    n_moment_channels: int | None = None,
    n_phi: int = 0,
    dens_indices: list[int] | None = None,
    moment_keys: tuple[str, ...] | None = None,
) -> FieldNormStats:
    """
    Estimate robust scales from a list of ``(C, H, W)`` stacks.

    Density / dispersion use median of positive pixels; velocity uses 90th
    percentile of |v|; potential uses MAD of Φ channels.
    """
    if moment_keys is None:
        moment_keys = cfg.moment_keys if cfg is not None else MOMENT_KEYS
    if dens_indices is None:
        if cfg is None:
            raise ValueError("need dens_indices or cfg")
        dens_indices = dens_channel_indices(cfg)
    if n_moment_channels is None:
        if cfg is None:
            raise ValueError("need n_moment_channels or cfg")
        n_moment_channels = len(cfg.components) * int(cfg.n_z) * len(moment_keys)

    dens_vals: list[np.ndarray] = []
    vel_vals: list[np.ndarray] = []
    disp_vals: list[np.ndarray] = []
    phi_vals: list[np.ndarray] = []
    n_mom_ch = int(n_moment_channels)
    for s in stacks:
        s = np.asarray(s)
        for i in dens_indices:
            if i >= s.shape[0]:
                continue
            d = s[i]
            dens_vals.append(d[d > 0])
        for ch in range(min(n_mom_ch, s.shape[0])):
            kind = _channel_kind(ch, moment_keys)
            if kind == "dens":
                continue
            if kind == "disp":
                d = s[ch]
                disp_vals.append(d[d > 0])
            elif kind == "vel":
                vel_vals.append(np.abs(s[ch]).ravel())
            # aniso (beta) uses fixed scale in normalize
        if n_phi > 0 and s.shape[0] >= n_mom_ch + n_phi:
            phi_vals.append(s[n_mom_ch : n_mom_ch + n_phi].ravel())

    dens_cat = np.concatenate(dens_vals) if dens_vals else np.asarray([1.0])
    dens_scale = float(np.median(dens_cat)) if dens_cat.size else 1.0
    dens_scale = max(dens_scale, 1e-12)

    vel_cat = np.concatenate(vel_vals) if vel_vals else np.asarray([1.0])
    vel_scale = float(np.percentile(vel_cat, 90)) if vel_cat.size else 2.0
    vel_scale = max(vel_scale, 0.1)

    disp_cat = np.concatenate(disp_vals) if disp_vals else np.asarray([1.0])
    disp_scale = float(np.median(disp_cat)) if disp_cat.size else 1.0
    disp_scale = max(disp_scale, 1e-6)

    phi_scale = 1.0
    phi_offset = 0.0
    if phi_vals:
        phi_cat = np.concatenate(phi_vals)
        phi_offset = float(np.median(phi_cat))
        phi_scale = float(np.median(np.abs(phi_cat - phi_offset)))
        phi_scale = max(phi_scale, 1e-6)

    return FieldNormStats(
        dens_scale=dens_scale,
        vel_scale=vel_scale,
        disp_scale=disp_scale,
        phi_scale=phi_scale,
        phi_offset=phi_offset,
        n_moment_channels=n_mom_ch,
        n_phi_channels=int(n_phi),
        n_mom=len(moment_keys),
        moment_keys=tuple(moment_keys),
    )


def estimate_norm_stats_component(
    stacks: list[np.ndarray],
    grid: ComponentSliceGrid,
    *,
    n_phi: int = 0,
) -> FieldNormStats:
    """Norm stats for one component's native ``(C, H, W)`` stacks."""
    return estimate_norm_stats(
        stacks,
        n_moment_channels=grid.n_moment_channels,
        n_phi=n_phi,
        dens_indices=dens_channel_indices_component(grid),
        moment_keys=grid.moment_keys,
    )


def normalize_stack(stack: np.ndarray, stats: FieldNormStats) -> np.ndarray:
    """Map physical channels → roughly O(1) training targets."""
    s = np.asarray(stack, dtype=np.float32).copy()
    n_mom = stats.n_moment_channels
    keys = stats.moment_keys or resolve_moment_keys("disp")
    for ch in range(min(n_mom, s.shape[0])):
        kind = _channel_kind(ch, keys)
        if kind == "dens":
            s[ch] = np.log1p(np.maximum(s[ch], 0.0) / stats.dens_scale)
        elif kind == "disp":
            s[ch] = np.log1p(np.maximum(s[ch], 0.0) / stats.disp_scale)
        elif kind == "aniso":
            s[ch] = np.tanh(s[ch] / 2.0)
        else:
            s[ch] = np.tanh(s[ch] / stats.vel_scale)
    if stats.n_phi_channels > 0 and s.shape[0] >= n_mom + stats.n_phi_channels:
        phi = s[n_mom : n_mom + stats.n_phi_channels]
        s[n_mom : n_mom + stats.n_phi_channels] = (
            (phi - stats.phi_offset) / stats.phi_scale
        )
    return s


def denormalize_stack(stack: np.ndarray, stats: FieldNormStats) -> np.ndarray:
    """Invert :func:`normalize_stack` (approx for tanh velocities)."""
    s = np.asarray(stack, dtype=np.float32).copy()
    n_mom = stats.n_moment_channels
    keys = stats.moment_keys or resolve_moment_keys("disp")
    for ch in range(min(n_mom, s.shape[0])):
        kind = _channel_kind(ch, keys)
        if kind == "dens":
            s[ch] = np.expm1(np.maximum(s[ch], 0.0)) * stats.dens_scale
            s[ch] = np.maximum(s[ch], 0.0)
        elif kind == "disp":
            s[ch] = np.expm1(np.maximum(s[ch], 0.0)) * stats.disp_scale
            s[ch] = np.maximum(s[ch], 0.0)
        elif kind == "aniso":
            s[ch] = np.clip(s[ch], -0.999, 0.999)
            s[ch] = np.arctanh(s[ch]) * 2.0
        else:
            s[ch] = np.clip(s[ch], -0.999, 0.999)
            s[ch] = np.arctanh(s[ch]) * stats.vel_scale
    if stats.n_phi_channels > 0 and s.shape[0] >= n_mom + stats.n_phi_channels:
        phi = s[n_mom : n_mom + stats.n_phi_channels]
        s[n_mom : n_mom + stats.n_phi_channels] = (
            phi * stats.phi_scale + stats.phi_offset
        )
    return s


def append_phi_channels(moment_stack: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Concatenate ``(n_z, H, W)`` Φ onto a moment stack as trailing channels."""
    mom = np.asarray(moment_stack, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    if phi.ndim != 3:
        raise ValueError(f"phi must be (n_z, H, W), got {phi.shape}")
    if phi.shape[-2:] != mom.shape[-2:]:
        raise ValueError(f"phi spatial {phi.shape[-2:]} != stack {mom.shape[-2:]}")
    return np.concatenate([mom, phi.astype(np.float32)], axis=0)
