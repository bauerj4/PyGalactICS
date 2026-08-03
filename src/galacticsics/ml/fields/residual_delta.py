"""Phase-B learned dens-contrast residual on GalactICS f0 midplane maps.

Predicts a multiplicative midplane contrast field that is injected with the
same axisym / ring-renorm guards as Phase A
(``inject_predicted_contrast_on_f0_dens``).

Architectures
-------------
- ``TinyDelta``: f0-only (2ch) — prior smoke; typically collapses to δ≈0.
- ``CondDelta``: f0 + morph-hint contrast (3ch) — learns to sharpen / correct
  a soft morph chart into a data-like m=2 residual.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class TinyDelta(nn.Module):
    """f0-only midplane contrast CNN (2 → 1)."""

    def __init__(self, width: int = 32, n_in: int = 2):
        super().__init__()
        self.n_in = int(n_in)
        self.width = int(width)
        w = self.width
        self.net = nn.Sequential(
            nn.Conv2d(self.n_in, w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CondDelta(nn.Module):
    """Morph-conditioned residual CNN with a light U-Net skip.

    Channels: [log1p dens_f0, log1p ax_f0, morph_contrast_hint] → contrast.
    """

    def __init__(self, width: int = 48, n_in: int = 3):
        super().__init__()
        self.n_in = int(n_in)
        self.width = int(width)
        w = self.width
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.n_in, w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(w, 2 * w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = nn.Sequential(
            nn.Conv2d(3 * w, w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        u = self.up(e2)
        if u.shape[-2:] != e1.shape[-2:]:
            u = nn.functional.interpolate(
                u, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.dec(torch.cat([u, e1], dim=1))


def build_delta_model(
    *,
    arch: str = "cond",
    width: int = 48,
    n_in: int | None = None,
) -> nn.Module:
    arch = str(arch).lower().strip()
    if arch in ("cond", "cond_delta", "unet"):
        return CondDelta(width=width, n_in=int(n_in if n_in is not None else 3))
    if arch in ("tiny", "tiny_delta", "f0"):
        return TinyDelta(width=width, n_in=int(n_in if n_in is not None else 2))
    raise ValueError(f"unknown Phase-B arch={arch!r}")


def _v1_tiny() -> nn.Module:
    """Legacy 3-layer width-16 net from phase_b/ smoke."""

    class _TinyDeltaV1(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1),
            )

        def forward(self, xx):
            return self.net(xx)

    return _TinyDeltaV1()


def load_delta_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a Phase-B ckpt; returns (eval-mode model, meta dict)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt["model"]
    arch = str(ckpt.get("arch", "")).lower().strip()
    width = int(ckpt.get("width", 32))
    n_in = ckpt.get("n_in")
    if n_in is None:
        # Infer from first conv weight.
        w0 = next(v for k, v in state.items() if k.endswith("weight") and v.ndim == 4)
        n_in = int(w0.shape[1])
    n_in = int(n_in)

    # Legacy v1: flat Sequential with 3 convs, n_in=2, width=16.
    n_layers = sum(1 for k in state if k.startswith("net.") and k.endswith(".weight"))
    if not arch and n_layers == 3 and n_in == 2:
        net = _v1_tiny()
        net.load_state_dict(state)
        net.eval()
        return net, ckpt

    if not arch:
        # Heuristic: CondDelta has enc1.* keys; TinyDelta has net.*
        arch = "cond" if any(k.startswith("enc1.") for k in state) else "tiny"

    net = build_delta_model(arch=arch, width=width, n_in=n_in)
    net.load_state_dict(state)
    net.eval()
    return net, ckpt


def predict_contrast(
    net: nn.Module,
    dens_f0: np.ndarray,
    *,
    morph_contrast: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run net on midplane maps; returns float32 contrast (H, W)."""
    from galacticsics.ml.fields.resample import _axisym_and_rbin

    dens0 = np.asarray(dens_f0, dtype=np.float32)
    ax0, _ = _axisym_and_rbin(np.maximum(dens0.astype(np.float64), 0.0))
    chans = [
        np.log1p(np.maximum(dens0, 0.0)),
        np.log1p(np.maximum(ax0.astype(np.float32), 0.0)),
    ]
    n_in = int(getattr(net, "n_in", 2))
    if n_in >= 3:
        if morph_contrast is None:
            hint = np.zeros_like(dens0, dtype=np.float32)
        else:
            hint = np.asarray(morph_contrast, dtype=np.float32)
            if hint.shape != dens0.shape:
                raise ValueError(
                    f"morph_contrast shape {hint.shape} != dens {dens0.shape}"
                )
        chans.append(hint)
    x = np.stack(chans[:n_in], axis=0)[None]
    dev = torch.device(device)
    net = net.to(dev)
    with torch.no_grad():
        y = net(torch.from_numpy(x).to(dev)).detach().cpu().numpy()[0, 0]
    return np.asarray(y, dtype=np.float32)


def midplane_contrast_from_dens(
    dens: np.ndarray,
    *,
    eps: float = 1e-8,
    contrast_clip: float = 3.0,
    m2_only: bool = True,
    r_max: float = 14.0,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Axisym-locked multiplicative contrast, optional blur + m=2 project."""
    from galacticsics.ml.fields.resample import m2_residual_field

    d = np.asarray(dens, dtype=np.float64)
    if blur_sigma and float(blur_sigma) > 0.0:
        from scipy.ndimage import gaussian_filter

        d = gaussian_filter(d, sigma=float(blur_sigma))
    # Local soft axisym (same bins as inject).
    from galacticsics.ml.fields.resample import _axisym_and_rbin

    ax, _ = _axisym_and_rbin(np.maximum(d, 0.0))
    resid = d - ax
    if m2_only:
        resid = m2_residual_field(resid, r_max=float(r_max), n_bins=16)
    contrast = resid / np.maximum(ax, float(eps))
    return np.clip(contrast, -float(contrast_clip), float(contrast_clip)).astype(
        np.float32
    )
