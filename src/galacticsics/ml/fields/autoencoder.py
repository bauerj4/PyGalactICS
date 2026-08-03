"""2-D CNN / U-Net autoencoders for multi-scale slice stacks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from galacticsics.ml.fields.binning import MultiScaleSliceConfig


@dataclass
class SliceAutoencoderConfig:
    """Architecture knobs for :class:`SliceAutoencoder` / :class:`SliceUNet`."""

    in_channels: int = 70
    base_channels: int = 32
    latent_channels: int = 64
    n_pix: int = 64
    n_z: int = 10
    n_mom: int = 7
    separate_heads: bool = True
    deep_heads: bool = True
    """If True, dens/moment heads are 3×3 stacks; False = 1×1 (crisp_2026-07-24)."""


class SliceAutoencoder(nn.Module):
    """
    Compact conv autoencoder: ``(B, C, H, W) → (B, C, H, W)``.

    Two stride-2 downsamples to ``H/4``, bottleneck 1×1 conv, then two
    transposed-conv upsamples.  Kept for small / baseline runs.
    """

    def __init__(self, cfg: SliceAutoencoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SliceAutoencoderConfig()
        c_in = int(self.cfg.in_channels)
        c = int(self.cfg.base_channels)
        z = int(self.cfg.latent_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(c_in, c, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * c, z, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z, 2 * c, 1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c, c_in, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SliceUNet(nn.Module):
    """
    U-Net with skip connections and optional separate dens / moment heads.

    Three downsamples (suitable for 48–128²).  Density head predicts the dens
    channel of each z-slab; moment head predicts the remaining channels.  This
    keeps bar morphology from being washed out by velocity MSE.
    """

    def __init__(self, cfg: SliceAutoencoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SliceAutoencoderConfig()
        c_in = int(self.cfg.in_channels)
        c = int(self.cfg.base_channels)
        z = int(self.cfg.latent_channels)
        self.n_mom = int(self.cfg.n_mom)
        self.n_z = int(self.cfg.n_z)
        self.separate_heads = bool(self.cfg.separate_heads)

        self.enc1 = _ConvBlock(c_in, c)
        self.down1 = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.enc2 = _ConvBlock(c, 2 * c)
        self.down2 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)
        self.enc3 = _ConvBlock(2 * c, 4 * c)
        self.down3 = nn.Conv2d(4 * c, 4 * c, 3, stride=2, padding=1)
        # Deeper bottleneck (two residual-ish blocks) helps 96–128² morphology.
        self.bottleneck = nn.Sequential(
            _ConvBlock(4 * c, z),
            _ConvBlock(z, z),
            nn.Conv2d(z, 4 * c, 1),
            nn.GELU(),
        )
        self.up3 = nn.ConvTranspose2d(4 * c, 4 * c, 4, stride=2, padding=1)
        self.dec3 = _ConvBlock(8 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1)
        self.dec2 = _ConvBlock(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)
        self.dec1 = _ConvBlock(2 * c, c)

        if self.separate_heads and self.n_mom > 0 and c_in % self.n_mom == 0:
            n_z = c_in // self.n_mom
            n_dens = n_z  # one dens channel per slab
            n_other = c_in - n_dens
            if bool(cfg.deep_heads):
                # Deeper heads (not 1×1 alone) preserve high-frequency dens/σ structure.
                self.dens_head = nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(c, c, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(c, n_dens, 1),
                )
                self.moment_head = nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(c, n_other, 1),
                )
            else:
                # crisp_2026-07-24 checkpoint layout (best particle A₂).
                self.dens_head = nn.Conv2d(c, n_dens, 1)
                self.moment_head = nn.Conv2d(c, n_other, 1)
            self._n_dens = n_dens
            self._n_other = n_other
            self.out_proj = None
        else:
            self.dens_head = None
            self.moment_head = None
            self.out_proj = nn.Conv2d(c, c_in, 3, padding=1)
            self._n_dens = 0
            self._n_other = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, _skips = self.encode_with_skips(x)
        return b

    def encode_with_skips(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return bottleneck + U-Net skips ``(e1, e2, e3)`` (bars live in skips)."""
        e1 = self.enc1(x)
        e2 = self.enc2(F.gelu(self.down1(e1)))
        e3 = self.enc3(F.gelu(self.down2(e2)))
        b = self.bottleneck(F.gelu(self.down3(e3)))
        return b, (e1, e2, e3)

    def decode_from_features(
        self,
        bottleneck: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        target_hw: tuple[int, int] | None = None,
        n_out_channels: int | None = None,
    ) -> torch.Tensor:
        """Decode from bottleneck + skips (no re-encode). Used by latent skip synth."""
        e1, e2, e3 = skips
        d3 = self.up3(bottleneck)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.dec1(torch.cat([d1, e1], dim=1))
        if target_hw is not None and feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return self._heads(feat, n_out_channels=n_out_channels)

    def _decode_features(self, x: torch.Tensor) -> torch.Tensor:
        b, skips = self.encode_with_skips(x)
        e1, e2, e3 = skips
        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec1(torch.cat([d1, e1], dim=1))

    def _interleave_heads(self, dens: torch.Tensor, moments: torch.Tensor) -> torch.Tensor:
        """Pack dens / other channels back into ``(B, n_z · n_mom, H, W)``."""
        b, _, h, w = dens.shape
        n_z = self._n_dens
        n_other_per = self.n_mom - 1
        dens_z = dens.view(b, n_z, 1, h, w)
        mom_z = moments.view(b, n_z, n_other_per, h, w)
        return torch.cat([dens_z, mom_z], dim=2).reshape(b, n_z * self.n_mom, h, w)

    def _heads(
        self, feat: torch.Tensor, *, n_out_channels: int | None = None
    ) -> torch.Tensor:
        if self.dens_head is None:
            assert self.out_proj is not None
            return self.out_proj(feat)
        dens = self.dens_head(feat)
        moments = self.moment_head(feat)
        n_core = dens.shape[1] * self.n_mom
        core = self._interleave_heads(dens, moments)
        n_out = int(n_out_channels) if n_out_channels is not None else n_core
        if n_out == n_core:
            return core
        extra_n = n_out - n_core
        if not hasattr(self, "extra_head") or self.extra_head is None:
            self.extra_head = nn.Conv2d(feat.shape[1], extra_n, 1).to(feat.device)
        return torch.cat([core, self.extra_head(feat)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._decode_features(x)
        return self._heads(feat, n_out_channels=x.shape[1])


class CrossTowerFusion(nn.Module):
    """
    Light attention across tower bottleneck vectors (disk ↔ bulge ↔ halo).

    Each tower still reconstructs at native resolution; fusion only mixes a
    pooled latent so morphology cues can transfer without resampling FOVs.
    """

    def __init__(self, n_towers: int, latent_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_towers = int(n_towers)
        self.proj_in = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(n_towers)]
        )
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.proj_out = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(n_towers)]
        )

    def forward(self, pooled: list[torch.Tensor]) -> list[torch.Tensor]:
        # pooled[i]: (B, D)
        tokens = torch.stack(
            [proj(p) for proj, p in zip(self.proj_in, pooled)], dim=1
        )
        fused, _ = self.attn(tokens, tokens, tokens)
        return [proj(fused[:, i]) for i, proj in enumerate(self.proj_out)]


class MultiTowerSliceAE(nn.Module):
    """
    Independent slice AE / U-Net per component (native FOV / channels / ``n_pix``).

    Optional cross-tower attention mixes pooled encoder cues so disk/bulge/halo
    towers can share morphology signals without resampling FOVs.
    """

    def __init__(
        self,
        cfg: MultiScaleSliceConfig,
        *,
        include_potential: bool = False,
        base_channels: int = 32,
        latent_channels: int = 64,
        arch: str = "unet",
        separate_heads: bool = True,
        cross_tower_attention: bool = True,
        deep_heads: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.include_potential = bool(include_potential)
        self.arch = str(arch)
        self.cross_tower_attention = bool(cross_tower_attention) and arch == "unet"
        towers = {}
        for g in cfg.grids:
            n_ch = g.n_moment_channels + (g.n_z if include_potential else 0)
            acfg = SliceAutoencoderConfig(
                in_channels=n_ch,
                base_channels=base_channels,
                latent_channels=latent_channels,
                n_pix=g.n_pix,
                n_z=g.n_z,
                n_mom=g.n_mom,
                separate_heads=separate_heads,
                deep_heads=deep_heads,
            )
            if arch == "unet":
                towers[g.name] = SliceUNet(acfg)
            else:
                towers[g.name] = SliceAutoencoder(acfg)
        self.towers = nn.ModuleDict(towers)
        self._tower_order = list(towers.keys())

        if self.cross_tower_attention and len(towers) > 1:
            zdim = int(latent_channels)
            self.fusion = CrossTowerFusion(len(towers), zdim)
            self.pool_to_z = nn.ModuleDict(
                {name: nn.Linear(base_channels, zdim) for name in towers}
            )
            self.cue_to_ch = nn.ModuleDict(
                {
                    name: nn.Linear(zdim, towers[name].cfg.in_channels)
                    for name in towers
                }
            )
        else:
            self.fusion = None
            self.pool_to_z = None
            self.cue_to_ch = None

    def encode_features(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]]:
        """Per-tower bottleneck + skips (no fusion). Bars live in skips."""
        out: dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]] = {}
        for name in self._tower_order:
            if name not in batch:
                continue
            tower = self.towers[name]
            if not hasattr(tower, "encode_with_skips"):
                raise TypeError(f"tower {name} lacks encode_with_skips (need arch=unet)")
            b, skips = tower.encode_with_skips(batch[name])
            out[name] = {"bottleneck": b, "skips": skips}
        return out

    def decode_features(
        self,
        features: dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]],
        *,
        target_shapes: dict[str, tuple[int, int]] | None = None,
        n_channels: dict[str, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode from teacher/student bottleneck+skips with frozen tower weights.

        Feature tensors may live on CPU (library storage); move them onto the
        tower device before decode so CUDA teachers work with CPU-cached skips.
        """
        out: dict[str, torch.Tensor] = {}
        for name in self._tower_order:
            if name not in features:
                continue
            tower = self.towers[name]
            device = next(tower.parameters()).device
            feat = features[name]
            b = feat["bottleneck"]  # type: ignore[assignment]
            skips = feat["skips"]  # type: ignore[assignment]
            assert isinstance(b, torch.Tensor)
            b = b.to(device=device, dtype=torch.float32, non_blocking=True)
            skips = tuple(
                s.to(device=device, dtype=torch.float32, non_blocking=True) for s in skips
            )
            hw = None if target_shapes is None else target_shapes.get(name)
            n_ch = None if n_channels is None else n_channels.get(name)
            out[name] = tower.decode_from_features(
                b, skips, target_hw=hw, n_out_channels=n_ch  # type: ignore[arg-type]
            )
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.fusion is None:
            return {
                name: self.towers[name](x)
                for name, x in batch.items()
                if name in self.towers
            }

        pooled = []
        for name in self._tower_order:
            tower = self.towers[name]
            e1 = tower.enc1(batch[name])
            vec = F.adaptive_avg_pool2d(e1, 1).flatten(1)
            pooled.append(self.pool_to_z[name](vec))
        fused = self.fusion(pooled)

        out: dict[str, torch.Tensor] = {}
        for name, cue in zip(self._tower_order, fused):
            y = self.towers[name](batch[name])
            shift = self.cue_to_ch[name](cue).view(y.shape[0], -1, 1, 1)
            out[name] = y + 0.05 * shift
        return out


def load_compatible_towers(
    model: MultiTowerSliceAE,
    state_dict: dict[str, torch.Tensor],
    *,
    towers: tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    """
    Load matching tower / fusion weights; skip incompatible tensors.

    Used to warm-start disk/halo from an FFT-long teacher while leaving a
    reconfigured (finer) bulge tower randomly initialised.  Returns
    ``{"loaded": [...], "skipped": [...]}`` key lists for logging.
    """
    own = model.state_dict()
    allow = None if towers is None else set(towers)
    loaded: list[str] = []
    skipped: list[str] = []
    filtered: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key not in own:
            skipped.append(key)
            continue
        if allow is not None:
            # Keys look like towers.disk.... / pool_to_z.disk.... / cue_to_ch.disk....
            parts = key.split(".")
            tower_name = None
            if parts[0] == "towers" and len(parts) > 1:
                tower_name = parts[1]
            elif parts[0] in ("pool_to_z", "cue_to_ch") and len(parts) > 1:
                tower_name = parts[1]
            if tower_name is not None and tower_name not in allow:
                skipped.append(key)
                continue
            if parts[0] == "fusion" and allow is not None:
                # Fusion mixes all towers; skip unless all towers are allowed.
                if set(model._tower_order) - allow:
                    skipped.append(key)
                    continue
        if tuple(own[key].shape) != tuple(tensor.shape):
            skipped.append(key)
            continue
        filtered[key] = tensor
        loaded.append(key)
    missing = model.load_state_dict(filtered, strict=False)
    skipped.extend(list(missing.missing_keys))
    skipped.extend(list(missing.unexpected_keys))
    return {"loaded": loaded, "skipped": skipped}


def soft_am_radial_from_dens_maps(
    dens: torch.Tensor,
    *,
    m: int = 2,
    n_bins: int = 12,
    map_half_width: float = 1.0,
    soft_width: float | None = None,
) -> dict[str, torch.Tensor]:
    """
    Soft radial profiles of azimuthal Fourier moments from ``(B, H, W)`` dens maps.

    ``A_m`` is **not** constant in radius — bars peak at finite ``R``, spirals
    wind, lopsidedness varies — so matching a single scalar (median / map-wide
    ``|A_m|/A₀``) washes out morphology.  This mirrors
    :func:`galacticsics.ml.profiles.soft_azimuthal_fourier` on particle clouds,
    but deposits dens-map pixels into soft cylindrical annuli.

    Map coordinates span ``[-map_half_width, map_half_width]²`` (default
    normalised FOV edge = 1).  Returns phase-aware ``cos`` / ``sin`` =
    ``Re/Im(a_m)/a_0`` and amplitude ``|a_m|/a_0`` per radial bin.
    """
    m = int(m)
    if m < 1:
        raise ValueError(f"Fourier order m must be >= 1, got {m}")
    _b, h, w = dens.shape
    half = float(map_half_width)
    yy, xx = torch.meshgrid(
        torch.linspace(-half, half, h, device=dens.device, dtype=dens.dtype),
        torch.linspace(-half, half, w, device=dens.device, dtype=dens.dtype),
        indexing="ij",
    )
    r = torch.sqrt(xx * xx + yy * yy + 1e-16)
    phi = torch.atan2(yy, xx)
    n_bins = int(n_bins)
    edges = torch.linspace(0.0, half, n_bins + 1, device=dens.device, dtype=dens.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = float(half) / max(n_bins, 1)
    sigma = float(soft_width) if soft_width is not None else 0.5 * bin_w
    # Soft radial membership (H, W, n_bins), mass-conserving across bins.
    logits = -0.5 * ((r.unsqueeze(-1) - centers.view(1, 1, -1)) / max(sigma, 1e-6)) ** 2
    inside = (r <= half * 1.05).to(dens.dtype)
    attn = torch.softmax(logits, dim=-1) * inside.unsqueeze(-1)
    wgt = dens.clamp_min(0.0).unsqueeze(-1)  # (B, H, W, 1)
    a0 = (wgt * attn).sum(dim=(-3, -2)).clamp_min(1e-8)  # (B, n_bins)
    m_f = float(m)
    cos_m = (wgt * attn * torch.cos(m_f * phi).unsqueeze(-1)).sum(dim=(-3, -2))
    sin_m = (wgt * attn * torch.sin(m_f * phi).unsqueeze(-1)).sum(dim=(-3, -2))
    cos_n = cos_m / a0
    sin_n = sin_m / a0
    amp = torch.sqrt(cos_n * cos_n + sin_n * sin_n + 1e-16)
    return {"a0": a0, "cos": cos_n, "sin": sin_n, "amp": amp, "r_mid": centers}


def soft_am_from_dens_maps(dens: torch.Tensor, m: int = 2) -> torch.Tensor:
    """
    Map-wide soft ``|A_m| / A₀`` (no radial bins) — diagnostics / legacy only.

    Prefer :func:`soft_am_radial_from_dens_maps` for losses: ``A_m`` varies with
    ``R``.  ``m`` is the harmonic order (1=lopsidedness, 2=bar/spiral, 3=triangular).
    """
    m = int(m)
    if m < 1:
        raise ValueError(f"Fourier order m must be >= 1, got {m}")
    _b, h, w = dens.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=dens.device, dtype=dens.dtype),
        torch.linspace(-1.0, 1.0, w, device=dens.device, dtype=dens.dtype),
        indexing="ij",
    )
    phi = torch.atan2(yy, xx)
    cm = torch.cos(float(m) * phi)
    sm = torch.sin(float(m) * phi)
    wgt = dens.clamp_min(0.0)
    a0 = wgt.sum(dim=(-2, -1)).clamp_min(1e-8)
    ac = (wgt * cm).sum(dim=(-2, -1))
    as_ = (wgt * sm).sum(dim=(-2, -1))
    return torch.sqrt(ac * ac + as_ * as_) / a0


def soft_a2_from_dens_maps(dens: torch.Tensor) -> torch.Tensor:
    """Map-wide soft m=2 Fourier amplitude (legacy scalar diagnostic)."""
    return soft_am_from_dens_maps(dens, m=2)


def radial_focus_bin_weights(
    r_mid: torch.Tensor,
    r_focus: torch.Tensor | float | None,
    *,
    band_lo_frac: float = 0.5,
    band_hi_frac: float = 1.5,
    peak: float = 4.0,
    floor: float = 0.25,
) -> torch.Tensor | None:
    """
    Soft radial bin weights peaking near ``r_focus`` (e.g. ``R_d``).

    ``r_mid`` is ``(nbins,)`` in the same units as ``r_focus`` (typically
    normalised FOV coords with ``map_half_width=1``).  ``r_focus`` may be a
    scalar or ``(B,)``.  Returns ``(B, nbins)`` or ``(1, nbins)``, or ``None``
    when focus is disabled.
    """
    if r_focus is None:
        return None
    if not torch.is_tensor(r_focus):
        rf = r_mid.new_tensor(float(r_focus)).view(1)
    else:
        rf = r_focus.to(device=r_mid.device, dtype=r_mid.dtype).reshape(-1)
    rf = rf.clamp_min(1e-4)
    # (B, 1) vs (nbins,)
    r = r_mid.view(1, -1)
    rf_b = rf.view(-1, 1)
    lo = float(band_lo_frac) * rf_b
    hi = float(band_hi_frac) * rf_b
    # Flat boost on [0.5, 1.5] R_d plus a Gaussian peak at R_d.
    in_band = ((r >= lo) & (r <= hi)).to(r_mid.dtype)
    sigma = (0.35 * rf_b).clamp_min(1e-3)
    gauss = torch.exp(-0.5 * ((r - rf_b) / sigma) ** 2)
    w = float(floor) + (1.0 - float(floor)) * in_band + (float(peak) - 1.0) * gauss
    return w


def soft_interp_amp_at_r(
    rad: dict[str, torch.Tensor],
    r_eval: torch.Tensor | float,
) -> torch.Tensor:
    """Linear-interp soft ``A_m(R)`` amplitude onto ``r_eval`` → ``(B,)``."""
    amp = rad["amp"]  # (B, nbins)
    r_mid = rad["r_mid"]  # (nbins,)
    if not torch.is_tensor(r_eval):
        re = amp.new_full((amp.shape[0],), float(r_eval))
    else:
        re = r_eval.to(device=amp.device, dtype=amp.dtype).reshape(-1)
        if re.numel() == 1 and amp.shape[0] > 1:
            re = re.expand(amp.shape[0])
    # Clamp into [r_mid[0], r_mid[-1]] for stable interp.
    re = re.clamp(min=float(r_mid[0]), max=float(r_mid[-1]))
    # Soft two-bin blend via distances (differentiable).
    # Find neighbors with soft weights ~ inverse distance to bin centers.
    d = (r_mid.view(1, -1) - re.view(-1, 1)).abs() + 1e-6
    # Focus on the two nearest bins via softmax over -d / bin_width.
    bin_w = float(r_mid[1] - r_mid[0]) if r_mid.numel() > 1 else 1.0
    attn = torch.softmax(-d / max(bin_w, 1e-6), dim=-1)
    return (attn * amp).sum(dim=-1)


def soft_a2_at_r_match_loss(
    pred_dens: torch.Tensor,
    target_dens: torch.Tensor,
    *,
    r_focus: torch.Tensor | float,
    n_bins: int = 12,
    map_half_width: float = 1.0,
    quiet_gate_floor: float = 0.05,
    quiet_gate_temp: float = 0.03,
    undershoot_weight: float = 2.0,
) -> torch.Tensor:
    """
    Match ``A₂(R = r_focus)`` (e.g. disk scale length ``R_d``).

    Absolute + relative amp MSE at the focus radius, with an extra hinge that
    penalises under-predicting bar strength at ``R_d`` (the α=1 fade mode).
    Quiet-gated by target ``A₂(R_d)``.
    """
    p = soft_am_radial_from_dens_maps(
        pred_dens, m=2, n_bins=n_bins, map_half_width=map_half_width
    )
    t = soft_am_radial_from_dens_maps(
        target_dens, m=2, n_bins=n_bins, map_half_width=map_half_width
    )
    pa = soft_interp_amp_at_r(p, r_focus)
    ta = soft_interp_amp_at_r(t, r_focus)
    d = pa - ta
    loss_abs = (d**2).mean()
    loss_rel = ((d / (ta + 0.08)) ** 2).mean()
    undershoot = torch.relu(ta - pa)
    loss_u = (undershoot**2).mean()
    gate = torch.sigmoid(
        (ta - float(quiet_gate_floor)) / max(float(quiet_gate_temp), 1e-6)
    )
    return (
        0.5 * loss_abs + 0.5 * loss_rel + float(undershoot_weight) * loss_u
    ) * gate.mean()


def soft_fourier_match_loss(
    pred_dens: torch.Tensor,
    target_dens: torch.Tensor,
    *,
    modes: tuple[int, ...] = (1, 2, 3),
    n_bins: int = 12,
    map_half_width: float = 1.0,
    lambda_phase: float = 1.0,
    amp_weight: float = 2.0,
    weight_by_a0: bool = True,
    weight_by_target_amp: bool = True,
    quiet_gate_floor: float = 0.05,
    quiet_gate_temp: float = 0.03,
    r_focus: torch.Tensor | float | None = None,
    r_focus_band: tuple[float, float] = (0.5, 1.5),
    r_focus_peak: float = 4.0,
    r_focus_floor: float = 0.25,
) -> torch.Tensor:
    """
    Match radial ``A_m(R)`` (amp + phase cos/sin) between dens maps.

    Defaults to modes ``m=1,2,3``.  Phase-aware terms keep bar angle / spiral
    handedness.  Extra amp weight + A₀/target-amp bin weights push peak heights
    to match (crisp Fourier-off under-predicted ``A_m``).  ``quiet_gate_*``
    soft-gates the loss by target strength so quiet ICs are not forced to invent
    bars (the failure mode of ungated Fourier).

    When ``r_focus`` is set (e.g. ``R_d / r_max`` in normalised FOV coords),
    radial bins near that ring (default band ``0.5–1.5×R_d``) are up-weighted
    so morphology match targets bar strength at the disk scale length.
    """
    losses: list[torch.Tensor] = []
    for m in modes:
        p = soft_am_radial_from_dens_maps(
            pred_dens, m=int(m), n_bins=n_bins, map_half_width=map_half_width
        )
        t = soft_am_radial_from_dens_maps(
            target_dens, m=int(m), n_bins=n_bins, map_half_width=map_half_width
        )
        # Per-bin weights: mass in ring × where the target actually has structure.
        w = torch.ones_like(t["amp"])
        if weight_by_a0:
            a0 = t["a0"]
            w = w * (a0 / a0.mean(dim=-1, keepdim=True).clamp_min(1e-8))
        if weight_by_target_amp:
            w = w * (1.0 + 4.0 * t["amp"])
        rf_w = radial_focus_bin_weights(
            t["r_mid"],
            r_focus,
            band_lo_frac=float(r_focus_band[0]),
            band_hi_frac=float(r_focus_band[1]),
            peak=float(r_focus_peak),
            floor=float(r_focus_floor),
        )
        if rf_w is not None:
            if rf_w.shape[0] == 1 and w.shape[0] > 1:
                rf_w = rf_w.expand(w.shape[0], -1)
            w = w * rf_w
        w = w / w.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        loss_phase = (
            ((p["cos"] - t["cos"]) ** 2 * w).mean()
            + ((p["sin"] - t["sin"]) ** 2 * w).mean()
        )
        # Absolute + relative amp so tall peaks are not washed by flat outer bins.
        d_amp = p["amp"] - t["amp"]
        loss_amp = (d_amp**2 * w).mean()
        loss_amp_rel = ((d_amp / (t["amp"] + 0.08)) ** 2 * w).mean()
        per = float(lambda_phase) * loss_phase + float(amp_weight) * (
            0.5 * loss_amp + 0.5 * loss_amp_rel
        )
        # Soft quiet gate from mean target amp (map-wide morphology strength).
        strength = t["amp"].mean(dim=-1)  # (B,)
        gate = torch.sigmoid(
            (strength - float(quiet_gate_floor)) / max(float(quiet_gate_temp), 1e-6)
        )
        losses.append(per * gate.mean())
    return torch.stack(losses).mean() if losses else pred_dens.new_zeros(())

def linearize_log1p_dens(
    dens_log: torch.Tensor,
    dens_scale: float,
    *,
    max_over_scale: float = 80.0,
) -> torch.Tensor:
    """Invert ``log1p(dens / dens_scale)`` used by :func:`normalize_stack`.

    Clamps the reconstructed dens to ``max_over_scale * dens_scale`` so a bad
    early prediction cannot explode Fourier / residual losses.
    """
    phys = torch.expm1(dens_log.clamp(min=-0.999, max=20.0)) * float(dens_scale)
    return phys.clamp(min=0.0, max=float(dens_scale) * float(max_over_scale))


def axisym_residual_dens(
    dens: torch.Tensor,
    *,
    n_bins: int = 16,
    map_half_width: float = 1.0,
) -> torch.Tensor:
    """
    Axisym-subtracted midplane dens residual ``R = Σ_n − ⟨Σ_n⟩_φ``.

    Dens is mean-normalized first so residuals are scale-free across FOVs /
    components.  Quiet maps → tiny ``R``; barred / spiral maps retain the
    non-axisymmetric morphology that scalar dens MSE under-weights.
    """
    _b, h, w = dens.shape
    half = float(map_half_width)
    yy, xx = torch.meshgrid(
        torch.linspace(-half, half, h, device=dens.device, dtype=dens.dtype),
        torch.linspace(-half, half, w, device=dens.device, dtype=dens.dtype),
        indexing="ij",
    )
    r = torch.sqrt(xx * xx + yy * yy + 1e-16)
    edges = torch.linspace(0.0, half, int(n_bins) + 1, device=dens.device, dtype=dens.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = float(half) / max(int(n_bins), 1)
    sigma = 0.5 * bin_w
    logits = -0.5 * ((r.unsqueeze(-1) - centers.view(1, 1, -1)) / max(sigma, 1e-6)) ** 2
    inside = (r <= half * 1.05).to(dens.dtype)
    attn = torch.softmax(logits, dim=-1) * inside.unsqueeze(-1)  # (H,W,nbins)
    dens_n = dens / dens.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    wgt = dens_n.clamp_min(0.0).unsqueeze(-1)  # (B,H,W,1)
    # Soft radial *mean* (not sum) so constant maps have ~0 residual.
    bin_mass = attn.sum(dim=(-3, -2)).clamp_min(1e-8)  # (nbins,)
    a0_mean = (wgt * attn).sum(dim=(-3, -2)) / bin_mass  # (B, nbins)
    axisym = (a0_mean.view(-1, 1, 1, int(n_bins)) * attn.unsqueeze(0)).sum(dim=-1)
    # Zero residual outside the circular FOV (corners have no azimuthal mean).
    return (dens_n - axisym) * inside


def spatial_ring_weight_map(
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    map_half_width: float = 1.0,
    r_focus: torch.Tensor | float | None = None,
    band_lo_frac: float = 0.5,
    band_hi_frac: float = 1.5,
    peak: float = 4.0,
    floor: float = 0.25,
) -> torch.Tensor | None:
    """``(B,H,W)`` or ``(1,H,W)`` soft ring weight around ``r_focus``, or ``None``."""
    if r_focus is None:
        return None
    half = float(map_half_width)
    yy, xx = torch.meshgrid(
        torch.linspace(-half, half, h, device=device, dtype=dtype),
        torch.linspace(-half, half, w, device=device, dtype=dtype),
        indexing="ij",
    )
    r = torch.sqrt(xx * xx + yy * yy + 1e-16)  # (H,W)
    if not torch.is_tensor(r_focus):
        rf = r.new_tensor(float(r_focus)).view(1)
    else:
        rf = r_focus.to(device=device, dtype=dtype).reshape(-1)
    rf = rf.clamp_min(1e-4).view(-1, 1, 1)
    r_b = r.unsqueeze(0)
    lo = float(band_lo_frac) * rf
    hi = float(band_hi_frac) * rf
    in_band = ((r_b >= lo) & (r_b <= hi)).to(dtype)
    sigma = (0.35 * rf).clamp_min(1e-3)
    gauss = torch.exp(-0.5 * ((r_b - rf) / sigma) ** 2)
    return float(floor) + (1.0 - float(floor)) * in_band + (float(peak) - 1.0) * gauss


def axisym_residual_dens_loss(
    pred_dens: torch.Tensor,
    target_dens: torch.Tensor,
    *,
    n_bins: int = 16,
    map_half_width: float = 1.0,
    r_focus: torch.Tensor | float | None = None,
    r_focus_band: tuple[float, float] = (0.5, 1.5),
    r_focus_peak: float = 4.0,
    r_focus_floor: float = 0.25,
) -> torch.Tensor:
    """
    MSE on dens after subtracting the soft radial (axisymmetric) mean.

    Quiet maps have tiny residuals → little pressure to invent bars.  Barred maps
    force the non-axisymmetric residual (the actual morphological features) to
    match, which is what scalar dens MSE under-weights.

    Optional ``r_focus`` up-weights the ring near ``R_d`` (and ``0.5–1.5 R_d``).
    """
    pr = axisym_residual_dens(
        pred_dens, n_bins=n_bins, map_half_width=map_half_width
    )
    tr = axisym_residual_dens(
        target_dens, n_bins=n_bins, map_half_width=map_half_width
    )
    diff2 = (pr - tr) ** 2
    sw = spatial_ring_weight_map(
        pr.shape[-2],
        pr.shape[-1],
        device=pr.device,
        dtype=pr.dtype,
        map_half_width=map_half_width,
        r_focus=r_focus,
        band_lo_frac=float(r_focus_band[0]),
        band_hi_frac=float(r_focus_band[1]),
        peak=float(r_focus_peak),
        floor=float(r_focus_floor),
    )
    if sw is None:
        return diff2.mean()
    if sw.shape[0] == 1 and diff2.shape[0] > 1:
        sw = sw.expand(diff2.shape[0], -1, -1)
    sw = sw / sw.mean().clamp_min(1e-8)
    return (diff2 * sw).mean()


def _fft_k_band_weights(
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    k_floor: float = 0.08,
    k_boost_power: float = 1.0,
) -> torch.Tensor:
    """Soft high-pass + mid/high-``k`` boost for ``rfft2`` magnitude grids."""
    fy = torch.fft.fftfreq(int(h), d=1.0, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(int(w), d=1.0, device=device, dtype=dtype)
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    k = torch.sqrt(kx * kx + ky * ky + 1e-16)
    k_nyq = 0.5
    # Damp k≈0 (axisym / global mean already removed); boost mid/high k.
    highpass = 1.0 - torch.exp(-(k / max(float(k_floor), 1e-6)) ** 2)
    boost = (k / k_nyq).clamp(max=1.0) ** float(k_boost_power)
    w_k = highpass * boost
    return w_k / w_k.mean().clamp_min(1e-8)


def spatial_fft_morphology_loss(
    pred_dens: torch.Tensor,
    target_dens: torch.Tensor,
    *,
    n_bins: int = 16,
    map_half_width: float = 1.0,
    lambda_phase: float = 0.15,
    k_floor: float = 0.08,
    k_boost_power: float = 1.0,
    quiet_gate_floor: float = 0.02,
    quiet_gate_temp: float = 0.01,
    r_focus: torch.Tensor | float | None = None,
    r_focus_band: tuple[float, float] = (0.5, 1.5),
    r_focus_peak: float = 4.0,
    r_focus_floor: float = 0.25,
) -> torch.Tensor:
    """
    Match spatial FFT of axisym-residual dens (full morphology, not just ``A_m``).

    Primary term: relative / log1p magnitude MSE on ``|FFT(R)|`` with k-band
    weights (damp ``k≈0``, boost mid/high ``k``) so outer spiral / bar arms are
    not washed by large-scale power.  Optional light phase cosine term.
    Soft-gated by target residual RMS so quiet ICs are not forced to invent
    structure (same spirit as :func:`soft_fourier_match_loss`).

    Optional ``r_focus`` spatially windows the residual toward the ``R_d`` ring
    before the FFT so morphology match concentrates near the disk scale length.
    """
    r_pred = axisym_residual_dens(
        pred_dens, n_bins=n_bins, map_half_width=map_half_width
    )
    r_tgt = axisym_residual_dens(
        target_dens, n_bins=n_bins, map_half_width=map_half_width
    )
    # Clamp residual amplitude so a bad early pred cannot explode the FFT.
    r_pred = r_pred.clamp(-20.0, 20.0)
    r_tgt = r_tgt.clamp(-20.0, 20.0)
    sw = spatial_ring_weight_map(
        r_pred.shape[-2],
        r_pred.shape[-1],
        device=r_pred.device,
        dtype=r_pred.dtype,
        map_half_width=map_half_width,
        r_focus=r_focus,
        band_lo_frac=float(r_focus_band[0]),
        band_hi_frac=float(r_focus_band[1]),
        peak=float(r_focus_peak),
        floor=float(r_focus_floor),
    )
    if sw is not None:
        if sw.shape[0] == 1 and r_pred.shape[0] > 1:
            sw = sw.expand(r_pred.shape[0], -1, -1)
        # Keep mean scale so quiet-gate RMS stays comparable.
        sw = sw / sw.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        r_pred = r_pred * sw
        r_tgt = r_tgt * sw


    fp = torch.fft.rfft2(r_pred, norm="ortho")
    ft = torch.fft.rfft2(r_tgt, norm="ortho")
    _b, h, w = r_pred.shape
    w_k = _fft_k_band_weights(
        h,
        w,
        device=r_pred.device,
        dtype=r_pred.dtype,
        k_floor=k_floor,
        k_boost_power=k_boost_power,
    )

    # log1p(|F|) keeps spectra numerically tame; relative denom avoids
    # over-weighting bright DC-adjacent bins that survive the high-pass.
    mag_p = torch.log1p(fp.abs())
    mag_t = torch.log1p(ft.abs())
    loss_mag = (((mag_p - mag_t) / (mag_t + 0.15)) ** 2 * w_k).mean()

    loss_phase = r_pred.new_zeros(())
    lp = float(lambda_phase)
    if lp > 0.0:
        denom = (fp.abs() * ft.abs()).clamp_min(1e-8)
        cos_dphi = (fp.real * ft.real + fp.imag * ft.imag) / denom
        # Weight phase by target power so quiet bins don't dominate.
        phase_w = w_k * (ft.abs() / ft.abs().mean().clamp_min(1e-8))
        loss_phase = ((1.0 - cos_dphi).clamp(min=0.0) * phase_w).mean()

    # Quiet gate from residual RMS (morphology strength on the target).
    strength = r_tgt.pow(2).mean(dim=(-2, -1)).sqrt()  # (B,)
    gate = torch.sigmoid(
        (strength - float(quiet_gate_floor)) / max(float(quiet_gate_temp), 1e-6)
    )
    return (loss_mag + lp * loss_phase) * gate.mean()


def dens_map_azimuthal_fourier_numpy(
    dens,
    *,
    m: int = 2,
    n_bins: int = 12,
    r_max: float = 12.0,
) -> dict:
    """
    Hard-binned ``A_m(R)`` diagnostic on a face-on dens map (numpy).

    Coordinates span ``[-r_max, r_max]²``.  Returns ``r_mid``, ``a_m_over_a0``,
    ``cos``, ``sin``, and median amplitude (for quick summaries only — prefer
    the full curve).
    """
    import numpy as np

    dens = np.asarray(dens, dtype=np.float64)
    h, w = dens.shape
    yy, xx = np.meshgrid(
        np.linspace(-float(r_max), float(r_max), h),
        np.linspace(-float(r_max), float(r_max), w),
        indexing="ij",
    )
    r = np.hypot(xx, yy)
    phi = np.arctan2(yy, xx)
    edges = np.linspace(0.0, float(r_max), int(n_bins) + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    amp = np.full(int(n_bins), np.nan)
    cos_n = np.full(int(n_bins), np.nan)
    sin_n = np.full(int(n_bins), np.nan)
    wgt = np.maximum(dens, 0.0)
    m_f = float(m)
    for i in range(int(n_bins)):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        a0 = float(wgt[mask].sum())
        if a0 <= 0:
            continue
        ac = float((wgt[mask] * np.cos(m_f * phi[mask])).sum())
        as_ = float((wgt[mask] * np.sin(m_f * phi[mask])).sum())
        cos_n[i] = ac / a0
        sin_n[i] = as_ / a0
        amp[i] = float(np.hypot(cos_n[i], sin_n[i]))
    valid = np.isfinite(amp)
    median = float(np.median(amp[valid])) if valid.any() else float("nan")
    return {
        "m": int(m),
        "r_mid": r_mid,
        "a_m_over_a0": amp,
        "cos": cos_n,
        "sin": sin_n,
        "a_m_over_a0_median": median,
    }


def dens_spatial_grad_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dens_channel_indices: list[int],
) -> torch.Tensor:
    """Match finite-difference ∇ dens so bars/spirals stay sharp (not just mean dens)."""
    if not dens_channel_indices:
        return pred.new_zeros(())
    p = pred[:, dens_channel_indices]
    t = target[:, dens_channel_indices]
    # Central differences along H, W.
    dp_y = p[:, :, 1:, :] - p[:, :, :-1, :]
    dt_y = t[:, :, 1:, :] - t[:, :, :-1, :]
    dp_x = p[:, :, :, 1:] - p[:, :, :, :-1]
    dt_x = t[:, :, :, 1:] - t[:, :, :, :-1]
    return ((dp_y - dt_y) ** 2).mean() + ((dp_x - dt_x) ** 2).mean()


def dens_weighted_moment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dens_channel_indices: list[int],
    dens_scale: float | None,
    n_mom: int | None = None,
) -> torch.Tensor:
    """
    ``‖√Σ₊ (m̂ − m)‖²`` on non-dens channels (normalized space).

    ``Σ`` is linearized dens per slab (``expm1`` of log1p channels when
    ``dens_scale`` is set). Weighting puts kinematics error where the mass
    lives — the joint dens+moments DF objective (option A).
    """
    if not dens_channel_indices:
        return pred.new_zeros(())
    dens_set = set(int(i) for i in dens_channel_indices)
    mom_idx = [i for i in range(pred.shape[1]) if i not in dens_set]
    if not mom_idx:
        return pred.new_zeros(())
    if n_mom is None:
        if len(dens_channel_indices) >= 2:
            n_mom = int(dens_channel_indices[1] - dens_channel_indices[0])
        else:
            n_mom = max(int(pred.shape[1] // max(len(dens_channel_indices), 1)), 1)
    n_mom = max(int(n_mom), 1)

    dens_ch = target[:, dens_channel_indices]
    if dens_scale is None:
        sigma = dens_ch.clamp_min(0.0)
    else:
        sigma = linearize_log1p_dens(dens_ch, float(dens_scale))
    # Map each dens channel index → its slab's √Σ weight broadcast to moments.
    # dens channels are interleaved as dens, vx, … at offsets 0, n_mom, …
    w_maps: dict[int, torch.Tensor] = {}
    for j, di in enumerate(dens_channel_indices):
        w_maps[int(di)] = torch.sqrt(sigma[:, j].clamp_min(0.0) + 1e-12)

    err2 = (pred - target) ** 2
    acc = pred.new_zeros(())
    n_terms = 0
    for mi in mom_idx:
        # dens channel for this slab: floor(mi / n_mom) * n_mom
        dens_i = (int(mi) // n_mom) * n_mom
        w = w_maps.get(dens_i)
        if w is None:
            # Fallback: nearest dens channel ≤ mi
            cand = [d for d in dens_channel_indices if d <= mi]
            dens_i = int(cand[-1]) if cand else int(dens_channel_indices[0])
            w = w_maps[dens_i]
        acc = acc + (err2[:, mi] * w).mean()
        n_terms += 1
    return acc / max(n_terms, 1)


def dens_weighted_vphi_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dens_channel_indices: list[int],
    dens_scale: float | None,
    n_mom: int | None = None,
    vx_offset: int = 1,
    vy_offset: int = 2,
) -> torch.Tensor:
    """
    Midplane-style dens-weighted ``⟨v_φ⟩`` map MSE from Cartesian ``vx,vy``.

    For each z-slab dens channel, build ``v_φ = (−y vx + x vy)/R`` on the pixel
    grid and penalise ``‖√Σ (v̂_φ − v_φ)‖²``. Attacks the OOD ``⟨v_φ⟩(R)`` gap
    without a cylindrical deposit channel.
    """
    if not dens_channel_indices:
        return pred.new_zeros(())
    if n_mom is None:
        if len(dens_channel_indices) >= 2:
            n_mom = int(dens_channel_indices[1] - dens_channel_indices[0])
        else:
            n_mom = max(int(pred.shape[1] // max(len(dens_channel_indices), 1)), 1)
    n_mom = max(int(n_mom), 1)
    _b, _c, h, w = pred.shape
    device, dtype = pred.device, pred.dtype
    # Pixel centres in a unit square FOV (scale cancels in v_φ / R).
    # Match ``histogram2d`` deposit layout: channel[i, j] ↔ (x_i, y_j).
    # (Older code used ``yy, xx = meshgrid(..., ij)`` which swapped axes and
    # made ⟨v_φ⟩≈0 for a rotating disk — useless as a DF kinematics loss.)
    xx, yy = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    r = torch.sqrt(xx * xx + yy * yy + 1e-16)

    dens_ch = target[:, dens_channel_indices]
    if dens_scale is None:
        sigma = dens_ch.clamp_min(0.0)
    else:
        sigma = linearize_log1p_dens(dens_ch, float(dens_scale))

    acc = pred.new_zeros(())
    n_terms = 0
    for j, di in enumerate(dens_channel_indices):
        base = int(di)
        ivx = base + int(vx_offset)
        ivy = base + int(vy_offset)
        if ivx >= pred.shape[1] or ivy >= pred.shape[1]:
            continue
        w = torch.sqrt(sigma[:, j].clamp_min(0.0) + 1e-12)

        def _vphi(t: torch.Tensor) -> torch.Tensor:
            return (-yy * t[:, ivx] + xx * t[:, ivy]) / r

        diff = _vphi(pred) - _vphi(target)
        acc = acc + (diff * diff * w).mean()
        n_terms += 1
    if n_terms == 0:
        return pred.new_zeros(())
    return acc / n_terms


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    dens_channel_indices: list[int],
    dens_weight: float = 5.0,
    moment_weight: float | None = None,
    dens_grad_weight: float = 0.0,
    dens_resid_weight: float = 0.0,
    a2_weight: float = 0.0,
    fourier_weight: float | None = None,
    fourier_modes: tuple[int, ...] = (1, 2, 3),
    fourier_n_bins: int = 12,
    fourier_lambda_phase: float = 1.0,
    fourier_amp_weight: float = 2.0,
    fourier_quiet_gate_floor: float = 0.05,
    fft_weight: float = 0.0,
    fft_lambda_phase: float = 0.15,
    fft_quiet_gate_floor: float | None = None,
    fft_k_floor: float = 0.08,
    dens_scale: float | None = None,
    dens_mass_weight: float = 0.0,
    moment_phys_weight: float = 0.0,
    vphi_phys_weight: float = 0.0,
    n_mom: int | None = None,
    r_focus: torch.Tensor | float | None = None,
    r_focus_band: tuple[float, float] = (0.5, 1.5),
    r_focus_peak: float = 4.0,
    r_focus_floor: float = 0.25,
    a2_rd_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Channel-weighted MSE; optional radial Fourier / axisym-residual / FFT match.

    ``moment_weight`` (default = ``dens_weight``) applies to all non-density
    channels (⟨v⟩, σ, …) so phase-space moments are not under-weighted relative
    to dens.  When ``dens_scale`` is set, Fourier / FFT morphology terms run on
    linearized dens (``expm1`` of log1p-normalized channels) so ``A_m(R)`` and
    residual spectra match physical map amplitudes — not log-space contrast.
    Keep ``dens_grad_weight=0`` (hurt bars in crisp_v3).

    ``dens_mass_weight`` > 0 reweights dens MSE by
    ``1 + α · target_dens / mean(target_dens)`` so cusp / high-Σ voxels are not
    drowned by the many near-empty outer cells (critical for the bulge tower).

    ``moment_phys_weight`` / ``vphi_phys_weight``: dens-weighted (√Σ) moment /
    ``⟨v_φ⟩`` field losses on linearized dens (DF-matching option A).

    ``r_focus`` / ``a2_rd_weight``: focus quiet-gated Am / dens-residual / FFT
    morphology on the ring near disk scale length ``R_d``, plus an optional
    dedicated soft ``A₂(R_d)`` match (train-for-persistence at the bar scale).
    """
    err = (pred - target) ** 2
    dens_mask = torch.zeros(err.shape[1], device=err.device, dtype=err.dtype)
    for i in dens_channel_indices:
        if 0 <= i < err.shape[1]:
            dens_mask[i] = 1.0
    mom_mask = 1.0 - dens_mask
    dw = float(dens_weight)
    mw = float(dens_weight if moment_weight is None else moment_weight)
    w = dw * dens_mask + mw * mom_mask
    weighted = err * w.view(1, -1, 1, 1)
    dm_w = float(dens_mass_weight)
    if dm_w > 0.0 and dens_channel_indices:
        td = target[:, dens_channel_indices].detach().clamp_min(0.0)
        tmean = td.mean().clamp_min(1e-4)
        spat_w = 1.0 + dm_w * (td / tmean)
        dens_slice = err[:, dens_channel_indices] * spat_w
        # Replace dens contribution in the channel-weighted mean with cusp-aware dens.
        dens_part = dens_slice.mean() * dw
        mom_idx = [i for i in range(err.shape[1]) if dens_mask[i] < 0.5]
        mom_part = (
            (err[:, mom_idx].mean() * mw) if mom_idx else err.new_zeros(())
        )
        # Preserve relative dens/moment channel count weighting of the flat mean.
        n_d = max(len(dens_channel_indices), 1)
        n_m = max(err.shape[1] - n_d, 1)
        n_tot = float(err.shape[1])
        loss = (n_d * dens_part + n_m * mom_part) / n_tot
        dens_loss = dens_slice.mean()
    else:
        loss = weighted.mean()
        dens_loss = err[:, dens_channel_indices].mean() if dens_channel_indices else loss
    mom_idx = [i for i in range(err.shape[1]) if dens_mask[i] < 0.5]
    mom_loss = err[:, mom_idx].mean() if mom_idx else loss.new_zeros(())
    metrics = {
        "loss": loss,
        "mse": err.mean(),
        "mse_dens": dens_loss,
        "mse_mom": mom_loss,
    }
    gw = float(dens_grad_weight)
    if gw > 0.0 and dens_channel_indices:
        g_l = dens_spatial_grad_loss(pred, target, dens_channel_indices)
        metrics["loss"] = metrics["loss"] + gw * g_l
        metrics["dens_grad_mse"] = g_l

    def _log_dens(x: torch.Tensor) -> torch.Tensor:
        return x[:, dens_channel_indices].sum(dim=1).clamp_min(0.0)

    def _phys_dens(x: torch.Tensor) -> torch.Tensor:
        ch = x[:, dens_channel_indices]
        if dens_scale is None:
            return ch.sum(dim=1).clamp_min(0.0)
        # Invert log1p per channel, then collapse slices/components.
        return linearize_log1p_dens(ch, float(dens_scale)).sum(dim=1)

    rw = float(dens_resid_weight)
    if rw > 0.0 and dens_channel_indices:
        # Residual on log1p dens (training space): scale-free, no expm1 blow-ups.
        r_l = axisym_residual_dens_loss(
            _log_dens(pred),
            _log_dens(target),
            r_focus=r_focus,
            r_focus_band=r_focus_band,
            r_focus_peak=r_focus_peak,
            r_focus_floor=r_focus_floor,
        )
        metrics["loss"] = metrics["loss"] + rw * r_l
        metrics["dens_resid_mse"] = r_l

    mpw = float(moment_phys_weight)
    if mpw > 0.0 and dens_channel_indices:
        mp_l = dens_weighted_moment_loss(
            pred,
            target,
            dens_channel_indices=dens_channel_indices,
            dens_scale=dens_scale,
            n_mom=n_mom,
        )
        metrics["loss"] = metrics["loss"] + mpw * mp_l
        metrics["moment_phys_mse"] = mp_l

    vpw = float(vphi_phys_weight)
    if vpw > 0.0 and dens_channel_indices:
        vp_l = dens_weighted_vphi_field_loss(
            pred,
            target,
            dens_channel_indices=dens_channel_indices,
            dens_scale=dens_scale,
            n_mom=n_mom,
        )
        metrics["loss"] = metrics["loss"] + vpw * vp_l
        metrics["vphi_phys_mse"] = vp_l

    fw = float(a2_weight if fourier_weight is None else fourier_weight)
    fft_w = float(fft_weight)
    a2rd_w = float(a2_rd_weight)
    need_phys = dens_channel_indices and (fw > 0.0 or fft_w > 0.0 or a2rd_w > 0.0)
    pred_d = tgt_d = None
    if need_phys:
        pred_d = _phys_dens(pred)
        tgt_d = _phys_dens(target)

    if fw > 0.0 and dens_channel_indices:
        assert pred_d is not None and tgt_d is not None
        modes = tuple(int(m) for m in fourier_modes) or (2,)
        f_l = soft_fourier_match_loss(
            pred_d,
            tgt_d,
            modes=modes,
            n_bins=fourier_n_bins,
            lambda_phase=fourier_lambda_phase,
            amp_weight=fourier_amp_weight,
            quiet_gate_floor=fourier_quiet_gate_floor,
            r_focus=r_focus,
            r_focus_band=r_focus_band,
            r_focus_peak=r_focus_peak,
            r_focus_floor=r_focus_floor,
        )
        # a2_mse = radial m=2 amplitude profile MSE (not a single scalar).
        p2 = soft_am_radial_from_dens_maps(pred_d, m=2, n_bins=fourier_n_bins)
        t2 = soft_am_radial_from_dens_maps(tgt_d, m=2, n_bins=fourier_n_bins)
        a2_l = ((p2["amp"] - t2["amp"]) ** 2).mean()
        metrics["loss"] = metrics["loss"] + fw * f_l
        metrics["a2_mse"] = a2_l
        metrics["fourier_mse"] = f_l

    if a2rd_w > 0.0 and dens_channel_indices and r_focus is not None:
        assert pred_d is not None and tgt_d is not None
        a2rd_l = soft_a2_at_r_match_loss(
            pred_d,
            tgt_d,
            r_focus=r_focus,
            n_bins=fourier_n_bins,
            quiet_gate_floor=fourier_quiet_gate_floor,
        )
        metrics["loss"] = metrics["loss"] + a2rd_w * a2rd_l
        metrics["a2_rd_mse"] = a2rd_l

    if fft_w > 0.0 and dens_channel_indices:
        assert pred_d is not None and tgt_d is not None
        fft_floor = (
            float(fourier_quiet_gate_floor)
            if fft_quiet_gate_floor is None
            else float(fft_quiet_gate_floor)
        )
        fft_l = spatial_fft_morphology_loss(
            pred_d,
            tgt_d,
            lambda_phase=fft_lambda_phase,
            k_floor=fft_k_floor,
            quiet_gate_floor=fft_floor,
            r_focus=r_focus,
            r_focus_band=r_focus_band,
            r_focus_peak=r_focus_peak,
            r_focus_floor=r_focus_floor,
        )
        metrics["loss"] = metrics["loss"] + fft_w * fft_l
        metrics["fft_mse"] = fft_l
    return metrics


def multitower_reconstruction_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    dens_indices: dict[str, list[int]],
    dens_weight: float = 5.0,
    moment_weight: float | None = None,
    dens_grad_weight: float = 0.0,
    dens_resid_weight: float = 0.0,
    a2_weight: float = 0.0,
    fourier_weight: float | None = None,
    fourier_modes: tuple[int, ...] = (1, 2, 3),
    fourier_n_bins: int = 12,
    fourier_lambda_phase: float = 1.0,
    fourier_amp_weight: float = 2.0,
    fourier_quiet_gate_floor: float = 0.05,
    fft_weight: float = 0.0,
    fft_lambda_phase: float = 0.15,
    fft_quiet_gate_floor: float | None = None,
    fft_k_floor: float = 0.08,
    dens_scales: dict[str, float] | None = None,
    component_weights: dict[str, float] | None = None,
    fourier_component_scale: dict[str, float] | None = None,
    fft_component_scale: dict[str, float] | None = None,
    dens_mass_weight: float = 0.0,
    dens_mass_weight_by_component: dict[str, float] | None = None,
    moment_phys_weight: float = 0.0,
    vphi_phys_weight: float = 0.0,
    n_mom_by_component: dict[str, int] | None = None,
    r_focus_by_component: dict[str, torch.Tensor | float | None] | None = None,
    r_focus_band: tuple[float, float] = (0.5, 1.5),
    r_focus_peak: float = 4.0,
    r_focus_floor: float = 0.25,
    a2_rd_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Sum of per-tower :func:`reconstruction_loss` (optional component weights)."""
    total = None
    dens_total = None
    mom_total = None
    dens_grad_total = None
    dens_resid_total = None
    a2_total = None
    fourier_total = None
    fft_total = None
    a2_rd_total = None
    moment_phys_total = None
    vphi_phys_total = None
    cw = component_weights or {}
    dscales = dens_scales or {}
    dmw_by = dens_mass_weight_by_component or {}
    nmom_by = n_mom_by_component or {}
    rfocus_by = r_focus_by_component or {}
    # Disk carries full Fourier / FFT weight; bulge/halo get a lighter share so
    # non-axisymmetric residual is not washed out to axisym.
    fscale = fourier_component_scale or {"disk": 1.0, "bulge": 0.15, "halo": 0.05}
    fftscale = fft_component_scale or {"disk": 1.0, "bulge": 0.1, "halo": 0.05}
    rscale = {"disk": 1.0, "bulge": 0.25, "halo": 0.1}
    # Joint DF terms: disk-heavy (where ⟨v_φ⟩ matters); light on bulge/halo.
    mphys_scale = {"disk": 1.0, "bulge": 0.25, "halo": 0.1}
    vphi_scale = {"disk": 1.0, "bulge": 0.0, "halo": 0.0}
    # A₂(R_d) is a disk-scale metric — disk-only.
    a2rd_scale = {"disk": 1.0, "bulge": 0.0, "halo": 0.0}
    base_fw = float(a2_weight if fourier_weight is None else fourier_weight)
    base_fft = float(fft_weight)
    base_rw = float(dens_resid_weight)
    base_mpw = float(moment_phys_weight)
    base_vpw = float(vphi_phys_weight)
    base_a2rd = float(a2_rd_weight)
    for name, y in pred.items():
        fw = base_fw * float(fscale.get(name, 0.0))
        ffw = base_fft * float(fftscale.get(name, 0.0))
        rw = base_rw * float(rscale.get(name, 0.0))
        mpw = base_mpw * float(mphys_scale.get(name, 0.0))
        vpw = base_vpw * float(vphi_scale.get(name, 0.0))
        a2rdw = base_a2rd * float(a2rd_scale.get(name, 0.0))
        dmw = float(dmw_by.get(name, dens_mass_weight))
        m = reconstruction_loss(
            y,
            target[name],
            dens_channel_indices=dens_indices[name],
            dens_weight=dens_weight,
            moment_weight=moment_weight,
            dens_grad_weight=dens_grad_weight,
            dens_resid_weight=rw,
            fourier_weight=fw,
            fourier_modes=fourier_modes,
            fourier_n_bins=fourier_n_bins,
            fourier_lambda_phase=fourier_lambda_phase,
            fourier_amp_weight=fourier_amp_weight,
            fourier_quiet_gate_floor=fourier_quiet_gate_floor,
            fft_weight=ffw,
            fft_lambda_phase=fft_lambda_phase,
            fft_quiet_gate_floor=fft_quiet_gate_floor,
            fft_k_floor=fft_k_floor,
            dens_scale=dscales.get(name),
            dens_mass_weight=dmw,
            moment_phys_weight=mpw,
            vphi_phys_weight=vpw,
            n_mom=nmom_by.get(name),
            r_focus=rfocus_by.get(name),
            r_focus_band=r_focus_band,
            r_focus_peak=r_focus_peak,
            r_focus_floor=r_focus_floor,
            a2_rd_weight=a2rdw,
        )
        w = float(cw.get(name, 1.0))
        piece = w * m["loss"]
        total = piece if total is None else total + piece
        dens_total = m["mse_dens"] if dens_total is None else dens_total + m["mse_dens"]
        mom_total = m["mse_mom"] if mom_total is None else mom_total + m["mse_mom"]
        if "dens_grad_mse" in m:
            dens_grad_total = (
                m["dens_grad_mse"]
                if dens_grad_total is None
                else dens_grad_total + m["dens_grad_mse"]
            )
        if "dens_resid_mse" in m:
            dens_resid_total = (
                m["dens_resid_mse"]
                if dens_resid_total is None
                else dens_resid_total + m["dens_resid_mse"]
            )
        if "a2_mse" in m:
            a2_total = m["a2_mse"] if a2_total is None else a2_total + m["a2_mse"]
        if "fourier_mse" in m:
            fourier_total = (
                m["fourier_mse"] if fourier_total is None else fourier_total + m["fourier_mse"]
            )
        if "fft_mse" in m:
            fft_total = m["fft_mse"] if fft_total is None else fft_total + m["fft_mse"]
        if "a2_rd_mse" in m:
            a2_rd_total = (
                m["a2_rd_mse"] if a2_rd_total is None else a2_rd_total + m["a2_rd_mse"]
            )
        if "moment_phys_mse" in m:
            moment_phys_total = (
                m["moment_phys_mse"]
                if moment_phys_total is None
                else moment_phys_total + m["moment_phys_mse"]
            )
        if "vphi_phys_mse" in m:
            vphi_phys_total = (
                m["vphi_phys_mse"]
                if vphi_phys_total is None
                else vphi_phys_total + m["vphi_phys_mse"]
            )
    assert total is not None and dens_total is not None
    out = {"loss": total, "mse_dens": dens_total, "mse_mom": mom_total}
    if dens_grad_total is not None:
        out["dens_grad_mse"] = dens_grad_total
    if dens_resid_total is not None:
        out["dens_resid_mse"] = dens_resid_total
    if a2_total is not None:
        out["a2_mse"] = a2_total
    if fourier_total is not None:
        out["fourier_mse"] = fourier_total
    if fft_total is not None:
        out["fft_mse"] = fft_total
    if a2_rd_total is not None:
        out["a2_rd_mse"] = a2_rd_total
    if moment_phys_total is not None:
        out["moment_phys_mse"] = moment_phys_total
    if vphi_phys_total is not None:
        out["vphi_phys_mse"] = vphi_phys_total
    return out
