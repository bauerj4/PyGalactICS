"""Conditional multi-tower field VAE: encode slices → global ``z``, decode ``(z, θ)``.

Scientific role (vs pure recon AE in :mod:`galacticsics.ml.fields.autoencoder`):

* Encode a snapshot (or condition on structural ``θ``) → **one** latent ``z``.
* Sample / interpolate ``z ~ N(0, I) | θ`` to generate non-equilibrium field stacks
  (bars, spirals, …), then resample particles.
* Preserve dens + velocity-moment recon quality with U-Net-style towers, FiLM
  conditioning, and a weak KL.

Particle-set :class:`~galacticsics.ml.models.sequence_vae.SequenceVAE` remains a
phase-space baseline; this module is the recommended morphology path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from galacticsics.ml.fields.autoencoder import (
    SliceAutoencoderConfig,
    _ConvBlock,
    multitower_reconstruction_loss,
)
from galacticsics.ml.fields.binning import MultiScaleSliceConfig


@dataclass
class FieldVAEConfig:
    """Knobs for :class:`MultiTowerSliceVAE`."""

    latent_dim: int = 64
    theta_dim: int = 10
    base_channels: int = 32
    bottleneck_channels: int = 64
    enc_grid: int = 4
    """Per-tower bottleneck is resized to ``enc_grid²`` before flattening into ``z``."""
    separate_heads: bool = True
    beta: float = 1e-4
    free_bits: float = 0.1
    skip_dropout: float = 1.0
    """Fraction of batches that drop encoder skips (1.0 = always prior path)."""
    prior_decode_weight: float = 0.0
    """Extra prior-path term (unused when skip_dropout=1; kept for ablations)."""
    skip_recon_weight: float = 0.1
    """Optional weak skip-assisted recon (sharpening only; not used at sample time)."""


class FiLM(nn.Module):
    """Feature-wise linear modulation: ``x * (1 + γ) + β`` from a condition vector."""

    def __init__(self, cond_dim: int, n_channels: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, n_channels)
        self.to_beta = nn.Linear(cond_dim, n_channels)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.to_gamma(cond).view(x.shape[0], x.shape[1], 1, 1)
        b = self.to_beta(cond).view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + g) + b


class SliceUNetEncoder(nn.Module):
    """U-Net encoder half: returns skip tensors + spatial bottleneck."""

    def __init__(self, cfg: SliceAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        c = int(cfg.base_channels)
        z = int(cfg.latent_channels)
        self.enc1 = _ConvBlock(c_in, c)
        self.down1 = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.enc2 = _ConvBlock(c, 2 * c)
        self.down2 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)
        self.enc3 = _ConvBlock(2 * c, 4 * c)
        self.down3 = nn.Conv2d(4 * c, 4 * c, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            _ConvBlock(4 * c, z),
            _ConvBlock(z, z),
        )
        self.out_channels = z
        self.pool_dim = z

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.gelu(self.down1(e1)))
        e3 = self.enc3(F.gelu(self.down2(e2)))
        b = self.bottleneck(F.gelu(self.down3(e3)))
        return b, (e1, e2, e3)


class SliceUNetDecoder(nn.Module):
    """U-Net decoder half with FiLM at each upsample level + dens/moment heads."""

    def __init__(self, cfg: SliceAutoencoderConfig, *, cond_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        c = int(cfg.base_channels)
        z = int(cfg.latent_channels)
        self.n_mom = int(cfg.n_mom)
        self.separate_heads = bool(cfg.separate_heads)
        self.cond_dim = int(cond_dim)

        self.proj_in = nn.Conv2d(z, 4 * c, 1)
        self.film_b = FiLM(cond_dim, 4 * c)
        self.up3 = nn.ConvTranspose2d(4 * c, 4 * c, 4, stride=2, padding=1)
        self.dec3 = _ConvBlock(8 * c, 4 * c)
        self.film3 = FiLM(cond_dim, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1)
        self.dec2 = _ConvBlock(4 * c, 2 * c)
        self.film2 = FiLM(cond_dim, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)
        self.dec1 = _ConvBlock(2 * c, c)
        self.film1 = FiLM(cond_dim, c)

        if self.separate_heads and self.n_mom > 0 and c_in % self.n_mom == 0:
            n_z = c_in // self.n_mom
            self.dens_head = nn.Conv2d(c, n_z, 1)
            self.moment_head = nn.Conv2d(c, c_in - n_z, 1)
            self._n_dens = n_z
            self.out_proj = None
        else:
            self.dens_head = None
            self.moment_head = None
            self.out_proj = nn.Conv2d(c, c_in, 3, padding=1)
            self._n_dens = 0

        # Spatially-constant bias + cond→spatial skip maps for prior / skip-dropped
        # decode. Constant-only fillers erase bars; z/θ must inject structure at
        # each U-Net scale for scientifically usable latent samples.
        self.skip1 = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.skip2 = nn.Parameter(torch.zeros(1, 2 * c, 1, 1))
        self.skip3 = nn.Parameter(torch.zeros(1, 4 * c, 1, 1))
        g = 4  # coarse layout before bilinear resize to target skip HW
        self.skip1_from_cond = nn.Linear(cond_dim, c * g * g)
        self.skip2_from_cond = nn.Linear(cond_dim, 2 * c * g * g)
        self.skip3_from_cond = nn.Linear(cond_dim, 4 * c * g * g)
        self._skip_grid = g
        for lin in (self.skip1_from_cond, self.skip2_from_cond, self.skip3_from_cond):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _cond_skip_map(
        self, linear: nn.Linear, channels: int, cond: torch.Tensor, hw: tuple[int, int]
    ) -> torch.Tensor:
        g = self._skip_grid
        x = linear(cond).view(cond.shape[0], channels, g, g)
        if x.shape[-2:] != hw:
            x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
        return x

    def prior_skips(
        self, cond: torch.Tensor, shapes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (e1, e2, e3)-shaped skip tensors from the FiLM condition ``(z, θ)``."""
        c = int(self.cfg.base_channels)
        e1 = self._cond_skip_map(self.skip1_from_cond, c, cond, shapes[0]) + self.skip1
        e2 = self._cond_skip_map(self.skip2_from_cond, 2 * c, cond, shapes[1]) + self.skip2
        e3 = self._cond_skip_map(self.skip3_from_cond, 4 * c, cond, shapes[2]) + self.skip3
        return e1, e2, e3

    def prior_skip_level(
        self, level: int, cond: torch.Tensor, hw: tuple[int, int]
    ) -> torch.Tensor:
        """Single prior-path skip map at U-Net level 1/2/3."""
        c = int(self.cfg.base_channels)
        if level == 1:
            return self._cond_skip_map(self.skip1_from_cond, c, cond, hw) + self.skip1
        if level == 2:
            return self._cond_skip_map(self.skip2_from_cond, 2 * c, cond, hw) + self.skip2
        if level == 3:
            return self._cond_skip_map(self.skip3_from_cond, 4 * c, cond, hw) + self.skip3
        raise ValueError(f"skip level must be 1..3, got {level}")

    def _interleave(self, dens: torch.Tensor, moments: torch.Tensor) -> torch.Tensor:
        b, _, h, w = dens.shape
        n_z = self._n_dens
        n_other = self.n_mom - 1
        dens_z = dens.view(b, n_z, 1, h, w)
        mom_z = moments.view(b, n_z, n_other, h, w)
        return torch.cat([dens_z, mom_z], dim=2).reshape(b, n_z * self.n_mom, h, w)

    def forward(
        self,
        bottleneck: torch.Tensor,
        cond: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        *,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        e1, e2, e3 = skips if skips is not None else (None, None, None)
        h = self.film_b(self.proj_in(bottleneck), cond)

        d3 = self.up3(h)
        if e3 is not None:
            if d3.shape[-2:] != e3.shape[-2:]:
                d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
            sk3 = e3
        else:
            sk3 = self.prior_skip_level(3, cond, (d3.shape[2], d3.shape[3]))
        d3 = self.film3(self.dec3(torch.cat([d3, sk3], dim=1)), cond)

        d2 = self.up2(d3)
        if e2 is not None:
            if d2.shape[-2:] != e2.shape[-2:]:
                d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            sk2 = e2
        else:
            sk2 = self.prior_skip_level(2, cond, (d2.shape[2], d2.shape[3]))
        d2 = self.film2(self.dec2(torch.cat([d2, sk2], dim=1)), cond)

        d1 = self.up1(d2)
        if e1 is not None:
            if d1.shape[-2:] != e1.shape[-2:]:
                d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            sk1 = e1
        else:
            sk1 = self.prior_skip_level(1, cond, (d1.shape[2], d1.shape[3]))
        feat = self.film1(self.dec1(torch.cat([d1, sk1], dim=1)), cond)

        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)

        if self.dens_head is None:
            assert self.out_proj is not None
            return self.out_proj(feat)
        dens = self.dens_head(feat)
        moments = self.moment_head(feat)
        n_core = dens.shape[1] * self.n_mom
        core = self._interleave(dens, moments)
        if self.cfg.in_channels == n_core:
            return core
        # Trailing Φ etc.: pad zeros (smoke trains without Φ).
        extra = feat.new_zeros(feat.shape[0], self.cfg.in_channels - n_core, *feat.shape[-2:])
        return torch.cat([core, extra], dim=1)


class MultiTowerSliceVAE(nn.Module):
    """
    Multi-tower conditional VAE over per-component slice stacks.

    * **Encode** each component with a U-Net encoder, fuse pooled cues →
      ``μ, logσ²`` → global ``z``.
    * **Decode** ``(z, θ)`` with FiLM-conditioned U-Net decoders.  Posterior
      decode may use encoder skips; prior decode uses learned skip fillers.
    * Training mixes posterior recon + skip-dropout + a prior-decode recon term
      so ``z ~ N(0,I)|θ`` samples stay morphologically useful.
    """

    def __init__(
        self,
        slice_cfg: MultiScaleSliceConfig,
        *,
        vae_cfg: FieldVAEConfig | None = None,
        include_potential: bool = False,
    ) -> None:
        super().__init__()
        self.slice_cfg = slice_cfg
        self.cfg = vae_cfg or FieldVAEConfig()
        self.include_potential = bool(include_potential)
        vc = self.cfg

        self.theta_mlp = nn.Sequential(
            nn.Linear(vc.theta_dim, vc.bottleneck_channels),
            nn.GELU(),
            nn.Linear(vc.bottleneck_channels, vc.bottleneck_channels),
            nn.GELU(),
        )
        cond_dim = vc.latent_dim + vc.bottleneck_channels
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        encoders = {}
        decoders = {}
        z_to_spatial = {}
        spatial_sizes: dict[str, tuple[int, int]] = {}
        for g in slice_cfg.grids:
            n_ch = g.n_moment_channels + (g.n_z if include_potential else 0)
            acfg = SliceAutoencoderConfig(
                in_channels=n_ch,
                base_channels=vc.base_channels,
                latent_channels=vc.bottleneck_channels,
                n_pix=g.n_pix,
                n_z=g.n_z,
                n_mom=g.n_mom,
                separate_heads=vc.separate_heads,
            )
            encoders[g.name] = SliceUNetEncoder(acfg)
            decoders[g.name] = SliceUNetDecoder(acfg, cond_dim=cond_dim)
            # Bottleneck spatial size after 3× stride-2.
            h = max(1, int(g.n_pix) // 8)
            w = h
            spatial_sizes[g.name] = (h, w)
            z_to_spatial[g.name] = nn.Sequential(
                nn.Linear(vc.latent_dim, vc.bottleneck_channels * h * w),
                nn.GELU(),
            )
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.z_to_spatial = nn.ModuleDict(z_to_spatial)
        self._spatial_sizes = spatial_sizes
        self._tower_order = list(encoders.keys())

        # Spatial flatten → global z (avg-pool alone erases bars).
        g = int(vc.enc_grid)
        self.enc_grid = g
        flat_dim = len(encoders) * vc.bottleneck_channels * g * g
        hid = max(vc.latent_dim * 2, 128)
        self.encode_mlp = nn.Sequential(
            nn.Linear(flat_dim, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
        )
        self.to_mu = nn.Linear(hid, vc.latent_dim)
        self.to_logvar = nn.Linear(hid, vc.latent_dim)
        # Small init (not zero) so rare morphologies can leave the prior mean.
        nn.init.normal_(self.to_mu.weight, std=0.01)
        nn.init.zeros_(self.to_mu.bias)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -2.0)

    def _condition(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        th = self.theta_mlp(theta)
        return self.cond_proj(torch.cat([z, th], dim=-1))

    def encode(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        bottlenecks: dict[str, torch.Tensor] = {}
        skips: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        flats: list[torch.Tensor] = []
        g = self.enc_grid
        for name in self._tower_order:
            b, sk = self.encoders[name](batch[name])
            bottlenecks[name] = b
            skips[name] = sk
            # Keep a coarse spatial layout so bars survive into z.
            spat = F.adaptive_avg_pool2d(b, (g, g))
            flats.append(spat.flatten(1))
        h = self.encode_mlp(torch.cat(flats, dim=-1))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-6.0, 2.0)
        return mu, logvar, bottlenecks, skips

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _bottleneck_from_z(
        self,
        name: str,
        z: torch.Tensor,
        enc_b: torch.Tensor | None,
        *,
        use_skip: bool,
    ) -> torch.Tensor:
        h, w = self._spatial_sizes[name]
        base = self.z_to_spatial[name](z).view(z.shape[0], -1, h, w)
        if use_skip and enc_b is not None:
            if enc_b.shape[-2:] != (h, w):
                enc_b = F.interpolate(enc_b, size=(h, w), mode="bilinear", align_corners=False)
            # Blend encoder bottleneck into z-spatial (posterior path).
            return base + enc_b
        return base

    def decode(
        self,
        z: torch.Tensor,
        theta: torch.Tensor,
        *,
        bottlenecks: dict[str, torch.Tensor] | None = None,
        skips: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
        use_encoder_skips: bool = True,
        target_shapes: dict[str, tuple[int, int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        cond = self._condition(z, theta)
        out: dict[str, torch.Tensor] = {}
        for name in self._tower_order:
            enc_b = None if bottlenecks is None else bottlenecks.get(name)
            sk = None
            if use_encoder_skips and skips is not None:
                sk = skips.get(name)
            bmap = self._bottleneck_from_z(
                name, z, enc_b, use_skip=use_encoder_skips and enc_b is not None
            )
            # FiLM the bottleneck with (z,θ) even on the posterior path.
            dec = self.decoders[name]
            projected = dec.film_b(dec.proj_in(bmap), cond)
            hw = target_shapes[name] if target_shapes else self._infer_hw(name)
            out[name] = self._decode_tower(name, projected, cond, sk, hw)
        return out

    def _infer_hw(self, name: str) -> tuple[int, int]:
        g = self.slice_cfg.grid_for(name)
        return (int(g.n_pix), int(g.n_pix))

    def _decode_tower(
        self,
        name: str,
        projected_b: torch.Tensor,
        cond: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Decode from already FiLM-projected bottleneck features."""
        dec = self.decoders[name]
        e1, e2, e3 = skips if skips is not None else (None, None, None)
        h = projected_b

        d3 = dec.up3(h)
        if e3 is not None:
            if d3.shape[-2:] != e3.shape[-2:]:
                d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
            sk3 = e3
        else:
            sk3 = dec.prior_skip_level(3, cond, (d3.shape[2], d3.shape[3]))
        d3 = dec.film3(dec.dec3(torch.cat([d3, sk3], dim=1)), cond)

        d2 = dec.up2(d3)
        if e2 is not None:
            if d2.shape[-2:] != e2.shape[-2:]:
                d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            sk2 = e2
        else:
            sk2 = dec.prior_skip_level(2, cond, (d2.shape[2], d2.shape[3]))
        d2 = dec.film2(dec.dec2(torch.cat([d2, sk2], dim=1)), cond)

        d1 = dec.up1(d2)
        if e1 is not None:
            if d1.shape[-2:] != e1.shape[-2:]:
                d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            sk1 = e1
        else:
            sk1 = dec.prior_skip_level(1, cond, (d1.shape[2], d1.shape[3]))
        feat = dec.film1(dec.dec1(torch.cat([d1, sk1], dim=1)), cond)
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)

        if dec.dens_head is None:
            assert dec.out_proj is not None
            return dec.out_proj(feat)
        dens = dec.dens_head(feat)
        moments = dec.moment_head(feat)
        n_core = dens.shape[1] * dec.n_mom
        core = dec._interleave(dens, moments)
        if dec.cfg.in_channels == n_core:
            return core
        extra = feat.new_zeros(
            feat.shape[0], dec.cfg.in_channels - n_core, *feat.shape[-2:]
        )
        return torch.cat([core, extra], dim=1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        theta: torch.Tensor,
        *,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        mu, logvar, bottlenecks, skips = self.encode(batch)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        # Default generative path: decode from z only (no encoder skips).
        use_skips = False
        if self.training and float(self.cfg.skip_dropout) < 1.0:
            if torch.rand(()) >= float(self.cfg.skip_dropout):
                use_skips = True
        target_shapes = {k: (v.shape[-2], v.shape[-1]) for k, v in batch.items()}
        recon = self.decode(
            z,
            theta,
            bottlenecks=bottlenecks if use_skips else None,
            skips=skips if use_skips else None,
            use_encoder_skips=use_skips,
            target_shapes=target_shapes,
        )
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "bottlenecks": bottlenecks,
            "skips": skips,
            "used_skips": use_skips,
        }

    def sample(
        self,
        theta: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        target_shapes: dict[str, tuple[int, int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prior / interpolation sample: ``z ~ N(0,I)`` (or provided) → field stacks."""
        b = theta.shape[0]
        if z is None:
            z = torch.randn(b, self.cfg.latent_dim, device=theta.device, dtype=theta.dtype)
        shapes = target_shapes or {name: self._infer_hw(name) for name in self._tower_order}
        return self.decode(
            z,
            theta,
            bottlenecks=None,
            skips=None,
            use_encoder_skips=False,
            target_shapes=shapes,
        )

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Per-batch mean KL; optional free-bits floor per latent dim."""
        kl_el = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        if self.cfg.free_bits > 0.0:
            kl_el = torch.clamp(kl_el, min=float(self.cfg.free_bits))
        return kl_el.sum(dim=-1).mean()


def field_vae_loss(
    model: MultiTowerSliceVAE,
    batch: dict[str, torch.Tensor],
    theta: torch.Tensor,
    *,
    dens_indices: dict[str, list[int]],
    dens_weight: float = 4.0,
    moment_weight: float = 6.0,
    a2_weight: float = 0.0,
    fourier_modes: tuple[int, ...] = (1, 2, 3),
    fourier_n_bins: int = 12,
    component_weights: dict[str, float] | None = None,
    beta: float | None = None,
) -> dict[str, torch.Tensor]:
    """
    Primary recon on decode(``z``) without skips + ``β`` KL.

    Optional weak skip-assisted recon (``skip_recon_weight``) can sharpen maps
    without becoming the sampling path. Dens / moment weights match the crisp AE.
    """
    out = model(batch, theta, sample_posterior=True)
    # Force generative-path recon even if this batch used skips in forward.
    target_shapes = {k: (v.shape[-2], v.shape[-1]) for k, v in batch.items()}
    gen_recon = model.decode(
        out["z"],
        theta,
        bottlenecks=None,
        skips=None,
        use_encoder_skips=False,
        target_shapes=target_shapes,
    )
    recon_m = multitower_reconstruction_loss(
        gen_recon,
        batch,
        dens_indices=dens_indices,
        dens_weight=dens_weight,
        moment_weight=moment_weight,
        a2_weight=a2_weight,
        fourier_modes=fourier_modes,
        fourier_n_bins=fourier_n_bins,
        component_weights=component_weights,
    )
    kl = model.kl_loss(out["mu"], out["logvar"])
    beta_v = float(model.cfg.beta if beta is None else beta)
    loss = recon_m["loss"] + beta_v * kl

    metrics: dict[str, torch.Tensor] = {
        "loss": loss,
        "recon": recon_m["loss"],
        "mse_dens": recon_m["mse_dens"],
        "mse_mom": recon_m["mse_mom"],
        "kl": kl,
        "beta": loss.new_tensor(beta_v),
        "prior_recon": recon_m["loss"],
    }
    if "fourier_mse" in recon_m:
        metrics["fourier_mse"] = recon_m["fourier_mse"]
    if "a2_mse" in recon_m:
        metrics["a2_mse"] = recon_m["a2_mse"]

    skip_w = float(model.cfg.skip_recon_weight)
    if skip_w > 0.0:
        skip_recon = model.decode(
            out["z"],
            theta,
            bottlenecks=out["bottlenecks"],
            skips=out["skips"],
            use_encoder_skips=True,
            target_shapes=target_shapes,
        )
        skip_m = multitower_reconstruction_loss(
            skip_recon,
            batch,
            dens_indices=dens_indices,
            dens_weight=dens_weight,
            moment_weight=moment_weight,
            a2_weight=a2_weight * 0.5,
            fourier_modes=fourier_modes,
            fourier_n_bins=fourier_n_bins,
            component_weights=component_weights,
        )
        metrics["loss"] = loss + skip_w * skip_m["loss"]
        metrics["skip_recon"] = skip_m["loss"]

    metrics["recon_stacks"] = gen_recon
    metrics["mu"] = out["mu"]
    metrics["logvar"] = out["logvar"]
    metrics["z"] = out["z"]
    return metrics
