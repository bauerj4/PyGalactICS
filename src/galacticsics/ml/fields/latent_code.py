"""Frozen crisp AE + single-``z`` skip distillation for generative sampling.

Root cause of overnight field-VAE washout: bars live in U-Net **skips**, not in
a global ``z``.  Pure ``decode(z, θ)`` with weak skip fillers erases A₂.

This module separates concerns:

1. **Frozen teacher** (:class:`~galacticsics.ml.fields.autoencoder.MultiTowerSliceAE`)
   keeps crisp dens+moment decode.
2. **Code encoder** pools teacher bottlenecks → single ``μ, logσ² → z``.
3. **Skip synthesizer** maps ``(z, θ[, morph])`` → bottleneck + skip maps
   (coarse grid → bilinear), distilled to match teacher features.
4. **Sample API** stays ``z ~ N(0,I) | θ`` → fields (one latent vector).

Optional RealNVP prior on teacher codes for sharper morphology sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from galacticsics.ml.fields.autoencoder import (
    MultiTowerSliceAE,
    multitower_reconstruction_loss,
    soft_a2_from_dens_maps,
)
from galacticsics.ml.fields.binning import MultiScaleSliceConfig


@dataclass
class LatentCodeConfig:
    """Knobs for :class:`FrozenAECodeVAE`."""

    latent_dim: int = 128
    theta_dim: int = 10
    morph_dim: int = 0
    """Extra morphology conditioning (e.g. A₂ summary); 0 = structural θ only."""
    enc_grid: int = 4
    synth_grid: int = 8
    beta: float = 1e-3
    free_bits: float = 0.05
    skip_weight: float = 1.0
    bottleneck_weight: float = 1.0
    recon_weight: float = 1.0
    # Skip level weights (e1 carries high-freq bar; keep ≥ e3).
    w_e1: float = 1.5
    w_e2: float = 1.0
    w_e3: float = 0.75
    deterministic: bool = False
    """If True, encode → μ only (no KL); pair with a flow prior on codes."""


class FiLM1d(nn.Module):
    def __init__(self, cond_dim: int, n_features: int) -> None:
        super().__init__()
        self.to_g = nn.Linear(cond_dim, n_features)
        self.to_b = nn.Linear(cond_dim, n_features)
        nn.init.zeros_(self.to_g.weight)
        nn.init.zeros_(self.to_g.bias)
        nn.init.zeros_(self.to_b.weight)
        nn.init.zeros_(self.to_b.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.to_g(cond)) + self.to_b(cond)


class SkipSynthTower(nn.Module):
    """Predict bottleneck + (e1,e2,e3) from a condition vector at coarse grid."""

    def __init__(
        self,
        *,
        cond_dim: int,
        base_channels: int,
        bottleneck_channels: int,
        synth_grid: int = 8,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        # Teacher SliceUNet bottleneck ends at 4·c channels.
        self.bn_ch = 4 * c
        self.base_c = c
        self.g = int(synth_grid)
        hid = max(cond_dim, 256)
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
        )
        g2 = self.g * self.g
        self.to_bn = nn.Linear(hid, self.bn_ch * g2)
        self.to_e3 = nn.Linear(hid, 4 * c * g2)
        self.to_e2 = nn.Linear(hid, 2 * c * g2)
        self.to_e1 = nn.Linear(hid, c * g2)
        # Small residual biases so synth can start near zero maps.
        self.bias_bn = nn.Parameter(torch.zeros(1, self.bn_ch, 1, 1))
        self.bias_e1 = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.bias_e2 = nn.Parameter(torch.zeros(1, 2 * c, 1, 1))
        self.bias_e3 = nn.Parameter(torch.zeros(1, 4 * c, 1, 1))
        for lin in (self.to_bn, self.to_e3, self.to_e2, self.to_e1):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def _map(self, linear: nn.Linear, ch: int, h: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        g = self.g
        x = linear(h).view(h.shape[0], ch, g, g)
        if x.shape[-2:] != hw:
            x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
        return x

    def forward(
        self,
        cond: torch.Tensor,
        *,
        bn_hw: tuple[int, int],
        e1_hw: tuple[int, int],
        e2_hw: tuple[int, int],
        e3_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h = self.trunk(cond)
        bn = self._map(self.to_bn, self.bn_ch, h, bn_hw) + self.bias_bn
        e3 = self._map(self.to_e3, 4 * self.base_c, h, e3_hw) + self.bias_e3
        e2 = self._map(self.to_e2, 2 * self.base_c, h, e2_hw) + self.bias_e2
        e1 = self._map(self.to_e1, self.base_c, h, e1_hw) + self.bias_e1
        return {"bottleneck": bn, "skips": (e1, e2, e3)}


class AffineCoupling(nn.Module):
    """RealNVP-style coupling for a cheap conditional prior on codes."""

    def __init__(self, dim: int, cond_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.dim = int(dim)
        self.split = self.dim // 2
        n_out = self.dim - self.split
        self.net = nn.Sequential(
            nn.Linear(self.split + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * n_out),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, *, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[:, : self.split], x[:, self.split :]
        h = self.net(torch.cat([x1, cond], dim=-1))
        log_s, t = h.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * 2.0  # stabilize
        if reverse:
            y2 = (x2 - t) * torch.exp(-log_s)
            log_det = -log_s.sum(dim=-1)
        else:
            y2 = x2 * torch.exp(log_s) + t
            log_det = log_s.sum(dim=-1)
        return torch.cat([x1, y2], dim=-1), log_det


class CondRealNVP(nn.Module):
    """Stack of affine couplings; base measure ``N(0,I)``."""

    def __init__(
        self, dim: int, cond_dim: int, *, n_flows: int = 4, hidden: int = 128
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.flows = nn.ModuleList(
            [AffineCoupling(dim, cond_dim, hidden=hidden) for _ in range(n_flows)]
        )
        # Alternate which half is transformed via a fixed permutation.
        self.register_buffer("perm", torch.arange(dim).flip(0), persistent=True)

    def _permute(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm]

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor, *, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = z.new_zeros(z.shape[0])
        if reverse:
            x = z
            for flow in reversed(self.flows):
                x = self._permute(x)
                x, ld = flow(x, cond, reverse=True)
                log_det = log_det + ld
            return x, log_det
        x = z
        for flow in self.flows:
            x, ld = flow(x, cond, reverse=False)
            log_det = log_det + ld
            x = self._permute(x)
        return x, log_det

    def log_prob(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        u, log_det = self.forward(z, cond, reverse=False)
        log_pu = -0.5 * (u.pow(2).sum(dim=-1) + self.dim * 1.83787706641)  # log(2π)
        return log_pu + log_det

    def sample(self, cond: torch.Tensor, n: int | None = None) -> torch.Tensor:
        b = cond.shape[0] if n is None else int(n)
        u = torch.randn(b, self.dim, device=cond.device, dtype=cond.dtype)
        if n is not None and n != cond.shape[0]:
            cond = cond[:1].expand(b, -1)
        z, _ = self.forward(u, cond, reverse=True)
        return z


class FrozenAECodeVAE(nn.Module):
    """
    Single-``z`` generative model on top of a **frozen** multi-tower AE.

    * Encode fields with teacher → pool → ``μ,logσ² → z``.
    * Synthesize bottleneck+skips from ``(z, θ[, morph])``.
    * Decode with **frozen** teacher decoder (preserves crisp dens/moment heads).
    """

    def __init__(
        self,
        teacher: MultiTowerSliceAE,
        *,
        cfg: LatentCodeConfig | None = None,
        use_flow_prior: bool = False,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.cfg = cfg or LatentCodeConfig()
        self.use_flow_prior = bool(use_flow_prior)
        # Freeze teacher completely.
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        vc = self.cfg
        base_c = int(next(iter(teacher.towers.values())).cfg.base_channels)
        self.base_channels = base_c
        cond_in = vc.latent_dim + vc.theta_dim + int(vc.morph_dim)
        self.theta_mlp = nn.Sequential(
            nn.Linear(vc.theta_dim + int(vc.morph_dim), 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.cond_dim = vc.latent_dim + 128
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.GELU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        g = int(vc.enc_grid)
        self.enc_grid = g
        n_towers = len(teacher._tower_order)
        # Teacher bottleneck channels = 4 · base_c.
        bn_ch = 4 * base_c
        flat = n_towers * bn_ch * g * g
        hid = max(vc.latent_dim * 2, 256)
        self.encode_mlp = nn.Sequential(
            nn.Linear(flat, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
        )
        self.to_mu = nn.Linear(hid, vc.latent_dim)
        self.to_logvar = nn.Linear(hid, vc.latent_dim)
        nn.init.normal_(self.to_mu.weight, std=0.01)
        nn.init.zeros_(self.to_mu.bias)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -2.0)

        synth = {}
        for name in teacher._tower_order:
            synth[name] = SkipSynthTower(
                cond_dim=self.cond_dim,
                base_channels=base_c,
                bottleneck_channels=bn_ch,
                synth_grid=vc.synth_grid,
            )
        self.synth = nn.ModuleDict(synth)
        self._tower_order = list(teacher._tower_order)

        if self.use_flow_prior:
            self.flow = CondRealNVP(vc.latent_dim, 128, n_flows=4, hidden=128)
        else:
            self.flow = None

    def train(self, mode: bool = True):  # noqa: A003
        # Keep teacher in eval even when student trains.
        super().train(mode)
        self.teacher.eval()
        return self

    def _theta_feat(self, theta: torch.Tensor, morph: torch.Tensor | None) -> torch.Tensor:
        if morph is not None and morph.numel() > 0:
            return self.theta_mlp(torch.cat([theta, morph], dim=-1))
        if self.cfg.morph_dim > 0:
            pad = theta.new_zeros(theta.shape[0], int(self.cfg.morph_dim))
            return self.theta_mlp(torch.cat([theta, pad], dim=-1))
        return self.theta_mlp(theta)

    def _condition(
        self, z: torch.Tensor, theta: torch.Tensor, morph: torch.Tensor | None = None
    ) -> torch.Tensor:
        th = self._theta_feat(theta, morph)
        return self.cond_proj(torch.cat([z, th], dim=-1))

    @torch.no_grad()
    def teacher_features(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]]:
        return self.teacher.encode_features(batch)

    def encode_from_features(
        self, features: dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flats: list[torch.Tensor] = []
        g = self.enc_grid
        for name in self._tower_order:
            b = features[name]["bottleneck"]
            assert isinstance(b, torch.Tensor)
            spat = F.adaptive_avg_pool2d(b, (g, g))
            flats.append(spat.flatten(1))
        h = self.encode_mlp(torch.cat(flats, dim=-1))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-6.0, 2.0)
        return mu, logvar

    def encode(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            feats = self.teacher_features(batch)
        return self.encode_from_features(feats)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def synthesize(
        self,
        z: torch.Tensor,
        theta: torch.Tensor,
        *,
        morph: torch.Tensor | None = None,
        ref_features: dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]]
        | None = None,
        target_shapes: dict[str, tuple[int, int]] | None = None,
    ) -> dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]]:
        cond = self._condition(z, theta, morph)
        out: dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]] = {}
        for name in self._tower_order:
            if ref_features is not None and name in ref_features:
                tb = ref_features[name]["bottleneck"]
                te1, te2, te3 = ref_features[name]["skips"]  # type: ignore[misc]
                assert isinstance(tb, torch.Tensor)
                bn_hw = (tb.shape[2], tb.shape[3])
                e1_hw = (te1.shape[2], te1.shape[3])
                e2_hw = (te2.shape[2], te2.shape[3])
                e3_hw = (te3.shape[2], te3.shape[3])
            else:
                # Infer from SliceUNet geometry: 3× stride-2 → H/8 bottleneck.
                pix = (
                    int(target_shapes[name][0])
                    if target_shapes is not None and name in target_shapes
                    else int(self.teacher.towers[name].cfg.n_pix)
                )
                e1_hw = (pix, pix)
                e2_hw = (max(1, pix // 2), max(1, pix // 2))
                e3_hw = (max(1, pix // 4), max(1, pix // 4))
                bn_hw = (max(1, pix // 8), max(1, pix // 8))
            out[name] = self.synth[name](
                cond, bn_hw=bn_hw, e1_hw=e1_hw, e2_hw=e2_hw, e3_hw=e3_hw
            )
        return out

    def decode_features(
        self,
        features: dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]],
        *,
        target_shapes: dict[str, tuple[int, int]] | None = None,
        n_channels: dict[str, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.teacher.decode_features(
            features, target_shapes=target_shapes, n_channels=n_channels
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        theta: torch.Tensor,
        *,
        morph: torch.Tensor | None = None,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher_feats = self.teacher_features(batch)
        mu, logvar = self.encode_from_features(teacher_feats)
        if bool(self.cfg.deterministic):
            z = mu
        else:
            z = self.reparameterize(mu, logvar) if sample_posterior else mu
        target_shapes = {k: (v.shape[-2], v.shape[-1]) for k, v in batch.items()}
        n_ch = {k: int(v.shape[1]) for k, v in batch.items()}
        synth = self.synthesize(
            z, theta, morph=morph, ref_features=teacher_feats, target_shapes=target_shapes
        )
        recon = self.decode_features(synth, target_shapes=target_shapes, n_channels=n_ch)
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "synth_features": synth,
            "teacher_features": teacher_feats,
        }

    @torch.no_grad()
    def sample(
        self,
        theta: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        morph: torch.Tensor | None = None,
        target_shapes: dict[str, tuple[int, int]] | None = None,
        n_channels: dict[str, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prior / given-``z`` sample → normalized field stacks."""
        b = theta.shape[0]
        if z is None:
            if self.flow is not None:
                th = self._theta_feat(theta, morph)
                z = self.flow.sample(th)
            else:
                z = torch.randn(b, self.cfg.latent_dim, device=theta.device, dtype=theta.dtype)
        synth = self.synthesize(z, theta, morph=morph, target_shapes=target_shapes)
        return self.decode_features(synth, target_shapes=target_shapes, n_channels=n_channels)

    @torch.no_grad()
    def encode_mu(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mu, _ = self.encode(batch)
        return mu


def _skip_distill_loss(
    synth: dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]],
    teacher: dict[str, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]],
    *,
    w_bn: float,
    w_e1: float,
    w_e2: float,
    w_e3: float,
    component_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    total = None
    for name, tfeat in teacher.items():
        if name not in synth:
            continue
        cw = 1.0 if component_weights is None else float(component_weights.get(name, 1.0))
        sb = synth[name]["bottleneck"]
        tb = tfeat["bottleneck"]
        assert isinstance(sb, torch.Tensor) and isinstance(tb, torch.Tensor)
        se1, se2, se3 = synth[name]["skips"]  # type: ignore[misc]
        te1, te2, te3 = tfeat["skips"]  # type: ignore[misc]
        term = (
            float(w_bn) * F.mse_loss(sb, tb)
            + float(w_e1) * F.mse_loss(se1, te1)
            + float(w_e2) * F.mse_loss(se2, te2)
            + float(w_e3) * F.mse_loss(se3, te3)
        )
        term = term * cw
        total = term if total is None else total + term
    assert total is not None
    return total


def morph_a2_summary(
    dens_map: torch.Tensor, *, n_bins: int = 8
) -> torch.Tensor:
    """Compact morphology vector from a face-on dens collapse ``(B,H,W)``.

    Returns ``(B, 4)``: soft map-wide A₂, mean/max radial A₂ amp, and A₂ at
    mid-radius — for conditioning the prior without inventing quiet bars.
    """
    from galacticsics.ml.fields.autoencoder import soft_am_radial_from_dens_maps

    a2 = soft_a2_from_dens_maps(dens_map).unsqueeze(-1)
    rad = soft_am_radial_from_dens_maps(dens_map, m=2, n_bins=n_bins)
    amp = rad["amp"]
    mean_a = amp.mean(dim=-1, keepdim=True)
    max_a = amp.max(dim=-1, keepdim=True).values
    mid = amp[:, n_bins // 2 : n_bins // 2 + 1]
    return torch.cat([a2, mean_a, max_a, mid], dim=-1)


def disk_dens_collapse(
    stack: torch.Tensor, *, n_z: int, n_mom: int
) -> torch.Tensor:
    dens = stack.new_zeros(stack.shape[0], stack.shape[-2], stack.shape[-1])
    for iz in range(int(n_z)):
        dens = dens + stack[:, iz * n_mom].clamp_min(0.0)
    return dens


def latent_code_loss(
    model: FrozenAECodeVAE,
    batch: dict[str, torch.Tensor],
    theta: torch.Tensor,
    *,
    dens_indices: dict[str, list[int]],
    dens_weight: float = 4.0,
    moment_weight: float = 6.0,
    a2_weight: float = 0.0,
    component_weights: dict[str, float] | None = None,
    morph: torch.Tensor | None = None,
    flow_nll_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Skip distill + frozen-decoder recon + KL (+ optional flow NLL)."""
    out = model(batch, theta, morph=morph, sample_posterior=True)
    cfg = model.cfg
    recon_m = multitower_reconstruction_loss(
        out["recon"],
        batch,
        dens_indices=dens_indices,
        dens_weight=dens_weight,
        moment_weight=moment_weight,
        a2_weight=a2_weight,
        component_weights=component_weights,
    )
    skip_l = _skip_distill_loss(
        out["synth_features"],
        out["teacher_features"],
        w_bn=cfg.bottleneck_weight,
        w_e1=cfg.w_e1,
        w_e2=cfg.w_e2,
        w_e3=cfg.w_e3,
        component_weights=component_weights,
    )
    mu, logvar = out["mu"], out["logvar"]
    if bool(cfg.deterministic):
        kl = mu.new_zeros(())
    else:
        # Free-bits KL.
        kl_el = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl_el = torch.clamp(kl_el, min=float(cfg.free_bits))
        kl = kl_el.sum(dim=-1).mean()

    loss = (
        float(cfg.recon_weight) * recon_m["loss"]
        + float(cfg.skip_weight) * skip_l
        + (0.0 if bool(cfg.deterministic) else float(cfg.beta) * kl)
    )
    metrics: dict[str, torch.Tensor] = {
        "loss": loss,
        "mse_dens": recon_m["mse_dens"],
        "mse_mom": recon_m["mse_mom"],
        "skip_distill": skip_l,
        "kl": kl,
        "recon": recon_m["loss"],
    }
    if flow_nll_weight > 0.0 and model.flow is not None:
        th = model._theta_feat(theta, morph)
        # Train flow on posterior μ (detached) as codes.
        nll = -model.flow.log_prob(mu.detach(), th).mean()
        metrics["loss"] = metrics["loss"] + float(flow_nll_weight) * nll
        metrics["flow_nll"] = nll
    return metrics


def load_frozen_teacher(
    ckpt_path,
    slice_cfg: MultiScaleSliceConfig,
    *,
    device: str = "cpu",
) -> tuple[MultiTowerSliceAE, dict]:
    """Load crisp AE checkpoint as a frozen teacher."""
    from pathlib import Path

    path = Path(ckpt_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    # Auto-detect 1×1 vs deep dens/moment heads from checkpoint keys.
    sd = ckpt["model"]
    deep_heads = any(k.endswith("dens_head.0.weight") for k in sd)
    teacher = MultiTowerSliceAE(
        slice_cfg,
        include_potential=bool(ckpt.get("include_potential", False)),
        base_channels=int(args.get("base_channels", 48)),
        latent_channels=int(args.get("latent_channels", 128)),
        arch=str(ckpt.get("arch", "unet")),
        separate_heads=True,
        cross_tower_attention=not bool(args.get("no_cross_tower", False)),
        deep_heads=deep_heads,
    )
    teacher.load_state_dict(sd, strict=True)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, ckpt
