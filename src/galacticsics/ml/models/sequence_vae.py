"""Conditional set VAE over Morton-subsampled galaxy particles.

The model is a **conditional variational autoencoder** ``p(particles | θ)``
where ``θ`` is a vector of galaxy structural parameters (disk mass, scale
lengths, Toomre Q, halo/bulge params, snapshot time, …).

**Component story (architecture, not a hyperparam).** Disk / halo / bulge are
dynamically and spatially distinct, so identity must be almost determined by
phase space:

1. **Generate.** Sample ``c`` from an easy ``θ``-conditioned mix prior (or
   hard stratified counts), then decode ``x, v | c, θ, z`` with strong
   component-conditional geometric bases.
2. **Encode.** Pool *per-component* dynamics stats; classify particles from
   ``(x, v)`` features — per-particle CE should be nearly trivial.
3. **Mix supervision.** Fraction MSE / stratified counts only — never fight
   a global ``mix_head(z, θ)`` CE against soft targets that never sees
   per-particle phase space.

Particles are represented as tokens ``(c, Δm, x, v)`` from
:func:`~galacticsics.ml.morton.tokenize.tokenize_morton`, but the decoder is a
**parallel set decoder**: given latent ``z`` and ``θ``, it draws ``N``
independent noise queries and maps them to tokens in one shot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("galacticsics[ml] / torch required for SequenceVAE") from exc

from galacticsics.ml.profiles import profile_reconstruction_loss, virial_consistency_loss


@dataclass
class SequenceVAEConfig:
    """
    Hyperparameters for :class:`SequenceVAE`.

    Attributes
    ----------
    n_particles : int
        Default set size ``N`` used when decoding without an explicit ``n``.
    theta_dim : int
        Dimension of the conditioning vector ``θ``.
    d_model : int
        Transformer / MLP width.
    latent_dim : int
        Size of the global latent ``z``.
    n_layers : int
        Optional light attention layers on a subsample (DeepSets is primary).
    n_heads : int
        Attention heads (must divide ``d_model``).
    n_decode_layers : int
        Unused (kept for checkpoint compatibility); geometric decoder only.
    beta : float
        KL weight in the β-VAE objective.
    lambda_recon : float
        Weight on Chamfer / token recon.
    lambda_chamfer : float
        Weight on symmetric Chamfer L2 for ``(x, v)``.
    lambda_ce : float
        Weight on **per-particle** phase-space CE (should be easy).
    lambda_mix : float
        Weight on easy mix-fraction MSE from the ``θ`` prior.
    lambda_index : float
        Weight on index-aligned token MSE (keep ``0`` for unordered sets).
    dropout : float
        Dropout inside optional encoder attention.
    lambda_sigma, lambda_rho, lambda_vphi : float
        Disk ``Σ(R)``, spheroid ``ρ(r)``, disk ``⟨v_φ⟩(R)`` profile weights.
    lambda_nonaxisym : float
        Soft azimuthal Fourier matching (bars / spirals).
    lambda_maps : float
        Soft face-on / edge-on density map matching.
    lambda_virial : float
        Stratified-subsample Plummer virial consistency.
    virial_n_sub : int
        Particles for the differentiable pairwise virial term (``O(n²)``).
    virial_eps : float
        Plummer softening [kpc] inside the virial term.
    lambda_virial_target : float
        Extra pull of predicted ``2K/|W|`` toward ``virial_target_ratio``.
    virial_target_ratio : float
        Soft self-gravity prior (``~1``).
    fourier_modes : tuple of int
        Modes ``m`` for ``A_m/A_0`` and phase (cos/sin) reconstruction.
    fourier_z_max : float
        Soft midplane cut [kpc] for Fourier / face-on map estimates.
    map_n_pix : int
        Pixels per side for map auxiliaries.
    profile_n_bins : int
        Soft radial bins for profile auxiliaries.
    profile_r_max_disk, profile_r_max_sph : float
        Outer radii [kpc] for disk and spheroid profiles.
    """

    n_particles: int = 128
    theta_dim: int = 10
    d_model: int = 64
    latent_dim: int = 32
    n_layers: int = 1
    n_heads: int = 2
    n_decode_layers: int = 1
    beta: float = 0.01
    lambda_recon: float = 1.0
    lambda_chamfer: float = 1.0
    lambda_ce: float = 1.0
    lambda_mix: float = 1.0
    lambda_index: float = 0.0
    dropout: float = 0.1
    lambda_sigma: float = 2.0
    lambda_rho: float = 1.0
    lambda_vphi: float = 1.0
    lambda_nonaxisym: float = 6.0
    lambda_maps: float = 6.0
    lambda_virial: float = 2.0
    virial_n_sub: int = 256
    virial_eps: float = 0.1
    lambda_virial_target: float = 1.5
    virial_target_ratio: float = 1.0
    fourier_modes: tuple[int, ...] = (1, 2)
    fourier_z_max: float = 0.5
    map_n_pix: int = 32
    profile_n_bins: int = 16
    profile_r_max_disk: float = 15.0
    profile_r_max_sph: float = 40.0
    query_noise_scale: float = 1.0
    slot_scale: float = 0.0
    chamfer_max_n: int = 512
    enc_attn_n: int = 256
    use_deepsets_encoder: bool = True
    # Corpus default disk:halo:bulge ≈ 4:2:1
    default_mix: tuple[float, float, float] = (4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0)


def _norm_dm(dm: "torch.Tensor") -> "torch.Tensor":
    """Compress Morton-key increments for features / MSE."""
    return torch.log1p(dm.clamp_min(0.0))


def _norm_pos(x: "torch.Tensor") -> "torch.Tensor":
    """Signed ``log1p`` positions so halo scales do not drown the disk."""
    return torch.sign(x) * torch.log1p(x.abs())


def _phase_features(dx: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
    """
    Dynamics / geometry features for component classification (no ``c``).

    Shape ``(B, N, 10)``: signed-log ``x``, ``v``, ``R``, ``|z|``, ``r``, ``|v|``.
    Disk / halo / bulge separate cleanly on these coordinates.
    """
    R = torch.sqrt(dx[..., 0] ** 2 + dx[..., 1] ** 2 + 1e-8)
    z_abs = dx[..., 2].abs()
    r = torch.linalg.norm(dx, dim=-1)
    speed = torch.linalg.norm(v, dim=-1)
    return torch.cat(
        [
            _norm_pos(dx),
            v,
            R.unsqueeze(-1),
            z_abs.unsqueeze(-1),
            r.unsqueeze(-1),
            speed.unsqueeze(-1),
        ],
        dim=-1,
    )


def _masked_mean_max(
    h: "torch.Tensor",
    mask: "torch.Tensor",
) -> "torch.Tensor":
    """
    Mean and max pool over a boolean particle mask.

    Parameters
    ----------
    h : Tensor, shape (B, N, D)
    mask : Tensor, shape (B, N), bool

    Returns
    -------
    pooled : Tensor, shape (B, 2D)
        ``[masked_mean || masked_max]``; empty masks → zeros.
    """
    m = mask.unsqueeze(-1).to(dtype=h.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    mean = (h * m).sum(dim=1) / denom
    neg_inf = torch.finfo(h.dtype).min
    h_masked = h.masked_fill(~mask.unsqueeze(-1), neg_inf)
    mx = h_masked.amax(dim=1)
    mx = torch.where(mask.any(dim=1, keepdim=True), mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


def chamfer_l2(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Symmetric Chamfer L2 between point sets ``a, b`` of shape ``(B, N, D)``."""
    d = torch.cdist(a, b, p=2)
    return d.min(dim=-1).values.mean() + d.min(dim=-2).values.mean()


def chamfer_l2_subsampled(
    a: "torch.Tensor",
    b: "torch.Tensor",
    *,
    max_n: int = 512,
) -> "torch.Tensor":
    """Chamfer on a random subset of each cloud (keeps O(max_n²) for large N)."""
    n = int(a.shape[1])
    m = int(max_n)
    if n > m:
        idx_a = torch.randperm(n, device=a.device)[:m]
        idx_b = torch.randperm(n, device=b.device)[:m]
        a = a[:, idx_a]
        b = b[:, idx_b]
    return chamfer_l2(a, b)


def stratified_component_ids(
    mix_logits: "torch.Tensor",
    n: int,
    *,
    generator: "torch.Generator | None" = None,
) -> "torch.Tensor":
    """
    Hard stratified ``c`` draws from a mix prior (exact counts ≈ softmax).

    Prefer this over i.i.d. Categorical sampling so generated clouds keep the
    corpus 4:2:1 (or learned) fractions without CE fighting.
    """
    b = mix_logits.shape[0]
    probs = torch.softmax(mix_logits, dim=-1)
    raw = (probs * float(n)).floor().long()
    out = []
    for bi in range(b):
        counts = raw[bi].clone()
        leftover = int(n - int(counts.sum().item()))
        if leftover > 0:
            extra = torch.multinomial(probs[bi], leftover, replacement=True)
            counts.scatter_add_(0, extra, torch.ones_like(extra))
        elif leftover < 0:
            for _ in range(-leftover):
                j = int(counts.argmax())
                if counts[j] > 0:
                    counts[j] -= 1
        parts = [
            torch.full((int(counts[k]),), k, device=mix_logits.device, dtype=torch.long)
            for k in range(3)
            if int(counts[k]) > 0
        ]
        row = (
            torch.cat(parts, dim=0)
            if parts
            else torch.zeros(0, device=mix_logits.device, dtype=torch.long)
        )
        if row.numel() < n:
            pad = torch.multinomial(probs[bi], n - int(row.numel()), replacement=True)
            row = torch.cat([row, pad], dim=0)
        row = row[:n]
        perm = torch.randperm(n, device=row.device, generator=generator)
        out.append(row[perm])
    return torch.stack(out, dim=0)


class SequenceVAE(nn.Module):
    """
    Conditional set VAE for galaxy particle clouds.

    **Encoder.** Phase-space MLP → per-component mean/max pools → ``(μ, log σ²)``.
    Labels only stratify the pools; features themselves carry no one-hot ``c``.

    **Classifier.** Small MLP on ``(x, v)`` dynamics → per-particle logits.
    Hard CE against component labels is the primary identity loss (should be easy).

    **Decoder.** Sample ``c ∼ MixPrior(θ)`` (stratified), then geometric
    ``x, v | c, θ, z`` bases + residual.  Mix prior is supervised with fraction
    MSE only — never a global soft-CE against ``mix_head(z, θ)``.
    """

    def __init__(self, config: SequenceVAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or SequenceVAEConfig()
        cfg = self.config
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")

        self.theta_mlp = nn.Sequential(
            nn.Linear(cfg.theta_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        # Dynamics-only particle encoder (no one-hot c)
        self.enc_mlp = nn.Sequential(
            nn.Linear(10, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        # 3 components × (mean||max) + θ embedding
        self.enc_fuse = nn.Sequential(
            nn.Linear(cfg.d_model * 6 + cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        if cfg.n_layers > 0 and cfg.enc_attn_n > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
            self.attn_fuse = nn.Linear(cfg.d_model, cfg.d_model)
        else:
            self.encoder = None
            self.attn_fuse = None
        self.to_mu = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.to_logvar = nn.Linear(cfg.d_model, cfg.latent_dim)

        # Per-particle phase-space classifier (CE should be nearly trivial)
        self.c_classifier = nn.Sequential(
            nn.Linear(10, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 3),
        )

        # Easy θ → mix prior (fraction MSE only; init near corpus 4:2:1)
        self.mix_prior = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 3),
        )
        with torch.no_grad():
            prior = torch.tensor(cfg.default_mix, dtype=torch.float32).clamp_min(1e-6)
            self.mix_prior[-1].bias.copy_(torch.log(prior))
            self.mix_prior[-1].weight.zero_()

        # Parallel set decoder: geometric component bases + residual MLP
        self.cond_proj = nn.Linear(cfg.latent_dim + cfg.d_model, cfg.d_model)
        self.c_embed = nn.Embedding(3, cfg.d_model)
        self.noise_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.scale_head = nn.Linear(cfg.d_model, 4)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        self.resid_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 6),  # Δx(3) + Δv(3)
        )
        nn.init.zeros_(self.resid_mlp[-1].weight)
        nn.init.zeros_(self.resid_mlp[-1].bias)
        self.bar_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 2),  # (amp, phase)
        )
        nn.init.zeros_(self.bar_head[-1].weight)
        nn.init.zeros_(self.bar_head[-1].bias)
        self.resid_scale = nn.Parameter(torch.tensor(0.25))
        self.v_scale = nn.Parameter(torch.tensor(0.5))
        # Legacy aliases kept so old checkpoints load with strict=False
        self.mix_head = self.mix_prior[-1]
        self.in_proj = nn.Linear(10, cfg.d_model)
        self.decoder = None
        self.slot_proj = nn.Identity()
        self.query_mix = nn.Identity()
        self.head_cont = nn.Identity()
        self.x_scale = nn.Parameter(torch.tensor(1.0))

    def classify_particles(
        self,
        dx: "torch.Tensor",
        v: "torch.Tensor",
    ) -> "torch.Tensor":
        """Per-particle component logits from phase space, shape ``(B, N, 3)``."""
        return self.c_classifier(_phase_features(dx, v))

    def mix_logits_from_theta(self, theta: "torch.Tensor") -> "torch.Tensor":
        """Easy ``θ``-conditioned mix prior logits, shape ``(B, 3)``."""
        return self.mix_prior(self.theta_mlp(theta))

    def encode(
        self,
        c: "torch.Tensor",
        dm: "torch.Tensor",
        dx: "torch.Tensor",
        v: "torch.Tensor",
        theta: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Encode a particle set into variational parameters.

        Pools dynamics features **per component** (labels only stratify the
        pool). Optional light attention on a subsample adds morphology cues.
        """
        del dm  # Morton Δm unused in the set encoder
        cfg = self.config
        h = self.enc_mlp(_phase_features(dx, v))
        th = self.theta_mlp(theta)
        c = c.long().clamp(0, 2)
        pools = [_masked_mean_max(h, c == k) for k in range(3)]
        fused = self.enc_fuse(torch.cat(pools + [th], dim=-1))
        if self.encoder is not None:
            n = h.shape[1]
            k = min(int(cfg.enc_attn_n), n)
            if k < n:
                idx = torch.randperm(n, device=h.device)[:k]
                attn_pool = self.encoder(h[:, idx] + th.unsqueeze(1)).mean(dim=1)
            else:
                attn_pool = self.encoder(h + th.unsqueeze(1)).mean(dim=1)
            fused = fused + 0.5 * self.attn_fuse(attn_pool)
        return self.to_mu(fused), self.to_logvar(fused)

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        """Draw ``z = μ + σ ⊙ ε`` with ``ε ~ N(0, I)``."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _geometric_particles(
        self,
        c_ids: "torch.Tensor",
        eps: "torch.Tensor",
        theta: "torch.Tensor",
        cond: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Component-conditional geometric base + small residual.

        Disk: exponential / sech-like cylinder from ``θ`` scale lengths.
        Halo / bulge: isotropic exponential spheres with ``θ`` scale radii.
        """
        if theta.shape[-1] < 9:
            pad = theta.new_zeros(theta.shape[0], 10)
            pad[:, : theta.shape[-1]] = theta
            pad[:, 1] = pad[:, 1].clamp(min=0.5) + (pad[:, 1] == 0).float() * 2.5
            pad[:, 2] = pad[:, 2].clamp(min=0.05) + (pad[:, 2] == 0).float() * 0.3
            pad[:, 6] = pad[:, 6].clamp(min=2.0) + (pad[:, 6] == 0).float() * 25.0
            pad[:, 8] = pad[:, 8].clamp(min=0.2) + (pad[:, 8] == 0).float() * 0.5
            theta = pad
        rd = theta[:, 1].clamp(min=0.5).unsqueeze(1)
        zd = theta[:, 2].clamp(min=0.05).unsqueeze(1)
        ah = theta[:, 6].clamp(min=2.0).unsqueeze(1)
        ab = theta[:, 8].clamp(min=0.2).unsqueeze(1)
        smult = torch.exp(0.6 * torch.tanh(self.scale_head(cond)))
        rd = rd * smult[:, 0:1]
        zd = zd * smult[:, 1:2]
        ah = ah * smult[:, 2:3]
        ab = ab * smult[:, 3:4]
        b, n, _ = eps.shape
        u = torch.rand(b, n, 3, device=eps.device, dtype=eps.dtype).clamp(1e-4, 1.0 - 1e-4)
        direction = nn.functional.normalize(
            torch.randn(b, n, 3, device=eps.device, dtype=eps.dtype), dim=-1
        )

        R = -rd * torch.log(u[..., 0])
        phi = 2.0 * torch.pi * u[..., 1]
        uu = u[..., 2]
        z_d = zd * torch.sign(2.0 * uu - 1.0) * (
            -torch.log(1.0 - (2.0 * uu - 1.0).abs().clamp(max=0.999))
        )
        bar = self.bar_head(cond)
        t_gyr = theta[:, 9].clamp(min=0.0) if theta.shape[-1] > 9 else theta.new_zeros(theta.shape[0])
        time_gate = torch.tanh(t_gyr / 0.5)
        eps_bar = (
            0.06 * time_gate + 0.42 * torch.tanh(bar[:, 0]) * (0.12 + 0.88 * time_gate)
        ).view(-1, 1)
        phase = bar[:, 1].view(-1, 1)
        ux, uy = torch.cos(phase), torch.sin(phase)
        rx, ry = R * torch.cos(phi), R * torch.sin(phi)
        par = rx * ux + ry * uy
        perp = -rx * uy + ry * ux
        par = par * (1.0 + eps_bar)
        perp = perp * (1.0 - eps_bar)
        x_disk = torch.stack([par * ux - perp * uy, par * uy + perp * ux, z_d], dim=-1)
        v_phi = (2.0 / (1.0 + R / rd)).clamp(0.1, 3.0)
        v_disk = torch.stack([-v_phi * torch.sin(phi), v_phi * torch.cos(phi), torch.zeros_like(R)], dim=-1)

        r_h = -ah * torch.log(u[..., 0])
        r_b = -ab * torch.log(u[..., 0])
        x_halo = r_h.unsqueeze(-1) * direction
        x_bulge = r_b.unsqueeze(-1) * direction
        v_iso = 0.3 * torch.randn(b, n, 3, device=eps.device, dtype=eps.dtype)

        c = c_ids.unsqueeze(-1)
        x = torch.where(c == 0, x_disk, torch.where(c == 1, x_halo, x_bulge))
        v = torch.where(c == 0, v_disk, v_iso)

        h = self.noise_proj(eps) + cond.unsqueeze(1) + self.c_embed(c_ids)
        resid = self.resid_mlp(h)
        x = x + self.resid_scale * resid[..., :3]
        v = v + self.v_scale * resid[..., 3:6]
        return x, v

    def decode(
        self,
        z: "torch.Tensor",
        theta: "torch.Tensor",
        n: int | None = None,
        *,
        eps: "torch.Tensor | None" = None,
        c_ids: "torch.Tensor | None" = None,
        slot_start: int = 0,
        slot_total: int | None = None,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Sample ``c`` from the ``θ`` mix prior (if needed), then decode ``x, v | c``.

        ``slot_*`` kept for API compatibility (unordered set decode).
        """
        del slot_start, slot_total
        cfg = self.config
        n = int(n or cfg.n_particles)
        b = z.shape[0]
        th = self.theta_mlp(theta)
        cond = self.cond_proj(torch.cat([z, th], dim=-1))
        mix_logits = self.mix_prior(th)
        if eps is None:
            eps = torch.randn(b, n, cfg.latent_dim, device=z.device, dtype=z.dtype)
        if c_ids is None:
            c_ids = stratified_component_ids(mix_logits, n)
        else:
            c_ids = c_ids.long().clamp(0, 2)
        dx, v = self._geometric_particles(c_ids, eps, theta, cond)
        dm = torch.zeros(b, n, device=z.device, dtype=z.dtype)
        return mix_logits, c_ids, dm, dx, v

    def forward(
        self,
        c: "torch.Tensor",
        dm: "torch.Tensor",
        dx: "torch.Tensor",
        v: "torch.Tensor",
        theta: "torch.Tensor",
    ) -> dict[str, "torch.Tensor"]:
        """
        Encode → reparameterize → decode.

        Teacher-forces ``c`` into the geometric decoder so ``x, v`` learn
        conditionally; identity is supervised by classifying phase space, not
        by a global mix CE.
        """
        mu, logvar = self.encode(c, dm, dx, v, theta)
        z = self.reparameterize(mu, logvar)
        mix_logits, c_ids, dm_hat, dx_hat, v_hat = self.decode(
            z, theta, n=c.shape[1], c_ids=c
        )
        # Classify both data and reconstruction — CE on data is the easy signal
        logits_data = self.classify_particles(dx, v)
        logits_pred = self.classify_particles(dx_hat, v_hat)
        return {
            "logits_c": logits_pred,
            "logits_c_data": logits_data,
            "mix_logits": mix_logits,
            "c_ids": c_ids,
            "dm": dm_hat,
            "dx": dx_hat,
            "v": v_hat,
            "mu": mu,
            "logvar": logvar,
        }

    def loss(self, batch: dict, outputs: dict) -> dict[str, "torch.Tensor"]:
        """β-VAE + per-particle CE + easy mix MSE + Chamfer + profiles + virial."""
        cfg = self.config
        c = batch["c"].long().clamp(0, 2)

        # --- Component identity: hard per-particle CE from phase space ---
        logits_data = outputs["logits_c_data"]
        ce = nn.functional.cross_entropy(
            logits_data.reshape(-1, 3),
            c.reshape(-1),
        )
        # Consistency: reconstructed particles (teacher-forced c) should also classify
        ce_pred = nn.functional.cross_entropy(
            outputs["logits_c"].reshape(-1, 3),
            c.reshape(-1),
        )
        ce = ce + 0.5 * ce_pred
        with torch.no_grad():
            pred_cls = logits_data.argmax(dim=-1)
            ce_acc = (pred_cls == c).float().mean()

        # --- Easy mix prior: fraction MSE only (no soft global CE) ---
        frac_tgt = torch.nn.functional.one_hot(c, num_classes=3).float().mean(dim=1)
        frac_pred = torch.softmax(outputs["mix_logits"], dim=-1)
        mix_mse = nn.functional.mse_loss(frac_pred, frac_tgt)

        # --- Permutation-invariant recon ---
        mse_dm = nn.functional.mse_loss(_norm_dm(outputs["dm"]), _norm_dm(batch["dm"]))
        mse_dx = nn.functional.mse_loss(_norm_pos(outputs["dx"]), _norm_pos(batch["dx"]))
        mse_v = nn.functional.mse_loss(outputs["v"], batch["v"])
        max_n = int(cfg.chamfer_max_n)
        chamfer = chamfer_l2_subsampled(outputs["dx"], batch["dx"], max_n=max_n)
        chamfer = chamfer + 0.25 * chamfer_l2_subsampled(outputs["v"], batch["v"], max_n=max_n)
        comp_ch = outputs["dx"].new_zeros(())
        n_comp = 0
        for k in range(3):
            parts = []
            for bi in range(c.shape[0]):
                m = c[bi] == k
                nk = int(m.sum())
                if nk < 2:
                    continue
                a = outputs["dx"][bi, m]
                t = batch["dx"][bi, m]
                if nk > max_n // 3:
                    idx = torch.randperm(nk, device=a.device)[: max_n // 3]
                    a, t = a[idx], t[idx]
                d = torch.cdist(a.unsqueeze(0), t.unsqueeze(0))
                parts.append(d.min(-1).values.mean() + d.min(-2).values.mean())
            if parts:
                comp_ch = comp_ch + torch.stack(parts).mean()
                n_comp += 1
        if n_comp:
            comp_ch = comp_ch / n_comp
        chamfer = chamfer + 0.5 * comp_ch
        index_term = mse_dm + mse_dx + mse_v
        recon = float(cfg.lambda_chamfer) * chamfer + float(cfg.lambda_index) * index_term

        mu, logvar = outputs["mu"], outputs["logvar"]
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.clamp(-20.0, 20.0).exp())
        # Profile soft-weights from phase-space classifier on the reconstruction
        prof = profile_reconstruction_loss(
            batch["dx"],
            batch["v"],
            batch["c"],
            outputs["dx"],
            outputs["v"],
            outputs["logits_c"],
            n_bins=cfg.profile_n_bins,
            r_max_disk=cfg.profile_r_max_disk,
            r_max_sph=cfg.profile_r_max_sph,
            lambda_sigma=cfg.lambda_sigma,
            lambda_rho=cfg.lambda_rho,
            lambda_vphi=cfg.lambda_vphi,
            lambda_nonaxisym=cfg.lambda_nonaxisym,
            lambda_maps=cfg.lambda_maps,
            fourier_modes=tuple(cfg.fourier_modes),
            fourier_z_max=cfg.fourier_z_max,
            map_n_pix=cfg.map_n_pix,
        )
        vir_term = outputs["dx"].new_zeros(())
        vir_metrics: dict[str, "torch.Tensor"] = {
            "virial": vir_term,
            "virial_ratio": vir_term,
            "virial_ke": vir_term,
            "virial_com": vir_term,
            "ratio_data": vir_term,
            "ratio_pred": vir_term,
        }
        if float(cfg.lambda_virial) > 0.0:
            # Equal-mass pairwise 2K/|W| on GalactICS dumps is hot (~10–20);
            # matching it teaches unbound BH clouds. Prefer soft Q≈1 + KE/COM.
            vir = virial_consistency_loss(
                batch["dx"],
                batch["v"],
                batch["c"],
                outputs["dx"],
                outputs["v"],
                n_sub=cfg.virial_n_sub,
                eps=cfg.virial_eps,
                lambda_ratio=0.15,
                lambda_ke=1.0,
                lambda_com=1.0,
                lambda_target=float(cfg.lambda_virial_target),
                target_ratio=float(cfg.virial_target_ratio),
            )
            vir_term = vir["virial"]
            vir_metrics = {
                "virial": vir["virial"],
                "virial_ratio": vir["virial_ratio"],
                "virial_ke": vir["ke"],
                "virial_com": vir["com"],
                "ratio_data": vir["ratio_data"],
                "ratio_pred": vir["ratio_pred"],
            }
        total = (
            cfg.lambda_recon * recon
            + float(cfg.lambda_ce) * ce
            + float(cfg.lambda_mix) * mix_mse
            + cfg.beta * kl
            + prof["profile"]
            + float(cfg.lambda_virial) * vir_term
        )
        out = {
            "loss": total,
            "recon": recon,
            "mse_dm": mse_dm,
            "mse_dx": mse_dx,
            "mse_v": mse_v,
            "chamfer": chamfer,
            "ce": ce,
            "ce_acc": ce_acc,
            "mix_mse": mix_mse,
            "kl": kl,
            "profile": prof["profile"],
            "sigma": prof["sigma"],
            "rho": prof["rho"],
            "vphi": prof["vphi"],
            "nonaxisym": prof["nonaxisym"],
            "maps": prof["maps"],
            "map_xy": prof["map_xy"],
            "map_xz": prof["map_xz"],
            **vir_metrics,
        }
        for key, val in prof.items():
            if key.startswith("am"):
                out[key] = val
        return out

    @torch.no_grad()
    def generate(
        self,
        theta: "torch.Tensor",
        n: int | None = None,
        *,
        z: "torch.Tensor | None" = None,
        chunk_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Draw a particle set from ``p(particles | θ)``.

        Samples stratified ``c`` from the ``θ`` mix prior, then decodes
        ``x, v | c, θ, z``.
        """
        self.eval()
        cfg = self.config
        n = int(n or cfg.n_particles)
        chunk = int(chunk_size or cfg.n_particles)
        b = theta.shape[0]
        if z is None:
            z = torch.randn(b, cfg.latent_dim, device=theta.device, dtype=theta.dtype)

        # One mix draw for the whole set so chunked decode keeps global fractions
        mix_logits = self.mix_logits_from_theta(theta)
        c_all = stratified_component_ids(mix_logits, n)

        c_parts: list[torch.Tensor] = []
        dm_parts: list[torch.Tensor] = []
        dx_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        offset = 0
        remaining = n
        while remaining > 0:
            take = min(chunk, remaining)
            c_chunk = c_all[:, offset : offset + take]
            _mix, c_ids, dm, dx, v = self.decode(
                z, theta, n=take, c_ids=c_chunk, slot_start=offset, slot_total=n
            )
            c_parts.append(c_ids)
            dm_parts.append(dm)
            dx_parts.append(dx)
            v_parts.append(v)
            offset += take
            remaining -= take

        return {
            "c": torch.cat(c_parts, dim=1).cpu().numpy(),
            "dm": torch.cat(dm_parts, dim=1).cpu().numpy(),
            "dx": torch.cat(dx_parts, dim=1).cpu().numpy(),
            "v": torch.cat(v_parts, dim=1).cpu().numpy(),
        }
