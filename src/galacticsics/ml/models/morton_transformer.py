"""Conditional autoregressive transformer over Morton particle tokens.

This model is the **sequential** baseline complementary to the parallel set
VAE (:class:`~galacticsics.ml.models.sequence_vae.SequenceVAE`).  It factorises

``p(z_1,…,z_N | θ) = ∏_i p(z_i | z_<i, θ)``

over Morton-ordered (or randomly ordered) tokens ``z_i = (c, Δm, x, v)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("galacticsics[ml] / torch required for MortonTransformer") from exc


@dataclass
class MortonTransformerConfig:
    """
    Hyperparameters for :class:`MortonTransformer`.

    Attributes
    ----------
    n_particles : int
        Maximum sequence length (positional embedding table size).
    theta_dim : int
        Conditioning vector dimension.
    d_model, n_layers, n_heads, dropout
        Standard transformer width / depth / regularisation.
    """

    n_particles: int = 512
    theta_dim: int = 10
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1


def _causal_mask(n: int, device) -> "torch.Tensor":
    """Boolean upper-triangular mask (``True`` = blocked) for causal attention."""
    return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)


class MortonTransformer(nn.Module):
    """
    Autoregressive Morton-token language model conditioned on galaxy params θ.

    Parameters
    ----------
    config : MortonTransformerConfig, optional
        Architecture hyperparameters.

    Notes
    -----
    Training uses teacher forcing: the network sees ``[θ_BOS, z_0, …, z_{N-2}]``
    and predicts ``(z_0, …, z_{N-1})``.  Generation is strictly left-to-right
    and therefore sequential in wall time (unlike the set VAE decoder).

    Loss
    ----
    ``L = MSE(Δm, x, v) + CE(c)`` with equal MSE weight on each continuous head.
    """

    def __init__(self, config: MortonTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or MortonTransformerConfig()
        cfg = self.config
        self.theta_mlp = nn.Sequential(
            nn.Linear(cfg.theta_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.token_emb = nn.Linear(10, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.n_particles + 1, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head_c = nn.Linear(cfg.d_model, 3)
        self.head_cont = nn.Linear(cfg.d_model, 7)

    def _embed_tokens(self, c, dm, dx, v) -> "torch.Tensor":
        """Embed ``(c, log1p(Δm), x, v)`` tokens to ``d_model``."""
        c_oh = torch.nn.functional.one_hot(c.long().clamp(0, 2), num_classes=3).float()
        dm_n = torch.log1p(dm.clamp_min(0.0))
        feats = torch.cat([c_oh, dm_n.unsqueeze(-1), dx, v], dim=-1)
        return self.token_emb(feats)


    def forward(self, c, dm, dx, v, theta):
        """
        Teacher-forced next-token prediction.

        Parameters
        ----------
        c : Tensor, shape (B, N)
        dm : Tensor, shape (B, N)
        dx : Tensor, shape (B, N, 3)
        v : Tensor, shape (B, N, 3)
        theta : Tensor, shape (B, theta_dim)

        Returns
        -------
        dict
            ``logits_c`` (B, N, 3), ``dm``/``dx``/``v`` predictions aligned to
            target tokens ``0…N-1``.
        """
        b, n, _ = dx.shape
        th = self.theta_mlp(theta).unsqueeze(1)  # (B,1,D)
        tok = self._embed_tokens(c, dm, dx, v)
        seq = torch.cat([th, tok[:, :-1, :]], dim=1)
        pos = torch.arange(n, device=seq.device).unsqueeze(0).expand(b, -1)
        h = self.blocks(seq + self.pos_emb(pos), mask=_causal_mask(n, seq.device))
        logits_c = self.head_c(h)
        cont = self.head_cont(h)
        return {
            "logits_c": logits_c,
            "dm": cont[..., 0],
            "dx": cont[..., 1:4],
            "v": cont[..., 4:7],
        }

    def loss(self, batch: dict, outputs: dict) -> dict[str, "torch.Tensor"]:
        """
        Autoregressive training objective.

        Parameters
        ----------
        batch : dict
            Target tokens ``c``, ``dm``, ``dx``, ``v``.
        outputs : dict
            Predictions from :meth:`forward`.

        Returns
        -------
        dict
            ``loss``, ``recon`` (sum of MSEs), ``ce``.
        """
        ce = nn.functional.cross_entropy(
            outputs["logits_c"].reshape(-1, 3),
            batch["c"].long().reshape(-1),
        )
        # log1p(Δm): raw Morton increments are O(10^5) and drown x,v MSE
        dm_t = torch.log1p(batch["dm"].clamp_min(0.0))
        dm_p = torch.log1p(outputs["dm"].clamp_min(0.0))
        recon = (
            nn.functional.mse_loss(dm_p, dm_t)
            + nn.functional.mse_loss(outputs["dx"], batch["dx"])
            + nn.functional.mse_loss(outputs["v"], batch["v"])
        )
        return {"loss": recon + ce, "recon": recon, "ce": ce}

    @torch.no_grad()
    def generate(self, theta: "torch.Tensor", n: int | None = None) -> dict[str, np.ndarray]:
        """
        Ancestral sample of an ``n``-token Morton sequence.

        Parameters
        ----------
        theta : Tensor, shape (B, theta_dim)
        n : int, optional
            Sequence length (default ``config.n_particles``).

        Returns
        -------
        dict of ndarray
            ``c``, ``dm``, ``dx``, ``v``.
        """
        self.eval()
        cfg = self.config
        n = n or cfg.n_particles
        b = theta.shape[0]
        device = theta.device
        th = self.theta_mlp(theta).unsqueeze(1)
        c = torch.zeros(b, n, dtype=torch.long, device=device)
        dm = torch.zeros(b, n, device=device)
        dx = torch.zeros(b, n, 3, device=device)
        v = torch.zeros(b, n, 3, device=device)
        for i in range(n):
            if i == 0:
                seq = th
            else:
                tok = self._embed_tokens(c[:, :i], dm[:, :i], dx[:, :i], v[:, :i])
                seq = torch.cat([th, tok], dim=1)
            pos = torch.arange(seq.shape[1], device=device).unsqueeze(0).expand(b, -1)
            pos = pos.clamp(max=cfg.n_particles)
            h = self.blocks(seq + self.pos_emb(pos), mask=_causal_mask(seq.shape[1], device))
            h_i = h[:, -1, :]
            logits = self.head_c(h_i)
            cont = self.head_cont(h_i)
            c[:, i] = logits.argmax(dim=-1)
            dm[:, i] = cont[:, 0].clamp(min=0.0)
            dx[:, i] = cont[:, 1:4]
            v[:, i] = cont[:, 4:7]
        return {
            "c": c.cpu().numpy(),
            "dm": dm.cpu().numpy(),
            "dx": dx.cpu().numpy(),
            "v": v.cpu().numpy(),
        }
