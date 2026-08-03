"""Frozen-teacher feature library for latent-controlled IC sampling.

**Teacher** = the frozen crisp multi-tower field autoencoder (encoder
bottleneck+skips + decoder), not a human or LLM. **Students** are models that
try to synthesize those skips from a compact ``z``.

Bars live in U-Net skips.  End-to-end ``decode(z)`` washes them; this path keeps
teacher bottleneck+skips in a library, compresses pooled bottlenecks to a PCA
``z``, and samples by retrieving / morphing library members.

Recommended methods (measured 2026-07-25 marathon, stratified library):

- ``amplify_residual`` / ``amplify_knn_hybrid`` — bar A₂ ≈ **0.39–0.42**, quiet ≈ 0.02
- ``sample_features_z_amplify`` — continuous ``z`` → bar A₂ ≈ **0.43±0.02**, quiet ≈ 0.02
- ``strong_knn`` / ``uniform_knn`` — bar A₂ ≈ 0.29–0.32 (weaker but safe)
- ``exact_a2_weighted`` / ``topk_exact`` — ceiling / retrieval checks
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from galacticsics.ml.fields.binning import MultiScaleSliceConfig, bin_multiscale_slice_stacks
from galacticsics.ml.fields.frame import prepare_shared_frame
from galacticsics.ml.fields.latent_code import load_frozen_teacher
from galacticsics.ml.fields.normalize import FieldNormStats, normalize_stack
from galacticsics.ml.morton.polygon import _component_ids

SampleKind = Literal["barred", "quiet"]
SampleMethod = Literal[
    "uniform_knn",
    "a2_weighted_knn",
    "exact_a2_weighted",
    "amplify_residual",
    "local_pca",
    "kde_retrieve",
    "hier_morph",
    "strong_knn",
    "amplify_knn_hybrid",
    "topk_exact",
    "disk_only_amplify",
]


def _as_stats(raw: dict) -> dict[str, FieldNormStats]:
    return {
        k: (v if isinstance(v, FieldNormStats) else FieldNormStats(**v))
        for k, v in raw.items()
    }


def pool_bottleneck_z(features: dict, enc_grid: int = 4) -> torch.Tensor:
    flats = []
    for name in sorted(features.keys()):
        b = features[name]["bottleneck"]
        spat = torch.nn.functional.adaptive_avg_pool2d(b, (enc_grid, enc_grid))
        flats.append(spat.flatten(1))
    return torch.cat(flats, dim=-1)


IndexMethod = Literal["pca", "lda_concat", "whiten_pca", "pls_a2", "multiclass_lda"]


def fit_index_basis(
    Z: np.ndarray,
    a2: np.ndarray,
    *,
    method: IndexMethod = "pca",
    n_comp: int = 64,
    bar_floor: float = 0.20,
    quiet_ceil: float = 0.06,
    n_a2_bins: int = 4,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a linear index ``codes = (Z - mean) @ W`` for library retrieval.

    Methods
    -------
    pca
        Unsupervised SVD variance axes (current production path).
    lda_concat
        Fisher bar/quiet direction prepended to a reduced PCA basis.
    whiten_pca
        Within-class (bar∪quiet) whitening, then PCA — class-conditional metric
        without collapsing to 1-D LDA.
    pls_a2
        1-target PLS / supervised PCA vs continuous A₂ (NIPALS), padded with
        residual PCA for remaining dims.
    multiclass_lda
        LDA on A₂ quantile strata (better than binary for morph diversity),
        padded with residual PCA.
    """
    Z = np.asarray(Z, dtype=np.float64)
    a2 = np.asarray(a2, dtype=np.float64).ravel()
    n, d = Z.shape
    n_comp = int(max(1, min(n_comp, d, max(n - 1, 1))))
    mean = Z.mean(0)
    Zc = Z - mean
    meta: dict[str, Any] = {"method": method, "n_comp": n_comp}

    def _pca_w(Xc: np.ndarray, k: int) -> np.ndarray:
        if Xc.shape[0] < 2 or k <= 0:
            return np.zeros((Xc.shape[1], 0), dtype=np.float64)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        return vt[:k].T

    def _ortho_pad(W: np.ndarray, Xc: np.ndarray, k: int) -> np.ndarray:
        """Pad columns of W with PCA of residual to reach k dims."""
        if W.size == 0:
            return _pca_w(Xc, k)
        # Orthonormalize existing columns.
        q, _ = np.linalg.qr(W, mode="reduced")
        proj = Xc @ q @ q.T
        resid = Xc - proj
        need = max(0, k - q.shape[1])
        if need <= 0:
            return q[:, :k]
        Wp = _pca_w(resid, need)
        if Wp.size == 0:
            return q[:, :k]
        return np.concatenate([q, Wp], axis=1)[:, :k]

    bar = a2 >= float(bar_floor)
    quiet = a2 <= float(quiet_ceil)
    meta["n_bar"] = int(bar.sum())
    meta["n_quiet"] = int(quiet.sum())

    if method == "pca":
        W = _pca_w(Zc, n_comp)
        return mean, W, Zc @ W, meta

    if method == "whiten_pca":
        mask = bar | quiet
        if int(mask.sum()) < 4:
            W = _pca_w(Zc, n_comp)
            meta["fallback"] = "pca_insufficient_labels"
            return mean, W, Zc @ W, meta
        # Pooled within-class covariance (each class centered on its own mean).
        chunks = []
        for m in (bar, quiet):
            if int(m.sum()) < 2:
                continue
            X = Zc[m]
            chunks.append(X - X.mean(0, keepdims=True))
        Xw = np.concatenate(chunks, axis=0) if chunks else Zc[mask]
        cov = (Xw.T @ Xw) / max(Xw.shape[0] - 1, 1)
        cov = cov + float(ridge) * np.eye(d)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 1e-8)
        # Whitening: Σ^{-1/2}
        white = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        Zw = Zc @ white
        Wp = _pca_w(Zw, n_comp)
        W = white @ Wp  # so (Z-mean)@W = Zw @ Wp
        meta["ridge"] = float(ridge)
        return mean, W, Zc @ W, meta

    if method == "lda_concat":
        if int(bar.sum()) < 2 or int(quiet.sum()) < 2:
            W = _pca_w(Zc, n_comp)
            meta["fallback"] = "pca_insufficient_labels"
            return mean, W, Zc @ W, meta
        mu_b = Zc[bar].mean(0)
        mu_q = Zc[quiet].mean(0)
        chunks = []
        for m in (bar, quiet):
            X = Zc[m]
            chunks.append(X - X.mean(0, keepdims=True))
        Xw = np.concatenate(chunks, axis=0)
        sw = (Xw.T @ Xw) / max(Xw.shape[0] - 1, 1) + float(ridge) * np.eye(d)
        w = np.linalg.solve(sw, mu_b - mu_q)
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-12:
            W = _pca_w(Zc, n_comp)
            meta["fallback"] = "pca_degenerate_lda"
            return mean, W, Zc @ W, meta
        w = (w / nrm).reshape(-1, 1)
        W = _ortho_pad(w, Zc, n_comp)
        meta["ridge"] = float(ridge)
        meta["fisher_sep"] = float(np.abs((mu_b - mu_q) @ w.ravel()))
        return mean, W, Zc @ W, meta

    if method == "multiclass_lda":
        # Quantile strata on continuous A₂ (drop empty edges).
        qs = np.linspace(0.0, 1.0, int(n_a2_bins) + 1)
        edges = np.unique(np.quantile(a2, qs))
        if edges.size < 3:
            W = _pca_w(Zc, n_comp)
            meta["fallback"] = "pca_insufficient_strata"
            return mean, W, Zc @ W, meta
        labels = np.digitize(a2, edges[1:-1], right=True)
        classes = np.unique(labels)
        if classes.size < 2:
            W = _pca_w(Zc, n_comp)
            meta["fallback"] = "pca_single_class"
            return mean, W, Zc @ W, meta
        mu = Zc.mean(0)
        sb = np.zeros((d, d), dtype=np.float64)
        sw = np.zeros((d, d), dtype=np.float64)
        for c in classes:
            X = Zc[labels == c]
            if X.shape[0] < 2:
                continue
            mc = X.mean(0)
            diff = (mc - mu).reshape(-1, 1)
            sb += X.shape[0] * (diff @ diff.T)
            Xc = X - mc
            sw += Xc.T @ Xc
        sw = sw + float(ridge) * np.eye(d)
        # Solve Sb v = λ Sw v via whitened eigenproblem.
        try:
            sw_inv_sqrt = np.linalg.inv(np.linalg.cholesky(sw))
            m = sw_inv_sqrt @ sb @ sw_inv_sqrt.T
            evals, evecs = np.linalg.eigh(m)
            order = np.argsort(evals)[::-1]
            n_lda = min(classes.size - 1, n_comp, int(np.sum(evals[order] > 1e-8)))
            V = sw_inv_sqrt.T @ evecs[:, order[:n_lda]]
        except np.linalg.LinAlgError:
            W = _pca_w(Zc, n_comp)
            meta["fallback"] = "pca_lda_linalg"
            return mean, W, Zc @ W, meta
        W = _ortho_pad(V, Zc, n_comp)
        meta["n_a2_bins"] = int(n_a2_bins)
        meta["n_lda"] = int(V.shape[1])
        meta["ridge"] = float(ridge)
        return mean, W, Zc @ W, meta

    if method == "pls_a2":
        # One-target NIPALS PLS; pad with residual PCA for remaining dims.
        y = a2.copy()
        y = y - y.mean()
        ys = float(np.std(y))
        if ys < 1e-12:
            W = _pca_w(Zc, n_comp)
            meta["fallback"] = "pca_constant_a2"
            return mean, W, Zc @ W, meta
        y = y / ys
        X = Zc.copy()
        n_pls = min(2, n_comp)  # A₂ is 1-D; 1–2 components capture supervised signal
        W_pls = []
        for _ in range(n_pls):
            w = X.T @ y
            nrm = float(np.linalg.norm(w))
            if nrm < 1e-12:
                break
            w = w / nrm
            t = X @ w
            tt = float(t @ t) + 1e-30
            p = (X.T @ t) / tt
            q = float(y @ t) / tt
            X = X - np.outer(t, p)
            y = y - q * t
            W_pls.append(w)
        if not W_pls:
            W = _pca_w(Zc, n_comp)
            meta["fallback"] = "pca_pls_failed"
            return mean, W, Zc @ W, meta
        W0 = np.stack(W_pls, axis=1)
        W = _ortho_pad(W0, Zc, n_comp)
        meta["n_pls"] = int(W0.shape[1])
        return mean, W, Zc @ W, meta

    raise ValueError(f"unknown index method {method!r}")


def blend_features(members: list[dict], weights: np.ndarray) -> dict:
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-30)
    out: dict = {}
    for name in members[0]:
        bn = sum(float(wi) * m[name]["bottleneck"] for wi, m in zip(w, members))
        n_sk = len(members[0][name]["skips"])
        skips = tuple(
            sum(float(wi) * m[name]["skips"][si] for wi, m in zip(w, members))
            for si in range(n_sk)
        )
        out[name] = {"bottleneck": bn, "skips": skips}
    return out


def scale_residual(base: dict, target: dict, alpha: float) -> dict:
    """``base + α (target − base)``; α>1 amplifies non-axisym residual."""
    out = {}
    for name in base:
        bb, bt = base[name]["bottleneck"], target[name]["bottleneck"]
        sb, st = base[name]["skips"], target[name]["skips"]
        out[name] = {
            "bottleneck": bb + float(alpha) * (bt - bb),
            "skips": tuple(b + float(alpha) * (t - b) for b, t in zip(sb, st)),
        }
    return out


def _hash_from_path(path: str | Path) -> str:
    """Extract ``mw_morton_corpus_v2/<hash>`` campaign id from a dump path."""
    parts = Path(str(path)).parts
    if "mw_morton_corpus_v2" in parts:
        i = parts.index("mw_morton_corpus_v2")
        if i + 1 < len(parts):
            return str(parts[i + 1])
    return ""


def _softmax_neg(d: np.ndarray, temp: float) -> np.ndarray:
    x = -d / max(temp, 1e-8)
    x = x - x.max()
    e = np.exp(x)
    return e / max(float(e.sum()), 1e-30)


@dataclass
class FeatureLibraryConfig:
    enc_grid: int = 4
    bar_floor: float = 0.20
    quiet_ceil: float = 0.06
    n_pc: int = 64


class TeacherFeatureLibrary:
    """In-memory teacher feature library + PCA ``z`` sampling API."""

    def __init__(
        self,
        *,
        teacher: torch.nn.Module,
        cfg: MultiScaleSliceConfig,
        stats: dict[str, FieldNormStats],
        feats: list[dict],
        codes: np.ndarray,
        a2: np.ndarray,
        meta: list[dict],
        pca_mean: np.ndarray,
        pca_w: np.ndarray,
        lib_cfg: FeatureLibraryConfig | None = None,
        theta: np.ndarray | None = None,
    ) -> None:
        self.teacher = teacher
        self.cfg = cfg
        self.stats = stats
        self.feats = feats
        self.codes = np.asarray(codes, dtype=np.float64)
        self.a2 = np.asarray(a2, dtype=np.float64)
        self.meta = meta
        self.pca_mean = np.asarray(pca_mean, dtype=np.float64)
        self.pca_w = np.asarray(pca_w, dtype=np.float64)
        self.lib_cfg = lib_cfg or FeatureLibraryConfig()
        self.theta = (
            np.asarray(theta, dtype=np.float64)
            if theta is not None
            else np.zeros((len(self.codes), 0), dtype=np.float64)
        )
        self.hashes = np.asarray(
            [str(m.get("run_hash", "") or _hash_from_path(m.get("path", ""))) for m in self.meta],
            dtype=object,
        )
        self.bar_pool = np.where(self.a2 >= self.lib_cfg.bar_floor)[0]
        self.quiet_pool = np.where(self.a2 <= self.lib_cfg.quiet_ceil)[0]
        if self.bar_pool.size == 0:
            self.bar_pool = np.where(self.a2 >= float(np.quantile(self.a2, 0.75)))[0]
        if self.quiet_pool.size == 0:
            self.quiet_pool = np.where(self.a2 <= float(np.quantile(self.a2, 0.25)))[0]
        self._quiet_mean = (
            blend_features([self.feats[i] for i in self.quiet_pool], np.ones(len(self.quiet_pool)))
            if self.quiet_pool.size
            else None
        )

    @property
    def latent_dim(self) -> int:
        return int(self.codes.shape[1])

    def pool_to_z(self, features: dict) -> np.ndarray:
        z_raw = pool_bottleneck_z(features, enc_grid=self.lib_cfg.enc_grid)[0].detach().cpu().numpy()
        return (z_raw - self.pca_mean) @ self.pca_w

    def decode_features(self, feat: dict) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.teacher.decode_features(
                feat,
                target_shapes={g.name: (g.n_pix, g.n_pix) for g in self.cfg.grids},
                n_channels={g.name: g.n_moment_channels for g in self.cfg.grids},
            )

    def filter_pool(
        self,
        pool: np.ndarray | None = None,
        *,
        exclude_hashes: set[str] | frozenset[str] | None = None,
        exclude_paths: set[str] | frozenset[str] | None = None,
        exclude_indices: set[int] | frozenset[int] | None = None,
    ) -> np.ndarray:
        """Return pool indices with optional leave-one-out / held-out exclusions."""
        idx = np.arange(len(self.codes)) if pool is None else np.asarray(pool, dtype=int)
        keep = np.ones(len(idx), dtype=bool)
        if exclude_hashes:
            keep &= np.array([h not in exclude_hashes for h in self.hashes[idx]], dtype=bool)
        if exclude_paths:
            paths = [str(self.meta[i].get("path", "")) for i in idx]
            keep &= np.array([p not in exclude_paths for p in paths], dtype=bool)
        if exclude_indices:
            keep &= np.array([int(i) not in exclude_indices for i in idx], dtype=bool)
        out = idx[keep]
        if out.size == 0:
            raise ValueError("filter_pool emptied the candidate set")
        return out

    def pool_near_theta(
        self,
        theta: np.ndarray,
        *,
        k: int = 24,
        exclude_hashes: set[str] | frozenset[str] | None = None,
        pool: np.ndarray | None = None,
        structural_only: bool = True,
    ) -> np.ndarray:
        """Nearest library members in structural θ (drops ``t_gyr`` by default)."""
        if self.theta.size == 0 or self.theta.shape[1] == 0:
            base = np.arange(len(self.codes)) if pool is None else np.asarray(pool, dtype=int)
            return self.filter_pool(base, exclude_hashes=exclude_hashes)
        th = np.asarray(theta, dtype=np.float64).reshape(-1)
        T = self.theta
        if structural_only and T.shape[1] >= 10:
            # DEFAULT_THETA_KEYS ends with t_gyr — drop last coord for equilibrium family.
            T = T[:, :-1]
            th = th[: T.shape[1]]
        n = min(len(th), T.shape[1])
        d = np.linalg.norm(T[:, :n] - th[:n].reshape(1, -1), axis=1)
        base = np.arange(len(self.codes)) if pool is None else np.asarray(pool, dtype=int)
        base = self.filter_pool(base, exclude_hashes=exclude_hashes)
        order = np.argsort(d[base])[: min(k, len(base))]
        return base[order]

    def encode_to_z(
        self,
        path: str | Path,
        *,
        phi: float | None = None,
    ) -> tuple[np.ndarray, dict, float]:
        """Encode a dump through the frozen teacher → PCA ``z`` + raw features."""
        feat, z_raw, a2 = encode_snapshot_features(
            path,
            teacher=self.teacher,
            cfg=self.cfg,
            stats=self.stats,
            phi=phi,
            enc_grid=self.lib_cfg.enc_grid,
        )
        z = (z_raw - self.pca_mean) @ self.pca_w
        return z, feat, float(a2)

    def retrieve_loo(
        self,
        z: np.ndarray,
        *,
        k: int = 5,
        temp: float = 6.0,
        exclude_hashes: set[str] | frozenset[str] | None = None,
        exclude_paths: set[str] | frozenset[str] | None = None,
        exclude_indices: set[int] | frozenset[int] | None = None,
        theta: np.ndarray | None = None,
        theta_k: int = 32,
        amplify: float | None = None,
    ) -> tuple[dict, dict]:
        """Leave-one-out / held-out retrieve in ``z`` (optionally θ-gated).

        Generative handle: encode → ``z`` → blend nearest *other* library features
        → decode. Never returns the excluded dump's stored features.
        """
        if theta is not None and self.theta.size and self.theta.shape[1] > 0:
            pool = self.pool_near_theta(
                theta, k=theta_k, exclude_hashes=exclude_hashes
            )
        else:
            pool = self.filter_pool(
                exclude_hashes=exclude_hashes,
                exclude_paths=exclude_paths,
                exclude_indices=exclude_indices,
            )
        pool = self.filter_pool(
            pool,
            exclude_hashes=exclude_hashes,
            exclude_paths=exclude_paths,
            exclude_indices=exclude_indices,
        )
        feat, meta = self._retrieve_from_z(
            np.asarray(z, dtype=np.float64),
            pool,
            "barred",
            np.random.default_rng(0),
            k=k,
            temp=temp,
            a2_power=0.0,  # full-system similarity: do not force high-A₂
        )
        if amplify is not None and self._quiet_mean is not None:
            feat = scale_residual(self._quiet_mean, feat, float(amplify))
            meta["amplify"] = float(amplify)
        meta["method"] = "retrieve_loo"
        meta["pool_size"] = int(len(pool))
        return feat, meta

    def sample_features_theta_z(
        self,
        theta: np.ndarray,
        *,
        kind: SampleKind = "barred",
        z: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        exclude_hashes: set[str] | frozenset[str] | None = None,
        theta_k: int = 28,
        method: SampleMethod = "amplify_knn_hybrid",
        **kwargs: Any,
    ) -> tuple[dict, dict]:
        """Sample features conditioned on structural θ then morphology kind / ``z``.

        Restricts the retrieve/amplify pool to θ-neighbors (excluding held-out
        campaigns), then runs the usual library sampler inside that pool.
        """
        rng = rng or np.random.default_rng()
        morph_pool = self.bar_pool if kind == "barred" else self.quiet_pool
        pool = self.pool_near_theta(
            theta, k=theta_k, exclude_hashes=exclude_hashes, pool=morph_pool
        )
        if z is not None:
            feat, meta = self._retrieve_from_z(
                np.asarray(z, dtype=np.float64), pool, kind, rng, **kwargs
            )
        elif method == "amplify_knn_hybrid":
            # Local strong_knn inside θ pool then amplify vs global quiet mean.
            sk = {
                k: kwargs[k]
                for k in ("strong_floor", "alpha_max", "a2_power")
                if k in kwargs
            }
            feat_knn, meta_knn = self._strong_knn(pool, rng, **sk)
            base = self._quiet_mean if self._quiet_mean is not None else self.feats[int(pool[0])]
            alpha = float(rng.uniform(kwargs.get("alpha_lo", 1.05), kwargs.get("alpha_hi", 1.30)))
            feat = scale_residual(base, feat_knn, alpha)
            meta = {"method": "theta_amplify_knn_hybrid", "alpha": alpha, **meta_knn}
        elif method == "z_amplify" or method == "amplify_residual":
            if z is None:
                # Local jitter inside θ-gated pool.
                i0 = int(rng.choice(pool))
                local_std = self.codes[pool].std(0) + 1e-3
                z = self.codes[i0] + rng.normal(0.0, 0.22, size=local_std.shape) * local_std
            feat, meta = self.sample_features_z_amplify(kind=kind, z=z, rng=rng, **kwargs)
            meta = {**meta, "theta_pool": pool.tolist(), "method": f"theta_{meta.get('method', method)}"}
            return feat, meta
        else:
            # Fall back: exact member from θ pool.
            feat, meta = self._exact(pool, rng, prefer_low_a2=(kind == "quiet"))
            meta["method"] = f"theta_{meta['method']}"
        meta["theta_pool"] = pool.tolist()
        meta["exclude_hashes"] = sorted(exclude_hashes) if exclude_hashes else []
        return feat, meta

    def interpolate_features(
        self,
        feat_a: dict,
        feat_b: dict,
        t: float,
    ) -> dict:
        """Linear feature interp ``(1-t) a + t b`` in bottleneck+skip space."""
        t = float(np.clip(t, 0.0, 1.0))
        return blend_features([feat_a, feat_b], np.array([1.0 - t, t]))

    def sample_features(
        self,
        *,
        kind: SampleKind = "barred",
        method: SampleMethod = "uniform_knn",
        z: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> tuple[dict, dict]:
        """Sample teacher features. Returns ``(features, meta)``.

        If ``z`` is given, retrieve A₂-aware neighbors of that code (ignores
        ``kind`` pool only for distance; still reweights by morphology).
        """
        rng = rng or np.random.default_rng()
        pool = self.bar_pool if kind == "barred" else self.quiet_pool
        if z is not None:
            return self._retrieve_from_z(np.asarray(z, dtype=np.float64), pool, kind, rng, **kwargs)
        if kind == "quiet" and method in (
            "amplify_residual",
            "hier_morph",
            "exact_a2_weighted",
            "amplify_knn_hybrid",
            "topk_exact",
            "strong_knn",
            "disk_only_amplify",
        ):
            return self._exact(pool, rng, prefer_low_a2=True)
        if kind == "quiet" and method in ("local_pca", "kde_retrieve", "a2_weighted_knn"):
            return self._uniform_knn(pool, rng, alpha_max=0.25)
        samplers = {
            "uniform_knn": lambda: self._uniform_knn(pool, rng, **kwargs),
            "a2_weighted_knn": lambda: self._a2_weighted_knn(pool, rng, **kwargs),
            "exact_a2_weighted": lambda: self._exact(pool, rng, prefer_low_a2=False),
            "amplify_residual": lambda: self._amplify(pool, rng, **kwargs),
            "local_pca": lambda: self._local_pca(pool, rng, **kwargs),
            "kde_retrieve": lambda: self._kde(pool, rng, **kwargs),
            "hier_morph": lambda: self._hier(pool, rng, **kwargs),
            "strong_knn": lambda: self._strong_knn(pool, rng, **kwargs),
            "amplify_knn_hybrid": lambda: self._amplify_knn_hybrid(pool, rng, **kwargs),
            "topk_exact": lambda: self._topk_exact(pool, rng, **kwargs),
            "disk_only_amplify": lambda: self._disk_only_amplify(pool, rng, **kwargs),
        }
        if method not in samplers:
            raise ValueError(f"unknown method {method}")
        return samplers[method]()

    def sample_fields(
        self,
        *,
        kind: SampleKind = "barred",
        method: SampleMethod = "uniform_knn",
        z: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        feat, meta = self.sample_features(kind=kind, method=method, z=z, rng=rng, **kwargs)
        return self.decode_features(feat), meta

    def _retrieve_from_z(
        self,
        z: np.ndarray,
        pool: np.ndarray,
        kind: SampleKind,
        rng: np.random.Generator,
        k: int = 5,
        temp: float = 6.0,
        a2_power: float = 3.5,
    ) -> tuple[dict, dict]:
        dd = np.linalg.norm(self.codes[pool] - z.reshape(-1), axis=1)
        order = np.argsort(dd)[: min(k, len(pool))]
        nn = pool[order]
        w = _softmax_neg(dd[order], temp=temp)
        if kind == "barred":
            w = w * (np.maximum(self.a2[nn], 1e-6) ** a2_power)
        else:
            w = w * (np.maximum(self.lib_cfg.quiet_ceil + 1e-3 - self.a2[nn], 1e-6) ** a2_power)
        w = w / w.sum()
        return blend_features([self.feats[i] for i in nn], w), {
            "method": "retrieve_z",
            "nn": nn.tolist(),
            "w": w.tolist(),
        }

    def _uniform_knn(
        self, pool: np.ndarray, rng: np.random.Generator, alpha_max: float = 0.35
    ) -> tuple[dict, dict]:
        i0 = int(rng.choice(pool))
        d = np.linalg.norm(self.codes - self.codes[i0], axis=1)
        d[i0] = np.inf
        mask = np.ones(len(self.codes), dtype=bool)
        mask[pool] = False
        d[mask] = np.inf
        i1 = int(np.argmin(d))
        alpha = float(rng.uniform(0.0, alpha_max))
        feat = blend_features(
            [self.feats[i0], self.feats[i1]], np.array([1.0 - alpha, alpha])
        )
        return feat, {"method": "uniform_knn", "i0": i0, "i1": i1, "alpha": alpha}

    def _a2_weighted_knn(
        self,
        pool: np.ndarray,
        rng: np.random.Generator,
        k: int = 5,
        temp: float = 8.0,
        a2_power: float = 2.0,
    ) -> tuple[dict, dict]:
        i0 = int(rng.choice(pool))
        d = np.linalg.norm(self.codes[pool] - self.codes[i0], axis=1)
        d = np.maximum(d, 1e-6)
        order = np.argsort(d)[:k]
        nn = pool[order]
        w = _softmax_neg(d[order], temp=temp) * (np.maximum(self.a2[nn], 1e-6) ** a2_power)
        w = w / w.sum()
        return blend_features([self.feats[i] for i in nn], w), {
            "method": "a2_weighted_knn",
            "i0": i0,
            "nn": nn.tolist(),
            "w": w.tolist(),
        }

    def _exact(
        self, pool: np.ndarray, rng: np.random.Generator, prefer_low_a2: bool = False
    ) -> tuple[dict, dict]:
        if prefer_low_a2:
            w = np.maximum(self.lib_cfg.quiet_ceil + 1e-3 - self.a2[pool], 1e-6) ** 2
        else:
            w = np.maximum(self.a2[pool], 1e-6) ** 2
        w = w / w.sum()
        i0 = int(rng.choice(pool, p=w))
        return self.feats[i0], {"method": "exact_a2_weighted", "i0": i0}

    def _amplify(
        self,
        pool: np.ndarray,
        rng: np.random.Generator,
        alpha_lo: float = 1.0,
        alpha_hi: float = 1.35,
    ) -> tuple[dict, dict]:
        base = self._quiet_mean if self._quiet_mean is not None else self.feats[int(pool[0])]
        # Prefer high-A2 targets
        w = np.maximum(self.a2[pool], 1e-6) ** 2
        w = w / w.sum()
        i0 = int(rng.choice(pool, p=w))
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        return scale_residual(base, self.feats[i0], alpha), {
            "method": "amplify_residual",
            "i0": i0,
            "alpha": alpha,
        }

    def _local_pca(
        self,
        pool: np.ndarray,
        rng: np.random.Generator,
        k: int = 12,
        n_comp: int = 6,
        sigma: float = 0.55,
    ) -> tuple[dict, dict]:
        i0 = int(rng.choice(pool))
        d = np.linalg.norm(self.codes[pool] - self.codes[i0], axis=1)
        order = np.argsort(d)[: min(k, len(pool))]
        nn = pool[order]
        X = self.codes[nn]
        mu = X.mean(0)
        Xc = X - mu
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        n_c = min(n_comp, vt.shape[0], max(len(nn) - 1, 1))
        P = vt[:n_c].T
        scale = (np.linalg.norm(Xc @ P, axis=0) + 1e-6) / np.sqrt(max(len(nn), 1))
        coef = rng.normal(0.0, sigma, size=n_c) * scale
        z_s = mu + P @ coef
        dd = np.linalg.norm(self.codes[nn] - z_s, axis=1)
        w = _softmax_neg(dd, temp=6.0) * (np.maximum(self.a2[nn], 1e-6) ** 2)
        w = w / w.sum()
        return blend_features([self.feats[i] for i in nn], w), {
            "method": "local_pca",
            "i0": i0,
            "nn": nn.tolist(),
            "w": w.tolist(),
        }

    def _kde(
        self, pool: np.ndarray, rng: np.random.Generator, k: int = 7
    ) -> tuple[dict, dict]:
        mu = self.codes[pool].mean(0)
        std = self.codes[pool].std(0) + 1e-3
        z_s = mu + rng.normal(0.0, 1.0, size=mu.shape) * std
        dd = np.linalg.norm(self.codes[pool] - z_s, axis=1)
        order = np.argsort(dd)[:k]
        nn = pool[order]
        w = _softmax_neg(dd[order], temp=5.0) * (np.maximum(self.a2[nn], 1e-6) ** 2.5)
        w = w / w.sum()
        return blend_features([self.feats[i] for i in nn], w), {
            "method": "kde_retrieve",
            "nn": nn.tolist(),
            "w": w.tolist(),
            "z": z_s.tolist(),
        }

    def _hier(
        self,
        pool: np.ndarray,
        rng: np.random.Generator,
        alpha_lo: float = 0.85,
        alpha_hi: float = 1.25,
    ) -> tuple[dict, dict]:
        i_eq = int(rng.choice(self.quiet_pool)) if self.quiet_pool.size else int(rng.choice(len(self.codes)))
        w = np.maximum(self.a2[pool], 1e-6) ** 2
        w = w / w.sum()
        i_m = int(rng.choice(pool, p=w))
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        return scale_residual(self.feats[i_eq], self.feats[i_m], alpha), {
            "method": "hier_morph",
            "i_eq": i_eq,
            "i_m": i_m,
            "alpha": alpha,
        }

    def _strong_knn(
        self,
        pool: np.ndarray,
        rng: np.random.Generator,
        strong_floor: float = 0.28,
        alpha_max: float = 0.22,
        a2_power: float = 3.0,
    ) -> tuple[dict, dict]:
        """KNN only among strong bars; A₂-biased seed; small blend."""
        strong = pool[self.a2[pool] >= strong_floor]
        if strong.size < 2:
            strong = pool
        w = np.maximum(self.a2[strong], 1e-6) ** a2_power
        w = w / w.sum()
        i0 = int(rng.choice(strong, p=w))
        d = np.linalg.norm(self.codes[strong] - self.codes[i0], axis=1)
        d[strong == i0] = np.inf
        i1 = int(strong[int(np.argmin(d))])
        alpha = float(rng.uniform(0.0, alpha_max))
        feat = blend_features(
            [self.feats[i0], self.feats[i1]], np.array([1.0 - alpha, alpha])
        )
        return feat, {
            "method": "strong_knn",
            "i0": i0,
            "i1": i1,
            "alpha": alpha,
            "strong_floor": strong_floor,
        }

    def _amplify_knn_hybrid(
        self,
        pool: np.ndarray,
        rng: np.random.Generator,
        alpha_lo: float = 1.05,
        alpha_hi: float = 1.30,
        knn_alpha_max: float = 0.18,
        strong_floor: float = 0.25,
    ) -> tuple[dict, dict]:
        """Blend two strong bars, then amplify residual vs quiet mean."""
        feat_knn, meta_knn = self._strong_knn(
            pool, rng, strong_floor=strong_floor, alpha_max=knn_alpha_max
        )
        base = self._quiet_mean if self._quiet_mean is not None else self.feats[int(pool[0])]
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        feat = scale_residual(base, feat_knn, alpha)
        return feat, {
            "method": "amplify_knn_hybrid",
            "alpha": alpha,
            **{f"knn_{k}": v for k, v in meta_knn.items() if k != "method"},
        }

    def _topk_exact(
        self,
        pool: np.ndarray,
        rng: np.random.Generator,
        top_frac: float = 0.35,
        a2_power: float = 4.0,
    ) -> tuple[dict, dict]:
        """Exact decode of a top-fraction A₂ library member (near ceiling)."""
        order = np.argsort(self.a2[pool])[::-1]
        n_keep = max(2, int(np.ceil(top_frac * len(pool))))
        top = pool[order[:n_keep]]
        w = np.maximum(self.a2[top], 1e-6) ** a2_power
        w = w / w.sum()
        i0 = int(rng.choice(top, p=w))
        return self.feats[i0], {"method": "topk_exact", "i0": i0, "n_top": int(n_keep)}

    def _disk_only_amplify(
        self,
        pool: np.ndarray,
        rng: np.random.Generator,
        alpha_lo: float = 1.15,
        alpha_hi: float = 1.45,
        bh_mix: float = 0.35,
    ) -> tuple[dict, dict]:
        """Amplify disk residual vs quiet mean; mild BH blend (not α-amplified)."""
        base = self._quiet_mean if self._quiet_mean is not None else self.feats[int(pool[0])]
        w = np.maximum(self.a2[pool], 1e-6) ** 3
        w = w / w.sum()
        i0 = int(rng.choice(pool, p=w))
        target = self.feats[i0]
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        beta = min(1.0, bh_mix * alpha)
        out = {}
        for name in base:
            bb, bt = base[name]["bottleneck"], target[name]["bottleneck"]
            sb, st = base[name]["skips"], target[name]["skips"]
            if name == "disk":
                out[name] = {
                    "bottleneck": bb + alpha * (bt - bb),
                    "skips": tuple(b + alpha * (t - b) for b, t in zip(sb, st)),
                }
            else:
                out[name] = {
                    "bottleneck": (1 - beta) * bb + beta * bt,
                    "skips": tuple((1 - beta) * b + beta * t for b, t in zip(sb, st)),
                }
        return out, {"method": "disk_only_amplify", "i0": i0, "alpha": alpha, "bh_mix": beta}

    def sample_z_kde(
        self,
        kind: SampleKind = "barred",
        rng: np.random.Generator | None = None,
        *,
        jitter: float = 0.35,
        k_local: int = 8,
        global_mix: float = 0.15,
    ) -> np.ndarray:
        """Draw a continuous ``z`` near a pool member (local jitter + light global).

        Global diagonal Gaussian alone washes bars under retrieve (~0.17).
        Jittering a random library code in its local neighborhood keeps morphology.
        """
        rng = rng or np.random.default_rng()
        pool = self.bar_pool if kind == "barred" else self.quiet_pool
        # Prefer high/low A2 seeds for bar/quiet
        if kind == "barred":
            w = np.maximum(self.a2[pool], 1e-6) ** 2
        else:
            w = np.maximum(self.lib_cfg.quiet_ceil + 1e-3 - self.a2[pool], 1e-6) ** 2
        w = w / w.sum()
        i0 = int(rng.choice(pool, p=w))
        d = np.linalg.norm(self.codes[pool] - self.codes[i0], axis=1)
        order = np.argsort(d)[: min(k_local, len(pool))]
        nn = pool[order]
        local_std = self.codes[nn].std(0) + 1e-3
        z_loc = self.codes[i0] + rng.normal(0.0, jitter, size=local_std.shape) * local_std
        mu_g = self.codes[pool].mean(0)
        std_g = self.codes[pool].std(0) + 1e-3
        z_g = mu_g + rng.normal(0.0, 1.0, size=mu_g.shape) * std_g
        return (1.0 - global_mix) * z_loc + global_mix * z_g

    def sample_features_z_amplify(
        self,
        *,
        kind: SampleKind = "barred",
        z: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        alpha_lo: float = 1.10,
        alpha_hi: float = 1.40,
        k: int = 3,
        temp: float = 4.0,
        a2_power: float = 5.0,
        strong_floor: float = 0.28,
    ) -> tuple[dict, dict]:
        """Continuous ``z`` → strong-bar retrieve → amplify vs quiet mean.

        Quiet samples stay exact low-A₂ library members.  Prefer nearest
        *strong* bars so residual amplify does not start from washed blends.
        """
        rng = rng or np.random.default_rng()
        pool = self.bar_pool if kind == "barred" else self.quiet_pool
        if kind == "quiet":
            return self._exact(pool, rng, prefer_low_a2=True)
        if z is None:
            z = self.sample_z_kde(
                kind="barred", rng=rng, jitter=0.22, global_mix=0.05, k_local=6
            )
        strong = pool[self.a2[pool] >= strong_floor]
        if strong.size < 2:
            strong = pool
        feat_ret, meta_ret = self._retrieve_from_z(
            np.asarray(z, dtype=np.float64),
            strong,
            "barred",
            rng,
            k=k,
            temp=temp,
            a2_power=a2_power,
        )
        base = self._quiet_mean if self._quiet_mean is not None else self.feats[int(pool[0])]
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        feat = scale_residual(base, feat_ret, alpha)
        return feat, {
            "method": "z_amplify",
            "alpha": alpha,
            "z": np.asarray(z, dtype=np.float64).tolist(),
            **{f"ret_{kk}": v for kk, v in meta_ret.items() if kk != "method"},
        }

    def save_codes(self, path: Path | str) -> None:
        path = Path(path)
        payload: dict[str, Any] = {
            "codes": self.codes,
            "a2": self.a2,
            "mean": self.pca_mean,
            "pca_w": self.pca_w,
            "paths": np.asarray([m.get("path", "") for m in self.meta]),
            "kinds": np.asarray([m.get("kind", "") for m in self.meta]),
            "hashes": np.asarray(self.hashes, dtype=object),
        }
        if self.theta.size and self.theta.shape[1] > 0:
            payload["theta"] = self.theta
        np.savez_compressed(path, **payload)


def encode_snapshot_features(
    path: str | Path,
    *,
    teacher: torch.nn.Module,
    cfg: MultiScaleSliceConfig,
    stats: dict[str, FieldNormStats],
    phi: float | None = None,
    enc_grid: int = 4,
) -> tuple[dict, np.ndarray, float]:
    """Encode one snapshot (optional common rotation) → features, raw-z, data A₂."""
    from ntropy.analysis.disk_density import disk_azimuthal_fourier

    with np.load(path, allow_pickle=True) as data:
        arr = {k: data[k] for k in data.files}
    pos = np.asarray(arr["pos"], dtype=np.float64)
    vel = np.asarray(arr["vel"], dtype=np.float64)
    mass = np.asarray(arr["mass"], dtype=np.float64)
    cid = _component_ids(arr.get("tags"), arr.get("type_id"), pos.shape[0])
    pos, vel, _ = prepare_shared_frame(
        pos, vel, mass, center=True, rotate=phi is not None, phi=phi
    )
    a2 = float(
        disk_azimuthal_fourier(
            pos[cid == 0], mass[cid == 0], m=2, r_max=12.0, n_bins=12, z_max=0.5, min_count=10
        )["a_m_over_a0_median"]
    )
    binned = bin_multiscale_slice_stacks(pos, vel, mass, cid, cfg=cfg)
    maps = {k: v[0] for k, v in binned.items()}
    device = next(teacher.parameters()).device
    stacks = {
        k: torch.as_tensor(
            normalize_stack(v, stats[k])[None], dtype=torch.float32, device=device
        )
        for k, v in maps.items()
    }
    with torch.no_grad():
        feat = teacher.encode_features(stacks)
        z = pool_bottleneck_z(feat, enc_grid=enc_grid)[0].cpu().numpy()
    feat_cpu = {
        name: {
            "bottleneck": feat[name]["bottleneck"].detach().cpu().contiguous(),
            "skips": tuple(s.detach().cpu().contiguous() for s in feat[name]["skips"]),
        }
        for name in feat
    }
    return feat_cpu, z, a2


def _default_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_frozen_teacher_bundle(
    teacher_path: Path | str,
    *,
    device: str | None = None,
) -> tuple[torch.nn.Module, MultiScaleSliceConfig, dict[str, FieldNormStats]]:
    teacher_path = Path(teacher_path)
    if device is None:
        device = _default_torch_device()
    probe = torch.load(teacher_path, map_location="cpu", weights_only=False)
    t_args = probe.get("args", {})
    cfg = MultiScaleSliceConfig.progressive_defaults(
        disk_n_pix=int(t_args.get("disk_n_pix", 128)),
        include_potential=False,
        moment_set=str(t_args.get("moment_set", "disp")),
    )
    teacher, ckpt = load_frozen_teacher(teacher_path, cfg, device=device)
    return teacher, cfg, _as_stats(ckpt["norm"])
