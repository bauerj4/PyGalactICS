"""Campaign training bundle export for ML encoders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from galacticsics.ml.heads import TrainingLabels
from galacticsics.representations.learned import (
    EncoderBackend,
    LearnedEncoderConfig,
    build_encoder,
)
from galacticsics.representations.particle_features import (
    FEATURE_NAMES,
    particle_state_to_batch,
)


@dataclass
class CampaignTrainingRecord:
    """
    One training example: IC features + evolution labels.

    Attributes
    ----------
    run_hash : str
        Campaign model directory hash.
    label : str
        Human-readable grid label.
    latent : ndarray or None
        Encoded vector when ``encode=True``.
    features_path : str or None
        Path to ``features.npz`` when written.
    labels : TrainingLabels
        Scalar targets from diagnostics and manifest.
    """

    run_hash: str
    label: str
    labels: TrainingLabels
    latent: np.ndarray | None = None
    features_path: str | None = None
    field_path: str | None = None


@dataclass
class CampaignTrainingBundle:
    """Aggregate export from a campaign work root."""

    work_root: Path
    records: list[CampaignTrainingRecord] = field(default_factory=list)
    encoder_backend: str = EncoderBackend.MEAN_POOL.value

    def to_manifest(self) -> dict[str, Any]:
        return {
            "work_root": str(self.work_root),
            "encoder_backend": self.encoder_backend,
            "n_records": len(self.records),
            "records": [
                {
                    "run_hash": r.run_hash,
                    "label": r.label,
                    "features_path": r.features_path,
                    "field_path": r.field_path,
                    "latent_shape": list(r.latent.shape) if r.latent is not None else None,
                    "labels": {
                        "scalars": r.labels.scalars,
                        "params": r.labels.params,
                    },
                }
                for r in self.records
            ],
        }

    def write(self, output_dir: Path | str) -> Path:
        """Write ``training_manifest.json`` and per-run arrays."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "training_manifest.json"
        manifest_path.write_text(json.dumps(self.to_manifest(), indent=2))
        return manifest_path


def _load_diagnostics_scalars(run_dir: Path) -> dict[str, float]:
    """Read final-row scalars from ``evolution/diagnostics.csv``."""
    csv_path = run_dir / "evolution" / "diagnostics.csv"
    if not csv_path.is_file():
        return {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    out: dict[str, float] = {}
    for key in ("dE_over_E0", "active_fraction", "energy", "t_gyr"):
        if key in last and last[key] not in ("", None):
            try:
                out[key] = float(last[key])
            except ValueError:
                pass
    if "mean_bin" in last:
        try:
            out["mean_bin"] = float(last["mean_bin"])
        except (ValueError, TypeError):
            pass
    return out


def _params_from_model_json(run_dir: Path) -> dict[str, float]:
    """Extract sweep-relevant floats from ``model.json``."""
    model_path = run_dir / "model.json"
    if not model_path.is_file():
        return {}
    data = json.loads(model_path.read_text())
    out: dict[str, float] = {}

    halo = data.get("halo") or {}
    if "v0" in halo:
        out["halo.v0"] = float(halo["v0"])

    disk = data.get("disk") or {}
    if "mass" in disk:
        out["disk.mass"] = float(disk["mass"])

    return out


def _particle_set_to_state(ps) -> "ParticleState":
    from ntropy.particles import ParticleState

    n = len(ps.data)
    pos = np.column_stack([ps.data["x"], ps.data["y"], ps.data["z"]])
    vel = np.column_stack([ps.data["vx"], ps.data["vy"], ps.data["vz"]])
    mass = ps.data["mass"].copy()
    eps = np.full(n, 0.01, dtype=float)
    return ParticleState.from_arrays(pos, vel, mass, eps)


def _load_particle_state(run_dir: Path):
    from ntropy.integrations.galacticsics import merge_galacticsics_components
    from ntropy.particle_types import TypeRegistry

    merged = run_dir / "merged.dat"
    if merged.is_file():
        from galacticsics.sampling.particles import ParticleSet

        return _particle_set_to_state(ParticleSet.from_ascii(merged, component="merged"))

    registry = TypeRegistry.default_galaxy()
    particles: dict = {}
    for name in ("halo", "bulge", "disk"):
        comp = run_dir / name
        if comp.exists():
            from galacticsics.sampling.particles import ParticleSet

            particles[name] = ParticleSet.from_ascii(comp, component=name)
    if particles:
        return merge_galacticsics_components(particles, type_registry=registry)

    raise FileNotFoundError(f"No particle files in {run_dir}")


def export_campaign_training_bundle(
    work_root: Path | str,
    *,
    output_dir: Path | str | None = None,
    encoder_config: LearnedEncoderConfig | None = None,
    encode: bool = True,
    write_features: bool = True,
    write_fields: bool = False,
) -> CampaignTrainingBundle:
    """
    Export a campaign work tree as an ML training bundle.

    Parameters
    ----------
    work_root : path
        Campaign root containing per-hash run directories.
    output_dir : path, optional
        Defaults to ``work_root / ml_training``.
    encoder_config : LearnedEncoderConfig, optional
        Encoder used when ``encode=True``.
    encode : bool
        Compute latent vectors per run.
    write_features : bool
        Write ``features.npz`` per run under ``output_dir``.
    write_fields : bool
        Write binned ``field.npy`` when using field encoder.

    Returns
    -------
    bundle : CampaignTrainingBundle
    """
    work_root = Path(work_root)
    out = Path(output_dir) if output_dir else work_root / "ml_training"
    out.mkdir(parents=True, exist_ok=True)

    cfg = encoder_config or LearnedEncoderConfig(backend=EncoderBackend.MEAN_POOL)
    encoder = build_encoder(cfg) if encode else None

    bundle = CampaignTrainingBundle(
        work_root=work_root,
        encoder_backend=cfg.backend.value,
    )

    jsonl = work_root / "manifest.jsonl"
    manifest_rows: list[dict[str, Any]] = []
    if jsonl.is_file():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                manifest_rows.append(json.loads(line))

    if not manifest_rows:
        # Discover run dirs by model.json
        for model_json in sorted(work_root.glob("*/model.json")):
            run_dir = model_json.parent
            manifest_rows.append(
                {"hash": run_dir.name, "label": run_dir.name, "path": str(run_dir)}
            )

    for row in manifest_rows:
        run_hash = row.get("hash", row.get("run_hash", "unknown"))
        label = row.get("label", run_hash)
        run_dir = Path(row.get("path", work_root / run_hash))
        if not run_dir.is_dir():
            continue

        scalars = _load_diagnostics_scalars(run_dir)
        params = _params_from_model_json(run_dir)
        if "dE_over_E0" in row and "dE_over_E0" not in scalars:
            scalars["dE_over_E0"] = float(row["dE_over_E0"])

        labels = TrainingLabels(run_hash=run_hash, label=label, scalars=scalars, params=params)

        latent = None
        features_path = None
        field_path = None

        try:
            state = _load_particle_state(run_dir)
            batch = particle_state_to_batch(state)

            run_out = out / run_hash
            if write_features:
                run_out.mkdir(parents=True, exist_ok=True)
                features_path = str(run_out / "features.npz")
                np.savez(
                    features_path,
                    feature_names=np.array(FEATURE_NAMES),
                    **batch.as_dict(),
                )

            if encode and encoder is not None:
                rep = encoder.encode(batch)
                latent = rep.latent
                if write_fields and rep.field is not None:
                    run_out.mkdir(parents=True, exist_ok=True)
                    field_path = str(run_out / "field.npy")
                    np.save(field_path, rep.field)
                if encode:
                    run_out.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        run_out / "latent.npz",
                        latent=latent,
                        backend=cfg.backend.value,
                        trained=rep.metadata.trained if rep.metadata else False,
                    )
        except FileNotFoundError:
            pass

        bundle.records.append(
            CampaignTrainingRecord(
                run_hash=run_hash,
                label=label,
                labels=labels,
                latent=latent,
                features_path=features_path,
                field_path=field_path,
            )
        )

    bundle.write(out)
    return bundle
