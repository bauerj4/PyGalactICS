"""Generate reference potential and sampling artifacts via the Python API."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from galacticsics.artifacts.paths import default_artifact_dir, reference_model
from galacticsics.artifacts.verify import verify_artifact_consistency
from galacticsics.io.legacy_inputs import write_diskdf_input
from galacticsics.legacy.paths import require_binary
from galacticsics.legacy.runner import LegacyRunner
from galacticsics.models import GalaxyModel
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.sampler import SampleConfig, sample_galaxy


def _write_manifest(path: Path, model: GalaxyModel, *, extra: dict[str, Any] | None = None) -> None:
    """Record model parameters used to produce artifacts."""
    payload: dict[str, Any] = {
        "model": "reference_disk_halo",
        "grid": {"dr": model.grid.dr, "nr": model.grid.nr, "lmax": model.grid.lmax},
        "halo": {
            "r_outer_kpc": model.halo.r_outer if model.halo else None,
            "v0_100kms": model.halo.v0 if model.halo else None,
            "a_kpc": model.halo.a if model.halo else None,
        },
        "disk": {
            "mass_galactics": model.disk.mass if model.disk else None,
            "scale_length_kpc": model.disk.scale_length if model.disk else None,
        },
        "disk_kinematics": asdict(model.disk_kinematics),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def generate_reference_artifacts(
    output_dir: Path | str | None = None,
    *,
    model: GalaxyModel | None = None,
    sample_disk: int = 2000,
    sample_halo: int = 2000,
    seed: int = -42,
    verify: bool = True,
    force: bool = False,
) -> Path:
    """
    Build the full reference artifact tree used by tests.

    Steps
    -----
    1. ``solve_potential`` (self-consistent disk + halo) → ``dbh.dat``, ``h.dat``, …
    2. ``getfreqs`` + ``diskdf`` → ``freqdbh.dat``, ``cordbh.dat``, ``toomre2.5``
    3. ``gendisk`` / ``genhalo`` → ``disk``, ``halo`` particle files

    Parameters
    ----------
    output_dir : path-like, optional
        Destination directory.  Default :func:`default_artifact_dir`.
    model : GalaxyModel, optional
        Model to solve.  Default :func:`reference_model`.
    sample_disk, sample_halo : int
        Particle counts for sampling outputs.
    seed : int
        RNG seed for samplers.
    verify : bool
        Run :func:`verify_artifact_consistency` before returning.
    force : bool
        Regenerate even if ``manifest.json`` already exists.

    Returns
    -------
    Path
        ``output_dir`` containing generated files.

    Raises
    ------
    FileNotFoundError
        If legacy binaries are not built.
    RuntimeError
        If consistency checks fail.
    """
    out = Path(output_dir) if output_dir is not None else default_artifact_dir()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"

    if manifest.is_file() and not force:
        if verify:
            verify_artifact_consistency(out, model=model or reference_model())
        return out

    model = model or reference_model()
    require_binary("dbh")
    require_binary("getfreqs")
    require_binary("diskdf")
    require_binary("gendisk")
    require_binary("genhalo")

    if force:
        for child in out.iterdir():
            if child.is_file():
                child.unlink()

    solve = solve_potential(model, work_dir=out, cleanup=False, timeout=None)
    if not (out / "h.dat").is_file():
        raise RuntimeError(f"dbh did not produce h.dat in {out}")

    runner = LegacyRunner(out)
    runner.run("getfreqs")
    write_diskdf_input(model, out / "in.diskdf")
    runner.run("diskdf", stdin_path=out / "in.diskdf")

    config = SampleConfig(
        n_disk=sample_disk,
        n_halo=sample_halo,
        seed_disk=seed,
        seed_halo=seed,
        run_diskdf=False,
    )
    sample_galaxy(model, config, work_dir=out, artifact_dir=out, cleanup=False)

    _write_manifest(
        manifest,
        model,
        extra={
            "tidal_radius_kpc": solve.diagnostics.tidal_radius,
            "component_masses": solve.diagnostics.component_masses,
            "sample_disk": sample_disk,
            "sample_halo": sample_halo,
            "seed": seed,
        },
    )

    if verify:
        report = verify_artifact_consistency(out, model=model)
        if not report.ok:
            shutil.rmtree(out, ignore_errors=True)
            raise RuntimeError("artifact consistency checks failed:\n" + report.summary())

    return out
