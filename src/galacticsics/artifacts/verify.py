"""Consistency checks for generated reference artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from galacticsics.artifacts.paths import reference_model
from galacticsics.io.formats import (
    merge_harmonic_potentials,
    read_component_masses,
    read_disk_correction,
    read_frequency_table,
    read_halo_harmonics,
    read_harmonic_potential,
    read_rtidal,
    read_toomre_q,
    write_harmonic_potential,
)
from galacticsics.models import GalaxyModel
from galacticsics.potential.evaluate import evaluate_force, evaluate_potential


@dataclass
class ConsistencyReport:
    """Result of :func:`verify_artifact_consistency`."""

    ok: bool
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = self.checks + [f"ERROR: {e}" for e in self.errors]
        return "\n".join(lines)


def verify_artifact_consistency(
    artifact_dir: Path | str,
    *,
    model: GalaxyModel | None = None,
) -> ConsistencyReport:
    """
    Confirm generated artifacts are internally consistent.

    Checks
    ------
    - Required files exist (``dbh.dat``, ``h.dat``, ``cordbh.dat``, …)
    - ``manifest.json`` grid matches the expected model
    - Harmonic round-trip read/write preserves coefficients
    - Halo + baryon merge is finite and same shape as ``dbh.dat``
    - Potential/force sanity at :math:`R=8` kpc, :math:`z=0`
    - ``mr.dat`` masses are positive; disk mass order matches input
    - Epicycle frequencies positive at :math:`R=5` kpc
    - Toomre :math:`Q` is finite and positive

    Parameters
    ----------
    artifact_dir : path-like
        Directory produced by :func:`~galacticsics.artifacts.generate_reference_artifacts`.
    model : GalaxyModel, optional
        Expected model parameters.

    Returns
    -------
    ConsistencyReport
    """
    root = Path(artifact_dir)
    model = model or reference_model()
    report = ConsistencyReport(ok=True)

    required = (
        "dbh.dat",
        "h.dat",
        "cordbh.dat",
        "freqdbh.dat",
        "mr.dat",
        "rtidal.dat",
        "manifest.json",
        "disk",
        "halo",
    )
    for name in required:
        if not (root / name).is_file():
            report.errors.append(f"missing {name}")
    if report.errors:
        report.ok = False
        return report

    manifest = json.loads((root / "manifest.json").read_text())
    g = manifest["grid"]
    if (g["dr"], g["nr"], g["lmax"]) != (model.grid.dr, model.grid.nr, model.grid.lmax):
        report.errors.append(
            f"manifest grid {g} != model ({model.grid.dr}, {model.grid.nr}, {model.grid.lmax})"
        )

    pot = read_harmonic_potential(root / "dbh.dat")
    halo = read_halo_harmonics(root / "h.dat")

    if pot.model.disk and pot.model.disk.mass != model.disk.mass:
        report.errors.append("dbh.dat disk mass does not match model")
    if pot.nr != model.grid.nr or pot.lmax != model.grid.lmax:
        report.errors.append("dbh.dat grid does not match model")

    roundtrip = root / "_roundtrip_dbh.dat"
    write_harmonic_potential(pot, roundtrip)
    pot2 = read_harmonic_potential(roundtrip)
    if not np.allclose(pot.apot, pot2.apot, rtol=1e-5, atol=1e-8):
        report.errors.append("apot round-trip mismatch")
    else:
        report.checks.append("apot round-trip OK")
    roundtrip.unlink(missing_ok=True)

    merged = merge_harmonic_potentials(halo, pot, model=pot.model)
    if not np.all(np.isfinite(merged.apot)):
        report.errors.append("merged harmonics are not finite")
    else:
        report.checks.append("halo merge finite")

    psi = evaluate_potential(pot, 8.0, 0.0)
    fr, fz, _ = evaluate_force(pot, 8.0, 0.0)
    if not (psi > 0 and fr < 0):
        report.errors.append(f"unexpected potential/force at R=8: psi={psi}, fr={fr}")
    else:
        report.checks.append("potential/force at R=8 OK")
    if abs(fz) > 0.1:
        report.errors.append(f"midplane vertical force too large: fz={fz}")
    else:
        report.checks.append("midplane fz ~ 0 OK")

    v_circ = math.sqrt(max(0.0, -fr * 8.0))
    if not (1.0 < v_circ < 4.0):
        report.errors.append(f"v_circ at 8 kpc out of range: {v_circ}")
    else:
        report.checks.append(f"v_circ(8 kpc)={v_circ:.3f} OK")

    masses = read_component_masses(root / "mr.dat")
    dm, _ = masses["disk"]
    hm, _ = masses["halo"]
    if dm <= 0 or hm <= 0:
        report.errors.append("non-positive component masses in mr.dat")
    elif hm < dm:
        report.errors.append("expected halo mass > disk mass")
    else:
        report.checks.append("mr.dat masses OK")

    freq = read_frequency_table(root / "freqdbh.dat")
    if freq.omega(5.0) <= 0 or freq.kappa(5.0) <= 0:
        report.errors.append("non-positive epicycle frequencies at R=5")
    else:
        report.checks.append("epicycle frequencies OK")

    corr = read_disk_correction(root / "cordbh.dat")
    if corr.f_d_at(0.0) < 0.5:
        report.errors.append("cordbh f_d(0) too small")
    else:
        report.checks.append("cordbh OK")

    rtidal = read_rtidal(root / "rtidal.dat")
    if rtidal <= 0:
        report.errors.append("non-positive tidal radius")
    else:
        report.checks.append(f"rtidal={rtidal:.2f} kpc OK")

    toomre = read_toomre_q(root / "toomre2.5")
    if not (0.5 < toomre < 3.0):
        report.errors.append(f"Toomre Q out of range: {toomre}")
    else:
        report.checks.append(f"Toomre Q={toomre:.3f} OK")

    report.ok = len(report.errors) == 0
    return report
