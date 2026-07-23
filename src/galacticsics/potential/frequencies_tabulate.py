"""Python port of legacy ``getfreqs``."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from galacticsics.io import read_harmonic_potential
from galacticsics.io.formats import write_frequency_table
from galacticsics.potential.evaluate import evaluate_potential, evaluate_potential_harmonic_only
from galacticsics.potential.harmonics import HarmonicPotential


def _radial_dpsi_dr(pot: HarmonicPotential, r: float) -> float:
    """Legacy ``getfreqs`` uses ``pot(r,0)`` with radial-force harmonics as ``apot``."""
    pot_force = replace(pot, apot=pot.fr)
    return evaluate_potential_harmonic_only(pot_force, r, 0.0)


def _d2psi_dr2(pot: HarmonicPotential, r: float) -> float:
    """Legacy ``getfreqs`` uses ``pot(r,0)`` with ``d²Psi/dr²`` harmonics as ``apot``."""
    if pot.fr2 is None:
        raise ValueError("dbh.dat is missing the d²Psi/dr² harmonic block required for freqdbh.dat")
    pot_d2 = replace(pot, apot=pot.fr2)
    return evaluate_potential_harmonic_only(pot_d2, r, 0.0)


def tabulate_frequencies(
    work_dir: Path,
    *,
    theta_incline: float = 0.05,
) -> None:
    """
    Build ``freqdbh.dat`` from ``dbh.dat`` and ``h.dat`` using Python potential eval.
    """
    work_dir = Path(work_dir)
    total = read_harmonic_potential(work_dir / "dbh.dat")
    halo_path = work_dir / "h.dat"
    halo = read_harmonic_potential(halo_path) if halo_path.is_file() else total

    dr = total.dr
    nr = total.nr
    radius = np.linspace(0, nr * dr, nr + 1)
    omega_h = np.zeros(nr + 1, dtype=float)
    nu_h = np.zeros(nr + 1, dtype=float)
    sigma_d = np.zeros(nr + 1, dtype=float)
    v_circ_total = np.zeros(nr + 1, dtype=float)
    bulge_vc = np.zeros(nr + 1, dtype=float)
    bulge_nu = np.zeros(nr + 1, dtype=float)
    psi_mid = np.zeros(nr + 1, dtype=float)
    d2psi = np.zeros(nr + 1, dtype=float)

    if total.flags.disk and total.model.disk:
        from galacticsics.potential.poisson.densities import disk_surface_density

        for ir in range(nr + 1):
            sigma_d[ir] = disk_surface_density(ir * dr, total.model)

    vcmaj = np.zeros(nr + 1, dtype=float)
    vertfreq = np.zeros(nr + 1, dtype=float)
    for ir in range(nr + 1):
        r = ir * dr
        potmaj = evaluate_potential(halo, r, 0.0)
        potup = evaluate_potential(halo, r * math.cos(theta_incline), r * math.sin(theta_incline))
        if ir == 0:
            vcmaj[ir] = 0.0
            vertfreq[ir] = 0.0
        else:
            vcmaj[ir] = math.sqrt(max(0.0, r * _radial_dpsi_dr(halo, r)))
            vertfreq[ir] = (
                math.sqrt(
                    max(
                        0.0,
                        vcmaj[ir] ** 2 + 2.0 * (potup - potmaj) / theta_incline**2,
                    )
                )
                / r
            )
        omega_h[ir] = vcmaj[ir] / r if ir > 0 else 0.0
        nu_h[ir] = vertfreq[ir]

    if nr >= 1:
        omega_h[0] = vcmaj[1] / dr
        nu_h[0] = (4.0 * vertfreq[1] - vertfreq[2]) / 3.0 if nr >= 2 else vertfreq[1]

    for ir in range(nr + 1):
        r = ir * dr
        psi_mid[ir] = evaluate_potential(total, r, 0.0)
        if ir > 0:
            dpsi_dr = _radial_dpsi_dr(total, r)
            v_circ_total[ir] = math.sqrt(max(0.0, r * dpsi_dr))
        d2psi[ir] = _d2psi_dr2(total, r)

    write_frequency_table(
        work_dir / "freqdbh.dat",
        radius=radius,
        omega_h=omega_h,
        nu_h=nu_h,
        sigma_d=sigma_d,
        v_circ_total=v_circ_total,
        v_circ_bulge=bulge_vc,
        nu_bulge=bulge_nu,
        psi_midplane=psi_mid,
        d2psi_dr2=d2psi,
    )
