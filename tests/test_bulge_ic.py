"""Bulge IC morphology, densψ amplitude, and spherical sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from galacticsics.models import GalaxyModel, NFWHalo, PotentialGrid, SersicBulge
from galacticsics.physics.backend import PhysicsBackendKind
from galacticsics.potential.evaluate import evaluate_potential
from galacticsics.potential.poisson.densities import bulge_density_psi
from galacticsics.potential.poisson.sersic import bulge_density_spherical, sersic_params_from_bulge
from galacticsics.potential.solver import solve_potential
from galacticsics.io import read_harmonic_potential
from galacticsics.sampling.sampler import SampleConfig, sample_galaxy

pytestmark = [pytest.mark.essential, pytest.mark.physics_python]


def _bulge_halo_model(*, dr: float = 0.1, nr: int = 80) -> GalaxyModel:
    return GalaxyModel(
        halo=NFWHalo(r_outer=40.0, v0=2.0, a=6.0, dr_trunc=6.0, enabled=True),
        bulge=SersicBulge(n_sersic=4.0, ppp=0.5, v0=1.5, a=0.4, enabled=True),
        grid=PotentialGrid(dr=dr, nr=nr, lmax=0),
    )


@pytest.mark.physics_python
def test_bulge_denspsi_tracks_analytic(tmp_path: Path) -> None:
    """denspsibulge must track analytic Sersic ρ(r), not DF round-trip amplitude."""
    model = _bulge_halo_model()
    work = tmp_path / "denspsi"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    pot = read_harmonic_potential(work / "dbh.dat")
    table = np.loadtxt(work / "denspsibulge.dat")
    params = sersic_params_from_bulge(model.bulge)
    for r in (0.1, 0.2, 0.4, 0.8):
        psi = evaluate_potential(pot, r, 0.0)
        analytic = bulge_density_spherical(r, model.bulge, params)
        from_tab = bulge_density_psi(
            psi,
            table[:, 0],
            table[:, 1],
            psi0=pot.psi0,
            psic=pot.psic,
            psid=getattr(pot, "psid", pot.psi0),
        )
        assert analytic > 0.0
        assert abs(from_tab - analytic) / analytic < 0.25, (
            f"r={r}: denspsi={from_tab} analytic={analytic}"
        )


@pytest.mark.physics_python
def test_bulge_ic_is_spherical_not_cylinder(tmp_path: Path) -> None:
    """Spherical sampling: axis ratios near isotropic sphere, not cylindrical envelope."""
    model = _bulge_halo_model()
    work = tmp_path / "sphere"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    result = sample_galaxy(
        model,
        SampleConfig(
            n_disk=0,
            n_halo=0,
            n_bulge=3000,
            run_diskdf=False,
            stream_bulge=0.5,
        ),
        work_dir=work,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )
    d = result.particles["bulge"].data
    x, y, z = d["x"], d["y"], d["z"]
    mass = d["mass"]
    R = np.hypot(x, y)
    r = np.sqrt(R * R + z * z)
    Rrms = float(np.sqrt(np.average(R * R, weights=mass)))
    zrms = float(np.sqrt(np.average(z * z, weights=mass)))
    # Sphere: z_rms / R_rms = 1/√2 ≈ 0.707
    ratio = zrms / Rrms
    assert 0.55 < ratio < 0.90, f"zrms/Rrms={ratio} (cylinder-like if ≫0.9 or ≪0.55)"

    bulgeedge = float((work / "mr.dat").read_text().splitlines()[1].split()[1])
    # Must not pile up on the old cylindrical wall R=edge, |z|=2*edge
    assert float(np.percentile(R, 99)) < 0.95 * bulgeedge
    assert float(np.percentile(np.abs(z), 99)) < 1.5 * bulgeedge

    # ρ(r) shape vs analytic inside ~2a
    params = sersic_params_from_bulge(model.bulge)
    a = model.bulge.a
    bins = np.linspace(0.05 * a, 2.0 * a, 8)
    mids = 0.5 * (bins[:-1] + bins[1:])
    dig = np.digitize(r, bins) - 1
    for i, rm in enumerate(mids):
        sel = dig == i
        if np.count_nonzero(sel) < 20:
            continue
        vol = (4.0 / 3.0) * np.pi * (bins[i + 1] ** 3 - bins[i] ** 3)
        rho_p = float(np.sum(mass[sel]) / vol)
        rho_a = bulge_density_spherical(float(rm), model.bulge, params)
        # Loose: sampling noise + grid; catch order-of-magnitude failures
        assert 0.2 < rho_p / rho_a < 5.0, f"r={rm}: particle ρ={rho_p} analytic={rho_a}"


@pytest.mark.physics_python
def test_bulge_df_monotonic_no_floor(tmp_path: Path) -> None:
    """dfsersic must rise toward high E — not a tiny floor that breaks fmax=f(ψ)."""
    model = _bulge_halo_model()
    work = tmp_path / "df"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    df = np.loadtxt(work / "dfsersic.dat")
    order = np.argsort(df[:, 0])
    e = df[order, 0]
    logf = df[order, 1]
    # Upper third of the energy grid should not sit on a constant floor.
    n = len(e)
    hi = logf[2 * n // 3 :]
    assert float(np.std(hi)) > 0.05, "high-E DF still flat (Eddington floor bug)"
    assert float(logf[-1]) > float(logf[n // 2]), "f(E) should increase toward centre"


@pytest.mark.physics_python
def test_bulge_velocity_moments_match_df(tmp_path: Path) -> None:
    """Inner bulge <v²> must track the Eddington DF, not escape-speed overheating."""
    from scipy.interpolate import interp1d

    model = _bulge_halo_model()
    work = tmp_path / "moments"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    pot = read_harmonic_potential(work / "dbh.dat")
    df = np.loadtxt(work / "dfsersic.dat")
    order = np.argsort(df[:, 0])
    log_interp = interp1d(
        df[order, 0],
        df[order, 1],
        kind="linear",
        bounds_error=False,
        fill_value=(float(df[order, 1][0]), float(df[order, 1][-1])),
        assume_sorted=True,
    )
    fcut = float(np.exp(log_interp(pot.psic)))

    def sigma2_df(psi: float, n: int = 2000) -> float:
        vmax2 = 2.0 * (psi - pot.psic)
        if vmax2 <= 0.0:
            return 0.0
        v = np.linspace(0.0, float(np.sqrt(vmax2)), n)
        e = psi - 0.5 * v * v
        f = np.maximum(np.exp(log_interp(e)) - fcut, 0.0)
        num = float(np.trapezoid(v**4 * f, v))
        den = float(np.trapezoid(v**2 * f, v))
        return num / den if den > 0.0 else 0.0

    result = sample_galaxy(
        model,
        SampleConfig(n_disk=0, n_halo=0, n_bulge=8000, run_diskdf=False, stream_bulge=0.5),
        work_dir=work,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )
    d = result.particles["bulge"].data
    r = np.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)
    v2 = d["vx"] ** 2 + d["vy"] ** 2 + d["vz"] ** 2
    for r_lo, r_hi in ((0.05, 0.15), (0.15, 0.30), (0.40, 0.80)):
        sel = (r >= r_lo) & (r < r_hi)
        if int(np.count_nonzero(sel)) < 80:
            continue
        rc = 0.5 * (r_lo + r_hi)
        psi = evaluate_potential(pot, rc, 0.0)
        ratio = float(np.mean(v2[sel])) / max(sigma2_df(psi), 1e-30)
        assert 0.45 < ratio < 2.2, f"r~{rc}: <v²>_samp/<v²>_DF={ratio}"


@pytest.mark.physics_python
def test_bulge_potential_virial_equilibrium(tmp_path: Path) -> None:
    """Bulge (+halo) ICs satisfy 2K+|W|≈0 with a=∇Ψ from dbh (not softened N-body)."""
    from galacticsics.diagnostics.virial import virial_diagnostic_potential

    model = _bulge_halo_model()
    work = tmp_path / "virial"
    solve_potential(model, work_dir=work, cleanup=False, backend=PhysicsBackendKind.PYTHON)
    result = sample_galaxy(
        model,
        SampleConfig(n_disk=0, n_halo=4000, n_bulge=4000, run_diskdf=False, stream_bulge=0.5),
        work_dir=work,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )
    pos_list = []
    vel_list = []
    mass_list = []
    for name in ("bulge", "halo"):
        d = result.particles[name].data
        pos_list.append(np.stack([d["x"], d["y"], d["z"]], axis=1))
        vel_list.append(np.stack([d["vx"], d["vy"], d["vz"]], axis=1))
        mass_list.append(d["mass"])
    pos = np.vstack(pos_list)
    vel = np.vstack(vel_list)
    mass = np.concatenate(mass_list)
    diag = virial_diagnostic_potential(pos, vel, mass, work / "dbh.dat", rtol=0.25)
    assert diag["is_virial_equilibrium"], diag
    assert 0.75 <= diag["virial_ratio"] <= 1.25, diag
