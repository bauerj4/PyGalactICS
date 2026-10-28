"""Regressions for the projected stellar-disk surface-density contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, quad
from scipy.stats import ks_2samp, kstest

from galacticsics.io import read_harmonic_potential
from galacticsics.campaign.serialize import model_from_dict, model_to_dict
from galacticsics.models import GalaxyModel, NFWHalo, PotentialGrid, SersicBulge
from galacticsics.potential.evaluate import evaluate_potential
from galacticsics.potential.poisson.densities import (
    disk_density_psi,
    disk_density_psi_batch,
    disk_surface_density,
    disk_surface_density_array,
)
from galacticsics.potential.poisson.disk_vertical import (
    disk_vertical_normalization_table,
    interpolate_disk_vertical_normalization,
)
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.openmp import extension_available, sample_disk_openmp
from galacticsics.sampling.python.samplers import sample_disk_python
from galacticsics.sampling.sampler import SampleConfig


def _surface_normalized_disk_bulge_halo_model() -> GalaxyModel:
    """Compact, centrally concentrated model that exposes column-density errors."""
    base = GalaxyModel.reference_disk_halo()
    assert base.disk is not None
    return replace(
        base,
        disk=replace(
            base.disk,
            mass=6.0,
            scale_length=2.0,
            outer_radius=8.0,
            trunc_width=1.0,
            vertical_density_mode="surface_normalized",
        ),
        halo=NFWHalo(
            r_outer=40.0,
            v0=3.2,
            a=25.0,
            dr_trunc=6.0,
            enabled=True,
        ),
        bulge=SersicBulge(
            n_sersic=4.0,
            ppp=0.5,
            v0=1.5,
            a=0.4,
            enabled=True,
        ),
        # diskdf requires a smooth central frequency table.  dr=0.15 kpc is
        # adequate for the column integral but aliases the velocity DF; the
        # production corpus and this regression both use 0.05 kpc.
        grid=PotentialGrid(dr=0.05, nr=800, lmax=2),
    )


@pytest.fixture(scope="module")
def surface_normalized_potential(tmp_path_factory: pytest.TempPathFactory):
    model = _surface_normalized_disk_bulge_halo_model()
    work = tmp_path_factory.mktemp("surface_normalized_disk")
    result = solve_potential(
        model,
        work_dir=work,
        cleanup=False,
        npsi=256,
        nint=12,
        n_workers=0,
        backend="python",
    )
    return model, result.potential, Path(work)


@pytest.fixture(scope="module")
def surface_normalized_sampler_work(surface_normalized_potential) -> Path:
    from galacticsics.distribution.diskdf_solve import solve_diskdf_python

    model, _, work = surface_normalized_potential
    corr = solve_diskdf_python(model, work, n_iterations=10, n_radial_steps=50)
    interior = corr.radius <= model.disk.outer_radius
    assert float(np.min(corr.f_d[interior])) > 0.2
    assert float(np.min(corr.f_sz[interior])) > 0.2
    return work


@pytest.mark.essential
def test_disk_density_radial_factor_uses_cylindrical_radius() -> None:
    """Changing z alone must not change the radial exponential factor."""
    model = _surface_normalized_disk_bulge_halo_model()
    radius = 0.4
    height = 0.8
    normalization = 1.7
    psi_mid = 10.0
    psi_3zd = 9.0

    expected = normalization * disk_surface_density(radius, model)
    midplane = disk_density_psi(
        radius,
        0.0,
        psi_mid,
        psi_mid,
        psi_3zd,
        model,
        vertical_normalization=normalization,
    )
    off_plane = disk_density_psi(
        radius,
        height,
        psi_mid,
        psi_mid,
        psi_3zd,
        model,
        vertical_normalization=normalization,
    )
    assert midplane == pytest.approx(expected, rel=1e-13)
    assert off_plane == pytest.approx(expected, rel=1e-13)

    batch = disk_density_psi_batch(
        np.array([radius, radius]),
        np.array([0.0, height]),
        np.array([psi_mid, psi_mid]),
        np.array([psi_mid, psi_mid]),
        np.array([psi_3zd, psi_3zd]),
        model,
        vertical_normalization=np.array([normalization, normalization]),
    )
    np.testing.assert_allclose(batch, expected, rtol=1e-13, atol=0.0)


@pytest.mark.essential
def test_find_fmax_scans_the_full_velocity_proposal(monkeypatch) -> None:
    """A displaced DF peak must not be hidden by a small local scan."""
    from galacticsics.distribution import diskdf_solve

    vpmax = 5.0
    vsigp = 2.0
    peak_velocity = vpmax - 3.125 * vsigp

    def sharply_peaked_log_df(_vr, vt, _vz, *_args) -> float:
        return -((vt - peak_velocity) / 1.0e-4) ** 2

    monkeypatch.setattr(diskdf_solve, "_log_diskdf5ez", sharply_peaked_log_df)
    selected_value = diskdf_solve._find_fmax(
        vpmax,
        1.0,
        0.0,
        object(),
        object(),
        object(),
        object(),
        object(),
        vsigp,
    )
    assert selected_value == pytest.approx(1.0, rel=1.0e-3)
    selected_value, selected_velocity = diskdf_solve._find_fmax_and_velocity(
        vpmax,
        1.0,
        0.0,
        object(),
        object(),
        object(),
        object(),
        object(),
        vsigp,
    )
    assert selected_value == pytest.approx(1.0, rel=1.0e-3)
    assert selected_velocity == pytest.approx(peak_velocity, abs=1.0e-3)


@pytest.mark.essential
def test_radial_df_correction_does_not_rescale_sigma_r() -> None:
    """GalactICS f_d corrects fnamidden, not the prescribed sigma_R profile."""
    from galacticsics.distribution.diskdf_solve import _sigma_r2

    model = _surface_normalized_disk_bulge_halo_model()
    assert _sigma_r2(1.2, model, 0.2) == pytest.approx(
        _sigma_r2(1.2, model, 1.6),
        rel=0.0,
        abs=0.0,
    )


@pytest.mark.physics_python
@pytest.mark.essential
def test_vertical_dispersion_is_anchored_at_one_scale_height(
    surface_normalized_potential,
) -> None:
    """sigz2.f uses Psi(R,z_d) and sech^2(1), not Psi(R,3 z_d)."""
    from galacticsics.distribution.diskdf_solve import _sigma_z2

    model, potential, work = surface_normalized_potential
    disk = model.disk
    assert disk is not None
    radius = 0.6
    f_sz = 0.7
    expected = (
        evaluate_potential(potential, radius, disk.scale_height)
        - evaluate_potential(potential, radius, 0.0)
    ) / np.log(0.419974) * f_sz
    assert _sigma_z2(radius, potential, disk.scale_height, f_sz) == pytest.approx(
        expected,
        rel=2.0e-13,
    )


@pytest.mark.essential
def test_rcirc_inversion_is_monotone_and_continuous(surface_normalized_potential) -> None:
    from galacticsics.distribution.diskdf_solve import _RcircInterpolator
    from galacticsics.io import read_frequency_table

    _, potential, work = surface_normalized_potential
    frequency = read_frequency_table(work / "freqdbh.dat")
    rcirc = _RcircInterpolator(potential, frequency)
    radii = np.arange(1, potential.model.grid.nr, dtype=float) * potential.model.grid.dr
    angular_momentum = np.array(
        [frequency.omega(float(radius)) * radius**2 for radius in radii]
    )
    recovered = np.array([rcirc(float(am)) for am in angular_momentum])
    np.testing.assert_allclose(recovered, radii, rtol=2.0e-12, atol=2.0e-12)

    am_edge = float(angular_momentum[-1])
    r_edge = float(radii[-1])
    assert rcirc(am_edge * (1.0 + 1.0e-9)) == pytest.approx(
        r_edge * (1.0 + 1.0e-9) ** 2,
        rel=2.0e-12,
    )


@pytest.mark.essential
def test_old_model_json_preserves_legacy_vertical_density_contract() -> None:
    model = _surface_normalized_disk_bulge_halo_model()
    encoded = model_to_dict(model)
    encoded["disk"].pop("vertical_density_mode")
    migrated = model_from_dict(encoded)
    assert migrated.disk is not None
    assert migrated.disk.vertical_density_mode == "legacy_midplane"


@pytest.mark.physics_python
@pytest.mark.essential
def test_vertical_density_integrates_to_requested_surface_density(
    surface_normalized_potential,
) -> None:
    """A bulge-compressed disk must still recover Sigma(R) after z integration."""
    model, potential, work = surface_normalized_potential
    disk = model.disk
    assert disk is not None
    norm_radii, norm_values = disk_vertical_normalization_table(potential, model)
    assert (
        read_harmonic_potential(work / "dbh.dat").model.disk.vertical_density_mode
        == "surface_normalized"
    )

    for radius in (0.15, 0.6, 2.0, 4.0):
        normalization = interpolate_disk_vertical_normalization(
            radius,
            norm_radii,
            norm_values,
        )
        psi_mid = evaluate_potential(potential, radius, 0.0)
        psi_3zd = evaluate_potential(potential, radius, 3.0 * disk.scale_height)

        def density_at_height(height: float) -> float:
            psi = evaluate_potential(potential, radius, height)
            return disk_density_psi(
                radius,
                height,
                psi,
                psi_mid,
                psi_3zd,
                model,
                vertical_normalization=normalization,
            )

        half_column, _integration_error = quad(
            density_at_height,
            0.0,
            30.0 * disk.scale_height,
            points=[
                disk.scale_height,
                2.0 * disk.scale_height,
                3.0 * disk.scale_height,
                5.0 * disk.scale_height,
                8.0 * disk.scale_height,
            ],
            epsabs=1.0e-10,
            epsrel=2.0e-6,
            limit=200,
        )
        measured_column = 2.0 * half_column
        requested_column = disk_surface_density(radius, model)
        # The normalization table is interpolated independently of this
        # adaptive integral; allow its documented sub-cell interpolation error.
        assert measured_column == pytest.approx(requested_column, rel=1.5e-2)


@pytest.mark.physics_python
@pytest.mark.slow
@pytest.mark.skipif(not extension_available(), reason="OpenMP sampler not built")
def test_surface_normalized_openmp_sampler_matches_python(
    surface_normalized_sampler_work: Path,
) -> None:
    """Both samplers must draw the same normalized spatial disk distribution."""
    n_particles = 4_000
    seed = 314159
    python_particles = sample_disk_python(
        surface_normalized_sampler_work,
        n_particles=n_particles,
        seed=seed,
        center=False,
        config=SampleConfig(use_openmp=False),
    )
    openmp_particles = sample_disk_openmp(
        surface_normalized_sampler_work,
        n_particles=n_particles,
        seed=seed,
        center=False,
        n_threads=1,
    )

    python_radius = np.hypot(python_particles.data["x"], python_particles.data["y"])
    openmp_radius = np.hypot(openmp_particles.data["x"], openmp_particles.data["y"])
    python_height = np.abs(python_particles.data["z"])
    openmp_height = np.abs(openmp_particles.data["z"])

    assert ks_2samp(python_radius, openmp_radius).statistic < 0.055
    assert ks_2samp(python_height, openmp_height).statistic < 0.055
    assert np.mean(python_radius < 1.0) == pytest.approx(
        np.mean(openmp_radius < 1.0),
        abs=0.012,
    )
    assert np.median(python_height) == pytest.approx(
        np.median(openmp_height),
        rel=0.12,
    )

    # Parity alone could let both implementations agree on a biased disk.
    # Compare their combined empirical radial CDF with the requested
    # projected profile, including the sampler's finite outer cutoff.
    potential = read_harmonic_potential(surface_normalized_sampler_work / "dbh.dat")
    disk = potential.model.disk
    assert disk is not None
    outer_cutoff = disk.outer_radius + 2.0 * disk.trunc_width
    target_radius = np.linspace(0.0, outer_cutoff, 20_001)
    target_pdf = target_radius * disk_surface_density_array(target_radius, potential.model)
    target_cdf = cumulative_trapezoid(target_pdf, target_radius, initial=0.0)
    target_cdf /= target_cdf[-1]

    def requested_radial_cdf(radius):
        return np.interp(radius, target_radius, target_cdf, left=0.0, right=1.0)

    combined_radius = np.concatenate([python_radius, openmp_radius])
    assert kstest(combined_radius, requested_radial_cdf).statistic < 0.025
    for inner_radius in (0.5, 1.0):
        requested_fraction = float(requested_radial_cdf(inner_radius))
        measured_fraction = float(np.mean(combined_radius < inner_radius))
        binomial_sigma = np.sqrt(
            requested_fraction
            * (1.0 - requested_fraction)
            / combined_radius.size
        )
        assert measured_fraction == pytest.approx(
            requested_fraction,
            abs=3.5 * binomial_sigma,
        )
