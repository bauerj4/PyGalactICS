"""
Optional long DBH diagnostic tests: potential parity, dynamic stability, literature checks.

Run the full suite with::

    pytest tests/test_dbh_diagnostics.py -m slow -v

Quick parity checks (no ``slow`` marker) run in the default CI subset.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from galacticsics.campaign.benchmarks import (
    diagnose_ic_stability,
    explain_ic_instability,
    ic_looks_stable,
    invalidate_ic_artifacts,
)
from galacticsics.campaign.spec import preview_dbh_model
from galacticsics.distribution.toomre import compute_toomre_q
from galacticsics.io import read_harmonic_potential
from galacticsics.io.formats import cordbh_is_valid, read_disk_correction, read_frequency_table
from galacticsics.models import GalaxyModel
from galacticsics.physics.python_backend import python_ensure_disk_df, python_sample_disk
from galacticsics.potential.evaluate import evaluate_potential
from galacticsics.potential.poisson.appdisk import disk_densestimate
from galacticsics.potential.poisson.densities import (
    disk_density_estimate,
    total_density_harmonic,
)
from galacticsics.potential.poisson.df_tables import build_monopole_estimates
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.sampler import SampleConfig

MW_DBH = Path(__file__).resolve().parents[1] / "models" / "MilkyWay" / "dbh.dat"
# Literature MW dbh.dat central potential (see docs/dbh_python_backend.md).
# models/MilkyWay/dbh.dat is gitignored; use this when the file is absent.
MW_PSI0_LITERATURE = 18.1


@pytest.mark.physics_python
def test_explain_ic_instability_reports_failed_checks() -> None:
    collapsed = {
        "r_max": 200.0,
        "disk_v_median": 0.3,
        "disk_v_max": 12.0,
        "halo_v_median": 5.0,
        "halo_v_max": 6.0,
    }
    msg = explain_ic_instability(diag=collapsed)
    assert "disk median" in msg
    assert "disk v_max" in msg

    good = {
        "r_max": 200.0,
        "disk_v_median": 1.7,
        "disk_v_max": 2.5,
        "halo_v_median": 1.5,
        "halo_v_max": 5.0,
    }
    assert "unknown" in explain_ic_instability(diag=good).lower()


@pytest.mark.physics_python
def test_ic_looks_stable_accepts_single_hot_outlier_at_large_n() -> None:
    """One particle above 6.0 must not fail a healthy 1e6-scale disk (p99.99 gate)."""
    from ntropy.particles import ParticleState

    n = 20_000
    pos = np.zeros((n, 3))
    pos[:, 0] = 5.0
    vel = np.zeros((n, 3))
    vel[:, 1] = 1.48  # circular-ish bulk
    vel[-1] = [6.03, 0.0, 0.0]  # single extreme outlier like the corpus failure
    state = ParticleState(
        pos=pos,
        vel=vel,
        mass=np.ones(n),
        eps=np.full(n, 0.1),
        tags=np.full(n, "disk"),
    )
    diag = diagnose_ic_stability(state)
    assert diag["disk_v_max"] == pytest.approx(6.03, rel=0.01)
    assert diag["disk_v_p9999"] < 6.0
    assert ic_looks_stable(state)


@pytest.mark.physics_python
def test_ic_looks_stable_accepts_marginal_coarse_grid_tail() -> None:
    """N≈20k coarse-grid MW runs can have v_max≈5.1 with healthy median ~1.1."""
    from ntropy.particles import ParticleState

    n = 1000
    speed = 1.147 / np.sqrt(3)
    vel = np.full((n, 3), speed)
    vel[-1] = [5.095, 0.0, 0.0]
    state = ParticleState(
        pos=np.zeros((n, 3)),
        vel=vel,
        mass=np.ones(n),
        eps=np.full(n, 0.1),
        tags=np.full(n, "disk"),
    )
    diag = diagnose_ic_stability(state)
    assert diag["disk_v_median"] == pytest.approx(1.147, rel=0.01)
    assert diag["disk_v_max"] == pytest.approx(5.095, rel=0.01)
    assert ic_looks_stable(state)
    assert "unknown" in explain_ic_instability(diag=diag).lower()


@pytest.mark.physics_python
def test_ic_looks_stable_rejects_hot_tail_ratio() -> None:
    from ntropy.particles import ParticleState

    n = 100
    speed = 1.0 / np.sqrt(3)
    vel = np.full((n, 3), speed)
    vel[-1] = [5.8, 0.0, 0.0]
    state = ParticleState(
        pos=np.zeros((n, 3)),
        vel=vel,
        mass=np.ones(n),
        eps=np.full(n, 0.1),
        tags=np.full(n, "disk"),
    )
    assert not ic_looks_stable(state)
    msg = explain_ic_instability(state=state)
    assert "v_max/v_median" in msg


@pytest.mark.physics_python
def test_invalidate_ic_artifacts_removes_cached_files(tmp_path: Path) -> None:
    work = tmp_path / "model"
    work.mkdir()
    (work / "evolution").mkdir()
    for name in (".done_sample", "cordbh.dat", "disk", "evolution/final.dat"):
        (work / name).write_text("x")
    (work / "dbh.dat").write_text("keep")
    removed = invalidate_ic_artifacts(work)
    assert ".done_sample" in removed
    assert "cordbh.dat" in removed
    assert "disk" in removed
    assert "evolution/final.dat" in removed
    assert (work / "dbh.dat").is_file()


@pytest.mark.physics_python
def test_monopole_psi0_matches_mw_reference(tmp_path: Path) -> None:
    """Python coarse-grid solve should anchor psi0 within 1% of literature MW dbh."""
    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")
    pot = read_harmonic_potential(tmp_path / "dbh.dat")
    ref_psi0 = (
        float(read_harmonic_potential(MW_DBH).psi0)
        if MW_DBH.is_file()
        else MW_PSI0_LITERATURE
    )
    assert pot.psi0 == pytest.approx(ref_psi0, rel=0.01)


@pytest.mark.physics_python
def test_diskdensestimate_differs_from_appdiskdens() -> None:
    """Monopole seeding uses diskdensestimate, not appdiskdens (distinct formulas)."""
    model = GalaxyModel.reference_disk_halo()
    s, z = 2.0, 0.3
    densest = disk_densestimate(s, z, model)
    appdisk = disk_density_estimate(s, z, model)
    assert densest > 0.0
    assert appdisk > 0.0
    assert not math.isclose(densest, appdisk, rel_tol=0.05)


@pytest.mark.physics_python
def test_appdisk_subtraction_keeps_harmonic_density_nonnegative(tmp_path: Path) -> None:
    """totdens - appdisk must not go negative at sampled midplane nodes."""
    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")
    pot = read_harmonic_potential(tmp_path / "dbh.dat")
    disk = model.disk
    assert disk is not None
    for r in (1.0, 2.5 * disk.scale_length, 5.0):
        psi = evaluate_potential(pot, r, 0.0)
        psi_mid = psi
        psi_3zd = evaluate_potential(pot, r, 3 * disk.scale_height)
        rho = total_density_harmonic(
            r,
            0.0,
            psi,
            psi_mid,
            psi_3zd,
            model,
            dens_psi_halo=lambda _e: 0.0,
            dens_psi_bulge=None,
            psic=pot.psic,
        )
        assert rho >= 0.0


@pytest.mark.physics_python
def test_freq_table_sane_after_python_solve(tmp_path: Path) -> None:
    """kappa and omega must be positive at the solar cylinder after tabulation."""
    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")
    freq = read_frequency_table(tmp_path / "freqdbh.dat")
    r = 2.5 * model.disk.scale_length
    assert freq.kappa(r) > 0.05
    assert freq.omega(r) > 0.05


@pytest.mark.physics_python
def test_rotation_curve_order_of_magnitude_vs_reference(
    tmp_path: Path, reference_artifacts_dir: Path
) -> None:
    """Omega(R) at 2.5 scale lengths should match generated reference within 25%."""
    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")
    freq = read_frequency_table(tmp_path / "freqdbh.dat")
    ref = read_frequency_table(reference_artifacts_dir / "freqdbh.dat")
    r = 2.5 * model.disk.scale_length
    assert freq.omega(r) == pytest.approx(ref.omega(r), rel=0.25)


@pytest.mark.physics_python
@pytest.mark.slow
def test_python_diskdf_valid_cordbh_mw_coarse(tmp_path: Path) -> None:
    """End-to-end Python diskdf on MW coarse grid produces literature-like f_d."""
    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    work = tmp_path / "mw"
    solve_potential(model, work_dir=work, cleanup=False, backend="python")
    python_ensure_disk_df(model, work)
    assert cordbh_is_valid(work / "cordbh.dat")
    corr = read_disk_correction(work / "cordbh.dat")
    med = float(np.median(corr.f_d[1:]))
    assert 0.5 < med < 1.2
    q, _ = compute_toomre_q(model, work)
    assert math.isfinite(q)
    assert 0.5 < q < 3.0


@pytest.mark.physics_python
@pytest.mark.slow
def test_toomre_q_target_diskdf_valid_mw_coarse(tmp_path: Path) -> None:
    """Toomre-Q target scaling must not break Python diskdf on the coarse MW grid."""
    from galacticsics.campaign.spec import _apply_patch

    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    model = _apply_patch(model, {"disk_kinematics.toomre_q_target": 1.5})
    work = tmp_path / "toomre_coarse"
    solve_potential(model, work_dir=work, cleanup=False, backend="python")
    python_ensure_disk_df(model, work)
    assert cordbh_is_valid(work / "cordbh.dat")
    corr = read_disk_correction(work / "cordbh.dat")
    med = float(np.median(corr.f_d[1:]))
    assert 0.5 < med < 1.2
    assert float(np.max(corr.f_d[1:])) <= 1.65


@pytest.mark.physics_python
@pytest.mark.slow
def test_python_ic_sampling_stable_mw_coarse_with_toomre(tmp_path: Path) -> None:
    """Notebook walkthrough patch (Toomre Q=1.5) should yield stable disk ICs on coarse grid."""
    from galacticsics.campaign.benchmarks import ic_looks_stable
    from galacticsics.campaign.spec import _apply_patch
    from galacticsics.sampling.particles import ParticleSet
    from ntropy.integrations.galacticsics import merge_galacticsics_components

    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    model = _apply_patch(model, {"disk_kinematics.toomre_q_target": 1.5})
    work = tmp_path / "ic_toomre"
    solve_potential(model, work_dir=work, cleanup=False, backend="python")
    python_ensure_disk_df(model, work)
    config = SampleConfig(n_disk=4000, n_halo=4000, n_bulge=0, seed_disk=42, seed_halo=43)
    python_sample_disk(work, config)
    from galacticsics.physics.python_backend import python_sample_halo

    python_sample_halo(work, config)
    particles = {
        "disk": ParticleSet.from_ascii(work / "disk", component="disk", max_particles=4000),
        "halo": ParticleSet.from_ascii(work / "halo", component="halo", max_particles=4000),
    }
    state = merge_galacticsics_components(particles)
    diag = diagnose_ic_stability(state)
    assert diag["disk_v_median"] > 0.8
    assert diag["disk_v_max"] < 6.0
    assert ic_looks_stable(state, disk_v_max=6.0)


@pytest.mark.physics_python
@pytest.mark.slow
def test_python_ic_sampling_stable_mw_coarse(tmp_path: Path) -> None:
    """Sampled disk ICs should pass ic_looks_stable after full Python pipeline."""
    from galacticsics.sampling.particles import ParticleSet
    from ntropy.integrations.galacticsics import merge_galacticsics_components

    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    work = tmp_path / "ic"
    solve_potential(model, work_dir=work, cleanup=False, backend="python")
    python_ensure_disk_df(model, work)
    config = SampleConfig(n_disk=8000, n_halo=8000, n_bulge=0, seed_disk=42, seed_halo=43)
    python_sample_disk(work, config)
    from galacticsics.physics.python_backend import python_sample_halo

    python_sample_halo(work, config)
    particles = {
        "disk": ParticleSet.from_ascii(work / "disk", component="disk", max_particles=5000),
        "halo": ParticleSet.from_ascii(work / "halo", component="halo", max_particles=5000),
    }
    state = merge_galacticsics_components(particles)
    diag = diagnose_ic_stability(state)
    assert diag["disk_v_max"] < 6.0
    assert diag["disk_v_median"] < 3.0
    assert ic_looks_stable(state, disk_v_max=6.0)


@pytest.mark.physics_python
@pytest.mark.slow
def test_python_solved_density_stable_100_myr(tmp_path: Path) -> None:
    """
    Disk Σ(R) and halo ρ(r) remain stable over ~100 Myr after a Python solve.

    Uses campaign-like ``bh_c`` gravity (theta=0.5, optimized preset) and the
    tiered leapfrog integrator. Skips when the C extension is not built.
    """
    from galacticsics.campaign.analysis import density_drift_metrics
    from galacticsics.physics.python_backend import python_ensure_disk_df, python_sample_halo
    from galacticsics.sampling.particles import ParticleSet
    from galacticsics.sampling.sampler import SampleConfig
    from ntropy.config import BhOptimizationsConfig, ForceConfig, IntegratorConfig, ParallelConfig, RunConfig, TimestepConfig
    from ntropy.forces.bhtree_c import extension_available
    from ntropy.integrations.galacticsics import merge_galacticsics_components
    from ntropy.simulation import Simulation

    if not extension_available():
        pytest.skip("bh_c extension not built")

    target_myr = 100.0
    end_time_gyr = target_myr / 1000.0

    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    work = tmp_path / "evolve"
    solve_potential(model, work_dir=work, cleanup=False, backend="python")
    python_ensure_disk_df(model, work)

    n_particles = 8000
    config = SampleConfig(
        n_disk=n_particles,
        n_halo=n_particles,
        n_bulge=0,
        seed_disk=11,
        seed_halo=12,
    )
    python_sample_disk(work, config)
    python_sample_halo(work, config)
    particles = {
        "disk": ParticleSet.from_ascii(work / "disk", component="disk", max_particles=n_particles),
        "halo": ParticleSet.from_ascii(work / "halo", component="halo", max_particles=n_particles),
    }
    state = merge_galacticsics_components(particles)
    assert ic_looks_stable(state, disk_v_max=6.0)

    cfg = RunConfig()
    cfg.integrator = IntegratorConfig(
        type="tiered_leapfrog",
        order=2,
        dt_base=0.05,
        end_time_gyr=end_time_gyr,
        timestep=TimestepConfig(eta=0.025, dt_base=0.05, max_bin=6, update_every=1),
    )
    cfg.force = ForceConfig(
        method="bh_c",
        theta=0.5,
        rebuild_every=5,
        active_subset=True,
        bh_optimizations=BhOptimizationsConfig.from_preset("optimized"),  # type: ignore[arg-type]
    )
    cfg.parallel = ParallelConfig(enabled=False)
    cfg.output.write_final = False
    cfg.output.every = 0

    result = Simulation(cfg, state=state.copy()).run()
    drift = density_drift_metrics(state, result.final_state)

    # Disk Σ(R) stays within campaign-like bounds; halo shells are noisier at 8k
    # particles on the coarse Python grid (see assert_density_sanity at 0.75).
    assert drift["halo_rho_drift"] < 0.75, (
        f"halo ρ(r) drift {drift['halo_rho_drift']:.3f} over {target_myr:.0f} Myr"
    )
    # Align with campaign assert_density_sanity (0.75); coarse 8k-particle runs
    # routinely sit near ~0.6 under bh_c + tiered leapfrog.
    assert drift["disk_sigma_drift"] < 0.75, (
        f"disk Σ(R) drift {drift['disk_sigma_drift']:.3f} over {target_myr:.0f} Myr"
    )


@pytest.mark.physics_python
@pytest.mark.slow
def test_monopole_estimates_match_solved_psi0(tmp_path: Path) -> None:
    """DF monopole seed psi0 should match converged dbh.dat psi0."""
    model = preview_dbh_model(base="milky_way_disk_halo", coarse=True)
    hpot, _, _, dpot, _, bpot, _ = build_monopole_estimates(model)
    psi0_seed = float(hpot[0] + dpot[0] + bpot[0])
    solve_potential(model, work_dir=tmp_path, cleanup=False, backend="python")
    pot = read_harmonic_potential(tmp_path / "dbh.dat")
    assert pot.psi0 == pytest.approx(psi0_seed, rel=1e-4)
