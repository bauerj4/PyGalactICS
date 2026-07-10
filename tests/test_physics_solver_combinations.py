"""
Parametrized Python ``dbh`` solver tests across galaxy component combinations.

Exercises supported halo / disk / bulge layouts across grid resolutions
(``nr``, ``dr``) and multipole orders (``lmax`` up to 20), and verifies that
unsupported component mixes fail with actionable errors.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from galacticsics.io import read_component_masses, read_harmonic_potential
from galacticsics.models import (
    BlackHole,
    ExponentialDisk,
    GalaxyModel,
    GasDisk,
    NFWHalo,
    PotentialGrid,
    Sech2Disk,
    SersicBulge,
)
from galacticsics.physics.backend import PhysicsBackendKind
from galacticsics.potential.evaluate import evaluate_potential
from galacticsics.potential.solver import solve_potential
from galacticsics.sampling.sampler import SampleConfig, ensure_disk_df, sample_galaxy


def _ensure_cordbh_for_sampling(
    model: GalaxyModel,
    work: Path,
    *,
    reference_artifacts_dir: Path | None = None,
) -> None:
    """Run Python diskdf when the grid allows it; else copy reference cordbh for sampling smoke."""
    import shutil

    from galacticsics.campaign.spec import assert_dbh_grid_diskdf_compatible

    try:
        assert_dbh_grid_diskdf_compatible(model)
        ensure_disk_df(model, work, backend=PhysicsBackendKind.PYTHON)
    except ValueError:
        if reference_artifacts_dir is None:
            pytest.skip("grid too coarse for diskdf and no reference artifacts")
        shutil.copy2(reference_artifacts_dir / "cordbh.dat", work / "cordbh.dat")


def _grid(nr: int, dr: float, lmax: int) -> PotentialGrid:
    return PotentialGrid(dr=dr, nr=nr, lmax=lmax)


def _nr_for_high_lmax(lmax: int) -> int:
    """Shrink ``nr`` as ``lmax`` grows; keep even radial bins for quadrature."""
    nr = max(24, 36 - max(0, lmax - 8) // 2)
    if nr % 2 == 1:
        nr += 1
    return nr


def _coarse_grid(*, nr: int = 44, dr: float = 0.18, lmax: int = 2) -> PotentialGrid:
    """Alias kept for readability in component-combination cases."""
    return _grid(nr=nr, dr=dr, lmax=lmax)


def _compact_halo(**kwargs) -> NFWHalo:
    defaults = dict(r_outer=42.0, v0=2.1, a=7.0, dr_trunc=6.0, cusp=1.0, enabled=True)
    defaults.update(kwargs)
    return NFWHalo(**defaults)


def _compact_bulge(**kwargs) -> SersicBulge:
    defaults = dict(n_sersic=4.0, ppp=0.5, v0=1.4, a=0.45, enabled=True)
    defaults.update(kwargs)
    return SersicBulge(**defaults)


def _disk_from_reference() -> ExponentialDisk:
    disk = GalaxyModel.reference_disk_halo().disk
    assert disk is not None
    return disk


@dataclass(frozen=True)
class SolveCase:
    """One supported Python-solver configuration."""

    label: str
    builder: Callable[[], GalaxyModel]
    expect_disk: bool = False
    expect_bulge: bool = False
    expect_halo: bool = True
    halo_tables: bool = True
    bulge_tables: bool = False
    expect_lmax: int | None = None
    expect_nr: int | None = None
    expect_dr: float | None = None
    expect_disk_asymmetry: bool = False


@dataclass(frozen=True)
class RejectCase:
    """Model that must be refused before numerics run."""

    label: str
    builder: Callable[[], GalaxyModel]
    exc_type: type[Exception]
    message: str


SOLVE_CASES: tuple[SolveCase, ...] = (
    SolveCase("halo_only_l0", lambda: replace(GalaxyModel.nfw_halo_only(), grid=_coarse_grid(lmax=0))),
    SolveCase(
        "disk_halo_l0",
        lambda: replace(GalaxyModel.reference_disk_halo(), grid=_coarse_grid(lmax=0)),
        expect_disk=True,
    ),
    SolveCase(
        "disk_halo_l2",
        lambda: replace(GalaxyModel.reference_disk_halo(), grid=_coarse_grid(lmax=2)),
        expect_disk=True,
    ),
    SolveCase(
        "reference_disk_halo",
        lambda: replace(GalaxyModel.reference_disk_halo(), grid=_coarse_grid(nr=48, lmax=2)),
        expect_disk=True,
    ),
    SolveCase(
        "milky_way_disk_halo",
        lambda: replace(GalaxyModel.milky_way_disk_halo(), grid=_coarse_grid(nr=52, lmax=2)),
        expect_disk=True,
    ),
    SolveCase(
        "bulge_halo_l0",
        lambda: GalaxyModel(halo=_compact_halo(), bulge=_compact_bulge(), grid=_coarse_grid(lmax=0)),
        expect_bulge=True,
        bulge_tables=True,
    ),
    SolveCase(
        "bulge_halo_l2",
        lambda: GalaxyModel(halo=_compact_halo(), bulge=_compact_bulge(), grid=_coarse_grid(lmax=2)),
        expect_bulge=True,
        bulge_tables=True,
    ),
    SolveCase(
        "disk_bulge_halo",
        lambda: GalaxyModel(
            halo=_compact_halo(),
            disk=_disk_from_reference(),
            bulge=_compact_bulge(),
            grid=_coarse_grid(lmax=2),
        ),
        expect_disk=True,
        expect_bulge=True,
        bulge_tables=True,
    ),
    SolveCase(
        "nfw_plus_bulge_coarse",
        lambda: replace(GalaxyModel.nfw_plus_bulge(), grid=_coarse_grid(nr=56, lmax=2)),
        expect_bulge=True,
        bulge_tables=True,
    ),
    SolveCase(
        "halo_from_mw_halo_only",
        lambda: replace(
            GalaxyModel.milky_way_disk_halo().with_halo_only(),
            grid=_coarse_grid(lmax=0),
        ),
    ),
    SolveCase(
        "bulge_sersic_n1",
        lambda: GalaxyModel(
            halo=_compact_halo(),
            bulge=_compact_bulge(n_sersic=1.0),
            grid=_coarse_grid(lmax=0),
        ),
        expect_bulge=True,
        bulge_tables=True,
    ),
    SolveCase(
        "bulge_sersic_n6",
        lambda: GalaxyModel(
            halo=_compact_halo(),
            bulge=_compact_bulge(n_sersic=6.0),
            grid=_coarse_grid(lmax=0),
        ),
        expect_bulge=True,
        bulge_tables=True,
    ),
    SolveCase(
        "halo_cusp_0p5",
        lambda: replace(GalaxyModel.nfw_halo_only(), halo=_compact_halo(cusp=0.5), grid=_coarse_grid(lmax=0)),
    ),
    SolveCase(
        "halo_cusp_1p5",
        lambda: replace(GalaxyModel.nfw_halo_only(), halo=_compact_halo(cusp=1.5), grid=_coarse_grid(lmax=0)),
    ),
    SolveCase(
        "halo_v0_low",
        lambda: replace(GalaxyModel.nfw_halo_only(), halo=_compact_halo(v0=1.6), grid=_coarse_grid(lmax=0)),
    ),
    SolveCase(
        "halo_v0_high",
        lambda: replace(GalaxyModel.nfw_halo_only(), halo=_compact_halo(v0=3.2), grid=_coarse_grid(lmax=0)),
    ),
    SolveCase(
        "disk_mass_low",
        lambda: replace(
            GalaxyModel.reference_disk_halo(),
            disk=replace(_disk_from_reference(), mass=8.0),
            grid=_coarse_grid(lmax=2),
        ),
        expect_disk=True,
    ),
    SolveCase(
        "disk_mass_high",
        lambda: replace(
            GalaxyModel.reference_disk_halo(),
            disk=replace(_disk_from_reference(), mass=22.0),
            grid=_coarse_grid(lmax=2),
        ),
        expect_disk=True,
    ),
)

# Grid resolution × multipole order for axisymmetric (disk-bearing) layouts.
_GRID_LMAX_SPECS: tuple[tuple[str, int, float, int, str], ...] = (
    # label_suffix, nr, dr, lmax, layout
    ("ultra_coarse", 32, 0.25, 2, "disk_halo"),
    ("coarse_l4", 44, 0.18, 4, "disk_halo"),
    ("medium_l4", 60, 0.15, 4, "disk_halo"),
    ("fine_l4", 72, 0.12, 4, "disk_halo"),
    ("medium_l6", 48, 0.18, 6, "disk_halo"),
    ("wide_extent_l4", 40, 0.20, 4, "disk_halo"),
    ("coarse_l4", 48, 0.18, 4, "reference_disk_halo"),
    ("medium_l4", 52, 0.18, 4, "milky_way_disk_halo"),
    ("medium_l4", 56, 0.16, 4, "disk_bulge_halo"),
    ("coarse_l4", 48, 0.18, 4, "bulge_halo"),
    ("high_l8", 36, 0.20, 8, "disk_halo"),
    ("high_l12", 32, 0.22, 12, "disk_bulge_halo"),
    ("high_l16", 32, 0.20, 16, "milky_way_disk_halo"),
    ("high_l20", 28, 0.25, 20, "disk_bulge_halo"),
)

# Dedicated ladder for production-scale multipole orders (even degrees 0…lmax).
HIGH_LMAX_VALUES: tuple[int, ...] = (8, 10, 12, 16, 20)


def _layout_model(layout: str, *, nr: int, dr: float, lmax: int) -> GalaxyModel:
    grid = _grid(nr=nr, dr=dr, lmax=lmax)
    if layout == "disk_halo":
        return replace(GalaxyModel.reference_disk_halo(), grid=grid)
    if layout == "reference_disk_halo":
        return replace(GalaxyModel.reference_disk_halo(), grid=grid)
    if layout == "milky_way_disk_halo":
        return replace(GalaxyModel.milky_way_disk_halo(), grid=grid)
    if layout == "disk_bulge_halo":
        return GalaxyModel(
            halo=_compact_halo(),
            disk=_disk_from_reference(),
            bulge=_compact_bulge(),
            grid=grid,
        )
    if layout == "bulge_halo":
        return GalaxyModel(halo=_compact_halo(), bulge=_compact_bulge(), grid=grid)
    raise ValueError(f"unknown layout {layout!r}")


def _grid_lmax_cases() -> tuple[SolveCase, ...]:
    cases: list[SolveCase] = []
    for suffix, nr, dr, lmax, layout in _GRID_LMAX_SPECS:
        expect_disk = layout in ("disk_halo", "reference_disk_halo", "milky_way_disk_halo", "disk_bulge_halo")
        expect_bulge = layout in ("disk_bulge_halo", "bulge_halo")
        label = f"{layout}_{suffix}_nr{nr}_dr{dr}_l{lmax}"
        cases.append(
            SolveCase(
                label,
                lambda lay=layout, n=nr, d=dr, lm=lmax: _layout_model(lay, nr=n, dr=d, lmax=lm),
                expect_disk=expect_disk,
                expect_bulge=expect_bulge,
                bulge_tables=expect_bulge,
                expect_lmax=lmax,
                expect_nr=nr,
                expect_dr=dr,
                expect_disk_asymmetry=expect_disk and lmax >= 2,
            )
        )
    return tuple(cases)


SOLVE_CASES = SOLVE_CASES + _grid_lmax_cases()

REJECT_CASES: tuple[RejectCase, ...] = (
    RejectCase(
        "disk_without_halo",
        lambda: replace(
            GalaxyModel.reference_disk_halo(),
            halo=replace(GalaxyModel.reference_disk_halo().halo, enabled=False),  # type: ignore[arg-type]
            grid=_coarse_grid(lmax=0),
        ),
        ValueError,
        "enabled halo",
    ),
    RejectCase(
        "bulge_without_halo",
        lambda: GalaxyModel(bulge=_compact_bulge(), grid=_coarse_grid(lmax=0)),
        ValueError,
        "enabled halo",
    ),
    RejectCase(
        "halo_only_lmax_positive",
        lambda: replace(GalaxyModel.nfw_halo_only(), grid=_coarse_grid(lmax=2)),
        ValueError,
        "lmax=0",
    ),
    RejectCase(
        "halo_only_lmax6_fine_grid",
        lambda: replace(
            GalaxyModel.nfw_halo_only(),
            grid=_grid(nr=72, dr=0.12, lmax=6),
        ),
        ValueError,
        "lmax=0",
    ),
    RejectCase(
        "halo_only_lmax20",
        lambda: replace(
            GalaxyModel.nfw_halo_only(),
            grid=_grid(nr=28, dr=0.25, lmax=20),
        ),
        ValueError,
        "lmax=0",
    ),
    RejectCase(
        "gas_enabled",
        lambda: replace(
            GalaxyModel.reference_disk_halo(),
            gas=GasDisk(
                mass=2.0,
                scale_length=4.0,
                outer_radius=25.0,
                z_scale=0.12,
                trunc_width=2.0,
                rz_scale=1.0,
                z_max=6.0,
                gamma=1.2,
                enabled=True,
            ),
            grid=_coarse_grid(lmax=2),
        ),
        NotImplementedError,
        "gas",
    ),
    RejectCase(
        "disk2_enabled",
        lambda: replace(
            GalaxyModel.reference_disk_halo(),
            disk2=Sech2Disk(
                mass=1.5,
                scale_length=2.5,
                outer_radius=18.0,
                scale_height=0.35,
                trunc_width=2.0,
                enabled=True,
            ),
            grid=_coarse_grid(lmax=2),
        ),
        NotImplementedError,
        "disk2",
    ),
    RejectCase(
        "black_hole_enabled",
        lambda: replace(
            GalaxyModel.reference_disk_halo(),
            black_hole=BlackHole(mass=0.02, enabled=True),
            grid=_coarse_grid(lmax=2),
        ),
        NotImplementedError,
        "black hole",
    ),
    RejectCase(
        "baryons_without_halo",
        lambda: GalaxyModel.milky_way_disk_halo().with_baryons_only(halo_grid=_coarse_grid(lmax=2)),
        ValueError,
        "enabled halo",
    ),
)


def _python_solve(model: GalaxyModel, work_dir: Path):
    return solve_potential(
        model,
        work_dir=work_dir,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )


def _assert_solve_artifacts(work: Path, case: SolveCase) -> None:
    for name in ("dbh.dat", "h.dat", "mr.dat", "rtidal.dat", "freqdbh.dat"):
        assert (work / name).is_file(), f"missing {name} for {case.label}"
    if case.halo_tables:
        for name in ("dfnfw.dat", "denspsihalo.dat"):
            assert (work / name).is_file(), f"missing {name} for {case.label}"
    if case.bulge_tables:
        for name in ("dfsersic.dat", "denspsibulge.dat"):
            assert (work / name).is_file(), f"missing {name} for {case.label}"

    pot = read_harmonic_potential(work / "dbh.dat")
    assert pot.psi0 > 0.0
    assert pot.flags.halo == case.expect_halo
    assert pot.flags.disk == case.expect_disk
    assert pot.flags.bulge == case.expect_bulge

    if case.expect_lmax is not None:
        assert pot.lmax == case.expect_lmax, f"lmax mismatch for {case.label}"
        n_harm = case.expect_lmax // 2 + 1
        assert pot.n_harmonics == n_harm
        assert pot.apot.shape == (n_harm, pot.nr + 1)
        assert pot.adens.shape == pot.apot.shape
        assert pot.fr.shape == pot.apot.shape
    if case.expect_nr is not None:
        assert pot.nr == case.expect_nr
        if case.expect_dr is not None:
            assert pot.r_edge == pytest.approx(case.expect_nr * case.expect_dr)
    if case.expect_dr is not None:
        assert pot.dr == pytest.approx(case.expect_dr)

    if case.expect_disk_asymmetry:
        psi_mid = evaluate_potential(pot, 2.0, 0.0)
        psi_off = evaluate_potential(pot, 2.0, 0.5)
        assert abs(psi_mid - psi_off) > 1e-4, (
            f"disk should break axisymmetry in evaluated potential for {case.label}"
        )

    for r, z in ((0.0, 0.0), (0.8, 0.0), (2.0, 0.3), (5.0, 1.0)):
        psi = evaluate_potential(pot, r, z)
        assert math.isfinite(psi), f"non-finite Psi at ({r}, {z}) for {case.label}"
        assert psi > 0.0 or r == 0.0

    if not case.expect_disk:
        psi_mid_1 = evaluate_potential(pot, 1.0, 0.0)
        psi_mid_5 = evaluate_potential(pot, 5.0, 0.0)
        assert psi_mid_1 >= psi_mid_5, f"midplane Psi should decrease outward for {case.label}"

    masses = read_component_masses(work / "mr.dat")
    if case.expect_disk:
        assert masses["disk"][0] > 0.0
    else:
        assert masses["disk"][0] == 0.0
    if case.expect_bulge:
        assert masses["bulge"][0] > 0.0
    else:
        assert masses["bulge"][0] == 0.0
    assert masses["halo"][0] > 0.0
    assert masses["halo"][1] > 0.0


@pytest.mark.physics_python
@pytest.mark.parametrize("case", SOLVE_CASES, ids=lambda c: c.label)
def test_python_solve_supported_combination(case: SolveCase, tmp_path: Path) -> None:
    """Python solver succeeds and writes consistent artifacts for each layout."""
    model = case.builder()
    work = tmp_path / case.label
    _python_solve(model, work)
    _assert_solve_artifacts(work, case)


@pytest.mark.physics_python
@pytest.mark.parametrize("lmax", HIGH_LMAX_VALUES)
def test_python_solve_high_lmax_disk_halo(lmax: int, tmp_path: Path) -> None:
    """Disk+halo solves succeed for production-scale multipole orders up to lmax=20."""
    nr = _nr_for_high_lmax(lmax)
    dr = 0.2
    case = SolveCase(
        f"disk_halo_lmax{lmax}",
        lambda lm=lmax, n=nr: replace(
            GalaxyModel.reference_disk_halo(),
            grid=_grid(nr=n, dr=dr, lmax=lm),
        ),
        expect_disk=True,
        expect_lmax=lmax,
        expect_nr=nr,
        expect_dr=dr,
        expect_disk_asymmetry=True,
    )
    work = tmp_path / case.label
    _python_solve(case.builder(), work)
    _assert_solve_artifacts(work, case)

    halo_pot = read_harmonic_potential(work / "h.dat")
    assert halo_pot.lmax == lmax
    assert halo_pot.apot.shape[0] == lmax // 2 + 1


@pytest.mark.physics_python
def test_python_solve_odd_nr_high_lmax_disk_halo(tmp_path: Path) -> None:
    """Odd shell counts remain stable at high lmax (legacy radial Simpson padding)."""
    case = SolveCase(
        "disk_halo_odd_nr_l10",
        lambda: replace(
            GalaxyModel.reference_disk_halo(),
            grid=_grid(nr=35, dr=0.2, lmax=10),
        ),
        expect_disk=True,
        expect_lmax=10,
        expect_nr=35,
        expect_dr=0.2,
        expect_disk_asymmetry=True,
    )
    work = tmp_path / case.label
    _python_solve(case.builder(), work)
    _assert_solve_artifacts(work, case)


@pytest.mark.physics_python
@pytest.mark.parametrize("case", REJECT_CASES, ids=lambda c: c.label)
def test_python_solve_rejects_unsupported_combination(case: RejectCase, tmp_path: Path) -> None:
    """Unsupported mixes fail fast with a clear, stable error."""
    model = case.builder()
    with pytest.raises(case.exc_type, match=re.escape(case.message)):
        _python_solve(model, tmp_path / case.label)


SAMPLE_CASES: tuple[tuple[str, Callable[[], GalaxyModel], SampleConfig], ...] = (
    (
        "halo_only",
        lambda: replace(GalaxyModel.nfw_halo_only(), grid=_coarse_grid(lmax=0)),
        SampleConfig(n_disk=0, n_halo=12, n_bulge=0, run_diskdf=False),
    ),
    (
        "disk_halo",
        lambda: replace(GalaxyModel.reference_disk_halo(), grid=_coarse_grid(lmax=2)),
        SampleConfig(n_disk=12, n_halo=12, n_bulge=0, run_diskdf=False),
    ),
    (
        "bulge_halo",
        lambda: GalaxyModel(halo=_compact_halo(), bulge=_compact_bulge(), grid=_coarse_grid(lmax=0)),
        SampleConfig(n_disk=0, n_halo=12, n_bulge=12, run_diskdf=False),
    ),
    (
        "disk_halo_l4",
        lambda: replace(GalaxyModel.reference_disk_halo(), grid=_grid(nr=60, dr=0.15, lmax=4)),
        SampleConfig(n_disk=10, n_halo=10, n_bulge=0, run_diskdf=False),
    ),
    (
        "disk_halo_l12",
        lambda: replace(
            GalaxyModel.reference_disk_halo(),
            grid=_grid(nr=_nr_for_high_lmax(12), dr=0.2, lmax=12),
        ),
        SampleConfig(n_disk=8, n_halo=8, n_bulge=0, run_diskdf=False),
    ),
    (
        "disk_bulge_halo",
        lambda: GalaxyModel(
            halo=_compact_halo(),
            disk=_disk_from_reference(),
            bulge=_compact_bulge(),
            grid=_coarse_grid(lmax=2),
        ),
        SampleConfig(n_disk=10, n_halo=10, n_bulge=10, run_diskdf=False),
    ),
)


@pytest.mark.physics_python
@pytest.mark.parametrize(
    "label,builder,config",
    SAMPLE_CASES,
    ids=[case[0] for case in SAMPLE_CASES],
)
def test_python_sample_supported_combination(
    label: str,
    builder: Callable[[], GalaxyModel],
    config: SampleConfig,
    tmp_path: Path,
    reference_artifacts_dir: Path,
) -> None:
    """Solve + sample smoke test for each supported particle layout."""
    model = builder()
    work = tmp_path / label
    _python_solve(model, work)
    if config.n_disk > 0:
        _ensure_cordbh_for_sampling(model, work, reference_artifacts_dir=reference_artifacts_dir)
    result = sample_galaxy(
        model,
        config,
        work_dir=work,
        cleanup=False,
        backend=PhysicsBackendKind.PYTHON,
    )
    if config.n_disk > 0:
        assert len(result.particles["disk"]) == config.n_disk
    if config.n_halo > 0:
        assert len(result.particles["halo"]) == config.n_halo
    if config.n_bulge > 0:
        assert len(result.particles["bulge"]) == config.n_bulge


@pytest.mark.physics_python
def test_python_solve_idempotent_on_same_directory(tmp_path: Path) -> None:
    """Re-solving into the same work directory overwrites artifacts cleanly."""
    model = replace(GalaxyModel.reference_disk_halo(), grid=_coarse_grid(lmax=2))
    work = tmp_path / "repeat"
    _python_solve(model, work)
    first_mtime = (work / "dbh.dat").stat().st_mtime_ns
    _python_solve(model, work)
    second_mtime = (work / "dbh.dat").stat().st_mtime_ns
    assert second_mtime >= first_mtime
    _assert_solve_artifacts(
        work,
        SolveCase("disk_halo_l2", lambda: model, expect_disk=True),
    )
