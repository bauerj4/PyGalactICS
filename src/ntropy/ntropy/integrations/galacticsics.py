"""
Bridge between **galacticsics** (GalactICS IC generation) and **ntropy** (N-body tests).

This module is the intended production path for integrated tests: build a
:class:`~galacticsics.models.GalaxyModel`, run the legacy ``dbh`` + ``genhalo``
pipeline, convert particles to :class:`~ntropy.particles.ParticleState`, and
evolve with ntropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ntropy.particles import ParticleState

if TYPE_CHECKING:
    from galacticsics.models import GalaxyModel
    from galacticsics.sampling.particles import ParticleSet
    from galacticsics.sampling.sampler import SampleConfig


@dataclass
class GalactICSSampleResult:
    """
    Output of a GalactICS → ntropy sampling workflow.

    Attributes
    ----------
    state : ParticleState
        Combined particle state ready for ntropy.
    work_dir : Path
        Directory containing ``dbh.dat``, ``halo``, etc.
    components : dict of str
        Per-component particle counts.
    model_name : str
        Short label for the galaxy model used.
    """

    state: ParticleState
    work_dir: Path
    components: dict[str, int]
    model_name: str = "galaxy"


def galacticsics_available() -> bool:
    """
    Return True when galacticsics and required legacy binaries are present.

    Checks for ``dbh`` and ``genhalo`` in ``legacy/bin/``.
    """
    try:
        from galacticsics.legacy.paths import require_binary

        require_binary("dbh")
        require_binary("genhalo")
        return True
    except (ImportError, FileNotFoundError):
        return False


def require_galacticsics() -> None:
    """Raise ``ImportError`` or ``FileNotFoundError`` if integration is unavailable."""
    try:
        import galacticsics  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "galacticsics is required for this workflow. "
            "Install from the repo root: pip install -e '.[dev]'"
        ) from exc
    from galacticsics.legacy.paths import require_binary

    require_binary("dbh")
    require_binary("genhalo")


def particle_state_from_galacticsics(
    particle_set: ParticleSet,
    *,
    eps: float = 0.05,
    tag: str | None = None,
) -> ParticleState:
    """
    Convert a galacticsics :class:`~galacticsics.sampling.particles.ParticleSet`
    to ntropy :class:`~ntropy.particles.ParticleState`.

    Parameters
    ----------
    particle_set : ParticleSet
        Component particles from ``genhalo`` / ``gendisk`` / ``genbulge``.
    eps : float
        Gravitational softening length for every particle [kpc].
    tag : str, optional
        Component label stored on ``state.tags``.

    Returns
    -------
    ParticleState
    """
    data = particle_set.data
    tag_value = tag or particle_set.component
    state = ParticleState.from_arrays(
        pos=np.column_stack([data["x"], data["y"], data["z"]]),
        vel=np.column_stack([data["vx"], data["vy"], data["vz"]]),
        mass=data["mass"].copy(),
        eps=eps,
    )
    state.tags = np.full(state.n, tag_value, dtype=object)
    return state


def merge_galacticsics_components(
    particles: dict[str, ParticleSet],
    *,
    eps_by_component: dict[str, float] | None = None,
    default_eps: float = 0.05,
) -> ParticleState:
    """
    Merge multiple galacticsics components into one ntropy state.

    Parameters
    ----------
    particles : dict
        Maps component name to :class:`~galacticsics.sampling.particles.ParticleSet`.
    eps_by_component : dict, optional
        Per-component softening lengths.
    default_eps : float
        Fallback softening when a component is not in ``eps_by_component``.

    Returns
    -------
    ParticleState
        Combined state with ``tags`` set; center of mass removed.
    """
    eps_by_component = eps_by_component or {}
    parts: list[ParticleState] = []
    for name, ps in particles.items():
        eps = eps_by_component.get(name, default_eps)
        parts.append(particle_state_from_galacticsics(ps, eps=eps, tag=name))
    if not parts:
        raise ValueError("No particle components to merge")

    pos = np.vstack([p.pos for p in parts])
    vel = np.vstack([p.vel for p in parts])
    mass = np.concatenate([p.mass for p in parts])
    eps = np.concatenate([p.eps for p in parts])
    tags = np.concatenate([p.tags for p in parts])
    state = ParticleState.from_arrays(pos, vel, mass, eps)
    state.tags = tags
    state.remove_center_of_mass()
    return state


def nfw_halo_model_fast():
    """
    Smaller NFW-only :class:`~galacticsics.models.GalaxyModel` for notebooks/tests.

    Uses ``nr=800``, ``lmax=0``, and a reduced outer radius so ``dbh`` finishes
    in seconds on a laptop (same spirit as
    :meth:`~galacticsics.models.GalaxyModel.reference_disk_halo`).
    """
    require_galacticsics()
    from galacticsics.models import GalaxyModel, NFWHalo, PotentialGrid

    return GalaxyModel(
        halo=NFWHalo(r_outer=60.0, v0=2.0, a=8.0, dr_trunc=8.0, cusp=1.0, enabled=True),
        grid=PotentialGrid(dr=0.1, nr=800, lmax=0),
    )


def sample_galacticsics_halo(
    model: GalaxyModel | None = None,
    *,
    n_particles: int = 256,
    seed: int = -42,
    work_dir: Path | str | None = None,
    eps: float = 0.05,
    solve: bool = True,
    timeout: float | None = 600.0,
    cleanup: bool = False,
) -> GalactICSSampleResult:
    """
    Full GalactICS halo IC pipeline: ``dbh`` → ``genhalo`` → ntropy state.

    Parameters
    ----------
    model : GalaxyModel or None
        Galaxy model.  Defaults to :func:`nfw_halo_model_fast`.
    n_particles : int
        Halo particle count for ``genhalo``.
    seed : int
        Legacy RNG seed (negative integer, ``ran3`` convention).
    work_dir : Path, optional
        Run directory.  Created under a temp folder when ``None``.
    eps : float
        ntropy softening length [kpc].
    solve : bool
        When ``True``, run :func:`~galacticsics.potential.solver.solve_potential`
        before sampling.  Set ``False`` when reusing an existing ``work_dir``
        that already contains ``dbh.dat`` and DF tables.
    timeout : float or None
        ``dbh`` subprocess timeout [s].
    cleanup : bool
        Remove temporary ``work_dir`` when ntropy created it.

    Returns
    -------
    GalactICSSampleResult
    """
    require_galacticsics()
    import tempfile

    from galacticsics.potential.solver import solve_potential
    from galacticsics.sampling.sampler import SampleConfig, sample_galaxy

    model = model or nfw_halo_model_fast()
    owned = work_dir is None
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if owned:
        tmp = tempfile.TemporaryDirectory(prefix="galacticsics_ntropy_")
        work_dir = Path(tmp.name)
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    if solve:
        solve_potential(model, work_dir=work_dir, cleanup=False, timeout=timeout)

    config = SampleConfig(
        n_disk=0,
        n_halo=n_particles,
        n_bulge=0,
        seed_halo=seed,
        run_diskdf=False,
        center=True,
    )
    sample_result = sample_galaxy(
        model,
        config,
        work_dir=work_dir,
        cleanup=False,
    )
    if "halo" not in sample_result.particles:
        raise RuntimeError("genhalo did not produce halo particles")

    state = particle_state_from_galacticsics(
        sample_result.particles["halo"], eps=eps, tag="halo"
    )
    state.remove_center_of_mass()

    if owned and cleanup and tmp is not None:
        tmp.cleanup()

    return GalactICSSampleResult(
        state=state,
        work_dir=work_dir,
        components={"halo": state.n},
        model_name="nfw_halo",
    )


def sample_galacticsics_galaxy(
    model: GalaxyModel,
    config: SampleConfig,
    *,
    work_dir: Path | str | None = None,
    artifact_dir: Path | str | None = None,
    eps_by_component: dict[str, float] | None = None,
    default_eps: float = 0.05,
    solve: bool = True,
    timeout: float | None = 600.0,
) -> GalactICSSampleResult:
    """
    General GalactICS IC pipeline for disk / halo / bulge → ntropy.

    When ``artifact_dir`` is provided (e.g. ``tests/generated/reference``),
    ``solve`` is skipped and precomputed ``dbh.dat`` / DF tables are copied in.

    Parameters
    ----------
    model : GalaxyModel
        Galaxy configuration.
    config : SampleConfig
        Per-component particle counts and seeds.
    work_dir : Path, optional
        Working directory for the legacy samplers.
    artifact_dir : Path, optional
        Pre-solved artifacts (skips ``dbh`` when set).
    eps_by_component : dict, optional
        Per-component ntropy softening.
    default_eps : float
        Default softening length [kpc].
    solve : bool
        Run ``solve_potential`` when ``artifact_dir`` is ``None``.
    timeout : float or None
        ``dbh`` timeout when solving.

    Returns
    -------
    GalactICSSampleResult
    """
    require_galacticsics()
    import tempfile

    from galacticsics.potential.solver import solve_potential
    from galacticsics.sampling.sampler import sample_galaxy

    owned = work_dir is None
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if owned:
        tmp = tempfile.TemporaryDirectory(prefix="galacticsics_ntropy_")
        work_dir = Path(tmp.name)
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = Path(artifact_dir) if artifact_dir is not None else None
    if artifact_path is None and solve:
        solve_potential(model, work_dir=work_dir, cleanup=False, timeout=timeout)

    sample_result = sample_galaxy(
        model,
        config,
        work_dir=work_dir,
        artifact_dir=artifact_path,
        cleanup=False,
    )
    if not sample_result.particles:
        raise RuntimeError("No particles were sampled")

    state = merge_galacticsics_components(
        sample_result.particles,
        eps_by_component=eps_by_component,
        default_eps=default_eps,
    )
    components = {name: len(ps) for name, ps in sample_result.particles.items()}

    return GalactICSSampleResult(
        state=state,
        work_dir=work_dir,
        components=components,
        model_name="galaxy",
    )
