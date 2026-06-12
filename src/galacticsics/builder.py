"""High-level pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from galacticsics.distribution.diskdf import DiskCorrectionTable
from galacticsics.distribution.frequencies import FrequencyTable
from galacticsics.io import read_disk_correction, read_frequency_table, read_harmonic_potential
from galacticsics.models import GalaxyModel
from galacticsics.potential.harmonics import HarmonicPotential
from galacticsics.potential.halo_first import (
    BaryonsInFixedHaloResult,
    HaloFirstStepResult,
    run_halo_first_workflow,
    solve_baryons_in_fixed_halo,
    solve_halo_potential,
)
from galacticsics.sampling.particles import ParticleSet


@dataclass
class GalaxyBuilder:
    """
    Orchestrate potential solve, DF correction, and sampling.

    Supports both the **single-pass** self-consistent solve (all components in
    one ``dbh`` run) and the **halo-first** two-step workflow (halo potential
    fixed before baryons are added).  See README § Two-step halo-first workflow.
    """

    model: GalaxyModel
    model_dir: Optional[str] = None
    potential: Optional[HarmonicPotential] = None
    halo_potential: Optional[HarmonicPotential] = None
    disk_correction: Optional[DiskCorrectionTable] = None
    frequencies: Optional[FrequencyTable] = None
    particles: dict[str, ParticleSet] = field(default_factory=dict)
    halo_work_dir: Optional[Path] = None

    def load_artifacts(self) -> GalaxyBuilder:
        """
        Load ``dbh.dat``, ``cordbh.dat``, and ``freqdbh.dat`` from ``model_dir``.

        Returns
        -------
        GalaxyBuilder
            ``self`` with ``potential``, ``disk_correction``, and
            ``frequencies`` populated when files exist.
        """
        if self.model_dir is None:
            raise ValueError("model_dir required to load artifacts")
        root = Path(self.model_dir)
        self.potential = read_harmonic_potential(root / "dbh.dat")
        if (root / "h.dat").exists():
            from galacticsics.io.formats import read_halo_harmonics

            self.halo_potential = read_halo_harmonics(root / "h.dat")
        if (root / "cordbh.dat").exists():
            self.disk_correction = read_disk_correction(root / "cordbh.dat")
        if (root / "freqdbh.dat").exists():
            self.frequencies = read_frequency_table(root / "freqdbh.dat")
        return self

    def load_particles(self) -> GalaxyBuilder:
        """
        Load pre-sampled ``disk``, ``halo``, ``bulge``, or ``gasdisk`` files.

        Returns
        -------
        GalaxyBuilder
            ``self`` with ``particles`` keyed by component name.
        """
        if self.model_dir is None:
            raise ValueError("model_dir required")
        root = Path(self.model_dir)
        for name in ("disk", "halo", "bulge", "gasdisk", "Xhalo"):
            path = root / name
            if path.exists():
                comp = "halo" if name == "Xhalo" else name
                self.particles[comp] = ParticleSet.from_ascii(path, component=comp)
        return self

    def solve_potential(
        self,
        *,
        work_dir: str | None = None,
        cleanup: bool = True,
    ) -> HarmonicPotential:
        """
        Solve the self-consistent multipole potential via legacy ``dbh``.

        All enabled components (halo, disk, bulge, gas) are iterated together.
        For a fixed halo from particles or a prior solve, use
        :meth:`solve_halo_first` and :meth:`solve_baryons_in_fixed_halo`.

        Parameters
        ----------
        work_dir : str, optional
            Working directory for the Fortran solver.
        cleanup : bool, optional
            Remove temporary work directory after success.

        Returns
        -------
        HarmonicPotential
            Stored on ``self.potential``.
        """
        from galacticsics.potential.solver import solve_potential

        result = solve_potential(
            self.model,
            work_dir=work_dir,
            cleanup=cleanup,
        )
        self.potential = result.potential
        return self.potential

    def solve_halo_first(
        self,
        *,
        work_dir: str | None = None,
        cleanup: bool = False,
    ) -> HaloFirstStepResult:
        """
        Step 1 of the halo-first workflow: halo-only ``dbh`` → ``h.dat``.

        Parameters
        ----------
        work_dir : str, optional
            Directory for halo solve artifacts.
        cleanup : bool, optional
            Default ``False`` so ``h.dat`` remains for step 2.

        Returns
        -------
        HaloFirstStepResult
            Sets ``self.halo_potential`` and ``self.halo_work_dir``.
        """
        result = solve_halo_potential(
            self.model,
            work_dir=work_dir,
            cleanup=cleanup,
        )
        self.halo_potential = result.halo_potential
        self.potential = result.total_potential
        self.halo_work_dir = result.work_dir
        return result

    def solve_baryons_in_fixed_halo(
        self,
        *,
        halo_work_dir: str | Path | None = None,
        work_dir: str | None = None,
        cleanup: bool = False,
        run_getfreqs: bool = True,
    ) -> BaryonsInFixedHaloResult:
        """
        Step 2: solve baryons with halo fixed via ``h.dat``.

        Parameters
        ----------
        halo_work_dir : path-like, optional
            Step 1 directory.  Uses ``self.halo_work_dir`` when omitted.
        work_dir, cleanup, run_getfreqs
            Passed to :func:`~galacticsics.potential.halo_first.solve_baryons_in_fixed_halo`.

        Returns
        -------
        BaryonsInFixedHaloResult
            Merged potential on ``self.potential``.
        """
        hdir = Path(halo_work_dir) if halo_work_dir is not None else self.halo_work_dir
        if hdir is None:
            raise ValueError("solve_halo_first() or halo_work_dir required")
        result = solve_baryons_in_fixed_halo(
            self.model,
            hdir,
            work_dir=work_dir,
            cleanup=cleanup,
            run_getfreqs=run_getfreqs,
        )
        self.potential = result.potential
        self.halo_potential = result.halo_potential
        return result

    def run_halo_first_workflow(
        self,
        *,
        work_dir: str | None = None,
        cleanup: bool = False,
    ) -> BaryonsInFixedHaloResult:
        """
        Run both halo-first steps and store the merged potential.

        Parameters
        ----------
        work_dir : str, optional
            Parent directory (contains ``halo/`` and ``baryons/``).
        cleanup : bool, optional
            Default ``False`` to preserve artifacts.

        Returns
        -------
        BaryonsInFixedHaloResult
        """
        result = run_halo_first_workflow(
            self.model,
            work_dir=work_dir,
            cleanup=cleanup,
        )
        self.potential = result.potential
        self.halo_potential = result.halo_potential
        self.halo_work_dir = result.work_dir.parent / "halo"
        return result

    def solve_disk_df(self) -> DiskCorrectionTable:
        """Run legacy ``diskdf`` (requires ``dbh.dat`` and ``h.dat``)."""
        from galacticsics.legacy.runner import LegacyRunner
        from galacticsics.sampling.sampler import ensure_disk_df

        if self.model_dir is None:
            raise ValueError("model_dir required for solve_disk_df")
        root = Path(self.model_dir)
        runner = LegacyRunner(root)
        ensure_disk_df(self.model, root, runner)
        self.disk_correction = read_disk_correction(root / "cordbh.dat")
        return self.disk_correction

    def sample(
        self,
        *,
        n_disk: int = 10_000,
        n_halo: int = 50_000,
        n_bulge: int = 0,
        seed: int = -1,
        center: bool = True,
        work_dir: str | None = None,
        cleanup: bool = True,
        external_halo_path: str | Path | None = None,
    ) -> dict[str, ParticleSet]:
        """
        Sample N-body particles via legacy ``gendisk`` / ``genhalo`` / ``genbulge``.

        Parameters
        ----------
        n_disk, n_halo, n_bulge : int
            Particle counts (zero skips a component).
        seed : int
            RNG seed passed to all components.
        center : bool
            Center each component on the origin.
        work_dir : str, optional
            Run directory; uses a temp dir if omitted.
        cleanup : bool
            Delete temporary work directory after sampling.
        external_halo_path : path-like, optional
            If set, load halo particles from this file (e.g. ``Xhalo``) instead
            of calling ``genhalo``.  Use with the halo-first workflow when the
            halo is specified by an existing N-body set.

        Returns
        -------
        dict[str, ParticleSet]
            Keys ``"disk"``, ``"halo"``, ``"bulge"`` for components sampled.
        """
        from galacticsics.io.formats import read_particles_ascii
        from galacticsics.sampling.sampler import SampleConfig, sample_galaxy

        artifact = Path(self.model_dir) if self.model_dir else None
        if artifact is not None and not (artifact / "dbh.dat").is_file():
            artifact = None
        if self.potential is None and artifact is None:
            raise ValueError("load_artifacts() or solve_potential() required before sample()")

        config = SampleConfig(
            n_disk=n_disk,
            n_halo=0 if external_halo_path else n_halo,
            n_bulge=n_bulge,
            seed_disk=seed,
            seed_halo=seed,
            seed_bulge=seed,
            center=center,
        )
        result = sample_galaxy(
            self.model,
            config,
            work_dir=Path(work_dir) if work_dir else None,
            artifact_dir=artifact,
            cleanup=cleanup,
        )
        self.particles = result.particles
        if external_halo_path is not None:
            data = read_particles_ascii(external_halo_path)
            self.particles["halo"] = ParticleSet(data, component="halo")
        return self.particles
