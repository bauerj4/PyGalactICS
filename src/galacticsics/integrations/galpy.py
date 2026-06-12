"""galpy potential wrapper using in-memory harmonic evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from galacticsics.potential.evaluate import evaluate_force, evaluate_potential
from galacticsics.potential.harmonics import HarmonicPotential

if TYPE_CHECKING:
    import galpy.potential


class GalactICSPotential:
    """Wrap a HarmonicPotential as a galpy-compatible potential.

    Uses direct harmonic evaluation by default. Optional grid interpolation
    via :meth:`from_grid` for legacy potential_out files.
    """

    def __init__(self, harmonic: HarmonicPotential, *, amp: float = 1.0):
        self._harmonic = harmonic
        self._amp = amp
        self._galpy_cls = self._import_galpy_potential()

    @staticmethod
    def _import_galpy_potential():
        try:
            import galpy.potential as gp
        except ImportError as exc:
            raise ImportError(
                "galpy is required for GalactICSPotential; install with pip install galacticsics[galpy]"
            ) from exc
        return gp.Potential

    @classmethod
    def from_harmonic(cls, harmonic: HarmonicPotential, *, amp: float = 1.0) -> GalactICSPotential:
        return cls(harmonic, amp=amp)

    @classmethod
    def from_grid(
        cls,
        path: str,
        *,
        r_ext: float = 30.0,
        z_ext: float = 30.0,
        amp: float = 1.0,
    ) -> GalactICSPotential:
        """Load from potentialgrid ASCII output (legacy potential_out format)."""
        blocks: list[list[tuple[float, float, float, float, float]]] = []
        current: list[tuple[float, float, float, float, float]] = []
        for line in open(path):
            if not line.strip():
                if current:
                    blocks.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) >= 5:
                current.append(tuple(map(float, parts[:5])))
        if current:
            blocks.append(current)

        r_vals = sorted({row[0] for row in blocks[0]})
        z_vals = sorted({row[1] for row in blocks[0]})
        psi = np.zeros((len(z_vals), len(r_vals)))
        fr = np.zeros_like(psi)
        fz = np.zeros_like(psi)
        lookup = {(row[0], row[1]): row for row in blocks[0]}
        for iz, z in enumerate(z_vals):
            for ir, r in enumerate(r_vals):
                row = lookup[(r, z)]
                psi[iz, ir] = row[2]
                fr[iz, ir] = row[3]
                fz[iz, ir] = row[4]

        obj = cls.__new__(cls)
        obj._harmonic = None
        obj._amp = amp
        obj._galpy_cls = cls._import_galpy_potential()
        obj._psi_interp = RegularGridInterpolator((z_vals, r_vals), psi, bounds_error=False, fill_value=None)
        obj._fr_interp = RegularGridInterpolator((z_vals, r_vals), fr, bounds_error=False, fill_value=None)
        obj._fz_interp = RegularGridInterpolator((z_vals, r_vals), fz, bounds_error=False, fill_value=None)
        return obj

    def _evaluate(self, R: float, z: float) -> float:
        if self._harmonic is not None:
            return self._amp * evaluate_potential(self._harmonic, R, z)
        return float(self._psi_interp((z, R)))

    def _rforce(self, R: float, z: float) -> float:
        if self._harmonic is not None:
            fr, _, _ = evaluate_force(self._harmonic, R, z)
            return self._amp * fr
        return float(self._fr_interp((z, R)))

    def _zforce(self, R: float, z: float) -> float:
        if self._harmonic is not None:
            _, fz, _ = evaluate_force(self._harmonic, R, z)
            return self._amp * fz
        return float(self._fz_interp((z, R)))

    def to_galpy(self):
        """Return an anonymous galpy Potential subclass instance."""
        parent = self

        class _GalactICSWrapper(self._galpy_cls):
            _amp = 1.0
            hasC = True

            def _evaluate(self, R, t=0.0):
                return parent._evaluate(float(R))

            def _Rforce(self, R, z=0.0, t=0.0):
                return parent._rforce(float(R), float(z))

            def _zforce(self, R, z=0.0, t=0.0):
                return parent._zforce(float(R), float(z))

        return _GalactICSWrapper()
