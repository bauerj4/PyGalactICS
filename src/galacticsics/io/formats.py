"""Readers and writers for legacy GalactICS data files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import numpy as np

from galacticsics.models import (
    BlackHole,
    ExponentialDisk,
    GasDisk,
    GalaxyModel,
    NFWHalo,
    PotentialGrid,
    Sech2Disk,
    SersicBulge,
)
from galacticsics.potential.harmonics import ComponentFlags, HarmonicPotential

PathLike = Union[str, Path]


def _strip_comment(line: str) -> str:
    """Remove Fortran-style ``#`` comment prefix."""
    s = line.strip()
    if s.startswith("#"):
        s = s[1:].strip()
    return s


def _looks_numeric(content: str) -> bool:
    """True if *content* contains only numeric tokens (no parameter labels)."""
    content = _strip_comment(content)
    if not content:
        return False
    for tok in content.split():
        try:
            float(tok)
        except ValueError:
            return False
    return True


def _parse_floats(line: str) -> list[float]:
    """Extract floating-point tokens from a line, ignoring comment prefixes."""
    line = _strip_comment(line)
    out: list[float] = []
    for tok in line.split():
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


def _parse_ints(line: str) -> list[int]:
    line = _strip_comment(line)
    return [int(float(tok)) for tok in line.split()]


def _parse_harmonic_row(line: str, n_harm: int) -> list[float] | None:
    """Parse one harmonic row; pad missing trailing coefficients with zero."""
    vals = _parse_floats(line)
    if len(vals) < 2:
        return None
    if len(vals) < n_harm + 1:
        vals = vals + [0.0] * (n_harm + 1 - len(vals))
    return vals[: n_harm + 1]


def _read_data_block(lines: list[str], start: int, nr: int, n_harm: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Read nr+1 rows of (radius, harmonic coefficients)."""
    radii = np.zeros(nr + 1, dtype=float)
    coeffs = np.zeros((n_harm, nr + 1), dtype=float)
    idx = start
    count = 0
    while idx < len(lines) and count <= nr:
        vals = _parse_harmonic_row(lines[idx], n_harm)
        if vals is not None:
            radii[count] = vals[0]
            coeffs[:, count] = vals[1 : 1 + n_harm]
            count += 1
        idx += 1
    if count != nr + 1:
        raise ValueError(f"expected {nr + 1} harmonic rows, found {count}")
    return radii, coeffs, idx


def read_harmonic_potential(path: PathLike) -> HarmonicPotential:
    """
    Read a ``dbh.dat`` multipole potential file.

    Parses the Fortran ``readharmfile`` layout: labeled header lines followed by
    three coefficient blocks (``adens``, ``apot``, ``fr``) on radii
    ``r_i = i * dr`` for ``i = 0, …, nr``.

    Parameters
    ----------
    path : path-like
        Path to ``dbh.dat`` (or ``h.dat``; see :func:`read_halo_harmonics`).

    Returns
    -------
    HarmonicPotential
        Parsed model metadata and harmonic arrays.

    Raises
    ------
    ValueError
        If header lines or row counts do not match ``nr`` / ``lmax``.
    """
    text = Path(path).read_text()
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]

    # Skip label lines; data lines follow Fortran readharmfile pattern.
    # Collect non-comment numeric header lines in order.
    header_vals: list[list[float]] = []
    flags: list[int] | None = None
    data_start = 0

    numeric_line_indices: list[int] = []
    for i, line in enumerate(lines):
        content = _strip_comment(line)
        if not content:
            continue
        if not _looks_numeric(content):
            continue
        floats = _parse_floats(content)
        if not floats:
            continue
        if len(floats) == 6 and all(abs(v) <= 1 and v == int(v) for v in floats):
            flags = [int(v) for v in floats]
        else:
            header_vals.append(floats)

    if len(header_vals) < 7:
        raise ValueError(f"unexpected dbh.dat header in {path}")

    (
        chalo,
        v0,
        a,
        nnn,
        v0bulge,
        abulge,
        dr,
        nr,
        lmax,
    ) = (
        header_vals[0][0],
        header_vals[0][1],
        header_vals[0][2],
        header_vals[0][3],
        header_vals[0][4],
        header_vals[0][5],
        header_vals[0][6],
        int(header_vals[0][7]),
        int(header_vals[0][8]),
    )
    psi0, haloconst, bulgeconst = header_vals[1][:3]
    (
        rmdisk,
        rdisk,
        zdisk,
        outdisk,
        drtrunc,
        rhole,
        rcore,
    ) = header_vals[2][:7]
    rmdisk2, rdisk2, zdisk2, outdisk2, drtrunc2 = header_vals[3][:5]
    rmgas, rgas, outgas, zgas0, drtruncgas, rzgas, zgasmax, gamma = header_vals[4][:8]
    psic, psi0_minus_psid, bhmass = header_vals[5][:3]
    psid = psi0 - psi0_minus_psid

    if flags is None:
        flags = [1, 0, 0, 0, 1, 0]

    n_harm = lmax // 2 + 1
    first_data = None
    for i, line in enumerate(lines):
        if "OUTPUT FROM DBH" in line:
            first_data = i + 1
            break
    if first_data is None:
        raise ValueError("could not locate harmonic data block header")
    while first_data < len(lines) and _parse_harmonic_row(lines[first_data], n_harm) is None:
        first_data += 1

    radii, adens, next_idx = _read_data_block(lines, first_data, nr, n_harm)
    while next_idx < len(lines) and _parse_harmonic_row(lines[next_idx], n_harm) is None:
        next_idx += 1
    _, apot, next_idx = _read_data_block(lines, next_idx, nr, n_harm)
    while next_idx < len(lines) and _parse_harmonic_row(lines[next_idx], n_harm) is None:
        next_idx += 1
    _, fr, _ = _read_data_block(lines, next_idx, nr, n_harm)

    model = GalaxyModel(
        halo=NFWHalo(r_outer=chalo, v0=v0, a=a, enabled=bool(flags[4])),
        disk=ExponentialDisk(
            mass=rmdisk,
            scale_length=rdisk,
            outer_radius=outdisk,
            scale_height=zdisk,
            trunc_width=drtrunc,
            hole_radius=rhole,
            core_radius=rcore,
            enabled=bool(flags[0]),
        ),
        disk2=Sech2Disk(
            mass=rmdisk2,
            scale_length=rdisk2,
            outer_radius=outdisk2,
            scale_height=zdisk2,
            trunc_width=drtrunc2,
            enabled=bool(flags[1]),
        ),
        gas=GasDisk(
            mass=rmgas,
            scale_length=rgas,
            outer_radius=outgas,
            z_scale=zgas0,
            trunc_width=drtruncgas,
            rz_scale=rzgas,
            z_max=zgasmax,
            gamma=gamma,
            enabled=bool(flags[2]),
        ),
        bulge=SersicBulge(n_sersic=nnn, ppp=0.0, v0=v0bulge, a=abulge, enabled=bool(flags[3])),
        black_hole=BlackHole(mass=bhmass, enabled=bool(flags[5])),
        grid=PotentialGrid(dr=dr, nr=nr, lmax=lmax),
    )

    return HarmonicPotential(
        model=model,
        psi0=psi0,
        haloconst=haloconst,
        bulgeconst=bulgeconst,
        psic=psic,
        psid=psid,
        flags=ComponentFlags(
            disk=bool(flags[0]),
            disk2=bool(flags[1]),
            gas=bool(flags[2]),
            bulge=bool(flags[3]),
            halo=bool(flags[4]),
            black_hole=bool(flags[5]),
        ),
        radii=radii,
        adens=adens,
        apot=apot,
        fr=fr,
    )


def read_halo_harmonics(path: PathLike) -> HarmonicPotential:
    """
    Read halo-only multipole coefficients from ``h.dat``.

    The file format matches ``dbh.dat`` (see ``legacy/fortran/halopotential.f``),
    but the second and third harmonic blocks store the isolated halo potential
    ``halopot(l, r)`` and radial force ``halofr(l, r)`` rather than the total
    model.  :func:`read_harmonic_potential` parses the same layout; this alias
    documents the intended use in the halo-first workflow.

    Parameters
    ----------
    path : path-like
        Path to ``h.dat`` produced by ``dbh`` when ``ihaloflag=1``.

    Returns
    -------
    HarmonicPotential
        Halo contribution with ``flags.halo=True``.

    See Also
    --------
    merge_harmonic_potentials
    galacticsics.potential.halo_first.solve_halo_potential
    """
    from dataclasses import replace

    pot = read_harmonic_potential(path)
    return replace(pot, flags=replace(pot.flags, halo=True))


def merge_harmonic_potentials(
    halo: HarmonicPotential,
    baryon: HarmonicPotential,
    *,
    model: GalaxyModel | None = None,
) -> HarmonicPotential:
    """
    Sum halo and baryon multipole coefficients on a common radial grid.

    For each even Legendre order :math:`l` and radius index ``ir``:

    .. math::

       \\Phi_l^{\\mathrm{tot}}(r_{ir}) =
       \\Phi_l^{\\mathrm{halo}}(r_{ir}) + \\Phi_l^{\\mathrm{baryon}}(r_{ir})

    and identically for the ``fr`` (radial force) and ``adens`` arrays.  This
    implements the superposition used when disk/bulge are solved in a fixed
    external halo (see README § Two-step workflow).

    Parameters
    ----------
    halo : HarmonicPotential
        Fixed halo harmonics from ``h.dat``.
    baryon : HarmonicPotential
        Baryon-only harmonics from a ``dbh`` run with ``ihaloflag=0``.
    model : GalaxyModel, optional
        Combined model metadata for the output.  Defaults to ``baryon.model``
        with ``halo`` copied from ``halo.model``.

    Returns
    -------
    HarmonicPotential
        Superposed potential suitable for ``write_harmonic_potential(..., "dbh.dat")``.

    Raises
    ------
    ValueError
        If radial grids (``dr``, ``nr``, ``lmax``) differ between inputs.
    """
    g0, g1 = halo.model.grid, baryon.model.grid
    if (g0.dr, g0.nr, g0.lmax) != (g1.dr, g1.nr, g1.lmax):
        raise ValueError(
            f"grid mismatch: halo ({g0.dr}, {g0.nr}, {g0.lmax}) vs "
            f"baryon ({g1.dr}, {g1.nr}, {g1.lmax})"
        )

    from dataclasses import replace

    from galacticsics.potential.harmonics import ComponentFlags

    out_model = model
    if out_model is None:
        out_model = replace(
            baryon.model,
            halo=halo.model.halo,
        )

    flags = ComponentFlags(
        disk=baryon.flags.disk or halo.flags.disk,
        disk2=baryon.flags.disk2 or halo.flags.disk2,
        gas=baryon.flags.gas or halo.flags.gas,
        bulge=baryon.flags.bulge or halo.flags.bulge,
        halo=True,
        black_hole=baryon.flags.black_hole or halo.flags.black_hole,
    )

    return HarmonicPotential(
        model=out_model,
        psi0=halo.psi0,
        haloconst=halo.haloconst,
        bulgeconst=baryon.bulgeconst,
        psic=halo.psic,
        psid=halo.psid,
        flags=flags,
        radii=halo.radii.copy(),
        adens=halo.adens + baryon.adens,
        apot=halo.apot + baryon.apot,
        fr=halo.fr + baryon.fr,
    )


def write_harmonic_potential(potential: HarmonicPotential, path: PathLike) -> None:
    """Write dbh.dat format (compatible with legacy readharmfile)."""
    p = potential
    m = p.model
    g = m.grid
    flags = p.flags
    lines: list[str] = []

    def hdr(label: str, values: str) -> None:
        lines.append(f"# {label}")
        lines.append(f"# {values}")

    hdr(
        "chalo,v0,a,nnn,v0bulge,abulge,dr,nr,lmax=",
        f"{m.halo.r_outer if m.halo else 0:12.5f} {m.halo.v0 if m.halo else 0:12.4f} "
        f"{m.halo.a if m.halo else 0:12.3f} {m.bulge.n_sersic if m.bulge else 0:12.4f} "
        f"{m.bulge.v0 if m.bulge else 0:12.4f} {m.bulge.a if m.bulge else 0:12.4f} "
        f"{g.dr:12.5E} {g.nr:5d} {g.lmax:4d}",
    )
    hdr(
        "psi0, haloconst, bulgeconst:",
        f"{p.psi0:12.6f} {p.haloconst:12.8E} {p.bulgeconst:12.8E}",
    )
    d = m.disk
    hdr(
        "Mdisk, rdisk, zdisk, outdisk, drtrunc",
        f"{d.mass if d else 0:12.5f} {d.scale_length if d else 0:12.4f} "
        f"{d.scale_height if d else 0:12.5f} {d.outer_radius if d else 0:12.5f} "
        f"{d.trunc_width if d else 0:12.4f} {d.hole_radius if d else 0:12.4f} "
        f"{d.core_radius if d else 0:12.4f}",
    )
    d2 = m.disk2
    hdr(
        "Mdisk2, rdisk2, zdisk2, outdisk2, drtrunc2",
        f"{d2.mass if d2 else 0:12.4f} {d2.scale_length if d2 else 0:12.4f} "
        f"{d2.scale_height if d2 else 0:12.4f} {d2.outer_radius if d2 else 0:12.4f} "
        f"{d2.trunc_width if d2 else 0:12.4f}",
    )
    gas = m.gas
    hdr(
        "Mgas, rg, zg, outg, drtruncg,rzg,zgmax,gam",
        f"{gas.mass if gas else 0:12.4f} {gas.scale_length if gas else 0:12.4f} "
        f"{gas.outer_radius if gas else 0:12.4f} {gas.z_scale if gas else 0:12.4f} "
        f"{gas.trunc_width if gas else 0:12.4f} {gas.rz_scale if gas else 0:12.4f} "
        f"{gas.z_max if gas else 0:12.4f} {gas.gamma if gas else 0:12.4f}",
    )
    hdr(
        "psic, psi0-psid, bhmass",
        f"{p.psic:12.4f} {p.psi0 - p.psid:12.5E} {m.black_hole.mass:12.4f}",
    )
    lines.append(
        f"# {int(flags.disk):4d} {int(flags.disk2):4d} {int(flags.gas):4d} "
        f"{int(flags.bulge):4d} {int(flags.halo):4d} {int(flags.black_hole):4d}"
    )
    lines.append("#  OUTPUT FROM DBH8. TOTAL POTENTIAL.")

    n_harm = g.lmax // 2 + 1
    for arr in (p.adens, p.apot, p.fr):
        for ir in range(g.nr + 1):
            vals = " ".join(f"{arr[i, ir]:16.8E}" for i in range(n_harm))
            lines.append(f"  {p.radii[ir]:16.8E} {vals}")

    Path(path).write_text("\n".join(lines) + "\n")


def read_disk_correction(path: PathLike):
    """Read cordbh.dat disk DF correction table."""
    from galacticsics.distribution.diskdf import DiskCorrectionTable

    lines = Path(path).read_text().splitlines()
    header = _parse_floats(_strip_comment(lines[0]))
    if len(header) < 3:
        raise ValueError(f"invalid cordbh header in {path}")
    sigr0, disksr, nrspl = header[0], header[1], int(header[2])
    rr, fdrat, fszrat = [], [], []
    for line in lines[1:]:
        vals = _parse_floats(line)
        if len(vals) >= 3:
            rr.append(vals[0])
            fdrat.append(vals[1])
            fszrat.append(vals[2])
    return DiskCorrectionTable(
        sigma_r0=sigr0,
        sigma_r_scale=disksr,
        radius=np.asarray(rr, dtype=float),
        f_d=np.asarray(fdrat, dtype=float),
        f_sz=np.asarray(fszrat, dtype=float),
    )


def read_frequency_table(path: PathLike):
    """Read freqdbh.dat epicycle frequency table."""
    from galacticsics.distribution.frequencies import FrequencyTable

    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        vals = _parse_floats(line)
        if len(vals) >= 9:
            rows.append(vals[:9])
    data = np.asarray(rows, dtype=float)
    return FrequencyTable(
        radius=data[:, 0],
        omega_h=data[:, 1],
        nu_h=data[:, 2],
        sigma_d=data[:, 3],
        v_circ_total=data[:, 4],
        v_circ_bulge=data[:, 5],
        nu_b=data[:, 6],
        psi_midplane=data[:, 7],
        d2psi_dr2=data[:, 8],
    )


def read_component_masses(path: PathLike) -> dict[str, tuple[float, float]]:
    """Read mr.dat component masses and scale radii."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"mr.dat needs 3 lines, got {len(lines)}")
    dm, dr = _parse_floats(lines[0])[:2]
    bm, br = _parse_floats(lines[1])[:2]
    hm, hr = _parse_floats(lines[2])[:2]
    return {
        "disk": (dm, dr),
        "bulge": (bm, br),
        "halo": (hm, hr),
    }


def read_particles_ascii(path: PathLike, *, max_particles: int | None = None) -> np.ndarray:
    """Read ASCII N-body particle file (mass, x,y,z, vx,vy,vz).

    Skips an optional first-line header ``nobj flag`` written by gendisk.
    """
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if max_particles is not None and len(rows) >= max_particles:
            break
        vals = _parse_floats(line.split("#", 1)[0])
        if len(vals) >= 7:
            rows.append(vals[:7])
        elif i == 0 and len(vals) == 2:
            continue  # particle count header
    from galacticsics.sampling.particles import PARTICLE_DTYPE

    arr = np.zeros(len(rows), dtype=PARTICLE_DTYPE)
    for i, row in enumerate(rows):
        for j, name in enumerate(PARTICLE_DTYPE.names):
            arr[name][i] = row[j]
    return arr


def read_rtidal(path: PathLike) -> float:
    return float(_parse_floats(Path(path).read_text())[0])


def read_toomre_q(path: PathLike) -> float:
    return float(_parse_floats(Path(path).read_text())[0])
