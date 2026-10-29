# Equilibrium consistency regressions

The corpus rerun exposed two independent cases where an artifact advertised
the requested configuration even though the numerical calculation had used a
different one. Both defects can drive an immediate transient that is easy to
mistake for physical disk evolution.

## Multipole ramp stalled at the monopole

The Poisson solver starts at `lmax_active = 0` and raises the active harmonic
degree after the halo cutoff radius stabilizes. When the cutoff crossing was
outside the finite radial grid, no tidal radius could be measured. The solver
represented that absence as a perpetually unconverged radius, so it never
advanced beyond the monopole. It nevertheless wrote `dbh.dat` with the
configured `lmax` in the header and zero coefficients in every higher-order
row.

For a disk model this is a serious mismatch. The disk DF is then constructed
in a sphericalized radial field, while the N-body realization immediately
feels the flattened particle disk. In the affected cold-disk runs, the exact
initial particle force was 34--52 percent stronger than the smooth solved
radial force at 0.5--1 kpc. The result was an inner contraction, outer
expansion, and vertical heating during the first snapshots even though angular
momentum, energy, and the integrator diagnostics were healthy.

An absent cutoff crossing is now treated as an inapplicable convergence
diagnostic, not as a reason to block the harmonic ramp. Convergence at the
requested degree also requires the potential coefficients themselves to stop
changing. A disk solve with `lmax > 0` refuses to write an all-monopole result,
and the regression test inspects the stored higher-order coefficients and
compares the potential at two angles on the same spherical shell.

## Toomre-Q adjustment used stale disk kinematics

The optional Toomre-Q target rescales `sigma_r0` after `dbh.dat` has been
written. `cordbh.dat` correctly recorded the rescaled value, and the OpenMP
sampler used it. The Python disk-DF correction and Python sampler, however,
evaluated parts of the DF through the model embedded in `dbh.dat`, which still
contained the pre-adjustment value. The correction table could therefore be
solved with one radial-dispersion law and sampled with another; Python and
OpenMP sampling also disagreed.

The current disk kinematics are now bound explicitly before every Python DF
solve, sample, or validation calculation. During the correction solve they
come from the Q-adjusted model; after the table has been written they come from
the `cordbh.dat` header, which is the same source already used by the OpenMP
sampler. A regression test verifies that radial-dispersion evaluation follows
the bound correction-table value instead of the stale potential header.

## Why the original tests missed this

The initial tests checked file production, configured `lmax`, and potential
values at `(R, z) = (2, 0)` and `(2, 0.5)`. The last comparison changed the
spherical radius, so even a pure monopole passed it. No test inspected the
higher-order coefficient rows. The Q-target tests checked the value written to
`cordbh.dat`, but not the kinematics used inside the correction integral.

The new assertions target those internal contracts directly. A nominal
multipole file must contain nonzero higher-order coefficients for a disk, and
all disk-DF paths must evaluate the same dispersion law recorded in the
correction table.
