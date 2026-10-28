# Disk column normalization and central-ring artifact

## Symptom

Disk initial conditions generated in a combined disk, bulge, and halo potential
could show a depleted center surrounded by a ring even though the requested
surface-density profile was monotonic.

## Density contract

The input profile

\[
\Sigma(R)=\Sigma_0 e^{-R/R_d}T(R)
\]

is a projected surface density. The corresponding three-dimensional density
must therefore satisfy

\[
\int_{-\infty}^{\infty}\rho(R,z)\,dz=\Sigma(R)
\]

at every cylindrical radius.

GalactICS uses a potential-dependent vertical factor

\[
C(R,z)=0.009866^{
  [\Psi(R,0)-\Psi(R,z)]/
  [\Psi(R,0)-\Psi(R,3z_d)]
}.
\]

The old Python path evaluated the radial profile at
\(\sqrt{R^2+z^2}\) and multiplied by the fixed factor \(1/(2z_d)\).
Those are separate errors:

- the exponential disk is a function of cylindrical \(R\), not spherical
  distance; and
- \(\int C\,dz=2z_d\) only for the exact reference
  \(\operatorname{sech}^2(z/z_d)\) shape. A bulge or halo changes that
  shape, especially near the center.

The corrected density is

\[
\rho(R,z)=\Sigma(R)N(R)C(R,z),\qquad
N(R)=\left[\int C(R,z)\,dz\right]^{-1}.
\]

Here “vertical normalization” means normalizing each vertical column. It does
not rescale particle heights or adjust the disk mass after sampling.

## Coupled sampling issue

Correcting the analytic density is not sufficient if rejection sampling can
silently bias the accepted positions. The previous disk sampler estimated its
spatial envelope only on the midplane. It could also discard an already
accepted \((R,z)\) after a short, radius-dependent velocity rejection loop.
Several discrepancies from the native GalactICS DF made those failures more
common.

The corrected path scans the full spatial proposal, fails closed if either
rejection envelope is exceeded, and searches for the local azimuthal-velocity
mode before drawing velocities. Python and OpenMP use the same normalization
table and DF conventions.

## Compatibility

New Python-backend models use the projected-column contract. Harmonic files
record the selected density mode. Existing model JSON and harmonic files that
lack this metadata are read as `legacy_midplane`, preserving the original
fixed \(1/(2z_d)\) prefactor. This is not a bit-for-bit replay mode: the
cylindrical-radius correction and the consistent Gaussian-width interpretation
of the truncation parameter apply in both modes. The legacy Fortran backend
rejects `surface_normalized` models because it does not implement this fixed
point.

The normalization is recomputed during the Poisson iteration and convergence
requires both the tidal radius and the disk-column factor to stabilize.

## Why the earlier tests missed it

The earlier unit tests checked individual density and sampling operations, but
did not integrate the three-dimensional disk density over height in a solved
disk, bulge, and halo potential. Some sampling fixtures could also reuse a
`cordbh.dat` correction table generated for a different model. Those tests
could therefore pass while the full initial-condition pipeline produced a
central ring.

The regression coverage now verifies the column integral at several radii,
generates each correction table from the same model under test, compares the
Python and OpenMP samplers, and checks the sampled radial distribution against
the requested projected profile.
