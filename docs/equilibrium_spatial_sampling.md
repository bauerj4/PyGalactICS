# Final-potential spheroid sampling

## The failure mode

The Python Poisson solver converges the halo and bulge as isotropic
distribution functions, so their density in the final disk-plus-spheroid
potential is `rho(Psi(R, z))`. The particle samplers previously used a
different spatial model: analytic spherical NFW and Sérsic profiles. They then
drew velocities from the converged `f(E)` tables. Positions, velocities, and
particle masses therefore did not describe the same equilibrium.

The mass bookkeeping amplified the mismatch. `mr.dat` reported the analytic
halo mass over the radial grid and the nearly complete Sérsic mass, while the
bulge sampler placed that mass only inside `3a`. For the cold-disk model used in
the bar corpus, the old halo mass was `417.60` although the final halo
`rho(Psi)` integrated to `248.97`; the old bulge mass was `1.177` and was
confined to `r < 1.8`, where the analytic enclosed mass is only `0.653`.

The resulting realization was globally subvirial. It contracted and heated
while conserving total energy, producing central rings and rapid vertical
thickening that could be mistaken for physical secular evolution.

## The invariant

For each isotropic spheroid, all three parts of the initial conditions now use
one final-potential contract:

1. The Poisson solve uses `rho(Psi)` from `denspsihalo.dat` or
   `denspsibulge.dat`.
2. The particle mass is the integral of that same field over the solved radial
   domain.
3. Positions are sampled from that same axisymmetric field, and velocities are
   sampled from its corresponding `f(E)`.

The solver writes `halo_spatial_profile.npz` and, when enabled,
`bulge_spatial_profile.npz`. Each profile contains the integrated mass, a dense
`rho(Psi)` interpolation, and a radial rejection proposal. The proposal draws
radius proportional to `r^2 max_mu rho(Psi(r, mu))`, then performs only the
angular rejection. This retains the OpenMP sampler's throughput without
assuming spherical density in the final disk potential.

`mr.dat` now records the profile masses and solved radial edge. The disk entry
records the configured disk mass, which is also the mass used by the disk
sampler.

## Validation

The regression tests require `mr.dat` to match the persisted final-potential
profiles and require independent halo and bulge samples to satisfy
`0.90 <= 2K/|W| <= 1.10` in the converged continuous potential.

On the corpus model that exposed the bug, a 20,000-particle control changed the
continuous-potential virial ratio from `0.925` to `1.008`. A separate
50,000-particle softened pair estimate gave `1.081 +/- 0.001`, compared with
`0.827` in the original 1.75-million-particle realization. The restarted
full-resolution corpus remains the end-to-end morphology check.
