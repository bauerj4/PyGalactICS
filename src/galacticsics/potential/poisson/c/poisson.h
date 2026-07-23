#ifndef GALACTICSICS_POISSON_H
#define GALACTICSICS_POISSON_H

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int nr;
    int lmax_active;
    int n_harm;
    int has_disk;
    int has_halo;
    int has_bulge;
    double dr;
    double psic;
    double psi0;
    double halo_dens_at_psi0;
    double bulge_psi0;
    double bulge_psid;
    double bulge_dens_at_psi0;
    const double *apot; /* n_harm * (nr+1), row-major */

    double disk_const;
    double disk_rd;
    double disk_zd;
    double disk_rtrunc;
    double disk_trunc_width;
    double disk_hole_radius;
    double disk_core_radius;

    int npsi_halo;
    const double *halo_energies;
    const double *halo_dens_psi;

    int npsi_bulge;
    const double *bulge_energies;
    const double *bulge_dens_psi;
} PoissonPack;

int fill_polar_density_harmonics_omp(
    double *adens,
    const PoissonPack *pack,
    double radial_step,
    int active_lmax,
    int n_polar_nodes,
    int n_threads
);

#endif
