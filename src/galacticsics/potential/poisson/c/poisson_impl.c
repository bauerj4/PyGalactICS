/*
 * poisson_impl.c — OpenMP polar shell integration for the Python Poisson solver.
 */

#include "poisson.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double legendre_P(int n, double x) {
    if (n == 0) {
        return 1.0;
    }
    if (n == 1) {
        return x;
    }
    double p0 = 1.0;
    double p1 = x;
    for (int i = 2; i <= n; i++) {
        double p2 = ((2.0 * i - 1.0) * x * p1 - (i - 1.0) * p0) / i;
        p0 = p1;
        p1 = p2;
    }
    return p1;
}

static double interp1d(const double *x, const double *y, int n, double xq) {
    if (n <= 0) {
        return 0.0;
    }
    if (xq <= x[0]) {
        return y[0];
    }
    if (xq >= x[n - 1]) {
        return y[n - 1];
    }
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (x[mid] <= xq) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    double t = (xq - x[lo]) / (x[hi] - x[lo]);
    return y[lo] * (1.0 - t) + y[hi] * t;
}

static void truncation_factors(
    double r, double outer_radius, double trunc_width, double *eerfc, double *eexp
) {
    double t = sqrt(0.5) * (r - outer_radius) / trunc_width;
    double t2 = t * t;
    if (t < -4.0) {
        *eerfc = 1.0;
        *eexp = 0.0;
        return;
    }
    if (t < 4.0) {
        *eexp = exp(-t2) / sqrt(2.0 * M_PI) / trunc_width;
        *eerfc = 0.5 * erfc(t);
        return;
    }
    *eerfc = 0.0;
    *eexp = 0.0;
}

static void disk_surface_derivatives(
    const PoissonPack *pack, double r, double *f, double *f1r, double *f2
) {
    *f = 0.0;
    *f1r = 0.0;
    *f2 = 0.0;
    if (!pack->has_disk || r <= 0.0) {
        return;
    }
    double eerfc, eexp;
    truncation_factors(r, pack->disk_rtrunc, pack->disk_trunc_width, &eerfc, &eexp);
    if (eerfc == 0.0) {
        return;
    }
    double dc = pack->disk_const;
    double rd = pack->disk_rd;
    double arg1 = -r / rd;
    double sg, sg1, sg2;
    if (pack->disk_hole_radius == 0.0) {
        sg = dc * exp(arg1);
        sg1 = -dc / rd * exp(arg1);
        sg2 = dc / (rd * rd) * exp(arg1);
    } else {
        double tmp2 = sqrt(r * r + pack->disk_hole_radius * pack->disk_hole_radius);
        double arg2 = -tmp2 / pack->disk_core_radius;
        sg = dc * (exp(arg1) - exp(arg2));
        sg1 = dc * (-exp(arg1) / rd + r / pack->disk_core_radius / tmp2 * exp(arg2));
        sg2 = dc
            * (exp(arg1) / (rd * rd)
               + exp(arg2)
                     * (pack->disk_core_radius * pack->disk_hole_radius * pack->disk_hole_radius
                        - r * r * tmp2)
                     / (pack->disk_core_radius * pack->disk_core_radius * tmp2 * tmp2 * tmp2));
    }
    *f = sg * eerfc;
    *f1r = (sg1 * eerfc + eexp * (*f)) / r;
    *f2 = sg2 * eerfc + 2.0 * sg1 * eexp + eexp * ((r - pack->disk_rtrunc) / (pack->disk_trunc_width * pack->disk_trunc_width)) * (*f);
}

static void disk_vertical_derivatives(
    double z, double zdisk, double *g, double *g1, double *g2
) {
    double zz = z / zdisk;
    if (fabs(zz) > 50.0) {
        *g = fabs(zz);
        *g1 = (zz >= 0.0) ? 1.0 : -1.0;
        *g2 = 0.0;
        return;
    }
    double cosh_zz = cosh(zz);
    *g = log(cosh_zz);
    *g1 = tanh(zz);
    *g2 = 1.0 / (cosh_zz * cosh_zz);
}

static double approx_disk_potential(const PoissonPack *pack, double s, double z) {
    if (!pack->has_disk) {
        return 0.0;
    }
    double r = hypot(s, z);
    double f, f1r, f2;
    disk_surface_derivatives(pack, r, &f, &f1r, &f2);
    if (f == 0.0) {
        return 0.0;
    }
    double g, g1, g2;
    disk_vertical_derivatives(z, pack->disk_zd, &g, &g1, &g2);
    return -2.0 * M_PI * f * pack->disk_zd * g;
}

static double approx_disk_density(const PoissonPack *pack, double s, double z) {
    if (!pack->has_disk) {
        return 0.0;
    }
    double r = hypot(s, z);
    double f, f1r, f2;
    disk_surface_derivatives(pack, r, &f, &f1r, &f2);
    double g, g1, g2;
    disk_vertical_derivatives(z, pack->disk_zd, &g, &g1, &g2);
    double h = pack->disk_zd;
    return 0.5 * (f2 * h * g + 2.0 * f1r * g * h + 2.0 * f1r * g1 * z + f * g2 / h);
}

static double disk_surface_sigma(const PoissonPack *pack, double r) {
    if (!pack->has_disk || r <= 0.0) {
        return 0.0;
    }
    double eerfc, eexp;
    truncation_factors(r, pack->disk_rtrunc, pack->disk_trunc_width, &eerfc, &eexp);
    if (eerfc == 0.0) {
        return 0.0;
    }
    return pack->disk_const * exp(-r / pack->disk_rd) * eerfc;
}

static double disk_density_psi(
    const PoissonPack *pack, double s, double z, double psi, double psi_mid, double psi_3zd
) {
    if (!pack->has_disk) {
        return 0.0;
    }
    if (fabs(z / pack->disk_zd) > 30.0) {
        return 0.0;
    }
    double con;
    if (z == 0.0) {
        con = 1.0;
    } else {
        double dpsizh = psi_mid - psi_3zd;
        double dpsi = psi_mid - psi;
        double coeff = (fabs(dpsizh) > 0.0) ? dpsi / dpsizh : 0.0;
        if (coeff > 16.0 || coeff < 0.0) {
            con = 0.0;
        } else {
            con = pow(0.009866, coeff);
        }
    }
    double surface = disk_surface_sigma(pack, hypot(s, z));
    return 0.5 / pack->disk_zd * surface * con;
}

static double halo_density_psi(const PoissonPack *pack, double psi) {
    if (!pack->has_halo || pack->npsi_halo <= 0) {
        return 0.0;
    }
    if (psi < pack->psic) {
        return 0.0;
    }
    if (psi >= pack->psi0) {
        return pack->halo_dens_at_psi0;
    }
    return interp1d(pack->halo_energies, pack->halo_dens_psi, pack->npsi_halo, psi);
}

static double bulge_density_psi(const PoissonPack *pack, double psi) {
    if (!pack->has_bulge || pack->npsi_bulge <= 0) {
        return 0.0;
    }
    if (psi < pack->psic) {
        return 0.0;
    }
    if (psi >= pack->bulge_psi0) {
        return pack->bulge_dens_at_psi0;
    }
    int npsi = pack->npsi_bulge;
    double log_num = log((pack->bulge_psi0 - psi) / fmax(pack->bulge_psi0 - pack->bulge_psid, 1e-30));
    double log_den = log((pack->bulge_psi0 - pack->psic) / fmax(pack->bulge_psi0 - pack->bulge_psid, 1e-30));
    double rj = 1.0 + (double)(npsi - 1) * log_num / log_den;
    int j = (int)rj;
    if (j < 1) {
        j = 1;
    }
    if (j > npsi - 1) {
        j = npsi - 1;
    }
    double frac = rj - (double)j;
    return pack->bulge_dens_psi[j - 1] + frac * (pack->bulge_dens_psi[j] - pack->bulge_dens_psi[j - 1]);
}

static double harmonic_potential(
    const PoissonPack *pack, double s, double z, int include_disk
) {
    double r = hypot(s, z);
    if (r == 0.0) {
        return pack->apot[0] / sqrt(4.0 * M_PI);
    }
    int nr = pack->nr;
    double dr = pack->dr;
    int ihi = (int)floor(r / dr) + 1;
    if (ihi < 1) {
        ihi = 1;
    }
    if (ihi > nr) {
        ihi = nr;
    }
    double r1 = dr * (ihi - 1);
    double r2 = dr * ihi;
    double t = (r2 > r1) ? (r - r1) / (r2 - r1) : 0.0;
    double costheta = z / r;
    double psi = 0.0;
    int li = 0;
    for (int ell = 0; ell <= pack->lmax_active; ell += 2) {
        double pl = legendre_P(ell, costheta);
        double plcon = sqrt((2.0 * ell + 1.0) / (4.0 * M_PI));
        int ir_lo = (ihi - 1 > 0) ? ihi - 1 : 0;
        double apot_hi = pack->apot[li * (nr + 1) + ihi];
        double apot_lo = pack->apot[li * (nr + 1) + ir_lo];
        double apot_i = apot_hi * t + apot_lo * (1.0 - t);
        psi += plcon * pl * apot_i;
        li++;
    }
    if (include_disk && pack->has_disk) {
        psi += approx_disk_potential(pack, s, z);
    }
    return psi;
}

static double simpson_uniform(const double *y, int n, double dx) {
    if (n < 3 || n % 2 == 0) {
        return 0.0;
    }
    double sum = y[0] + y[n - 1];
    for (int i = 1; i < n - 1; i += 2) {
        sum += 4.0 * y[i];
    }
    for (int i = 2; i < n - 2; i += 2) {
        sum += 2.0 * y[i];
    }
    return dx / 3.0 * sum;
}

static void integrate_shell(
    const PoissonPack *pack,
    double shell_radius,
    int ntheta,
    int active_lmax,
    double *moments
) {
    int n_active = active_lmax / 2 + 1;
    for (int li = 0; li < n_active; li++) {
        moments[li] = 0.0;
    }
    if (shell_radius <= 0.0) {
        return;
    }

    double dctheta = 1.0 / (double)(ntheta - 1);
    double *rho = (double *)malloc((size_t)ntheta * sizeof(double));
    double *weighted = (double *)malloc((size_t)ntheta * sizeof(double));
    if (!rho || !weighted) {
        free(rho);
        free(weighted);
        return;
    }

    for (int it = 0; it < ntheta; it++) {
        double ctheta = (double)it * dctheta;
        double z = shell_radius * ctheta;
        double s = shell_radius * sqrt(fmax(0.0, 1.0 - ctheta * ctheta));
        double psi = harmonic_potential(pack, s, z, 1);
        double psi_mid = harmonic_potential(pack, s, 0.0, 1);
        double psi_3zd = harmonic_potential(pack, s, 3.0 * pack->disk_zd, 1);

        double dens = 0.0;
        if (psi >= pack->psic) {
            dens += halo_density_psi(pack, psi);
            dens += bulge_density_psi(pack, psi);
        }
        dens += disk_density_psi(pack, s, z, psi, psi_mid, psi_3zd);
        dens -= approx_disk_density(pack, s, z);
        if (dens < 0.0) {
            dens = 0.0;
        }
        rho[it] = dens;
    }

    int li = 0;
    for (int ell = 0; ell <= active_lmax; ell += 2) {
        double plcon = sqrt((2.0 * ell + 1.0) / (4.0 * M_PI));
        for (int it = 0; it < ntheta; it++) {
            double ctheta = (double)it * dctheta;
            weighted[it] = rho[it] * legendre_P(ell, ctheta) * plcon;
        }
        moments[li] = simpson_uniform(weighted, ntheta, dctheta) * 4.0 * M_PI;
        li++;
    }

    free(rho);
    free(weighted);
}

int fill_polar_density_harmonics_omp(
    double *adens,
    const PoissonPack *pack,
    double radial_step,
    int active_lmax,
    int n_polar_nodes,
    int n_threads
) {
    if (!adens || !pack) {
        return -1;
    }
    int n_active = active_lmax / 2 + 1;
    int ntheta = n_polar_nodes;
    if (ntheta % 2 == 0) {
        ntheta += 1;
    }
    if (ntheta < 3) {
        return -1;
    }

#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
#endif

    int nr = pack->nr;
#pragma omp parallel for schedule(static) if (nr > 32)
    for (int shell_index = 1; shell_index <= nr; shell_index++) {
        double shell_radius = shell_index * radial_step;
        double *moments = (double *)malloc((size_t)n_active * sizeof(double));
        if (!moments) {
            continue;
        }
        integrate_shell(pack, shell_radius, ntheta, active_lmax, moments);
        for (int li = 0; li < n_active; li++) {
            adens[li * (nr + 1) + shell_index] = moments[li];
        }
        free(moments);
    }
    return 0;
}
