/*
 * sampler_impl.c — OpenMP particle samplers (gendisk / genhalo).
 */

#include "sampler.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double erfc_approx(double x) {
    return erfc(x);
}

/* Invert u = -(1+x)exp(-x) (legacy gendisk invu); PDF ∝ x exp(-x). */
static double invu(double u) {
    double rg = 1.0;
    for (int i = 0; i < 20; i++) {
        double e = exp(-rg);
        double f = -(1.0 + rg) * e - u;
        double df = rg * e;
        if (df <= 1e-30) {
            break;
        }
        double rnew = rg - f / df;
        if (fabs(rnew - rg) < 1e-8) {
            return (rnew > 0.0) ? rnew : 0.0;
        }
        rg = rnew;
    }
    return (rg > 0.0) ? rg : 0.0;
}

/* --- RNG (splitmix64-based, per-thread) --- */

void sampler_rng_seed(SamplerRng *rng, int seed, int thread_id) {
    uint64_t s = (uint64_t)(seed == 0 ? 42 : (seed < 0 ? -seed : seed));
    s ^= (uint64_t)(thread_id + 1) * 0x9e3779b97f4a7c15ULL;
    s += 0x9e3779b97f4a7c15ULL;
    s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9ULL;
    s = (s ^ (s >> 27)) * 0x94d049bb133111ebULL;
    rng->state = s ^ (s >> 31);
    if (rng->state == 0) {
        rng->state = 1;
    }
}

static uint64_t rng_next(SamplerRng *rng) {
    uint64_t z = (rng->state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double sampler_rng_uniform(SamplerRng *rng) {
    return (rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

double sampler_rng_range(SamplerRng *rng, double lo, double hi) {
    return lo + (hi - lo) * sampler_rng_uniform(rng);
}

/* --- interpolation --- */

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

double freq_omega(const FreqPack *f, double r) {
    return interp1d(f->radius, f->omega, f->n, r);
}

double freq_kappa(const FreqPack *f, double r) {
    return interp1d(f->radius, f->kappa, f->n, r);
}

double corr_f_d(const CorrPack *c, double r) {
    return interp1d(c->radius, c->f_d, c->n, r);
}

double corr_f_sz(const CorrPack *c, double r) {
    return interp1d(c->radius, c->f_sz, c->n, r);
}

double rcirc_from_am(const RcircPack *rc, double am) {
    double aam = fabs(am);
    if (aam <= 0.0) {
        return 0.0;
    }
    if (rc->n <= 0) {
        return 0.0;
    }
    if (aam > rc->am[rc->n - 1]) {
        double ratio = aam / rc->am[rc->n - 1];
        return rc->inv_sqrt_am[rc->n - 1] * ratio * ratio;
    }
    double inv_sqrt = interp1d(rc->am, rc->inv_sqrt_am, rc->n, aam);
    double rc_val = inv_sqrt * sqrt(aam);
    return (rc_val > 0.0) ? rc_val : 0.0;
}

double df_eval(const DfPack *df, double psi) {
    if (psi <= df->psic) {
        return 0.0;
    }
    double logf = interp1d(df->energy, df->log_df, df->n, psi);
    return exp(logf);
}

/* --- Legendre (even l) --- */

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

static void legendre_even_l(double costheta, int lmax, double *p, double *dp, int n_harm) {
    double sintheta = sqrt(fmax(0.0, 1.0 - costheta * costheta));
    int idx = 0;
    for (int ell = 0; ell <= lmax; ell += 2) {
        p[idx] = legendre_P(ell, costheta);
        if (fabs(sintheta) < 1e-14 || ell == 0) {
            dp[idx] = 0.0;
        } else {
            double p_lm1 = legendre_P(ell - 1, costheta);
            dp[idx] = ell * (costheta * p[idx] - p_lm1) / sintheta;
        }
        idx++;
    }
    (void)n_harm;
}

static double disk_surface_density(const PotPack *pot, double r) {
    if (!pot->has_disk || r <= 0.0) {
        return 0.0;
    }
    /* Match appdisk.f / Python disk_surface_radial_derivatives truncation. */
    double t = sqrt(0.5) * (r - pot->disk_rtrunc) / pot->disk_trunc_width;
    double trunc;
    if (t < -4.0) {
        trunc = 1.0;
    } else if (t > 4.0) {
        return 0.0;
    } else {
        trunc = 0.5 * erfc_approx(t);
    }
    return pot->disk_const * exp(-r / pot->disk_rd) * trunc;
}

static double disk_vertical_log_cosh(double z, double zdisk) {
    if (zdisk <= 0.0) {
        return 0.0;
    }
    double zz = z / zdisk;
    double ax = fabs(zz);
    if (ax > 50.0) {
        return ax;
    }
    return log(cosh(zz));
}

static double approx_disk_potential(const PotPack *pot, double s, double z) {
    if (!pot->has_disk) {
        return 0.0;
    }
    double r = hypot(s, z);
    double f = disk_surface_density(pot, r);
    if (f == 0.0) {
        return 0.0;
    }
    /* Legacy appdiskpot.f / Python approximate_disk_potential (log cosh vertical). */
    double g = disk_vertical_log_cosh(z, pot->disk_zd);
    return -2.0 * M_PI * f * pot->disk_zd * g;
}

double pot_eval(const PotPack *pot, double s, double z) {
    double r = hypot(s, z);
    double psi = 0.0;
    if (r == 0.0) {
        psi = pot->apot[0] / sqrt(4.0 * M_PI);
    } else {
        int ihi = (int)(r / pot->dr) + 1;
        if (ihi < 1) {
            ihi = 1;
        }
        if (ihi > pot->nr) {
            ihi = pot->nr;
        }
        double r1 = pot->dr * (ihi - 1);
        double r2 = pot->dr * ihi;
        double t = (r2 > r1) ? (r - r1) / (r2 - r1) : 0.0;
        double costheta = z / r;
        int n_harm = pot->n_harm;
        double p[16];
        double dp[16];
        int lmaxx = pot->lmax;
        legendre_even_l(costheta, lmaxx, p, dp, n_harm);
        int ihim1 = ihi - 1;
        if (ihim1 < 0) {
            ihim1 = 0;
        }
        for (int i = 0; i < n_harm; i++) {
            double ap_hi = pot->apot[i * (pot->nr + 1) + ihi];
            double ap_lo = pot->apot[i * (pot->nr + 1) + ihim1];
            psi += p[i] * pot->plcon[i] * (t * ap_hi + (1.0 - t) * ap_lo);
        }
    }
    if (pot->has_disk) {
        psi += approx_disk_potential(pot, s, z);
    }
    return psi;
}

double halo_density_spherical(const HaloParams *h, double r) {
    if (r <= 0.0) {
        return 0.0;
    }
    double s = r / h->a;
    double haloconst = pow(2.0, 1.0 - h->cusp) * h->v0 * h->v0 / (4.0 * M_PI * h->a * h->a);
    double rho = haloconst / pow(s, h->cusp) / pow(1.0 + s, 3.0 - h->cusp);
    double t = sqrt(0.5) * (r - h->r_outer) / h->dr_trunc;
    double trunc;
    if (t < -4.0) {
        trunc = 1.0;
    } else if (t > 4.0) {
        trunc = 0.0;
    } else {
        trunc = 0.5 * erfc_approx(t);
    }
    return rho * trunc;
}

double sigma_r2(const CorrPack *c, double r) {
    double fd = corr_f_d(c, r);
    if (fd < 1e-6) {
        fd = 1e-6;
    }
    return c->sigma_r0 * c->sigma_r0 * exp(-r / c->sigma_r_scale) * fd;
}

double sigma_z2(const PotPack *pot, const CorrPack *c, double r) {
    double zdisk = pot->disk_scale_height;
    double psizh = pot_eval(pot, r, 3.0 * zdisk);
    double psi00 = pot_eval(pot, r, 0.0);
    double fsz = corr_f_sz(c, r);
    double base = (psizh - psi00) / log(0.419974);
    double sz2 = base * fsz;
    return (sz2 > 1e-10) ? sz2 : 1e-10;
}

static double disk_density_psi(
    const PotPack *pot, double s, double z, double psi, double psi_mid, double psi_at_3zd
) {
    if (!pot->has_disk) {
        return 0.0;
    }
    double zdisk = pot->disk_scale_height;
    if (fabs(z / zdisk) > 30.0) {
        return 0.0;
    }
    double con;
    if (z == 0.0) {
        con = 1.0;
    } else {
        double dpsizh = psi_mid - psi_at_3zd;
        double dpsi = psi_mid - psi;
        double coeff = (fabs(dpsizh) > 0.0) ? dpsi / dpsizh : 0.0;
        if (coeff > 16.0 || coeff < 0.0) {
            con = 0.0;
        } else {
            con = pow(0.009866, coeff);
        }
    }
    double f = disk_surface_density(pot, hypot(s, z));
    return 0.5 / zdisk * f * con;
}

double disk_midplane_density(const PotPack *pot, double r, double z) {
    double psi = pot_eval(pot, r, z);
    double psi_mid = pot_eval(pot, r, 0.0);
    double zdisk = pot->disk_scale_height;
    double psi_3zd = pot_eval(pot, r, 3.0 * zdisk);
    return disk_density_psi(pot, r, z, psi, psi_mid, psi_3zd);
}

static double diskdf3ez(
    double ep, double am, double ez, double r,
    const PotPack *pot, const FreqPack *freq, const CorrPack *corr, const RcircPack *rcirc
) {
    double rc = rcirc_from_am(rcirc, am);
    if (rc <= 0.0) {
        return 0.0;
    }
    double omega = freq_omega(freq, rc);
    double kappa = freq_kappa(freq, rc);
    if (kappa <= 0.0 || omega <= 0.0) {
        return 0.0;
    }
    double vc = rc * omega;
    double psir0 = pot_eval(pot, rc, 0.0);
    double ec = -psir0 + 0.5 * vc * vc;
    /* Legacy diskdf3ez.f: suppress counter-rotating orbits. */
    if (am < 0.0) {
        double psi00 = pot_eval(pot, 0.0, 0.0);
        ec = -2.0 * psi00 - ec;
    }
    double f_d = corr_f_d(corr, rc);
    double f_sz = corr_f_sz(corr, rc);
    double sr2 = sigma_r2(corr, rc);
    double sz2 = sigma_z2(pot, corr, rc);
    if (sz2 <= 0.0 || sr2 <= 0.0) {
        return 0.0;
    }
    double rho_mid = disk_midplane_density(pot, rc, 0.0) * fmax(f_d, 1e-6);
    double fvert = rho_mid * exp(fmin(-ez / sz2, 700.0)) / sqrt(2.0 * M_PI * sz2);
    double val = omega / (M_PI * kappa) / sr2 * exp(fmin(-(ep - ec) / sr2, 700.0)) * fvert;
    return (val > 0.0) ? val : 0.0;
    (void)r;
}

double diskdf5ez(
    double vr, double vt, double vz, double r, double z,
    const PotPack *pot, const FreqPack *freq, const CorrPack *corr, const RcircPack *rcirc
) {
    double psir0 = pot_eval(pot, r, 0.0);
    double psirz = (z == 0.0) ? psir0 : pot_eval(pot, r, z);
    double ep = 0.5 * (vr * vr + vt * vt) - psir0;
    double am = r * vt;
    double ez = 0.5 * vz * vz - psirz + psir0;
    return diskdf3ez(ep, am, ez, r, pot, freq, corr, rcirc);
}

double find_fmax(
    double vpmax, double r, double z,
    const PotPack *pot, const FreqPack *freq, const CorrPack *corr, const RcircPack *rcirc,
    double vsigp
) {
    double dv = 0.1 * vsigp;
    double v0 = vpmax - dv;
    double v1 = vpmax + dv;
    double f0 = diskdf5ez(0.0, v0, 0.0, r, z, pot, freq, corr, rcirc);
    double fmid = diskdf5ez(0.0, vpmax, 0.0, r, z, pot, freq, corr, rcirc);
    double f1 = diskdf5ez(0.0, v1, 0.0, r, z, pot, freq, corr, rcirc);
    if (fmid >= f0 && fmid >= f1 && fmid > 0.0) {
        return fmid;
    }
    double fmax = f0;
    double vmax = v0;
    for (int i = 0; i <= 100; i++) {
        double v = v0 + i * 2.0 * dv;
        double f = diskdf5ez(0.0, v, 0.0, r, z, pot, freq, corr, rcirc);
        if (f > fmax) {
            fmax = f;
            vmax = v;
        }
    }
    return diskdf5ez(0.0, vmax, 0.0, r, z, pot, freq, corr, rcirc);
}

/* --- OpenMP samplers --- */

static void center_particles(double *out, int n) {
    double msum = 0.0;
    double com[3] = {0.0, 0.0, 0.0};
    double vcom[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < n; i++) {
        double m = out[i * 7 + 0];
        msum += m;
        com[0] += m * out[i * 7 + 1];
        com[1] += m * out[i * 7 + 2];
        com[2] += m * out[i * 7 + 3];
        vcom[0] += m * out[i * 7 + 4];
        vcom[1] += m * out[i * 7 + 5];
        vcom[2] += m * out[i * 7 + 6];
    }
    if (msum <= 0.0) {
        return;
    }
    for (int k = 0; k < 3; k++) {
        com[k] /= msum;
        vcom[k] /= msum;
    }
    for (int i = 0; i < n; i++) {
        out[i * 7 + 1] -= com[0];
        out[i * 7 + 2] -= com[1];
        out[i * 7 + 3] -= com[2];
        out[i * 7 + 4] -= vcom[0];
        out[i * 7 + 5] -= vcom[1];
        out[i * 7 + 6] -= vcom[2];
    }
}

int sample_halo_omp(
    double *out,
    int n_particles,
    int seed,
    double mass,
    double haloedge,
    double rhomax,
    double rhomin,
    double streaming,
    const PotPack *pot,
    const HaloParams *halo,
    const DfPack *df,
    const SamplerOpts *opts
) {
    int n_threads = opts->n_threads;
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
#endif
    int accepted = 0;
    int failed = 0;
    int max_attempts = opts->max_attempts;

#pragma omp parallel
    {
        SamplerRng rng;
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        sampler_rng_seed(&rng, seed, tid);
        int local_attempts = 0;

        while (1) {
            int slot;
#pragma omp atomic read
            slot = accepted;
            if (slot >= n_particles) {
                break;
            }
            if (local_attempts >= max_attempts) {
#pragma omp critical
                {
                    failed = 1;
                }
                break;
            }
            local_attempts++;

            double u1 = haloedge * sampler_rng_uniform(&rng);
            double v1 = M_PI * (sampler_rng_uniform(&rng) * 2.0 - 1.0);
            double r_cyl = u1;
            double z = r_cyl * tan(v1);
            if (fabs(z) > 2.0 * haloedge) {
                continue;
            }
            double rad = hypot(r_cyl, z);
            double rhotst = halo_density_spherical(halo, rad) * (r_cyl * r_cyl + z * z);
            if (rhotst < rhomin || (rhomax - rhomin) * sampler_rng_uniform(&rng) > rhotst) {
                continue;
            }
            double phi = sampler_rng_range(&rng, 0.0, 2.0 * M_PI);
            double x = r_cyl * cos(phi);
            double y = r_cyl * sin(phi);
            double psi = pot_eval(pot, r_cyl, z);
            if (psi < pot->psic) {
                continue;
            }
            double vmax2 = 2.0 * (psi - pot->psic);
            double vmax = sqrt(fmax(vmax2, 0.0));
            double fmax = df_eval(df, psi) - df->fcut;
            if (fmax <= 0.0) {
                fmax = 0.0;
                continue;
            }
            int vel_ok = 0;
            double vR = 0.0, vp = 0.0, vz = 0.0;
            for (int inner = 0; inner < 10000; inner++) {
                double v2 = 1.1 * vmax2;
                while (v2 > vmax2) {
                    vR = 2.0 * vmax * (sampler_rng_uniform(&rng) - 0.5);
                    vp = 2.0 * vmax * (sampler_rng_uniform(&rng) - 0.5);
                    vz = 2.0 * vmax * (sampler_rng_uniform(&rng) - 0.5);
                    v2 = vR * vR + vp * vp + vz * vz;
                }
                double energy = psi - 0.5 * v2;
                double f0 = df_eval(df, energy) - df->fcut;
                if (f0 < 0.0) {
                    f0 = 0.0;
                }
                if (fmax * sampler_rng_uniform(&rng) <= f0) {
                    vel_ok = 1;
                    break;
                }
            }
            if (!vel_ok) {
                continue;
            }
            if (sampler_rng_uniform(&rng) < streaming) {
                vp = fabs(vp);
            } else {
                vp = -fabs(vp);
            }
            double r_cyl_pos = hypot(x, y);
            double vx, vy;
            if (r_cyl_pos > 0.0) {
                double cph = x / r_cyl_pos;
                double sph = y / r_cyl_pos;
                vx = vR * cph - vp * sph;
                vy = vR * sph + vp * cph;
            } else {
                vx = vR;
                vy = vp;
            }

            int idx;
#pragma omp atomic capture
            idx = accepted++;
            if (idx < n_particles) {
                out[idx * 7 + 0] = mass;
                out[idx * 7 + 1] = x;
                out[idx * 7 + 2] = y;
                out[idx * 7 + 3] = z;
                out[idx * 7 + 4] = vx;
                out[idx * 7 + 5] = vy;
                out[idx * 7 + 6] = vz;
            }
        }
    }

    if (failed || accepted < n_particles) {
        return -1;
    }
    if (opts->center) {
        center_particles(out, n_particles);
    }
    return 0;
}

int sample_disk_omp(
    double *out,
    int n_particles,
    int seed,
    double mass,
    double rd,
    double zd,
    double rtrunc,
    double rhomax,
    double rhomin,
    const PotPack *pot,
    const FreqPack *freq,
    const CorrPack *corr,
    const RcircPack *rcirc,
    const SamplerOpts *opts
) {
    int n_threads = opts->n_threads;
#ifdef _OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }
#endif
    int accepted = 0;
    int failed = 0;
    int max_attempts = opts->max_attempts;

#pragma omp parallel
    {
        SamplerRng rng;
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        sampler_rng_seed(&rng, seed, tid);
        int local_attempts = 0;

        while (1) {
            int slot;
#pragma omp atomic read
            slot = accepted;
            if (slot >= n_particles) {
                break;
            }
            if (local_attempts >= max_attempts) {
#pragma omp critical
                {
                    failed = 1;
                }
                break;
            }
            local_attempts++;

            double r_try = 2.0 * rtrunc;
            double z_try = 0.0;
            while (r_try > rtrunc) {
                /* Legacy gendisk: u1 = -ran; R = rd * invu(u1) ⇒ P(R)∝R exp(-R/rd). */
                double u1 = -fmax(sampler_rng_uniform(&rng), 1e-30);
                double v1 = sampler_rng_uniform(&rng) * 2.0 - 1.0;
                r_try = rd * invu(u1);
                z_try = zd * atanh(fmax(fmin(v1, 0.999), -0.999));
            }
            double rhoguess = exp(-r_try / rd) / pow(cosh(z_try / zd), 2.0);
            double rhotst = disk_midplane_density(pot, r_try, z_try) / fmax(rhoguess, 1e-30);
            if (rhotst < rhomin || (rhomax - rhomin) * sampler_rng_uniform(&rng) > rhotst) {
                continue;
            }

            double phi = sampler_rng_range(&rng, 0.0, 2.0 * M_PI);
            double x = r_try * cos(phi);
            double y = r_try * sin(phi);
            double omega = freq_omega(freq, r_try);
            double kappa = freq_kappa(freq, r_try);
            double vphimax = omega * r_try;
            double vsigR = sqrt(sigma_r2(corr, r_try));
            double vsigp = (omega > 0.0) ? kappa / (2.0 * omega) * vsigR : 0.0;
            double vsigz = sqrt(sigma_z2(pot, corr, r_try));
            double fmax = 1.1 * find_fmax(vphimax, r_try, z_try, pot, freq, corr, rcirc, vsigp);
            if (fmax <= 0.0) {
                continue;
            }

            int vel_ok = 0;
            double vR = 0.0, vp = 0.0, vz = 0.0;
            for (int k = 0; k < 200; k++) {
                double gr = 8.0 * (sampler_rng_uniform(&rng) - 0.5);
                double gp = 16.0 * (sampler_rng_uniform(&rng) - 0.5);
                double gz = 8.0 * (sampler_rng_uniform(&rng) - 0.5);
                if (gr * gr / 16.0 + gp * gp / 64.0 + gz * gz / 16.0 > 1.0) {
                    continue;
                }
                vR = vsigR * gr;
                vp = vphimax + vsigp * gp;
                vz = vsigz * gz;
                double f0 = diskdf5ez(vR, vp, vz, r_try, z_try, pot, freq, corr, rcirc);
                if (fmax * sampler_rng_uniform(&rng) <= f0) {
                    vel_ok = 1;
                    break;
                }
            }
            if (!vel_ok) {
                continue;
            }

            /* Cylindrical (vR, vφ, vz) → Cartesian (vx, vy, vz); matches halo path. */
            double r_cyl_pos = hypot(x, y);
            double vx, vy;
            if (r_cyl_pos > 0.0) {
                double cph = x / r_cyl_pos;
                double sph = y / r_cyl_pos;
                vx = vR * cph - vp * sph;
                vy = vR * sph + vp * cph;
            } else {
                vx = vR;
                vy = vp;
            }

            int idx;
#pragma omp atomic capture
            idx = accepted++;
            if (idx < n_particles) {
                out[idx * 7 + 0] = mass;
                out[idx * 7 + 1] = x;
                out[idx * 7 + 2] = y;
                out[idx * 7 + 3] = z_try;
                out[idx * 7 + 4] = vx;
                out[idx * 7 + 5] = vy;
                out[idx * 7 + 6] = vz;
            }
        }
    }

    if (failed || accepted < n_particles) {
        return -1;
    }
    if (opts->center) {
        center_particles(out, n_particles);
    }
    return 0;
}
