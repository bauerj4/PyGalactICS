#ifndef GALACTICSICS_SAMPLER_H
#define GALACTICSICS_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    uint64_t state;
} SamplerRng;

typedef struct {
    int nr;
    int n_harm;
    int lmax;
    int has_disk;
    double dr;
    double psic;
    double disk_const;
    double disk_rd;
    double disk_zd;
    double disk_scale_height;
    double disk_rtrunc;
    double disk_trunc_width;
    const double *apot;   /* n_harm * (nr+1), row-major l then r */
    const double *plcon;  /* n_harm */
} PotPack;

typedef struct {
    int n;
    const double *radius;
    const double *omega;
    const double *kappa;
} FreqPack;

/* Maps specific angular momentum L = R v_phi to circular radius (rcirc.f). */
typedef struct {
    int n;
    const double *am;
    const double *inv_sqrt_am; /* 1/sqrt(omega) on the DBH radial grid */
} RcircPack;

typedef struct {
    int n;
    const double *radius;
    const double *f_d;
    const double *f_sz;
    double sigma_r0;
    double sigma_r_scale;
} CorrPack;

typedef struct {
    int n;
    const double *energy;
    const double *log_df;
    /* Optional running max of max(exp(log_df)-fcut,0) on ascending energy; NULL → f(ψ)-fcut. */
    const double *fmax_cum;
    double psic;
    double fcut;
} DfPack;

typedef struct {
    double a;
    double v0;
    double cusp;
    double r_outer;
    double dr_trunc;
} HaloParams;

typedef struct {
    double n;     /* Sersic index */
    double ppp;   /* inner slope */
    double Re;    /* effective radius */
    double butt;  /* truncation parameter */
    double rho0;  /* central density normalization */
} BulgeParams;

typedef struct {
    int n_threads;
    int max_attempts;
    int center;
} SamplerOpts;

void sampler_rng_seed(SamplerRng *rng, int seed, int thread_id);
double sampler_rng_uniform(SamplerRng *rng);
double sampler_rng_range(SamplerRng *rng, double lo, double hi);

double pot_eval(const PotPack *pot, double s, double z);
double halo_density_spherical(const HaloParams *h, double r);
double sersic_density_spherical(const BulgeParams *b, double r);
double df_eval(const DfPack *df, double psi);
double df_fmax_at(const DfPack *df, double psi);

double freq_omega(const FreqPack *f, double r);
double freq_kappa(const FreqPack *f, double r);
double corr_f_d(const CorrPack *c, double r);
double corr_f_sz(const CorrPack *c, double r);
double sigma_r2(const CorrPack *c, double r);
double sigma_z2(const PotPack *pot, const CorrPack *c, double r);
double disk_midplane_density(const PotPack *pot, double r, double z);
double rcirc_from_am(const RcircPack *rc, double am);
double diskdf5ez(
    double vr, double vt, double vz, double r, double z,
    const PotPack *pot, const FreqPack *freq, const CorrPack *corr, const RcircPack *rcirc
);
double find_fmax(
    double vpmax, double r, double z,
    const PotPack *pot, const FreqPack *freq, const CorrPack *corr, const RcircPack *rcirc,
    double vsigp
);

int sample_halo_omp(
    double *out, /* 7 * n_particles */
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
);

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
);

int sample_bulge_omp(
    double *out, /* 7 * n_particles */
    int n_particles,
    int seed,
    double mass,
    double bulgeedge,
    double wmax,
    double wmin,
    double streaming,
    const PotPack *pot,
    const BulgeParams *bulge,
    const DfPack *df,
    const SamplerOpts *opts
);

#endif /* GALACTICSICS_SAMPLER_H */
