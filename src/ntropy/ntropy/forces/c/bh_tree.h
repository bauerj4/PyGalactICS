/*
 * bh_tree.h — Barnes–Hut octree for ntropy (C API)
 *
 * Flat, index-based octree used by the Python extension ntropy.forces._bh_c.
 * Physics matches ntropy.forces.bhtree (Python reference).
 *
 * Optional optimizations are controlled by BHTreeOpts (all off = legacy behaviour).
 */

#ifndef NTROPY_BH_TREE_H
#define NTROPY_BH_TREE_H

#include <stddef.h>

#define NTROPY_BH_G 1.0
#define NTROPY_BH_MIN_LEAF_SIZE 1e-12
#define NTROPY_BH_MAX_WALK_STACK 256

#define NTROPY_BH_PACK_LEGACY 0
#define NTROPY_BH_PACK_NATIVE 1

typedef struct {
    int fast_inv_r3;
    int squared_opening;
    int iterative_walk;
    int morton_build;
    int borrow_arrays;
    int fast_coincident_check;
    int native_pack;
    int accel_all_fast;
    int simd_leaves;
    int omp_schedule; /* 0=static, 1=guided, 2=dynamic */
} BHTreeOpts;

typedef struct {
    double center[3];
    double com[3];
    double size;
    double mass;
    int is_leaf;
    int child[8];
    int leaf_start;
    int leaf_count;
} BHNode;

typedef struct {
    BHNode *nodes;
    int n_nodes;
    int cap_nodes;
    int *leaf_indices;
    int n_leaf_indices;
    int cap_leaf_indices;
    int n_particles;
    double *pos;   /* row-major layout: pos[3*i + d] == (i, d) */
    double *mass;
    double *eps;
    int owns_particle_arrays;
    BHTreeOpts opts;
} BHTree;

/* Default opts: all optimizations disabled (legacy). */
void bh_tree_opts_default(BHTreeOpts *opts);

/*
 * bh_tree_build — allocate and populate an octree from N particles.
 * Copies pos/mass/eps unless opts->borrow_arrays is set.
 */
int bh_tree_build(
    BHTree *tree,
    const double *pos,
    const double *mass,
    const double *eps,
    int n,
    const BHTreeOpts *opts
);

int bh_tree_from_packed(
    BHTree *tree,
    const BHNode *nodes,
    int n_nodes,
    const int *leaf_indices,
    int n_leaf_indices,
    const double *pos,
    const double *mass,
    const double *eps,
    int n_particles,
    const BHTreeOpts *opts
);

void bh_tree_free(BHTree *tree);

int bh_tree_pack(
    const BHTree *tree,
    BHNode **out_nodes,
    int *out_n_nodes,
    int **out_leaf_indices,
    int *out_n_leaf_indices
);

void bh_tree_accel_one(
    const BHTree *tree,
    int target_index,
    double theta,
    const double *pos,
    const double *eps,
    double acc_out[3]
);

int bh_tree_accel_targets(
    const BHTree *tree,
    const int *target_indices,
    int n_targets,
    double theta,
    const double *pos,
    const double *eps,
    double *acc_out
);

int bh_tree_accel_all(
    const BHTree *tree,
    double theta,
    const double *pos,
    const double *eps,
    double *acc_out
);

#endif /* NTROPY_BH_TREE_H */
