/*
 * bh_tree.h — Barnes–Hut octree for ntropy (C API)
 *
 * Flat, index-based octree used by the Python extension ntropy.forces._bh_c.
 * Physics matches ntropy.forces.bhtree (Python reference):
 *
 *   - Monopole cell approximation when (size / distance) < theta
 *   - Leaf cells: pairwise Plummer sum with h_ij = 0.5 * (eps_i + eps_j)
 *   - Opened cells: monopole with h = eps_target
 *   - Self-interactions excluded; G = NTROPY_BH_G (1.0)
 *
 * Memory model
 * ------------
 * BHTree stores nodes in a growable array (indices, not pointers) so realloc
 * never invalidates parent/child links.  bh_tree_build copies pos/mass/eps
 * into owned buffers (owns_particle_arrays = 1).  bh_tree_from_packed binds
 * external particle arrays for the MPI unpack path.
 *
 * MPI phase 1 (see PARALLEL.md)
 * -----------------------------
 *   rank 0: bh_tree_build → pack via Python pack_buffers → MPI_Bcast
 *   all ranks: bh_tree_from_packed → bh_tree_accel_targets(local_targets)
 *
 * Packed node row (19 doubles, NODE_PACK_WIDTH in bh_module.c)
 * ------------------------------------------------------------
 *   [0:3] center, [3:6] com, [6] size, [7] mass, [8] is_leaf,
 *   [9:17] child[8], [17] leaf_start, [18] leaf_count
 */

#ifndef NTROPY_BH_TREE_H
#define NTROPY_BH_TREE_H

#include <stddef.h>

#define NTROPY_BH_G 1.0
#define NTROPY_BH_MIN_LEAF_SIZE 1e-12
#define NTROPY_BH_MAX_WALK_STACK 128

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
} BHTree;

/*
 * bh_tree_build — allocate and populate an octree from N particles.
 * Copies pos/mass/eps into owned buffers.  Returns 0 on success, -1 on error.
 */
int bh_tree_build(BHTree *tree, const double *pos, const double *mass, const double *eps, int n);

/*
 * bh_tree_from_packed — reconstruct tree topology from flat node + leaf buffers.
 * Does not copy pos/mass/eps (caller keeps arrays alive).  Returns 0 / -1.
 */
int bh_tree_from_packed(
    BHTree *tree,
    const BHNode *nodes,
    int n_nodes,
    const int *leaf_indices,
    int n_leaf_indices,
    const double *pos,
    const double *mass,
    const double *eps,
    int n_particles
);

/* bh_tree_free — release nodes, leaf pool, and owned particle copies. */
void bh_tree_free(BHTree *tree);

/*
 * bh_tree_pack — deep-copy nodes and leaf_indices for C-level serialization.
 * Python wrapper uses pack_buffers() instead; kept for tests / future MPI C API.
 */
int bh_tree_pack(
    const BHTree *tree,
    BHNode **out_nodes,
    int *out_n_nodes,
    int **out_leaf_indices,
    int *out_n_leaf_indices
);

/*
 * bh_tree_accel_one — acceleration on a single target index.
 * pos/eps may differ from build-time arrays (integrator trial positions).
 */
void bh_tree_accel_one(
    const BHTree *tree,
    int target_index,
    double theta,
    const double *pos,
    const double *eps,
    double acc_out[3]
);

/*
 * bh_tree_accel_targets — accelerations for many targets.
 * acc_out layout: [ax0, ay0, az0, ax1, ay1, az1, ...], length 3 * n_targets.
 * Returns 0 on success, -1 on invalid arguments.
 */
int bh_tree_accel_targets(
    const BHTree *tree,
    const int *target_indices,
    int n_targets,
    double theta,
    const double *pos,
    const double *eps,
    double *acc_out
);

#endif /* NTROPY_BH_TREE_H */
