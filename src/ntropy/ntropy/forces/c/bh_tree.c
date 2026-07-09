/*
 * bh_tree.c — Barnes–Hut octree build, pack, and walk.
 *
 * Implementation notes:
 *   - Child links are integer indices; node pointers are never cached across
 *     bh_tree_alloc_node (realloc may move the nodes array).
 *   - bh_tree_build copies particle arrays; bh_tree_from_packed borrows them.
 *   - Walk is recursive (matches ntropy.forces.bhtree._walk).
 *   - Leaf split uses heap-allocated index lists (no fixed stack cap).
 *
 * See bh_tree.h for the public C API and bhtree_c.py for Python docstrings.
 */

#include "bh_tree.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int bh_child_index(const double center[3], const double point[3]) {
    int idx = 0;
    if (point[0] >= center[0]) {
        idx |= 1;
    }
    if (point[1] >= center[1]) {
        idx |= 2;
    }
    if (point[2] >= center[2]) {
        idx |= 4;
    }
    return idx;
}

static void bh_child_center(
    const double parent_center[3],
    double parent_size,
    int child_idx,
    double out_center[3]
) {
    const double offset = parent_size * 0.25;
    out_center[0] = parent_center[0] + ((child_idx & 1) ? offset : -offset);
    out_center[1] = parent_center[1] + ((child_idx & 2) ? offset : -offset);
    out_center[2] = parent_center[2] + ((child_idx & 4) ? offset : -offset);
}

static int bh_tree_grow_nodes(BHTree *tree) {
    const int old_cap = tree->cap_nodes;
    int new_cap = old_cap < 16 ? 16 : old_cap * 2;
    BHNode *new_nodes = (BHNode *)realloc(tree->nodes, (size_t)new_cap * sizeof(BHNode));
    if (!new_nodes) {
        return -1;
    }
    memset(new_nodes + old_cap, 0, (size_t)(new_cap - old_cap) * sizeof(BHNode));
    tree->nodes = new_nodes;
    tree->cap_nodes = new_cap;
    return 0;
}

static int bh_tree_alloc_node(BHTree *tree) {
    if (tree->n_nodes >= tree->cap_nodes) {
        if (bh_tree_grow_nodes(tree) != 0) {
            return -1;
        }
    }
    const int idx = tree->n_nodes++;
    BHNode *node = &tree->nodes[idx];
    memset(node, 0, sizeof(BHNode));
    node->center[0] = node->center[1] = node->center[2] = 0.0;
    node->com[0] = node->com[1] = node->com[2] = 0.0;
    node->size = 0.0;
    node->mass = 0.0;
    node->is_leaf = 1;
    for (int c = 0; c < 8; ++c) {
        node->child[c] = -1;
    }
    node->leaf_start = -1;
    node->leaf_count = 0;
    return idx;
}

static int bh_tree_grow_leaf_pool(BHTree *tree, int extra) {
    int needed = tree->n_leaf_indices + extra;
    if (needed <= tree->cap_leaf_indices) {
        return 0;
    }
    int new_cap = tree->cap_leaf_indices < 16 ? 16 : tree->cap_leaf_indices;
    while (new_cap < needed) {
        new_cap *= 2;
    }
    int *new_pool = (int *)realloc(tree->leaf_indices, (size_t)new_cap * sizeof(int));
    if (!new_pool) {
        return -1;
    }
    tree->leaf_indices = new_pool;
    tree->cap_leaf_indices = new_cap;
    return 0;
}

static int bh_tree_append_leaf(BHTree *tree, int node_index, int particle_index) {
    BHNode *node = &tree->nodes[node_index];
    if (node->leaf_count == 0) {
        if (bh_tree_grow_leaf_pool(tree, 1) != 0) {
            return -1;
        }
        node->leaf_start = tree->n_leaf_indices;
    } else if (bh_tree_grow_leaf_pool(tree, 1) != 0) {
        return -1;
    }
    tree->leaf_indices[tree->n_leaf_indices++] = particle_index;
    node->leaf_count += 1;
    return 0;
}

static void bh_tree_update_leaf(const BHTree *tree, int node_index) {
    BHNode *node = &tree->nodes[node_index];
    if (node->leaf_count <= 0) {
        node->mass = 0.0;
        node->com[0] = node->center[0];
        node->com[1] = node->center[1];
        node->com[2] = node->center[2];
        return;
    }
    double total_mass = 0.0;
    double com[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < node->leaf_count; ++i) {
        const int p = tree->leaf_indices[node->leaf_start + i];
        const double m = tree->mass[p];
        total_mass += m;
        com[0] += m * tree->pos[3 * p + 0];
        com[1] += m * tree->pos[3 * p + 1];
        com[2] += m * tree->pos[3 * p + 2];
    }
    node->mass = total_mass;
    if (total_mass > 0.0) {
        node->com[0] = com[0] / total_mass;
        node->com[1] = com[1] / total_mass;
        node->com[2] = com[2] / total_mass;
    } else {
        const int p = tree->leaf_indices[node->leaf_start];
        node->com[0] = tree->pos[3 * p + 0];
        node->com[1] = tree->pos[3 * p + 1];
        node->com[2] = tree->pos[3 * p + 2];
    }
}

static int bh_tree_coincident_in_leaf(
    const BHTree *tree,
    const BHNode *node,
    int particle_index
) {
    const double *point = &tree->pos[3 * particle_index];
    const double tol = node->size * 1e-9 > 1e-15 ? node->size * 1e-9 : 1e-15;
    for (int i = 0; i < node->leaf_count; ++i) {
        const int p = tree->leaf_indices[node->leaf_start + i];
        const double dx = point[0] - tree->pos[3 * p + 0];
        const double dy = point[1] - tree->pos[3 * p + 1];
        const double dz = point[2] - tree->pos[3 * p + 2];
        const double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist <= tol) {
            return 1;
        }
    }
    return 0;
}

static int bh_tree_insert_child(BHTree *tree, int node_index, int particle_index);

static int bh_tree_insert(BHTree *tree, int node_index, int particle_index) {
    BHNode *node = &tree->nodes[node_index];
    const double *point = &tree->pos[3 * particle_index];

    if (node->is_leaf) {
        if (node->leaf_count == 0) {
            return bh_tree_append_leaf(tree, node_index, particle_index);
        }
        if (
            bh_tree_coincident_in_leaf(tree, node, particle_index)
            || node->size <= NTROPY_BH_MIN_LEAF_SIZE
        ) {
            if (bh_tree_append_leaf(tree, node_index, particle_index) != 0) {
                return -1;
            }
            bh_tree_update_leaf(tree, node_index);
            return 0;
        }

        const int saved_count = node->leaf_count;
        int *saved = (int *)malloc((size_t)(saved_count + 1) * sizeof(int));
        if (!saved) {
            return -1;
        }
        for (int i = 0; i < saved_count; ++i) {
            saved[i] = tree->leaf_indices[node->leaf_start + i];
        }
        saved[saved_count] = particle_index;

        node->leaf_count = 0;
        node->leaf_start = -1;
        node->is_leaf = 0;
        node->mass = 0.0;
        node->com[0] = node->com[1] = node->com[2] = 0.0;

        for (int i = 0; i < saved_count + 1; ++i) {
            if (bh_tree_insert_child(tree, node_index, saved[i]) != 0) {
                free(saved);
                return -1;
            }
        }
        free(saved);
        return 0;
    }

    return bh_tree_insert_child(tree, node_index, particle_index);
}

static int bh_tree_insert_child(BHTree *tree, int node_index, int particle_index) {
    const double *point = &tree->pos[3 * particle_index];
    const int child_idx = bh_child_index(tree->nodes[node_index].center, point);
    if (tree->nodes[node_index].child[child_idx] < 0) {
        const int child_node = bh_tree_alloc_node(tree);
        if (child_node < 0) {
            return -1;
        }
        /* Re-fetch node_index after realloc in bh_tree_alloc_node. */
        tree->nodes[node_index].child[child_idx] = child_node;
        bh_child_center(
            tree->nodes[node_index].center,
            tree->nodes[node_index].size,
            child_idx,
            tree->nodes[child_node].center
        );
        tree->nodes[child_node].size = 0.5 * tree->nodes[node_index].size;
    }
    return bh_tree_insert(tree, tree->nodes[node_index].child[child_idx], particle_index);
}

static void bh_tree_aggregate(BHTree *tree, int node_index) {
    BHNode *node = &tree->nodes[node_index];
    if (node->is_leaf) {
        bh_tree_update_leaf(tree, node_index);
        return;
    }
    double total_mass = 0.0;
    double com[3] = {0.0, 0.0, 0.0};
    for (int c = 0; c < 8; ++c) {
        const int child = node->child[c];
        if (child < 0) {
            continue;
        }
        bh_tree_aggregate(tree, child);
        const BHNode *child_node = &tree->nodes[child];
        total_mass += child_node->mass;
        com[0] += child_node->mass * child_node->com[0];
        com[1] += child_node->mass * child_node->com[1];
        com[2] += child_node->mass * child_node->com[2];
    }
    node->mass = total_mass;
    if (total_mass > 0.0) {
        node->com[0] = com[0] / total_mass;
        node->com[1] = com[1] / total_mass;
        node->com[2] = com[2] / total_mass;
    } else {
        node->com[0] = node->center[0];
        node->com[1] = node->center[1];
        node->com[2] = node->center[2];
    }
}

void bh_tree_free(BHTree *tree) {
    if (!tree) {
        return;
    }
    free(tree->nodes);
    free(tree->leaf_indices);
    if (tree->owns_particle_arrays) {
        free(tree->pos);
        free(tree->mass);
        free(tree->eps);
    }
    memset(tree, 0, sizeof(BHTree));
}

static int bh_tree_copy_particles(
    BHTree *tree,
    const double *pos,
    const double *mass,
    const double *eps,
    int n
) {
    tree->n_particles = n;
    tree->owns_particle_arrays = 1;
    tree->pos = (double *)malloc((size_t)n * 3 * sizeof(double));
    tree->mass = (double *)malloc((size_t)n * sizeof(double));
    tree->eps = (double *)malloc((size_t)n * sizeof(double));
    if (!tree->pos || !tree->mass || !tree->eps) {
        return -1;
    }
    memcpy(tree->pos, pos, (size_t)n * 3 * sizeof(double));
    memcpy(tree->mass, mass, (size_t)n * sizeof(double));
    memcpy(tree->eps, eps, (size_t)n * sizeof(double));
    return 0;
}

int bh_tree_build(BHTree *tree, const double *pos, const double *mass, const double *eps, int n) {
    if (!tree || !pos || !mass || !eps || n <= 0) {
        return -1;
    }
    memset(tree, 0, sizeof(BHTree));
    if (bh_tree_copy_particles(tree, pos, mass, eps, n) != 0) {
        bh_tree_free(tree);
        return -1;
    }

    const double *build_pos = tree->pos;
    double mins[3] = {build_pos[0], build_pos[1], build_pos[2]};
    double maxs[3] = {build_pos[0], build_pos[1], build_pos[2]};
    for (int i = 1; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double v = build_pos[3 * i + d];
            if (v < mins[d]) {
                mins[d] = v;
            }
            if (v > maxs[d]) {
                maxs[d] = v;
            }
        }
    }
    const int root_index = bh_tree_alloc_node(tree);
    if (root_index < 0) {
        bh_tree_free(tree);
        return -1;
    }
    BHNode *root = &tree->nodes[root_index];
    root->center[0] = 0.5 * (mins[0] + maxs[0]);
    root->center[1] = 0.5 * (mins[1] + maxs[1]);
    root->center[2] = 0.5 * (mins[2] + maxs[2]);
    double half = 0.5 * (maxs[0] - mins[0]);
    for (int d = 1; d < 3; ++d) {
        const double span = 0.5 * (maxs[d] - mins[d]);
        if (span > half) {
            half = span;
        }
    }
    if (half == 0.0) {
        half = 1.0;
    }
    root->size = 2.0 * half * 1.01;

    for (int i = 0; i < n; ++i) {
        if (bh_tree_insert(tree, root_index, i) != 0) {
            bh_tree_free(tree);
            return -1;
        }
    }
    bh_tree_aggregate(tree, root_index);
    return 0;
}

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
) {
    if (!tree || !nodes || n_nodes <= 0 || !leaf_indices || !pos || !mass || !eps || n_particles <= 0) {
        return -1;
    }
    memset(tree, 0, sizeof(BHTree));
    tree->n_nodes = n_nodes;
    tree->cap_nodes = n_nodes;
    tree->nodes = (BHNode *)malloc((size_t)n_nodes * sizeof(BHNode));
    if (!tree->nodes) {
        return -1;
    }
    memcpy(tree->nodes, nodes, (size_t)n_nodes * sizeof(BHNode));

    tree->n_leaf_indices = n_leaf_indices;
    tree->cap_leaf_indices = n_leaf_indices;
    tree->leaf_indices = (int *)malloc((size_t)n_leaf_indices * sizeof(int));
    if (!tree->leaf_indices) {
        bh_tree_free(tree);
        return -1;
    }
    memcpy(tree->leaf_indices, leaf_indices, (size_t)n_leaf_indices * sizeof(int));

    tree->pos = (double *)pos;
    tree->mass = (double *)mass;
    tree->eps = (double *)eps;
    tree->n_particles = n_particles;
    tree->owns_particle_arrays = 0;
    return 0;
}

int bh_tree_pack(
    const BHTree *tree,
    BHNode **out_nodes,
    int *out_n_nodes,
    int **out_leaf_indices,
    int *out_n_leaf_indices
) {
    if (!tree || !out_nodes || !out_n_nodes || !out_leaf_indices || !out_n_leaf_indices) {
        return -1;
    }
    BHNode *nodes_copy = (BHNode *)malloc((size_t)tree->n_nodes * sizeof(BHNode));
    if (!nodes_copy) {
        return -1;
    }
    memcpy(nodes_copy, tree->nodes, (size_t)tree->n_nodes * sizeof(BHNode));

    int *leaf_copy = (int *)malloc((size_t)tree->n_leaf_indices * sizeof(int));
    if (!leaf_copy) {
        free(nodes_copy);
        return -1;
    }
    memcpy(leaf_copy, tree->leaf_indices, (size_t)tree->n_leaf_indices * sizeof(int));

    *out_nodes = nodes_copy;
    *out_n_nodes = tree->n_nodes;
    *out_leaf_indices = leaf_copy;
    *out_n_leaf_indices = tree->n_leaf_indices;
    return 0;
}

static void bh_tree_walk(
    const BHTree *tree,
    int node_index,
    int target_index,
    double theta,
    const double *pos,
    const double *eps,
    double acc[3]
) {
    const BHNode *node = &tree->nodes[node_index];
    if (node->mass <= 0.0) {
        return;
    }

    const double *target_pos = &pos[3 * target_index];
    const double dr[3] = {
        node->com[0] - target_pos[0],
        node->com[1] - target_pos[1],
        node->com[2] - target_pos[2],
    };
    const double dist = sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);

    if (node->is_leaf) {
        for (int i = 0; i < node->leaf_count; ++i) {
            const int source = tree->leaf_indices[node->leaf_start + i];
            if (source == target_index) {
                continue;
            }
            const double rx = pos[3 * source + 0] - target_pos[0];
            const double ry = pos[3 * source + 1] - target_pos[1];
            const double rz = pos[3 * source + 2] - target_pos[2];
            const double r2 = rx * rx + ry * ry + rz * rz;
            if (r2 == 0.0) {
                continue;
            }
            const double h = 0.5 * (eps[target_index] + eps[source]);
            const double denom = pow(r2 + h * h, 1.5);
            const double factor = NTROPY_BH_G * tree->mass[source] / denom;
            acc[0] += factor * rx;
            acc[1] += factor * ry;
            acc[2] += factor * rz;
        }
        return;
    }

    if (node->size / (dist > 1e-30 ? dist : 1e-30) < theta) {
        const double h = eps[target_index];
        const double denom = pow(dist * dist + h * h, 1.5);
        const double factor = NTROPY_BH_G * node->mass / denom;
        acc[0] += factor * dr[0];
        acc[1] += factor * dr[1];
        acc[2] += factor * dr[2];
        return;
    }

    for (int c = 0; c < 8; ++c) {
        if (node->child[c] >= 0) {
            bh_tree_walk(tree, node->child[c], target_index, theta, pos, eps, acc);
        }
    }
}

void bh_tree_accel_one(
    const BHTree *tree,
    int target_index,
    double theta,
    const double *pos,
    const double *eps,
    double acc_out[3]
) {
    acc_out[0] = acc_out[1] = acc_out[2] = 0.0;
    if (!tree || tree->n_nodes <= 0 || target_index < 0 || target_index >= tree->n_particles) {
        return;
    }
    bh_tree_walk(tree, 0, target_index, theta, pos, eps, acc_out);
}

int bh_tree_accel_targets(
    const BHTree *tree,
    const int *target_indices,
    int n_targets,
    double theta,
    const double *pos,
    const double *eps,
    double *acc_out
) {
    if (!tree || !target_indices || !acc_out || n_targets < 0) {
        return -1;
    }
    for (int k = 0; k < n_targets; ++k) {
        double acc[3];
        bh_tree_accel_one(tree, target_indices[k], theta, pos, eps, acc);
        acc_out[3 * k + 0] = acc[0];
        acc_out[3 * k + 1] = acc[1];
        acc_out[3 * k + 2] = acc[2];
    }
    return 0;
}
