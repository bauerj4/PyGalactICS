/*
 * bh_module.c — Python C-API bindings for the Barnes–Hut tree.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "bh_tree.h"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>

#define BH_NODE_PACK_WIDTH 19

typedef struct {
    PyObject_HEAD
    BHTree tree;
    PyObject *pos_ref;
    PyObject *mass_ref;
    PyObject *eps_ref;
} PyBHTreeObject;

static PyTypeObject PyBHTreeType;

static int bh_check_array(PyArrayObject *arr, int ndim, int typenum) {
    if (!arr) {
        PyErr_SetString(PyExc_ValueError, "expected numpy array");
        return -1;
    }
    if (PyArray_NDIM(arr) != ndim) {
        PyErr_Format(PyExc_ValueError, "expected %d-D array", ndim);
        return -1;
    }
    if (PyArray_TYPE(arr) != typenum) {
        PyErr_SetString(PyExc_TypeError, "unexpected array dtype");
        return -1;
    }
    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_ValueError, "array must be C-contiguous");
        return -1;
    }
    return 0;
}

static int bh_read_opt_flag(PyObject *opts, const char *key, int *out) {
    PyObject *val = PyDict_GetItemString(opts, key);
    if (!val) {
        return 0;
    }
    *out = PyObject_IsTrue(val) ? 1 : 0;
    return 0;
}

static int bh_read_opt_schedule(PyObject *opts, int *out) {
    PyObject *val = PyDict_GetItemString(opts, "omp_schedule");
    if (!val) {
        return 0;
    }
    if (PyLong_Check(val)) {
        *out = (int)PyLong_AsLong(val);
        return 0;
    }
    if (PyUnicode_Check(val)) {
        if (PyUnicode_CompareWithASCIIString(val, "guided") == 0) {
            *out = 1;
            return 0;
        }
        if (PyUnicode_CompareWithASCIIString(val, "dynamic") == 0) {
            *out = 2;
            return 0;
        }
        *out = 0;
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "omp_schedule must be int or str");
    return -1;
}

static int bh_parse_opts(PyObject *opts_obj, BHTreeOpts *out) {
    bh_tree_opts_default(out);
    if (!opts_obj || opts_obj == Py_None) {
        return 0;
    }
    if (!PyDict_Check(opts_obj)) {
        PyErr_SetString(PyExc_TypeError, "opts must be a dict or None");
        return -1;
    }
    if (bh_read_opt_flag(opts_obj, "fast_inv_r3", &out->fast_inv_r3) != 0) {
        return -1;
    }
    if (bh_read_opt_flag(opts_obj, "squared_opening", &out->squared_opening) != 0) {
        return -1;
    }
    if (bh_read_opt_flag(opts_obj, "iterative_walk", &out->iterative_walk) != 0) {
        return -1;
    }
    if (bh_read_opt_flag(opts_obj, "morton_build", &out->morton_build) != 0) {
        return -1;
    }
    if (bh_read_opt_flag(opts_obj, "borrow_arrays", &out->borrow_arrays) != 0) {
        return -1;
    }
    if (bh_read_opt_flag(opts_obj, "fast_coincident_check", &out->fast_coincident_check) != 0) {
        return -1;
    }
    if (bh_read_opt_flag(opts_obj, "native_pack", &out->native_pack) != 0) {
        return -1;
    }
    if (bh_read_opt_flag(opts_obj, "accel_all_fast", &out->accel_all_fast) != 0) {
        return -1;
    }
    if (bh_read_opt_flag(opts_obj, "simd_leaves", &out->simd_leaves) != 0) {
        return -1;
    }
    return bh_read_opt_schedule(opts_obj, &out->omp_schedule);
}

static int bh_bind_particle_arrays(
    PyBHTreeObject *self,
    PyArrayObject *pos,
    PyArrayObject *mass,
    PyArrayObject *eps
) {
    if (bh_check_array(pos, 2, NPY_FLOAT64) != 0) {
        return -1;
    }
    if (bh_check_array(mass, 1, NPY_FLOAT64) != 0) {
        return -1;
    }
    if (bh_check_array(eps, 1, NPY_FLOAT64) != 0) {
        return -1;
    }
    const npy_intp n = PyArray_DIM(mass, 0);
    if (PyArray_DIM(pos, 0) != n || PyArray_DIM(pos, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "pos must have shape (N, 3)");
        return -1;
    }
    if (PyArray_DIM(eps, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "eps must have shape (N,)");
        return -1;
    }

    Py_XDECREF(self->pos_ref);
    Py_XDECREF(self->mass_ref);
    Py_XDECREF(self->eps_ref);
    self->pos_ref = (PyObject *)pos;
    self->mass_ref = (PyObject *)mass;
    self->eps_ref = (PyObject *)eps;
    Py_INCREF(self->pos_ref);
    Py_INCREF(self->mass_ref);
    Py_INCREF(self->eps_ref);

    self->tree.n_particles = (int)n;
    return 0;
}

static void bh_fill_node_row(const BHNode *node, double *row) {
    row[0] = node->center[0];
    row[1] = node->center[1];
    row[2] = node->center[2];
    row[3] = node->com[0];
    row[4] = node->com[1];
    row[5] = node->com[2];
    row[6] = node->size;
    row[7] = node->mass;
    row[8] = (double)node->is_leaf;
    for (int c = 0; c < 8; ++c) {
        row[9 + c] = (double)node->child[c];
    }
    row[17] = (double)node->leaf_start;
    row[18] = (double)node->leaf_count;
}

static void bh_row_to_node(const double *row, BHNode *node) {
    node->center[0] = row[0];
    node->center[1] = row[1];
    node->center[2] = row[2];
    node->com[0] = row[3];
    node->com[1] = row[4];
    node->com[2] = row[5];
    node->size = row[6];
    node->mass = row[7];
    node->is_leaf = (int)row[8];
    for (int c = 0; c < 8; ++c) {
        node->child[c] = (int)row[9 + c];
    }
    node->leaf_start = (int)row[17];
    node->leaf_count = (int)row[18];
}

static void PyBHTree_dealloc(PyBHTreeObject *self) {
    bh_tree_free(&self->tree);
    Py_XDECREF(self->pos_ref);
    Py_XDECREF(self->mass_ref);
    Py_XDECREF(self->eps_ref);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyBHTree_new(PyTypeObject *cls, PyObject *args, PyObject *kwds) {
    (void)args;
    (void)kwds;
    PyBHTreeObject *self = (PyBHTreeObject *)cls->tp_alloc(cls, 0);
    if (!self) {
        return NULL;
    }
    memset(&self->tree, 0, sizeof(BHTree));
    self->pos_ref = NULL;
    self->mass_ref = NULL;
    self->eps_ref = NULL;
    return (PyObject *)self;
}

static PyObject *PyBHTree_build(PyTypeObject *cls, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"pos", "mass", "eps", "opts", NULL};
    PyArrayObject *pos = NULL;
    PyArrayObject *mass = NULL;
    PyArrayObject *eps = NULL;
    PyObject *opts_obj = Py_None;
    if (
        !PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "O!O!O!|O",
            kwlist,
            &PyArray_Type,
            &pos,
            &PyArray_Type,
            &mass,
            &PyArray_Type,
            &eps,
            &opts_obj
        )
    ) {
        return NULL;
    }

    BHTreeOpts opts;
    if (bh_parse_opts(opts_obj, &opts) != 0) {
        return NULL;
    }

    PyBHTreeObject *self = (PyBHTreeObject *)PyBHTree_new(cls, NULL, NULL);
    if (!self) {
        return NULL;
    }
    if (bh_bind_particle_arrays(self, pos, mass, eps) != 0) {
        Py_DECREF(self);
        return NULL;
    }
    if (
        bh_tree_build(
            &self->tree,
            (const double *)PyArray_DATA(pos),
            (const double *)PyArray_DATA(mass),
            (const double *)PyArray_DATA(eps),
            self->tree.n_particles,
            &opts
        ) != 0
    ) {
        PyErr_SetString(PyExc_MemoryError, "failed to build Barnes-Hut tree");
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *)self;
}

static PyObject *PyBHTree_from_packed(PyTypeObject *cls, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "nodes",
        "leaf_indices",
        "pos",
        "mass",
        "eps",
        "pack_format",
        "nodes_native",
        "opts",
        NULL,
    };
    PyArrayObject *nodes = NULL;
    PyArrayObject *leaf_indices = NULL;
    PyArrayObject *pos = NULL;
    PyArrayObject *mass = NULL;
    PyArrayObject *eps = NULL;
    long pack_format = NTROPY_BH_PACK_LEGACY;
    PyObject *nodes_native_obj = Py_None;
    PyObject *opts_obj = Py_None;

    if (
        !PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "O!O!O!O!O!|lOO",
            kwlist,
            &PyArray_Type,
            &nodes,
            &PyArray_Type,
            &leaf_indices,
            &PyArray_Type,
            &pos,
            &PyArray_Type,
            &mass,
            &PyArray_Type,
            &eps,
            &pack_format,
            &nodes_native_obj,
            &opts_obj
        )
    ) {
        return NULL;
    }

    BHTreeOpts opts;
    if (bh_parse_opts(opts_obj, &opts) != 0) {
        return NULL;
    }

    const int n_nodes = pack_format == NTROPY_BH_PACK_NATIVE
        ? 0
        : (int)PyArray_DIM(nodes, 0);
    BHNode *node_buf = NULL;

    if (pack_format == NTROPY_BH_PACK_NATIVE) {
        if (nodes_native_obj == Py_None) {
            PyErr_SetString(PyExc_ValueError, "nodes_native required for native pack format");
            return NULL;
        }
        PyArrayObject *native = (PyArrayObject *)nodes_native_obj;
        if (PyArray_TYPE(native) != NPY_UINT8 || !PyArray_ISCONTIGUOUS(native)) {
            PyErr_SetString(PyExc_ValueError, "nodes_native must be contiguous uint8");
            return NULL;
        }
        const npy_intp nbytes = PyArray_DIM(native, 0);
        if (nbytes % (npy_intp)sizeof(BHNode) != 0) {
            PyErr_SetString(PyExc_ValueError, "nodes_native length must be multiple of BHNode size");
            return NULL;
        }
        const int native_nodes = (int)(nbytes / (npy_intp)sizeof(BHNode));
        node_buf = (BHNode *)malloc((size_t)native_nodes * sizeof(BHNode));
        if (!node_buf) {
            return PyErr_NoMemory();
        }
        memcpy(node_buf, PyArray_DATA(native), (size_t)native_nodes * sizeof(BHNode));
        if (bh_check_array(leaf_indices, 1, NPY_INT32) != 0) {
            free(node_buf);
            return NULL;
        }
        PyBHTreeObject *self = (PyBHTreeObject *)PyBHTree_new(cls, NULL, NULL);
        if (!self) {
            free(node_buf);
            return NULL;
        }
        if (bh_bind_particle_arrays(self, pos, mass, eps) != 0) {
            free(node_buf);
            Py_DECREF(self);
            return NULL;
        }
        const int n_leaf = (int)PyArray_DIM(leaf_indices, 0);
        if (
            bh_tree_from_packed(
                &self->tree,
                node_buf,
                native_nodes,
                (const int *)PyArray_DATA(leaf_indices),
                n_leaf,
                (const double *)PyArray_DATA(pos),
                (const double *)PyArray_DATA(mass),
                (const double *)PyArray_DATA(eps),
                self->tree.n_particles,
                &opts
            ) != 0
        ) {
            free(node_buf);
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        free(node_buf);
        return (PyObject *)self;
    }

    if (bh_check_array(nodes, 2, NPY_FLOAT64) != 0) {
        return NULL;
    }
    if (PyArray_DIM(nodes, 1) != BH_NODE_PACK_WIDTH) {
        PyErr_Format(PyExc_ValueError, "nodes must have shape (N, %d)", BH_NODE_PACK_WIDTH);
        return NULL;
    }
    if (bh_check_array(leaf_indices, 1, NPY_INT32) != 0) {
        return NULL;
    }

    node_buf = (BHNode *)malloc((size_t)n_nodes * sizeof(BHNode));
    if (!node_buf) {
        return PyErr_NoMemory();
    }
    const double *rows = (const double *)PyArray_DATA(nodes);
    for (int i = 0; i < n_nodes; ++i) {
        bh_row_to_node(rows + i * BH_NODE_PACK_WIDTH, &node_buf[i]);
    }
    PyBHTreeObject *self = (PyBHTreeObject *)PyBHTree_new(cls, NULL, NULL);
    if (!self) {
        free(node_buf);
        return NULL;
    }
    if (bh_bind_particle_arrays(self, pos, mass, eps) != 0) {
        free(node_buf);
        Py_DECREF(self);
        return NULL;
    }
    const int n_leaf = (int)PyArray_DIM(leaf_indices, 0);
    if (
        bh_tree_from_packed(
            &self->tree,
            node_buf,
            n_nodes,
            (const int *)PyArray_DATA(leaf_indices),
            n_leaf,
            (const double *)PyArray_DATA(pos),
            (const double *)PyArray_DATA(mass),
            (const double *)PyArray_DATA(eps),
            self->tree.n_particles,
            &opts
        ) != 0
    ) {
        free(node_buf);
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    free(node_buf);
    return (PyObject *)self;
}

static PyObject *PyBHTree_accel_targets(PyBHTreeObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"target_indices", "theta", "pos", "eps", NULL};
    PyArrayObject *targets = NULL;
    double theta = 0.5;
    PyObject *pos_obj = Py_None;
    PyObject *eps_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!d|OO", kwlist, &PyArray_Type, &targets, &theta, &pos_obj, &eps_obj)) {
        return NULL;
    }
    if (PyArray_TYPE(targets) != NPY_INT32 && PyArray_TYPE(targets) != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError, "target_indices must be int32 or int64");
        return NULL;
    }
    if (PyArray_NDIM(targets) != 1) {
        PyErr_SetString(PyExc_ValueError, "target_indices must be 1-D");
        return NULL;
    }

    const double *pos = self->tree.pos;
    const double *eps = self->tree.eps;
    if (pos_obj != Py_None) {
        PyArrayObject *pos_arr = (PyArrayObject *)pos_obj;
        if (bh_check_array(pos_arr, 2, NPY_FLOAT64) != 0) {
            return NULL;
        }
        pos = (double *)PyArray_DATA(pos_arr);
    }
    if (eps_obj != Py_None) {
        PyArrayObject *eps_arr = (PyArrayObject *)eps_obj;
        if (bh_check_array(eps_arr, 1, NPY_FLOAT64) != 0) {
            return NULL;
        }
        eps = (double *)PyArray_DATA(eps_arr);
    }

    const npy_intp n_targets = PyArray_DIM(targets, 0);
    npy_intp dims[2] = {n_targets, 3};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (!out) {
        return NULL;
    }

    const int *target_ptr = NULL;
    int *target_buf = NULL;
    if (PyArray_TYPE(targets) == NPY_INT32 && PyArray_ISCONTIGUOUS(targets)) {
        target_ptr = (const int *)PyArray_DATA(targets);
    } else {
        target_buf = (int *)malloc((size_t)n_targets * sizeof(int));
        if (!target_buf) {
            Py_DECREF(out);
            return PyErr_NoMemory();
        }
        if (PyArray_TYPE(targets) == NPY_INT64) {
            const npy_int64 *src = (const npy_int64 *)PyArray_DATA(targets);
            for (npy_intp i = 0; i < n_targets; ++i) {
                target_buf[i] = (int)src[i];
            }
        } else {
            memcpy(target_buf, PyArray_DATA(targets), (size_t)n_targets * sizeof(int));
        }
        target_ptr = target_buf;
    }

    if (
        bh_tree_accel_targets(
            &self->tree,
            target_ptr,
            (int)n_targets,
            theta,
            pos,
            eps,
            (double *)PyArray_DATA(out)
        ) != 0
    ) {
        free(target_buf);
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "acceleration evaluation failed");
        return NULL;
    }
    free(target_buf);
    return (PyObject *)out;
}

static PyObject *PyBHTree_accel_all(PyBHTreeObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"theta", "pos", "eps", NULL};
    double theta = 0.5;
    PyObject *pos_obj = Py_None;
    PyObject *eps_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|OO", kwlist, &theta, &pos_obj, &eps_obj)) {
        return NULL;
    }

    const double *pos = self->tree.pos;
    const double *eps = self->tree.eps;
    if (pos_obj != Py_None) {
        PyArrayObject *pos_arr = (PyArrayObject *)pos_obj;
        if (bh_check_array(pos_arr, 2, NPY_FLOAT64) != 0) {
            return NULL;
        }
        pos = (double *)PyArray_DATA(pos_arr);
    }
    if (eps_obj != Py_None) {
        PyArrayObject *eps_arr = (PyArrayObject *)eps_obj;
        if (bh_check_array(eps_arr, 1, NPY_FLOAT64) != 0) {
            return NULL;
        }
        eps = (double *)PyArray_DATA(eps_arr);
    }

    const int n = self->tree.n_particles;
    npy_intp dims[2] = {n, 3};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (!out) {
        return NULL;
    }

    int status = 0;
    if (self->tree.opts.accel_all_fast) {
        status = bh_tree_accel_all(&self->tree, theta, pos, eps, (double *)PyArray_DATA(out));
    } else {
        npy_intp tdims[1] = {n};
        PyArrayObject *targets = (PyArrayObject *)PyArray_SimpleNew(1, tdims, NPY_INT32);
        if (!targets) {
            Py_DECREF(out);
            return NULL;
        }
        int *tdata = (int *)PyArray_DATA(targets);
        for (int i = 0; i < n; ++i) {
            tdata[i] = i;
        }
        status = bh_tree_accel_targets(
            &self->tree,
            tdata,
            n,
            theta,
            pos,
            eps,
            (double *)PyArray_DATA(out)
        );
        Py_DECREF(targets);
    }

    if (status != 0) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "acceleration evaluation failed");
        return NULL;
    }
    return (PyObject *)out;
}

static PyObject *PyBHTree_pack_buffers(PyBHTreeObject *self, PyObject *Py_UNUSED(ignored)) {
    const int n_nodes = self->tree.n_nodes;
    const int n_leaf = self->tree.n_leaf_indices;
    const int pack_format = self->tree.opts.native_pack ? NTROPY_BH_PACK_NATIVE : NTROPY_BH_PACK_LEGACY;

    npy_intp leaf_dims[1] = {n_leaf};
    PyArrayObject *leaf_arr = (PyArrayObject *)PyArray_SimpleNew(1, leaf_dims, NPY_INT32);
    npy_intp meta_dims[1] = {3};
    PyArrayObject *meta_arr = (PyArrayObject *)PyArray_SimpleNew(1, meta_dims, NPY_INT64);
    PyArrayObject *nodes_arr = NULL;
    PyArrayObject *native_arr = NULL;

    if (!leaf_arr || !meta_arr) {
        Py_XDECREF(leaf_arr);
        Py_XDECREF(meta_arr);
        return PyErr_NoMemory();
    }

    memcpy(PyArray_DATA(leaf_arr), self->tree.leaf_indices, (size_t)n_leaf * sizeof(int));
    npy_int64 *meta = (npy_int64 *)PyArray_DATA(meta_arr);
    meta[0] = n_nodes;
    meta[1] = n_leaf;
    meta[2] = pack_format;

    if (pack_format == NTROPY_BH_PACK_NATIVE) {
        npy_intp native_dims[1] = {(npy_intp)n_nodes * (npy_intp)sizeof(BHNode)};
        native_arr = (PyArrayObject *)PyArray_SimpleNew(1, native_dims, NPY_UINT8);
        if (!native_arr) {
            Py_DECREF(leaf_arr);
            Py_DECREF(meta_arr);
            return PyErr_NoMemory();
        }
        memcpy(PyArray_DATA(native_arr), self->tree.nodes, (size_t)native_dims[0]);
        npy_intp legacy_dims[2] = {0, BH_NODE_PACK_WIDTH};
        nodes_arr = (PyArrayObject *)PyArray_SimpleNew(2, legacy_dims, NPY_FLOAT64);
        if (!nodes_arr) {
            Py_DECREF(leaf_arr);
            Py_DECREF(meta_arr);
            Py_DECREF(native_arr);
            return PyErr_NoMemory();
        }
    } else {
        npy_intp node_dims[2] = {n_nodes, BH_NODE_PACK_WIDTH};
        nodes_arr = (PyArrayObject *)PyArray_SimpleNew(2, node_dims, NPY_FLOAT64);
        if (!nodes_arr) {
            Py_DECREF(leaf_arr);
            Py_DECREF(meta_arr);
            return PyErr_NoMemory();
        }
        double *rows = (double *)PyArray_DATA(nodes_arr);
        for (int i = 0; i < n_nodes; ++i) {
            bh_fill_node_row(&self->tree.nodes[i], rows + i * BH_NODE_PACK_WIDTH);
        }
        npy_intp native_dims[1] = {0};
        native_arr = (PyArrayObject *)PyArray_SimpleNew(1, native_dims, NPY_UINT8);
        if (!native_arr) {
            Py_DECREF(leaf_arr);
            Py_DECREF(meta_arr);
            Py_DECREF(nodes_arr);
            return PyErr_NoMemory();
        }
    }

    return Py_BuildValue(
        "{s:N,s:N,s:N,s:N,s:O,s:O,s:O,s:i}",
        "nodes",
        nodes_arr,
        "nodes_native",
        native_arr,
        "leaf_indices",
        leaf_arr,
        "meta",
        meta_arr,
        "pos",
        self->pos_ref,
        "mass",
        self->mass_ref,
        "eps",
        self->eps_ref,
        "pack_format",
        pack_format
    );
}

static PyObject *PyBHTree_get_n_nodes(PyBHTreeObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->tree.n_nodes);
}

static PyObject *PyBHTree_get_n_particles(PyBHTreeObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->tree.n_particles);
}

static PyGetSetDef PyBHTree_getset[] = {
    {"n_nodes", (getter)PyBHTree_get_n_nodes, NULL, "Number of octree nodes", NULL},
    {"n_particles", (getter)PyBHTree_get_n_particles, NULL, "Number of particles", NULL},
    {NULL},
};

static PyMethodDef PyBHTree_methods[] = {
    {"accel_targets", (PyCFunction)PyBHTree_accel_targets, METH_VARARGS | METH_KEYWORDS, "Compute accelerations on target indices"},
    {"accel_all", (PyCFunction)PyBHTree_accel_all, METH_VARARGS | METH_KEYWORDS, "Compute accelerations on all particles"},
    {"pack_buffers", (PyCFunction)PyBHTree_pack_buffers, METH_NOARGS, "Export flat node buffers for MPI broadcast"},
    {NULL},
};

static PyTypeObject PyBHTreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ntropy.forces._bh_c.Tree",
    .tp_basicsize = sizeof(PyBHTreeObject),
    .tp_dealloc = (destructor)PyBHTree_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_methods = PyBHTree_methods,
    .tp_getset = PyBHTree_getset,
    .tp_new = PyBHTree_new,
};

static PyObject *mod_build_tree(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    return PyBHTree_build(&PyBHTreeType, args, kwds);
}

static PyObject *mod_tree_from_packed(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    return PyBHTree_from_packed(&PyBHTreeType, args, kwds);
}

static PyMethodDef bh_module_methods[] = {
    {"build_tree", (PyCFunction)mod_build_tree, METH_VARARGS | METH_KEYWORDS, "Build a Barnes-Hut tree"},
    {"tree_from_packed", (PyCFunction)mod_tree_from_packed, METH_VARARGS | METH_KEYWORDS, "Reconstruct a tree from packed buffers"},
    {NULL},
};

static struct PyModuleDef bh_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ntropy.forces._bh_c",
    .m_doc = "C Barnes-Hut tree for ntropy",
    .m_size = -1,
    .m_methods = bh_module_methods,
};

PyMODINIT_FUNC PyInit__bh_c(void) {
    import_array();
    PyBHTreeType.tp_new = PyBHTree_new;
    if (PyType_Ready(&PyBHTreeType) < 0) {
        return NULL;
    }
    PyObject *module = PyModule_Create(&bh_module);
    if (!module) {
        return NULL;
    }
    Py_INCREF(&PyBHTreeType);
    if (PyModule_AddObject(module, "Tree", (PyObject *)&PyBHTreeType) < 0) {
        Py_DECREF(&PyBHTreeType);
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddIntConstant(module, "NODE_PACK_WIDTH", BH_NODE_PACK_WIDTH) < 0) {
        Py_DECREF(&PyBHTreeType);
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddIntConstant(module, "PACK_FORMAT_LEGACY", NTROPY_BH_PACK_LEGACY) < 0) {
        Py_DECREF(&PyBHTreeType);
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddIntConstant(module, "PACK_FORMAT_NATIVE", NTROPY_BH_PACK_NATIVE) < 0) {
        Py_DECREF(&PyBHTreeType);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
