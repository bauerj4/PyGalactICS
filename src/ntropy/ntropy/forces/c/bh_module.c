/*
 * bh_module.c — Python C-API bindings for the Barnes–Hut tree.
 *
 * Exposes module ntropy.forces._bh_c:
 *   build_tree(pos, mass, eps)       -> Tree
 *   tree_from_packed(nodes, ...)     -> Tree
 *   Tree.accel_targets(indices, theta, pos=None, eps=None)
 *   Tree.accel_all(theta, pos=None, eps=None)
 *   Tree.pack_buffers()              -> dict for MPI bcast
 *   NODE_PACK_WIDTH = 19
 *
 * Python users should import ntropy.forces.bhtree_c instead; this file is the
 * low-level bridge documented in bhtree_c.py and forces/c/PARALLEL.md.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "bh_tree.h"

#include <Python.h>
#include <numpy/arrayobject.h>

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
    static char *kwlist[] = {"pos", "mass", "eps", NULL};
    PyArrayObject *pos = NULL;
    PyArrayObject *mass = NULL;
    PyArrayObject *eps = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!", kwlist, &PyArray_Type, &pos, &PyArray_Type, &mass, &PyArray_Type, &eps)) {
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
            self->tree.n_particles
        ) != 0
    ) {
        PyErr_SetString(PyExc_MemoryError, "failed to build Barnes-Hut tree");
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *)self;
}

static PyObject *PyBHTree_from_packed(PyTypeObject *cls, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"nodes", "leaf_indices", "pos", "mass", "eps", NULL};
    PyArrayObject *nodes = NULL;
    PyArrayObject *leaf_indices = NULL;
    PyArrayObject *pos = NULL;
    PyArrayObject *mass = NULL;
    PyArrayObject *eps = NULL;
    if (
        !PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "O!O!O!O!O!",
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
            &eps
        )
    ) {
        return NULL;
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

    const int n_nodes = (int)PyArray_DIM(nodes, 0);
    BHNode *node_buf = (BHNode *)malloc((size_t)n_nodes * sizeof(BHNode));
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
            self->tree.n_particles
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

    int *target_buf = (int *)malloc((size_t)n_targets * sizeof(int));
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

    if (
        bh_tree_accel_targets(
            &self->tree,
            target_buf,
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
    npy_intp dims[1] = {self->tree.n_particles};
    PyArrayObject *targets = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
    if (!targets) {
        return NULL;
    }
    int *tdata = (int *)PyArray_DATA(targets);
    for (int i = 0; i < self->tree.n_particles; ++i) {
        tdata[i] = i;
    }
    PyObject *call_args = Py_BuildValue("(OdOO)", targets, theta, pos_obj, eps_obj);
    Py_DECREF(targets);
    if (!call_args) {
        return NULL;
    }
    PyObject *result = PyBHTree_accel_targets(self, call_args, NULL);
    Py_DECREF(call_args);
    return result;
}

static PyObject *PyBHTree_pack_buffers(PyBHTreeObject *self, PyObject *Py_UNUSED(ignored)) {
    const int n_nodes = self->tree.n_nodes;
    const int n_leaf = self->tree.n_leaf_indices;

    npy_intp node_dims[2] = {n_nodes, BH_NODE_PACK_WIDTH};
    npy_intp leaf_dims[1] = {n_leaf};
    PyArrayObject *nodes_arr = (PyArrayObject *)PyArray_SimpleNew(2, node_dims, NPY_FLOAT64);
    PyArrayObject *leaf_arr = (PyArrayObject *)PyArray_SimpleNew(1, leaf_dims, NPY_INT32);
    npy_intp meta_dims[1] = {2};
    PyArrayObject *meta_arr = (PyArrayObject *)PyArray_SimpleNew(1, meta_dims, NPY_INT64);
    if (!nodes_arr || !leaf_arr || !meta_arr) {
        Py_XDECREF(nodes_arr);
        Py_XDECREF(leaf_arr);
        Py_XDECREF(meta_arr);
        return PyErr_NoMemory();
    }

    double *rows = (double *)PyArray_DATA(nodes_arr);
    for (int i = 0; i < n_nodes; ++i) {
        bh_fill_node_row(&self->tree.nodes[i], rows + i * BH_NODE_PACK_WIDTH);
    }
    memcpy(PyArray_DATA(leaf_arr), self->tree.leaf_indices, (size_t)n_leaf * sizeof(int));
    npy_int64 *meta = (npy_int64 *)PyArray_DATA(meta_arr);
    meta[0] = n_nodes;
    meta[1] = n_leaf;

    return Py_BuildValue(
        "{s:O,s:O,s:O,s:O,s:O,s:O}",
        "nodes",
        nodes_arr,
        "leaf_indices",
        leaf_arr,
        "meta",
        meta_arr,
        "pos",
        self->pos_ref,
        "mass",
        self->mass_ref,
        "eps",
        self->eps_ref
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
    return module;
}
