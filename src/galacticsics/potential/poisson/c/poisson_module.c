/*
 * poisson_module.c — Python bindings for OpenMP polar shell integration.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "poisson.h"

#include <Python.h>
#include <numpy/arrayobject.h>

static int get_required_double(PyObject *dict, const char *key, double *out) {
    PyObject *val = PyDict_GetItemString(dict, key);
    if (!val) {
        PyErr_Format(PyExc_KeyError, "missing pack key '%s'", key);
        return -1;
    }
    *out = PyFloat_AsDouble(val);
    return PyErr_Occurred() ? -1 : 0;
}

static int get_required_int(PyObject *dict, const char *key, int *out) {
    PyObject *val = PyDict_GetItemString(dict, key);
    if (!val) {
        PyErr_Format(PyExc_KeyError, "missing pack key '%s'", key);
        return -1;
    }
    *out = (int)PyLong_AsLong(val);
    return PyErr_Occurred() ? -1 : 0;
}

static int get_optional_int(PyObject *dict, const char *key, int default_value, int *out) {
    PyObject *val = PyDict_GetItemString(dict, key);
    if (!val) {
        *out = default_value;
        return 0;
    }
    *out = (int)PyLong_AsLong(val);
    return PyErr_Occurred() ? -1 : 0;
}

static PyArrayObject *get_required_array(PyObject *dict, const char *key, int ndim) {
    PyObject *val = PyDict_GetItemString(dict, key);
    if (!val) {
        PyErr_Format(PyExc_KeyError, "missing pack key '%s'", key);
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(val, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (!arr) {
        return NULL;
    }
    if (PyArray_NDIM(arr) != ndim) {
        Py_DECREF(arr);
        PyErr_Format(PyExc_ValueError, "array '%s' must be %d-D", key, ndim);
        return NULL;
    }
    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
        Py_DECREF(arr);
        PyErr_Format(PyExc_ValueError, "array '%s' must be C-contiguous", key);
        return NULL;
    }
    return arr;
}

static PyArrayObject *get_optional_array(PyObject *dict, const char *key, int ndim) {
    PyObject *val = PyDict_GetItemString(dict, key);
    if (!val) {
        return NULL;
    }
    return get_required_array(dict, key, ndim);
}

static int fill_poisson_pack(PyObject *pack_obj, PoissonPack *pack, PyArrayObject **apot_out) {
    PyArrayObject *apot = get_required_array(pack_obj, "apot", 2);
    if (!apot) {
        return -1;
    }
    memset(pack, 0, sizeof(*pack));
    if (get_required_int(pack_obj, "nr", &pack->nr) < 0
        || get_required_int(pack_obj, "lmax_active", &pack->lmax_active) < 0
        || get_required_double(pack_obj, "dr", &pack->dr) < 0
        || get_required_double(pack_obj, "psic", &pack->psic) < 0
        || get_optional_int(pack_obj, "has_disk", 0, &pack->has_disk) < 0
        || get_optional_int(pack_obj, "has_halo", 0, &pack->has_halo) < 0
        || get_optional_int(pack_obj, "has_bulge", 0, &pack->has_bulge) < 0) {
        Py_DECREF(apot);
        return -1;
    }
    pack->n_harm = pack->lmax_active / 2 + 1;
    pack->apot = (const double *)PyArray_DATA(apot);
    *apot_out = apot;

    if (pack->has_disk) {
        if (get_required_double(pack_obj, "disk_const", &pack->disk_const) < 0
            || get_required_double(pack_obj, "disk_rd", &pack->disk_rd) < 0
            || get_required_double(pack_obj, "disk_zd", &pack->disk_zd) < 0
            || get_required_double(pack_obj, "disk_rtrunc", &pack->disk_rtrunc) < 0
            || get_required_double(pack_obj, "disk_trunc_width", &pack->disk_trunc_width) < 0
            || get_required_double(pack_obj, "disk_hole_radius", &pack->disk_hole_radius) < 0
            || get_required_double(pack_obj, "disk_core_radius", &pack->disk_core_radius) < 0) {
            Py_DECREF(apot);
            return -1;
        }
    }

    PyArrayObject *halo_e = get_optional_array(pack_obj, "halo_energies", 1);
    PyArrayObject *halo_d = get_optional_array(pack_obj, "halo_dens_psi", 1);
    if (halo_e && halo_d) {
        pack->npsi_halo = (int)PyArray_DIM(halo_e, 0);
        pack->halo_energies = (const double *)PyArray_DATA(halo_e);
        pack->halo_dens_psi = (const double *)PyArray_DATA(halo_d);
        if (get_required_double(pack_obj, "psi0", &pack->psi0) < 0
            || get_required_double(pack_obj, "halo_dens_at_psi0", &pack->halo_dens_at_psi0) < 0) {
            Py_DECREF(apot);
            return -1;
        }
    }

    PyArrayObject *bulge_e = get_optional_array(pack_obj, "bulge_energies", 1);
    PyArrayObject *bulge_d = get_optional_array(pack_obj, "bulge_dens_psi", 1);
    if (bulge_e && bulge_d) {
        pack->npsi_bulge = (int)PyArray_DIM(bulge_e, 0);
        pack->bulge_energies = (const double *)PyArray_DATA(bulge_e);
        pack->bulge_dens_psi = (const double *)PyArray_DATA(bulge_d);
        if (get_required_double(pack_obj, "bulge_psi0", &pack->bulge_psi0) < 0
            || get_required_double(pack_obj, "bulge_psid", &pack->bulge_psid) < 0
            || get_required_double(pack_obj, "bulge_dens_at_psi0", &pack->bulge_dens_at_psi0) < 0) {
            Py_DECREF(apot);
            return -1;
        }
    }
    return 0;
}

static PyObject *py_extension_available(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
#ifdef _OPENMP
    return Py_True;
#else
    return Py_False;
#endif
}

static PyObject *py_fill_polar_density_harmonics(
    PyObject *self, PyObject *args, PyObject *kwargs
) {
    (void)self;
    static char *kwlist[] = {
        "adens", "pack", "radial_step", "active_lmax", "n_polar_nodes", "n_threads", NULL
    };
    PyArrayObject *adens_arr = NULL;
    PyObject *pack_obj = NULL;
    double radial_step = 0.0;
    int active_lmax = 0;
    int n_polar_nodes = 0;
    int n_threads = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O!O!diii",
            kwlist,
            &PyArray_Type,
            &adens_arr,
            &PyDict_Type,
            &pack_obj,
            &radial_step,
            &active_lmax,
            &n_polar_nodes,
            &n_threads)) {
        return NULL;
    }
    if (!(PyArray_FLAGS(adens_arr) & NPY_ARRAY_WRITEABLE)) {
        PyErr_SetString(PyExc_ValueError, "adens must be writeable");
        return NULL;
    }

    PoissonPack pack = {0};
    PyArrayObject *apot_arr = NULL;
    PyArrayObject *halo_e = NULL;
    PyArrayObject *halo_d = NULL;
    PyArrayObject *bulge_e = NULL;
    PyArrayObject *bulge_d = NULL;
    if (fill_poisson_pack(pack_obj, &pack, &apot_arr) < 0) {
        return NULL;
    }
    halo_e = get_optional_array(pack_obj, "halo_energies", 1);
    halo_d = get_optional_array(pack_obj, "halo_dens_psi", 1);
    bulge_e = get_optional_array(pack_obj, "bulge_energies", 1);
    bulge_d = get_optional_array(pack_obj, "bulge_dens_psi", 1);

    int rc = fill_polar_density_harmonics_omp(
        (double *)PyArray_DATA(adens_arr),
        &pack,
        radial_step,
        active_lmax,
        n_polar_nodes,
        n_threads
    );
    Py_DECREF(apot_arr);
    Py_XDECREF(halo_e);
    Py_XDECREF(halo_d);
    Py_XDECREF(bulge_e);
    Py_XDECREF(bulge_d);
    if (rc != 0) {
        PyErr_SetString(PyExc_RuntimeError, "polar density harmonic fill failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {
        "extension_available",
        (PyCFunction)py_extension_available,
        METH_NOARGS,
        "True if built with OpenMP",
    },
    {
        "fill_polar_density_harmonics",
        (PyCFunction)py_fill_polar_density_harmonics,
        METH_VARARGS | METH_KEYWORDS,
        "OpenMP polar shell integration",
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "galacticsics.potential.poisson._poisson_c",
    .m_doc = "OpenMP Poisson polar shell integration",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__poisson_c(void) {
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    return m;
}
