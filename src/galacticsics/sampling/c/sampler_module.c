/*
 * sampler_module.c — Python bindings for OpenMP samplers.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "sampler.h"

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

static int fill_pot_pack(
    PyObject *pack, PotPack *pot, PyArrayObject **apot_out, PyArrayObject **plcon_out
) {
    PyArrayObject *apot = get_required_array(pack, "apot", 2);
    PyArrayObject *plcon = get_required_array(pack, "plcon", 1);
    if (!apot || !plcon) {
        Py_XDECREF(apot);
        Py_XDECREF(plcon);
        return -1;
    }
    if (get_required_int(pack, "nr", &pot->nr) < 0
        || get_required_int(pack, "n_harm", &pot->n_harm) < 0
        || get_required_int(pack, "lmax", &pot->lmax) < 0
        || get_required_int(pack, "has_disk", &pot->has_disk) < 0
        || get_required_double(pack, "dr", &pot->dr) < 0
        || get_required_double(pack, "psic", &pot->psic) < 0
        || get_required_double(pack, "disk_const", &pot->disk_const) < 0
        || get_required_double(pack, "disk_rd", &pot->disk_rd) < 0
        || get_required_double(pack, "disk_zd", &pot->disk_zd) < 0
        || get_required_double(pack, "disk_scale_height", &pot->disk_scale_height) < 0
        || get_required_double(pack, "disk_rtrunc", &pot->disk_rtrunc) < 0
        || get_required_double(pack, "disk_trunc_width", &pot->disk_trunc_width) < 0) {
        Py_DECREF(apot);
        Py_DECREF(plcon);
        return -1;
    }
    pot->apot = (const double *)PyArray_DATA(apot);
    pot->plcon = (const double *)PyArray_DATA(plcon);
    *apot_out = apot;
    *plcon_out = plcon;
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

static PyObject *py_sample_halo(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {
        "pack", "n_particles", "seed", "mass", "haloedge", "rhomax", "rhomin",
        "streaming", "n_threads", "max_attempts", "center", NULL
    };
    PyObject *pack_obj = NULL;
    int n_particles = 0, seed = -1, n_threads = 0, max_attempts = 0, center = 1;
    double mass = 0, haloedge = 0, rhomax = 0, rhomin = 0, streaming = 0.5;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Oiidddddiii", kwlist,
            &pack_obj, &n_particles, &seed, &mass, &haloedge, &rhomax, &rhomin,
            &streaming, &n_threads, &max_attempts, &center)) {
        return NULL;
    }
    if (!PyDict_Check(pack_obj)) {
        PyErr_SetString(PyExc_TypeError, "pack must be a dict");
        return NULL;
    }

    PotPack pot = {0};
    HaloParams halo = {0};
    DfPack df = {0};
    PyArrayObject *apot_arr = NULL;
    PyArrayObject *plcon_arr = NULL;
    if (fill_pot_pack(pack_obj, &pot, &apot_arr, &plcon_arr) < 0) {
        return NULL;
    }
    if (get_required_double(pack_obj, "halo_a", &halo.a) < 0
        || get_required_double(pack_obj, "halo_v0", &halo.v0) < 0
        || get_required_double(pack_obj, "halo_cusp", &halo.cusp) < 0
        || get_required_double(pack_obj, "halo_r_outer", &halo.r_outer) < 0
        || get_required_double(pack_obj, "halo_dr_trunc", &halo.dr_trunc) < 0
        || get_required_double(pack_obj, "df_fcut", &df.fcut) < 0) {
        return NULL;
    }
    df.psic = pot.psic;

    PyArrayObject *df_energy = get_required_array(pack_obj, "df_energy", 1);
    PyArrayObject *df_log = get_required_array(pack_obj, "df_log_df", 1);
    if (!df_energy || !df_log) {
        Py_XDECREF(df_energy);
        Py_XDECREF(df_log);
        return NULL;
    }
    df.n = (int)PyArray_DIM(df_energy, 0);
    df.energy = (const double *)PyArray_DATA(df_energy);
    df.log_df = (const double *)PyArray_DATA(df_log);
    df.fmax_cum = NULL;

    npy_intp dims[1] = {(npy_intp)(n_particles * 7)};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!out) {
        Py_DECREF(df_energy);
        Py_DECREF(df_log);
        return NULL;
    }

    SamplerOpts opts = {.n_threads = n_threads, .max_attempts = max_attempts, .center = center};
    int rc = sample_halo_omp(
        (double *)PyArray_DATA(out), n_particles, seed, mass, haloedge, rhomax, rhomin,
        streaming, &pot, &halo, &df, &opts
    );
    Py_DECREF(df_energy);
    Py_DECREF(df_log);
    Py_DECREF(apot_arr);
    Py_DECREF(plcon_arr);
    if (rc != 0) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "halo sampling failed");
        return NULL;
    }
    return (PyObject *)out;
}

static PyObject *py_sample_disk(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {
        "pack", "n_particles", "seed", "mass", "rd", "zd", "rtrunc", "rhomax", "rhomin",
        "n_threads", "max_attempts", "center", NULL
    };
    PyObject *pack_obj = NULL;
    int n_particles = 0, seed = -1, n_threads = 0, max_attempts = 0, center = 1;
    double mass = 0, rd = 0, zd = 0, rtrunc = 0, rhomax = 0, rhomin = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Oiiddddddiii", kwlist,
            &pack_obj, &n_particles, &seed, &mass, &rd, &zd, &rtrunc, &rhomax, &rhomin,
            &n_threads, &max_attempts, &center)) {
        return NULL;
    }
    if (!PyDict_Check(pack_obj)) {
        PyErr_SetString(PyExc_TypeError, "pack must be a dict");
        return NULL;
    }

    PotPack pot = {0};
    CorrPack corr = {0};
    FreqPack freq = {0};
    RcircPack rcirc = {0};
    PyArrayObject *apot_arr = NULL;
    PyArrayObject *plcon_arr = NULL;
    if (fill_pot_pack(pack_obj, &pot, &apot_arr, &plcon_arr) < 0) {
        return NULL;
    }
    if (get_required_double(pack_obj, "sigma_r0", &corr.sigma_r0) < 0
        || get_required_double(pack_obj, "sigma_r_scale", &corr.sigma_r_scale) < 0) {
        return NULL;
    }

    PyArrayObject *corr_r = get_required_array(pack_obj, "corr_radius", 1);
    PyArrayObject *corr_fd = get_required_array(pack_obj, "corr_f_d", 1);
    PyArrayObject *corr_fsz = get_required_array(pack_obj, "corr_f_sz", 1);
    PyArrayObject *freq_r = get_required_array(pack_obj, "freq_radius", 1);
    PyArrayObject *freq_o = get_required_array(pack_obj, "freq_omega", 1);
    PyArrayObject *freq_k = get_required_array(pack_obj, "freq_kappa", 1);
    PyArrayObject *rcirc_am = get_required_array(pack_obj, "rcirc_am", 1);
    PyArrayObject *rcirc_inv = get_required_array(pack_obj, "rcirc_inv_sqrt_am", 1);
    if (!corr_r || !corr_fd || !corr_fsz || !freq_r || !freq_o || !freq_k || !rcirc_am || !rcirc_inv) {
        Py_XDECREF(corr_r);
        Py_XDECREF(corr_fd);
        Py_XDECREF(corr_fsz);
        Py_XDECREF(freq_r);
        Py_XDECREF(freq_o);
        Py_XDECREF(freq_k);
        Py_XDECREF(rcirc_am);
        Py_XDECREF(rcirc_inv);
        return NULL;
    }
    corr.n = (int)PyArray_DIM(corr_r, 0);
    corr.radius = (const double *)PyArray_DATA(corr_r);
    corr.f_d = (const double *)PyArray_DATA(corr_fd);
    corr.f_sz = (const double *)PyArray_DATA(corr_fsz);
    freq.n = (int)PyArray_DIM(freq_r, 0);
    freq.radius = (const double *)PyArray_DATA(freq_r);
    freq.omega = (const double *)PyArray_DATA(freq_o);
    freq.kappa = (const double *)PyArray_DATA(freq_k);
    rcirc.n = (int)PyArray_DIM(rcirc_am, 0);
    rcirc.am = (const double *)PyArray_DATA(rcirc_am);
    rcirc.inv_sqrt_am = (const double *)PyArray_DATA(rcirc_inv);

    npy_intp dims[1] = {(npy_intp)(n_particles * 7)};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!out) {
        Py_DECREF(corr_r);
        Py_DECREF(corr_fd);
        Py_DECREF(corr_fsz);
        Py_DECREF(freq_r);
        Py_DECREF(freq_o);
        Py_DECREF(freq_k);
        Py_DECREF(rcirc_am);
        Py_DECREF(rcirc_inv);
        return NULL;
    }

    SamplerOpts opts = {.n_threads = n_threads, .max_attempts = max_attempts, .center = center};
    int rc = sample_disk_omp(
        (double *)PyArray_DATA(out), n_particles, seed, mass, rd, zd, rtrunc, rhomax, rhomin,
        &pot, &freq, &corr, &rcirc, &opts
    );
    Py_DECREF(corr_r);
    Py_DECREF(corr_fd);
    Py_DECREF(corr_fsz);
    Py_DECREF(freq_r);
    Py_DECREF(freq_o);
    Py_DECREF(freq_k);
    Py_DECREF(rcirc_am);
    Py_DECREF(rcirc_inv);
    Py_DECREF(apot_arr);
    Py_DECREF(plcon_arr);
    if (rc != 0) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "disk sampling failed");
        return NULL;
    }
    return (PyObject *)out;
}

static PyObject *py_sample_bulge(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {
        "pack", "n_particles", "seed", "mass", "bulgeedge", "wmax", "wmin",
        "streaming", "n_threads", "max_attempts", "center", NULL
    };
    PyObject *pack_obj = NULL;
    int n_particles = 0, seed = -1, n_threads = 0, max_attempts = 0, center = 1;
    double mass = 0, bulgeedge = 0, wmax = 0, wmin = 0, streaming = 0.5;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Oiidddddiii", kwlist,
            &pack_obj, &n_particles, &seed, &mass, &bulgeedge, &wmax, &wmin,
            &streaming, &n_threads, &max_attempts, &center)) {
        return NULL;
    }
    if (!PyDict_Check(pack_obj)) {
        PyErr_SetString(PyExc_TypeError, "pack must be a dict");
        return NULL;
    }

    PotPack pot = {0};
    BulgeParams bulge = {0};
    DfPack df = {0};
    PyArrayObject *apot_arr = NULL;
    PyArrayObject *plcon_arr = NULL;
    if (fill_pot_pack(pack_obj, &pot, &apot_arr, &plcon_arr) < 0) {
        return NULL;
    }
    if (get_required_double(pack_obj, "bulge_n", &bulge.n) < 0
        || get_required_double(pack_obj, "bulge_ppp", &bulge.ppp) < 0
        || get_required_double(pack_obj, "bulge_Re", &bulge.Re) < 0
        || get_required_double(pack_obj, "bulge_butt", &bulge.butt) < 0
        || get_required_double(pack_obj, "bulge_rho0", &bulge.rho0) < 0
        || get_required_double(pack_obj, "df_fcut", &df.fcut) < 0) {
        Py_DECREF(apot_arr);
        Py_DECREF(plcon_arr);
        return NULL;
    }
    df.psic = pot.psic;

    PyArrayObject *df_energy = get_required_array(pack_obj, "df_energy", 1);
    PyArrayObject *df_log = get_required_array(pack_obj, "df_log_df", 1);
    PyArrayObject *df_fmax = get_required_array(pack_obj, "df_fmax_cum", 1);
    if (!df_energy || !df_log || !df_fmax) {
        Py_XDECREF(df_energy);
        Py_XDECREF(df_log);
        Py_XDECREF(df_fmax);
        Py_DECREF(apot_arr);
        Py_DECREF(plcon_arr);
        return NULL;
    }
    df.n = (int)PyArray_DIM(df_energy, 0);
    df.energy = (const double *)PyArray_DATA(df_energy);
    df.log_df = (const double *)PyArray_DATA(df_log);
    df.fmax_cum = (const double *)PyArray_DATA(df_fmax);

    npy_intp dims[1] = {(npy_intp)(n_particles * 7)};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!out) {
        Py_DECREF(df_energy);
        Py_DECREF(df_log);
        Py_DECREF(df_fmax);
        Py_DECREF(apot_arr);
        Py_DECREF(plcon_arr);
        return NULL;
    }

    SamplerOpts opts = {.n_threads = n_threads, .max_attempts = max_attempts, .center = center};
    int rc = sample_bulge_omp(
        (double *)PyArray_DATA(out), n_particles, seed, mass, bulgeedge, wmax, wmin,
        streaming, &pot, &bulge, &df, &opts
    );
    Py_DECREF(df_energy);
    Py_DECREF(df_log);
    Py_DECREF(df_fmax);
    Py_DECREF(apot_arr);
    Py_DECREF(plcon_arr);
    if (rc != 0) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "bulge sampling failed");
        return NULL;
    }
    return (PyObject *)out;
}

static PyMethodDef module_methods[] = {
    {"extension_available", (PyCFunction)py_extension_available, METH_NOARGS, "True if built with OpenMP"},
    {"sample_halo", (PyCFunction)py_sample_halo, METH_VARARGS | METH_KEYWORDS, "OpenMP halo sampler"},
    {"sample_disk", (PyCFunction)py_sample_disk, METH_VARARGS | METH_KEYWORDS, "OpenMP disk sampler"},
    {"sample_bulge", (PyCFunction)py_sample_bulge, METH_VARARGS | METH_KEYWORDS, "OpenMP bulge sampler"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "galacticsics.sampling._sampler_c",
    .m_doc = "OpenMP particle samplers",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__sampler_c(void) {
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    return m;
}
