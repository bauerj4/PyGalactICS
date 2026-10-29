/*
 * sampler_module.c — Python bindings for OpenMP samplers.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "sampler.h"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>

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
    PyObject *pack,
    PotPack *pot,
    PyArrayObject **apot_out,
    PyArrayObject **plcon_out,
    PyArrayObject **disk_norm_radii_out,
    PyArrayObject **disk_norm_values_out
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
    if (pot->has_disk) {
        PyArrayObject *norm_radii = get_required_array(
            pack, "disk_vertical_norm_radii", 1
        );
        PyArrayObject *norm_values = get_required_array(
            pack, "disk_vertical_norm_values", 1
        );
        if (!norm_radii || !norm_values) {
            Py_XDECREF(norm_radii);
            Py_XDECREF(norm_values);
            Py_DECREF(apot);
            Py_DECREF(plcon);
            return -1;
        }
        if (PyArray_DIM(norm_radii, 0) != PyArray_DIM(norm_values, 0)
            || PyArray_DIM(norm_radii, 0) < 1) {
            Py_DECREF(norm_radii);
            Py_DECREF(norm_values);
            Py_DECREF(apot);
            Py_DECREF(plcon);
            PyErr_SetString(
                PyExc_ValueError,
                "disk vertical normalization arrays must have equal nonzero length"
            );
            return -1;
        }
        pot->n_disk_vertical_norm = (int)PyArray_DIM(norm_radii, 0);
        pot->disk_vertical_norm_radii = (const double *)PyArray_DATA(norm_radii);
        pot->disk_vertical_norm_values = (const double *)PyArray_DATA(norm_values);
        *disk_norm_radii_out = norm_radii;
        *disk_norm_values_out = norm_values;
    }
    *apot_out = apot;
    *plcon_out = plcon;
    return 0;
}

static void release_pot_pack_arrays(
    PyArrayObject *apot,
    PyArrayObject *plcon,
    PyArrayObject *disk_norm_radii,
    PyArrayObject *disk_norm_values
) {
    Py_XDECREF(apot);
    Py_XDECREF(plcon);
    Py_XDECREF(disk_norm_radii);
    Py_XDECREF(disk_norm_values);
}

static int fill_spatial_pack(
    PyObject *pack,
    SpatialPack *spatial,
    PyArrayObject **radius_out,
    PyArrayObject **envelope_out,
    PyArrayObject **cdf_out,
    PyArrayObject **psi_out,
    PyArrayObject **density_out
) {
    PyArrayObject *radius = get_required_array(pack, "spatial_radius", 1);
    PyArrayObject *envelope = get_required_array(pack, "spatial_envelope", 1);
    PyArrayObject *cdf = get_required_array(pack, "spatial_cdf", 1);
    PyArrayObject *psi = get_required_array(pack, "spatial_psi", 1);
    PyArrayObject *density = get_required_array(pack, "spatial_density", 1);
    if (!radius || !envelope || !cdf || !psi || !density) {
        Py_XDECREF(radius);
        Py_XDECREF(envelope);
        Py_XDECREF(cdf);
        Py_XDECREF(psi);
        Py_XDECREF(density);
        return -1;
    }
    if (PyArray_DIM(radius, 0) != PyArray_DIM(envelope, 0)
        || PyArray_DIM(radius, 0) != PyArray_DIM(cdf, 0)
        || PyArray_DIM(radius, 0) < 2
        || PyArray_DIM(psi, 0) != PyArray_DIM(density, 0)
        || PyArray_DIM(psi, 0) < 2) {
        Py_DECREF(radius);
        Py_DECREF(envelope);
        Py_DECREF(cdf);
        Py_DECREF(psi);
        Py_DECREF(density);
        PyErr_SetString(
            PyExc_ValueError,
            "spatial profile arrays must have matching nontrivial lengths"
        );
        return -1;
    }
    spatial->n_radius = (int)PyArray_DIM(radius, 0);
    spatial->radius = (const double *)PyArray_DATA(radius);
    spatial->envelope = (const double *)PyArray_DATA(envelope);
    spatial->cdf = (const double *)PyArray_DATA(cdf);
    spatial->n_psi = (int)PyArray_DIM(psi, 0);
    spatial->psi = (const double *)PyArray_DATA(psi);
    spatial->density = (const double *)PyArray_DATA(density);
    *radius_out = radius;
    *envelope_out = envelope;
    *cdf_out = cdf;
    *psi_out = psi;
    *density_out = density;
    return 0;
}

static void release_spatial_pack_arrays(
    PyArrayObject *radius,
    PyArrayObject *envelope,
    PyArrayObject *cdf,
    PyArrayObject *psi,
    PyArrayObject *density
) {
    Py_XDECREF(radius);
    Py_XDECREF(envelope);
    Py_XDECREF(cdf);
    Py_XDECREF(psi);
    Py_XDECREF(density);
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
        "pack", "n_particles", "seed", "mass", "streaming", "n_threads",
        "max_attempts", "center", NULL
    };
    PyObject *pack_obj = NULL;
    int n_particles = 0, seed = -1, n_threads = 0, max_attempts = 0, center = 1;
    double mass = 0, streaming = 0.5;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Oiiddiii", kwlist,
            &pack_obj, &n_particles, &seed, &mass, &streaming,
            &n_threads, &max_attempts, &center)) {
        return NULL;
    }
    if (!PyDict_Check(pack_obj)) {
        PyErr_SetString(PyExc_TypeError, "pack must be a dict");
        return NULL;
    }

    PotPack pot = {0};
    SpatialPack spatial = {0};
    DfPack df = {0};
    PyArrayObject *apot_arr = NULL;
    PyArrayObject *plcon_arr = NULL;
    PyArrayObject *disk_norm_radii = NULL;
    PyArrayObject *disk_norm_values = NULL;
    if (fill_pot_pack(
            pack_obj,
            &pot,
            &apot_arr,
            &plcon_arr,
            &disk_norm_radii,
            &disk_norm_values) < 0) {
        return NULL;
    }
    if (get_required_double(pack_obj, "df_fcut", &df.fcut) < 0) {
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
        return NULL;
    }
    df.psic = pot.psic;

    PyArrayObject *spatial_radius = NULL;
    PyArrayObject *spatial_envelope = NULL;
    PyArrayObject *spatial_cdf = NULL;
    PyArrayObject *spatial_psi = NULL;
    PyArrayObject *spatial_density = NULL;
    if (fill_spatial_pack(
            pack_obj,
            &spatial,
            &spatial_radius,
            &spatial_envelope,
            &spatial_cdf,
            &spatial_psi,
            &spatial_density) < 0) {
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
        return NULL;
    }

    PyArrayObject *df_energy = get_required_array(pack_obj, "df_energy", 1);
    PyArrayObject *df_log = get_required_array(pack_obj, "df_log_df", 1);
    PyArrayObject *df_fmax = get_required_array(pack_obj, "df_fmax_cum", 1);
    if (!df_energy || !df_log || !df_fmax) {
        Py_XDECREF(df_energy);
        Py_XDECREF(df_log);
        Py_XDECREF(df_fmax);
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
        release_spatial_pack_arrays(
            spatial_radius, spatial_envelope, spatial_cdf, spatial_psi,
            spatial_density
        );
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
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
        release_spatial_pack_arrays(
            spatial_radius, spatial_envelope, spatial_cdf, spatial_psi,
            spatial_density
        );
        return NULL;
    }

    SamplerOpts opts = {.n_threads = n_threads, .max_attempts = max_attempts, .center = center};
    int rc = sample_halo_omp(
        (double *)PyArray_DATA(out), n_particles, seed, mass,
        streaming, &pot, &spatial, &df, &opts
    );
    Py_DECREF(df_energy);
    Py_DECREF(df_log);
    Py_DECREF(df_fmax);
    release_pot_pack_arrays(
        apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
    );
    release_spatial_pack_arrays(
        spatial_radius, spatial_envelope, spatial_cdf, spatial_psi,
        spatial_density
    );
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
    PyArrayObject *disk_norm_radii = NULL;
    PyArrayObject *disk_norm_values = NULL;
    if (fill_pot_pack(
            pack_obj,
            &pot,
            &apot_arr,
            &plcon_arr,
            &disk_norm_radii,
            &disk_norm_values) < 0) {
        return NULL;
    }
    if (get_required_double(pack_obj, "sigma_r0", &corr.sigma_r0) < 0
        || get_required_double(pack_obj, "sigma_r_scale", &corr.sigma_r_scale) < 0) {
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
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
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
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
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
        return NULL;
    }

    SamplerOpts opts = {.n_threads = n_threads, .max_attempts = max_attempts, .center = center};
    double failure_detail[4] = {0.0, 0.0, 0.0, 0.0};
    int rc = sample_disk_omp(
        (double *)PyArray_DATA(out), n_particles, seed, mass, rd, zd, rtrunc, rhomax, rhomin,
        &pot, &freq, &corr, &rcirc, &opts, failure_detail
    );
    Py_DECREF(corr_r);
    Py_DECREF(corr_fd);
    Py_DECREF(corr_fsz);
    Py_DECREF(freq_r);
    Py_DECREF(freq_o);
    Py_DECREF(freq_k);
    Py_DECREF(rcirc_am);
    Py_DECREF(rcirc_inv);
    release_pot_pack_arrays(
        apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
    );
    if (rc != 0) {
        Py_DECREF(out);
        const char *message = "disk sampling failed";
        switch (rc) {
            case -1: message = "disk sampling exceeded the spatial attempt limit"; break;
            case -2: message = "disk spatial rejection envelope was exceeded"; break;
            case -3: message = "disk velocity envelope is non-positive"; break;
            case -4: message = "disk velocity rejection envelope was exceeded"; break;
            case -5: message = "disk velocity sampling exceeded 20000 attempts"; break;
            case -6: message = "disk sampler stopped before filling the output"; break;
        }
        char detail_message[320];
        snprintf(
            detail_message,
            sizeof(detail_message),
            "%s (detail: %.9g, %.9g, %.9g, %.9g)",
            message,
            failure_detail[0],
            failure_detail[1],
            failure_detail[2],
            failure_detail[3]
        );
        PyErr_SetString(PyExc_RuntimeError, detail_message);
        return NULL;
    }
    return (PyObject *)out;
}

static PyObject *py_sample_bulge(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {
        "pack", "n_particles", "seed", "mass", "streaming", "n_threads",
        "max_attempts", "center", NULL
    };
    PyObject *pack_obj = NULL;
    int n_particles = 0, seed = -1, n_threads = 0, max_attempts = 0, center = 1;
    double mass = 0, streaming = 0.5;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Oiiddiii", kwlist,
            &pack_obj, &n_particles, &seed, &mass, &streaming,
            &n_threads, &max_attempts, &center)) {
        return NULL;
    }
    if (!PyDict_Check(pack_obj)) {
        PyErr_SetString(PyExc_TypeError, "pack must be a dict");
        return NULL;
    }

    PotPack pot = {0};
    SpatialPack spatial = {0};
    DfPack df = {0};
    PyArrayObject *apot_arr = NULL;
    PyArrayObject *plcon_arr = NULL;
    PyArrayObject *disk_norm_radii = NULL;
    PyArrayObject *disk_norm_values = NULL;
    if (fill_pot_pack(
            pack_obj,
            &pot,
            &apot_arr,
            &plcon_arr,
            &disk_norm_radii,
            &disk_norm_values) < 0) {
        return NULL;
    }
    if (get_required_double(pack_obj, "df_fcut", &df.fcut) < 0) {
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
        return NULL;
    }
    df.psic = pot.psic;

    PyArrayObject *spatial_radius = NULL;
    PyArrayObject *spatial_envelope = NULL;
    PyArrayObject *spatial_cdf = NULL;
    PyArrayObject *spatial_psi = NULL;
    PyArrayObject *spatial_density = NULL;
    if (fill_spatial_pack(
            pack_obj,
            &spatial,
            &spatial_radius,
            &spatial_envelope,
            &spatial_cdf,
            &spatial_psi,
            &spatial_density) < 0) {
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
        return NULL;
    }

    PyArrayObject *df_energy = get_required_array(pack_obj, "df_energy", 1);
    PyArrayObject *df_log = get_required_array(pack_obj, "df_log_df", 1);
    PyArrayObject *df_fmax = get_required_array(pack_obj, "df_fmax_cum", 1);
    if (!df_energy || !df_log || !df_fmax) {
        Py_XDECREF(df_energy);
        Py_XDECREF(df_log);
        Py_XDECREF(df_fmax);
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
        release_spatial_pack_arrays(
            spatial_radius, spatial_envelope, spatial_cdf, spatial_psi,
            spatial_density
        );
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
        release_pot_pack_arrays(
            apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
        );
        release_spatial_pack_arrays(
            spatial_radius, spatial_envelope, spatial_cdf, spatial_psi,
            spatial_density
        );
        return NULL;
    }

    SamplerOpts opts = {.n_threads = n_threads, .max_attempts = max_attempts, .center = center};
    int rc = sample_bulge_omp(
        (double *)PyArray_DATA(out), n_particles, seed, mass,
        streaming, &pot, &spatial, &df, &opts
    );
    Py_DECREF(df_energy);
    Py_DECREF(df_log);
    Py_DECREF(df_fmax);
    release_pot_pack_arrays(
        apot_arr, plcon_arr, disk_norm_radii, disk_norm_values
    );
    release_spatial_pack_arrays(
        spatial_radius, spatial_envelope, spatial_cdf, spatial_psi,
        spatial_density
    );
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
