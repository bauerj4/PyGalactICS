"""Build configuration for galacticsics OpenMP sampler extension."""

from setuptools import Extension, setup

import numpy as np

setup(
    ext_modules=[
        Extension(
            "galacticsics.sampling._sampler_c",
            sources=[
                "src/galacticsics/sampling/c/sampler_impl.c",
                "src/galacticsics/sampling/c/sampler_module.c",
            ],
            include_dirs=[np.get_include(), "src/galacticsics/sampling/c"],
            extra_compile_args=["-O3", "-std=c11", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
        Extension(
            "galacticsics.potential.poisson._poisson_c",
            sources=[
                "src/galacticsics/potential/poisson/c/poisson_impl.c",
                "src/galacticsics/potential/poisson/c/poisson_module.c",
            ],
            include_dirs=[np.get_include(), "src/galacticsics/potential/poisson/c"],
            extra_compile_args=["-O3", "-std=c11", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
    ],
    package_dir={"": "src"},
)
