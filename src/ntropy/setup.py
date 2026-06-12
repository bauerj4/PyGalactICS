"""Build configuration for ntropy C extensions."""

from setuptools import Extension, setup

import numpy as np

setup(
    ext_modules=[
        Extension(
            "ntropy.forces._bh_c",
            sources=[
                "ntropy/forces/c/bh_tree.c",
                "ntropy/forces/c/bh_module.c",
            ],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-std=c11"],
        ),
    ],
)
