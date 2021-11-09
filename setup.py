"""Packaging of ptsa"""
from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

# For linetracing:
# from Cython.Compiler.Options import get_directive_defaults
# directive_defaults = get_directive_defaults()
# directive_defaults['linetrace'] = True

extensions = [
    Extension(
        "*",
        ["ptsa/{,lattice/,special/}*.pyx"],
        include_dirs=[np.get_include()],
        # define_macros=[('CYTHON_TRACE_NOGIL', '1')],  # for linetracing
    )
]
setup(
    name="ptsa",
    version="0.1",
    packages=["ptsa", "ptsa.special", "ptsa.lattice"],
    install_requires=["numpy", "scipy>=1.6", "cython"],
    extras_require={"test": ["pytest", "pytest-cov"], "docs": ["sphinx"]},
    ext_modules=cythonize(extensions, language_level="3"),
    package_data={
        "ptsa/lattice": ["cython_lattice.pxd"],
        "ptsa/special": ["cython_special.pxd"],
    },
)
