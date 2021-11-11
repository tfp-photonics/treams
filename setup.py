"""Packaging of ptsa"""
import os
from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

keys = {"include_dirs": [np.get_include()]}
compiler_directives = {}
if os.environ.get("CYTHON_COVERAGE", False):
    keys["define_macros"] = [("CYTHON_TRACE_NOGIL", "1")]
    compiler_directives["linetrace"] = True

extensions = [Extension("*", ["ptsa/{,lattice/,special/}*.pyx"], **keys,)]
setup(
    name="ptsa",
    version="0.1",
    packages=["ptsa", "ptsa.special", "ptsa.lattice"],
    install_requires=["numpy", "scipy>=1.6", "cython"],
    extras_require={"test": ["pytest", "pytest-cov"], "docs": ["sphinx"]},
    ext_modules=cythonize(
        extensions, language_level="3", compiler_directives=compiler_directives
    ),
    package_data={
        "ptsa/lattice": ["cython_lattice.pxd"],
        "ptsa/special": ["cython_special.pxd"],
    },
)
