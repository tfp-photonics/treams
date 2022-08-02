"""Packaging of ptsa"""
import os

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


if os.name == "nt":

    link_args = [
        "-static-libgcc",
        "-static-libstdc++",
        "-Wl,-Bstatic,--whole-archive",
        "-lwinpthread",
        "-Wl,--no-whole-archive",
    ]
    compile_args = ["-DMS_WIN64"]

    class build_ext(_build_ext):
        # Enforce the gcc compiler under windows
        def finalize_options(self):
            super().finalize_options()
            self.compiler = "mingw32"

        # https://cython.readthedocs.io/en/latest/src/tutorial/appendix.html
        def build_extensions(self):
            if self.compiler.compiler_type == "mingw32":
                for e in self.extensions:
                    e.extra_compile_args = compile_args
                    e.extra_link_args = link_args
            super().build_extensions()


else:
    build_ext = _build_ext


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


keys = {"include_dirs": [np.get_include()]}
compiler_directives = {"language_level": "3"}
if os.environ.get("CYTHON_COVERAGE", False):
    keys["define_macros"] = [("CYTHON_TRACE_NOGIL", "1")]
    compiler_directives["linetrace"] = True

extension_names = [
    "ptsa.coeffs",
    "ptsa.config",
    "ptsa.cw",
    "ptsa.pw",
    "ptsa.sw",
    "ptsa.lattice._dsum",
    "ptsa.lattice._esum",
    "ptsa.lattice._gufuncs",
    "ptsa.lattice._misc",
    "ptsa.lattice.cython_lattice",
    "ptsa.special._bessel",
    "ptsa.special._coord",
    "ptsa.special._gufuncs",
    "ptsa.special._integrals",
    "ptsa.special._misc",
    "ptsa.special._ufuncs",
    "ptsa.special._waves",
    "ptsa.special._wigner3j",
    "ptsa.special._wignerd",
    "ptsa.special.cython_special",
]
extensions = [
    Extension(name, [f"{name.replace('.', '/')}.pyx"], **keys)
    for name in extension_names
]

if cythonize is not None:
    try:
        extensions = cythonize(extensions, compiler_directives=compiler_directives)
    except ValueError:
        extensions = no_cythonize(extensions)
else:
    extensions = no_cythonize(extensions)

setup(ext_modules=extensions, cmdclass={"build_ext": build_ext})
