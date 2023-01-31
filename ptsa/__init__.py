"""
============
Package ptsa
============

.. currentmodule:: ptsa

Classes
=======

.. autosummary::
   :toctree: generated/

   TMatrix
   TMatrixC
   QMatrix
   PlaneWaveBasis
   SphericalWaveBasis
   CylindricalWaveBasis
   Lattice
   Material
   PhysicsArray


Modules
=======

These modules provide basic functionality for transformations within one basis set, i.e.
one module, like translations and rotations as well as transformations among them.
The functions in there provide an intermediate stage between the purely mathematical
functions found in the two subpackages :mod:`lattice` and :mod:`special` and the
higher-level classes.

.. toctree::
   :maxdepth: 1

   ptsa.sw
   ptsa.cw
   ptsa.pw

This module is for loading and storing data in HDF5 files and also for creating meshes
of sphere ensembles.

.. toctree::
   :maxdepth: 1

   ptsa.io

Finally, two modules for calculating scattering coefficients and doing miscellaneous
tasks.

.. toctree::
   :maxdepth: 1

   ptsa.coeffs
   ptsa.misc
"""

from ptsa._core import (  # noqa: F401
    CylindricalWaveBasis,
    PhysicsArray,
    PlaneWaveBasis,
    PlaneWaveBasisPartial,
    SphericalWaveBasis,
)
from ptsa._lattice import Lattice  # noqa: F401
from ptsa._material import Material  # noqa: F401
from ptsa._operators import (  # noqa: F401
    changepoltype,
    efield,
    expand,
    expandlattice,
    permute,
    rotate,
    translate,
)
from ptsa._qmatrix import QMatrix  # noqa: F401
from ptsa._tmatrix import TMatrix  # noqa: F401
from ptsa._tmatrixc import TMatrixC  # noqa: F401
