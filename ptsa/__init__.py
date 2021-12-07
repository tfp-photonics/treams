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

from ptsa._qmatrix import QMatrix
from ptsa._tmatrix import TMatrix
from ptsa._tmatrixc import TMatrixC
