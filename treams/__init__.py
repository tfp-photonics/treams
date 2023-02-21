"""TREAMS: T-Matrix scattering code for nanophotonic computations.

.. currentmodule:: treams

Classes
=======

The top-level classes and functions allow a high-level access to the functionality.

Basis sets
----------

.. autosummary::
   :toctree: generated/

   CylindricalWaveBasis
   PlaneWaveBasis
   PlaneWaveBasisPartial
   SphericalWaveBasis

Matrices and Arrays
-------------------

.. autosummary::
   :toctree: generated/

   PhysicsArray
   SMatrix
   TMatrix
   TMatrixC

Other
-----

.. autosummary::
   :toctree: generated/

   Lattice
   Material

Functions
=========

.. autosummary::
   :toctree: generated/

   bfield
   changepoltype
   dfield
   efield
   expand
   expandlattice
   hfield
   permute
   rotate
   translate

"""

from treams._core import (  # noqa: F401
    CylindricalWaveBasis,
    PhysicsArray,
    PlaneWaveBasis,
    PlaneWaveBasisPartial,
    SphericalWaveBasis,
)
from treams._lattice import Lattice  # noqa: F401
from treams._material import Material  # noqa: F401
from treams._operators import (  # noqa: F401
    bfield,
    changepoltype,
    dfield,
    efield,
    expand,
    expandlattice,
    hfield,
    permute,
    rotate,
    translate,
)
from treams._smatrix import SMatrix  # noqa: F401
from treams._tmatrix import TMatrix, TMatrixC, plane_wave, spherical_wave  # noqa: F401
