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
   PlaneWaveBasisByUnitVector
   PlaneWaveBasisByComp
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
    PlaneWaveBasisByUnitVector,
    PlaneWaveBasisByComp,
    SphericalWaveBasis,
)
from treams._lattice import Lattice, WaveVector  # noqa: F401
from treams._material import Material  # noqa: F401
from treams._operators import (  # noqa: F401
    BField,
    bfield,
    ChangePoltype,
    changepoltype,
    DField,
    dfield,
    EField,
    efield,
    Expand,
    expand,
    ExpandLattice,
    expandlattice,
    FField,
    ffield,
    GField,
    gfield,
    HField,
    hfield,
    Permute,
    permute,
    Rotate,
    rotate,
    Translate,
    translate,
)
from treams._smatrix import (  # noqa: F401
    SMatrices,
    SMatrix,
    chirality_density,
    poynting_avg_z,
)
from treams._tmatrix import (  # noqa: F401
    TMatrix,
    TMatrixC,
    plane_wave,
    plane_wave_angle,
    cylindrical_wave,
    spherical_wave,
)
