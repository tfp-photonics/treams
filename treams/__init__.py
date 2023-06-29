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
    PlaneWaveBasisByComp,
    PlaneWaveBasisByUnitVector,
    SphericalWaveBasis,
)
from treams._lattice import Lattice, WaveVector  # noqa: F401
from treams._material import Material  # noqa: F401
from treams._operators import (  # noqa: F401
    BField,
    ChangePoltype,
    DField,
    EField,
    Expand,
    ExpandLattice,
    FField,
    GField,
    HField,
    Permute,
    Rotate,
    Translate,
    bfield,
    changepoltype,
    dfield,
    efield,
    expand,
    expandlattice,
    ffield,
    gfield,
    hfield,
    permute,
    rotate,
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
    cylindrical_wave,
    plane_wave,
    plane_wave_angle,
    spherical_wave,
)
