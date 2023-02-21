Reference
=========

.. automodule:: treams
   :no-members:
   :no-inherited-members:
   :no-special-members:


Subpackages
-----------

.. autosummary::
   :toctree: generated/
   :template: bare-module

   treams.lattice
   treams.special


Modules
=======

These modules provide basic functionality for transformations within one basis set, i.e.
one module, like translations and rotations as well as transformations among them.
The functions in there provide an intermediate stage between the purely mathematical
functions found in the two subpackages :mod:`lattice` and :mod:`special` and the
higher-level classes.

.. toctree::
   :maxdepth: 1

   treams.sw
   treams.cw
   treams.pw

This module is for loading and storing data in HDF5 files and also for creating meshes
of sphere ensembles.

.. toctree::
   :maxdepth: 1

   treams.io

Finally, two modules for calculating scattering coefficients and doing miscellaneous
tasks.

.. toctree::
   :maxdepth: 1

   treams.coeffs
   treams.misc

Global configuration variables are stored in

.. toctree::
   :maxdepth: 1

   treams.config

.. autosummary::
   :toctree: generated/

   treams.util
