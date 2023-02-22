=========
Reference
=========

.. automodule:: treams
   :no-members:
   :no-inherited-members:
   :no-special-members:


Modules
=======

These modules provide basic functionality for transformations within one basis set, i.e.
one module, like translations and rotations as well as transformations among them.
The functions in there provide an intermediate stage between the purely mathematical
functions found in the two subpackages :mod:`lattice` and :mod:`special` and the
higher-level classes and functions.

.. autosummary::
   :toctree: generated/
   :template: bare-module

   ~treams.sw
   ~treams.cw
   ~treams.pw

This module is for loading and storing data in HDF5 files and also for creating meshes
of sphere ensembles.

.. autosummary::
   :toctree: generated/

   ~treams.io

Finally, two modules for calculating scattering coefficients and doing miscellaneous
tasks.

.. autosummary::
   :toctree: generated/

   ~treams.coeffs
   ~treams.misc

Global configuration variables are stored in

.. autosummary::
   :toctree: generated/

   ~treams.config

Some convenience classes for the implementation are defined in

.. autosummary::
   :toctree: generated/

   ~treams.util


Subpackages
===========

These subpackages allow a low-level access to the implementation of the lattice sums and
mathematical functions

.. autosummary::
   :toctree: generated/
   :template: bare-module

   ~treams.lattice
   ~treams.special
