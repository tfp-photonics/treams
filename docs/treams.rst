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

   sw
   cw
   pw

This module is for loading and storing data in HDF5 files and also for creating meshes
of sphere ensembles.

.. autosummary::
   :toctree: generated/

   io

Finally, two modules for calculating scattering coefficients and doing miscellaneous
tasks.

.. autosummary::
   :toctree: generated/

   coeffs
   misc

Global configuration variables are stored in

.. autosummary::
   :toctree: generated/

   config

Some convenience classes for the implementation are defined in

.. autosummary::
   :toctree: generated/

   util

Subpackages
===========

These subpackages allow a low-level access to the implementation of the lattice sums and
mathematical functions

.. autosummary::
   :toctree: generated/

   lattice
   special
