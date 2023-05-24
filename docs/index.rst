=================================
Welcome to treams' documentation!
=================================

.. toctree::
   :maxdepth: 1

   gettingstarted
   examples
   intro
   theory
   treams
   dev
   about

The package **treams** provides a framework to simplify computations of the
electromagnetic scattering of waves at finite and periodically infinite arrangements of
particles. All methods are suitable for the use of chiral materials. The periodic
systems can have one-, two-, or three-dimensional lattices. The lattice computations are
accelerated by converting the occurring slowly converging summations to exponentially
fast convergent series.

To accommodate the periodic structures of different dimensionalities, three types of
solutions to the vectorial Helmholtz equation are employed: plane waves, cylindrical
waves, and spherical waves. For each of those solution sets, the typical manipulations,
e.g. translations and rotations, are implemented, as well as transformations between
them.

The package contains two subpackages: :mod:`lattice` and :mod:`special`. The former
contains mainly the functions for computing the lattice series. The latter can be seen
as an addition to the special functions implemented in :py:mod:`scipy.special`. It
contains the mathematical functions that are typically necessary in T-Matrix method
computations.

Finally, three classes are the main point of interaction for the user. They allow access
to the underlying functions operating directly on the spherical and cylindrical
T-matrices or the Q-matrices based on the plane wave solutions.

.. todo:: clean up intro

Features
========

* T-matrix calculations using a spherical or cylindrical wave basis set
* Calculations in helicity and parity (TE/TM) basis
* Scattering from clusters of particles
* Scattering from particles and clusters arranged in 3d-, 2d-, and 1d-lattices
* Calculation of light propagation in stratified media
* Band calculation in crystal structures


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
