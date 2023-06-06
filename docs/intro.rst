============
Introduction
============

*treams* is a program that covers various aspects of T-matrix calculations and
associated topics. The functionality can be roughly separated into three levels:
low-level functions, intermediate-level functions, and high-level functions and classes.

The low-level functions implement the underlying mathematical functions, that build
the foundation of T-matrix calculations. They are mainly located in the two subpackages
:mod:`treams.special` and :mod:`treams.lattice`. The first one contains, e.g., the
various solutions to the Helmholtz equation and their translation coefficients, the
second subpackage contains functions that are associated with computations in lattices.

On the intermediate-level those underlying functions are combined to provide functions
as they are often needed for T-matrix calculations, e.g., the Mie coefficients or the
expansion coefficients of vector plane waves into spherical waves. The low- and
intermediate-level functions are mostly focused on speed, they are usually implemented
in compiled code and are often vectorized. The latter aspect also helps with the
integration into the framework provided by numpy.

The high-level functionality is more focused on the usability. We attempt to create an
useful interface to the underlying functions, that reduces redundancy and that is less
error prone than using the pure functions, while still integrating nicely with numpy
functions. It consists of a combination of different classes and functions. At first,
there are the different basis sets, that can be used together with other important
parameters for the calculation, for example the embedding materials or the lattice
definitions. Then, there are the "physics-aware arrays", which keep track of these
parameters during the computation, and operators that can be applied to these arrays.
Finally, we will introduce, how these previous concepts can be applied to T-Matrices for
vector spherical and cylindrical solutions and to S-Matrices for vector plane wave
solutions.

.. toctree::
    :maxdepth: 1

    params
    physicsarray
    operators
    tmatrix

.. operators, tmatrix, tmatrixc, smatrix
