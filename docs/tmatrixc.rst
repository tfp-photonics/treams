.. highlight:: python

.. only:: builder_html

======================
Cylindrical T-matrices
======================

.. contents:: Table of contents
   :local:

Here, we will cover cylindrical T-matrices, which are distinguished from the more
conventional (spherical) T-matrices through the use of vector cylindrical functions
instead of vector spherical functions. These waves are parametrized by the z-component
of the wave vector :math:`k_z`, that describes their behavior in the z-direction
:math:`\mathrm e^{\mathrm i k_z z}`, and the azimuthal order :math:`m`, that is also
used in vector spherical functions. Furthermore, there modes of well-defined parity and
helicity are available.

The cylindrical T-matrices are suited for structures that are periodic in one dimension
(conventionally set along the z-axis). Similarly to T-matrices of spheres that contain
the analytically known Mie coefficients, the cylindrical T-matrices for infinitely long
cylinders can also be calculated analytically.

Another similarity to spherical T-matrices are the possibilities to describe clusters of
objects in a local and global basis and to place these objects in a lattice. The
lattices can only extend in one and two dimensions; the z-direction is implicitly
periodic already.

(Infinitely long) Cylinders
===========================

The first simple object, for which we'll calculate the cylindrical T-matrix is an
infinitely long cylinder. (Similar to the case of spheres and T-matrices those
cylinders could also have multiple concentric shells.) Due to the rotation symmetry
about the z-axis this matrix is diagonal with respect to :code:`m` and due to the
translation symmetry it is also diagonal with respect to :code:`kz`.

.. literalinclude:: examples/cylinder.py
    :language: python
    :lines: 6-11

For such infinitely long structures it makes more sense to talk about cross width
instead of cross section. We obtain the averaged scattering and extinction cross width
by 

.. literalinclude:: examples/cylinder.py
    :language: python
    :lines: 13-15

Again, we can also select specific modes only, for example the modes with :math:`m = 0`.

.. literalinclude:: examples/cylinder.py
    :language: python
    :lines: 16-19

to calculate their cross width. Evaluating the field intensity in the x-y-plane is
equally simple.

.. literalinclude:: examples/cylinder.py
    :language: python
    :lines: 16-19

.. plot:: examples/cylinder.py

Cylindrical T-matrices for one-dimensional arrays of spherical T-matrices
=========================================================================

For our next example we want to look at the system of spheres on a one-dimensional
lattice again (:ref:`tmatrix:One-dimensional arrays (along z)`). They fulfil all
properties that define structures where the use of cylindrical waves is beneficial,
namely they have a finite extent in the x-y-plane and they are periodic along the
x-y-direction.

So, the initial setup of our calculation starts with spheres in the spherical wave
basis.



Clusters
========

One-dimensional arrays (along the x-axis)
=========================================

Two-dimensional arrays (in the x-y-plane)
=========================================
