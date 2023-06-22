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

.. literalinclude:: examples/cylinder_tmatrixc.py
    :language: python
    :lines: 6-11

For such infinitely long structures it makes more sense to talk about cross width
instead of cross section. We obtain the averaged scattering and extinction cross width
by 

.. literalinclude:: examples/cylinder_tmatrixc.py
    :language: python
    :lines: 13-15

Again, we can also select specific modes only, for example the modes with :math:`m = 0`.

.. literalinclude:: examples/cylinder_tmatrixc.py
    :language: python
    :lines: 16-19

to calculate their cross width. Evaluating the field intensity in the x-y-plane is
equally simple.

.. literalinclude:: examples/cylinder_tmatrixc.py
    :language: python
    :lines: 16-19

.. plot:: examples/cylinder_tmatrixc.py

Cylindrical T-matrices for one-dimensional arrays of spherical T-matrices
=========================================================================

For our next example we want to look at the system of spheres on a one-dimensional
lattice again (:ref:`tmatrix:One-dimensional arrays (along z)`). They fulfil all
properties that define structures where the use of cylindrical waves is beneficial,
namely they have a finite extent in the x-y-plane and they are periodic along the
z-direction.

So, the initial setup of our calculation starts with spheres in the spherical wave
basis and place them in a chain. This is the same procedure as in
:ref:`tmatrix:One-dimensional arrays (along z)`.

.. literalinclude:: examples/cylinder_tmatrixc.py
    :language: python
    :lines: 6-15

Next, we convert this chain in the spherical wave basis to a suitable cylindrical wave
basis.

.. literalinclude:: examples/cylinder_tmatrixc.py
    :language: python
    :lines: 17-19

We chose to add the first three diffraction orders (plus a 0.1 margin to avoid problems
with floating point comparisons).

Finally, we set-up the illumination and calculate the scattering with the usual
procedure.

.. literalinclude:: examples/cylinder_tmatrixc.py
    :language: python
    :lines: 21-27

We evaluate the fields in two regions. Outside of the circumscribing cylinders we can
use the fast cylindrical wave expansion. Inside of the circumscribing cylinders but
outside of the spheres we can use the method of
:ref:`tmatrix:One-dimensional arrays (along z)`.

Finally, we can plot the results. To illustrate the periodicity better, three unit cells
are shown.

.. plot:: examples/chain_tmatrixc.py


Clusters
========

Similarly to the case of spheres we can also calculate the response from a cluster of
objects. For the example want to simulate a cylinder together with a chain of spheres
in the cylindrical wave basis as described in the previous section.

So, we set up first the spheres in the chain and convert them to the cylindrical wave
basis as before

.. literalinclude:: examples/cluster_tmatrixc.py
    :language: python
    :lines: 6-18

Then, we create the cylinder T-matrix in the cylindrical wave basis

.. literalinclude:: examples/cluster_tmatrixc.py
    :language: python
    :lines: 20

Finally, we construct the cluster and let the interaction be solved

.. literalinclude:: examples/cluster_tmatrixc.py
    :language: python
    :lines: 22-29

and we solve calculate the scattered field coefficients, whose field representation we
then plot

.. plot:: examples/cluster_tmatrixc.py


One-dimensional arrays (along the x-axis)
=========================================

Two-dimensional arrays (in the x-y-plane)
=========================================
