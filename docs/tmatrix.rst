.. highlight:: python

.. only:: builder_html

==========
T-Matrices
==========

.. contents:: Table of contents
   :local:

One of the main objects for electromagnetic scattering calculations within *treams* are
T-matrices. They describe the scattering response of an object by encoding the linear
relationship between incident and scattered fields. These fields are expanded using
the vector spherical wave functions.

The T-matrices of spheres or multiple layers of spherical shells, present for example in
core-shell particles, can be obtained analytically. For more complicated shapes
numerical methods are necessary to compute the T-matrix. Once the T-matrix of a single
object is known, the electromagnetic interaction between particle cluster can be
calculated efficiently. Such clusters can be analyzed in their local description, where
the field expansions are centered at each particle of the cluster, or in a global
description treating the whole cluster as a single object.

*treams* is particularly aimed at analyzing scattering within lattices. These lattices
can be periodic in one, two, or all three spatial dimensions. The unit cell of those
lattices can consist of an arbitrary number of objects described by a T-matrix.


Spheres
=======

It's possible to calculate the T-matrix of a single (multi-layered) sphere with the
method :meth:`~treams.TMatrix.sphere`. We start by defining the relevant parameters for
our calculation and creating the T-matrices themselves.

.. literalinclude:: examples/sphere.py
   :language: python
   :lines: 6-10

Now, we can easily access quantities like the scattering and extinction cross
sections

.. literalinclude:: examples/sphere.py
   :language: python
   :lines: 12-13

From the parameter ``lmax = 4`` we see that the T-matrix is calculated up to the forth
multipolar order. To restrict the T-matrix to the dipolar coefficients only, we can
select a basis containing only those coefficients.

.. literalinclude:: examples/sphere.py
   :language: python
   :lines: 15-18

Now, we can look at the results by plotting them and observe, unsurprisingly, that for
larger frequencies the dipolar approximation is not giving an accurate result. Finally,
we visualize the fields at the largest frequency.

.. literalinclude:: examples/sphere.py
   :language: python
   :lines: 20-29

We select the T-matrix and illuminate it with a plane wave. Next, we set up the grid
and choose valid points, namely those that are outside of our spherical object. Then,
we can calculate the fields as a superposition of incident and scattered fields.

.. plot:: examples/sphere.py


Clusters
========

Multi-scattering calculations in a cluster of particles is a typical application of the
T-matrix method. We first construct an object from four different spheres placed at the
corners of a tetrahedron. Using the definition of the relevant parameters

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 6-17

we can simply first create the spheres and put them together in a cluster, where we
immediately calculate the interaction.

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 19-20

Then, we can illuminate with a plane wave and get the scattered field coefficients and
the scattering and extinction cross sections for that particular illumination.

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 22-24

Finally, with few lines similar to the plotting of the field intensity of a single
sphere we can obtain the fields outside of the sphere

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 26-32

Up to here, we did all calculations for the cluster in the local basis. By expanding
the incident and scattered fields in a basis with a single origin we can describe the
same object. Often, a larger number of multipoles is needed to do so and some
information on fields between the particles is lost. But, the description in a global
basis can be more efficient in terms of matrix size.

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 57-68

A comparison of the calculated near-fields and the cross sections show good agreement
between the results of both, local and global, T-matrices.

.. plot:: examples/cluster.py

In the last figure, the T-matrix is rotated by 90 degrees about the y-axis and the
illumination is set accordingly to be a plane wave propagating in the negative
z-direction, such that the whole system is the same simply. It shows how the rotate
operator produces consistent results.

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 83-95


One-dimensional arrays (along z)
================================

Next, we turn to systems that are periodic in the z-direction. We calculate the
scattering from an array of spheres. Intentionally, we choose a unit cell with two
spheres that overlap along the z-direction, but are not placed exactly along the same
line. This is the most general case for the implemented lattice sums. After the common
setup of the parameters we simply create a cluster in a local basis.

.. literalinclude:: examples/chain.py
   :language: python
   :lines: 6-12

This time we let them interact specifying a one-dimensional lattice, so that the spheres
form a chain.

.. literalinclude:: examples/chain.py
   :language: python
   :lines: 14-15

Next, we choose set the illumination to be propagating along the x-direction and to be
polarized along z. The z-component of the plane wave has to match to the wave vector
component of the lattice interaction.

.. literalinclude:: examples/chain.py
   :language: python
   :lines: 17-18

There are efficient ways to calculate the many properties, especially in the far-field,
using cylindrical T-matrices. Those will be introduced in :doc:`tmatrixc`. Here, we will
stay in the expression of the fields as vector spherical waves. This allows the
calculation of the fields in the domain between the spheres. To get them accurately, we
expand the scattered fields in the whole lattice in dipolar approximation at each point
we want to probe.

.. literalinclude:: examples/chain.py
   :language: python
   :lines: 20-29

Here, we plot the z-component of the electric field. Note, that the values at the top
and bottom side match exactly, as required due to the periodic boundary conditions.

.. plot:: examples/chain.py

Two-dimensional arrays (in the x-y-plane)
=========================================

The case of periodicity in two directions is similar to the case of the previous section
with one-dimensional periodicity. Here, by convention the array has to be in the
x-y-plane.

.. literalinclude:: examples/grid.py
   :language: python
   :lines: 6-31

With few changes we get the fields in a square array of the same spheres as in the
previous examples. Most importantly we changed the value of the variable :code:`lattice`
to an instance of a two-dimensional :class:`~treams.Lattice` and set :code:`kpar`
accordingly. Most other changes are just resulting from the change of the coordinate
system.

Here, we show the z-component of the electric field.

.. plot:: examples/grid.py

Three-dimensional arrays
========================

In a three-dimensional lattice we're mostly concerned with finding eigenmodes of a
crystal. We want to restrict the example to calculating modes at the gamma point in
reciprocal space. The calculated system consists of a single sphere in a cubic lattice.
In our very crude analysis, we blindly select the lowest singular value of the lattice
interaction matrix. Besides the mode when the frequency tends to zero, there are two
additional modes at higher frequencies in the chosen range.

.. literalinclude:: examples/crystal.py
   :language: python
   :lines: 6-17

.. plot:: examples/crystal.py

