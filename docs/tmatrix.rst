.. highlight:: python

.. only:: builder_html

========
T-Matrix
========

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

It's possible to calculate the T-matrix of a single (mulit-layered) sphere with the
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
   :lines: 7-18

we can simply first create the spheres and put them together in a cluster, where we
immediately calculate the interaction.

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 20-21

Then, we can illuminate with a plane wave and get the scattered field coefficients and
the scattering and extinction cross sections for that particular illumination.

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 23-25

Finally, with few lines similar to the plotting of the field intensity of a single
sphere we can obtain the fields outside of the sphere

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 27-33

Up to here, we did all calculations for the cluster in the local basis. By expanding
the incident and scattered fields in a basis with a single origin we can describe the
same object. Often, a larger number of multipoles is needed to do so and some
information on fields between the particles is lost. But, the description in a global
basis can be more efficient in terms of matrix size.

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 58-69

A comparison of the calculated near-fields and the cross sections show good agreement
between the results of both, local and global, T-matrices.

.. plot:: examples/cluster.py

In the last figure, the T-matrix is rotated by 90 degrees about the y-axis and the
illumination is set accordingly to be a plane wave propagating in the negative
z-direction, such that the whole system is the same simply. It shows how the rotate
operator produces consistent results.

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 94-106


One-dimensional arrays (along z)
================================

Two-dimensional arrays (in the x-y-plane)
=========================================

Three-dimensional arrays
========================

