.. highlight:: python

.. only:: builder_html

========
Examples
========

.. contents:: Table of contents
   :local:

The computations implemented in ptsa mainly revolve around three different types of
matrices: the (spherical) T-matrix, the cylindrical T-matrix and the S-matrices. These
three types of matrices emerge from using either spherical, cylindrical, or plane waves
as solutions to Maxwell's equations. The examples are sorted into these three topics.
Each of these topics start with a short introduction on the particular type of matrix
before using it.

In all examples we assume that ``import numpy as np`` and ``import ptsa`` is used.

T-matrix examples
=================

Particle cluster
----------------

We want to build a small snowman made of four spheres. Two for the body and head and two
smaller ones for each hand:

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 6-10

Here, we simply set the wavelength to `1000`. For our constant material parameters we
can choose the scale of wavelength and sphere radii arbitrarily, but for our example we
could simply assume e.g. the unit of the wavelength and the radii to be nanometers.
Up to now we only defined parameters, now we want to create the T-matrices. For
convenience we store the T-matrices in a simple list:

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 12

Examining the call of `ptsa` in this example shows a simple structure. We use the
:class:`ptsa.TMatrix` for our object and then define the sphere. As a minimum the
maximal multipole order has to be given, then the vacuum wave number. Then, we define
the radius and finally the relative permittivities. The materials are given from the
inside to the outside.

.. note::

  The method :func:`ptsa.TMatrix.sphere` is actually more general. It can calculate the
  T-matrix of spherical core-shell particles with an arbitrary number of shells. Also,
  each material can be chiral.

Now, we want to create a cluster of these T-matrices:

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 14-15

We first create the cluster by giving the T-matrices of all its constituent particles
and specifying their positions. It is advised to always put the origin in a symmetry
point if possible. The invocation of :func:`ptsa.TMatrix.cluster` does create a
block-diagonal T-matrix from all its constituents. However, the interaction between all
these particles has to computed in a separate step by using the method
:func:`ptsa.TMatrix.interacted`. Now, ``snowman`` contains the *local* T-matrix. This
means that the full scattered field is described by spherical waves at multiple origins.
This has the benefit, that one can usually obtain correct fields closer to the geometric
object.

Now, we can calculate the scattering and extinction cross section of our little snowman
under a plane wave illumination along the x-axis:

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 17-19

Next, we can look at the field distribution. For this we first create a grid, that is
separated into three parts. First, the space occupied by the spheres, here, we cannot
easily calculate the fields. Second, the part that is outside the individual spheres but
inside of the circumscribing spheres. Here, we can calculate the fields using the local
T-matrix. Third, the part outside of the circumscribing sphere, where we can calculate
the fields using the local or the global T-matrix. Which calculation is more beneficial
and faster is determined by mainly by how many orders need to be included in both cases.
For the sake of a more diverse example we will do the latter:

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 21-27

Mostly, we set up here a grid of points and separate the different regions. Finally, we
take the region outside of the circumscribing sphere of the snowman. Using the global
T-Matrix we calculate the scattererd field coefficients. In the last line, we multiply
those coefficients with the corresponding values of the efield of those modes.

Now we want to calculate the field between the spheres for which we work with the local
T-matrix and mostly repeat the previous steps.

.. literalinclude:: examples/cluster.py
   :language: python
   :lines: 29-31

Now we can plot the full field around our snowman. We only add the illuminating plane
wave and arrive at:

.. .. plot:: examples/cluster.py

We remark, that the field outside the circumscribing sphere (red dashed line) does
nicely fit with the field inside, indicating that the chosen expansion order for the
global T-matrix was sufficient.

3D-arrays of particles
----------------------

Cylindrical T-matrix examples
=============================

Clusters of cylindrical rods
----------------------------

Due to the similarity of T-matrices and cylindrical T-matrices, we can build clusters
analogously. For this example we take infinitely extended cylinders, for which the
T-matrix entries can be computed analytically.

We begin with defining the cylinder parameters. For the embedding medium we choose a
non-zero value of the chirality parameter.

.. literalinclude:: examples/cylindercluster.py
   :language: python
   :lines: 7-16

In contrast to the case of spheres and spherical T-matrices, we now need to define the
value of the z-component of the wave vector. Multiple values have to be given. Later, we
will use one-dimensional periodic arrays (chains). Then, it's necessary that the
different values correspond to a set of diffraction orders. Here, however, we only take
a single value for our example. Additionally to the wave vector component the maximal
azimuthal order is chosen to be 4. So, `m` runs from -4 to +4 in the T-matrix modes.

The next step is similar to the cluster example. The most important difference is, that
we use for the illumination a property of the cluster `ks` which holds the wave numbers
in the chiral medium. This is done just out of convenience.

.. literalinclude:: examples/cylindercluster.py
   :language: python
   :lines: 18

Finally, we just set up a suitable grid and plot the results. This time we calculate
the whole field in the local system, but we could have also chosen to change to the
global T-matrix for the computation of the field outside of the T-matrix.

.. literalinclude:: examples/cylindercluster.py
   :language: python
   :lines: 20-27

.. .. plot:: examples/cylindercluster.py

1D-arrays of particles
----------------------

This, example now turns to an one-dimensional array of spheres (chain). The coupling
between the spheres will be calculated with the Ewald method. Then, one can transition
from a description of coupled spheres, described by T-matrices to a description using
the cylindrical T-matrix.

We start by defining two spheres, that are within one unit cell. This means we have a
nontrivial unit cell for our system, resulting finally in a local cylindrical T-matrix.

.. literalinclude:: examples/chain.py
   :language: python
   :lines: 6-15

The definition of the unit cell is exactly the same to the definition of a regular
cluster. Now, one only needs to call the interaction calculation within the lattice,
which for a chain is simply defined by its pitch. Also, one needs to specify the
z-component of the wave vector (up to a reciprocal lattice vector).

We will illuminate the chain with a plane wave traveling in z-direction. To calculate
the field, we define again different areas. By using the lattice sums for arbitrary
points in the unit cell, it is possible to calculate the all positions outside of the
spheres. However, this is comparably slow in comparison to the calculation using the
cylindrical wave expansion. This expansion is only valid outside of the circumscribing
cylinder with infinite length in z-direction.

.. literalinclude:: examples/chain.py
   :language: python
   :lines: 17-19

Having the definition of the different areas out of the way, one can do the calculation
now for the different points. First, we have to define the expansion modes around the
point, where we probe the field at. 

.. .. plot:: examples/chain.py

Q-matrix examples
=================

Interfaces and homogeneous media
--------------------------------

Metasurfaces
------------

Gratings
--------

Band structure
--------------

Loading and storing
===================
