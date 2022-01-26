.. highlight:: python

========
Examples
========

The computations implemented in ptsa mainly revolve around three different types of
matrices: the (spherical) T-matrix, the cylindrical T-matrix and the Q-matrices. These
three types of matrices emerge from using either spherical, cylindrical, or plane waves
as solutions to Maxwell's equations. The examples are sorted into these three topics.
Each of these topics start with a short introduction on the particular type of matrix
before using it.

In all examples we assume that ``import numpy as np`` and ``import ptsa`` is used.


T-matrix examples
=================

Particle clusters
-----------------

We want to build a small snowman made of four spheres. Two for the body and head and two
smaller ones for each hand::

    k0 = 2 * np.pi / 1000 # Wave number in vacuum
    epsilon_air = 1
    epsilon_snow = 16 + 1j # Permittivity of our snow
    lmax = 4  # Multipole order
    radii = [
        150,  # body
        100,  # head
        40,  # right hand
        30,  # left hand
    ]

Here we simply set the wavelength to `1000`. For our constant material parameters we can
choose the scale of wavelength and sphere radii arbitrarily, but for our example we
could simply assume e.g. the unit of the wavelength and the radii to be nanometers.
Up to now we only defined parameters, now we want to create the T-matrices. For
convenience we store the T-matrices in a simple list::

    snowballs = [
        ptsa.TMatrix.sphere(lmax, k0, r, [epsilon_snow, epsilon_air]) for r in radii
    ]

Examining the call first invocation of `ptsa` in this example shows a simple structure.
We use the :class:`ptsa.TMatrix` for our object and then define the sphere. As a
minimum the maximal multipole order has to be given, then the vacuum wave number. Then,
we define the radius and finally the relative permittivities. The materials are given
from the inside to the outside.

.. note::

  The method :func:`ptsa.TMatrix.sphere` is actually more general. It can calculate the
  T-matrix of spherical core-shell particles with an arbitrary number of shells. Also,
  each material can be chiral.

Now, we want to create a cluster of these T-matrices::

    positions = [
        [0, 0, -100],  # body
        [0, 0, 150],  # head
        [-85, 145, -5],  # right hand
        [-80, -135, -7.5],  # left hand
    ]
    snowman = ptsa.TMatrix.cluster(snowballs, positions)
    snowman.interact()

We first create the cluster by giving the T-matrices of all its constituent particles
and specifying their positions. It is advised to always put the origin in a symmetry
point if possible. The invocation of :func:`ptsa.TMatrix.cluster` does create a
block-diagonal T-matrix from all its constituents. However, the interaction between all
these particles has to computed in a separate step by using the method
:func:`ptsa.TMatrix.interact`. Now, ``snowman`` contains the *local* T-matrix. This
means that the full scattered field is described by spherical waves at multiple origins.
This has the benefit, that one can usually obtain correct fields closer to the geometric
object.

Now, we can calculate the scattering and extinction cross section of our little snowman
under a plane wave illumination along the x-axis::

    illu = snowman.illuminate_pw(k0, 0, 0, 0)
    xs = snowman.xs(illu)
    print(f"scattering cross section: {xs[0]}")
    print(f"extinction cross section: {xs[1]}")

Next, we can look at the field distribution. For this we first create a grid, that is
separated into three parts. First, the space occupied by the spheres. Here, we cannot
easily calculate the fields. Second, the part that is outside the individual spheres but
inside of the circumscribing spheres. Here, we can calculate the fields using the local
T-matrix. Third, the part outside of the circumscribing sphere. Here, we can calculate
the fields using the local or a global T-matrix. Which calculation is more beneficial
and faster is determined by mainly by how many orders need to be included in both cases.
For the sake of a more diverse example we will do the latter::

    grid = np.mgrid[-300:300:201j, 0:1, -300:300:201j].squeeze().transpose((1, 2, 0))
    scattered_field = np.zeros_like(grid, complex)
    outside = np.sum(np.power(grid, 2), axis=-1) > 250 * 250
    in_between = np.logical_and(
        np.logical_not(outside),
        np.logical_and(
            np.sum(np.power(grid - positions[0], 2), axis=-1) > radii[0] * radii[0],
            np.sum(np.power(grid - positions[1], 2), axis=-1) > radii[1] * radii[1],
        )
    )
    scattered_field_coeff = tm.field(grid[in_between, :])
    scattered_field[in_between, :] = np.sum(scattered_field_coeff * (tm.t @ illu), axis=-1)






3D-arrays of particles
----------------------

Cylindrical T-matrix examples
=============================

Clusters
--------

1D-arrays of particles
----------------------


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
