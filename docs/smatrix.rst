.. highlight:: python

.. only:: builder_html

==========================
S-Matrices for plane waves
==========================

In addition to the T-matrix method, where incident and scattered fields are related,
S-matrices relate incoming and outgoing fields. To describe scattering in the plane
wave basis, we use exactly such a S-matrix description. The incoming and outgoing waves
are defined with respect to a plane, typically the x-y-plane. This plane additionally
separates the whole space into two parts. This then separates the description into
four parts: the transmission of fields propagating in the positive and negative
z-direction, and the reflection of those fields.

Slabs
=====

The main object for plane wave computations is the class :class:`~treams.SMatrices`
which exactly collects those four individual S-matrices. For simple interfaces and the
propagation in a homogeneous medium these S-matrices can be obtained analytically.
Combining these two objects then allows the description of simple slabs.

.. literalinclude:: examples/slab.py
   :language: python
   :lines: 6-14

The setup is fairly simple. The materials are given in order from negative to positive
z-values. We simply loop over the wave number and calculate transmission and reflection
in the chiral medium for both helicites.

.. plot:: examples/slab.py

From T-matrix arrays
====================

While this example is simple we can build more complicated structures from
two-dimensional arrays of T-matrices. We take spheres on an thin film as an example.
This means we first calculate the S-matrices for the thin film and the array
individually and then couple those two systems.

.. literalinclude:: examples/array_spheres.py
   :language: python
   :lines: 6-12

Beforehand, we define all the necessary parameters. First the wave numbers, then the
parameters of the slab, and finally those for the lattice and the spheres. Then, we can
use a simple loop to solve the system for all wave numbers.

.. literalinclude:: examples/array_spheres.py
   :language: python
   :lines: 14-27

We set some oblique incidence and the array of spheres. Then, we define a linearly
polarized plane wave and the needed S-matrices: a slab, the distance between the
top interface of the slab to the center of the sphere array, and the array in the
S-matrix representation itself.

.. plot:: examples/array_spheres.py

From cylindrical T-matrix gratings
==================================

Now, we want to perform a small test of the methods. Instead of creating the
two-dimensional sphere array right away, we intermediately create a one-dimensional
array, then calculate cylindrical T-matrices, and place them in a second
one-dimensional lattice, thereby, obtaining the S-matrix from the previous section.

.. literalinclude:: examples/array_spheres_tmatrixc.py
   :language: python
   :lines: 6-13

The definition of the parameters is quite similar. We store the lattice pitch for later
use separately and define the maximal order :code:`mmax` separately.

.. literalinclude:: examples/array_spheres_tmatrixc.py
   :language: python
   :lines: 15-32

The first half of the loop is now a little bit different. After creating the spheres
we solve the interaction along the z-direction, then create the cylindrical T-matrix
and finally calculate the interaction along the x-direction. The second half is the
same as in the previous calculation.

The most important aspect to note here, is that the method
:meth:`treams.SMatrix.from_array` implicitly converts the lattice in the z-x-plane to
a lattice in the x-y-plane.

.. plot:: examples/array_spheres_tmatrixc.py


Band structure
==============

Finally, we want to compute the band structure of a system consisting of the periodic
repetition of an S-matrix along the z-direction. In principle, one can obtain this
band structure also from the lattice interaction in the T-matrix, but calculating it
from the S-matrix has two benefits. First, more complex systems can be analyzed, because
slabs and objects described by cylindrical T-matrices can be included. Second, one only
defines :math:`k_0`, :math:`k_x`, and :math:`k_y`. Then, the result of the calculation
are all components :math:`k_z` and the plane wave decomposition of the polarization from
an eigenvalue decomposition. So, instead of a 4-dimensional parameter sweep only a
3-dimensional sweep is necessary decreasing the computation time. The downside is, that
one is restricted to unit cells, where one vector points along the z-axis and is
perpendicular to the other two.

We take the array of spheres on top of a slab and continue this one infinitely along the
z-axis. Thus, the setup is

.. literalinclude:: examples/band_structure.py
   :language: python
   :lines: 6-13

where :code:`az` is the length of the lattice vector pointing in the z-direction. With a
simple loop we can get the band structure for :math:`k_x = 0 = k_y`.

.. literalinclude:: examples/band_structure.py
   :language: python
   :lines: 15-31

which looks as follows, after a cut on the imaginary part of the :math:`k_z` component.

.. plot:: examples/band_structure.py
