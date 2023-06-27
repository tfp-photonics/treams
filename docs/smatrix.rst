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


From cylindrical T-matrix gratings
==================================


Band structures
===============
