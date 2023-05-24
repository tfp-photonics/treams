====================
Physics-aware arrays
====================

.. testsetup::

   import numpy as np
   import treams

A core building block of the underlying features of treams are physics-aware arrays.
In most of their properties they behave similar to numpy arrays and one can easily
change the type and mix them

.. doctest::

    >>> np.array([1, 2]) * treams.PhysicsArray([2, 3])
    PhysicsArray(
        [2, 6],
    )
    >>> np.array([1, 2]) @ treams.PhysicsArray([2, 3])
    8

but they have mainly two features added. First, they derive from
:class:`treams.util.AnnotatedArray` so they can carry annotations with them, but several
keywords have a special meaning and are treated slightly differently. Second, they offer
special methods to create matrices for common transformations like rotations.

Special properties
==================

.. doctest::

    >>> treams.PhysicsArray([[0, 1], [2, 3]], k0=(1, 2))
    PhysicsArray(
        [[0, 1],
         [2, 3]],
        k0=(1.0, 2.0),
    )

While these annotations can have arbitrary names, some of them get a special meaning for
physics-aware arrays. You might have noticed that here the values were converted from
integers to floats. This happens because `k0` is interpreted as the vacuum wave number,
which in general is a real number. Thus, trying to use
:code:`tream.PhysicsArray([1], k0=1j)` will raise an error, because the complex number
cannot be interpreted as a float. Additional special keywords are `basis`, `kpar`,
`lattice`, `material`, `modetype`, and `poltype`. These properties can also be accessed
by setting the corresponding attribute

.. doctest:: 

    >>> m = treams.PhysicsArray([1, 2])
    >>> m.material = 4
    >>> m
    PhysicsArray(
        [1, 2],
        material=Material(4, 1, 0),
    )

where we now have a material with the relative permittivity 4. As with its parent class
these properties are also compared and merged when using operations on these objects

.. doctest::

    >>> treams.PhysicsArray([0, 1], k0=1) + treams.PhysicsArray([2, 3], material=2)
    PhysicsArray(
        [2, 4],
        k0=1.0,
        material=Material(2, 1, 0),
    )

and using conflicting values will raise a warning, for example

.. doctest::

    >>> treams.PhysicsArray([0, 1], k0=1) + treams.PhysicsArray([2, 3], k0=2)
    PhysicsArray(
        [2, 4],
        k0=1.0,
    )

emits :code:`treams/util.py:249: AnnotationWarning: at index 0: overwriting key 'k0'`.
The special properties have also a unique behavior when appearing in matrix
multiplications. If one of the two matrices has the special property not set, it becomes
"transparent" to it. Check out the difference between

.. doctest::

    >>> np.ones((2, 2)) @ treams.PhysicsArray([1, 2], k0=1.0)
    PhysicsArray(
        [3., 3.],
        k0=1.0,
    )

and 

.. doctest::

    >>> np.ones((2, 2)) @ treams.util.AnnotatedArray([1, 2], k0=(1.0,))
    AnnotatedArray(
        [3., 3.],
        AnnotationSequence(AnnotationDict({}),)
    )

where besides the obvious difference in array types, the property `k0` is preserved.

The full list of special properties is:

======== ===========================================================
Name     Description
======== ===========================================================
basis    Basis set: spherical, cylindrical, planar
k0       Vacuum wave number
kpar     Phase relation in lattices
lattice  Definition of a lattice (:class:`treams.Lattice`)
modetype Modetype, depends on wave (:ref:`params:Mode types`)
material Embedding material (:class:`treams.Material`)
poltype  Polarization types (:ref:`params:Polarizations`)
======== ===========================================================

Transformations
===============

The transformations of a the values in a array are usually represented by matrices. To
simplify creating these transformations, several operators are defined. We take the
example of a plane wave and want to expand it in the spherical basis. One way to create
such a plane wave could be:

.. doctest::

    >>> wave = treams.PhysicsArray([1.5], basis=treams.PlaneWaveBasisByComp([(3, 4, 0)]))
    >>> wave.expand(treams.SphericalWaveBasis.default(1))
    PhysicsArray(
        [[ 0.00000000e+00+0.j        ],
         [-4.19262711e+00+3.14447033j],
         [ 0.00000000e+00+0.j        ],
         [-1.87982067e-16-3.06998012j],
         [ 0.00000000e+00+0.j        ],
         [ 7.19341088e-01+0.53950582j]],
        basis=(SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ), PlaneWaveBasisByComp(
        kx=[3],
        ky=[4],
        pol=[0],
    )),
        material=Material(1, 1, 0),
        modetype=('regular', None),
        poltype='helicity',
    )

In the first line we define a plane wave by specifying a basis that consists of only one
entry namely that for a wave vector :math:`\boldsymbol k = 3 \boldsymbol{\hat x} +
4 \boldsymbol{\hat y} + 5 \boldsymbol{\hat z}` and the polarization 0
(:ref:`params:Polarizations`). The amplitude of the wave is 1.5. Then, we want
to expand the wave into the spherical basis up to dipolar order. The matrix that
represents this transformation is then returned. It contains as the given bases.
Additionally, a couple of other values are specified. This is, first, the material,
which defaults to vacuum (or air in good approximation). Second the mode type, which
depends on the basis sets used. Here, a plane wave is expanded into regular modes. The
last parameter added automatically is the polarization type. The default can be changed
by setting :code:`treams.config.POLTYPE = "parity"` for example.

Another transformation are translations, where additionally the inverse transformation
is implemented. We see that the result is roughly the unit matrix

.. doctest::

    >>> wave = treams.PhysicsArray(np.eye(6),
    ...    basis=treams.SphericalWaveBasis.default(1),
    ...    k0=1/100,
    ...    )
    >>> wave.translate([1, 2, 3]) @ wave @ wave.translate.inv([1, 2, 3])
    PhysicsArray(
        [[ 9.99777524e-01+1.73472348e-18j,  0.00000000e+00+0.00000000e+00j,
          -1.06049048e-05-2.12098096e-05j,  0.00000000e+00+0.00000000e+00j,
          -7.49880009e-06+9.99840012e-06j,  0.00000000e+00+0.00000000e+00j],
         [ 0.00000000e+00+0.00000000e+00j,  9.99777524e-01-1.73472348e-18j,
           0.00000000e+00+0.00000000e+00j, -1.06049048e-05-2.12098096e-05j,
           0.00000000e+00+0.00000000e+00j, -7.49880009e-06+9.99840012e-06j],
         [-1.06049048e-05+2.12098096e-05j,  0.00000000e+00+0.00000000e+00j,
           9.99745030e-01-2.29625731e-20j,  0.00000000e+00+0.00000000e+00j,
           1.06049048e-05+2.12098096e-05j,  0.00000000e+00+0.00000000e+00j],
         [ 0.00000000e+00+0.00000000e+00j, -1.06049048e-05+2.12098096e-05j,
           0.00000000e+00+0.00000000e+00j,  9.99745030e-01-2.25394264e-20j,
           0.00000000e+00+0.00000000e+00j,  1.06049048e-05+2.12098096e-05j],
         [-7.49880009e-06-9.99840012e-06j,  0.00000000e+00+0.00000000e+00j,
           1.06049048e-05-2.12098096e-05j,  0.00000000e+00+0.00000000e+00j,
           9.99777524e-01-1.73472348e-18j,  0.00000000e+00+0.00000000e+00j],
         [ 0.00000000e+00+0.00000000e+00j, -7.49880009e-06-9.99840012e-06j,
           0.00000000e+00+0.00000000e+00j,  1.06049048e-05-2.12098096e-05j,
           0.00000000e+00+0.00000000e+00j,  9.99777524e-01+1.73472348e-18j]],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=0.01,
        material=Material(1, 1, 0),
        poltype='helicity',
    )

as expected. Other transformations are:

============= =================================================
Name          Description
============= =================================================
rotate        Rotation
translate     Translation
expand        Expand in another basis
changepoltype Switch between parity and helicity polarizations
expandlattice Expand in another basis assuming a periodic array
============= =================================================
