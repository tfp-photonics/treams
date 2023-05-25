=========
Operators
=========

There are numerous operators implemented in *treams*. They replicate to some extend the
way active transformations with operators work on linear mapping :code:`T`

.. math::

    R(\varphi) T R^{-1}(\varphi)

where :code:`R` is the transformation we want to apply and and :code:`\varphi` is a
parameter of that transformation.

Rotation
========

As the notation already implies, for the sake of this section we could assume that this
is a rotation. At this point the notation is quite abstract. The representation of the
rotation operator depends on which basis we use. For example, for the T-matrix using the
spherical wave basis, such a rotation would be represented by the Wigner D-matrix
elements, but for plane waves this would look different. In *treams* we can create such
an abstract rotation operator by using :class:`~treams.Rotate`

.. doctest::

    >>> r = treams.Rotate(np.pi)

which can then be converted to a representation by calling it with the basis argument.

.. doctest::

    >>> r(basis=treams.SphericalWaveBasis.default(1))
    PhysicsArray(
        [[-1.+1.2246468e-16j,  0.+0.0000000e+00j,  0.+0.0000000e+00j,
        0.+0.0000000e+00j,  0.+0.0000000e+00j,  0.+0.0000000e+00j],
        [ 0.+0.0000000e+00j, -1.+1.2246468e-16j,  0.+0.0000000e+00j,
        0.+0.0000000e+00j,  0.+0.0000000e+00j,  0.+0.0000000e+00j],
        [ 0.+0.0000000e+00j,  0.+0.0000000e+00j,  1.+0.0000000e+00j,
        0.+0.0000000e+00j,  0.+0.0000000e+00j,  0.+0.0000000e+00j],
        [ 0.+0.0000000e+00j,  0.+0.0000000e+00j,  0.+0.0000000e+00j,
        1.+0.0000000e+00j,  0.+0.0000000e+00j,  0.+0.0000000e+00j],
        [ 0.-0.0000000e+00j,  0.+0.0000000e+00j,  0.-0.0000000e+00j,
        0.+0.0000000e+00j, -1.-1.2246468e-16j,  0.+0.0000000e+00j],
        [ 0.+0.0000000e+00j,  0.-0.0000000e+00j,  0.+0.0000000e+00j,
        0.-0.0000000e+00j,  0.+0.0000000e+00j, -1.-1.2246468e-16j]],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
    )

If it is multiplied with an array that defines the attribute `basis` it will
automatically take that attribute.

.. doctest::

    >>> t = treams.PhysicsArray(np.eye(6), basis=treams.SphericalWaveBasis.default(1))
    >>> r @ t @ r.inv
    PhysicsArray(
        [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
    )

Here, we also use the property `inv` to get the inverse rotation. Moreover, we for
instances of :class:`~treams.PhysicsArray` we can get the same result by using the
method apply

    >>> phi = 1
    >>> r = Rotate(phi)
    >>> r @ t @ r.inv == t.rotate.apply(phi)
    True

which also exists as `apply_left` and `apply_right` to only apply the operator from one
side. For some basis sets only rotations about the z-axis are possible, while other
basis sets allow rotations including all three Euler angles.

Translation
===========

The next transformation that is implemented are translations where the parameter is the
Cartesian translation vector.

.. doctest::

    >>> t = treams.PhysicsArray(np.eye(6), basis=treams.SphericalWaveBasis.default(1), k0=1)
    >>> t.translate.apply([1, 2, 3])
    PhysicsArray(
        [[ 0.13702594-1.38777878e-17j,  0.        +0.00000000e+00j,
        -0.02142403-4.28480668e-02j,  0.        +0.00000000e+00j,
        -0.01514908+2.01987724e-02j,  0.        +0.00000000e+00j],
        [ 0.        +0.00000000e+00j,  0.13702594+0.00000000e+00j,
        0.        +0.00000000e+00j, -0.02142403-4.28480668e-02j,
        0.        +0.00000000e+00j, -0.01514908+2.01987724e-02j],
        [-0.02142403+4.28480668e-02j,  0.        +0.00000000e+00j,
        0.07137993-6.84670061e-18j,  0.        +0.00000000e+00j,
        0.02142403+4.28480668e-02j,  0.        +0.00000000e+00j],
        [ 0.        +0.00000000e+00j, -0.02142403+4.28480668e-02j,
        0.        +0.00000000e+00j,  0.07137993-1.52119906e-17j,
        0.        +0.00000000e+00j,  0.02142403+4.28480668e-02j],
        [-0.01514908-2.01987724e-02j,  0.        +0.00000000e+00j,
        0.02142403-4.28480668e-02j,  0.        +0.00000000e+00j,
        0.13702594+6.93889390e-18j,  0.        +0.00000000e+00j],
        [ 0.        +0.00000000e+00j, -0.01514908-2.01987724e-02j,
        0.        +0.00000000e+00j,  0.02142403-4.28480668e-02j,
        0.        +0.00000000e+00j,  0.13702594-6.93889390e-18j]],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=1.0,
        material=Material(1, 1, 0),
        poltype='helicity',
    )

For the translation we have to specify the basis and the vacuum wave number. In the
result we can see that the default material of the embedding is vacuum and the default
polarization type is taken from :attr:`treams.config.POLTYPE`.

.. note::

    The rotation and translation operators applied to a spherical or cylindrical basis
    with multiple positions, will rotate or translate each position independently from
    the others. This results in block-diagonal matrices with respect to the different
    positions in such a case.

Expand in a different basis
===========================

The expansion in a different basis set is a little bit more complicated. Therefore, we
will treat each source basis set separately in the following.

Plane waves
-----------

Spherical waves
---------------

Cylindrical waves
-----------------

Expand in a different basis with periodic boundaries
====================================================

Change the polarization type
============================

Permute the axes
================

Evaluate the field
==================

