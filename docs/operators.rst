=========
Operators
=========

There are numerous operators implemented in *treams*. They replicate to some extend the
way active transformations with operators work on linear mapping :math:`M`

.. math::

    O(x) M O^{-1}(x)

where :math:`O` is the transformation we want to apply and and :math:`x` is a
parameter of that transformation. Similarly,

.. math::

    O(x) \psi

transforms state :math:`\psi` that can be represented by a vector. We attempt to
replicate this transformation notation in code and to extend it by other useful
functions. Later, the state :math:`\psi` can often be the expansion coefficients of
a wave, the linear mapping could be the T-matrix and the transformation operator could
be a rotation.

Each operator is implemented as a class. The class is instantiated with the parameter
:code:`op = Operator(x)`. At this stage it is only an abstract operator. A concrete
representation can be obtained by calling the operator with the necessary keyword
arguments, e.g. :code:`op(basis=concrete_basis)`, which will return a array-like
structure.

However, to be able to replicate the above mathematical notation it is also possible to
use the matrix multiplication operator between an array and the operator. The array
needs to have the necessary keywords as attributes, this is for what :doc:`physicsarray`
come in handy. For such an array :code:`arr` it is possible to type :code:`op @ arr` or
:code:`op @ arr @ op.inv`. The inverse of operators is implemented for many operators
but some have no inverse defined.

Sometimes, it can come in handy to directly apply an operator to an array without
defining the abstract operator before. This can be achieved by
:code:`arr.operator.apply_left(x)` and :code:`arr.operator.apply_right(x)`, which are
equivalent to :code:`op @ arr` and :code:`arr @ op.inv`, respectively. The function
:code:`arr.op.apply()` is also defined. For arrays with ``ndim`` equal to 1 or without
an inverse, it is equivalent to :code:`arr.op.apply_left()`, otherwise it corresponds to
:code:`op @ arr @ op.inv`.

Finally, it is possible to just get the transformation matrix for an array without
performing the matrix multiplication right away. For this the operator attribute can be
simply called :code:`arr.operator(x)` which gives an equal result as
:code:`op(basis=concrete_basis)`

Rotation
========

As already mentioned a common transformation are rotations. The representation of the
rotation operator depends on which basis we use. For example, for the T-matrix using the
spherical wave basis, such a rotation is represented by the Wigner D-matrix elements,
but for plane waves this would look different. In *treams* we can create such an
abstract rotation operator by using :class:`~treams.Rotate`

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

Plane waves can be expanded into a different set of plane waves and into regular
spherical and cylindrical waves. The expansion into a different set of plane waves
is basically just a matching of the wave vectors and polarizations.

.. doctest::

    >>> pw = treams.plane_wave([0, 3, 4], [.5, .5], k0=5, material=1)
    >>> treams.Expand(treams.PlaneWaveBasisByComp.default([[0, 3]])) @ pw
    PhysicsArray(
        [1., 1.],
        basis=PlaneWaveBasisByComp(
        kx=[0. 0.],
        ky=[3. 3.],
        pol=[1 0],
    ),
        k0=5.0,
        material=Material(1, 1, 0),
        modetype='up',
    )

For example, here we change from the expansion in
:class:`~treams.PlaneWaveBasisByUnitVector` to the expansion by x- and y- components.
For such a basis change, it is necessary that the material and the wave number is
specified.

Next, we can expand this plane wave also in cylindrical and in spherical waves.

.. doctest::

    >>> treams.Expand(treams.CylindricalWaveBasis.default([4], 1)) @ pw
    PhysicsArray(
        [0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j],
        basis=CylindricalWaveBasis(
        pidx=[0 0 0 0 0 0],
        kz=[4. 4. 4. 4. 4. 4.],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=5.0,
        material=Material(1, 1, 0),
        modetype='regular',
    )
    >>> treams.Expand(treams.SphericalWaveBasis.default(1)) @ pw
    PhysicsArray(
        [ 3.06998012e-01-3.75964133e-17j, -2.76298211e+00+3.38367720e-16j,
        -7.97540364e-17-1.30248226e+00j, -7.97540364e-17-1.30248226e+00j,
        -2.76298211e+00+0.00000000e+00j,  3.06998012e-01+0.00000000e+00j],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=5.0,
        material=Material(1, 1, 0),
        modetype='regular',
        poltype='helicity',
    )

Spherical waves
---------------

Next, we have spherical waves. In comparison to the plane waves, spherical waves have
the added difficulty of the categorization of "regular" and "singular" functions and the
distinction of global and local basis sets.

In a simple case we want to expand a spherical wave that is centered not at the origin
and expand it around the origin

.. doctest::

    >>> off_centered_swb = treams.SphericalWaveBasis.default(1, positions=[[1, 0, 0]])
    >>> sw = treams.spherical_wave(1, 0, 0, basis=off_centered_swb, k0=1, material=1, modetype="singular")
    >>> ex = treams.Expand(treams.SphericalWaveBasis.default(1))
    >>> ex @ sw
    PhysicsArray(
        [ 0.00000000e+00+0.00000000e+00j,  5.86797393e-17+3.19437623e-01j,
        0.00000000e+00+0.00000000e+00j,  8.10453459e-01+3.79855139e-18j,
        0.00000000e+00+0.00000000e+00j, -1.95599131e-17+3.19437623e-01j],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=1.0,
        material=Material(1, 1, 0),
        modetype='singular',
        poltype='helicity',
    )

We defined the wave as a singular wave and, if nothing is explicitly specified, the
expansion into other spherical waves is taken as the same type of field. So, a singular
field will be expanded again in singular modes and a regular field is expanded in
regular modes. However, we can also change the type of mode, when the field is expanded
around a different origin

.. doctest::

    >>> ex = treams.Expand(treams.SphericalWaveBasis.default(1), "regular")
    >>> ex @ sw
    PhysicsArray(
        [0.        +0.j        , 1.4655919 +0.31943762j,
        0.        +0.j        , 0.81045346+1.26220648j,
        0.        +0.j        , 1.4655919 +0.31943762j],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=1.0,
        material=Material(1, 1, 0),
        modetype='regular',
        poltype='helicity',
    )

for this we had to define the ``modetype`` for the expand operator.

Next, we want to look at the expansion of a global field into a local field at multiple
origins, which works quite similarly

.. doctest::

    >>> sw_global = treams.spherical_wave(1, 0, 0, k0=1, material=1, modetype="regular")
    >>> local_swb = treams.SphericalWaveBasis.default(1, 2, positions=[[0, 0, 1], [0, 0, -1]])
    >>> sw_global.expand.apply_left(local_swb)
    PhysicsArray(
        [0.        +0.00000000e+00j, 0.        +0.00000000e+00j,
        0.        +0.00000000e+00j, 0.90350604-7.59710279e-18j,
        0.        +0.00000000e+00j, 0.        +0.00000000e+00j,
        0.        +0.00000000e+00j, 0.        +0.00000000e+00j,
        0.        +0.00000000e+00j, 0.90350604-7.59710279e-18j,
        0.        +0.00000000e+00j, 0.        +0.00000000e+00j],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0 1 1 1 1 1 1],
        l=[1 1 1 1 1 1 1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1 -1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0 1 0 1 0 1 0],
        positions=[[ 0.  0.  1.], [ 0.  0. -1.]],
    ),
        k0=1.0,
        material=Material(1, 1, 0),
        modetype='regular',
        poltype='helicity',
    )

For the translations within only regular or only singular waves it is possible to
expand back into the same basis set which returns a unit matrix.

.. doctest::

    >>> sw_global.expand(treams.SphericalWaveBasis.default(1))
    PhysicsArray(
        [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 1.-0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.-0.j]],
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

For translations from singular to regular waves, the same basis set means that a
zero matrix is returned.

.. doctest::

    >>> sw_global.expand.inv(treams.SphericalWaveBasis.default(1), "singular")
    PhysicsArray(
        [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=1.0,
        material=Material(1, 1, 0),
        modetype=('regular', 'singular'),
        poltype='helicity',
    )

Besides that the expansion of spherical waves in different basis sets results in dense
matrices.

The expansion of spherical waves into cylindrical or plane waves is a continuous
spectrum and is currently not implemented.

Cylindrical waves
-----------------

Cylindrical waves are similar to spherical waves, in the sense, that they can be
separated into regular and singular modes and that they can be defined with multiple
origins within treams. The expansion within cylindrical waves follows the same
properties than spherical waves.

Expand in a different basis with periodic boundaries
====================================================

Change the polarization type
============================

Permute the axes
================

Evaluate the field
==================

