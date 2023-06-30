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
:code:`arr.op()` is also defined. For arrays with ``ndim`` equal to 1 or without an
inverse, it is equivalent to :code:`arr.op.apply_left()`, otherwise it corresponds to
:code:`op @ arr @ op.inv`.

Rotation
========

As already mentioned a common transformation are rotations. The representation of the
rotation operator depends on which basis we use. For example, for the T-matrix using the
spherical wave basis, such a rotation is represented by the Wigner D-matrix elements,
but for plane waves this would look different. In *treams* we can create such an
abstract rotation operator by using :class:`~treams.Rotate`

.. doctest::

    >>> r = Rotate(np.pi)

which can then be converted to a representation by calling it with the basis argument.

.. doctest::

    >>> r(basis=SphericalWaveBasis.default(1))
    PhysicsArray(
        [[-1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
         [ 0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
         [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
         [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
         [ 0.-0.j,  0.+0.j,  0.-0.j,  0.+0.j, -1.-0.j,  0.+0.j],
         [ 0.+0.j,  0.-0.j,  0.+0.j,  0.-0.j,  0.+0.j, -1.-0.j]],
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

    >>> t = PhysicsArray(np.eye(6), basis=SphericalWaveBasis.default(1))
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
instances of :class:`~treams.PhysicsArray` we can get the same result by calling the
correspondingly named attribute

    >>> phi = 1
    >>> r = Rotate(phi)
    >>> (r @ t @ r.inv == t.rotate(phi)).all()
    True

which also has the methods `apply_left` and `apply_right` to only apply the operator
from one side. For some basis sets only rotations about the z-axis are possible, while
other basis sets allow rotations including all three Euler angles.

Translation
===========

The next transformation that is implemented are translations where the parameter is the
Cartesian translation vector.

.. doctest::

    >>> t = PhysicsArray(np.eye(6), basis=SphericalWaveBasis.default(1), k0=1)
    >>> t.translate([1, 2, 3])
    PhysicsArray(
        [[ 0.137-0.j   ,  0.   +0.j   , -0.021-0.043j,  0.   +0.j   ,
          -0.015+0.02j ,  0.   +0.j   ],
         [ 0.   +0.j   ,  0.137+0.j   ,  0.   +0.j   , -0.021-0.043j,
           0.   +0.j   , -0.015+0.02j ],
         [-0.021+0.043j,  0.   +0.j   ,  0.071-0.j   ,  0.   +0.j   ,
           0.021+0.043j,  0.   +0.j   ],
         [ 0.   +0.j   , -0.021+0.043j,  0.   +0.j   ,  0.071-0.j   ,
           0.   +0.j   ,  0.021+0.043j],
         [-0.015-0.02j ,  0.   +0.j   ,  0.021-0.043j,  0.   +0.j   ,
           0.137+0.j   ,  0.   +0.j   ],
         [ 0.   +0.j   , -0.015-0.02j ,  0.   +0.j   ,  0.021-0.043j,
           0.   +0.j   ,  0.137-0.j   ]],
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

The expansion in a different basis set is a little bit more complicated due to the
number of possible combinations of which basis set can be expanded in which other basis
sets. Therefore, we will treat each source basis set separately in the following.

Also, here the notion of abstract operator and concrete representation breaks down to
some extent because it makes little sense to first define an abstract expansion in,
e.g., spherical waves without specifying the relevant multipoles. Thus, the concrete
representation of the target basis is the argument of the operator.

Plane waves
-----------

Plane waves can be expanded into a different set of plane waves and into regular
spherical and cylindrical waves. The expansion into a different set of plane waves
is basically just a matching of the wave vectors and polarizations.

.. doctest::

    >>> plw = plane_wave([0, 3, 4], [.5, .5], k0=5, material=1)
    >>> Expand(PlaneWaveBasisByComp.default([[0, 3]])) @ plw
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

    >>> Expand(CylindricalWaveBasis.default([4], 1)) @ plw
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
    >>> Expand(SphericalWaveBasis.default(1)) @ plw
    PhysicsArray(
        [ 0.307-0.j   , -2.763+0.j   , -0.   -1.302j, -0.   -1.302j,
         -2.763+0.j   ,  0.307+0.j   ],
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

    >>> off_centered_swb = SphericalWaveBasis.default(1, positions=[[1, 0, 0]])
    >>> spw = spherical_wave(1, 0, 0, basis=off_centered_swb, k0=1, material=1, modetype="singular")
    >>> ex = Expand(SphericalWaveBasis.default(1))
    >>> ex @ spw
    PhysicsArray(
        [ 0.  +0.j   ,  0.  +0.319j,  0.  +0.j   ,  0.81+0.j   ,
          0.  +0.j   , -0.  +0.319j],
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

    >>> ex = Expand(SphericalWaveBasis.default(1), "regular")
    >>> ex @ spw
    PhysicsArray(
        [0.   +0.j   , 1.466+0.319j, 0.   +0.j   , 0.81 +1.262j,
         0.   +0.j   , 1.466+0.319j],
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

    >>> sw_global = spherical_wave(1, 0, 0, k0=1, material=1, modetype="regular")
    >>> local_swb = SphericalWaveBasis.default(1, 2, positions=[[0, 0, 1], [0, 0, -1]])
    >>> sw_global.expand.apply_left(local_swb)
    PhysicsArray(
        [0.   +0.j, 0.   +0.j, 0.   +0.j, 0.904-0.j, 0.   +0.j, 0.   +0.j,
         0.   +0.j, 0.   +0.j, 0.   +0.j, 0.904-0.j, 0.   +0.j, 0.   +0.j],
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
expand back into the same basis set in this case corresponds to the multiplication by a
unit matrix.

.. doctest::

    >>> sw_global.expand(SphericalWaveBasis.default(1))
    PhysicsArray(
        [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
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

For translations from singular to regular waves, the same basis set means that a
zero matrix is returned.

.. doctest::

    >>> sw_global.expand.apply_right(SphericalWaveBasis.default(1), "singular")
    PhysicsArray(
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
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

Besides that the expansion of spherical waves in different basis sets results in dense
matrices.

The expansion of spherical waves into cylindrical or plane waves is a continuous
spectrum and is currently not implemented.

Cylindrical waves
-----------------

Cylindrical waves are similar to spherical waves, in the sense, that they can be
separated into regular and singular modes and that they can be defined with multiple
origins within treams. Therefore, the expansion within cylindrical waves follows the
same properties than spherical waves.

.. doctest::

    >>> off_centered_cwb = CylindricalWaveBasis.default([0], 1, positions=[[1, 0, 0]])
    >>> cyw = cylindrical_wave(0, 1, 0, basis=off_centered_cwb, k0=1, material=1, modetype="singular")
    >>> ex = Expand(CylindricalWaveBasis.default([0], 1))
    >>> ex @ cyw
    PhysicsArray(
        [ 0.   +0.j,  0.115-0.j,  0.   +0.j, -0.44 +0.j,  0.   +0.j,
          0.765+0.j],
        basis=CylindricalWaveBasis(
        pidx=[0 0 0 0 0 0],
        kz=[0. 0. 0. 0. 0. 0.],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=1.0,
        material=Material(1, 1, 0),
        modetype='singular',
    )

Additionally, it is possible to expand a cylindrical wave into spherical waves. Note,
that waves defined with multiple origins get each expanded separately. The positions
of the spherical and cylindrical waves must be equal.

.. doctest::

    >>> cyw = cylindrical_wave(0, 1, 0, k0=1, material=1, modetype="regular")
    >>> cyw.expand(SphericalWaveBasis.default(1))
    PhysicsArray(
        [0.  +0.j, 0.  +0.j, 0.  +0.j, 0.  +0.j, 0.  +0.j, 3.07+0.j],
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

The inverse of this expansion is not implemented.

The expansion of cylindrical waves into plane waves is a continuous spectrum and is not
implemented.

Expand in a different basis with periodic boundaries
====================================================

There is a special case of expansion implemented for the case of periodic boundaries
when using spherical or cylindrical waves. These expansions are needed to compute the
electromagnetic interaction between particles within a lattice. It is assumed that the
given basis with singular modes are repeated periodically in the given lattice
structure. Then, these fields are expanded as regular fields in a single unit cell.

.. doctest::

    >>> cyw = cylindrical_wave(0, 1, 0, k0=1, material=1, modetype="singular")
    >>> cyw.expandlattice(1, 0)
    PhysicsArray(
        [0.+0.j   , 2.-3.866j, 0.+0.j   , 0.+0.j   , 0.+0.j   , 1.+1.234j],
        basis=CylindricalWaveBasis(
        pidx=[0 0 0 0 0 0],
        kz=[0. 0. 0. 0. 0. 0.],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=1.0,
        kpar=WaveVector(0, nan, 0.0),
        lattice=Lattice(1.0, alignment='x'),
        material=Material(1, 1, 0),
        modetype='regular',
    )
    >>> spw = spherical_wave(1, 0, 0, k0=1, material=1)
    >>> spw.expandlattice([1, 2], [0, 0])
    PhysicsArray(
        [ 0.+0.j   ,  0.+0.j   ,  0.+0.j   , -1.+7.722j,  0.+0.j   ,
          0.+0.j   ],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=1.0,
        kpar=WaveVector(0, 0, nan),
        lattice=Lattice([[1. 0.]
             [0. 2.]], alignment='xy'),
        material=Material(1, 1, 0),
        modetype='regular',
        poltype='helicity',
    )

The inverse of this operator is not implemented. Additionally, it's possible to expand
the periodic field into a different basis set. Spherical waves in a one-dimensional
lattice along the z-axis can be expanded in cylindrical waves

.. doctest::

    >>> spw = spherical_wave(1, 0, 0, k0=1, material=1)
    >>> ex = ExpandLattice(basis=CylindricalWaveBasis.diffr_orders([.1], 0, 7, 1))
    >>> ex @ spw
    PhysicsArray(
        [ 0.+0.j   , -0.+0.094j,  0.+0.j   , -0.+0.154j,  0.+0.j   ,
         -0.+0.011j],
        basis=CylindricalWaveBasis(
        pidx=[0 0 0 0 0 0],
        kz=[-0.798 -0.798  0.1    0.1    0.998  0.998],
        m=[0 0 0 0 0 0],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        k0=1.0,
        kpar=WaveVector(nan, nan, 0.1),
        lattice=Lattice(7.0, alignment='z'),
        material=Material(1, 1, 0),
        modetype='singular',
        poltype='helicity',
    )

where the lattice and the wave vector are implicitly defined by the use of the
class method :func:`treams.CylindricalWaveBasis.diffr_orders`. Similarly, spherical
waves in a two-dimensional lattice in the x-y-plane can be expanded in plane waves.

.. doctest::

    >>> ex = ExpandLattice(basis=PlaneWaveBasisByComp.diffr_orders([.1, 0], [7, 7], 1))
    >>> ex @ spw
    PhysicsArray(
        [ 0.+0.j   , -0.+0.004j,  0.+0.j   , -0.+0.093j,  0.+0.j   ,
         -0.+0.093j,  0.+0.j   , -0.+0.638j,  0.+0.j   , -0.+0.059j],
        basis=PlaneWaveBasisByComp(
        kx=[ 0.1    0.1    0.1    0.1    0.1    0.1    0.998  0.998 -0.798 -0.798],
        ky=[ 0.     0.     0.898  0.898 -0.898 -0.898  0.     0.     0.     0.   ],
        pol=[1 0 1 0 1 0 1 0 1 0],
    ),
        k0=1.0,
        kpar=WaveVector(0.1, 0, nan),
        lattice=Lattice([[7. 0.]
             [0. 7.]], alignment='xy'),
        material=Material(1, 1, 0),
        modetype='up',
        poltype='helicity',
    )

Cylindrical waves, that themselves are periodic in the z-direction, in a one-dimensional
lattice along the x-axis can also be expanded in plane waves.

.. doctest::

    >>> cyw = cylindrical_wave(0, 1, 0, k0=1, material=1)
    >>> ex = ExpandLattice(basis=PlaneWaveBasisByComp.diffr_orders([0, .1], Lattice([7, 7], "zx"), 1))
    >>> ex @ cyw
    PhysicsArray(
        [0.   +0.j   , 0.286-0.029j, 0.   +0.j   , 0.286-4.115j,
         0.   +0.j   , 0.286+0.378j, 0.   +0.j   , 0.   +0.j   ,
         0.   +0.j   , 0.   +0.j   ],
        basis=PlaneWaveBasisByComp(
        kz=[ 0.     0.     0.     0.     0.     0.     0.898  0.898 -0.898 -0.898],
        kx=[ 0.1    0.1    0.998  0.998 -0.798 -0.798  0.1    0.1    0.1    0.1  ],
        pol=[1 0 1 0 1 0 1 0 1 0],
    ),
        k0=1.0,
        kpar=WaveVector(nan, nan, 0.0),
        lattice=Lattice(7.0, alignment='x'),
        material=Material(1, 1, 0),
        modetype='up',
    )

Change the polarization type
============================

Changing the polarization type is a simple operation. All waves can be expanded in
modes of well-defined helicity. For an achiral material these waves can equally be
expressed in modes of well-defined parity. The change between those polarization types
can be expressed as an operator.

.. doctest::

    >>> spw = spherical_wave(1, 0, 0, poltype="helicity")
    >>> spw.changepoltype("parity")
    PhysicsArray(
        [ 0.   ,  0.   ,  0.707, -0.707,  0.   ,  0.   ],
        basis=SphericalWaveBasis(
        pidx=[0 0 0 0 0 0],
        l=[1 1 1 1 1 1],
        m=[-1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    ),
        poltype='parity',
    )

Permute the axes
================

The permute operator is only implemented for plane waves, in particular for plane
waves that are defined by two of their components (and a direction of the modes).
For this type of waves, the rotation is only implemented about the z-axis. These
rotations then don't include a relabeling of the Cartesian axes, for example
:math:`(x', y', z') = (z, x, y)`. This operation is implemented separately as
permutation, meaning the axes labels get permuted.

.. doctest::

    >>> plw = plane_wave([2, 3, 6], 0)
    >>> plw
    PhysicsArray(
        [0, 1],
        basis=PlaneWaveBasisByUnitVector(
        qx=[0.286 0.286],
        qy=[0.429 0.429],
        qz=[0.857 0.857],
        pol=[1 0],
    ),
    )
    >>> plw.permute()
    PhysicsArray(
        [ 0.   +0.j   , -0.789+0.614j],
        basis=PlaneWaveBasisByUnitVector(
        qx=[0.857 0.857],
        qy=[0.286 0.286],
        qz=[0.429 0.429],
        pol=[1 0],
    ),
        poltype='helicity',
    )

Evaluate the field
==================

From a programming perspective, the evaluation of the field values at specified points
is also implemented by a couple of operators. The electric field :math:`\boldsymbol E`,
the magnetic field :math:`\boldsymbol H`, the displacement field :math:`\boldsymbol D`,
and the magnetic flux density :math:`\boldsymbol B` can be computed as well as two
different definitions of the Riemann-Silberstein vectors
:math:`\sqrt{2} \boldsymbol G_\pm = \boldsymbol E \pm \mathrm i Z_0 Z \boldsymbol H` and
:math:`\sqrt{2} \boldsymbol F_\pm = \frac{1}{\epsilon_0 \epsilon} \boldsymbol D \pm
\mathrm i \frac{c}{n} \boldsymbol B = \frac{n \pm \kappa}{n} G_\pm` (see also 
:doc:`maxwell`).

.. doctest::

    >>> spw = spherical_wave(1, 0, 0, k0=1, material=1, poltype="helicity", modetype="regular")
    >>> spw.efield([[0, 0, 0], [1, 0, 0]])
    PhysicsArray(
        [[0.+0.j   , 0.+0.j   , 0.+0.163j],
         [0.-0.j   , 0.-0.074j, 0.+0.132j]],
    )
