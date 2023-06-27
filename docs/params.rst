.. testsetup::

   import numpy as np
   import treams

====================================
Basis sets and other core parameters
====================================

Throughout the high-level functions and classes of *treams* a set of parameters appear
that define important underlying quantities for the calculation. First, these are the
different basis sets that are used to solve scattering processes: the spherical,
cylindrical, and plane wave solutions. Closely related to these basis sets are the
polarization types and the mode types. The other parameters are the vacuum wave numbers
and the materials as well as, in the case of calculations with periodicity involved,
the lattice definitions and the phase shift between lattice sites.

Basis sets
==========

As described in :doc:`maxwell` it is possible to solve Maxwells equations in different
coordinate systems. While being in principle equivalent, for different scenarios it is
beneficial to use suitable solution sets that represent the waves with sufficient
precision when truncated to a finite number of modes. The chosen finite number of
modes is given in the the the classes :class:`~treams.SphericalWaveBasis`,
:class:`~treams.CylindricalWaveBasis`, and :class:`~treams._core.PlaneWaveBasis`, which
are all children of the base call :class:`~treams._core.BasisSet`.

The modes of the spherical basis can are defined by their degree ``l``, the order ``m``,
and an index for the polarization ``pol``. The basis is then simply the collection of
multiple of these modes, each given in a tuple with exactly that order, for example

.. doctest::

    >>> treams.SphericalWaveBasis([(1, -1, 0), (1, 0, 0), (1, 1, 0)])
    SphericalWaveBasis(
        pidx=[0 0 0],
        l=[1 1 1],
        m=[-1  0  1],
        pol=[0 0 0],
        positions=[[0. 0. 0.]],
    )

results in a basis with three modes. All have the same degree and polarization, but the
order ``m`` goes from -1 to 1. We see that there are also the fields ``pidx`` and
``positions``. This is a special case for the spherical (and later also the
cyclindrical) wave basis. Sometimes, the fields are not expanded with respect to a
single point, but multiple positions. Then ``positions`` contains the their Cartesian
coordinates and ``pidx`` maps each mode to one of those coordinates. Here, the default
value of the expansion about a single origin is used. These basis sets behave mostly
like regular Python sets, we can check for example if a mode is in our basis set by

.. doctest::

    >>> (0, 1, 0, 0) in treams.SphericalWaveBasis([(1, -1, 0), (1, 0, 0), (1, 1, 0)])
    True

Equally, it is possible to use the regular comparisons and binary operators of Python
sets

.. doctest::

    >>> treams.SphericalWaveBasis([(1, -1, 0), (1, 0, 0), (1, 1, 0)]) > {(0, 1, 0, 0)}
    True
    >>> treams.SphericalWaveBasis([(1, -1, 0), (1, 0, 0), (1, 1, 0)]) & {(0, 1, 0, 0)}
    SphericalWaveBasis(
        pidx=[0],
        l=[1],
        m=[0],
        pol=[0],
        positions=[[0. 0. 0.]],
    )

However, because we want to use those basis sets later to index the rows and columns of
matrices, the order of the entries is fixed. Therefore, the equality operator is
stricter. Two basis sets are only considered equal when they have the same number modes
in the same order and the same positions.

.. doctest::

    >>> treams.SphericalWaveBasis([(1, 0, 0), (1, 1, 0)]) == treams.SphericalWaveBasis([(1, 1, 0), (1, 0, 0)])
    False

For convenience it is possible to create a default order up to a maximal multipolar
order

.. doctest::

    >>> treams.SphericalWaveBasis.default(2)
    SphericalWaveBasis(
        pidx=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
        l=[1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2],
        m=[-1 -1  0  0  1  1 -2 -2 -1 -1  0  0  1  1  2  2],
        pol=[1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    )

where we now have a spherical wave basis up do quadrupolar order.

The cyclindrical wave basis is mostly similar to the quadrupolar basis. Instead of a the
multipole ``l`` the z-component of the wave vector ``kz`` is used

.. doctest::

    >>> treams.CylindricalWaveBasis([(.1, -1, 0), (.1, 0, 0), (.1, 1, 0)])
    CylindricalWaveBasis(
        pidx=[0 0 0],
        kz=[0.1 0.1 0.1],
        m=[-1  0  1],
        pol=[0 0 0],
        positions=[[0. 0. 0.]],
    )

which is a real number. The default function takes a list of ``kz`` values a maximal
absolute value for ``m``.

.. doctest::

    >>> treams.CylindricalWaveBasis.default([-.5, .5], 1)
    CylindricalWaveBasis(
        pidx=[0 0 0 0 0 0 0 0 0 0 0 0],
        kz=[-0.5 -0.5 -0.5 -0.5 -0.5 -0.5  0.5  0.5  0.5  0.5  0.5  0.5],
        m=[-1 -1  0  0  1  1 -1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0 1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    )

The cylindrical wave basis is particularly useful for systems with periodicity in the
z-direction. Then, a basis with the diffraction orders up to a threshold can be obtained
by running

.. doctest::

    >>> treams.CylindricalWaveBasis.diffr_orders(kz=.1, mmax=1, lattice=2 * np.pi, bmax=1.05)
    CylindricalWaveBasis(
        pidx=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],
        kz=[-0.9 -0.9 -0.9 -0.9 -0.9 -0.9  0.1  0.1  0.1  0.1  0.1  0.1  1.1  1.1
      1.1  1.1  1.1  1.1],
        m=[-1 -1  0  0  1  1 -1 -1  0  0  1  1 -1 -1  0  0  1  1],
        pol=[1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0],
        positions=[[0. 0. 0.]],
    )

where ``bmax`` defines a distance in reciprocal space.

The plane wave basis behaves a little bit different. First, it is currently only defined
with respect to a single origin so the ``pidx`` and ``positions`` is not defined. Also,
the basis can be defined in two ways: :class:`PlaneWaveBasisByUnitVector` and
:class:`PlaneWaveBasisByComp`. In the first case, the definition is given by the unit
vector which, multiplied by the wave number in the medium, gives the full wave vector.
In the second case, two components of the wave vector are given and the remaining third
Cartesian component is defined such that it fulfils the dispersion relation.

.. doctest::

    >>> treams.PlaneWaveBasisByUnitVector([(4, 0, 3, 0), (4, 0, 3, 1)])
    PlaneWaveBasisByUnitVector(
        qx=[0.8 0.8],
        qy=[0. 0.],
        qz=[0.6 0.6],
        pol=[0 1],
    )
    >>> treams.PlaneWaveBasisByComp([(1, 0, 0), (1, 0, 1)])
    PlaneWaveBasisByComp(
        kx=[1 1],
        ky=[0 0],
        pol=[0 1],
    )

By default, it is assumed, that the x- and y- components are given for the latter class,
but other components can also be chosen.

It is possible to convert between those basis sets by using the corresponding
functions

.. doctest::

    >>> pwbc = treams.PlaneWaveBasisByComp([(3, 0, 0), (3, 0, 1)])
    >>> pwbc.byunitvector(5)
    PlaneWaveBasisByUnitVector(
        qx=[0.6+0.j 0.6+0.j],
        qy=[0.+0.j 0.+0.j],
        qz=[0.8+0.j 0.8+0.j],
        pol=[0 1],
    )
    >>> pwbuv = treams.PlaneWaveBasisByUnitVector([(0, 0, 1, 0), (0, 0, 1, 1)])    
    >>> pwbuv.bycomp(1)
    PlaneWaveBasisByComp(
        kx=[0. 0.],
        ky=[0. 0.],
        pol=[0 1],
    )

Additionally, similar to the case of cylindrical waves, the basis by components can be
used for a range of diffraction orders

    >>> treams.PlaneWaveBasisByComp.diffr_orders([0, 0], np.eye(2), 7)
    PlaneWaveBasisByComp(
        kx=[ 0.          0.          0.          0.          0.          0.
      6.28318531  6.28318531 -6.28318531 -6.28318531],
        ky=[ 0.          0.          6.28318531  6.28318531 -6.28318531 -6.28318531
      0.          0.          0.          0.        ],
        pol=[1 0 1 0 1 0 1 0 1 0],
    )


Polarizations
=============

The definitions of the basis sets above are not complete without specifying the
polarization types. In *treams* two polarization types are supported: `helicity` and
`parity`. The first allows the use of chiral material parameters. Each polarization type
contains two polarizations that are indicated by the integers `0` and `1` throughout the
code. For helicity polarizations `0` stands for negative helicity and `1` for positive
helicity. In the case of parity polarizations `0` stands for `TE` or `magnetic`
polarization and `1` for `TM` or `electric` polarizations. The magnetic parity waves are
defined in :func:`treams.special.vsw_M`, :func:`treams.special.vsw_rM`,
:func:`treams.special.vcw_M`, :func:`treams.special.vcw_rM`, and
:func:`treams.special.vpw_M`. For spherical waves they are transverse with respect to
the radial direction, for cylindrical and plane waves they are transverse to the z-axis.
The corresponding electric parity waves are :func:`treams.special.vsw_N`,
:func:`treams.special.vsw_rN`, :func:`treams.special.vcw_N`,
:func:`treams.special.vcw_rN`, and :func:`treams.special.vpw_N`.

The helicity waves are defined in :func:`treams.special.vsw_A`,
:func:`treams.special.vsw_rA`, :func:`treams.special.vcw_A`,
:func:`treams.special.vcw_rA`, and :func:`treams.special.vpw_A`.

The default polarization type to be used can by setting ``treams.config.POLTYPE`` to the
corresponding string.

Mode types
==========

For some basis sets there exist two different types of modes, that distinguish
propagation features. For the spherical and cylindrical basis theses are `regular`
and `singular` modes. The former come through the use of (spherical) Bessel Functions
and the latter through the use of (spherical) Hankel functions of the first kind. The
regular modes are finite in the whole space. Thus, they are suitable for describing
incident modes or to expand a plane wave. The singular modes fulfil the radiation
condition and as such are used for the scattered fields.

For the plane wave basis of type (:class:`~treams.PlaneWaveBasisByComp`) only two
components of the wave vector are given and the third component is only implicitly
defined by the wave number and the material parameters. The application for this basis
is mostly within stratified media that are uniform or periodic in the two other
dimensions. Thus, the two given components of the wave vectors are conserved up to
reciprocal lattice vectors. To lift the ambiguity of the definition of the third
component, the mode types `up` and `down` are possible. They define, if the modes
propagate -- or decay for evanescent modes -- along the positive or negative direction
with respect to the third axis.

Vacuum wave number
==================

All calculations are executed in frequency domain. Instead of defining the frequency
:math:`\nu` or the angular frequency :math:`\omega` itself, *treams* works by using the
vacuum wave number

.. math::

    k_0 = \frac{2 \pi \nu}{c} = \frac{\omega}{c}

where :math:`c` is the speed of light in vacuum. In the code this real-valued number is
usually referred to by ``k0``. Implicitly, it is assumed throughout that all quantities,
like wave numbers, wave vectors, distances, or lattice vectors are given in the same
unit of (inverse) length.

Materials
=========

For materials there exists the class :class:`~treams.Material`, which holds the values
of the relative permittivity, relative permeability, and the chirality parameter. The
default material is air and can be initialized without any parameters. For other cases,
the parameters can be given in the order above.

.. doctest::

    >>> treams.Material()
    Material(1, 1, 0)
    >>> treams.Material(3, 2, 1)
    Material(3, 2, 1)

It's also possible to get the parameters from the refractive index (or the refractive
indices for negative and positive helicity) and the impedance

.. doctest::

    >>> treams.Material.from_n(3)
    Material(9.0, 1.0, 0)
    >>> treams.Material.from_nmp((3, 5))
    Material(16.0, 1.0, 1.0)

Lattices
========

The periodicity of arrangements is given by defining an instance of the class
:class:`~treams.Lattice`. A lattice can be one-, two-, or three-dimensional.

.. doctest::

    >>> treams.Lattice(1)
    Lattice(1.0, alignment='z')
    >>> treams.Lattice([[1, .5], [-.5, 1]])
    Lattice([[ 1.   0.5]
             [-0.5  1. ]], alignment='xy')
    >>> treams.Lattice([1, 2, 3])
    Lattice([[1. 0. 0.]
             [0. 2. 0.]
             [0. 0. 3.]], alignment='xyz')

The one- and two-dimensional lattices have to be aligned with one and two, respectively,
Cartesian axes. The default alignments are along the z-axis for one-dimensional and in
the x-y-plane for the two-dimensional lattices. In the last example we see that it is
sufficient to just specify the diagonal entries. It's also possible to automatically
create special lattice shapes, for example

.. doctest::

    >>> treams.Lattice.hexagonal(2)
    Lattice([[2.         0.        ]
             [1.         1.73205081]], alignment='xy')

creates a hexagonal lattice with sidelength 2. It's also possible to extract a
lower-dimensional sublattice

.. doctest::

    >>> lat_3d = treams.Lattice([1, 2, 3])
    >>> treams.Lattice(lat_3d, "zx")
    Lattice([[0. 1.]
             [3. 0.]], alignment='zx')

or to combine and compare lattices

.. doctest::

    >>> treams.Lattice(1, "x") | treams.Lattice(2, "y")
    Lattice([[1. 0.]
             [0. 2.]], alignment='xy')
    >>> treams.Lattice([1, 2], "xy") & treams.Lattice([2, 3], "yz")
    Lattice(2.0, alignment='y')
    >>> treams.Lattice(1, "x") <= treams.Lattice([1, 2], "xy")
    True

The volume of the lattice can also be obtained

.. doctest::

    >>> treams.Lattice([[1, 0], [0, 1]]).volume
    1.0
    >>> treams.Lattice([[0, 1], [1, 0]]).volume
    -1.0

as we see the volume is "signed", i.e. it shows if the lattice vectors are in a
right-handed order, and the reciprocal lattice vectors can be computed

.. doctest::

    >>> treams.Lattice([1, 1]).reciprocal
    array([[ 6.28318531, -0.        ],
           [-0.        ,  6.28318531]])

Phase vector
============

The wave vector, often referred to as ``kpar``, specifies the phase relationship of
different lattice sites :math:`\exp(\mathrm i \boldsymbol k_\parallel \boldsymbol R)`.

.. doctest::

    >>> treams.WaveVector()
    WaveVector(nan, nan, nan)
    >>> treams.WaveVector(1)
    WaveVector(nan, nan, 1)
    >>> treams.WaveVector(1, "x")
    WaveVector(1, nan, nan)
    >>> treams.WaveVector((1, 2))
    WaveVector(1, 2, nan)
    >>> treams.WaveVector((1, 2, 3))
    WaveVector(1, 2, 3)

where unspecified directions are represented as ``nan``. The wave vectors can be
combined and compared.

.. doctest::

    >>> treams.WaveVector((1, 2)) | treams.WaveVector((2, 3), "yz")
    WaveVector(nan, 2, nan)
    >>> treams.WaveVector(1, "x") & treams.WaveVector(2, "y")
    WaveVector(1, 2, nan)
    >>> treams.WaveVector(1, "x") >= treams.WaveVector((1, 2))
    True

Note that the ordering is from less strict wave vector to the stricter one.
