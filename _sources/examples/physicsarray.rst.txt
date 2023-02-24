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
        [2, 6]
    )
    >>> np.array([1, 2]) @ treams.PhysicsArray([2, 3])
    8

but they have mainly two features add. First, they derive from
:class:`treams.util.AnnotatedArray` so they can carry annotations with them.

.. doctest::

    >>> treams.PhysicsArray([[0, 1], [2, 3]], k0=(1, 2))
    PhysicsArray(
        [[0, 1],
         [2, 3]],
        k0=(1.0, 2.0)
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
        material=Material(4, 1, 0)
    )

where we now have a material with the relative permittivity 4. As with its parent class
these properties are also compared and merged when using operations on these objects

.. doctest::

    >>> treams.PhysicsArray([0, 1], k0=1) + treams.PhysicsArray([2, 3], material=2)
    PhysicsArray(
        [2, 4],
        k0=1.0,
        material=Material(2, 1, 0)
    )

and using conflicting values will raise a warning, for example

.. doctest::

    >>> treams.PhysicsArray([0, 1], k0=1) + treams.PhysicsArray([2, 3], k0=2)
    PhysicsArray(
        [2, 4],
        k0=1.0
    )

emits :code:`treams/util.py:249: AnnotationWarning: at index 0: overwriting key 'k0'`.
