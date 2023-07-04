r"""Lattice sums.

.. currentmodule:: treams.lattice

Calculates the lattice sums of the forms

.. math::
   D_{lm}(k, \boldsymbol k_\parallel, \boldsymbol r, \Lambda_d)
   = \sideset{}{'}{\sum_{\boldsymbol R \in \Lambda_d}}
   h_l^{(1)}(k|\boldsymbol r + \boldsymbol R|)
   Y_{lm}
   (\theta_{-\boldsymbol r - \boldsymbol R}, \varphi_{-\boldsymbol r - \boldsymbol R})
   \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

and

.. math::
   D_{m}(k, \boldsymbol k_\parallel, \boldsymbol r, \Lambda_d)
   = \sideset{}{'}{\sum_{\boldsymbol R \in \Lambda_d}}
   H_m^{(1)}(k|\boldsymbol r + \boldsymbol R|)
   \mathrm e^{\mathrm i m \varphi_{-\boldsymbol r - \boldsymbol R}}
   \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

that arise when translating and summing the spherical and cylindrical solutions
of the Helmholtz equation in a periodic arrangement. These sums have a notoriously slow
convergence and at least for lattices with a dimension :math:`d > 1` it is not advisable
to use the direct approach. Fortunately, it is possible to convert them to exponentially
convergent series, which are implemented here. For details on the math, see the
references below.

The lattice of dimension :math:`d \leq 3` (:math:`d \leq 2` in the second case) is
denoted with :math:`\Lambda_d` and consists of all vectors
:math:`\boldsymbol R = \sum_{i=1}^{d} n_i \boldsymbol a_i` with :math:`\boldsymbol a_i`
being basis vectors of the lattice and :math:`n_i \in \mathbb Z`. For :math:`d = 2` the
lattice is in the `z = 0` plane and for :math:`d = 1` it is along the z-axis. The vector
:math:`\boldsymbol r` is arbitrary but for best convergence it should be reduced to the
Wigner-Seitz cell of the lattice. The summation excludes the point
:math:`\boldsymbol R + \boldsymbol r = 0` if it exists, which is indicated by the prime
next to the summation sign.

The wave in the lattice is defined by its -- possibly complex-valued -- wave number
:math:`k` and the (real) components of the wave vector
:math:`\boldsymbol k_\parallel \in \mathbb R^d` that are parallel to the lattice.

The expressions include the (spherical) Hankel functions of first kind :math:`H_m^{(1)}`
(:math:`h_l^{(1)}`) and the spherical harmonics :math:`Y_{lm}`. The angles
:math:`\theta` and :math:`\varphi` are the polar and azimuthal angle, when expressing
the points in spherical coordinates. In the first case, the degree is
:math:`l \in \mathbb N_0` and the order is :math:`\mathbb Z \ni m \leq l`. In the second
case :math:`m \in \mathbb Z`.


Available functions
===================

Accelerated lattice summations
------------------------------

.. autosummary::
   :toctree:

   lsumcw1d
   lsumcw1d_shift
   lsumcw2d
   lsumsw1d
   lsumsw1d_shift
   lsumsw2d
   lsumsw2d_shift
   lsumsw3d


Direct summations
-----------------

The functions are almost only here for benchmarking and comparison.

.. autosummary::
   :toctree:

   dsumcw1d
   dsumcw1d_shift
   dsumcw2d
   dsumsw1d
   dsumsw1d_shift
   dsumsw2d
   dsumsw2d_shift
   dsumsw3d


Miscellaneous functions
-----------------------

.. autosummary::
   :toctree:

   area
   cube
   cubeedge
   diffr_orders_circle
   reciprocal
   volume


Cython module
-------------

.. autosummary::
   :toctree:
   :template: cython-module

   cython_lattice


References
==========

* `[1] K. Kambe, Zeitschrift Für Naturforschung A 22, 3 (1967). <https://doi.org/10.1515/zna-1967-0305>`_
* `[2] K. Kambe, Zeitschrift Für Naturforschung A 23, 9 (1968). <https://doi.org/10.1515/zna-1968-0908>`_
* `[3] C. M. Linton, SIAM Rev. 52, 630 (2010). <https://doi.org/10.1137/09075130X>`_
* `[4] D. Beutel el al., J. Opt. Soc. Am. B (2021). <https://doi.org/10.1364/JOSAB.419645>`_
"""

import numpy as np

from treams.lattice import _misc
from treams.lattice._gufuncs import *  # noqa: F403
from treams.lattice._misc import cube, cubeedge  # noqa: F401


def diffr_orders_circle(b, rmax):
    """Diffraction orders in a circle.

    Given the reciprocal lattice defined by the vectors that make up the rows of `b`,
    return all diffraction orders within a circle of radius `rmax`.

    Args:
        b (float, (2, 2)-array): Reciprocal lattice vectors
        rmax (float): Maximal radius

    Returns:
        float array
    """
    return _misc.diffr_orders_circle(np.array(b), rmax)


def lsumsw(dim, l, m, k, kpar, a, r, eta, out=None, **kwargs):  # noqa: E741
    if dim == 1:
        return lsumsw1d_shift(l, m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    if dim == 2:
        return lsumsw2d_shift(l, m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    if dim == 3:
        return lsumsw3d(l, m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    raise ValueError(f"invalid dimension '{dim}'")


def realsumsw(dim, l, m, k, kpar, a, r, eta, out=None, **kwargs):  # noqa: E741
    if dim == 1:
        return realsumsw1d_shift(  # noqa: F405
            l, m, k, kpar, a, r, eta, out=out, **kwargs
        )
    if dim == 2:
        return realsumsw2d_shift(  # noqa: F405
            l, m, k, kpar, a, r, eta, out=out, **kwargs
        )
    if dim == 3:
        return realsumsw3d(l, m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    raise ValueError(f"invalid dimension '{dim}'")


def recsumsw(dim, l, m, k, kpar, a, r, eta, out=None, **kwargs):  # noqa: E741
    if dim == 1:
        return recsumsw1d_shift(  # noqa: F405
            l, m, k, kpar, a, r, eta, out=out, **kwargs
        )
    if dim == 2:
        return recsumsw2d_shift(  # noqa: F405
            l, m, k, kpar, a, r, eta, out=out, **kwargs
        )
    if dim == 3:
        return recsumsw3d(l, m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    raise ValueError(f"invalid dimension '{dim}'")


def lsumcw(dim, m, k, kpar, a, r, eta, out=None, **kwargs):  # noqa: E741
    if dim == 1:
        return lsumcw1d_shift(m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    if dim == 2:
        return lsumcw2d(m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    raise ValueError(f"invalid dimension '{dim}'")


def realsumcw(dim, m, k, kpar, a, r, eta, out=None, **kwargs):
    if dim == 1:
        return realsumcw1d_shift(m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    if dim == 2:
        return realsumcw2d(m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    raise ValueError(f"invalid dimension '{dim}'")


def recsumcw(dim, m, k, kpar, a, r, eta, out=None, **kwargs):
    if dim == 1:
        return recsumcw1d_shift(m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    if dim == 2:
        return recsumcw2d(m, k, kpar, a, r, eta, out=out, **kwargs)  # noqa: F405
    raise ValueError(f"invalid dimension '{dim}'")


def dsumsw(dim, l, m, k, kpar, a, r, i, out=None, **kwargs):  # noqa: E741
    if dim == 1:
        return dsumsw1d_shift(l, m, k, kpar, a, r, i, out=out, **kwargs)  # noqa: F405
    if dim == 2:
        return dsumsw2d_shift(l, m, k, kpar, a, r, i, out=out, **kwargs)  # noqa: F405
    if dim == 3:
        return dsumsw3d(l, m, k, kpar, a, r, i, out=out, **kwargs)  # noqa: F405
    raise ValueError(f"invalid dimension '{dim}'")


def dsumcw(dim, m, k, kpar, a, r, i, out=None, **kwargs):
    if dim == 1:
        return dsumcw1d_shift(m, k, kpar, a, r, i, out=out, **kwargs)  # noqa: F405
    if dim == 2:
        return dsumcw2d(m, k, kpar, a, r, i, out=out, **kwargs)  # noqa: F405
    raise ValueError(f"invalid dimension '{dim}'")
