r"""
======================================
Lattice summations :mod:`ptsa.lattice`
======================================

.. currentmodule:: ptsa.lattice


Calculates the lattice sums of the forms

.. math::
   D_{lm}(k, \boldsymbol k_\parallel, \boldsymbol r, \Lambda_d)
   = \sideset{}{'}{\sum_{\boldsymbol R \in \Lambda_d}}
   h_l^{(1)}(k|\boldsymbol r + \boldsymbol R|)
   Y_{lm}(\theta_{-\boldsymbol r - \boldsymbol R}, \varphi_{-\boldsymbol r - \boldsymbol R})
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

Accellerated lattice summations
-------------------------------

.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

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
   :toctree: generated/

   area
   cube
   cubeedge
   diffr_orders_circle
   reciprocal
   volume


References
==========

* `[1] K. Kambe, Zeitschrift Für Naturforschung A 22, 3 (1967). <https://doi.org/10.1515/zna-1967-0305>`_
* `[2] K. Kambe, Zeitschrift Für Naturforschung A 23, 9 (1968). <https://doi.org/10.1515/zna-1968-0908>`_
* `[3] C. M. Linton, SIAM Rev. 52, 630 (2010). <https://doi.org/10.1137/09075130X>`_
* `[4] D. Beutel el al., J. Opt. Soc. Am. B (2021). <https://doi.org/10.1364/JOSAB.419645>`_


"""

import numpy as np

from ptsa.lattice._gufuncs import *
from ptsa.lattice._misc import cube, cubeedge
from ptsa.lattice import _misc


def diffr_orders_circle(b, rmax):
    """
    Diffraction orders in a circle

    Given the reciprocal lattice defined by the vectors that make up the rows of `b`,
    return all diffraction orders within a circle of radius `rmax`.

    Args:
        b (float, (2, 2)-array): Reciprocal lattice vectors
        rmax (float): Maximal radius

    Returns:
        float array
    """
    return _misc.diffr_orders_circle(np.array(b), rmax)
