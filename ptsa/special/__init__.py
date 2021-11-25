"""
=====================================
Special functions :mod:`ptsa.special`
=====================================

.. currentmodule:: ptsa.special

Special mathematical functions used in :mod:`ptsa`. Some functions are reexported from
:py:mod:`scipy.special`. Most functions are available as Numpy universal functions
(:py:class:`numpy.ufunc`) or as generalized universal functions
(:std:doc:`reference/c-api/generalized-ufuncs`).

Available functions
===================

Bessel and Hankel functions, with their spherical counterparts, derivatives
---------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   hankel1_d
   hankel2_d
   jv_d
   yv_d
   spherical_hankel1
   spherical_hankel2
   spherical_hankel1_d
   spherical_hankel2_d


Those functions are just reexported from Scipy. So, one only needs to import this
subpackage within ptsa.

.. autosummary::
   :toctree: generated/

   hankel1
   hankel2
   jv
   yv
   spherical_jn
   spherical_yn


Those functions just wrap Scipy functions with special optional arguments to be able to
analogously access them like their non-spherical counterparts:

.. autosummary::
   :toctree: generated/

   spherical_jn_d
   spherical_yn_d


Scipy functions with enhanced domain
------------------------------------

.. autosummary::
   :toctree: generated/

   sph_harm
   lpmv


Integrals for the Ewald summation
---------------------------------

.. autosummary::
   :toctree: generated/

   incgamma
   intkambe


Wigner d- and Wigner D-matrix elements
--------------------------------------

.. autosummary::
   :toctree: generated/

   wignersmalld
   wignerd


Wigner 3j-symbols
-----------------

.. autosummary::
   :toctree: generated/

   wigner3j


Vector wave functions
--------------------------

.. autosummary::
   :toctree: generated/

   pi_fun
   tau_fun

Spherical waves and translation coefficients

.. autosummary::
   :toctree: generated/

   vsh_X
   vsh_Y
   vsh_Z
   vsw_M
   vsw_N
   vsw_A
   vsw_rM
   vsw_rN
   vsw_rA
   tl_vsw_A
   tl_vsw_B
   tl_vsw_rA
   tl_vsw_rB


Cylindrical waves

.. autosummary::
   :toctree: generated/

   vcw_M
   vcw_N
   vcw_A
   vcw_rM
   vcw_rN
   vcw_rA
   tl_vcw
   tl_vcw_r


Plane waves

.. autosummary::
   :toctree: generated/

   vpw_M
   vpw_N
   vpw_A


Coordinate system transformations
---------------------------------

.. autosummary::
   :toctree: generated/

   car2cyl
   car2sph
   cyl2car
   cyl2sph
   sph2car
   sph2cyl
   vcar2cyl
   vcar2sph
   vcyl2car
   vcyl2sph
   vsph2car
   vsph2cyl
   car2pol
   pol2car
   vcar2pol
   vpol2car
"""

from scipy.special import (
    hankel1,
    hankel2,
    jv,
    spherical_jn,
    spherical_yn,
    yv,
)

from ptsa.special import _ufuncs
from ptsa.special import _gufuncs
from ptsa.special._ufuncs import *
from ptsa.special._gufuncs import *


def spherical_jn_d(n, z):
    """Derivative of the spherical Bessel function of the first kind

    This is simply a wrapper for `scipy.special.spherical_jn(n, z, True)`, see
    :py:func:`scipy.special.spherical_jn`. It's here to have a consistent way of
    calling the derivative of a (spherical) Bessel or Hankel function.

    Args:
        n (int, array_like): Order
        z (float or complex, array_like): Argument

    Returns:
        float or complex

    References:
        - `DLMF 10.47 <https://dlmf.nist.gov/10.47>`_
        - `DLMF 10.51 <https://dlmf.nist.gov/10.51>`_
    """
    return spherical_jn(n, z, True)


def spherical_yn_d(n, z):
    """Derivative of the spherical Bessel function of the second kind

    This is simply a wrapper for `scipy.special.spherical_yn(n, z, True)`, see
    :py:func:`scipy.special.spherical_jn`. It's here to have a consistent way of
    calling the derivative of a (spherical) Bessel or Hankel function.

    Args:
        n (int, array_like): Order
        z (float or complex, array_like): Argument

    Returns:
        float or complex

    References:
        - `DLMF 10.47 <https://dlmf.nist.gov/10.47>`_
        - `DLMF 10.51 <https://dlmf.nist.gov/10.51>`_
    """
    return spherical_yn(n, z, True)
