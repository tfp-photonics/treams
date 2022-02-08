"""
=======================
Miscellaneous functions
=======================

.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   basischange
   firstbrillouin1d
   firstbrillouin2d
   firstbrillouin3d
   pickmodes
   refractive_index
   wave_vec_z
"""

import numpy as np

import ptsa.lattice as la


def refractive_index(epsilon=1, mu=1, kappa=0):
    r"""
    Refractive index of a (chiral) medium

    The refractive indeces in a chiral medium :math:`\sqrt{\epsilon\mu} \mp \kappa` are
    returned with the negative helicity result first.

    Args:
        epsilon (float or complex, array_like, optional): Relative permittivity,
            defaults to 1.
        mu (float or complex, array_like, optional): Relative permeability, defaults to 1.
        kappa (float or complex, array_like, optional): Chirality parameter, defaults to 0.

    Returns:
        float or complex, (2,)-array

    """
    epsilon = np.array(epsilon)
    n = np.sqrt(epsilon * mu)
    res = np.stack((n - kappa, n + kappa), axis=-1)
    res[np.imag(res) < 0] *= -1
    return res


def basischange(out, in_=None):
    """
    Coefficients for the basis change between helicity and parity modes

    Args:
        out (3- or 4-tuple of (M,)-arrays): Output modes, the last array is taken as
            polarization.
        in_ (3- or 4-tuple of (N,)-arrays, optional): Input modes, if none are given,
            equal to the output modes

    Returns:
        float, ((M, N)-array
    """
    if in_ is None:
        in_ = out
    out = np.array([*zip(*out)])
    in_ = np.array([*zip(*in_)])
    res = np.zeros((out.shape[0], in_.shape[0]))
    out = out[:, None, :]
    sqhalf = np.sqrt(0.5)
    equal = (out[:, :, :-1] == in_[:, :-1]).all(axis=-1)
    minus = np.logical_and(out[:, :, -1] == in_[:, -1], in_[:, -1] == 0)
    res[equal] = sqhalf
    res[np.logical_and(equal, minus)] = -sqhalf
    return res


def pickmodes(out, in_):
    """
    Coefficients to pick modes

    Args:
        out (3- or 4-tuple of (M,)-arrays): Output modes, the last array is taken as
            polarization.
        in_ (3- or 4-tuple of (N,)-arrays, optional): Input modes, if none are given,
            equal to the output modes

    Returns:
        float, ((M, N)-array
    """
    out = np.array([*zip(*out)])
    in_ = np.array([*zip(*in_)])
    return np.all(out[:, None, :] == in_, axis=-1)


# def wave_vec_zs(kpars, ks):
#     kpars = np.array(kpars)
#     ks = np.array(ks)
#     if ks.ndim == 0:
#         nmat = npol = 1
#     elif ks.ndim == 1:
#         npol = ks.shape[0]
#         nmat = 1
#     elif ks.ndim == 2:
#         nmat, npol = ks.shape
#     else:
#         raise ValueError("ks has invalid shape")
#     res = np.zeros((nmat, kpars.shape[0], npol), np.complex128)
#     for i, kpar in enumerate(kpars):
#         res[:, i, :] = wave_vec_z(kpar[0], kpar[1], ks)
#     return res


def wave_vec_z(kx, ky, k):
    r"""
    Z component of the wave vector with positive imaginary part

    The result is :math:`k_z = \sqrt{k^2 - k_x^2 - k_y^2}` with
    :math:`\arg k_z \in \[ 0, \pi )`.

    Args:
        kx (float, array_like): X component of the wave vector
        ky (float, array_like): Y component of the wave vector
        k (float or complex, array_like): Wave number

    Returns:
        complex
    """
    kx = np.array(kx)
    ky = np.array(ky)
    k = np.array(k, complex)
    res = np.sqrt(k * k - kx * kx - ky * ky)
    if res.ndim == 0 and res.imag < 0:
        res = -res
    elif res.ndim > 0:
        res[np.imag(res) < 0] *= -1
    return res


def firstbrillouin1d(kpar, b):
    """
    Reduce the 1d wave vector (actually just a number) to the first Brillouin zone, i.e.
    the range `(-b/2, b/2]`

    Args:
        kpar (float64): (parallel) wave vector
        b (float64): reciprocal lattice vector

    Returns:
        float64
    """
    kpar -= b * np.round(kpar / b)
    if kpar > 0.5 * b:
        kpar -= b
    if kpar <= -0.5 * b:
        kpar += b
    return kpar


def firstbrillouin2d(kpar, b, n=2):
    """
    Reduce the 2d wave vector to the first Brillouin zone.

    The reduction to the first Brillouin zone is first approximated roughly. From this
    approximated vector and its 8 neighbours, the shortest one is picked. As a
    sufficient approximation is not guaranteed (especially for extreme geometries),
    this process is iterated `n` times.

    Args:
        kpar (1d-array): parallel wave vector
        b (2d-array): reciprocal lattice vectors
        n (int): number of iterations

    Returns:
        (1d-array)
    """
    kparstart = kpar
    b1 = b[0, :]
    b2 = b[1, :]
    normsq1 = b1 @ b1
    normsq2 = b2 @ b2
    normsqp = (b1 + b2) @ (b1 + b2)
    normsqm = (b1 - b2) @ (b1 - b2)
    if (
        normsqp < normsq1 - 1e-16
        or normsqp < normsq2 - 1e-16
        or normsqm < normsq1 - 1e-16
        or normsqm < normsq2 - 1e-16
    ):
        raise ValueError("Lattice vectors are not of minimal length")
    # Rough estimate
    kpar -= b1 * np.round((kpar @ b1) / normsq1)
    kpar -= b2 * np.round((kpar @ b2) / normsq2)
    # todo: precise
    options = kpar + la.cube(2, 1) @ b
    for i, option in enumerate(options):
        if option @ option < kpar @ kpar:
            kpar = option
    if n == 0 or np.array_equal(kpar, kparstart):
        return kpar
    return firstbrillouin2d(kpar, b, n - 1)


def firstbrillouin3d(kpar, b, n=2):
    """
    Reduce the 3d wave vector to the first Brillouin zone.

    The reduction to the first Brillouin zone is first approximated roughly. From this
    approximated vector and its 26 neighbours, the shortest one is picked. As a
    sufficient approximation is not guaranteed (especially for extreme geometries),
    this process is iterated `n` times.

    Args:
        kpar (1d-array): parallel wave vector
        b (2d-array): reciprocal lattice vectors
        n (int): number of iterations

    Returns:
        (1d-array)
    """
    kparstart = kpar
    b1 = b[0, :]
    b2 = b[1, :]
    b3 = b[2, :]
    normsq1 = b1 @ b1
    normsq2 = b2 @ b2
    normsq3 = b3 @ b3
    # todo: Minimal length
    # Rough estimate
    kpar -= b1 * np.round((kpar @ b1) / normsq1)
    kpar -= b2 * np.round((kpar @ b2) / normsq2)
    kpar -= b3 * np.round((kpar @ b3) / normsq3)
    # todo: precise
    options = kpar + la.cube(3, 1) @ b
    for i, option in enumerate(options):
        if option @ option < kpar @ kpar:
            kpar = option
    if n == 0 or np.array_equal(kpar, kparstart):
        return kpar
    return firstbrillouin3d(kpar, b, n - 1)
