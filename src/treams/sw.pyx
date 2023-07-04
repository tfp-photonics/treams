"""Spherical wave module.

.. autosummary::
   :toctree:

   periodic_to_cw
   periodic_to_pw
   rotate
   translate
   translate_periodic

"""

import numpy as np

from treams import config, lattice

cimport numpy as np
from libc.math cimport atan2
from libc.math cimport exp as expd
from libc.math cimport lgamma, pi
from libc.math cimport sqrt as sqrtd
from libc.stdlib cimport labs

cimport treams.special.cython_special as cs
from treams.special._misc cimport abs, double_complex, minusonepow, sqrt


cdef extern from "<complex.h>" nogil:
    double complex cpow(double complex x, double complex y)
    double complex cexp(double complex z)
    double creal(double complex z)
    double cimag(double complex z)


ctypedef fused number_t:
    double complex
    double


cdef double complex _ctranslate_sh(long lambda_, long mu, long pol, long l, long m, long qol, number_t kr, double theta, double phi) nogil:
    if abs(kr) < 1e-16:
        return 0.0j
    if pol == qol and (pol == 0 or pol == 1):
        return cs.tl_vsw_A(lambda_, mu, l, m, kr, theta, phi) + (
            2 * pol - 1
        ) * cs.tl_vsw_B(lambda_, mu, l, m, kr, theta, phi)
    if (pol == 1 and qol == 0) or (pol == 0 and qol == 1):
        return 0.0j
    raise ValueError("Polarization must be defined by 0 or 1")


cdef double complex _ctranslate_rh(long lambda_, long mu, long pol, long l, long m, long qol, number_t kr, double theta, double phi) nogil:
    if pol == qol and (pol == 0 or pol == 1):
        return cs.tl_vsw_rA(lambda_, mu, l, m, kr, theta, phi) + (
            2 * pol - 1
        ) * cs.tl_vsw_rB(lambda_, mu, l, m, kr, theta, phi)
    if (pol == 1 and qol == 0) or (pol == 0 and qol == 1):
        return 0.0j
    raise ValueError("Polarization must be defined by 0 or 1")


cdef double complex _ctranslate_sp(long lambda_, long mu, long pol, long l, long m, long qol, number_t kr, double theta, double phi) nogil:
    if abs(kr) < 1e-16:
        return 0.0j
    if pol == qol and (pol == 0 or pol == 1):
        return cs.tl_vsw_A(lambda_, mu, l, m, kr, theta, phi)
    if (pol == 1 and qol == 0) or (pol == 0 and qol == 1):
        return cs.tl_vsw_B(lambda_, mu, l, m, kr, theta, phi)
    raise ValueError("Polarization must be defined by 0 or 1")


cdef double complex _ctranslate_rp(long lambda_, long mu, long pol, long l, long m, long qol, number_t kr, double theta, double phi) nogil:
    if pol == qol and (pol == 0 or pol == 1):
        return cs.tl_vsw_rA(lambda_, mu, l, m, kr, theta, phi)
    if (pol == 1 and qol == 0) or (pol == 0 and qol == 1):
        return cs.tl_vsw_rB(lambda_, mu, l, m, kr, theta, phi)
    raise ValueError("Polarization must be defined by 0 or 1")


cdef void _loop_D_llllllddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *ip7 = args[7]
    cdef char *ip8 = args[8]
    cdef char *op0 = args[9]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(long, long, long, long, long, long, double, double, double) nogil>func)(
            <long>(<long*>ip0)[0],
            <long>(<long*>ip1)[0],
            <long>(<long*>ip2)[0],
            <long>(<long*>ip3)[0],
            <long>(<long*>ip4)[0],
            <long>(<long*>ip5)[0],
            <double>(<double*>ip6)[0],
            <double>(<double*>ip7)[0],
            <double>(<double*>ip8)[0],
        )
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        ip7 += steps[7]
        ip8 += steps[8]
        op0 += steps[9]


cdef void _loop_D_llllllDdd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *ip7 = args[7]
    cdef char *ip8 = args[8]
    cdef char *op0 = args[9]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(long, long, long, long, long, long, double complex, double, double) nogil>func)(
            <long>(<long*>ip0)[0],
            <long>(<long*>ip1)[0],
            <long>(<long*>ip2)[0],
            <long>(<long*>ip3)[0],
            <long>(<long*>ip4)[0],
            <long>(<long*>ip5)[0],
            <double complex>(<double complex*>ip6)[0],
            <double>(<double*>ip7)[0],
            <double>(<double*>ip8)[0],
        )
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        ip7 += steps[7]
        ip8 += steps[8]
        op0 += steps[9]


np.import_array()
np.import_ufunc()


cdef np.PyUFuncGenericFunction ufunc_translate_loops[2]
cdef void *ufunc_translate_sh_data[2]
cdef void *ufunc_translate_rh_data[2]
cdef void *ufunc_translate_sp_data[2]
cdef void *ufunc_translate_rp_data[2]
cdef char ufunc_translate_types[2 * 10]

ufunc_translate_loops[0] = <np.PyUFuncGenericFunction>_loop_D_llllllddd
ufunc_translate_loops[1] = <np.PyUFuncGenericFunction>_loop_D_llllllDdd
ufunc_translate_types[0] = <char>np.NPY_LONG
ufunc_translate_types[1] = <char>np.NPY_LONG
ufunc_translate_types[2] = <char>np.NPY_LONG
ufunc_translate_types[3] = <char>np.NPY_LONG
ufunc_translate_types[4] = <char>np.NPY_LONG
ufunc_translate_types[5] = <char>np.NPY_LONG
ufunc_translate_types[6] = <char>np.NPY_DOUBLE
ufunc_translate_types[7] = <char>np.NPY_DOUBLE
ufunc_translate_types[8] = <char>np.NPY_DOUBLE
ufunc_translate_types[9] = <char>np.NPY_CDOUBLE
ufunc_translate_types[10] = <char>np.NPY_LONG
ufunc_translate_types[11] = <char>np.NPY_LONG
ufunc_translate_types[12] = <char>np.NPY_LONG
ufunc_translate_types[13] = <char>np.NPY_LONG
ufunc_translate_types[14] = <char>np.NPY_LONG
ufunc_translate_types[15] = <char>np.NPY_LONG
ufunc_translate_types[16] = <char>np.NPY_CDOUBLE
ufunc_translate_types[17] = <char>np.NPY_DOUBLE
ufunc_translate_types[18] = <char>np.NPY_DOUBLE
ufunc_translate_types[19] = <char>np.NPY_CDOUBLE
ufunc_translate_sh_data[0] = <void*>_ctranslate_sh[double]
ufunc_translate_sh_data[1] = <void*>_ctranslate_sh[double_complex]
ufunc_translate_rh_data[0] = <void*>_ctranslate_rh[double]
ufunc_translate_rh_data[1] = <void*>_ctranslate_rh[double_complex]
ufunc_translate_sp_data[0] = <void*>_ctranslate_sp[double]
ufunc_translate_sp_data[1] = <void*>_ctranslate_sp[double_complex]
ufunc_translate_rp_data[0] = <void*>_ctranslate_rp[double]
ufunc_translate_rp_data[1] = <void*>_ctranslate_rp[double_complex]

_translate_sh = np.PyUFunc_FromFuncAndData(
    ufunc_translate_loops,
    ufunc_translate_sh_data,
    ufunc_translate_types,
    2,
    9,
    1,
    0,
    "_translate_sh",
    "",
    0,
)
_translate_rh = np.PyUFunc_FromFuncAndData(
    ufunc_translate_loops,
    ufunc_translate_rh_data,
    ufunc_translate_types,
    2,
    9,
    1,
    0,
    "_translate_rh",
    "",
    0,
)
_translate_sp = np.PyUFunc_FromFuncAndData(
    ufunc_translate_loops,
    ufunc_translate_sp_data,
    ufunc_translate_types,
    2,
    9,
    1,
    0,
    "_translate_sp",
    "",
    0,
)
_translate_rp = np.PyUFunc_FromFuncAndData(
    ufunc_translate_loops,
    ufunc_translate_rp_data,
    ufunc_translate_types,
    2,
    9,
    1,
    0,
    "_translate_rp",
    "",
    0,
)


def translate(lambda_, mu, pol, l, m, qol, kr, theta, phi, poltype=None, singular=True, *args, **kwargs):
    """translate(lambda_, mu, pol, l, m, qol, kr, theta, phi, helicity=True, singular=True)

    Translation coefficient for spherical modes.

    Returns the correct translation coefficient from :func:`treams.special.tl_vsw_A`,
    :func:`treams.special.tl_vsw_B`, :func:`treams.special.tl_vsw_rA`, and
    :func:`treams.special.tl_vsw_rB` or a combination thereof for the specified mode and
    basis.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        lambda_ (int, array_like): Degree of the destination mode
        mu (int, array_like): Order of the destination mode
        pol (int, array_like): Polarization of the destination mode
        l (int, array_like): Degree of the source mode
        m (int, array_like): Order of the source mode
        qol (int, array_like): Polarization of the source mode
        kr (float or complex, array_like): Translation distance in units of the wave number
        theta (float, array_like): Polar angle
        phi (float, array_like): Azimuthal angle
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.
        singular (bool, optional): If true, singular translation coefficients are used,
            else regular coefficients. Defaults to ``True``.

    Returns:
        complex
    """
    poltype = config.POLTYPE if poltype is None else poltype
    if poltype == "helicity":
        if singular:
            return _translate_sh(lambda_, mu, pol, l, m, qol, kr, theta, phi, *args, **kwargs)
        return _translate_rh(lambda_, mu, pol, l, m, qol, kr, theta, phi, *args, **kwargs)
    elif poltype == "parity":
        if singular:
            return _translate_sp(lambda_, mu, pol, l, m, qol, kr, theta, phi, *args, **kwargs)
        return _translate_rp(lambda_, mu, pol, l, m, qol, kr, theta, phi, *args, **kwargs)
    raise ValueError(f"invalid poltype '{poltype}'")


cdef double complex _crotate(long lambda_, long mu, long pol1, long l, long m, long pol2, double phi, double theta, double psi) nogil:
    """
    Rotation coefficient for the rotation around the Euler angles phi, theta and psi.
    It is intended to be used to construct the rotation matrix manually.
    """
    if lambda_ == l and (pol1 == pol2 == 0 or pol1 == pol2 == 1):
        return cs.wignerd(l, mu, m, phi, theta, psi)
    if (pol1 == 0 or pol1 == 1) and (pol2 == 0 or pol2 == 1):
        return 0
    raise ValueError('TODO')


cdef np.PyUFuncGenericFunction ufunc_rotate_loops[1]
cdef void *ufunc_rotate_data[1]
cdef char ufunc_rotate_types[10]

ufunc_rotate_loops[0] = <np.PyUFuncGenericFunction>_loop_D_llllllddd
ufunc_rotate_types[0] = <char>np.NPY_LONG
ufunc_rotate_types[1] = <char>np.NPY_LONG
ufunc_rotate_types[2] = <char>np.NPY_LONG
ufunc_rotate_types[3] = <char>np.NPY_LONG
ufunc_rotate_types[4] = <char>np.NPY_LONG
ufunc_rotate_types[5] = <char>np.NPY_LONG
ufunc_rotate_types[6] = <char>np.NPY_DOUBLE
ufunc_rotate_types[7] = <char>np.NPY_DOUBLE
ufunc_rotate_types[8] = <char>np.NPY_DOUBLE
ufunc_rotate_types[9] = <char>np.NPY_CDOUBLE
ufunc_rotate_data[0] = <void*>_crotate

_rotate = np.PyUFunc_FromFuncAndData(
    ufunc_rotate_loops,
    ufunc_rotate_data,
    ufunc_rotate_types,
    1,
    9,
    1,
    0,
    "_rotate",
    "",
    0,
)


def rotate(lambda_, mu, pol, l, m, qol, phi, theta=0, psi=0, *args, **kwargs):
    """rotate(lambda_, mu, pol, l, m, qol, phi, theta=0, psi=0)

    Rotation coefficient for spherical modes.

    Returns the correct rotation coefficient from :func:`treams.special.wignerd`. The
    angles are given as Euler angles in `z-y-z`-convention. In the intrinsic (object
    fixed coordinate system) convention the rotations are applied in the order phi
    first, theta second, psi third. In the extrinsic (global or reference frame fixed
    coordinate system) the rotations are applied psi first, theta second, phi third.

    Args:
        lambda_ (int, array_like): Degree of the destination mode
        mu (int, array_like): Order of the destination mode
        pol (int, array_like): Polarization of the destination mode
        l (int, array_like): Degree of the source mode
        m (int, array_like): Order of the source mode
        qol (int, array_like): Polarization of the source mode
        phi (float or complex, array_like): First Euler angle
        theta (float, array_like): Second Euler angle
        psi (float, array_like): Third Euler angle

    Returns:
        complex
    """
    return _rotate(lambda_, mu, pol, l, m, qol, phi, theta, psi, *args, **kwargs)


cdef double complex _transl_A_lattice(long lambda_, long mu, long l, long m, double complex *dlms, long step) nogil:
    cdef double complex pref = (
        minusonepow(m)
        * sqrtd(
            pi
            * (2 * l + 1)
            * (2 * lambda_ + 1)
            / <double>(l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * cpow(1.0j, lambda_ - l)
    )
    cdef double complex dlm, res = 0
    cdef long p
    for p in range(l + lambda_, max(labs(lambda_ - l), labs(mu - m)) - 1, -2):
        dlm = dlms[(p * (p + 1) + m - mu) * step]
        res += (
            dlm
            * cpow(1.0j, p)
            * sqrtd(2 * p + 1)
            * cs.wigner3j(l, lambda_, p, m, -mu, -m + mu)
            * cs.wigner3j(l, lambda_, p, 0, 0, 0)
            * (l * (l + 1) + lambda_ * (lambda_ + 1) - p * (p + 1))
        )
    return res * pref


cdef double complex _transl_B_lattice(long lambda_, long mu, long l, long m, double complex *dlms, long step) nogil:
    cdef double complex pref = (
        minusonepow(m)
        * sqrtd(
            pi
            * (2 * l + 1)
            * (2 * lambda_ + 1)
            / <double>(l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * cpow(1.0j, lambda_ - l)
    )
    cdef double complex dlm, res = 0
    cdef long p
    for p in range(
        l + lambda_ - 1, max(labs(lambda_ - l) + 1, labs(mu - m)) - 1, -2
    ):
        dlm = dlms[(p * (p + 1) + m - mu) * step]
        res += (
            dlm
            * cpow(1.0j, p)
            * sqrtd(2 * p + 1)
            * cs.wigner3j(l, lambda_, p, m, -mu, -m + mu)
            * cs.wigner3j(l, lambda_, p - 1, 0, 0, 0)
            * sqrtd(
                (l + lambda_ + 1 + p)
                * (l + lambda_ + 1 - p)
                * (p - lambda_ + l)
                * (p + lambda_ - l)
            )
        )
    return res * pref


cdef double complex _ctranslate_periodic_p(long lambda_, long mu, long pol1, long l, long m, long pol2, double complex *dlms, long step) nogil:
    if pol1 == pol2:
        return _transl_A_lattice(lambda_, mu, l, m, dlms, step)
    return _transl_B_lattice(lambda_, mu, l, m, dlms, step)


cdef double complex _ctranslate_periodic_h(long lambda_, long mu, long pol1, long l, long m, long pol2, double complex *dlms, long step) nogil:
    if pol1 == pol2:
        return _transl_A_lattice(lambda_, mu, l, m, dlms, step) + (
            2 * pol1 - 1
        ) * _transl_B_lattice(lambda_, mu, l, m, dlms, step)
    return 0.0j


cdef void _loop_translate_periodic_h(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long dlmstep, step1 = steps[9] // sizeof(double complex), step2 = steps[10] // sizeof(double complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *ip7 = args[7]
    cdef char *op0 = args[8]
    cdef char *dlmchoice
    cdef double complex ov0
    for i in range(n):
        if <long>(<long*>ip2)[0] == 0:
            dlmchoice = ip6
            dlmstep = step1
        else:
            dlmchoice = ip7
            dlmstep = step2
        ov0 = (<double complex (*)(long, long, long, long, long, long, double complex*, long) nogil>func)(
            <long>(<long*>ip0)[0],
            <long>(<long*>ip1)[0],
            <long>(<long*>ip2)[0],
            <long>(<long*>ip3)[0],
            <long>(<long*>ip4)[0],
            <long>(<long*>ip5)[0],
            <double complex*>dlmchoice,
            dlmstep,
        )
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        ip7 += steps[7]
        op0 += steps[8]


cdef void _loop_translate_periodic_p(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long dlmstep = steps[8] // sizeof(double complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex (*)(long, long, long, long, long, long, double complex*, long) nogil>func)(
            <long>(<long*>ip0)[0],
            <long>(<long*>ip1)[0],
            <long>(<long*>ip2)[0],
            <long>(<long*>ip3)[0],
            <long>(<long*>ip4)[0],
            <long>(<long*>ip5)[0],
            <double complex*>ip6,
            dlmstep,
        )
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef np.PyUFuncGenericFunction gufunc_translate_periodic_h_loops[1]
cdef void *gufunc_translate_periodic_h_data[1]
cdef char gufunc_translate_periodic_h_types[9]

gufunc_translate_periodic_h_loops[0] = <np.PyUFuncGenericFunction>_loop_translate_periodic_h
gufunc_translate_periodic_h_types[0] = <char>np.NPY_LONG
gufunc_translate_periodic_h_types[1] = <char>np.NPY_LONG
gufunc_translate_periodic_h_types[2] = <char>np.NPY_LONG
gufunc_translate_periodic_h_types[3] = <char>np.NPY_LONG
gufunc_translate_periodic_h_types[4] = <char>np.NPY_LONG
gufunc_translate_periodic_h_types[5] = <char>np.NPY_LONG
gufunc_translate_periodic_h_types[6] = <char>np.NPY_CDOUBLE
gufunc_translate_periodic_h_types[7] = <char>np.NPY_CDOUBLE
gufunc_translate_periodic_h_types[8] = <char>np.NPY_CDOUBLE
gufunc_translate_periodic_h_data[0] = <void*>_ctranslate_periodic_h

_translate_periodic_h = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_translate_periodic_h_loops,
    gufunc_translate_periodic_h_data,
    gufunc_translate_periodic_h_types,
    1,
    8,
    1,
    0,
    "_translate_periodic_h",
    "",
    0,
    "(),(),(),(),(),(),(a),(a)->()",
)

cdef np.PyUFuncGenericFunction gufunc_translate_periodic_p_loops[1]
cdef void *gufunc_translate_periodic_p_data[1]
cdef char gufunc_translate_periodic_p_types[8]

gufunc_translate_periodic_p_loops[0] = <np.PyUFuncGenericFunction>_loop_translate_periodic_p
gufunc_translate_periodic_p_types[0] = <char>np.NPY_LONG
gufunc_translate_periodic_p_types[1] = <char>np.NPY_LONG
gufunc_translate_periodic_p_types[2] = <char>np.NPY_LONG
gufunc_translate_periodic_p_types[3] = <char>np.NPY_LONG
gufunc_translate_periodic_p_types[4] = <char>np.NPY_LONG
gufunc_translate_periodic_p_types[5] = <char>np.NPY_LONG
gufunc_translate_periodic_p_types[6] = <char>np.NPY_CDOUBLE
gufunc_translate_periodic_p_types[7] = <char>np.NPY_CDOUBLE
gufunc_translate_periodic_p_data[0] = <void*>_ctranslate_periodic_p

_translate_periodic_p = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_translate_periodic_p_loops,
    gufunc_translate_periodic_p_data,
    gufunc_translate_periodic_p_types,
    1,
    7,
    1,
    0,
    "_translate_periodic_p",
    "",
    0,
    "(),(),(),(),(),(),(a)->()",
)


def translate_periodic(ks, kpar, a, rs, out, in_=None, rsin=None, poltype=None, eta=0, func=lattice.lsumsw):
    """translate_periodic(ks, kpar, a, rs, out, in_=None, rsin=None, helicity=True, eta=0)

    Translation coefficients in a lattice.

    Returns the translation coefficents for the given modes in a lattice. The calculation
    uses the fast converging sums of :mod:`treams.lattice`.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        ks (float or complex, scalar or (2,)-array): Wave number(s) in the medium, use
            two values in chiral media, indexed analogous to the polarization values
        kpar (float, (D,)-array): Parallel component of the wave, defines the dimension with `1 <= D <= 3`
        a (float, (D,D)-array): Lattice vectors in each row of the array
        rs (float, (M, 3)-array): Shift vectors with respect to one lattice point
        out (3- or 4-tuple of integer arrays): Output modes
        in_ (3- or 4-tuple of integer arrays): Input modes, if none are given equal to
            the output modes
        rsin (float): Shift vectors to use with the input modes, if non are given equal
            to `rs`
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.
        eta (float or complex, optional): Cut between real and reciprocal space
            summation, if equal to zero, an estimation for the optimal value is done.

    Returns:
        complex array
    """
    poltype = config.POLTYPE if poltype is None else poltype
    if poltype not in ("helicity", "parity"):
        raise ValueError(f"invalid poltype '{poltype}'")
    if in_ is None:
        in_ = out
    out = (*(np.array(o) for o in out),)
    in_ = (*(np.array(i) for i in in_),)
    if len(out) < 3 or len(out) > 4:
        raise ValueError(f"invalid length of output modes {len(out)}, must be 3 or 4")
    elif len(out) == 3:
        out = (np.zeros_like(out[0]),) + out
    if len(in_) < 3 or len(in_) > 4:
        raise ValueError(f"invalid length of input modes {len(in_)}, must be 3 or 4")
    elif len(in_) == 3:
        in_ = (np.zeros_like(in_[0]),) + in_
    if rsin is None:
        rsin = rs
    modes = np.array([
        [l, m]
        for l in range(np.max(out[1]) + np.max(in_[1]) + 1)
        for m in range(-l, l + 1)
    ])
    ks = np.array(ks)
    ks = ks.reshape((-1, 1))
    if ks.shape[0] == 2 and ks[0, 0] == ks[1, 0]:
        ks = ks[:1, :]
    kpar = np.array(kpar)
    rs = np.array(rs)
    if rs.ndim == 1:
        rs = rs.reshape((1, -1))
    rsin = np.array(rsin)
    if rsin.ndim == 1:
        rsin = rsin.reshape((1, -1))
    rsdiff = -rs[:, None, None, None, :] + rsin[:, None, None, :]

    dim = 1 if kpar.ndim == 0 else kpar.shape[-1]
    # The result has the shape (n_rs, n_rs, n_ks, n_modes)
    dlms = func(dim, modes[:, 0], modes[:, 1], ks, kpar, a, rsdiff, eta)

    if poltype == "helicity":
        return _translate_periodic_h(
            *(o[:, None] for o in out[1:]),
            *in_[1:],
            dlms[out[0][:, None], in_[0], 0, :],
            dlms[out[0][:, None], in_[0], ks.shape[0] - 1, :],
        )
    return _translate_periodic_p(
        *(o[:, None] for o in out[1:]),
        *in_[1:],
        dlms[out[0][:, None], in_[0], 0, :],
    )


cdef double complex _cperiodic_to_pw_h(double kx, double ky, number_t kz, long polpw, long l, long m, long polvsw, double area) nogil:
    if polvsw != polpw:
        return 0.0j
    cdef number_t k = sqrt(kx * kx + ky * ky + kz * kz)
    cdef double complex costheta = kz / k
    cdef double phi = atan2(ky, kx)
    cdef double complex kz_s = kz
    if kz == 0:
        kz_s = 1e-20 + 1e-20j
    elif (cimag(kz_s) < 0) or (cimag(kz_s) == 0 and creal(kz_s) < 0):
        kz_s = -kz_s
    return (
        sqrtd(pi * (2 * l + 1) / <double>(l * (l + 1)))
        * expd((lgamma(l - m + 1) - lgamma(l + m + 1)) * 0.5)
        * cexp(1j * m * phi)
        * cpow(-1j, l)
        * (cs.tau_fun(l, m, costheta) + (2 * polpw - 1) * cs.pi_fun(l, m, costheta))
        / (area * k * kz_s)
    )


cdef double complex _cperiodic_to_pw_p(double kx, double ky, number_t kz, long polpw, long l, long m, long polvsw, double area) nogil:
    cdef number_t k = sqrt(kx * kx + ky * ky + kz * kz)
    cdef double complex costheta = kz / k
    cdef double phi = atan2(ky, kx)
    cdef double complex kz_s = kz
    if kz == 0:
        kz_s = 1e-20 + 1e-20j
    elif (cimag(kz_s) < 0) or (cimag(kz_s) == 0 and creal(kz_s) < 0):
        kz_s = -kz_s
    cdef double complex prefactor = (
        sqrtd(pi * (2 * l + 1) / <double>(l * (l + 1)))
        * expd((lgamma(l - m + 1) - lgamma(l + m + 1)) * 0.5)
        * cexp(1j * m * phi)
        * cpow(-1j, l)
        / (area * k * kz_s)
    )
    if polvsw == polpw:
        return prefactor * cs.tau_fun(l, m, costheta)
    return prefactor * cs.pi_fun(l, m, costheta)


cdef void _loop_periodic_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *ip7 = args[7]
    cdef char *op0 = args[8]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(double, double, double, long, long, long, long, double) nogil>func)(
            <double>(<double*>ip0)[0],
            <double>(<double*>ip1)[0],
            <double>(<double*>ip2)[0],
            <long>(<long*>ip3)[0],
            <long>(<long*>ip4)[0],
            <long>(<long*>ip5)[0],
            <long>(<long*>ip6)[0],
            <double>(<double*>ip7)[0],
        )
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        ip7 += steps[7]
        op0 += steps[8]


cdef void _loop_periodic_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *ip7 = args[7]
    cdef char *op0 = args[8]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(double, double, double complex, long, long, long, long, double) nogil>func)(
            <double>(<double*>ip0)[0],
            <double>(<double*>ip1)[0],
            <double complex>(<double complex*>ip2)[0],
            <long>(<long*>ip3)[0],
            <long>(<long*>ip4)[0],
            <long>(<long*>ip5)[0],
            <long>(<long*>ip6)[0],
            <double>(<double*>ip7)[0],
        )
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        ip7 += steps[7]
        op0 += steps[8]


cdef np.PyUFuncGenericFunction ufunc_pw_loops[2]
cdef void *ufunc_pw_h_data[2]
cdef void *ufunc_pw_p_data[2]
cdef char ufunc_pw_types[2 * 9]

ufunc_pw_loops[0] = <np.PyUFuncGenericFunction>_loop_periodic_d
ufunc_pw_loops[1] = <np.PyUFuncGenericFunction>_loop_periodic_D
ufunc_pw_types[0] = <char>np.NPY_DOUBLE
ufunc_pw_types[1] = <char>np.NPY_DOUBLE
ufunc_pw_types[2] = <char>np.NPY_DOUBLE
ufunc_pw_types[3] = <char>np.NPY_LONG
ufunc_pw_types[4] = <char>np.NPY_LONG
ufunc_pw_types[5] = <char>np.NPY_LONG
ufunc_pw_types[6] = <char>np.NPY_LONG
ufunc_pw_types[7] = <char>np.NPY_DOUBLE
ufunc_pw_types[8] = <char>np.NPY_CDOUBLE
ufunc_pw_types[9] = <char>np.NPY_DOUBLE
ufunc_pw_types[10] = <char>np.NPY_DOUBLE
ufunc_pw_types[11] = <char>np.NPY_CDOUBLE
ufunc_pw_types[12] = <char>np.NPY_LONG
ufunc_pw_types[13] = <char>np.NPY_LONG
ufunc_pw_types[14] = <char>np.NPY_LONG
ufunc_pw_types[15] = <char>np.NPY_LONG
ufunc_pw_types[16] = <char>np.NPY_DOUBLE
ufunc_pw_types[17] = <char>np.NPY_CDOUBLE
ufunc_pw_h_data[0] = <void*>_cperiodic_to_pw_h[double]
ufunc_pw_h_data[1] = <void*>_cperiodic_to_pw_h[double_complex]
ufunc_pw_p_data[0] = <void*>_cperiodic_to_pw_p[double]
ufunc_pw_p_data[1] = <void*>_cperiodic_to_pw_p[double_complex]

_periodic_to_pw_h = np.PyUFunc_FromFuncAndData(
    ufunc_pw_loops,
    ufunc_pw_h_data,
    ufunc_pw_types,
    2,
    8,
    1,
    0,
    "_periodic_to_pw_h",
    "",
    0,
)
_periodic_to_pw_p = np.PyUFunc_FromFuncAndData(
    ufunc_pw_loops,
    ufunc_pw_p_data,
    ufunc_pw_types,
    2,
    8,
    1,
    0,
    "_periodic_to_pw_p",
    "",
    0,
)


def periodic_to_pw(kx, ky, kz, pol, l, m, qol, area, poltype=None, *args, **kwargs):
    """periodic_to_pw(kx, ky, kz, pol, l, m, qol, area, helicity=True)

    Convert periodic spherical wave to plane wave.

    Returns the coefficient for the basis change in a periodic arrangement of spherical
    modes to plane waves. For multiple positions only diagonal values (with respect to
    the position) are returned. A correct phase factor is still necessary for the full
    result.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        kx (float, array_like): X component of destination mode wave vector
        ky (float, array_like): Y component of destination mode wave vector
        kz (float or complex, array_like): Z component of destination mode wave vector
        pol (int, array_like): Polarization of the destination mode
        l (int, array_like): Degree of the source mode
        m (int, array_like): Order of the source mode
        qol (int, array_like): Polarization of the source mode
        area (float, array_like): Unit cell area
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.

    Returns:
        complex
    """
    poltype = config.POLTYPE if poltype is None else poltype
    if poltype == "helicity":
        return _periodic_to_pw_h(kx, ky, kz, pol, l, m, qol, area, *args, **kwargs)
    elif poltype == "parity":
        return _periodic_to_pw_p(kx, ky, kz, pol, l, m, qol, area, *args, **kwargs)
    raise ValueError(f"invalid poltype '{poltype}'")


cdef double complex _cperiodic_to_cw_h(double kz, long mu, long polcw, long l, long m, long polsw, double complex k, double a) nogil:
    if mu != m or polsw != polcw:
        return 0
    cdef double complex costheta = kz / k
    return (
        sqrtd(pi * (2 * l + 1) / <double>(l * (l + 1)))
        * expd((lgamma(l - m + 1) - lgamma(l + m + 1)) * 0.5)
        * 0.5
        * cpow(-1j, l - m)
        * (cs.tau_fun(l, m, costheta) + (2 * polcw - 1) * cs.pi_fun(l, m, costheta))
        / (a * k)
    )


cdef double complex _cperiodic_to_cw_p(double kz, long mu, long polcw, long l, long m, long polsw, double complex k, double a) nogil:
    if mu != m:
        return 0
    cdef double complex costheta = kz / k
    cdef double complex prefactor = (
        sqrtd(pi * (2 * l + 1) / <double>(l * (l + 1)))
        * expd((lgamma(l - m + 1) - lgamma(l + m + 1)) * 0.5)
        * 0.5
        * cpow(-1j, l - m)
        / (a * k)
    )
    if polcw == polsw:
        return prefactor * cs.tau_fun(l, m, costheta)
    return prefactor * cs.pi_fun(l, m, costheta)


cdef void _loop_periodic_cw_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *ip7 = args[7]
    cdef char *op0 = args[8]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(double, long, long, long, long, long, double complex, double) nogil>func)(
            <double>(<double*>ip0)[0],
            <long>(<long*>ip1)[0],
            <long>(<long*>ip2)[0],
            <long>(<long*>ip3)[0],
            <long>(<long*>ip4)[0],
            <long>(<long*>ip5)[0],
            <double complex>(<double complex*>ip6)[0],
            <double>(<double*>ip7)[0],
        )
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        ip7 += steps[7]
        op0 += steps[8]


cdef np.PyUFuncGenericFunction ufunc_cw_loops[1]
cdef void *ufunc_cw_h_data[1]
cdef void *ufunc_cw_p_data[1]
cdef char ufunc_cw_types[9]

ufunc_cw_loops[0] = <np.PyUFuncGenericFunction>_loop_periodic_cw_D
ufunc_cw_types[0] = <char>np.NPY_DOUBLE
ufunc_cw_types[1] = <char>np.NPY_LONG
ufunc_cw_types[2] = <char>np.NPY_LONG
ufunc_cw_types[3] = <char>np.NPY_LONG
ufunc_cw_types[4] = <char>np.NPY_LONG
ufunc_cw_types[5] = <char>np.NPY_LONG
ufunc_cw_types[6] = <char>np.NPY_CDOUBLE
ufunc_cw_types[7] = <char>np.NPY_DOUBLE
ufunc_cw_types[8] = <char>np.NPY_CDOUBLE
ufunc_cw_h_data[0] = <void*>_cperiodic_to_cw_h
ufunc_cw_p_data[0] = <void*>_cperiodic_to_cw_p

_periodic_to_cw_h = np.PyUFunc_FromFuncAndData(
    ufunc_cw_loops,
    ufunc_cw_h_data,
    ufunc_cw_types,
    1,
    8,
    1,
    0,
    "_periodic_to_cw_h",
    "",
    0,
)
_periodic_to_cw_p = np.PyUFunc_FromFuncAndData(
    ufunc_cw_loops,
    ufunc_cw_p_data,
    ufunc_cw_types,
    1,
    8,
    1,
    0,
    "_periodic_to_cw_p",
    "",
    0,
)


def periodic_to_cw(kz, m, pol, l, mu, qol, k, area, poltype=None, *args, **kwargs):
    """periodic_to_cw(kz, m, pol, l, mu, qol, k, area, helicity=True)

    Convert periodic spherical wave to plane wave.

    Returns the coefficient for the basis change in a periodic arrangement of spherical
    modes to plane waves. For multiple positions only diagonal values (with respect to
    the position) are returned. A correct phase factor is still necessary for the full
    result.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        kz (float, array_like): Z component of destination mode wave vector
        m (int, array_like): Order or the destination mode
        pol (int, array_like): Polarization of the destination mode
        l (int, array_like): Degree of the source mode
        mu (int, array_like): Order of the source mode
        qol (int, array_like): Polarization of the source mode
        k (float or complex, array_like): Wave number
        area (float, array_like): Unit cell area
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.

    Returns:
        complex
    """
    poltype = config.POLTYPE if poltype is None else poltype
    if poltype == "helicity":
        return _periodic_to_cw_h(kz, m, pol, l, mu, qol, k, area, *args, **kwargs)
    elif poltype == "parity":
        return _periodic_to_cw_p(kz, m, pol, l, mu, qol, k, area, *args, **kwargs)
    raise ValueError(f"invalid poltype '{poltype}'")
