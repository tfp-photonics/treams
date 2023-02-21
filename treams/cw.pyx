"""Cylindrical wave module.

.. autosummary::
   :toctree: generated/

   periodic_to_pw
   rotate
   to_sw
   translate
   translate_periodic

"""

import numpy as np

from treams import config, lattice

cimport numpy as np
from libc.math cimport exp as expd
from libc.math cimport lgamma, pi
from libc.math cimport sqrt as sqrtd

cimport treams.special.cython_special as cs
from treams.special._misc cimport abs, double_complex, sqrt


cdef extern from "<complex.h>" nogil:
    double complex cexp(double complex z)
    double complex cpow(double complex x, double complex y)
    double creal(double complex z)
    double cimag(double complex z)


ctypedef fused number_t:
    double complex
    double


cdef double complex _ctranslate_r(double kz, long mu, long pol, double qz, long m, long qol, number_t krr, double phi, double z) nogil:
    if pol == qol and (pol == 0 or pol == 1):
        return cs.tl_vcw_r(kz, mu, qz, m, krr, phi, z)
    if (pol == 1 and qol == 0) or (pol == 0 and qol == 1):
        return 0.0j
    raise ValueError("Polarization must be defined by 0 or 1")


cdef double complex _ctranslate_s(double kz, long mu, long pol, double qz, long m, long qol, number_t krr, double phi, double z) nogil:
    if abs(krr) < 1e-16 and abs(z) < 1e-16:
        return 0.0j
    if pol == qol and (pol == 0 or pol == 1):
        return cs.tl_vcw(kz, mu, qz, m, krr, phi, z)
    if (pol == 1 and qol == 0) or (pol == 0 and qol == 1):
        return 0.0j
    raise ValueError("Polarization must be defined by 0 or 1")


cdef void _loop_D_dlldllddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        ov0 = (<double complex(*)(double, long, long, double, long, long, double, double, double) nogil>func)(
            <double>(<double*>ip0)[0],
            <long>(<long*>ip1)[0],
            <long>(<long*>ip2)[0],
            <double>(<double*>ip3)[0],
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


cdef void _loop_D_dlldllDdd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        ov0 = (<double complex(*)(double, long, long, double, long, long, double complex, double, double) nogil>func)(
            <double>(<double*>ip0)[0],
            <long>(<long*>ip1)[0],
            <long>(<long*>ip2)[0],
            <double>(<double*>ip3)[0],
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
cdef void *ufunc_translate_s_data[2]
cdef void *ufunc_translate_r_data[2]
cdef char ufunc_translate_types[2 * 10]

ufunc_translate_loops[0] = <np.PyUFuncGenericFunction>_loop_D_dlldllddd
ufunc_translate_loops[1] = <np.PyUFuncGenericFunction>_loop_D_dlldllDdd
ufunc_translate_types[0] = <char>np.NPY_DOUBLE
ufunc_translate_types[1] = <char>np.NPY_LONG
ufunc_translate_types[2] = <char>np.NPY_LONG
ufunc_translate_types[3] = <char>np.NPY_DOUBLE
ufunc_translate_types[4] = <char>np.NPY_LONG
ufunc_translate_types[5] = <char>np.NPY_LONG
ufunc_translate_types[6] = <char>np.NPY_DOUBLE
ufunc_translate_types[7] = <char>np.NPY_DOUBLE
ufunc_translate_types[8] = <char>np.NPY_DOUBLE
ufunc_translate_types[9] = <char>np.NPY_CDOUBLE
ufunc_translate_types[10] = <char>np.NPY_DOUBLE
ufunc_translate_types[11] = <char>np.NPY_LONG
ufunc_translate_types[12] = <char>np.NPY_LONG
ufunc_translate_types[13] = <char>np.NPY_DOUBLE
ufunc_translate_types[14] = <char>np.NPY_LONG
ufunc_translate_types[15] = <char>np.NPY_LONG
ufunc_translate_types[16] = <char>np.NPY_CDOUBLE
ufunc_translate_types[17] = <char>np.NPY_DOUBLE
ufunc_translate_types[18] = <char>np.NPY_DOUBLE
ufunc_translate_types[19] = <char>np.NPY_CDOUBLE
ufunc_translate_s_data[0] = <void*>_ctranslate_s[double]
ufunc_translate_s_data[1] = <void*>_ctranslate_s[double_complex]
ufunc_translate_r_data[0] = <void*>_ctranslate_r[double]
ufunc_translate_r_data[1] = <void*>_ctranslate_r[double_complex]

_translate_s = np.PyUFunc_FromFuncAndData(
    ufunc_translate_loops,
    ufunc_translate_s_data,
    ufunc_translate_types,
    2,
    9,
    1,
    0,
    '_translate_s',
    '',
    0,
)
_translate_r = np.PyUFunc_FromFuncAndData(
    ufunc_translate_loops,
    ufunc_translate_r_data,
    ufunc_translate_types,
    2,
    9,
    1,
    0,
    '_translate_r',
    '',
    0,
)


def translate(kz, mu, pol, qz, m, qol, krr, phi, z, singular=True, *args, **kwargs):
    r"""translate(kz, mu, pol, qz, m, qol, krr, phi, z, singular=True)

    Translation coefficient for cylindrical modes.

    Returns the correct translation coefficient from :func:`treams.special.tl_vcw` and
    :func:`treams.special.tl_vcw_r` or a combination thereof for the specified mode. A
    basis does not have to be specified since the coefficients are the same in helicity
    and parity modes.

    Args:
        kz (float, array_like): Z component of the destination mode
        mu (int, array_like): Order of the destination mode
        pol (int, array_like): Polarization of the destination mode
        qz (float, array_like): Z component of the source mode
        m (int, array_like): Order of the source mode
        qol (int, array_like): Polarization of the source mode
        krr (float or complex, array_like): Radial translation distance in units of the
            radial wave number :math:`k_\rho \rho`
        phi (float, array_like): Azimuthal angle
        z (float, array_like): Shift distance in z-direction
        singular (bool, optional): If true, singular translation coefficients are used,
            else regular coefficients. Defaults to ``True``.

    Returns:
        complex
    """
    if singular:
        return _translate_s(kz, mu, pol, qz, m, qol, krr, phi, z, *args, **kwargs)
    return _translate_r(kz, mu, pol, qz, m, qol, krr, phi, z, *args, **kwargs)


cdef double complex _crotate(double kz, long mu, long pol, double qz, long m, long qol, double phi) nogil:
    if (kz == qz) and (m == mu) and (pol == qol) and (pol == 0 or qol == 1):
        return cexp(1j * m * phi)
    elif (pol == 0 or pol == 1) and (qol == 0 or qol == 1):
        return 0
    raise ValueError("polarisation must be 0 or 1")


cdef void _loop_D_dlldlld(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
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
        ov0 = (<double complex(*)(double, long, long, double, long, long, double) nogil>func)(
            <double>(<double*>ip0)[0],
            <long>(<long*>ip1)[0],
            <long>(<long*>ip2)[0],
            <double>(<double*>ip3)[0],
            <long>(<long*>ip4)[0],
            <long>(<long*>ip5)[0],
            <double>(<double*>ip6)[0],
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


cdef np.PyUFuncGenericFunction ufunc_rotate_loops[1]
cdef void *ufunc_rotate_data[1]
cdef char ufunc_rotate_types[8]

ufunc_rotate_loops[0] = <np.PyUFuncGenericFunction>_loop_D_dlldlld
ufunc_rotate_types[0] = <char>np.NPY_DOUBLE
ufunc_rotate_types[1] = <char>np.NPY_LONG
ufunc_rotate_types[2] = <char>np.NPY_LONG
ufunc_rotate_types[3] = <char>np.NPY_DOUBLE
ufunc_rotate_types[4] = <char>np.NPY_LONG
ufunc_rotate_types[5] = <char>np.NPY_LONG
ufunc_rotate_types[6] = <char>np.NPY_DOUBLE
ufunc_rotate_types[7] = <char>np.NPY_CDOUBLE
ufunc_rotate_data[0] = <void*>_crotate

rotate = np.PyUFunc_FromFuncAndData(
    ufunc_rotate_loops,
    ufunc_rotate_data,
    ufunc_rotate_types,
    1,
    7,
    1,
    0,
    'rotate',
    """rotate(kz, mu, pol, qz, m, qol, phi)

Rotation coefficient for cylindrical modes.

Returns the correct rotation coefficient or a combination thereof for the specified
mode. A basis does not have to be specified since the coefficients are the same in
helicity and parity modes.

Args:
    kz (float, array_like): Z component of the destination mode
    mu (int, array_like): Order of the destination mode
    pol (int, array_like): Polarization of the destination mode
    qz (float, array_like): Z component of the source mode
    m (int, array_like): Order of the source mode
    qol (int, array_like): Polarization of the source mode
    phi (float, array_like): Rotation angle

Returns:
    complex
""",
    0,
)


cdef double complex _cperiodic_to_pw(double kx, number_t ky, double kzpw, long polpw, double kzcw, long m, long polcw, double a) nogil:
    cdef number_t krho = sqrt(kx * kx + ky * ky)
    cdef double complex ky_s = ky
    if (polcw == polpw == 0 or polcw == polpw == 1) and kzcw == kzpw:
        if ky_s == 0:
            ky_s = 1e-20 + 1e-20j
        elif cimag(ky_s) < 0 or (cimag(ky_s) == 0 and creal(ky_s) < 0):
            ky_s = -ky_s
        return 2 * cpow((-1j * kx + ky) / krho, m) / (abs(a) * ky_s)
    elif (polpw == 0 or polpw == 1) and (polcw == 0 or polcw == 1):
        return 0.0j
    raise ValueError("Polarization must be zero or one")


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
        ov0 = (<double complex(*)(double, double, double, long, double, long, long, double) nogil>func)(
            <double>(<double*>ip0)[0],
            <double>(<double*>ip1)[0],
            <double>(<double*>ip2)[0],
            <long>(<long*>ip3)[0],
            <double>(<double*>ip4)[0],
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
        ov0 = (<double complex(*)(double, double complex, double, long, double, long, long, double) nogil>func)(
            <double>(<double*>ip0)[0],
            <double complex>(<double complex*>ip1)[0],
            <double>(<double*>ip2)[0],
            <long>(<long*>ip3)[0],
            <double>(<double*>ip4)[0],
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
cdef void *ufunc_pw_data[2]
cdef char ufunc_pw_types[2 * 9]

ufunc_pw_loops[0] = <np.PyUFuncGenericFunction>_loop_periodic_d
ufunc_pw_loops[1] = <np.PyUFuncGenericFunction>_loop_periodic_D
ufunc_pw_types[0] = <char>np.NPY_DOUBLE
ufunc_pw_types[1] = <char>np.NPY_DOUBLE
ufunc_pw_types[2] = <char>np.NPY_DOUBLE
ufunc_pw_types[3] = <char>np.NPY_LONG
ufunc_pw_types[4] = <char>np.NPY_DOUBLE
ufunc_pw_types[5] = <char>np.NPY_LONG
ufunc_pw_types[6] = <char>np.NPY_LONG
ufunc_pw_types[7] = <char>np.NPY_DOUBLE
ufunc_pw_types[8] = <char>np.NPY_CDOUBLE
ufunc_pw_types[9] = <char>np.NPY_DOUBLE
ufunc_pw_types[10] = <char>np.NPY_CDOUBLE
ufunc_pw_types[11] = <char>np.NPY_DOUBLE
ufunc_pw_types[12] = <char>np.NPY_LONG
ufunc_pw_types[13] = <char>np.NPY_DOUBLE
ufunc_pw_types[14] = <char>np.NPY_LONG
ufunc_pw_types[15] = <char>np.NPY_LONG
ufunc_pw_types[16] = <char>np.NPY_DOUBLE
ufunc_pw_types[17] = <char>np.NPY_CDOUBLE
ufunc_pw_data[0] = <void*>_cperiodic_to_pw[double]
ufunc_pw_data[1] = <void*>_cperiodic_to_pw[double_complex]

periodic_to_pw = np.PyUFunc_FromFuncAndData(
    ufunc_pw_loops,
    ufunc_pw_data,
    ufunc_pw_types,
    2,
    8,
    1,
    0,
    'periodic_to_pw',
    """periodic_to_pw(kx, ky, kz, pol, qz, m, qol, area)

Convert periodic cylindrical wave to plane wave.

Returns the coefficient for the basis change in a periodic arrangement of cylindrical
modes to plane waves. For multiple positions only diagonal values (with respect to
the position) are returned.

The polarization values `0` and `1` refer to negative and positive helicity
waves or to TE and TM parity.

Args:
    kx (float, array_like): X component of destination mode wave vector
    ky (float, array_like): Y component of destination mode wave vector
    kz (float or complex, array_like): Z component of destination mode wave vector
    pol (int, array_like): Polarization of the destination mode
    qz (float, array_like): Z component of the source mode
    m (int, array_like): Order of the source mode
    qol (int, array_like): Polarization of the source mode
    area (float, array_like): Unit cell area

Returns:
    complex
""",
    0,
)


cdef double complex _cto_sw_h(long l, long m, long polpw, double kz, long mu, long polcw, double complex k) nogil:
    cdef double complex costheta
    if polpw == polcw == 0 or polpw == polcw == 1:
        if m == mu:
            costheta = kz / k
            return (
                cpow(1j, l - m)
                * sqrtd(4 * pi * (2 * l + 1) / <double>(l * (l + 1)))
                * expd(0.5 * (lgamma(l - m + 1) - lgamma(l + m + 1)))
                * (
                    cs.tau_fun(l, m, costheta)
                    + (2 * polpw - 1) * cs.pi_fun(l, m, costheta)
                )
            )
        return 0
    elif (polpw == 0 or polpw == 1) and (polcw == 0 or polcw == 1):
        return 0
    raise ValueError("Polarization must be zero or one")


cdef double complex _cto_sw_p(long l, long m, long polpw, double kz, long mu, long polcw, double complex k) nogil:
    if polpw == polcw == 0 or polpw == polcw == 1:
        if m == mu:
            return (
                cpow(1j, l - m)
                * sqrtd(4 * pi * (2 * l + 1) / <double>(l * (l + 1)))
                * expd(0.5 * (lgamma(l - m + 1) - lgamma(l + m + 1)))
                * cs.tau_fun(l, m, <double complex>(kz / k))
            )
        return 0
    elif (polpw == 1 and polcw == 0) or (polpw == 0 and polcw == 1):
        if m == mu:
            return (
                cpow(1j, l - m)
                * sqrtd(4 * pi * (2 * l + 1) / <double>(l * (l + 1)))
                * expd(0.5 * (lgamma(l - m + 1) - lgamma(l + m + 1)))
                * cs.pi_fun(l, m, <double complex>(kz / k))
            )
        return 0
    raise ValueError("Polarization must be zero or one")


cdef void _loop_sw_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
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
        ov0 = (<double complex(*)(long, long, long, double, long, long, double complex) nogil>func)(
            <long>(<long*>ip0)[0],
            <long>(<long*>ip1)[0],
            <long>(<long*>ip2)[0],
            <double>(<double*>ip3)[0],
            <long>(<long*>ip4)[0],
            <long>(<long*>ip5)[0],
            <double complex>(<double complex*>ip6)[0],
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


cdef np.PyUFuncGenericFunction ufunc_sw_loops[1]
cdef void *ufunc_sw_data_h[1]
cdef void *ufunc_sw_data_p[1]
cdef char ufunc_sw_types[8]

ufunc_sw_loops[0] = <np.PyUFuncGenericFunction>_loop_sw_D
ufunc_sw_types[0] = <char>np.NPY_LONG
ufunc_sw_types[1] = <char>np.NPY_LONG
ufunc_sw_types[2] = <char>np.NPY_LONG
ufunc_sw_types[3] = <char>np.NPY_DOUBLE
ufunc_sw_types[4] = <char>np.NPY_LONG
ufunc_sw_types[5] = <char>np.NPY_LONG
ufunc_sw_types[6] = <char>np.NPY_CDOUBLE
ufunc_sw_types[7] = <char>np.NPY_CDOUBLE
ufunc_sw_data_h[0] = <void*>_cto_sw_h
ufunc_sw_data_p[0] = <void*>_cto_sw_p

_to_sw_h = np.PyUFunc_FromFuncAndData(
    ufunc_sw_loops,
    ufunc_sw_data_h,
    ufunc_sw_types,
    1,
    7,
    1,
    0,
    '_to_sw_h',
    '',
    0,
)
_to_sw_p = np.PyUFunc_FromFuncAndData(
    ufunc_sw_loops,
    ufunc_sw_data_p,
    ufunc_sw_types,
    1,
    7,
    1,
    0,
    '_to_sw_p',
    '',
    0,
)


def to_sw(l, m, polsw, kz, mu, polcw, k, poltype=None, *args, **kwargs):
    """to_sw(l, m, polsw, kz, mu, polcw, k, helicity=True)

    Coefficient for the expansion of a cylindrical wave in spherical waves.

    Returns the coefficient for the basis change from a plane wave to a spherical wave.
    For multiple positions only diagonal values (with respect to the position) are
    returned.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        l (int, array_like): Degree of the spherical wave
        m (int, array_like): Order of the spherical wave
        polsw (int, array_like): Polarization of the destination mode
        kz (int, array_like): Z component of the cylindrical wave
        mu (int, array_like): Order of the cylindrical wave
        polcw (int, array_like): Polarization of the source mode
        k (float or complex, array_like): Wave number
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.

    Returns:
        complex
    """
    poltype = config.POLTYPE if poltype is None else poltype
    if poltype == "helicity":
        return _to_sw_h(l, m, polsw, kz, mu, polcw, k, *args, **kwargs)
    elif poltype == "parity":
        return _to_sw_p(l, m, polsw, kz, mu, polcw, k, *args, **kwargs)
    raise ValueError(f"invalid poltype '{poltype}'")


def translate_periodic(ks, kpar, a, rs, out, in_=None, rsin=None, eta=0):
    """translate_periodic(ks, kpar, a, rs, out, in_=None, rsin=None, eta=0)

    Translation coefficients in a lattice.

    Returns the translation coefficents for the given modes in a lattice. The
    calculation uses the fast converging sums of :mod:`treams.lattice`.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        ks (float or complex, scalar or (2,)-array): Wave number(s) in the medium, use
            two values in chiral media, indexed analogous to the polarization values
        kpar (float, (D,)-array): Parallel component of the wave, defines the dimension
            with `1 <= D <= 2`
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
    if in_ is None:
        in_ = out
    out = (*(np.array(o) for o in out),)
    in_ = (*(np.array(i) for i in in_),)
    if len(out) < 3 or len(out) > 4:
        raise ValueError(f"invalid length of output modes {len(out)}, must be 3 or 4")
    elif len(out) == 3:
        out = (np.zeros_like(out[0], int),) + out
    if len(in_) < 3 or len(in_) > 4:
        raise ValueError(f"invalid length of input modes {len(in_)}, must be 3 or 4")
    elif len(in_) == 3:
        in_ = (np.zeros_like(in_[0], int),) + in_
    if rsin is None:
        rsin = rs
    modes = -out[2][:, None] + in_[2]
    ks = np.array(ks)
    ks = ks.reshape((-1,))
    if ks.shape[0] == 2 and ks[0] == ks[1]:
        ks = ks[:1]
    if ks.shape[0] == 1 or ks[0] == ks[1]:
        krhos = np.sqrt((ks[0] * ks[0] - out[1][:, None] * in_[1]).astype(complex))  # todo: out not necessary this only simplifies krhos[compute] below
    else:
        krhos = np.sqrt((ks[in_[3]] * ks[in_[3]] - out[1][:, None] * in_[1]).astype(complex))
    krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
    compute = np.logical_and(
        np.equal(out[1][:, None], in_[1]),
        np.equal(out[3][:, None], in_[3]),
    )
    kpar = np.array(kpar)
    rs = np.array(rs)
    if rs.ndim == 1:
        rs = rs.reshape((1, -1))
    rsin = np.array(rsin)
    if rsin.ndim == 1:
        rsin = rsin.reshape((1, -1))
    rsdiff = -rs[out[0], None, :] + rsin[in_[0], :]
    if kpar.ndim == 0 or kpar.shape[-1] == 1:
        dlms = lattice.lsumcw1d_shift(modes[compute], krhos[compute], kpar, a, rsdiff[compute, :2], eta)
    else:
        dlms = lattice.lsumcw2d(modes[compute], krhos[compute], kpar, a, rsdiff[compute, :2], eta)
    res = np.zeros((out[0].shape[0], in_[0].shape[0]), complex)
    res[compute] = dlms
    res = res * np.exp(-1j * in_[1] * rsdiff[:, :, 2])
    return res
