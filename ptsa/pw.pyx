"""
=====================================
Functions associated with plane waves
=====================================

.. autosummary::
   :toctree: generated/

   flip_y_to_z
   to_cw
   to_sw
   translate

"""

import numpy as np

cimport numpy as np
from libc.math cimport atan2
from libc.math cimport exp as expd
from libc.math cimport lgamma, pi
from libc.math cimport sqrt as sqrtd

cimport ptsa.special.cython_special as cs
from ptsa.special._misc cimport acos, double_complex, sqrt


cdef extern from "<complex.h>" nogil:
    double complex cexp(double complex z)
    double complex cpow(double complex x, double complex y)
    double creal(double complex z)


ctypedef fused number_t:
    double complex
    double


np.import_array()
np.import_ufunc()


# cdef double complex  _ctranslate(
#         double complex kx, double complex ky, double complex kz, long pol,
#         double complex qx, double complex qy, double complex qz, long qol,
#         double x, double y, double z) nogil:
#     if kx == qx and ky == qy and kz == qz and pol == qol:
#         return cexp(1j * (kx * x + ky * y + kz * z))
#     else:
#         return 0
#
#
# cdef void _loop_translate_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
#     cdef np.npy_intp i, n = dims[0]
#     cdef void *func = <void*>data
#     cdef char *ip0 = args[0]
#     cdef char *ip1 = args[1]
#     cdef char *ip2 = args[2]
#     cdef char *ip3 = args[3]
#     cdef char *ip4 = args[4]
#     cdef char *ip5 = args[5]
#     cdef char *ip6 = args[6]
#     cdef char *ip7 = args[7]
#     cdef char *ip8 = args[8]
#     cdef char *ip9 = args[9]
#     cdef char *ip10 = args[10]
#     cdef char *ip11 = args[11]
#     cdef char *ip12 = args[12]
#     cdef char *op0 = args[13]
#     cdef double complex ov0
#     for i in range(n):
#         if <long>(<long*>ip11)[0] == <long>(<long*>ip12)[0]:
#             ov0 = (<double complex(*)(double complex, double complex, double complex, long, double complex, double complex, double complex, long, double, double, double) nogil>func)(
#                 <double complex>(<double complex*>ip0)[0],
#                 <double complex>(<double complex*>ip1)[0],
#                 <double complex>(<double complex*>ip2)[0],
#                 <long>(<long*>ip3)[0],
#                 <double complex>(<double complex*>ip4)[0],
#                 <double complex>(<double complex*>ip5)[0],
#                 <double complex>(<double complex*>ip6)[0],
#                 <long>(<long*>ip7)[0],
#                 <double>(<double*>ip8)[0],
#                 <double>(<double*>ip9)[0],
#                 <double>(<double*>ip10)[0],
#             )
#         else:
#             ov0 = 0
#         (<double complex*>op0)[0] = <double complex>ov0
#         ip0 += steps[0]
#         ip1 += steps[1]
#         ip2 += steps[2]
#         ip3 += steps[3]
#         ip4 += steps[4]
#         ip5 += steps[5]
#         ip6 += steps[6]
#         ip7 += steps[7]
#         ip8 += steps[8]
#         ip9 += steps[9]
#         ip10 += steps[10]
#         ip11 += steps[11]
#         ip12 += steps[12]
#         op0 += steps[13]
#
# cdef np.PyUFuncGenericFunction ufunc_translate_loops[1]
# cdef void *ufunc_translate_data[1]
# cdef char ufunc_translate_types[14]
#
# ufunc_translate_loops[0] = <np.PyUFuncGenericFunction>_loop_translate_D
# ufunc_translate_types[0] = <char>np.NPY_CDOUBLE
# ufunc_translate_types[1] = <char>np.NPY_CDOUBLE
# ufunc_translate_types[2] = <char>np.NPY_CDOUBLE
# ufunc_translate_types[3] = <char>np.NPY_LONG
# ufunc_translate_types[4] = <char>np.NPY_CDOUBLE
# ufunc_translate_types[5] = <char>np.NPY_CDOUBLE
# ufunc_translate_types[6] = <char>np.NPY_CDOUBLE
# ufunc_translate_types[7] = <char>np.NPY_LONG
# ufunc_translate_types[8] = <char>np.NPY_DOUBLE
# ufunc_translate_types[9] = <char>np.NPY_DOUBLE
# ufunc_translate_types[10] = <char>np.NPY_DOUBLE
# ufunc_translate_types[11] = <char>np.NPY_LONG
# ufunc_translate_types[12] = <char>np.NPY_LONG
# ufunc_translate_types[13] = <char>np.NPY_CDOUBLE
# ufunc_translate_data[0] = <void*>_ctranslate
#
# _translate = np.PyUFunc_FromFuncAndData(
#     ufunc_translate_loops,
#     ufunc_translate_data,
#     ufunc_translate_types,
#     1,
#     13,
#     1,
#     0,
#     '_translate',
#     '',
#     0,
# )
#
# def translate(kx, ky, kz, pol, qx, qy, qz, qol, x, y, z, posin=0, posout=0):
#     return _translate(kx, ky, kz, pol, qx, qy, qz, qol, x, y, z, posin, posout)


def translate(kx, ky, kz, x, y, z):
  """
  translate(kx, ky, kz, x, y, z)

  Translation coefficient for plane wave modes

  The translation coefficient is the phase factor
  :math:`\mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}`.

  Args:
      kx, ky, kz (float or complex, array_like): Wave vector components
      x, y, z (float, array_like): Translation vector components

  Returns:
      complex
  """
  return np.exp(1j * (kx * x + ky * y + kz * z))


cdef double complex _cto_sw_h(long l, long m, long polvsw, double kx, double ky, number_t kz, long polpw) nogil:
    if polvsw != polpw:
        return 0.0j
    cdef double phi = atan2(ky, kx)
    cdef number_t k = sqrt(kx * kx + ky * ky + kz * kz)
    cdef double complex costheta = kz / k
    return (
        2 * sqrtd(pi * (2 * l + 1) / <double>(l * (l + 1)))
        * expd(0.5 * (lgamma(l - m + 1) - lgamma(l + m + 1)))
        * cpow(1j, l)
        * cexp(-1j * m * phi)
    ) * (cs.tau_fun(l, m, costheta) + (2 * polpw - 1) * cs.pi_fun(l, m, costheta))


cdef double complex _cto_sw_p(long l, long m, long polvsw, double kx, double ky, number_t kz, long polpw) nogil:
    cdef double phi = atan2(ky, kx)
    cdef number_t k = sqrt(kx * kx + ky * ky + kz * kz)
    cdef double complex costheta = kz / k
    cdef double complex pref = (
        2 * sqrtd(pi * (2 * l + 1) / <double>(l * (l + 1)))
        * expd(0.5 * (lgamma(l - m + 1) - lgamma(l + m + 1)))
        * cpow(1j, l)
        * cexp(-1j * m * phi)
    )
    if polvsw == polpw:
        return pref * cs.tau_fun(l, m, costheta)
    return pref * cs.pi_fun(l, m, costheta)


cdef void _loop_sw_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        if <long>(<long*>ip7)[0] == <long>(<long*>ip8)[0]:
            ov0 = (<double complex(*)(long, long, long, double, double, double, long) nogil>func)(
                <long>(<long*>ip0)[0],
                <long>(<long*>ip1)[0],
                <long>(<long*>ip2)[0],
                <double>(<double*>ip3)[0],
                <double>(<double*>ip4)[0],
                <double>(<double*>ip5)[0],
                <long>(<long*>ip6)[0],
            )
        else:
            ov0 = 0
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
    cdef char *ip7 = args[7]
    cdef char *ip8 = args[8]
    cdef char *op0 = args[9]
    cdef double complex ov0
    for i in range(n):
        if <long>(<long*>ip7)[0] == <long>(<long*>ip8)[0]:
            ov0 = (<double complex(*)(long, long, long, double, double, double complex, long) nogil>func)(
                <long>(<long*>ip0)[0],
                <long>(<long*>ip1)[0],
                <long>(<long*>ip2)[0],
                <double>(<double*>ip3)[0],
                <double>(<double*>ip4)[0],
                <double complex>(<double complex*>ip5)[0],
                <long>(<long*>ip6)[0],
            )
        else:
            ov0 = 0
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


cdef np.PyUFuncGenericFunction ufunc_sw_loops[2]
cdef void *ufunc_sw_h_data[2]
cdef void *ufunc_sw_p_data[2]
cdef char ufunc_sw_types[2 * 10]

ufunc_sw_loops[0] = <np.PyUFuncGenericFunction>_loop_sw_d
ufunc_sw_loops[1] = <np.PyUFuncGenericFunction>_loop_sw_D
ufunc_sw_types[0] = <char>np.NPY_LONG
ufunc_sw_types[1] = <char>np.NPY_LONG
ufunc_sw_types[2] = <char>np.NPY_LONG
ufunc_sw_types[3] = <char>np.NPY_DOUBLE
ufunc_sw_types[4] = <char>np.NPY_DOUBLE
ufunc_sw_types[5] = <char>np.NPY_DOUBLE
ufunc_sw_types[6] = <char>np.NPY_LONG
ufunc_sw_types[7] = <char>np.NPY_LONG
ufunc_sw_types[8] = <char>np.NPY_LONG
ufunc_sw_types[9] = <char>np.NPY_CDOUBLE
ufunc_sw_types[10] = <char>np.NPY_LONG
ufunc_sw_types[11] = <char>np.NPY_LONG
ufunc_sw_types[12] = <char>np.NPY_LONG
ufunc_sw_types[13] = <char>np.NPY_DOUBLE
ufunc_sw_types[14] = <char>np.NPY_DOUBLE
ufunc_sw_types[15] = <char>np.NPY_CDOUBLE
ufunc_sw_types[16] = <char>np.NPY_LONG
ufunc_sw_types[17] = <char>np.NPY_LONG
ufunc_sw_types[18] = <char>np.NPY_LONG
ufunc_sw_types[19] = <char>np.NPY_CDOUBLE
ufunc_sw_h_data[0] = <void*>_cto_sw_h[double]
ufunc_sw_h_data[1] = <void*>_cto_sw_h[double_complex]
ufunc_sw_p_data[0] = <void*>_cto_sw_p[double]
ufunc_sw_p_data[1] = <void*>_cto_sw_p[double_complex]

_to_sw_h = np.PyUFunc_FromFuncAndData(
    ufunc_sw_loops,
    ufunc_sw_h_data,
    ufunc_sw_types,
    2,
    9,
    1,
    0,
    '_to_sw_h',
    '',
    0,
)
_to_sw_p = np.PyUFunc_FromFuncAndData(
    ufunc_sw_loops,
    ufunc_sw_p_data,
    ufunc_sw_types,
    2,
    9,
    1,
    0,
    '_to_sw_p',
    '',
    0,
)


def to_sw(l, m, polsw, kx, ky, kz, polpw, posout=0, posin=0, helicity=True):
    """
    to_sw(l, m, polsw, kx, ky, kz, polpw, posout=0, posin=0, helicity=True)

    Coefficient for the expansion of a plane wave in spherical waves

    Returns the coefficient for the basis change from a plane wave to a spherical wave.
    For multiple positions only diagonal values (with respect to the position) are
    returned.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        l (int, array_like): Degree of the spherical wave
        m (int, array_like): Order of the spherical wave
        polsw (int, array_like): Polarization of the destination mode
        kx (float, array_like): X component of plane wave's wave vector
        ky (float, array_like): Y component of plane wave's wave vector
        kz (float, array_like): Z component of plane wave's wave vector
        polpw (int, array_like): Polarization of the plane wave
        posout (int, optional): Output positions
        posin (int, optional): Input positions
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.

    Returns:
        complex
    """
    if helicity:
        return _to_sw_h(l, m, polsw, kx, ky, kz, polpw, posout, posin)
    return _to_sw_p(l, m, polsw, kx, ky, kz, polpw, posout, posin)


cdef double complex _cto_cw(double kzcw, long m, long polcw, double kx, number_t ky, double kzpw, long polpw) nogil:
    cdef number_t krho = sqrt(kx * kx + ky * ky)
    if (polcw == polpw == 0 or polcw == polpw == 1) and kzcw == kzpw:
        if m == 0:
            return 1
        return cpow((1j * kx + ky) / krho, m)
    elif (polpw == 1 and polcw == 0) or (polpw == 0 and polcw == 1) or kzcw != kzpw:
        return 0.0j
    raise ValueError("Polarization must be zero or one")


cdef void _loop_cw_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        if <long>(<long*>ip7)[0] == <long>(<long*>ip8)[0]:
            ov0 = (<double complex(*)(double, long, long, double, double, double, long) nogil>func)(
                <double>(<double*>ip0)[0],
                <long>(<long*>ip1)[0],
                <long>(<long*>ip2)[0],
                <double>(<double*>ip3)[0],
                <double>(<double*>ip4)[0],
                <double>(<double*>ip5)[0],
                <long>(<long*>ip6)[0],
            )
        else:
            ov0 = 0
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


cdef void _loop_cw_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        if <long>(<long*>ip7)[0] == <long>(<long*>ip8)[0]:
            ov0 = (<double complex(*)(double, long, long, double, double complex, double, long) nogil>func)(
                <double>(<double*>ip0)[0],
                <long>(<long*>ip1)[0],
                <long>(<long*>ip2)[0],
                <double>(<double*>ip3)[0],
                <double complex>(<double complex*>ip4)[0],
                <double>(<double*>ip5)[0],
                <long>(<long*>ip6)[0],
            )
        else:
            ov0 = 0
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


cdef np.PyUFuncGenericFunction ufunc_cw_loops[2]
cdef void *ufunc_cw_data[2]
cdef char ufunc_cw_types[2 * 10]

ufunc_cw_loops[0] = <np.PyUFuncGenericFunction>_loop_cw_d
ufunc_cw_loops[1] = <np.PyUFuncGenericFunction>_loop_cw_D
ufunc_cw_types[0] = <char>np.NPY_DOUBLE
ufunc_cw_types[1] = <char>np.NPY_LONG
ufunc_cw_types[2] = <char>np.NPY_LONG
ufunc_cw_types[3] = <char>np.NPY_DOUBLE
ufunc_cw_types[4] = <char>np.NPY_DOUBLE
ufunc_cw_types[5] = <char>np.NPY_DOUBLE
ufunc_cw_types[6] = <char>np.NPY_LONG
ufunc_cw_types[7] = <char>np.NPY_LONG
ufunc_cw_types[8] = <char>np.NPY_LONG
ufunc_cw_types[9] = <char>np.NPY_CDOUBLE
ufunc_cw_types[10] = <char>np.NPY_DOUBLE
ufunc_cw_types[11] = <char>np.NPY_LONG
ufunc_cw_types[12] = <char>np.NPY_LONG
ufunc_cw_types[13] = <char>np.NPY_DOUBLE
ufunc_cw_types[14] = <char>np.NPY_CDOUBLE
ufunc_cw_types[15] = <char>np.NPY_DOUBLE
ufunc_cw_types[16] = <char>np.NPY_LONG
ufunc_cw_types[17] = <char>np.NPY_LONG
ufunc_cw_types[18] = <char>np.NPY_LONG
ufunc_cw_types[19] = <char>np.NPY_CDOUBLE
ufunc_cw_data[0] = <void*>_cto_cw[double]
ufunc_cw_data[1] = <void*>_cto_cw[double_complex]

_to_cw = np.PyUFunc_FromFuncAndData(
    ufunc_cw_loops,
    ufunc_cw_data,
    ufunc_cw_types,
    2,
    9,
    1,
    0,
    '_to_cw',
    '',
    0,
)


def to_cw(kzcw, m, polcw, kx, ky, kzpw, polpw, posout=0, posin=0):
    """
    to_cw(kzcw, m, polcw, kx, ky, kzpw, polpw, posout=0, posin=0)

    Coefficient for the expansion of a plane wave in cylindricrical waves

    Returns the coefficient for the basis change from a plane wave to a cylindrical wave.
    For multiple positions only diagonal values (with respect to the position) are
    returned.

    The polarization values `0` and `1` refer to negative and positive helicity
    waves or, if ``helicity == False``, to TE and TM parity.

    Args:
        kzcw (float, array_like): Z component of cylindrical wave
        m (int, array_like): Order of the cylindrical wave
        polsw (int, array_like): Polarization of the destination mode
        kx (float, array_like): X component of plane wave's wave vector
        ky (float, array_like): Y component of plane wave's wave vector
        kzpw (float, array_like): Z component of plane wave's wave vector
        polpw (int, array_like): Polarization of the plane wave
        posout (int, optional): Output positions
        posin (int, optional): Input positions
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.

    Returns:
        complex
    """
    return _to_cw(kzcw, m, polcw, kx, ky, kzpw, polpw, posout, posin)


cdef double complex _cxyz_to_zxy_p(number_t kx, number_t ky, number_t kz, long polout, long polin) nogil:
    # We describe the system as if it was aligned along x, periodic along y and
    # principle direction of propagation was along z. But actually, the system is
    # in a system where all these arguments are permuted, such that alignment is along
    # z.
    cdef number_t kyz = sqrt(ky * ky + kz * kz)
    if kyz == 0:
        # In this case we have a propagation purely along the cylinder axis, which is
        # not possible to describe by the cylindrical waves we use.
        raise ValueError("kx**2 must not be equal to k**2")
    cdef number_t kxy = sqrt(kx * kx + ky * ky)
    if kxy == 0:
        if polout == polin:
            return -1 if creal(kz) > 0 else 1
        return 0
    if polout == polin:
        return -kx * kz / (kxy * kyz)
    cdef number_t k = sqrt(kx * kx + ky * ky + kz * kz)
    return 1j * k * ky / (kxy * kyz)


cdef double complex _cxyz_to_zxy_h(number_t kx, number_t ky, number_t kz, long polout, long polin) nogil:
    """
    Change the coordinate system and basis vectors of a plane wave

    The Q-matrices are expected to extend periodically in the x y plane, but the
    two-dimensional T-matrices have their cylinder axis along the z direction and the
    periodicity along x. To match the two computation techniques we need to permute the
    labels of the coordinates. Also, the polarization basis vectors have to be adjusted
    accordingly.
    """
    cdef number_t kyz = sqrt(ky * ky + kz * kz)
    if kyz == 0:
        # In this case we have a propagation purely along the cylinder axis, which is
        # not possible to describe by the cylindrical waves we use.
        raise ValueError("kx**2 must not be equal to k**2")
    if polout != polin:
        return 0.0j
    cdef number_t kxy = sqrt(kx * kx + ky * ky)
    if kxy == 0:
        return -1 if creal(kz) > 0 else 1
    cdef number_t k = sqrt(kx * kx + ky * ky + kz * kz)
    return (-kx * kz + (2 * polin - 1) * 1j * k * ky) / (kxy * kyz)


cdef double complex _cxyz_to_yzx_p(number_t kx, number_t ky, number_t kz, long polout, long polin) nogil:
    if polout == polin:
        return -_cxyz_to_zxy_p(kx, ky, kz, polout, polin)
    return _cxyz_to_zxy_p(kx, ky, kz, polout, polin)


cdef double complex _cxyz_to_yzx_h(number_t kx, number_t ky, number_t kz, long polout, long polin) nogil:
    return _cxyz_to_zxy_h(kx, ky, kz, 1 - polout, 1 - polin)


cdef void _loop_yz_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
    cdef char *ip9 = args[9]
    cdef char *op0 = args[10]
    cdef double complex ov0
    for i in range(n):
        if (
            <double>(<double*>ip0)[0] == <double>(<double*>ip4)[0]
            and <double>(<double*>ip1)[0] == <double>(<double*>ip5)[0]
            and <double>(<double*>ip2)[0] == <double>(<double*>ip6)[0]
            and <long>(<long*>ip8)[0] == <long>(<long*>ip9)[0]
        ):
            ov0 = (<double complex(*)(double, double, double, long, long) nogil>func)(
                <double>(<double*>ip0)[0],
                <double>(<double*>ip1)[0],
                <double>(<double*>ip2)[0],
                <long>(<long*>ip3)[0],
                <long>(<long*>ip7)[0],
            )
        else:
            ov0 = 0
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
        ip9 += steps[9]
        op0 += steps[10]


cdef void _loop_yz_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
    cdef char *ip9 = args[9]
    cdef char *op0 = args[10]
    cdef double complex ov0
    for i in range(n):
        if (
            <double complex>(<double complex*>ip0)[0] == <double complex>(<double complex*>ip4)[0]
            and <double complex>(<double complex*>ip1)[0] == <double complex>(<double complex*>ip5)[0]
            and <double complex>(<double complex*>ip2)[0] == <double complex>(<double complex*>ip6)[0]
            and <long>(<long*>ip8)[0] == <long>(<long*>ip9)[0]
        ):
            ov0 = (<double complex(*)(double complex, double complex, double complex, long, long) nogil>func)(
                <double complex>(<double complex*>ip0)[0],
                <double complex>(<double complex*>ip1)[0],
                <double complex>(<double complex*>ip2)[0],
                <long>(<long*>ip3)[0],
                <long>(<long*>ip7)[0],
            )
        else:
            ov0 = 0
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
        ip9 += steps[9]
        op0 += steps[10]


cdef np.PyUFuncGenericFunction ufunc_yz_loops[2]
cdef void *ufunc_yz_h_data[2]
cdef void *ufunc_yz_p_data[2]
cdef void *ufunc_zy_h_data[2]
cdef void *ufunc_zy_p_data[2]
cdef char ufunc_yz_types[2 * 11]

ufunc_yz_loops[0] = <np.PyUFuncGenericFunction>_loop_yz_d
ufunc_yz_loops[1] = <np.PyUFuncGenericFunction>_loop_yz_D
ufunc_yz_types[0] = <char>np.NPY_DOUBLE
ufunc_yz_types[1] = <char>np.NPY_DOUBLE
ufunc_yz_types[2] = <char>np.NPY_DOUBLE
ufunc_yz_types[3] = <char>np.NPY_LONG
ufunc_yz_types[4] = <char>np.NPY_DOUBLE
ufunc_yz_types[5] = <char>np.NPY_DOUBLE
ufunc_yz_types[6] = <char>np.NPY_DOUBLE
ufunc_yz_types[7] = <char>np.NPY_LONG
ufunc_yz_types[8] = <char>np.NPY_LONG
ufunc_yz_types[9] = <char>np.NPY_LONG
ufunc_yz_types[10] = <char>np.NPY_CDOUBLE
ufunc_yz_types[11] = <char>np.NPY_CDOUBLE
ufunc_yz_types[12] = <char>np.NPY_CDOUBLE
ufunc_yz_types[13] = <char>np.NPY_CDOUBLE
ufunc_yz_types[14] = <char>np.NPY_LONG
ufunc_yz_types[15] = <char>np.NPY_CDOUBLE
ufunc_yz_types[16] = <char>np.NPY_CDOUBLE
ufunc_yz_types[17] = <char>np.NPY_CDOUBLE
ufunc_yz_types[18] = <char>np.NPY_LONG
ufunc_yz_types[19] = <char>np.NPY_LONG
ufunc_yz_types[20] = <char>np.NPY_LONG
ufunc_yz_types[21] = <char>np.NPY_CDOUBLE
ufunc_yz_h_data[0] = <void*>_cxyz_to_zxy_h[double]
ufunc_yz_h_data[1] = <void*>_cxyz_to_zxy_h[double_complex]
ufunc_yz_p_data[0] = <void*>_cxyz_to_zxy_p[double]
ufunc_yz_p_data[1] = <void*>_cxyz_to_zxy_p[double_complex]
ufunc_zy_h_data[0] = <void*>_cxyz_to_yzx_h[double]
ufunc_zy_h_data[1] = <void*>_cxyz_to_yzx_h[double_complex]
ufunc_zy_p_data[0] = <void*>_cxyz_to_yzx_p[double]
ufunc_zy_p_data[1] = <void*>_cxyz_to_yzx_p[double_complex]

_xyz_to_zxy_h = np.PyUFunc_FromFuncAndData(
    ufunc_yz_loops,
    ufunc_yz_h_data,
    ufunc_yz_types,
    2,
    10,
    1,
    0,
    '_xyz_to_zxy_h',
    '',
    0,
)
_xyz_to_zxy_p = np.PyUFunc_FromFuncAndData(
    ufunc_yz_loops,
    ufunc_yz_p_data,
    ufunc_yz_types,
    2,
    10,
    1,
    0,
    '_xyz_to_zxy_p',
    '',
    0,
)
_xyz_to_yzx_h = np.PyUFunc_FromFuncAndData(
    ufunc_yz_loops,
    ufunc_zy_h_data,
    ufunc_yz_types,
    2,
    10,
    1,
    0,
    '_xyz_to_yzx_h',
    '',
    0,
)
_xyz_to_yzx_p = np.PyUFunc_FromFuncAndData(
    ufunc_yz_loops,
    ufunc_zy_p_data,
    ufunc_yz_types,
    2,
    10,
    1,
    0,
    '_xyz_to_yzx_p',
    '',
    0,
)


def xyz_to_zxy(kx, ky, kz, pol, qx, qy, qz, qol, posout=0, posin=0, helicity=True, inverse=False):
    """
    xyz_to_zxy(kx, ky, kz, pol, qx, qy, qz, qol, posout=0, posin=0, helicity=True, inverse=False)

    Change the coordinate system of the plane wave

    A plane wave in the coordinate system :math:`(x, y, z)` with primary direction of
    propagation along the z-axis is described in the system
    :math:`(x, y, z) = (z', x', y')`, where still the modes are described with
    :func:`ptsa.special.vpw_M`, :func:`ptsa.special.vpw_N`, and
    :func:`ptsa.special.vpw_A`. The inverse transformation is also possible.

    The function is essentially diagonal in the wave number.

    Args:
        kx (float, array_like): X component of destination mode wave vector
        ky (float, array_like): Y component of destination mode wave vector
        kz (float or complex, array_like): Z component of destination mode wave vector
        pol (int, array_like): Polarization of the destination mode
        kx (float, array_like): X component of source mode wave vector
        ky (float, array_like): Y component of source mode wave vector
        kz (float, array_like): Z component of source mode wave vector
        pol (int, array_like): Polarization of the source mode
        posout (int, optional): Output positions
        posin (int, optional): Input positions
        helicity (bool, optional): If true, helicity basis is assumed, else parity basis.
            Defaults to ``True``.
        inverse (bool, optional): Use the inverse transformation.

    Returns:
        complex
    """
    if helicity:
        if inverse:
            return _xyz_to_yzx_h(kx, ky, kz, pol, qx, qy, qz, qol, posout, posin)
        return _xyz_to_zxy_h(kx, ky, kz, pol, qx, qy, qz, qol, posout, posin)
    if inverse:
        return _xyz_to_yzx_p(kx, ky, kz, pol, qx, qy, qz, qol, posout, posin)
    return _xyz_to_zxy_p(kx, ky, kz, pol, qx, qy, qz, qol, posout, posin)
