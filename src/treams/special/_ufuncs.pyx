"""Universal functions for the special subpackage"""

cimport numpy as np
cimport scipy.special.cython_special as cs

from treams.special cimport _bessel, _integrals, _waves, _wigner3j, _wignerd
from treams.special._misc cimport double_complex

__all__ = [
    "hankel1_d",
    "hankel2_d",
    "incgamma",
    "intkambe",
    "jv_d",
    "lpmv",
    "pi_fun",
    "sph_harm",
    "spherical_hankel1",
    "spherical_hankel1_d",
    "spherical_hankel2",
    "spherical_hankel2_d",
    "tau_fun",
    "tl_vsw_A",
    "tl_vsw_B",
    "tl_vsw_rA",
    "tl_vsw_rB",
    "tl_vcw",
    "tl_vcw_r",
    "wignerd",
    "wignersmalld",
    "wigner3j",
    "yv_d",
    "_tl_vsw_helper",
]


cdef void loop_d_dd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *op0 = args[2]
    cdef double ov0
    for i in range(n):
        ov0 = (<double(*)(double, double) nogil>func)(<double>(<double*>ip0)[0], <double>(<double*>ip1)[0])
        (<double*>op0)[0] = <double>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        op0 += steps[2]


cdef void loop_D_dD(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *op0 = args[2]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(double, double complex) nogil>func)(<double>(<double*>ip0)[0], <double complex>(<double complex*>ip1)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        op0 += steps[2]


cdef void loop_d_ddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *op0 = args[3]
    cdef double ov0
    for i in range(n):
        ov0 = (<double(*)(double, double, double) nogil>func)(<double>(<double*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0])
        (<double*>op0)[0] = <double>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        op0 += steps[3]


cdef void loop_d_ldd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *op0 = args[3]
    cdef double ov0
    for i in range(n):
        ov0 = (<double(*)(long, double, double) nogil>func)(<long>(<long*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0])
        (<double*>op0)[0] = <double>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        op0 += steps[3]


cdef void loop_D_lDD(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *op0 = args[3]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(long, double complex, double complex) nogil>func)(<long>(<long*>ip0)[0], <double complex>(<double complex*>ip1)[0], <double complex>(<double complex*>ip2)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        op0 += steps[3]


cdef void loop_D_ddD(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *op0 = args[3]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(double, double, double complex) nogil>func)(<double>(<double*>ip0)[0], <double>(<double*>ip1)[0], <double complex>(<double complex*>ip2)[0])
        (<double complex *>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        op0 += steps[3]


cdef void loop_d_llld(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *op0 = args[4]
    cdef double ov0
    for i in range(n):
        ov0 = (<double(*)(long, long, long, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <long>(<long*>ip2)[0], <double>(<double*>ip3)[0])
        (<double*>op0)[0] = <double>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        op0 += steps[4]


cdef void loop_D_lllD(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *op0 = args[4]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(long, long, long, double complex) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <long>(<long*>ip2)[0], <double complex>(<double complex*>ip3)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        op0 += steps[4]


cdef void loop_D_dddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *op0 = args[4]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(double, double, double, double) nogil>func)(<double>(<double*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        op0 += steps[4]


cdef void loop_D_dddD(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *op0 = args[4]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(double, double, double, double complex) nogil>func)(<double>(<double*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0], <double complex>(<double complex*>ip3)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        op0 += steps[4]


cdef void loop_d_llllll(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double ov0
    for i in range(n):
        ov0 = (<double(*)(long, long, long, long, long, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <long>(<long*>ip2)[0], <long>(<long*>ip3)[0], <long>(<long*>ip4)[0], <long>(<long*>ip5)[0])
        (<double*>op0)[0] = <double>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_llllll(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(long, long, long, long, long, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <long>(<long*>ip2)[0], <long>(<long*>ip3)[0], <long>(<long*>ip4)[0], <long>(<long*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_lllddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(long, long, long, double, double, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <long>(<long*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double>(<double*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_llldDd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(long, long, long, double, double complex, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <long>(<long*>ip2)[0], <double>(<double*>ip3)[0], <double complex>(<double complex*>ip4)[0], <double>(<double*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_llllddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        ov0 = (<double complex(*)(long, long, long, long, double, double, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <long>(<long*>ip2)[0], <long>(<long*>ip3)[0], <double>(<double*>ip4)[0], <double>(<double*>ip5)[0], <double>(<double*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_D_llllDdd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        ov0 = (<double complex(*)(long, long, long, long, double complex, double, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <long>(<long*>ip2)[0], <long>(<long*>ip3)[0], <double complex>(<double complex*>ip4)[0], <double>(<double*>ip5)[0], <double>(<double*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_D_llllDDd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        ov0 = (<double complex(*)(long, long, long, long, double complex, double complex, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <long>(<long*>ip2)[0], <long>(<long*>ip3)[0], <double complex>(<double complex*>ip4)[0], <double complex>(<double complex*>ip5)[0], <double>(<double*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_D_dldlddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        ov0 = (<double complex(*)(double, long, double, long, double, double, double) nogil>func)(<double>(<double*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <long>(<long*>ip3)[0], <double>(<double*>ip4)[0], <double>(<double*>ip5)[0], <double>(<double*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_D_dldlDdd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
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
        ov0 = (<double complex(*)(double, long, double, long, double complex, double, double) nogil>func)(<double>(<double*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <long>(<long*>ip3)[0], <double complex>(<double complex*>ip4)[0], <double>(<double*>ip5)[0], <double>(<double*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


np.import_array()
np.import_ufunc()


cdef np.PyUFuncGenericFunction ufunc_lpmv_loops[2]
cdef void *ufunc_lpmv_data[2]
cdef char ufunc_lpmv_types[2 * 4]

ufunc_lpmv_loops[0] = <np.PyUFuncGenericFunction>loop_d_ddd
ufunc_lpmv_loops[1] = <np.PyUFuncGenericFunction>loop_D_ddD
ufunc_lpmv_types[0] = <char>np.NPY_DOUBLE
ufunc_lpmv_types[1] = <char>np.NPY_DOUBLE
ufunc_lpmv_types[2] = <char>np.NPY_DOUBLE
ufunc_lpmv_types[3] = <char>np.NPY_DOUBLE
ufunc_lpmv_types[4] = <char>np.NPY_DOUBLE
ufunc_lpmv_types[5] = <char>np.NPY_DOUBLE
ufunc_lpmv_types[6] = <char>np.NPY_CDOUBLE
ufunc_lpmv_types[7] = <char>np.NPY_CDOUBLE
ufunc_lpmv_data[0] = <void*>_waves.lpmv[double]
ufunc_lpmv_data[1] = <void*>_waves.lpmv[double_complex]

lpmv = np.PyUFunc_FromFuncAndData(
    ufunc_lpmv_loops,
    ufunc_lpmv_data,
    ufunc_lpmv_types,
    2,  # number of supported input types
    3,  # number of input args
    1,  # number of output args
    0,  # identity element
    "lpmv",  # function name
    r"""lpmv(m, v, z)

Associated legendre polynomials of real and complex argument

For complex arguments the branch cut is at `|x| > 1`. Additionally, only integer
orders and degrees are possible when using a complex argument. The computation is
using a recursion relation in this case. For a real argument
:py:data:`scipy.special.lpmv` is called.

The function is defined as

.. math::

    P_v^m (z) = \frac{(-1)^m}{2^\nu \nu!} (1 - z^2)^\frac{m}{2}
    \frac{\mathrm d^{\nu + m}}{\mathrm d z^{\nu + m}} (z^2 - 1)^\nu

for integer degree and order.

Warning:
    Please note the order of the arguments. It is kept this way to stay in line with
    Scipy. Most of the other functions, with :data:`sph_harm` as a notable exception
    for the same reason, use degree and order swapped around.

Args:
    m (float, array_like): Order
    v (float, array_like): Degree
    z (float or complex, array_like): Argument

Returns:
    float or complex

References:
    - `Wikipedia: Associated Legendre polynomials <https://en.wikipedia.org/wiki/Associated_Legendre_polynomials>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_sph_harm_loops[2]
cdef void *ufunc_sph_harm_data[2]
cdef char ufunc_sph_harm_types[2 * 5]

ufunc_sph_harm_loops[0] = <np.PyUFuncGenericFunction>loop_D_dddd
ufunc_sph_harm_loops[1] = <np.PyUFuncGenericFunction>loop_D_dddD
ufunc_sph_harm_types[0] = <char>np.NPY_DOUBLE
ufunc_sph_harm_types[1] = <char>np.NPY_DOUBLE
ufunc_sph_harm_types[2] = <char>np.NPY_DOUBLE
ufunc_sph_harm_types[3] = <char>np.NPY_DOUBLE
ufunc_sph_harm_types[4] = <char>np.NPY_CDOUBLE
ufunc_sph_harm_types[5] = <char>np.NPY_DOUBLE
ufunc_sph_harm_types[6] = <char>np.NPY_DOUBLE
ufunc_sph_harm_types[7] = <char>np.NPY_DOUBLE
ufunc_sph_harm_types[8] = <char>np.NPY_CDOUBLE
ufunc_sph_harm_types[9] = <char>np.NPY_CDOUBLE
ufunc_sph_harm_data[0] = <void*>cs.sph_harm[double]
ufunc_sph_harm_data[1] = <void*>_waves.csph_harm

sph_harm = np.PyUFunc_FromFuncAndData(
    ufunc_sph_harm_loops,
    ufunc_sph_harm_data,
    ufunc_sph_harm_types,
    2,  # number of supported input types
    4,  # number of input args
    1,  # number of output args
    0,  # identity element
    "sph_harm",  # function name
    r"""sph_harm(m, l, phi, theta)

Spherical harmonics of real and complex argument

Warning:
    In order to stay consistent with Scipy, the order of the arguments is kept the
    same. For most other functions, the postion of degree and order are swapped.

For complex argument the spherical harmonics are computed with

.. math::

    Y_{lm}(\theta, \varphi) = \sqrt{\frac{2l + 1}{4\pi}\frac{(l - m)!}{(l + m)!}}
    P_l^m(\cos\theta) \mathrm e^{\mathrm i m \varphi}

for real arguments :py:data:`scipy.special.sph_harm` is used.

Args:
    m (integer, array_like): Order
    l (integer, array_like): Degree, non-negative
    phi (float or complex, array_like): Azimuthal angle
    theta (float or complex, array_like): Polar angle

Returns:
    complex
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_incgamma_loops[2]
cdef void *ufunc_incgamma_data[2]
cdef char ufunc_incgamma_types[2 * 3]

ufunc_incgamma_loops[0] = <np.PyUFuncGenericFunction>loop_d_dd
ufunc_incgamma_loops[1] = <np.PyUFuncGenericFunction>loop_D_dD
ufunc_incgamma_types[0] = <char>np.NPY_DOUBLE
ufunc_incgamma_types[1] = <char>np.NPY_DOUBLE
ufunc_incgamma_types[2] = <char>np.NPY_DOUBLE
ufunc_incgamma_types[3] = <char>np.NPY_DOUBLE
ufunc_incgamma_types[4] = <char>np.NPY_CDOUBLE
ufunc_incgamma_types[5] = <char>np.NPY_CDOUBLE
ufunc_incgamma_data[0] = <void*>_integrals.incgamma[double]
ufunc_incgamma_data[1] = <void*>_integrals.incgamma[double_complex]

incgamma = np.PyUFunc_FromFuncAndData(
    ufunc_incgamma_loops,
    ufunc_incgamma_data,
    ufunc_incgamma_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    "incgamma",  # function name
    r"""incgamma(l, z)

Upper incomplete Gamma function of integer and half-integer degree and real and complex
argument

This function is defined as

.. math

    Gamma(n, z) = \int_z^\infty t^{n - 1} \mathrm e^{-t} \mathrm dt

The negative real axis is the branch cut of the implemented function.

Args:
    l (integer or float): Integer or half-integer order
    z (float or complex): Argument

Returns:
    float or complex

References:
    - `DLMF: 8.2 <https://dlmf.nist.gov/8.2>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_intkambe_loops[2]
cdef void *ufunc_intkambe_data[2]
cdef char ufunc_intkambe_types[2 * 4]

ufunc_intkambe_loops[0] = <np.PyUFuncGenericFunction>loop_d_ldd
ufunc_intkambe_loops[1] = <np.PyUFuncGenericFunction>loop_D_lDD
ufunc_intkambe_types[0] = <char>np.NPY_LONG
ufunc_intkambe_types[1] = <char>np.NPY_DOUBLE
ufunc_intkambe_types[2] = <char>np.NPY_DOUBLE
ufunc_intkambe_types[3] = <char>np.NPY_DOUBLE
ufunc_intkambe_types[4] = <char>np.NPY_LONG
ufunc_intkambe_types[5] = <char>np.NPY_CDOUBLE
ufunc_intkambe_types[6] = <char>np.NPY_CDOUBLE
ufunc_intkambe_types[7] = <char>np.NPY_CDOUBLE
ufunc_intkambe_data[0] = <void*>_integrals.intkambe[double]
ufunc_intkambe_data[1] = <void*>_integrals.intkambe[double_complex]

intkambe = np.PyUFunc_FromFuncAndData(
    ufunc_intkambe_loops,
    ufunc_intkambe_data,
    ufunc_intkambe_types,
    2,  # number of supported input types
    3,  # number of input args
    1,  # number of output args
    0,  # identity element
    "intkambe",  # function name
    r"""intkambe(n, z, eta)

Integral appearing in the accelerated lattice summations

Named here after its appearance (in a slightly different form) in equation (3.16),
in Kambe's paper [#]_.

This function is defined as

.. math::

    I_n(\eta, z)
    = \int_\eta^\infty t^n \mathrm e^{-\frac{z^2t^2}{2} + \frac{1}{2t^2}}
    \mathrm d t

and is calculated via recursion.

Args:
    n (integer, array_like): Order
    z (float or complex, array_like): Argument
    eta (float or complex, array_like): Integral cutoff

Returns:
    float or complex

References:
    .. [#] `K. Kambe, Zeitschrift Fuer Naturforschung A 23, 9 (1968). <https://doi.org/10.1515/zna-1968-0908>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_wignersmalld_loops[2]
cdef void *ufunc_wignersmalld_data[2]
cdef char ufunc_wignersmalld_types[2 * 5]

ufunc_wignersmalld_loops[0] = <np.PyUFuncGenericFunction>loop_d_llld
ufunc_wignersmalld_loops[1] = <np.PyUFuncGenericFunction>loop_D_lllD
ufunc_wignersmalld_types[0] = <char>np.NPY_LONG
ufunc_wignersmalld_types[1] = <char>np.NPY_LONG
ufunc_wignersmalld_types[2] = <char>np.NPY_LONG
ufunc_wignersmalld_types[3] = <char>np.NPY_DOUBLE
ufunc_wignersmalld_types[4] = <char>np.NPY_DOUBLE
ufunc_wignersmalld_types[5] = <char>np.NPY_LONG
ufunc_wignersmalld_types[6] = <char>np.NPY_LONG
ufunc_wignersmalld_types[7] = <char>np.NPY_LONG
ufunc_wignersmalld_types[8] = <char>np.NPY_CDOUBLE
ufunc_wignersmalld_types[9] = <char>np.NPY_CDOUBLE
ufunc_wignersmalld_data[0] = <void*>_wignerd.wignersmalld[double]
ufunc_wignersmalld_data[1] = <void*>_wignerd.wignersmalld[double_complex]

wignersmalld = np.PyUFunc_FromFuncAndData(
    ufunc_wignersmalld_loops,
    ufunc_wignersmalld_data,
    ufunc_wignersmalld_types,
    2,  # number of supported input types
    4,  # number of input args
    1,  # number of output args
    0,  # identity element
    "wignersmalld",  # function name
    r"""wignersmalld(l, m, k, theta)

Wigner-d matrix element

Wigner-(small)-d symbol :math:`d^l_{mk}(\theta)` calculated via forward or backward
recursion for integer arguments. The decision is based on the number of
recursive function calls necessary. The function return values are cached
during recursion.

Note:
    Mathematica and Dachsela use a different sign convention, which means taking
    the negative angle.

Args:
    l (integer): Degree :math:`l \geq 0`
    m (integer): Order :math:`|m| \leq l`
    k (integer): Order :math:`|k| \leq l`
    theta (float or complex) Azimuthal angle

Returns:
    float or complex

References:
    - `Wikipedia: Wigner D-matrix <https://en.wikipedia.org/wiki/Wigner_D-matrix>`_
    - `H. Dachsela, J. Chem. Phys. 142, 144115 (2006) <https://doi.org/10.1063/1.2194548>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_wignerD_loops[2]
cdef void *ufunc_wignerD_data[2]
cdef char ufunc_wignerD_types[2 * 7]

ufunc_wignerD_loops[0] = <np.PyUFuncGenericFunction>loop_D_lllddd
ufunc_wignerD_loops[1] = <np.PyUFuncGenericFunction>loop_D_llldDd
ufunc_wignerD_types[0] = <char>np.NPY_LONG
ufunc_wignerD_types[1] = <char>np.NPY_LONG
ufunc_wignerD_types[2] = <char>np.NPY_LONG
ufunc_wignerD_types[3] = <char>np.NPY_DOUBLE
ufunc_wignerD_types[4] = <char>np.NPY_DOUBLE
ufunc_wignerD_types[5] = <char>np.NPY_DOUBLE
ufunc_wignerD_types[6] = <char>np.NPY_CDOUBLE
ufunc_wignerD_types[7] = <char>np.NPY_LONG
ufunc_wignerD_types[8] = <char>np.NPY_LONG
ufunc_wignerD_types[9] = <char>np.NPY_LONG
ufunc_wignerD_types[10] = <char>np.NPY_CDOUBLE
ufunc_wignerD_types[11] = <char>np.NPY_CDOUBLE
ufunc_wignerD_types[12] = <char>np.NPY_CDOUBLE
ufunc_wignerD_types[13] = <char>np.NPY_CDOUBLE
ufunc_wignerD_data[0] = <void*>_wignerd.wignerd[double]
ufunc_wignerD_data[1] = <void*>_wignerd.wignerd[double_complex]

wignerd = np.PyUFunc_FromFuncAndData(
    ufunc_wignerD_loops,
    ufunc_wignerD_data,
    ufunc_wignerD_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # identity element
    "wignerd",  # function name
    r"""wignerd(l, m, k, phi, theta, psi)

Wigner-D matrix element

.. math::

    D^l_{mk}(\varphi, \theta, \psi)
    = \mathrm e^{-\mathrm i m \varphi} d^l_{mk}(\theta) \mathrm e^{-\mathrm i k \psi}

See also :func:`treams.special.wignersmalld`.

Note:
    Mathematica uses a different sign convention, which means taking the negative
    angles.

Args:
    l (integer): Degree :math:`l \geq 0`
    m (integer): Order :math:`|m| \leq l`
    k (integer): Order :math:`|k| \leq l`
    phi, theta, psi (float or complex): Angles

Returns:
    complex

References:
    - `Wikipedia: Wigner D-matrix <https://en.wikipedia.org/wiki/Wigner_D-matrix>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_wigner3j_loops[1]
cdef void *ufunc_wigner3j_data[1]
cdef char ufunc_wigner3j_types[7]

ufunc_wigner3j_loops[0] = <np.PyUFuncGenericFunction>loop_d_llllll
ufunc_wigner3j_types[0] = <char>np.NPY_LONG
ufunc_wigner3j_types[1] = <char>np.NPY_LONG
ufunc_wigner3j_types[2] = <char>np.NPY_LONG
ufunc_wigner3j_types[3] = <char>np.NPY_LONG
ufunc_wigner3j_types[4] = <char>np.NPY_LONG
ufunc_wigner3j_types[5] = <char>np.NPY_LONG
ufunc_wigner3j_types[6] = <char>np.NPY_DOUBLE
ufunc_wigner3j_data[0] = <void*>_wigner3j.wigner3j

wigner3j = np.PyUFunc_FromFuncAndData(
    ufunc_wigner3j_loops,
    ufunc_wigner3j_data,
    ufunc_wigner3j_types,
    1,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # identity element
    "wigner3j",  # function name
    r"""wigner3j(j1, j2, j3, m1, m2, m3)

Wigner-3j symbol

Calculate

.. math::

    \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}

recursively by forward or backward recurstion.
Starting points are the extremal values for `j3.` The recursive function
calls are cached. For unphysical value combinations `0.0` is returned,
similar to Mathematica's behavior.

Args:
    j1, j2, j3 (integer, array_like): Degrees
    m1, m2, m3 (integer, array_like): Orders

Returns:
    float

References:
    - `Y.-L. Xu, J. Comput. Phys. 139, 137 - 165 (1998) <https://doi.org/10.1006/jcph.1997.5867>`_
    - `DLMF: 34.3 <https://dlmf.nist.gov/34.3>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_jv_d_loops[2]
cdef void *ufunc_jv_d_data[2]
cdef char ufunc_jv_d_types[2 * 3]

ufunc_jv_d_loops[0] = <np.PyUFuncGenericFunction>loop_d_dd
ufunc_jv_d_loops[1] = <np.PyUFuncGenericFunction>loop_D_dD
ufunc_jv_d_types[0] = <char>np.NPY_DOUBLE
ufunc_jv_d_types[1] = <char>np.NPY_DOUBLE
ufunc_jv_d_types[2] = <char>np.NPY_DOUBLE
ufunc_jv_d_types[3] = <char>np.NPY_DOUBLE
ufunc_jv_d_types[4] = <char>np.NPY_CDOUBLE
ufunc_jv_d_types[5] = <char>np.NPY_CDOUBLE
ufunc_jv_d_data[0] = <void*>_bessel.jv_d[double]
ufunc_jv_d_data[1] = <void*>_bessel.jv_d[double_complex]

jv_d = np.PyUFunc_FromFuncAndData(
    ufunc_jv_d_loops,
    ufunc_jv_d_data,
    ufunc_jv_d_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    "jv_d",  # function name
    r"""jv_d(v, z)

Derivative of the Bessel function of the first kind

Computed by

.. math::

    J_\nu'(z) = \frac{J_{\nu - 1}(z) - J_{\nu + 1}(z)}{2}

Args:
    v (float, array_like): Order
    z (float or complex, array_like): Argument

Returns:
    float or complex

References:
    - `DLMF: 10.6 <https://dlmf.nist.gov/10.6>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_yv_d_loops[2]
cdef void *ufunc_yv_d_data[2]
cdef char ufunc_yv_d_types[2 * 3]

ufunc_yv_d_loops[0] = <np.PyUFuncGenericFunction>loop_d_dd
ufunc_yv_d_loops[1] = <np.PyUFuncGenericFunction>loop_D_dD
ufunc_yv_d_types[0] = <char>np.NPY_DOUBLE
ufunc_yv_d_types[1] = <char>np.NPY_DOUBLE
ufunc_yv_d_types[2] = <char>np.NPY_DOUBLE
ufunc_yv_d_types[3] = <char>np.NPY_DOUBLE
ufunc_yv_d_types[4] = <char>np.NPY_CDOUBLE
ufunc_yv_d_types[5] = <char>np.NPY_CDOUBLE
ufunc_yv_d_data[0] = <void*>_bessel.yv_d[double]
ufunc_yv_d_data[1] = <void*>_bessel.yv_d[double_complex]

yv_d = np.PyUFunc_FromFuncAndData(
    ufunc_yv_d_loops,
    ufunc_yv_d_data,
    ufunc_yv_d_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    "yv_d",  # function name
    r"""yv_d(v, z)

Derivative of the Bessel function of the second kind

Computed by

.. math::

    Y_\nu'(z) = \frac{Y_{\nu - 1}(z) - Y_{\nu + 1}(z)}{2}

Args:
    v (float, array_like): Order
    z (float or complex, array_like): Argument

Returns:
    float or complex

References:
    - `DLMF: 10.6 <https://dlmf.nist.gov/10.6>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_spherical_hankel1_loops[1]
cdef void *ufunc_spherical_hankel1_data[1]
cdef char ufunc_spherical_hankel1_types[3]

ufunc_spherical_hankel1_loops[0] = <np.PyUFuncGenericFunction>loop_D_dD
ufunc_spherical_hankel1_types[0] = <char>np.NPY_DOUBLE
ufunc_spherical_hankel1_types[1] = <char>np.NPY_CDOUBLE
ufunc_spherical_hankel1_types[2] = <char>np.NPY_CDOUBLE
ufunc_spherical_hankel1_data[0] = <void*>_bessel.spherical_hankel1

spherical_hankel1 = np.PyUFunc_FromFuncAndData(
    ufunc_spherical_hankel1_loops,
    ufunc_spherical_hankel1_data,
    ufunc_spherical_hankel1_types,
    1,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    "spherical_hankel1",  # function name
    r"""spherical_hankel1(n, z)

Spherical Hankel function of the first kind

Defined by

.. math::

    h_n^{(1)}(z) = \sqrt{\frac{\pi}{2z}} H_{n + \frac{1}{2}}^{(1)}(z)

where :math:`H_{n + \frac{1}{2}}^{(1)}` is the Hankel function of the first kind.

Args:
    n (integer, array_like): Order
    z (complex, array_like): Argument

Returns:
    complex

References:
    - `DLMF: 10.47 <https://dlmf.nist.gov/10.47>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_spherical_hankel2_loops[1]
cdef void *ufunc_spherical_hankel2_data[1]
cdef char ufunc_spherical_hankel2_types[3]

ufunc_spherical_hankel2_loops[0] = <np.PyUFuncGenericFunction>loop_D_dD
ufunc_spherical_hankel2_types[0] = <char>np.NPY_DOUBLE
ufunc_spherical_hankel2_types[1] = <char>np.NPY_CDOUBLE
ufunc_spherical_hankel2_types[2] = <char>np.NPY_CDOUBLE
ufunc_spherical_hankel2_data[0] = <void*>_bessel.spherical_hankel2

spherical_hankel2 = np.PyUFunc_FromFuncAndData(
    ufunc_spherical_hankel2_loops,
    ufunc_spherical_hankel2_data,
    ufunc_spherical_hankel2_types,
    1,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    "spherical_hankel2",  # function name
    r"""spherical_hankel2(n, z)

Spherical Hankel function of the second kind

Defined by

.. math::

    h_n^{(2)}(z) = \sqrt{\frac{\pi}{2z}} H_{n + \frac{1}{2}}^{(2)}(z)

where :math:`H_{n + \frac{1}{2}}^{(2)}` is the Hankel function of the second kind.

Args:
    n (integer, array_like): Order
    z (complex, array_like): Argument

Returns:
    complex

References:
    - `DLMF: 10.47 <https://dlmf.nist.gov/10.47>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_spherical_hankel1_d_loops[1]
cdef void *ufunc_spherical_hankel1_d_data[1]
cdef char ufunc_spherical_hankel1_d_types[3]

ufunc_spherical_hankel1_d_loops[0] = <np.PyUFuncGenericFunction>loop_D_dD
ufunc_spherical_hankel1_d_types[0] = <char>np.NPY_DOUBLE
ufunc_spherical_hankel1_d_types[1] = <char>np.NPY_CDOUBLE
ufunc_spherical_hankel1_d_types[2] = <char>np.NPY_CDOUBLE
ufunc_spherical_hankel1_d_data[0] = <void*>_bessel.spherical_hankel1_d

spherical_hankel1_d = np.PyUFunc_FromFuncAndData(
    ufunc_spherical_hankel1_d_loops,
    ufunc_spherical_hankel1_d_data,
    ufunc_spherical_hankel1_d_types,
    1,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    "spherical_hankel1_d",  # function name
    r"""spherical_hankel1_d(n, z)

Derivative of the spherical Hankel function of the first kind.

Computed by

.. math::

    {h_n^{(1)}}'(z)
    = \frac{n h_{n - 1}^{(1)}(z) - (n + 1) h_{n + 1}^{(1)}(z)}{2n + 1}

Args:
    n (integer, array_like): Order
    z (complex, array_like): Argument

Returns:
    complex

References:
    - `DLMF: 10.51 <https://dlmf.nist.gov/10.51>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_spherical_hankel2_d_loops[1]
cdef void *ufunc_spherical_hankel2_d_data[1]
cdef char ufunc_spherical_hankel2_d_types[3]

ufunc_spherical_hankel2_d_loops[0] = <np.PyUFuncGenericFunction>loop_D_dD
ufunc_spherical_hankel2_d_types[0] = <char>np.NPY_DOUBLE
ufunc_spherical_hankel2_d_types[1] = <char>np.NPY_CDOUBLE
ufunc_spherical_hankel2_d_types[2] = <char>np.NPY_CDOUBLE
ufunc_spherical_hankel2_d_data[0] = <void*>_bessel.spherical_hankel2_d

spherical_hankel2_d = np.PyUFunc_FromFuncAndData(
    ufunc_spherical_hankel2_d_loops,
    ufunc_spherical_hankel2_d_data,
    ufunc_spherical_hankel2_d_types,
    1,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    "spherical_hankel2_d",  # function name
    r"""spherical_hankel2_d(n, z)

Derivative of the spherical Hankel function of the second kind

Computed by

.. math::

    {h_n^{(2)}}'(z)
    = \frac{n h_{n - 1}^{(2)}(z) - (n + 1) h_{n + 1}^{(2)}(z)}{2n + 1}

Args:
    n (integer, array_like): Order
    z (complex, array_like): Argument

Returns:
    complex

References:
    - `DLMF: 10.51 <https://dlmf.nist.gov/10.51>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_hankel1_d_loops[1]
cdef void *ufunc_hankel1_d_data[1]
cdef char ufunc_hankel1_d_types[3]

ufunc_hankel1_d_loops[0] = <np.PyUFuncGenericFunction>loop_D_dD
ufunc_hankel1_d_types[0] = <char>np.NPY_DOUBLE
ufunc_hankel1_d_types[1] = <char>np.NPY_CDOUBLE
ufunc_hankel1_d_types[2] = <char>np.NPY_CDOUBLE
ufunc_hankel1_d_data[0] = <void*>_bessel.hankel1_d

hankel1_d = np.PyUFunc_FromFuncAndData(
    ufunc_hankel1_d_loops,
    ufunc_hankel1_d_data,
    ufunc_hankel1_d_types,
    1,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    "hankel1_d",  # function name
    r"""hankel1_d(v, z)

Derivative of the Hankel function of the first kind.

Computed by

.. math::

    {H_\nu^{(1)}}'(z) = \frac{H_{\nu - 1}^{(1)}(z) - H_{\nu + 1}^{(1)}(z)}{2}

Args:
    v (float, array_like): Order
    z (complex, array_like): Argument

Returns:
    complex

References:
    - `DLMF: 10.6 <https://dlmf.nist.gov/10.6>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_hankel2_d_loops[1]
cdef void *ufunc_hankel2_d_data[1]
cdef char ufunc_hankel2_d_types[3]

ufunc_hankel2_d_loops[0] = <np.PyUFuncGenericFunction>loop_D_dD
ufunc_hankel2_d_types[0] = <char>np.NPY_DOUBLE
ufunc_hankel2_d_types[1] = <char>np.NPY_CDOUBLE
ufunc_hankel2_d_types[2] = <char>np.NPY_CDOUBLE
ufunc_hankel2_d_data[0] = <void*>_bessel.hankel2_d

hankel2_d = np.PyUFunc_FromFuncAndData(
    ufunc_hankel2_d_loops,
    ufunc_hankel2_d_data,
    ufunc_hankel2_d_types,
    1,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    "hankel2_d",  # function name
    r"""hankel2_d(v, z)

Derivative of the Hankel function of the second kind

Computed by

.. math::

    {H_\nu^{(2)}}'(z) = \frac{H_{\nu - 1}^{(1)}(z) - H_{\nu + 1}^{(2)}(z)}{2}

Args:
    v (float, array_like): Order
    z (complex, array_like): Argument

Returns:
    complex

References:
    - `DLMF: 10.6 <https://dlmf.nist.gov/10.6>`_
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_pi_fun_loops[2]
cdef void *ufunc_pi_fun_data[2]
cdef char ufunc_pi_fun_types[2 * 4]

ufunc_pi_fun_loops[0] = <np.PyUFuncGenericFunction>loop_d_ddd
ufunc_pi_fun_loops[1] = <np.PyUFuncGenericFunction>loop_D_ddD
ufunc_pi_fun_types[0] = <char>np.NPY_DOUBLE
ufunc_pi_fun_types[1] = <char>np.NPY_DOUBLE
ufunc_pi_fun_types[2] = <char>np.NPY_DOUBLE
ufunc_pi_fun_types[3] = <char>np.NPY_DOUBLE
ufunc_pi_fun_types[4] = <char>np.NPY_DOUBLE
ufunc_pi_fun_types[5] = <char>np.NPY_DOUBLE
ufunc_pi_fun_types[6] = <char>np.NPY_CDOUBLE
ufunc_pi_fun_types[7] = <char>np.NPY_CDOUBLE
ufunc_pi_fun_data[0] = <void*>_waves.pi_fun[double]
ufunc_pi_fun_data[1] = <void*>_waves.pi_fun[double_complex]

pi_fun = np.PyUFunc_FromFuncAndData(
    ufunc_pi_fun_loops,
    ufunc_pi_fun_data,
    ufunc_pi_fun_types,
    2,  # number of supported input types
    3,  # number of input args
    1,  # number of output args
    0,  # identity element
    "pi_fun",  # function name
    r"""pi_fun(l, m, x)

Angular function pi

.. math::

    \pi^l_m(x) = \frac{m P_l^m(x)}{\sqrt{1 - x^2}}

where :math:`P^l_m` is the associated Legendre polynomial.

Args:
    l (integer, array_like): degree :math:`l \geq 0`
    m (integer, array_like): order :math:`|m| \leq l`
    x (float or complex, array_like): argument

Returns:
    float or complex
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_tau_fun_loops[2]
cdef void *ufunc_tau_fun_data[2]
cdef char ufunc_tau_fun_types[2 * 4]

ufunc_tau_fun_loops[0] = <np.PyUFuncGenericFunction>loop_d_ddd
ufunc_tau_fun_loops[1] = <np.PyUFuncGenericFunction>loop_D_ddD
ufunc_tau_fun_types[0] = <char>np.NPY_DOUBLE
ufunc_tau_fun_types[1] = <char>np.NPY_DOUBLE
ufunc_tau_fun_types[2] = <char>np.NPY_DOUBLE
ufunc_tau_fun_types[3] = <char>np.NPY_DOUBLE
ufunc_tau_fun_types[4] = <char>np.NPY_DOUBLE
ufunc_tau_fun_types[5] = <char>np.NPY_DOUBLE
ufunc_tau_fun_types[6] = <char>np.NPY_CDOUBLE
ufunc_tau_fun_types[7] = <char>np.NPY_CDOUBLE
ufunc_tau_fun_data[0] = <void*>_waves.tau_fun[double]
ufunc_tau_fun_data[1] = <void*>_waves.tau_fun[double_complex]

tau_fun = np.PyUFunc_FromFuncAndData(
    ufunc_tau_fun_loops,
    ufunc_tau_fun_data,
    ufunc_tau_fun_types,
    2,  # number of supported input types
    3,  # number of input args
    1,  # number of output args
    0,  # identity element
    "tau_fun",  # function name
    r"""tau_fun(l, m, x)

Angular function tau

.. math::

    \tau^l_m(x)
    = \left.\frac{\mathrm d}{\mathrm d \theta}P_l^m(\cos\theta)\right|_{x = \cos\theta}

where :math:`P^l_m` is the associated Legendre polynomial.

Args:
    l (integer, array_like): degree :math:`l \geq 0`
    m (integer, array_like): order :math:`|m| \leq l`
    x (float or complex, array_like): argument

Returns:
    float or complex
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_tl_vsw_rA_loops[2]
cdef void *ufunc_tl_vsw_rA_data[2]
cdef char ufunc_tl_vsw_rA_types[2 * 8]

ufunc_tl_vsw_rA_loops[0] = <np.PyUFuncGenericFunction>loop_D_llllddd
ufunc_tl_vsw_rA_loops[1] = <np.PyUFuncGenericFunction>loop_D_llllDDd
ufunc_tl_vsw_rA_types[0] = <char>np.NPY_LONG
ufunc_tl_vsw_rA_types[1] = <char>np.NPY_LONG
ufunc_tl_vsw_rA_types[2] = <char>np.NPY_LONG
ufunc_tl_vsw_rA_types[3] = <char>np.NPY_LONG
ufunc_tl_vsw_rA_types[4] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_rA_types[5] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_rA_types[6] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_rA_types[7] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_rA_types[8] = <char>np.NPY_LONG
ufunc_tl_vsw_rA_types[9] = <char>np.NPY_LONG
ufunc_tl_vsw_rA_types[10] = <char>np.NPY_LONG
ufunc_tl_vsw_rA_types[11] = <char>np.NPY_LONG
ufunc_tl_vsw_rA_types[12] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_rA_types[13] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_rA_types[14] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_rA_types[15] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_rA_data[0] = <void*>_waves.tl_vsw_rA[double]
ufunc_tl_vsw_rA_data[1] = <void*>_waves.tl_vsw_rA[double_complex]

tl_vsw_rA = np.PyUFunc_FromFuncAndData(
    ufunc_tl_vsw_rA_loops,
    ufunc_tl_vsw_rA_data,
    ufunc_tl_vsw_rA_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # identity element
    "tl_vsw_rA",  # function name
    r"""tl_vsw_rA(lambda, mu, l, m, x, theta, phi)

Translation coefficient for vector spherical waves with the same parity

Definded by [#]_

.. math::

    A_{\lambda\mu lm}^{(1)}(x, \theta, \varphi) =
    \frac{\gamma_{lm}}{\gamma_{\lambda\mu}}
    (-1)^m \frac{2\lambda + 1}{\lambda (\lambda + 1)} \mathrm i^{\lambda - l}
    \sqrt{\pi \frac{(l + m)!(\lambda - \mu)!}{(l - m)!(\lambda + \mu)!}} \\ \cdot
    \sum_{p} \mathrm i^p\sqrt{2p + 1}
    j_p(kr) Y_{p, m - \mu}(\theta, \varphi)
    \begin{pmatrix}
        l & \lambda & p \\
        m & -\mu & -m + \mu
    \end{pmatrix} \\ \cdot
    \begin{pmatrix}
        l & \lambda & p \\
        0 & 0 & 0
    \end{pmatrix}
    \left[l (l + 1) + \lambda (\lambda + 1) - p (p + 1)\right]

with

.. math::

    \gamma_{lm} = \mathrm i \sqrt{\frac{2l + 1}{4\pi l (l + 1)}\frac{(l - m)!}{(l + m)!}}

and the Wigner 3j-symbols (:func:`treams.special.wigner3j`) and the spherical Bessel
functions. The summation runs over all
:math:`p \in \{\lambda + l, \lambda + l - 2, \dots, \max(|\lambda - l|, |\mu - m|)\}`.

These coefficients are used to translate from incident to incident modes and from
scattered to scattered modes.

Args:
    lambda (integer, array_like): Degree of the destination mode
    mu (integer, array_like): Order of the destination mode
    l (integer, array_like): Degree of the source mode
    m (integer, array_like): Order of the source mode
    x (complex, array_like): Translation in units of the wave number
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angel

Returns:
    complex

References:
    .. [#] L. Tsang, J. A. Kong, and R. T. Shi, Theory of Microwave Remote Sensing (Wiley Series in Remote Sensing and Image Processing) (Wiley-Interscience, 1985).
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_tl_vsw_rB_loops[2]
cdef void *ufunc_tl_vsw_rB_data[2]
cdef char ufunc_tl_vsw_rB_types[2 * 8]

ufunc_tl_vsw_rB_loops[0] = <np.PyUFuncGenericFunction>loop_D_llllddd
ufunc_tl_vsw_rB_loops[1] = <np.PyUFuncGenericFunction>loop_D_llllDDd
ufunc_tl_vsw_rB_types[0] = <char>np.NPY_LONG
ufunc_tl_vsw_rB_types[1] = <char>np.NPY_LONG
ufunc_tl_vsw_rB_types[2] = <char>np.NPY_LONG
ufunc_tl_vsw_rB_types[3] = <char>np.NPY_LONG
ufunc_tl_vsw_rB_types[4] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_rB_types[5] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_rB_types[6] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_rB_types[7] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_rB_types[8] = <char>np.NPY_LONG
ufunc_tl_vsw_rB_types[9] = <char>np.NPY_LONG
ufunc_tl_vsw_rB_types[10] = <char>np.NPY_LONG
ufunc_tl_vsw_rB_types[11] = <char>np.NPY_LONG
ufunc_tl_vsw_rB_types[12] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_rB_types[13] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_rB_types[14] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_rB_types[15] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_rB_data[0] = <void*>_waves.tl_vsw_rB[double]
ufunc_tl_vsw_rB_data[1] = <void*>_waves.tl_vsw_rB[double_complex]

tl_vsw_rB = np.PyUFunc_FromFuncAndData(
    ufunc_tl_vsw_rB_loops,
    ufunc_tl_vsw_rB_data,
    ufunc_tl_vsw_rB_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # identity element
    "tl_vsw_rB",  # function name
    r"""tl_vsw_rB(lambda, mu, l, m, x, theta, phi)

Translation coefficient for vector spherical waves with opposite parity

Definded by [#]_

.. math::

    B_{\lambda\mu lm}^{(1)}(x, \theta, \varphi) =
    \frac{\gamma_{lm}}{\gamma_{\lambda\mu}}
    (-1)^m \frac{2\lambda + 1}{\lambda (\lambda + 1)} \mathrm i^{\lambda - l}
    \sqrt{\pi \frac{(l + m)!(\lambda - \mu)!}{(l - m)!(\lambda + \mu)!}} \\ \cdot
    \sum_{p} \mathrm i^p\sqrt{2p + 1}
    j_p(kr) Y_{p, m - \mu}(\theta, \varphi)
    \begin{pmatrix}
        l & \lambda & p \\
        m & -\mu & -m + \mu
    \end{pmatrix} \\ \cdot
    \begin{pmatrix}
        l & \lambda & p - 1 \\
        0 & 0 & 0
    \end{pmatrix}
    \sqrt{\left[(l + \lambda + 1)^2 - p^2\right]\left[p^2 - (l - \lambda)^2\right]}

with

.. math::

    \gamma_{lm} = \mathrm i \sqrt{\frac{2l + 1}{4\pi l (l + 1)}\frac{(l - m)!}{(l + m)!}}

and the Wigner 3j-symbols (:func:`treams.special.wigner3j`) and the spherical Bessel
functions. The summation runs over all
:math:`p \in \{\lambda + l - 1, \lambda + l - 3, \dots,
\max(|\lambda - l| + 1, |\mu - m|)\}`.

These coefficients are used to translate from incident to incident modes and from
scattered to scattered modes.

Args:
    lambda (integer, array_like): Degree of the destination mode
    mu (integer, array_like): Order of the destination mode
    l (integer, array_like): Degree of the source mode
    m (integer, array_like): Order of the source mode
    x (complex, array_like): Translation in units of the wave number
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angel

Returns:
    complex

References:
    .. [#] L. Tsang, J. A. Kong, and R. T. Shi, Theory of Microwave Remote Sensing (Wiley Series in Remote Sensing and Image Processing) (Wiley-Interscience, 1985).
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_tl_vsw_A_loops[2]
cdef void *ufunc_tl_vsw_A_data[2]
cdef char ufunc_tl_vsw_A_types[2 * 8]

ufunc_tl_vsw_A_loops[0] = <np.PyUFuncGenericFunction>loop_D_llllDdd
ufunc_tl_vsw_A_loops[1] = <np.PyUFuncGenericFunction>loop_D_llllDDd
ufunc_tl_vsw_A_types[0] = <char>np.NPY_LONG
ufunc_tl_vsw_A_types[1] = <char>np.NPY_LONG
ufunc_tl_vsw_A_types[2] = <char>np.NPY_LONG
ufunc_tl_vsw_A_types[3] = <char>np.NPY_LONG
ufunc_tl_vsw_A_types[4] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_A_types[5] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_A_types[6] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_A_types[7] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_A_types[8] = <char>np.NPY_LONG
ufunc_tl_vsw_A_types[9] = <char>np.NPY_LONG
ufunc_tl_vsw_A_types[10] = <char>np.NPY_LONG
ufunc_tl_vsw_A_types[11] = <char>np.NPY_LONG
ufunc_tl_vsw_A_types[12] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_A_types[13] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_A_types[14] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_A_types[15] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_A_data[0] = <void*>_waves.tl_vsw_A[double]
ufunc_tl_vsw_A_data[1] = <void*>_waves.tl_vsw_A[double_complex]

tl_vsw_A = np.PyUFunc_FromFuncAndData(
    ufunc_tl_vsw_A_loops,
    ufunc_tl_vsw_A_data,
    ufunc_tl_vsw_A_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # identity element
    "tl_vsw_A",  # function name
    r"""tl_vsw_A(lambda, mu, l, m, x, theta, phi)

Translation coefficient for vector spherical waves with the same parity

Definded by [#]_

.. math::

    A_{\lambda\mu lm}^{(3)}(x, \theta, \varphi) =
    \frac{\gamma_{lm}}{\gamma_{\lambda\mu}}
    (-1)^m \frac{2\lambda + 1}{\lambda (\lambda + 1)} \mathrm i^{\lambda - l}
    \sqrt{\pi \frac{(l + m)!(\lambda - \mu)!}{(l - m)!(\lambda + \mu)!}} \\ \cdot
    \sum_{p} \mathrm i^p\sqrt{2p + 1}
    h_p^{(1)}(kr) Y_{p, m - \mu}(\theta, \varphi)
    \begin{pmatrix}
        l & \lambda & p \\
        m & -\mu & -m + \mu
    \end{pmatrix} \\ \cdot
    \begin{pmatrix}
        l & \lambda & p \\
        0 & 0 & 0
    \end{pmatrix}
    \left[l (l + 1) + \lambda (\lambda + 1) - p (p + 1)\right]

with

.. math::

    \gamma_{lm}
    = \mathrm i \sqrt{\frac{2l + 1}{4\pi l (l + 1)}\frac{(l - m)!}{(l + m)!}}

and the Wigner 3j-symbols (:func:`treams.special.wigner3j`) and the spherical Hankel
functions. The summation runs over all
:math:`p \in \{\lambda + l, \lambda + l - 2, \dots, \max(|\lambda - l|, |\mu - m|)\}`.

These coefficients are used to translate from scattered to incident modes.

Args:
    lambda (integer, array_like): Degree of the destination mode
    mu (integer, array_like): Order of the destination mode
    l (integer, array_like): Degree of the source mode
    m (integer, array_like): Order of the source mode
    x (complex, array_like): Translation in units of the wave number
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angel

Returns:
    complex

References:
    .. [#] L. Tsang, J. A. Kong, and R. T. Shi, Theory of Microwave Remote Sensing (Wiley Series in Remote Sensing and Image Processing) (Wiley-Interscience, 1985).
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_tl_vsw_B_loops[2]
cdef void *ufunc_tl_vsw_B_data[2]
cdef char ufunc_tl_vsw_B_types[2 * 8]

ufunc_tl_vsw_B_loops[0] = <np.PyUFuncGenericFunction>loop_D_llllDdd
ufunc_tl_vsw_B_loops[1] = <np.PyUFuncGenericFunction>loop_D_llllDDd
ufunc_tl_vsw_B_types[0] = <char>np.NPY_LONG
ufunc_tl_vsw_B_types[1] = <char>np.NPY_LONG
ufunc_tl_vsw_B_types[2] = <char>np.NPY_LONG
ufunc_tl_vsw_B_types[3] = <char>np.NPY_LONG
ufunc_tl_vsw_B_types[4] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_B_types[5] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_B_types[6] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_B_types[7] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_B_types[8] = <char>np.NPY_LONG
ufunc_tl_vsw_B_types[9] = <char>np.NPY_LONG
ufunc_tl_vsw_B_types[10] = <char>np.NPY_LONG
ufunc_tl_vsw_B_types[11] = <char>np.NPY_LONG
ufunc_tl_vsw_B_types[12] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_B_types[13] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_B_types[14] = <char>np.NPY_DOUBLE
ufunc_tl_vsw_B_types[15] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_B_data[0] = <void*>_waves.tl_vsw_B[double]
ufunc_tl_vsw_B_data[1] = <void*>_waves.tl_vsw_B[double_complex]

tl_vsw_B = np.PyUFunc_FromFuncAndData(
    ufunc_tl_vsw_B_loops,
    ufunc_tl_vsw_B_data,
    ufunc_tl_vsw_B_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # identity element
    "tl_vsw_B",  # function name
    r"""tl_vsw_B(lambda, mu, l, m, x, theta, phi)

Translation coefficient for vector spherical waves with opposite parity

Definded by [#]_

.. math::

    B_{\lambda\mu lm}^{(3)}(x, \theta, \varphi) =
    \frac{\gamma_{lm}}{\gamma_{\lambda\mu}}
    (-1)^m \frac{2\lambda + 1}{\lambda (\lambda + 1)} \mathrm i^{\lambda - l}
    \sqrt{\pi \frac{(l + m)!(\lambda - \mu)!}{(l - m)!(\lambda + \mu)!}} \\ \cdot
    \sum_{p} \mathrm i^p\sqrt{2p + 1}
    h_p^{(1)}(kr) Y_{p, m - \mu}(\theta, \varphi)
    \begin{pmatrix}
        l & \lambda & p \\
        m & -\mu & -m + \mu
    \end{pmatrix} \\ \cdot
    \begin{pmatrix}
        l & \lambda & p - 1 \\
        0 & 0 & 0
    \end{pmatrix}
    \sqrt{\left[(l + \lambda + 1)^2 - p^2\right]\left[p^2 - (l - \lambda)^2\right]}

with

.. math::

    \gamma_{lm}
    = \mathrm i \sqrt{\frac{2l + 1}{4\pi l (l + 1)}\frac{(l - m)!}{(l + m)!}}

and the Winger 3j-symbols (:func:`treams.special.wigner3j`) and the spherical Hankel
functions. The summation runs over all
:math:`p \in \{\lambda + l - 1, \lambda + l - 3, \dots, \max(|\lambda - l| + 1, |\mu - m|)\}`.

These coefficients are used to translate from scattered to incident modes.

Args:
    lambda (integer, array_like): Degree of the destination mode
    mu (integer, array_like): Order of the destination mode
    l (integer, array_like): Degree of the source mode
    m (integer, array_like): Order of the source mode
    x (complex, array_like): Translation in units of the wave number
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angel

Returns:
    complex

References:
    .. [#] L. Tsang, J. A. Kong, and R. T. Shi, Theory of Microwave Remote Sensing (Wiley Series in Remote Sensing and Image Processing) (Wiley-Interscience, 1985).
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_tl_vcw_loops[1]
cdef void *ufunc_tl_vcw_data[1]
cdef char ufunc_tl_vcw_types[8]

ufunc_tl_vcw_loops[0] = <np.PyUFuncGenericFunction>loop_D_dldlDdd
ufunc_tl_vcw_types[0] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_types[1] = <char>np.NPY_LONG
ufunc_tl_vcw_types[2] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_types[3] = <char>np.NPY_LONG
ufunc_tl_vcw_types[4] = <char>np.NPY_CDOUBLE
ufunc_tl_vcw_types[5] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_types[6] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_types[7] = <char>np.NPY_CDOUBLE
ufunc_tl_vcw_data[0] = <void*>_waves.tl_vcw

tl_vcw = np.PyUFunc_FromFuncAndData(
    ufunc_tl_vcw_loops,
    ufunc_tl_vcw_data,
    ufunc_tl_vcw_types,
    1,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # identity element
    "tl_vcw",  # function name
    r"""tl_vcw(kz1, mu, kz2, m, k, xrho, phi, z)

Translation coefficient for vector cylindrical waves from scattered to incident modes

Definded by

.. math::

    \begin{cases}
        H_{m - \mu}^{(1)}(x_\rho) \mathrm e^{\mathrm i ((m - \mu) \varphi + k_{z,1}z)}
        & \text{if }k_{z,1} = k_{z,2} \\
        0 & \text{otherwise}
    \end{cases}

where :math:`H_{m - \mu}^{(1)}` are the Hankel functions.

These coefficients are used to translate from incident to incident modes and from
scattered to scattered modes.

Args:
    kz1 (float, array_like): Z component of the destination mode's wave vector
    mu (integer, array_like): Order of the destination mode
    kz2 (float, array_like): Z component of the source mode's wave vector
    m (integer, array_like): Order of the source mode
    xrho (complex, array_like): Translation in radial direction in units of the wave number
    phi (float, array_like): Azimuthal angel
    z (float, array_like): Translation in z direction

Returns:
    complex
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_tl_vcw_r_loops[2]
cdef void *ufunc_tl_vcw_r_data[2]
cdef char ufunc_tl_vcw_r_types[2 * 8]

ufunc_tl_vcw_r_loops[0] = <np.PyUFuncGenericFunction>loop_D_dldlddd
ufunc_tl_vcw_r_loops[1] = <np.PyUFuncGenericFunction>loop_D_dldlDdd
ufunc_tl_vcw_r_types[0] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_r_types[1] = <char>np.NPY_LONG
ufunc_tl_vcw_r_types[2] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_r_types[3] = <char>np.NPY_LONG
ufunc_tl_vcw_r_types[4] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_r_types[5] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_r_types[6] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_r_types[7] = <char>np.NPY_CDOUBLE
ufunc_tl_vcw_r_types[8] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_r_types[9] = <char>np.NPY_LONG
ufunc_tl_vcw_r_types[10] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_r_types[11] = <char>np.NPY_LONG
ufunc_tl_vcw_r_types[12] = <char>np.NPY_CDOUBLE
ufunc_tl_vcw_r_types[13] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_r_types[14] = <char>np.NPY_DOUBLE
ufunc_tl_vcw_r_types[15] = <char>np.NPY_CDOUBLE
ufunc_tl_vcw_r_data[0] = <void*>_waves.tl_vcw_r[double]
ufunc_tl_vcw_r_data[1] = <void*>_waves.tl_vcw_r[double_complex]

tl_vcw_r = np.PyUFunc_FromFuncAndData(
    ufunc_tl_vcw_r_loops,
    ufunc_tl_vcw_r_data,
    ufunc_tl_vcw_r_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # identity element
    "tl_vcw_r",  # function name
    r"""tl_vcw_r(kz1, mu, kz2, m, k, xrho, phi, z)

Translation coefficient for vector cylindrical waves of the same kind

Definded by

.. math::

    \begin{cases}
        J_{m - \mu}(x_\rho) \mathrm e^{\mathrm i ((m - \mu) \varphi + k_{z,1}z)}
        & \text{if }k_{z,1} = k_{z,2} \\
        0 & \text{otherwise}
    \end{cases}

where :math:`J_{m - \mu}^{(1)}` are the Bessel functions.

These coefficients are used to translate from incident to incident modes and from
scattered to scattered modes.

Args:
    kz1 (float, array_like): Z component of the destination mode's wave vector
    mu (integer, array_like): Order of the destination mode
    kz2 (float, array_like): Z component of the source mode's wave vector
    m (integer, array_like): Order of the source mode
    xrho (complex, array_like): Translation in radial direction in units of the wave
        number
    phi (float, array_like): Azimuthal angel
    z (float, array_like): Translation in z direction

Returns:
    complex
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction ufunc_tl_vsw_helper_loops[1]
cdef void *ufunc_tl_vsw_helper_data[1]
cdef char ufunc_tl_vsw_helper_types[7]

ufunc_tl_vsw_helper_loops[0] = <np.PyUFuncGenericFunction>loop_D_llllll
ufunc_tl_vsw_helper_types[0] = <char>np.NPY_LONG
ufunc_tl_vsw_helper_types[1] = <char>np.NPY_LONG
ufunc_tl_vsw_helper_types[2] = <char>np.NPY_LONG
ufunc_tl_vsw_helper_types[3] = <char>np.NPY_LONG
ufunc_tl_vsw_helper_types[4] = <char>np.NPY_LONG
ufunc_tl_vsw_helper_types[5] = <char>np.NPY_LONG
ufunc_tl_vsw_helper_types[6] = <char>np.NPY_CDOUBLE
ufunc_tl_vsw_helper_data[0] = <void*>_waves._tl_vsw_helper

_tl_vsw_helper = np.PyUFunc_FromFuncAndData(
    ufunc_tl_vsw_helper_loops,
    ufunc_tl_vsw_helper_data,
    ufunc_tl_vsw_helper_types,
    1,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # identity element
    "_tl_vsw_helper",  # function name
    r"""_tl_vsw_helper(l, m, lambda_, mu, p, q)""",  # docstring
    0,  # unused
)
