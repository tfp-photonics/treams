"""
==========================================
Scattering coefficients for selected cases
==========================================

Calculate the scattering coefficients for cases where they can be obtained analytically
easily. This is a (multilayered) sphere using spherical waves (Mie coefficients), a
(multilayered) cylinder using cylindrical waves, and an infinitely extended planar
interface (Fresnel coefficients).

.. note::
   To accomodate chiral materials, the solution is always calculated in helicity basis,
   which is less commonly used than parity basis.

.. autosummary::
   :toctree: generated/

   mie
   mie_cyl
   fresnel

"""

cimport numpy as np
cimport scipy.special.cython_special as sc
from libc.string cimport memcpy
from scipy.linalg.cython_blas cimport zgemm
from scipy.linalg.cython_lapack cimport zgesv

cimport ptsa.special.cython_special as cs
from ptsa.special._misc cimport double_complex, sqrt


cdef extern from "<complex.h>" nogil:
    double creal(double complex z)
    double cimag(double complex z)
    double complex csqrt(double complex z)

cdef void _interface(long l, number_t x[2][2], number_t *z, double complex m[4][4]) nogil:
    """
    Fill a matrix for the relation at a spherical interface

    Note:
        The result is stored in column major order, because it is later processed with
        Fortran routines.

    Args:
        l: Degree of the coefficient
        x: Size parameters in the corresponding media, with the first dimension indexing
            the side and the second parameter the helicity
        z: Impedances
        m: Return array, indexing regular and scattered modes and both helicities

    """
    cdef number_t sb[2][2]
    cdef double complex sh[2][2]
    cdef number_t psi[2][2]
    cdef double complex chi[2][2]
    cdef long i, j
    for i in range(2):
        for j in range(2):
            sb[i][j] = sc.spherical_jn(l, x[i][j], 0)
            sh[i][j] = cs.spherical_hankel1(l, x[i][j])
            psi[i][j] = sc.spherical_jn(l, x[i][j], 1) + sb[i][j] / x[i][j]
            chi[i][j] = cs.spherical_hankel1_d(l, x[i][j]) + sh[i][j] / x[i][j]
    cdef double complex zs = (z[1] + z[0]) / (2j * z[0])
    cdef double complex zd = (z[1] - z[0]) / (2j * z[0])
    # Column major order! The actual matrix we have in mind is transposed here
    m[0][0] = (chi[1][0] * sb[0][0] - sh[1][0] * psi[0][0]) * zs * x[1][0] * x[1][0]
    m[1][0] = (chi[1][0] * sb[0][1] + sh[1][0] * psi[0][1]) * zd * x[1][0] * x[1][0]
    m[2][0] = (chi[1][0] * sh[0][0] - sh[1][0] * chi[0][0]) * zs * x[1][0] * x[1][0]
    m[3][0] = (chi[1][0] * sh[0][1] + sh[1][0] * chi[0][1]) * zd * x[1][0] * x[1][0]

    m[0][1] = (chi[1][1] * sb[0][0] + sh[1][1] * psi[0][0]) * zd * x[1][1] * x[1][1]
    m[1][1] = (chi[1][1] * sb[0][1] - sh[1][1] * psi[0][1]) * zs * x[1][1] * x[1][1]
    m[2][1] = (chi[1][1] * sh[0][0] + sh[1][1] * chi[0][0]) * zd * x[1][1] * x[1][1]
    m[3][1] = (chi[1][1] * sh[0][1] - sh[1][1] * chi[0][1]) * zs * x[1][1] * x[1][1]

    m[0][2] = (-psi[1][0] * sb[0][0] + sb[1][0] * psi[0][0]) * zs * x[1][0] * x[1][0]
    m[1][2] = (-psi[1][0] * sb[0][1] - sb[1][0] * psi[0][1]) * zd * x[1][0] * x[1][0]
    m[2][2] = (-psi[1][0] * sh[0][0] + sb[1][0] * chi[0][0]) * zs * x[1][0] * x[1][0]
    m[3][2] = (-psi[1][0] * sh[0][1] - sb[1][0] * chi[0][1]) * zd * x[1][0] * x[1][0]

    m[0][3] = (-psi[1][1] * sb[0][0] - sb[1][1] * psi[0][0]) * zd * x[1][1] * x[1][1]
    m[1][3] = (-psi[1][1] * sb[0][1] + sb[1][1] * psi[0][1]) * zs * x[1][1] * x[1][1]
    m[2][3] = (-psi[1][1] * sh[0][0] - sb[1][1] * chi[0][0]) * zd * x[1][1] * x[1][1]
    m[3][3] = (-psi[1][1] * sh[0][1] + sb[1][1] * chi[0][1]) * zs * x[1][1] * x[1][1]


cdef void _innermost_interface(long l, number_t x[2][2], number_t *z, double complex m[2][4]) nogil:
    """
    Fill a matrix for the relation at the innermost spherical interface

    This function is essentially the same as `_interface` but neglecting the singular
    modes on the inner side, because they must vanish.

    Note:
        The result is stored in column major order, because it is later processed with
        Fortran routines.

    Args:
        l: Degree of the coefficient
        x: Size parameters in the corresponding media, with the first dimension indexing
            the side and the second parameter the helicity
        z: Impedances
        m: Return array, indexing regular and scattered modes and both helicities

    """
    cdef number_t sb[2][2]
    cdef double complex sh[2][2]
    cdef number_t psi[2][2]
    cdef double complex chi[2][2]
    cdef long i, j
    for i in range(2):
        for j in range(2):
            sb[i][j] = sc.spherical_jn(l, x[i][j], 0)
            sh[i][j] = cs.spherical_hankel1(l, x[i][j])
            psi[i][j] = sc.spherical_jn(l, x[i][j], 1) + sb[i][j] / x[i][j]
            chi[i][j] = cs.spherical_hankel1_d(l, x[i][j]) + sh[i][j] / x[i][j]
    cdef double complex zs = (z[1] + z[0]) / (2j * z[0])
    cdef double complex zd = (z[1] - z[0]) / (2j * z[0])
    # Column major order! The actual matrix we have in mind is transposed here
    m[0][0] = (chi[1][0] * sb[0][0] - sh[1][0] * psi[0][0]) * zs * x[1][0] * x[1][0]
    m[1][0] = (chi[1][0] * sb[0][1] + sh[1][0] * psi[0][1]) * zd * x[1][0] * x[1][0]

    m[0][1] = (chi[1][1] * sb[0][0] + sh[1][1] * psi[0][0]) * zd * x[1][1] * x[1][1]
    m[1][1] = (chi[1][1] * sb[0][1] - sh[1][1] * psi[0][1]) * zs * x[1][1] * x[1][1]

    m[0][2] = (-psi[1][0] * sb[0][0] + sb[1][0] * psi[0][0]) * zs * x[1][0] * x[1][0]
    m[1][2] = (-psi[1][0] * sb[0][1] - sb[1][0] * psi[0][1]) * zd * x[1][0] * x[1][0]

    m[0][3] = (-psi[1][1] * sb[0][0] - sb[1][1] * psi[0][0]) * zd * x[1][1] * x[1][1]
    m[1][3] = (-psi[1][1] * sb[0][1] + sb[1][1] * psi[0][1]) * zs * x[1][1] * x[1][1]


cdef void _mie(long l, double *x, number_t *epsilon, number_t *mu, number_t *kappa, long n, double complex res[2][2]) nogil:
    """
    Calculate the mie coefficients for degree l
    """
    cdef double complex mfull[4][4]
    cdef double complex mtmp[2][4]  # Column major
    cdef double complex m[2][4]  # Column major
    cdef number_t xn[2][2]
    cdef number_t z[2]
    cdef number_t nr[2]
    cdef char c = b'N'  # Since the matrices are filled accordingly
    cdef int two = 2, four = 4
    cdef double complex zone = 1, zzero = 0
    nr[1] = sqrt(epsilon[0] * mu[0])
    z[1] = sqrt(mu[0] / epsilon[0])
    cdef long i
    for i in range(n):
        z[0] = z[1]
        z[1] = sqrt(mu[i + 1] / epsilon[i + 1])
        nr[0] = nr[1]
        nr[1] = sqrt(epsilon[i + 1] * mu[i + 1])
        xn[0][0] = x[i] * (nr[0] - kappa[i])
        xn[0][1] = x[i] * (nr[0] + kappa[i])
        xn[1][0] = x[i] * (nr[1] - kappa[i + 1])
        xn[1][1] = x[i] * (nr[1] + kappa[i + 1])
        if i:
            _interface(l, xn, z, mfull)
            zgemm(&c, &c, &four, &two, &four, &zone, &mfull[0][0], &four, &m[0][0], &four, &zzero, &mtmp[0][0], &four)
            memcpy(&m, &mtmp, 8 * sizeof(double complex))
        else:
            _innermost_interface(l, xn, z, m)
    cdef double complex det = 1 / (m[0][0] * m[1][1] - m[1][0] * m[0][1])
    res[0][0] = (m[0][2] * m[1][1] - m[1][2] * m[0][1]) * det
    res[0][1] = (-m[0][2] * m[1][0] + m[1][2] * m[0][0]) * det
    res[1][0] = (m[0][3] * m[1][1] - m[1][3] * m[0][1]) * det
    res[1][1] = (-m[0][3] * m[1][0]+ m[1][3] * m[0][0]) * det


cdef void loop_mie_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double complex res[2][2]
    cdef double epsilon[20]
    cdef double mu[20]
    cdef double kappa[20]
    cdef double x[19]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *op0 = args[5]
    for i in range(n):
        for j in range(dims[1]):
            x[j] = (<double*>(ip1 + j * steps[6]))[0]
            epsilon[j] = (<double*>(ip2 + j * steps[7]))[0]
            mu[j] = (<double*>(ip3 + j * steps[8]))[0]
            kappa[j] = (<double*>(ip4 + j * steps[9]))[0]
        epsilon[dims[1]] = (<double*>(ip2 + dims[1] * steps[7]))[0]
        mu[dims[1]] = (<double*>(ip3 + dims[1] * steps[8]))[0]
        kappa[dims[1]] = (<double*>(ip4 + dims[1] * steps[9]))[0]
        (<void (*)(long, double*, double*, double*, double*, long, double complex[2][2]) nogil>func)(<long>(<long*>ip0)[0], x, epsilon, mu, kappa, dims[1], res)
        for j in range(2):
            (<double complex*>(op0 + j * steps[10]))[0] = res[j][0]
            (<double complex*>(op0 + j * steps[10] + steps[11]))[0] = res[j][1]
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        op0 += steps[5]

cdef void loop_mie_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double complex res[2][2]
    cdef double complex epsilon[20]
    cdef double complex mu[20]
    cdef double complex kappa[20]
    cdef double x[19]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *op0 = args[5]
    for i in range(n):
        for j in range(dims[1]):
            x[j] = (<double*>(ip1 + j * steps[6]))[0]
            epsilon[j] = (<double complex*>(ip2 + j * steps[7]))[0]
            mu[j] = (<double complex*>(ip3 + j * steps[8]))[0]
            kappa[j] = (<double complex*>(ip4 + j * steps[9]))[0]
        epsilon[dims[1]] = (<double complex*>(ip2 + dims[1] * steps[7]))[0]
        mu[dims[1]] = (<double complex*>(ip3 + dims[1] * steps[8]))[0]
        kappa[dims[1]] = (<double complex*>(ip4 + dims[1] * steps[9]))[0]
        (<void (*)(long, double*, double complex*, double complex*, double complex*, long, double complex[2][2]) nogil>func)(<long>(<long*>ip0)[0], x, epsilon, mu, kappa, dims[1], res)
        for j in range(2):
            (<double complex*>(op0 + j * steps[10]))[0] = res[j][0]
            (<double complex*>(op0 + j * steps[10] + steps[11]))[0] = res[j][1]
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        op0 += steps[5]

np.import_array()
np.import_ufunc()

cdef np.PyUFuncGenericFunction gufunc_mie_loops[2]
cdef void *gufunc_mie_data[2]
cdef char gufunc_mie_types[2*6]

gufunc_mie_loops[0] = <np.PyUFuncGenericFunction>loop_mie_d
gufunc_mie_loops[1] = <np.PyUFuncGenericFunction>loop_mie_D
gufunc_mie_types[0] = <char>np.NPY_LONG
gufunc_mie_types[1] = <char>np.NPY_DOUBLE
gufunc_mie_types[2] = <char>np.NPY_DOUBLE
gufunc_mie_types[3] = <char>np.NPY_DOUBLE
gufunc_mie_types[4] = <char>np.NPY_DOUBLE
gufunc_mie_types[5] = <char>np.NPY_CDOUBLE
gufunc_mie_types[6] = <char>np.NPY_LONG
gufunc_mie_types[7] = <char>np.NPY_DOUBLE
gufunc_mie_types[8] = <char>np.NPY_CDOUBLE
gufunc_mie_types[9] = <char>np.NPY_CDOUBLE
gufunc_mie_types[10] = <char>np.NPY_CDOUBLE
gufunc_mie_types[11] = <char>np.NPY_CDOUBLE
gufunc_mie_data[0] = <void*>_mie[double]
gufunc_mie_data[1] = <void*>_mie[double_complex]

mie = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_mie_loops,
    gufunc_mie_data,
    gufunc_mie_types,
    2,  # number of supported input types
    5,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'mie',  # function name
    r"""
    mie(l, x, epsilon, mu, kappa)

    Mie coefficient of degree l in helicity basis

    The sphere is defined by its size parameter :math:`k_0 r`, where :math:`r` is the
    radius and :math:`k_0` the wave number in vacuum. A multilayered sphere is defined
    by giving an array of ascending numbers, that define the size parameters of the
    sphere and its shells starting from the center.
    Likewise, the material parameters are given from inside to outside. These arrays
    are expected to be exactly one unit larger then the array `x`.

    The result is an array relating incident light of negative (index `0`) and
    positive (index `1`) helicity with the scattered modes, which are index in the same
    way. The first dimension of the array are the scattered and the second dimension the
    incident modes.

    Args:
        l (integer): Degree :math:`l \geq 0`
        x (float, array_like): Size parameters
        epsilon (float or complex, array_like): Relative permittivity
        mu (float or complex, array_like): Relative permeability
        kappa (float or complex, array_like): Chirality parameter

    Returns:
        complex (2, 2)-array
    """,  # docstring
    0,  # unused
    '(),(a),(b),(b),(b)->(2,2)'  # signature
)

cdef void _fresnel(number_t ks[2][2], number_t kzs[2][2], number_t zs[2], number_t res[2][2][2][2]) nogil:
    # First dimension is the side, second the polarization
    cdef number_t ap[2]
    ap[0] = ks[0][0] * kzs[0][1] + ks[0][1] * kzs[0][0]
    ap[1] = ks[1][0] * kzs[1][1] + ks[1][1] * kzs[1][0]
    cdef number_t bp = ks[0][1] * kzs[1][1] + ks[1][1] * kzs[0][1]
    cdef number_t cp = ks[0][0] * kzs[1][0] + ks[1][0] * kzs[0][0]
    cdef number_t am[2]
    am[0] = ks[0][0] * kzs[0][1] - ks[0][1] * kzs[0][0]
    am[1] = ks[1][0] * kzs[1][1] - ks[1][1] * kzs[1][0]

    cdef number_t zs_diff = zs[0] - zs[1]
    cdef number_t zs_prod = 4 * zs[0] * zs[1]
    cdef number_t pref = 1 / (zs_diff * zs_diff * ap[0] * ap[1] + zs_prod * bp * cp)
    cdef long i, j
    for i in range(2):
        j = 1 - i
        res[i][i][0][0] = (zs[0] + zs[1]) * ks[j][0] * kzs[i][0] * bp * zs[j] * 4 * pref
        res[i][i][0][1] = -(
            (zs[i] - zs[j])
            * ks[j][0]
            * kzs[i][1]
            * (ks[j][1] * kzs[i][0] - ks[i][0] * kzs[j][1])
        ) * zs[j] * 4 * pref
        res[i][i][1][0] = -(
            (zs[i] - zs[j])
            * ks[j][1]
            * kzs[i][0]
            * (ks[j][0] * kzs[i][1] - ks[i][1] * kzs[j][0])
        ) * zs[j] * 4 * pref
        res[i][i][1][1] = (zs[0] + zs[1]) * ks[j][1] * kzs[i][1] * cp * zs[j] * 4 * pref

        res[j][i][0][0] = (
            -zs_diff * zs_diff * am[i] * ap[j] - zs_prod * bp
            * (ks[i][0] * kzs[j][0] - ks[j][0] * kzs[i][0])
        ) * pref
        res[j][i][0][1] = -(
            2 * (zs[j] * zs[j] - zs[i] * zs[i]) * ks[i][0] * kzs[i][1] * ap[j]
        ) * pref
        res[j][i][1][0] = -(
            2 * (zs[j] * zs[j] - zs[i] * zs[i]) * ks[i][1] * kzs[i][0] * ap[j]
        ) * pref
        res[j][i][1][1] = (
            zs_diff * zs_diff * am[i] * ap[j] - zs_prod * cp
            * (ks[i][1] * kzs[j][1] - ks[j][1] * kzs[i][1])
        ) * pref

cdef void loop_fresnel_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double res[2][2][2][2]
    cdef double ks[2][2]
    cdef double kzs[2][2]
    cdef double zs[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *op0 = args[3]
    for i in range(n):
        for j in range(2):
            for k in range(2):
                ks[j][k] = (<double*>(ip0 + j * steps[4] + k * steps[5]))[0]
                kzs[j][k] = (<double*>(ip1 + j * steps[6] + k * steps[7]))[0]
            zs[j] = (<double*>(ip2 + j * steps[8]))[0]
        (<void (*)(double[2][2], double[2][2], double[2], double[2][2][2][2]) nogil>func)(ks, kzs, zs, res)
        for j in range(2):
            for k in range(2):
                (<double*>(op0 + j * steps[9] + k * steps[10]))[0] = res[j][k][0][0]
                (<double*>(op0 + j * steps[9] + k * steps[10] + steps[12]))[0] = res[j][k][0][1]
                (<double*>(op0 + j * steps[9] + k * steps[10] + steps[11]))[0] = res[j][k][1][0]
                (<double*>(op0 + j * steps[9] + k * steps[10] + steps[11] + steps[12]))[0] = res[j][k][1][1]
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        op0 += steps[3]

cdef void loop_fresnel_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double complex res[2][2][2][2]
    cdef double complex ks[2][2]
    cdef double complex kzs[2][2]
    cdef double complex zs[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *op0 = args[3]
    for i in range(n):
        for j in range(2):
            for k in range(2):
                ks[j][k] = (<double complex*>(ip0 + j * steps[4] + k * steps[5]))[0]
                kzs[j][k] = (<double complex*>(ip1 + j * steps[6] + k * steps[7]))[0]
            zs[j] = (<double complex*>(ip2 + j * steps[8]))[0]
        (<double complex (*)(double complex[2][2], double complex[2][2], double complex[2], double complex[2][2][2][2]) nogil>func)(ks, kzs, zs, res)
        for j in range(2):
            for k in range(2):
                (<double complex*>(op0 + j * steps[9] + k * steps[10]))[0] = res[j][k][0][0]
                (<double complex*>(op0 + j * steps[9] + k * steps[10] + steps[12]))[0] = res[j][k][0][1]
                (<double complex*>(op0 + j * steps[9] + k * steps[10] + steps[11]))[0] = res[j][k][1][0]
                (<double complex*>(op0 + j * steps[9] + k * steps[10] + steps[11] + steps[12]))[0] = res[j][k][1][1]
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        op0 += steps[3]

cdef np.PyUFuncGenericFunction gufunc_fresnel_loops[2]
cdef void *gufunc_fresnel_data[2]
cdef char gufunc_fresnel_types[2*4]

gufunc_fresnel_loops[0] = <np.PyUFuncGenericFunction>loop_fresnel_d
gufunc_fresnel_loops[1] = <np.PyUFuncGenericFunction>loop_fresnel_D
gufunc_fresnel_types[0] = <char>np.NPY_DOUBLE
gufunc_fresnel_types[1] = <char>np.NPY_DOUBLE
gufunc_fresnel_types[2] = <char>np.NPY_DOUBLE
gufunc_fresnel_types[3] = <char>np.NPY_DOUBLE
gufunc_fresnel_types[4] = <char>np.NPY_CDOUBLE
gufunc_fresnel_types[5] = <char>np.NPY_CDOUBLE
gufunc_fresnel_types[6] = <char>np.NPY_CDOUBLE
gufunc_fresnel_types[7] = <char>np.NPY_CDOUBLE
gufunc_fresnel_data[0] = <void*>_fresnel[double]
gufunc_fresnel_data[1] = <void*>_fresnel[double_complex]

fresnel = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_fresnel_loops,
    gufunc_fresnel_data,
    gufunc_fresnel_types,
    2,  # number of supported input types
    3,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'fresnel',  # function name
    r"""
    fresnel(ks, kzs, zs)

    Fresnel coefficient for a planar interface

    The first dimension contains the numbers for the two media, the second dimenison
    indexes the polarizations.

    The result is an array relating incident light of negative (index `0`) and
    positive (index `1`) helicity with the scattered modes, which are index in the same
    way. The first dimension of the array are the scattered and the second dimension the
    incident modes.

    Args:
        ks (float or complex): Wave numbers
        ks (float): Z component of the waves
        zs (float or complex): Impedances

    Returns:
        complex (2, 2)-array
    """,  # docstring
    0,  # unused
    '(2,2),(2,2),(2)->(2,2,2,2)',  # signature
)

cdef void _fill_cyl(long m, double complex kz, double complex ks[2], double rho, double complex z, double complex res[4][4]) nogil:
    cdef double complex krho, x
    cdef double complex jphi[2]
    cdef double complex jz[2]
    cdef double complex hphi[2]
    cdef double complex hz[2]
    cdef long i
    for i in range(2):
        krho = csqrt(ks[i] * ks[i] - kz * kz)
        x = krho * rho
        jphi[i] = -krho * sc.jv(m, x) / ks[i]
        jz[i] = -kz * m * sc.jv(m, x) / (ks[i] * x) + cs.jv_d(m, x) * (1 - 2 * i)
        hphi[i] = -krho * sc.hankel1(m, x) / ks[i]
        hz[i] = -kz * m * sc.hankel1(m, x) / (ks[i] * x) + cs.hankel1_d(m, x) * (1 - 2 * i)

    for i in range(2):
        # Column major layout
        res[i][0] = jphi[i]
        res[i + 2][0] = hphi[i]
        res[i][1] = jz[i]
        res[i + 2][1] = hz[i]
        res[i][2] = jphi[i] * (2 * i - 1) / z
        res[i + 2][2] = hphi[i] * (2 * i - 1) / z
        res[i][3] = jz[i] * (2 * i - 1) / z
        res[i + 2][3] = hz[i] * (2 * i - 1) / z

cdef void _fill_cyl_init(long m, double complex kz, double complex ks[2], double rho, double complex z, double complex res[2][4]) nogil:
    cdef double complex krho, x
    cdef double complex jphi[2]
    cdef double complex jz[2]
    cdef long i
    for i in range(2):
        krho = csqrt(ks[i] * ks[i] - kz * kz)
        x = krho * rho
        jphi[i] = -krho * sc.jv(m, x) / ks[i]
        jz[i] = -kz * m * sc.jv(m, x) / (ks[i] * x) + cs.jv_d(m, x) * (1 - 2 * i)

    for i in range(2):
        # Column major layout
        res[i][0] = jphi[i]
        res[i][1] = jz[i]
        res[i][2] = jphi[i] * (2 * i - 1) / z
        res[i][3] = jz[i] * (2 * i - 1) / z

cdef void _mie_cyl(double kz, long m, double k0, double *radii, double complex *epsilon, double complex *mu, double complex *kappa, long n, double complex res[2][2]) nogil:
    cdef double complex mfull[4][4]
    cdef double complex mtmp[2][4]
    cdef double complex mres[2][4]  # Column major
    cdef long i
    cdef double complex ks[2]
    cdef double complex z, zone = 1, zzero = 0
    cdef int two = 2, four = 4, info
    cdef int ipiv[4]
    cdef char c = b'N'
    for i in range(n):
        if i:
            _fill_cyl(m, kz, ks, radii[i], z, mfull)
            zgemm(&c, &c, &four, &two, &four, &zone, &mfull[0][0], &four, &mres[0][0], &four, &zzero, &mtmp[0][0], &four)
            memcpy(&mres, &mtmp, 8 * sizeof(double complex))
        else:
            ks[0] = k0 * (csqrt(epsilon[i] * mu[i]) - kappa[i])
            ks[1] = k0 * (csqrt(epsilon[i] * mu[i]) + kappa[i])
            z = csqrt(mu[i] / epsilon[i])
            _fill_cyl_init(m, kz, ks, radii[i], z, mres)
        ks[0] = k0 * (csqrt(epsilon[i + 1] * mu[i + 1]) - kappa[i + 1])
        ks[1] = k0 * (csqrt(epsilon[i + 1] * mu[i + 1]) + kappa[i + 1])
        z = csqrt(mu[i + 1] / epsilon[i + 1])
        _fill_cyl(m, kz, ks, radii[i], z, mfull)
        zgesv(&four, &two, &mfull[0][0], &four, ipiv, &mres[0][0], &four, &info)
    cdef double complex det = 1 / (mres[0][0] * mres[1][1] - mres[0][1] * mres[1][0])
    res[0][0] = (mres[0][2] * mres[1][1] - mres[1][2] * mres[0][1]) * det
    res[0][1] = (-mres[0][2] * mres[1][0] + mres[1][2] * mres[0][0]) * det
    res[1][0] = (mres[0][3] * mres[1][1] - mres[1][3] * mres[0][1]) * det
    res[1][1] = (-mres[0][3] * mres[1][0] + mres[1][3] * mres[0][0]) * det


cdef void loop_mie_cyl_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double complex res[2][2]
    cdef double complex epsilon[20]
    cdef double complex mu[20]
    cdef double complex kappa[20]
    cdef double radii[19]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    for i in range(n):
        for j in range(dims[1]):
            radii[j] = (<double*>(ip3 + j * steps[8]))[0]
            epsilon[j] = (<double complex*>(ip4 + j * steps[9]))[0]
            mu[j] = (<double complex*>(ip5 + j * steps[10]))[0]
            kappa[j] = (<double complex*>(ip6 + j * steps[11]))[0]
        epsilon[dims[1]] = (<double complex*>(ip4 + dims[1] * steps[9]))[0]
        mu[dims[1]] = (<double complex*>(ip5 + dims[1] * steps[10]))[0]
        kappa[dims[1]] = (<double complex*>(ip6 + dims[1] * steps[11]))[0]
        (<void (*)(double, long, double, double*, double complex*, double complex*, double complex*, long, double complex[2][2]) nogil>func)(<double>(<double*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], radii, epsilon, mu, kappa, dims[1], res)
        for j in range(2):
            (<double complex*>(op0 + j * steps[12]))[0] = res[j][0]
            (<double complex*>(op0 + j * steps[12] + steps[13]))[0] = res[j][1]
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]

cdef np.PyUFuncGenericFunction gufunc_mie_cyl_loops[1]
cdef void *gufunc_mie_cyl_data[1]
cdef char gufunc_mie_cyl_types[8]

gufunc_mie_cyl_loops[0] = <np.PyUFuncGenericFunction>loop_mie_cyl_D
gufunc_mie_cyl_types[0] = <char>np.NPY_DOUBLE
gufunc_mie_cyl_types[1] = <char>np.NPY_LONG
gufunc_mie_cyl_types[2] = <char>np.NPY_DOUBLE
gufunc_mie_cyl_types[3] = <char>np.NPY_DOUBLE
gufunc_mie_cyl_types[4] = <char>np.NPY_CDOUBLE
gufunc_mie_cyl_types[5] = <char>np.NPY_CDOUBLE
gufunc_mie_cyl_types[6] = <char>np.NPY_CDOUBLE
gufunc_mie_cyl_types[7] = <char>np.NPY_CDOUBLE
gufunc_mie_cyl_data[0] = <void*>_mie_cyl

mie_cyl = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_mie_cyl_loops,
    gufunc_mie_cyl_data,
    gufunc_mie_cyl_types,
    1,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'mie_cyl',  # function name
    r"""
    mie_cyl(kz, m, k0, radii, epsilon, mu, kappa)

    Coefficient for scattering at an infinite cylinder in helicity basis

    The cylinder is defined by its radii :math:`\rho`. A multilayered cylinder is defined
    by giving an array of ascending numbers, that define its shells starting from the center.
    Likewise, the material parameters are given from inside to outside. These arrays
    are expected to be exactly one unit larger then the array `x`.

    The result is an array relating incident light of negative (index `0`) and
    positive (index `1`) helicity with the scattered modes, which are index in the same
    way. The first dimension of the array are the scattered and the second dimension the
    incident modes.

    Args:
        kz (float): Z component of the wave
        m (integer): Order
        k0 (float or complex): Wave number in vacuum
        radii (float, array_like): Size parameters
        epsilon (float or complex, array_like): Relative permittivity
        mu (float or complex, array_like): Relative permeability
        kappa (float or complex, array_like): Chirality parameter

    Returns:
        complex (2, 2)-array
    """,  # docstring
    0,  # unused
    '(),(),(),(a),(b),(b),(b)->(2,2)',  # signature
)
