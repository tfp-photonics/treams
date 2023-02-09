"""Cython versions of special functions.

This module defines the public interface of the implemented Cython functions that are
underlying the ufuncs of the module.
"""

cimport scipy.special.cython_special as cs

from ptsa.special cimport _bessel, _coord, _integrals, _waves, _wigner3j, _wignerd
from ptsa.special._misc cimport double_complex


cpdef double complex hankel1_d(double n, double complex z) nogil:
    """hankel1_d(n, z)

    See the documentation for :func:`ptsa.special.hankel1_d`.
    """
    return _bessel.hankel1_d(n, z)
cpdef double complex hankel2_d(double n, double complex z) nogil:
    """hankel2_d(n, z)

    See the documentation for :func:`ptsa.special.hankel2_d`.
    """
    return _bessel.hankel2_d(n, z)
cpdef number_t jv_d(double n, number_t z) nogil:
    """jv_d(n, z)

    See the documentation for :func:`ptsa.special.jv_d`.
    """
    return _bessel.jv_d(n, z)
cpdef double complex spherical_hankel1(double n, number_t z) nogil:
    """spherical_hankel1(n, z)

    See the documentation for :func:`ptsa.special.spherical_hankel1`.
    """
    return _bessel.spherical_hankel1(n, z)
cpdef double complex spherical_hankel1_d(double n, double complex z) nogil:
    """spherical_hankel1_d(n, z)

    See the documentation for :func:`ptsa.special.spherical_hankel1_d`.
    """
    return _bessel.spherical_hankel1_d(n, z)
cpdef double complex spherical_hankel2(double n, number_t z) nogil:
    """spherical_hankel2(n, z)

    See the documentation for :func:`ptsa.special.spherical_hankel2`.
    """
    return _bessel.spherical_hankel2(n, z)
cpdef double complex spherical_hankel2_d(double n, double complex z) nogil:
    """spherical_hankel2_d(n, z)

    See the documentation for :func:`ptsa.special.spherical_hankel2_d`.
    """
    return _bessel.spherical_hankel2_d(n, z)
cpdef number_t yv_d(double n, number_t z) nogil:
    """yv_d(n, z)

    See the documentation for :func:`ptsa.special.yv_d`.
    """
    return _bessel.yv_d(n, z)


cdef void car2cyl(double *input, double *output) nogil:
    """See the documentation for :func:`ptsa.special.car2cyl`."""
    _coord.car2cyl(input, output, 1, 1)
cdef void car2pol(double *input, double *output) nogil:
    """See the documentation for :func:`ptsa.special.car2pol`."""
    _coord.car2pol(input, output, 1, 1)
cdef void car2sph(double *input, double *output) nogil:
    """See the documentation for :func:`ptsa.special.car2sph`."""
    _coord.car2sph(input, output, 1, 1)
cdef void cyl2car(double *input, double *output) nogil:
    """See the documentation for :func:`ptsa.special.cyl2car`."""
    _coord.cyl2car(input, output, 1, 1)
cdef void cyl2sph(double *input, double *output) nogil:
    """See the documentation for :func:`ptsa.special.cyl2sph`."""
    _coord.cyl2sph(input, output, 1, 1)
cdef void pol2car(double *input, double *output) nogil:
    """See the documentation for :func:`ptsa.special.pol2car`."""
    _coord.pol2car(input, output, 1, 1)
cdef void sph2car(double *input, double *output) nogil:
    """See the documentation for :func:`ptsa.special.sph2car`."""
    _coord.sph2car(input, output, 1, 1)
cdef void sph2cyl(double *input, double *output) nogil:
    """See the documentation for :func:`ptsa.special.sph2cyl`."""
    _coord.sph2cyl(input, output, 1, 1)
cdef void vcar2cyl(number_t *iv, double *ip, number_t *output) nogil:
    """See the documentation for :func:`ptsa.special.vcar2cyl`."""
    _coord.vcar2cyl(iv, ip, output, 1, 1, 1)
cdef void vcar2pol(number_t *iv, double *ip, number_t *output) nogil:
    """See the documentation for :func:`ptsa.special.vcar2pol`."""
    _coord.vcar2pol(iv, ip, output, 1, 1, 1)
cdef void vcar2sph(number_t *iv, double *ip, number_t *output) nogil:
    """See the documentation for :func:`ptsa.special.vcar2sph`."""
    _coord.vcar2sph(iv, ip, output, 1, 1, 1)
cdef void vcyl2car(number_t *iv, double *ip, number_t *output) nogil:
    """See the documentation for :func:`ptsa.special.vcyl2car`."""
    _coord.vcyl2car(iv, ip, output, 1, 1, 1)
cdef void vcyl2sph(number_t *iv, double *ip, number_t *output) nogil:
    """See the documentation for :func:`ptsa.special.vcyl2sph`."""
    _coord.vcyl2sph(iv, ip, output, 1, 1, 1)
cdef void vpol2car(number_t *iv, double *ip, number_t *output) nogil:
    """See the documentation for :func:`ptsa.special.vpol2car`."""
    _coord.vpol2car(iv, ip, output, 1, 1, 1)
cdef void vsph2car(number_t *iv, double *ip, number_t *output) nogil:
    """See the documentation for :func:`ptsa.special.vsph2car`."""
    _coord.vsph2car(iv, ip, output, 1, 1, 1)
cdef void vsph2cyl(number_t *iv, double *ip, number_t *output) nogil:
    """See the documentation for :func:`ptsa.special.vsph2cyl`."""
    _coord.vsph2cyl(iv, ip, output, 1, 1, 1)


cpdef number_t incgamma(double n, number_t z) nogil:
    """incgamma(n, z)

    See the documentation for :func:`ptsa.special.incgamma`.
    """
    return _integrals.incgamma(n, z)
cpdef number_t intkambe(long n, number_t z, number_t eta) nogil:
    """intkambe(n, z, eta)

    See the documentation for :func:`ptsa.special.intkambe`.
    """
    return _integrals.intkambe(n, z, eta)


cpdef number_t lpmv(double m, double l, number_t z) nogil:
    """lpmv(m, l, z)

    See the documentation for :func:`ptsa.special.lpmv`.
    """
    return _waves.lpmv(m, l, z)
cpdef number_t pi_fun(double l, double m, number_t x) nogil:
    """pi_fun(l, m, theta)

    See the documentation for :func:`ptsa.special.pi_fun`.
    """
    return _waves.pi_fun(l, m, x)
cpdef double complex sph_harm(double m, double l, double phi, number_t theta) nogil:
    """sph_harm(m, l, phi, theta)

    See the documentation for :func:`ptsa.special.sph_harm`.
    """
    return _waves.sph_harm(m, l, phi, theta)
cpdef number_t tau_fun(double l, double m, number_t x) nogil:
    """tau_fun(l, m, theta)

    See the documentation for :func:`ptsa.special.tau_fun`.
    """
    return _waves.tau_fun(l, m, x)

cpdef double complex _tl_vsw_helper(long l, long m, long lambda_, long mu, long p, long q) nogil:
    """Helper function for the translation coefficient of vector spherical waves."""
    return _waves._tl_vsw_helper(l, m, lambda_, mu, p, q)

cpdef double complex tl_vcw(double kz1, long mu, double kz2, long m, double complex krr, double phi, double z) nogil:
    """tl_vcw(kz1, mu, kz2, m, xrho, phi, z)

    See the documentation for :func:`ptsa.special.tl_vcw`.
    """
    return _waves.tl_vcw(kz1, mu, kz2, m, krr, phi, z)
cpdef double complex tl_vcw_r(double kz1, long mu, double kz2, long m, number_t krr, double phi, double z) nogil:
    """tl_vdw_r(kz1, mu, kz2, m, xrho, phi, z)

    See the documentation for :func:`ptsa.special.tl_vcw_r`.
    """
    return _waves.tl_vcw_r(kz1, mu, kz2, m, krr, phi, z)

cpdef double complex tl_vsw_A(long lambda_, long mu, long l, long m, double complex kr, number_t theta, double phi) nogil:
    """tl_vsw_A(lambda, mu, l, m, x, theta, phi)

    See the documentation for :func:`ptsa.special.tl_vsw_A`.
    """
    return _waves.tl_vsw_A(lambda_, mu, l, m, kr, theta, phi)
cpdef double complex tl_vsw_B(long lambda_, long mu, long l, long m, double complex kr, number_t theta, double phi) nogil:
    """tl_vsw_B(lambda, mu, l, m, x, theta, phi)

    See the documentation for :func:`ptsa.special.tl_vsw_B`.
    """
    return _waves.tl_vsw_B(lambda_, mu, l, m, kr, theta, phi)
cpdef double complex tl_vsw_rA(long lambda_, long mu, long l, long m, number_t kr, number_t theta, double phi) nogil:
    """tl_vsw_rA(lambda, mu, l, m, x, theta, phi)

    See the documentation for :func:`ptsa.special.tl_vsw_rA`.
    """
    return _waves.tl_vsw_rA(lambda_, mu, l, m, kr, theta, phi)
cpdef double complex tl_vsw_rB(long lambda_, long mu, long l, long m, number_t kr, number_t theta, double phi) nogil:
    """tl_vsw_rB(lambda, mu, l, m, x, theta, phi)

    See the documentation for :func:`ptsa.special.tl_vsw_rB`.
    """
    return _waves.tl_vsw_rB(lambda_, mu, l, m, kr, theta, phi)

cdef void vcw_A(double kz, long m, double complex krr, double phi, double z, double complex k, long pol, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vcw_A`."""
    _waves.vcw_A(kz, m, krr, phi, z, k, pol, out, 1)
cdef void vcw_M(double kz, long m, double complex krr, double phi, double z, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vcw_M`."""
    _waves.vcw_M(kz, m, krr, phi, z, out, 1)
cdef void vcw_N(double kz, long m, double complex krr, double phi, double z, double complex k, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vcw_N`."""
    _waves.vcw_N(kz, m, krr, phi, z, k, out, 1)
cdef void vcw_rA(double kz, long m, number_t krr, double phi, double z, double complex k, long pol, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vcw_rA`."""
    _waves.vcw_rA(kz, m, krr, phi, z, k, pol, out, 1)
cdef void vcw_rM(double kz, long m, number_t krr, double phi, double z, double complex *out, long i) nogil:
    """See the documentation for :func:`ptsa.special.vcw_rM`."""
    _waves.vcw_rM(kz, m, krr, phi, z, out, 1)
cdef void vcw_rN(double kz, long m, number_t krr, double phi, double z, double complex k, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vcw_rN`."""
    _waves.vcw_rN(kz, m, krr, phi, z, k, out, 1)

cdef void vpw_A(number_t kx, number_t ky, number_t kz, double x, double y, double z, long pol, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vpw_A`."""
    _waves.vpw_A(kx, ky, kz, x, y, z, pol, out, 1)
cdef void vpw_M(number_t kx, number_t ky, number_t kz, double x, double y, double z, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vpw_M`."""
    _waves.vpw_M(kx, ky, kz, x, y, z, out, 1)
cdef void vpw_N(number_t kx, number_t ky, number_t kz, double x, double y, double z, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vpw_N`."""
    _waves.vpw_N(kx, ky, kz, x, y, z, out, 1)

cdef void vsh_X(long l, long m, number_t theta, double phi, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vsh_X`."""
    _waves.vsh_X(l, m, theta, phi, out, 1)
cdef void vsh_Y(long l, long m, number_t theta, double phi, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vsh_Y`."""
    _waves.vsh_Y(l, m, theta, phi, out, 1)
cdef void vsh_Z(long l, long m, number_t theta, double phi, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vsh_Z`."""
    _waves.vsh_Z(l, m, theta, phi, out, 1)

cdef void vsw_A(long l, long m, double complex kr, number_t theta, double phi, long pol, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vsw_A`."""
    _waves.vsw_A(l, m, kr, theta, phi, pol, out, 1)
cdef void vsw_M(long l, long m, double complex kr, number_t theta, double phi, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vsw_M`."""
    _waves.vsw_M(l, m, kr, theta, phi, out, 1)
cdef void vsw_N(long l, long m, double complex kr, number_t theta, double phi, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vsw_N`."""
    _waves.vsw_N(l, m, kr, theta, phi, out, 1)
cdef void vsw_rA(long l, long m, number_t kr, number_t theta, double phi, long pol, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vsw_rA`."""
    _waves.vsw_rA(l, m, kr, theta, phi, pol, out, 1)
cdef void vsw_rM(long l, long m, number_t kr, number_t theta, double phi, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vsw_rM`."""
    _waves.vsw_rM(l, m, kr, theta, phi, out, 1)
cdef void vsw_rN(long l, long m, number_t kr, number_t theta, double phi, double complex *out) nogil:
    """See the documentation for :func:`ptsa.special.vsw_rN`."""
    _waves.vsw_rN(l, m, kr, theta, phi, out, 1)


cpdef double wigner3j(long j1, long j2, long j3, long m1, long m2, long m3) nogil:
    """wigner3j(j1, j2, j3, m1, m2, m3)

    See the documentation for :func:`ptsa.special.wigner3j`.
    """
    return _wigner3j.wigner3j(j1, j2, j3, m1, m2, m3)


cpdef number_t wignersmalld(long l, long m, long k, number_t theta) nogil:
    """wignersmalld(l, m, k, theta)

    See the documentation for :func:`ptsa.special.wignersmalld`.
    """
    return _wignerd.wignersmalld(l, m, k, theta)
cpdef double complex wignerd(long l, long m, long k, double phi, number_t theta, double psi) nogil:
    """wignerd(l, m, k, phi, theta, psi)

    See the documentation for :func:`ptsa.special.wignerd`.
    """
    return _wignerd.wignerd(l, m, k, phi, theta, psi)
