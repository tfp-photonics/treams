"""
(Spherical) Bessel and Hankel functions including their derivatives

Additions to the functions available in Scipy
"""

cimport scipy.special.cython_special as cs
from libc.math cimport pi

from treams.special._misc cimport sqrt


cdef double complex hankel1_d(double l, double complex z) nogil:
    """Derivative of the Hankel function of the first kind"""
    if l == 0:
        return -cs.hankel1(1, z)
    return 0.5 * (cs.hankel1(l - 1, z) - cs.hankel1(l + 1, z))


cdef double complex hankel2_d(double l, double complex z) nogil:
    """Derivative of the Hankel function of the second kind"""
    if l == 0:
        return -cs.hankel2(1, z)
    return 0.5 * (cs.hankel2(l - 1, z) - cs.hankel2(l + 1, z))


cdef number_t jv_d(double l, number_t z) nogil:
    """Derivative of the Bessel function of the first kind"""
    if l == 0:
        return -cs.jv(1, z)
    return 0.5 * (cs.jv(l - 1, z) - cs.jv(l + 1, z))


cdef double complex spherical_hankel1(double n, double complex z) nogil:
    """Spherical Hankel function of the first kind"""
    return sqrt(pi / (2 * z)) * cs.hankel1(n + 0.5, z)


cdef double complex spherical_hankel1_d(double l, double complex z) nogil:
    """Derivative of the spherical Hankel function of the first kind"""
    if l == 0:
        return -spherical_hankel1(1, z)
    return l * spherical_hankel1(l, z) / z - spherical_hankel1(l + 1, z)


cdef double complex spherical_hankel2(double n, double complex z) nogil:
    """Spherical Hankel function of the second kind"""
    return sqrt(pi / (2 * z)) * cs.hankel2(n + 0.5, z)


cdef double complex spherical_hankel2_d(double l, double complex z) nogil:
    """Derivative of the spherical Hankel function of the second kind"""
    if l == 0:
        return -spherical_hankel2(1, z)
    return l * spherical_hankel2(l, z) / z - spherical_hankel2(l + 1, z)


cdef number_t yv_d(double l, number_t z) nogil:
    """Derivative of the Bessel function of the second kind"""
    if l == 0:
        return -cs.yv(1, z)
    return 0.5 * (cs.yv(l - 1, z) - cs.yv(l + 1, z))
