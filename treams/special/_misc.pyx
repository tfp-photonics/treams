"""Miscellaneous functions for treams.special"""

from libc.math cimport cos as cosd
from libc.math cimport exp as expd
from libc.math cimport fabs, pi
from libc.math cimport pow as powd
from libc.math cimport sin as sind
from libc.math cimport sqrt as sqrtd


cdef extern from "<complex.h>" nogil:
    double cabs(double complex z)
    double complex ccos(double complex z)
    double complex cexp(double complex z)
    double complex cpow(double complex x, double complex y)
    double complex csin(double complex z)
    double complex csqrt(double complex z)


cdef double SQPI = sqrtd(pi)


cdef number_t sin(number_t x) nogil:
    """Fused type version of sin"""
    if number_t is double:
        return sind(x)
    elif number_t is double_complex:
        return csin(x)


cdef number_t cos(number_t x) nogil:
    """Fused type version of cos"""
    if number_t is double:
        return cosd(x)
    elif number_t is double_complex:
        return ccos(x)


cdef number_t exp(number_t x) nogil:
    """Fused type version of exp"""
    if number_t is double:
        return expd(x)
    elif number_t is double_complex:
        return cexp(x)


cdef number_t pow(number_t x, number_t y) nogil:
    """Fused type version of pow"""
    if number_t is double:
        return powd(x, y)
    elif number_t is double_complex:
        return cpow(x, y)


cdef number_t sqrt(number_t x) nogil:
    """Fused type version of sqrt"""
    if number_t is double:
        return sqrtd(x)
    elif number_t is double_complex:
        return csqrt(x)


cdef double abs(number_t x) nogil:
    """Fused type version of abs"""
    if number_t is double:
        return fabs(x)
    elif number_t is double_complex:
        return cabs(x)


cdef long minusonepow(long l) nogil:
    """Minus one to an integer power"""
    if l % 2 == 0:
        return 1
    return -1


cdef long array_zero(numeric *a, long n) nogil:
    """Check if all array entries are zero"""
    cdef long i
    for i in range(n):
        if a[i] != 0:
            return 0
    return 1


cdef long ipow(long base, long exponent) nogil:
    """Power function for integer base and non-negative integer exponent"""
    if exponent < 0:
        raise ValueError("negative exponent")
    cdef long res = 1
    while True:
        if exponent & 1:
            res *= base
        exponent >>= 1
        if exponent == 0:
            break
        base *= base
    return res
