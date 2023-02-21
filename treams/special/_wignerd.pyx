"""Wigner D-matrix elements"""

from libc.math cimport M_LN2, NAN, exp, isnan, lgamma, pi
from libc.math cimport sqrt as sqrtd
from libc.stdlib cimport free, labs, malloc
from scipy.special.cython_special cimport lpmv

from treams.special._misc cimport cos, double_complex, minusonepow, pow, sqrt
from treams.special._waves cimport clpmv


# The preprocessor directives correct the missing macro in mingw-w64 by
# substituting it with a corresponding function that is defined by cython anyway
cdef extern from "<complex.h>" nogil:
    """
    #ifndef CMPLX
    #define CMPLX __pyx_t_double_complex_from_parts
    #endif
    """
    double complex cexp(double complex z)
    double creal(double complex z)
    double cimag(double complex z)
    double cabs(double complex z)
    double complex CMPLX(double, double)


cdef number_t _wignerdforward(long l, long m, long k, number_t x, number_t *cache) nogil:
    """
    Forward recursion for Wigner d-matrix elements

    See equations (18) to (20) in [#]_.

    References:
        .. [#] `H. Dachsela, J. Chem. Phys. 142, 144115 (2006) <https://doi.org/10.1063/1.2194548>`_
    """
    cdef long pos = (2 * l + 1) * m + k + l
    if not isnan(creal(cache[pos])):
        return cache[pos]
    cdef number_t res
    if m == 0:  # Essentially Legendre polynomials
        if number_t is double:
            res = exp((lgamma(l - k + 1) - lgamma(l + k + 1)) * 0.5) * lpmv(k, l, x)
        elif number_t is double_complex:
            res = exp((lgamma(l - k + 1) - lgamma(l + k + 1)) * 0.5) * clpmv(k, l, x)
        cache[pos] = res
        return res
    cdef number_t y = sqrt(1 - x * x)
    if k == -l:
        res = (
            (l - m + 1)
            * y
            * _wignerdforward(l, m - 1, k, x, cache)
            / (sqrtd(l * (l + 1) - (m - 1) * m) * (1 + x))
        )
        cache[pos] = res
        return res
    cdef number_t a = (
        sqrtd(l * (l + 1) - k * (k - 1))
        * _wignerdforward(l, m - 1, k - 1, x, cache)
    )
    cdef number_t b = (m - 1 + k) * y * _wignerdforward(l, m - 1, k, x, cache) / (1 + x)
    res = (a - b) / sqrtd(l * (l + 1) - (m - 1) * m)
    cache[pos] = res
    return res


cdef number_t _wignerdlml(long l, long m, number_t x) nogil:
    """
    Wigner d-matrix element d^l_{ml}

    See equation (28) in [#]_.

    References:
        .. [#] `H. Dachsela, J. Chem. Phys. 142, 144115 (2006) <https://doi.org/10.1063/1.2194548>`_
    """
    return (
        minusonepow(l + m)
        * exp(
            -l * M_LN2
            + (lgamma(2 * l + 1) - lgamma(l - m + 1) - lgamma(l + m + 1)) * 0.5
        )
        * pow(1 + x, m)
        * pow(sqrt(1 - x * x), l - m)
    )


cdef number_t _wignerdbackward(long l, long m, long k, number_t x, number_t *cache) nogil:
    """
    Backward recursion for Wigner d-matrix elements

    See equations (25) to (27) in [#]_

    References:
        .. [#] `H. Dachsela, J. Chem. Phys. 142, 144115 (2006) <https://doi.org/10.1063/1.2194548>`_
    """
    cdef long pos = (2 * l + 1) * m + k + l
    if not isnan(creal(cache[pos])):
        return cache[pos]
    cdef number_t res
    if l == k:
        res = _wignerdlml(l, m, x)
        cache[pos] = res
        return res
    cdef number_t y = sqrt(1 - x * x)
    if l == m:
        res = (
            (l + k + 1)
            * y
            * _wignerdbackward(l, l, k + 1, x, cache)
            / (sqrtd(l * (l + 1) - k * (k + 1)) * (1 + x))
        )
        cache[pos] = res
        return res
    cdef number_t a = sqrtd(l * (l + 1) - m * (m + 1)) * _wignerdbackward(l, m + 1, k + 1, x, cache)
    cdef number_t b = (m + k + 1) * y * _wignerdbackward(l, m, k + 1, x, cache) / (1 + x)
    res = (a + b) / sqrtd(l * (l + 1) - k * (k + 1))
    cache[pos] = res
    return res


cdef bint _wignerd_f_or_b(long l, long m, long k) nogil:
    """Decision between forward and backward recursion by simple estimates"""
    cdef long forward = ((m + 1) * (m + 2)) // 2
    if l < m - k:
        forward -= (m - l - k) * (m - l - k + 1) // 2
    cdef long backward = ((l - k + 1) * (l - k + 2)) // 2
    if k < m:
        backward -= (m - k) * (m - k + 1) // 2
    return forward <= backward


cdef number_t wignersmalld(long l, long m, long k, number_t theta) nogil:
    r"""
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
    """
    if l < 0:
        raise ValueError(f"{l} is not non-negative")
    if labs(m) > l or labs(k) > l:
        return 0.0
    # There is an issue with the sign convention: Dachsela and Mathematica use one and
    # Yarshalovich, Wikipedia use the other. We want the latter convention. The
    # difference is essentially taking the opposite angle:
    theta = -theta
    # From now on, everything is according to Dachsela
    cdef double thetar = creal(theta) % (2 * pi)
    if thetar < 0:
        thetar += 2 * pi
    if number_t is double_complex:
        theta = CMPLX(thetar, cimag(theta))
    elif number_t is double:
        theta = thetar
    if theta == 0:
        if k == m:
            return 1.0
        return 0.0
    if cabs(theta - pi) < 1e-16:
        if k == -m:
            return minusonepow(l + k)
        return 0.0

    cdef number_t *cache = <number_t*>malloc((l + 1) * (2 * l + 1) * sizeof(number_t))
    cdef long i
    for i in range((l + 1) * (2 * l + 1)):
        cache[i] = <double>NAN

    # Recursion only for m >= 0, so swap sign if necessary
    cdef double pref_swap = 1
    if m < 0:
        pref_swap = minusonepow(m + k)
        m = -m
        k = -k
    # Shift argument to interval (0, pi/2)
    cdef double pref_shift = 1
    if 0.5 * pi <= creal(theta) < pi:
        pref_shift = minusonepow(l + m)
        k = -k
        theta = pi - theta
    elif pi <= creal(theta) < 1.5 * pi:
        pref_shift = minusonepow(l + k)
        k = -k
        theta = theta - pi
    elif 1.5 * pi <= creal(theta) < 2 * pi:  # TODO: < 2pi unnecessary ?
        pref_shift = minusonepow(m + k)
        theta = 2 * pi - theta
    cdef double pref_pure_complex = 1
    if creal(theta) == 0 and cimag(theta) < 0:
        pref_pure_complex = minusonepow(m + k)
    cdef number_t res
    if _wignerd_f_or_b(l, m, k):
        res = _wignerdforward(l, m, k, cos(theta), cache)
    else:
        res = _wignerdbackward(l, m, k, cos(theta), cache)
    free(cache)
    return pref_shift * pref_swap * pref_pure_complex * res


cdef double complex wignerd(
    long l,
    long m,
    long k,
    double phi,
    number_t theta,
    double psi
) nogil:
    r"""
    Wigner-D symbol

    .. math::

        D^l_{mk}(\varphi, \theta, \psi) = \mathrm e^{-\mathrm i m \varphi} d^l_{mk}(\theta) \mathrm e^{-\mathrm i k \psi}

    Note:
        Mathematica use a different sign convention, which means taking the negative
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
    """
    return cexp(-1j * (phi * m + psi * k)) * wignersmalld(l, m, k, theta)
