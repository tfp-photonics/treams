"""Numerical evaluation of integrals"""

cimport scipy.special.cython_special as cs
from libc.math cimport M_SQRT1_2, NAN
from libc.math cimport exp as expd
from libc.math cimport isnan, lgamma
from libc.stdlib cimport free, labs, malloc
from numpy.math cimport INFINITY, NAN

from treams.special._misc cimport SQPI, double_complex, exp, pow, sqrt


cdef extern from "<complex.h>" nogil:
    double creal(double complex x)
    double cimag(double complex x)


cdef number_t incgamma(double n, number_t z) nogil:
    r"""
    incgamma(n, z)

    Upper incomplete Gamma function of integer and half-integer degree and real and complex
    argument

    This function is defined as

    .. math

        \Gamma(n, z) = \int_z^\infty t^{n - 1} \mathrm e^{-t} \mathrm dt

    The negative real axis is the branch cut of the implemented function.

    References:
        - `DLMF: 8.2 <https://dlmf.nist.gov/8.2>`
    """
    cdef long twicen = <long>(2 * n)
    if twicen != 2 * n:
        raise ValueError(f"{n} is neither integer nor half-integer")
    if z == 0:
        if twicen <= 0:
            return INFINITY
        else:
            return expd(lgamma(n))
    if twicen == 0:
        return cs.exp1(z)
    if twicen == 2:
        return exp(-z)
    if twicen == 1:
        return SQPI * cs.erfc(sqrt(z))
    if twicen > 2:
        return (n - 1) * incgamma(n - 1, z) + pow(z, n - 1) * exp(-z)
    return (incgamma(n + 1, z) - pow(z, n) * exp(-z)) / n


cdef number_t _intkambe_m3(number_t z, number_t eta) nogil:
    r"""Kambe integral of order -3

    Implements

    .. math::

       I_{-3}(z, \eta) = \sum_{n = 0}^\infty \frac{1}{n!}
       \left(\frac{z^2}{4}\right)^{n + 1}
       \Gamma(-n - 1, \frac{z^2 \eta^2}{2})
    """
    cdef number_t acc = 0
    cdef number_t mult = z * z * 0.25
    cdef number_t macc = mult
    cdef long mu
    for mu in range(21):  # TODO: optimize loop
        acc += macc * incgamma(-1 - mu, z * z * eta * eta * 0.5)
        macc *= mult / (mu + 1)
    return acc


cdef number_t _intkambe_m1(number_t z, number_t eta) nogil:
    r"""Kambe integral of order -1

    Implements

    .. math::

       I_{-1}(z, \eta) = \frac{1}{2} \sum_{n = 0}^\infty \frac{1}{n!}
       \left(\frac{z^2}{4}\right)^{n}
       \Gamma(-n, \frac{z^2 \eta^2}{2})
    """
    cdef number_t acc = 0
    cdef number_t mult = z * z * 0.25
    cdef number_t macc = 0.5
    cdef long mu
    for mu in range(21):  # TODO: optimize loop
        acc += macc * incgamma(-mu, z * z * eta * eta * 0.5)
        macc *= mult / (mu + 1)
    return acc


cdef double complex _cintkambe_m2(double_complex z, double complex eta) nogil:
    """Kambe integral of order -2 for complex double"""
    if creal(z) < 0:
        z = -z
    cdef double complex faddp = cs.erfc((z * eta - 1j / eta) * M_SQRT1_2) * exp(-1j * z)
    cdef double complex faddm = cs.erfc((z * eta + 1j / eta) * M_SQRT1_2) * exp(1j * z)
    return -0.5j * (faddp - faddm) * SQPI * M_SQRT1_2


cdef double _dintkambe_m2(double z, double eta) nogil:
    """Kambe integral of order -2 for double"""
    if z < 0:
        z = -z
    cdef double faddp = cimag(cs.erfc((z * eta - 1j / eta) * M_SQRT1_2) * exp(-1j * z))
    cdef double faddm = cimag(cs.erfc((z * eta + 1j / eta) * M_SQRT1_2) * exp(1j * z))
    return 0.5 * (faddp - faddm) * SQPI * M_SQRT1_2


cdef number_t _intkambe_m2(number_t z, number_t eta) nogil:
    """Kambe integral of order 0 for fused type"""
    if number_t is double:
        return _dintkambe_m2(z, eta)
    elif number_t is double_complex:
        return _cintkambe_m2(z, eta)


cdef double complex _cintkambe_0(double complex z, double complex eta) nogil:
    """Kambe integral of order 0 for complex double"""
    if creal(z) < 0:
        z = -z
    cdef double complex faddp = cs.erfc((z * eta - 1j / eta) * M_SQRT1_2) * exp(-1j * z)
    cdef double complex faddm = cs.erfc((z * eta + 1j / eta) * M_SQRT1_2) * exp(1j * z)
    return (faddp + faddm) * SQPI * 0.5 * M_SQRT1_2 / z


cdef double _dintkambe_0(double z, double eta) nogil:
    """Kambe integral of order 0 for double"""
    if z < 0:  # TODO: necessary?
        z = -z
    cdef double faddp = creal(cs.erfc((z * eta - 1j / eta) * M_SQRT1_2) * exp(-1j * z))
    cdef double faddm = creal(cs.erfc((z * eta + 1j / eta) * M_SQRT1_2) * exp(1j * z))
    return (faddp + faddm) * SQPI * 0.5 * M_SQRT1_2 / z


cdef number_t _intkambe_0(number_t z, number_t eta) nogil:
    """Kambe integral of order 0 for fused type"""
    if number_t is double:
        return _dintkambe_0(z, eta)
    elif number_t is double_complex:
        return _cintkambe_0(z, eta)


cdef number_t _intkambe(long n, number_t z, number_t eta, number_t *cache) nogil:
    r"""
    Cached recursion for the Kambe integral

    Args:
        n (integer): Order
        z (float or complex): Argument
        eta (float or complex): Integral cutoff
        cache (float or complex, array): Cache of at least size `|n| + 1`

    Returns:
        float or complex, can return complex values for real argument

    References:
        .. [#] `K. Kambe, Zeitschrift Fuer Naturforschung A 23, 9 (1968). <https://doi.org/10.1515/zna-1968-0908>`_
    """
    if not isnan(creal(cache[labs(n)])):
        return cache[labs(n)]
    if eta == 0:
        cache[labs(n)] = INFINITY
    # n < -4 is handled regularly
    elif z == 0 and n == -4:
        cache[labs(n)] = -intkambe(-2, 0, eta) + exp(0.5 / (eta * eta)) / eta
    elif z == 0 and n == -3:
        cache[labs(n)] = exp(1 / (2 * eta * eta)) - 1
    # n == -2 is handled regularly
    elif z == 0 and n > -2:
        cache[labs(n)] = INFINITY
    elif n == -3:
        cache[labs(n)] = _intkambe_m3(z, eta)
    elif n == -2:
        cache[labs(n)] = _intkambe_m2(z, eta)
    elif n == -1:
        cache[labs(n)] = _intkambe_m1(z, eta)
    elif n == 0:
        cache[labs(n)] = _intkambe_0(z, eta)
    # Simplified recursion for n == 1
    elif n == 1:
        cache[labs(n)] = (
            exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5)
            - _intkambe_m3(z, eta)
        ) / (z * z)
    # Needed to avoid false cache hits
    elif n == 2:
        cache[labs(n)] = (
            _intkambe(0, z, eta, cache)
            - _intkambe_m2(z, eta)
            + eta * exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5)
        ) / (z * z)
    # Needed to avoid false cache hits
    elif n == 3:
        cache[labs(n)] = (
            2 * _intkambe(1, z, eta, cache)
            - _intkambe_m1(z, eta)
            + eta * eta * exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5)
        ) / (z * z)
    elif n < -3:
        cache[labs(n)] = (
            (n + 3) * _intkambe(n + 2, z, eta, cache)
            - z * z * _intkambe(n + 4, z, eta, cache)
            + pow(eta, n + 3) * exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5)
        )
    else:
        cache[labs(n)] = (
            (n - 1) * _intkambe(n - 2, z, eta, cache)
            - _intkambe(n - 4, z, eta, cache)
            + pow(eta, n - 1) * exp((-z * z * eta * eta + 1 / (eta * eta)) * 0.5)
        ) / (z * z)
    return cache[labs(n)]


cdef number_t intkambe(long n, number_t z, number_t eta) nogil:
    r"""
    Integral appearing in the accelerated lattice summations.

    Named here after its appearance (in a slightly different form) in equation (3.16),
    in Kambe's paper [#]_.

    This function is defined as

    .. math::

       I_n(\eta, z)
       = \int_\eta^\infty t^n \mathrm e^{-\frac{z^2t^2}{2} + \frac{1}{2t^2}}
       \mathrm d t

    and is calculated via recursion.

    Args:
        n (integer): Order
        z (float or complex): Argument
        eta (float or complex): Integral cutoff

    Returns:
        float or complex

    References:
        .. [#] `K. Kambe, Zeitschrift Fuer Naturforschung A 23, 9 (1968). <https://doi.org/10.1515/zna-1968-0908>`_
    """
    cdef number_t *cache = <number_t*>malloc((labs(n) + 1) * sizeof(number_t))
    cdef long i
    for i in range(labs(n) + 1):
        cache[i] = NAN
    cdef number_t res = _intkambe(n, z, eta, cache)
    free(cache)
    return res
