"""Wigner 3j-symbol evaluation by (cached) recursion"""

from libc.math cimport NAN, exp, isnan, lgamma, sqrt
from libc.stdlib cimport free, labs, malloc

from ptsa.special._misc cimport max, minusonepow


cdef double _coeffc(long j1, long j2, long j3, long m3) nogil:
    r"""
    Coefficient C for the recursion relation of the Wigner 3j-symbols

    See equation (7) in [#]_.

    References:
        .. [#] `Y.-L. Xu, J. Comput. Phys. 139, 137 - 165 (1998) <https://doi.org/10.1006/jcph.1997.5867>`_
    """
    cdef double x = (
        (j3 * j3 - (j1 - j2) * (j1 - j2))
        * ((j1 + j2 + 1) * (j1 + j2 + 1) - j3 * j3)
        * (j3 * j3 - m3 * m3)
    )
    return sqrt(x)


cdef long _coeffd(long j1, long j2, long j3, long m1, long m2, long m3) nogil:
    r"""
    Coefficient D for the recursion relation of the Wigner 3j-symbols

    See equation (7) in [#]_.

    References:
        .. [#] `Y.-L. Xu, J. Comput. Phys. 139, 137 - 165 (1998) <https://doi.org/10.1006/jcph.1997.5867>`_
    """
    return (2 * j3 + 1) * (
        j3 * (j3 + 1) * (m2 - m1) + j2 * (j2 + 1) * m3 - j1 * (j1 + 1) * m3
    )


cdef double _initforwardm(long j1, long j2, long m1, long m2) nogil:
    """
    Initial value for the forward recursion for j3 = |m1 + m2|

    See equation (16) in [#]_.

    References:
        .. [#] `Y.-L. Xu, J. Comput. Phys. 139, 137 - 165 (1998) <https://doi.org/10.1006/jcph.1997.5867>`_
    """
    return minusonepow(j2 + m2) * exp(
        (
            lgamma(j1 + m1 + 1)
            + lgamma(j2 + m2 + 1)
            + lgamma(j1 + j2 - m1 - m2 + 1)
            + lgamma(2 * m1 + 2 * m2 + 1)
            - lgamma(j1 - m1 + 1)
            - lgamma(j2 - m2 + 1)
            - lgamma(j1 - j2 + m1 + m2 + 1)
            - lgamma(j2 - j1 + m1 + m2 + 1)
            - lgamma(j1 + j2 + m1 + m2 + 2)
        )
        * 0.5
    )


cdef double _initforwardj(long j1, long j2, long m1, long m2) nogil:
    """
    Initial value for the forward recursion, if j3 = |j1 - j2|

    See equation (14) in [#]_.

    References:
        .. [#] `Y.-L. Xu, J. Comput. Phys. 139, 137 - 165 (1998) <https://doi.org/10.1006/jcph.1997.5867>`_
    """
    return minusonepow(j1 + m1) * exp(
        (
            lgamma(j1 - m1 + 1)
            + lgamma(j1 + m1 + 1)
            + lgamma(2 * j1 - 2 * j2 + 1)
            + lgamma(2 * j2 + 1)
            - lgamma(j2 - m2 + 1)
            - lgamma(j2 + m2 + 1)
            - lgamma(j1 - j2 - m1 - m2 + 1)
            - lgamma(j1 - j2 + m1 + m2 + 1)
            - lgamma(2 * j1 + 2)
        )
        * 0.5
    )


cdef double _wigner3jforward(
    long j1, long j2, long j3,
    long m1, long m2, long m3,
    double *cache
) nogil:
    """
    Forward recursion for Wigner 3j symbol

    See equation (6) in [#]_. The cache contains already computed values or nan, with
    j3 as index. It must be j3 + 1 elements large.

    References:
        .. [#] `Y.-L. Xu, J. Comput. Phys. 139, 137 - 165 (1998) <https://doi.org/10.1006/jcph.1997.5867>`_
    """
    if not isnan(cache[j3]):
        return cache[j3]
    cdef long j3min = max(labs(j1 - j2), labs(m1 + m2))
    cdef double res
    if j3 == j3min:  # TODO: Unnecessary for possible values of j3?
        if j3 == j1 - j2:
            res = _initforwardj(j1, j2, m1, m2)
        elif j3 == j2 - j1:
            res = _initforwardj(j2, j1, m2, m1)
        elif j3 == m1 + m2:
            res = _initforwardm(j1, j2, m1, m2)
        else:  # j3 == -m1 - m2:
            res = _initforwardm(j2, j1, -m2, -m1)
        cache[j3] = res
        return res
    # This if avoids divide by zero in the following if
    if j3min == 0 and j3 == 1:
        res = -_wigner3jforward(j1, j2, 0, m1, m2, m3, cache) * (m2 - m1) / _coeffc(j1, j2, 1, m3)
        cache[j3] = res
        return res
    cdef double res_prev = -_coeffd(j1, j2, j3 - 1, m1, m2, m3) * _wigner3jforward(
        j1, j2, j3 - 1, m1, m2, m3, cache
    )
    if j3 == j3min + 1:
        res = res_prev / (j3min * _coeffc(j1, j2, j3, m3))
        cache[j3] = res
        return res
    cdef double res_prev_prev = (
        -j3
        * _coeffc(j1, j2, j3 - 1, m3)
        * _wigner3jforward(j1, j2, j3 - 2, m1, m2, m3, cache)
    )
    res = (res_prev + res_prev_prev) / ((j3 - 1) * _coeffc(j1, j2, j3, m3))
    cache[j3] = res
    return res


cdef double _wigner3jbackward(
    long j1, long j2, long j3,
    long m1, long m2, long m3,
    double *cache
) nogil:
    """
    Backward recursion for Wigner 3j symbol

    See equation (6) in [#]_. The cache contains already computed values or nan, with
    j3 as index. It must be j1 + j2 + 1 elements large.

    References:
        .. [#] `Y.-L. Xu, J. Comput. Phys. 139, 137 - 165 (1998) <https://doi.org/10.1006/jcph.1997.5867>`_
    """
    if not isnan(cache[j3]):
        return cache[j3]
    cdef double res, res_prev
    if j3 == j1 + j2:
        res = minusonepow(j3 - m3) * exp(
            (
                lgamma(2 * j1 + 1)
                + lgamma(2 * j2 + 1)
                + lgamma(j3 + m3 + 1)
                + lgamma(j3 - m3 + 1)
                - lgamma(2 * j3 + 2)
                - lgamma(j1 - m1 + 1)
                - lgamma(j1 + m1 + 1)
                - lgamma(j2 - m2 + 1)
                - lgamma(j2 + m2 + 1)
            )
            * 0.5
        )
        cache[j3] = res
        return res
    res_prev = -_coeffd(j1, j2, j3 + 1, m1, m2, m3) * _wigner3jbackward(
        j1, j2, j3 + 1, m1, m2, m3, cache
    )
    if j3 == j1 + j2 - 1:
        res = res_prev / ((j3 + 2) * _coeffc(j1, j2, j3 + 1, m3))
        cache[j3] = res
        return res
    cdef double res_prev_prev = (
        -(j3 + 1)
        * _coeffc(j1, j2, j3 + 2, m3)
        * _wigner3jbackward(j1, j2, j3 + 2, m1, m2, m3, cache)
    )
    res = (res_prev + res_prev_prev) / ((j3 + 2) * _coeffc(j1, j2, j3 + 1, m3))
    cache[j3] = res
    return res


cdef double wigner3j(long j1, long j2, long j3, long m1, long m2, long m3) nogil:
    r"""
    Wigner-3j symbol

    Calculate

    .. math::

        \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}

    recursively by forward or backward recurstion.
    Starting points are the extremal values for `j3.` The recursive function
    calls are cached. For unphysical value combinations `0.0` is returned,
    similar to Mathematica's behavior.

    References:
        - `Y.-L. Xu, J. Comput. Phys. 139, 137 - 165 (1998) <https://doi.org/10.1006/jcph.1997.5867>`_
        - `DLMF 34.3 <https://dlmf.nist.gov/34.3>`_
    """
    if j1 < 0 or j2 < 0 or j3 < 0:
        raise ValueError("j values must be non-negative")
    if not labs(j1 - j2) <= j3 <= j1 + j2:
        return 0.0
    if labs(m1) > j1 or labs(m2) > j2 or labs(m3) > j3:
        return 0.0
    if m1 + m2 + m3 != 0:
        return 0.0

    cdef double *cache = <double*>malloc((j1 + j2 + 1) * sizeof(double))  # Size could be smaller by closely inspecting forward or backward recursion
    cdef long i
    for i in range(j1 + j2 + 1):
        cache[i] = <double>NAN

    cdef double res
    if (j1 + j2 - labs(j1 - j2)) // 4 + labs(j1 - j2) > j3:
        res = _wigner3jforward(j1, j2, j3, m1, m2, m3, cache)
    else:
        res = _wigner3jbackward(j1, j2, j3, m1, m2, m3, cache)
    free(cache)
    return res
