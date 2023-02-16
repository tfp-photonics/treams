"""Special mathematical functions related to waves"""

cimport scipy.special.cython_special as cs
from libc.math cimport INFINITY, M_2_SQRTPI, M_SQRT1_2
from libc.math cimport exp as expd
from libc.math cimport fabs, lgamma, pi
from libc.math cimport sqrt as sqrtd
from libc.stdlib cimport labs
from numpy.math cimport NAN

from ptsa.special._bessel cimport (
    hankel1_d,
    jv_d,
    spherical_hankel1,
    spherical_hankel1_d,
)
from ptsa.special._misc cimport cos, double_complex, minusonepow, pow, sin, sqrt
from ptsa.special._wigner3j cimport wigner3j


cdef extern from "<complex.h>" nogil:
    double complex csqrt(double complex z)
    double complex ccos(double complex z)
    double complex cexp(double complex z)
    double complex cpow(double complex x, double complex y)
    double creal(double complex z)
    double cimag(double complex z)


cdef double complex clpmv(double m, double l, double complex z) nogil:
    """Associated Legendre polynomials for complex argument"""
    cdef long li = <long>l
    if li != l:
        return NAN
    cdef long mi = <long>m
    if mi != m:
        return NAN
    if li < 0:
        raise ValueError(f"{li} is not non-negative")
    if labs(mi) > li:
        return 0.0
    cdef double complex p = 1.0
    cdef double complex w = csqrt(1 - z * z)
    cdef long k
    # Go to lpmv(m, m, z)
    if mi > 0:
        for k in range(1, mi + 1):
            p = -(2* k - 1) * w * p
    else:
        for k in range(1, 1 - mi):
            p = w * p / (2 * k)
    if li == labs(mi):
        return p
    cdef double complex p_prev = p
    cdef double complex p_prev_prev
    p = (2 * labs(mi) + 1) * z * p_prev / (labs(mi) - mi + 1.0)
    for k in range(labs(mi) + 2, li + 1):
        p_prev_prev = p_prev
        p_prev = p
        p = (2 * k - 1) / (k - m) * z * p_prev - (k + m - 1) / (k - m) * p_prev_prev
    return p


cdef double complex csph_harm(double m, double l, double phi, double complex theta) nogil:
    """Spherical harmonics for complex argument"""
    if <long>l != l:
        return NAN
    if <long>m != m:
        return NAN
    return (
        0.25 * sqrt(2 * l + 1) * M_2_SQRTPI
        * expd(0.5 * (lgamma(l - m + 1) - lgamma(l + m + 1)))
        * clpmv(m, l, ccos(theta))
        * cexp(1j * m * phi)
    )


cdef number_t lpmv(double m, double l, number_t z) nogil:
    """Associated Legendre polynomials for fused type"""
    if fabs(m) - 1e-8 > l:
        return 0
    if number_t is double:
        return cs.lpmv(m, l, z)
    elif number_t is double_complex:
        return clpmv(m, l, z)


cdef double complex sph_harm(double m, double l, double phi, number_t theta) nogil:
    """Spherical harmonics for fused type"""
    if number_t is double:
        return cs.sph_harm(m, l, phi, theta)
    elif number_t is double_complex:
        return csph_harm(m, l, phi, theta)


cdef number_t pi_fun(double l, double m, number_t x) nogil:
    r"""
    Angular function

    .. math::

        \pi_l^m(x) = \frac{m P_l^m(x)}{\sqrt{1 - x^2}}

    where :math:`P^l_m` is the associated Legendre polynomial.

    Args:
        l (int): degree :math:`l \geq 0`
        m (int): order :math:`|m| \leq l`
        x (float or complex): argument

    Returns:
        float or complex
    """
    if <long>l != l:
        return NAN
    if <long>m != m:
        return NAN
    cdef number_t st = sqrt(1 - x * x)
    cdef double str = creal(st), sti = cimag(st)
    if str * str + sti * sti < 1e-40:
        if m == 1:
            return -pow(x, l + 1) * l * (l + 1) * 0.5
        elif m == -1:
            return -pow(x, l + 1) * 0.5
        return 0.0
    return m * lpmv(m, l, x) / st


cdef number_t tau_fun(double l, double m, number_t x) nogil:
    r"""
    Angular function

    .. math::

        \tau_l^m(x) = \left.\frac{\mathrm d}{\mathrm d \theta}P_l^m(\cos\theta)\right|_{x = \cos\theta}

    where :math:`P_l^m` is the associated Legendre polynomial.

    Args:
        l (int): degree :math:`l \geq 0`
        m (int): order :math:`|m| \leq l`
        x (float or complex): argument

    Returns:
        float or complex
    """
    if <long>l != l:
        return NAN
    if <long>m != m:
        return NAN
    if l == m:
        return -l * lpmv(m - 1, l, x)
    elif l == -m:
        return 0.5 * lpmv(m + 1, l, x)
    return (lpmv(m + 1, l, x) - (l + m) * (l - m + 1) * lpmv(m - 1, l, x)) * 0.5


cdef void vsh_X(long l, long m, number_t theta, double phi, double complex *out, long i) nogil:
    """Vector spherical harmonic X"""
    if l == 0:
        out[0] = 0
        out[i] = 0
        out[2 * i] = 0
        return
    cdef double complex pref = (
        1j
        * sqrtd((2 * l + 1) / (4 * pi * l * (l + 1)))
        * expd((lgamma(l - m + 1) - lgamma(l + m + 1)) * 0.5)
    ) * cexp(1j * m * phi)
    out[0] = 0
    out[i] = 1j * pref * pi_fun(l, m, cos(theta))
    out[2 * i] = -pref * tau_fun(l, m, cos(theta))


cdef void vsh_Y(long l, long m, number_t theta, double phi, double complex *out, long i) nogil:
    """Vector spherical harmonic Y"""
    if l == 0:
        out[0] = 0
        out[i] = 0
        out[2 * i] = 0
        return
    cdef double complex pref = (
        1j
        * sqrtd((2 * l + 1) / (4 * pi * l * (l + 1)))
        * expd((lgamma(l - m + 1) - lgamma(l + m + 1)) * 0.5)
    ) * cexp(1j * m * phi)
    out[0] = 0
    out[i] = pref * tau_fun(l, m, cos(theta))
    out[2 * i] = pref * 1j * pi_fun(l, m, cos(theta))


cdef void vsh_Z(long l, long m, number_t theta, double phi, double complex *out, long i) nogil:
    """Vector spherical harmonic Z"""
    out[0] = 1j * sph_harm(m, l, phi, theta)
    out[i] = 0
    out[2 * i] = 0
    return


cdef void vsw_M(
        long l, long m,
        double complex kr, number_t theta, double phi,
        double complex *out, long i
) nogil:
    """Singular vector spherical wave M"""
    vsh_X(l, m, theta, phi, out, i)
    cdef double complex sh = spherical_hankel1(l, kr)
    out[i] *= sh
    out[2 * i] *= sh


cdef void vsw_rM(
        long l, long m,
        number_t kr, number_t theta, double phi,
        double complex *out, long i
) nogil:
    """Regular vector spherical wave M"""
    vsh_X(l, m, theta, phi, out, i)
    cdef double complex sh = cs.spherical_jn(l, kr)
    out[i] *= sh
    out[2 * i] *= sh


cdef double complex _spherical_hankel1_div_x(long l, double complex x) nogil:
    """Spherical Hankel function of the first kind devided by its argument"""
    return (spherical_hankel1(l - 1, x) + spherical_hankel1(l + 1, x)) / (2 * l + 1)


cdef number_t _spherical_jn_div_x(long l, number_t x) nogil:
    """Spherical Bessel function of the first kind devided by its argument"""
    return (cs.spherical_jn(l - 1, x) + cs.spherical_jn(l + 1, x)) / (2 * l + 1)


cdef void vsw_N(
        long l, long m,
        double complex kr, number_t theta, double phi,
        double complex *out, long i
) nogil:
    """Singular vector spherical wave N"""
    cdef double complex tmp[3]
    vsh_Z(l, m, theta, phi, tmp, 1)
    vsh_Y(l, m, theta, phi, out, i)
    cdef double complex pref = _spherical_hankel1_div_x(l, kr) + spherical_hankel1_d(l, kr)
    out[0] = tmp[0] * _spherical_hankel1_div_x(l, kr) * sqrtd(l * (l + 1))
    out[i] *= pref
    out[2 * i] *= pref


cdef void vsw_rN(
        long l, long m,
        number_t kr, number_t theta, double phi,
        double complex *out, long i
) nogil:
    """Regular vector spherical wave N"""
    cdef double complex tmp[3]
    vsh_Z(l, m, theta, phi, tmp, 1)
    vsh_Y(l, m, theta, phi, out, i)
    cdef double complex pref = _spherical_jn_div_x(l, kr) + cs.spherical_jn(l, kr, 1)
    out[0] = tmp[0] * _spherical_jn_div_x(l, kr) * sqrtd(l * (l + 1))
    out[i] *= pref
    out[2 * i] *= pref


cdef void vsw_A(
        long l, long m,
        double complex kr, number_t theta, double phi,
        long pol,
        double complex *out, long i
) nogil:
    """Singular vector spherical wave of well-defined helicity"""
    cdef double sign = 1 if pol > 0 else -1
    cdef double complex tmp[3]
    vsw_N(l, m, kr, theta, phi, out, i)
    vsw_M(l, m, kr, theta, phi, tmp, 1)
    out[0] = M_SQRT1_2 * (out[0] + sign * tmp[0])
    out[i] = M_SQRT1_2 * (out[i] + sign * tmp[1])
    out[2 * i] = M_SQRT1_2 * (out[2 * i] + sign * tmp[2])


cdef void vsw_rA(
        long l, long m,
        number_t kr, number_t theta, double phi,
        long pol,
        double complex *out, long i
) nogil:
    """Regular vector spherical wave of well-defined helicity"""
    cdef double sign = 1 if pol > 0 else -1
    cdef double complex tmp[3]
    vsw_rN(l, m, kr, theta, phi, out, i)
    vsw_rM(l, m, kr, theta, phi, tmp, 1)
    out[0] = M_SQRT1_2 * (out[0] + sign * tmp[0])
    out[i] = M_SQRT1_2 * (out[i] + sign * tmp[1])
    out[2 * i] = M_SQRT1_2 * (out[2 * i] + sign * tmp[2])


cdef void vcw_M(
        double kz, long m,
        double complex krr, double phi, double z,
        double complex *out, long i
) nogil:
    """Singular vector cylindrical wave M"""
    cdef double complex pref = cexp(1j * (m * phi + kz * z))
    out[0] = 1j * m * cs.hankel1(m, krr) * pref / krr
    out[i] = -hankel1_d(m, krr) * pref
    out[2 * i] = 0


cdef number_t _m_jv_div_x(long m, number_t x) nogil:
    """m times Bessel function devided by its argument"""
    if m == 0:
        return 0
    if x == 0:
        if fabs(m) == 1:
            return 0.5
        if m > 1 or (m % 1 == 0):
            return 0
    return m * cs.jv(m, x) / x


cdef void vcw_rM(
        double kz, long m,
        number_t krr, double phi, double z,
        double complex *out, long i
) nogil:
    """Regular vector cylindrical wave M"""
    cdef double complex pref = cexp(1j * (m * phi + kz * z))
    out[0] = 1j * _m_jv_div_x(m, krr) * pref
    out[i] = -jv_d(m, krr) * pref
    out[2 * i] = 0


cdef void vcw_N(
        double kz, long m,
        double complex krr, double phi, double z,
        double complex k,
        double complex *out, long i
) nogil:
    """Singular vector cylindrical wave N"""
    cdef double complex krho = csqrt(k * k - kz * kz)
    cdef double complex pref = cexp(1j * (m * phi + kz * z)) / k
    out[0] = 1j * kz * hankel1_d(m, krr) * pref
    out[i] = -m * kz * cs.hankel1(m, krr) * pref / krr
    out[2 * i] = krho * cs.hankel1(m, krr) * pref


cdef void vcw_rN(
        double kz, long m,
        number_t krr, double phi, double z,
        double complex k,
        double complex *out, long i
) nogil:
    """Regular vector cylindrical wave N"""
    cdef double complex krho = csqrt(k * k - kz * kz)
    cdef double complex pref = cexp(1j * (m * phi + kz * z)) / k
    out[0] = 1j * kz * jv_d(m, krr) * pref
    out[i] = -kz * _m_jv_div_x(m, krr) * pref
    out[2 * i] = krho * cs.jv(m, krr) * pref


cdef void vcw_A(
        double kz, long m,
        double complex krr, double phi, double z,
        double complex k, long pol,
        double complex *out, long i
) nogil:
    """Singular vector cylindrical wave of well-defined helicity"""
    cdef double sign = 1 if pol > 0 else -1
    cdef double complex tmp[3]
    vcw_N(kz, m, krr, phi, z, k, out, i)
    vcw_M(kz, m, krr, phi, z, tmp, 1)
    out[0] = M_SQRT1_2 * (out[0] + sign * tmp[0])
    out[i] = M_SQRT1_2 * (out[i] + sign * tmp[1])
    out[2 * i] = M_SQRT1_2 * (out[2 * i] + sign * tmp[2])


cdef void vcw_rA(
        double kz, long m,
        number_t krr, double phi, double z,
        double complex k, long pol,
        double complex *out, long i
) nogil:
    """Regular vector cylindrical wave of well-defined helicity"""
    cdef double sign = 1 if pol > 0 else -1
    cdef double complex tmp[3]
    vcw_rN(kz, m, krr, phi, z, k, out, i)
    vcw_rM(kz, m, krr, phi, z, tmp, 1)
    out[0] = M_SQRT1_2 * (out[0] + sign * tmp[0])
    out[i] = M_SQRT1_2 * (out[i] + sign * tmp[1])
    out[2 * i] = M_SQRT1_2 * (out[2 * i] + sign * tmp[2])


# def vsw_L(l, m, kr, theta, phi):
#     """Longitudinal singular vector spherical wave"""
#     return vsh_Z(l, m, theta, phi) * spherical_hankel1_d(l, kr) + np.sqrt(
#         l * (l + 1)
#     ) * vsh_Y(l, m, theta, phi) * _spherical_hankel1_div_x(l, kr)
#
#
# def vcw_L(kz, m, krr, phi, z, k):
#     """Longitudinal singular vector cylindrical wave"""
#     krho = np.sqrt(k * k - kz * kz)
#     return (
#         np.array(
#             [
#                 hankel1_d(m, krr),
#                 1j * m * hankel1(m, krr) / krr,
#                 1j * kz / krho * hankel1(m, krr),
#             ]
#         )
#         * np.exp(1j * (m * phi + kz * z))
#         / k
#     )
#
#
# def vsw_rL(l, m, kr, theta, phi):
#     """Longitudinal regular vector spherical wave"""
#     return vsh_Z(l, m, theta, phi) * spherical_jn_d(l, kr) + np.sqrt(
#         l * (l + 1)
#     ) * vsh_Y(l, m, theta, phi) * _spherical_jn_div_x(l, kr)
#
#
# def vcw_rL(kz, m, krr, phi, z, k):
#     """Longitudinal regular vector cylindrical wave"""
#     krho = np.sqrt(k * k - kz * kz)
#     return (
#         np.array(
#             [
#                 hankel1_d(m, krr),
#                 1j * m * hankel1(m, krr) / krr,
#                 1j * kz / krho * hankel1(m, krr),
#             ]
#         )
#         * np.exp(1j * (m * phi + kz * z))
#         / k
#     )


cdef double complex _tl_vsw_helper(long l, long m, long lambda_, long mu, long p, long q) nogil:
    """Helper function for the translation coefficient of vector spherical waves"""
    if (
        p < max(labs(m + mu), labs(l - lambda_))
        or p > labs(l + lambda_)
        or q < labs(l - lambda_)
        or q > labs(l + lambda_)
        or (q + l + lambda_) % 2 != 0
    ):
        return 0
    return (
        (2 * p + 1)
        * cpow(1j, lambda_ - l + p)
        * expd((lgamma(p - m - mu + 1) - lgamma(p + m + mu + 1)) * 0.5)
        * wigner3j(l, lambda_, p, m, mu, -(m + mu))
        * wigner3j(l, lambda_, q, 0, 0, 0)
    )


cdef double complex tl_vsw_A(
        long lambda_, long mu, long l, long m,
        double complex kr, number_t theta, double phi
) nogil:
    """Singular translation coefficient of vector spherical waves"""
    cdef double complex pref = (
        0.5
        * minusonepow(m)
        * sqrtd(
            (2 * l + 1) * (2 * lambda_ + 1) / <double>(l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * cexp(1j * (m - mu) * phi)
    )
    cdef double complex res = 0
    cdef long p
    for p in range(l + lambda_, max(labs(lambda_ - l), labs(m - mu)) - 1, -2):
        res += (
            _tl_vsw_helper(l, m, lambda_, -mu, p, p)
            * (l * (l + 1) + lambda_ * (lambda_ + 1) - p * (p + 1))
            * spherical_hankel1(p, kr)
            * lpmv(m - mu, p, cos(theta))
        )
    return res * pref


cdef double complex tl_vsw_rA(
        long lambda_, long mu, long l, long m,
        number_t kr, number_t theta, double phi
) nogil:
    """Regular translation coefficient of vector spherical waves"""
    cdef double complex pref = (
        0.5
        * minusonepow(m)
        * sqrtd(
            (2 * l + 1) * (2 * lambda_ + 1) / <double>(l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * cexp(1j * (m - mu) * phi)
    )
    cdef double complex res = 0
    cdef long p
    for p in range(l + lambda_, max(labs(lambda_ - l), labs(m - mu)) - 1, -2):
        res += (
            _tl_vsw_helper(l, m, lambda_, -mu, p, p)
            * (l * (l + 1) + lambda_ * (lambda_ + 1) - p * (p + 1))
            * cs.spherical_jn(p, kr)
            * lpmv(m - mu, p, cos(theta))
        )
    return res * pref


cdef double complex tl_vsw_B(
        long lambda_, long mu, long l, long m,
        double complex kr, number_t theta, double phi
) nogil:
    """Singular translation coefficient of vector spherical waves"""
    cdef double complex pref = (
        0.5
        * minusonepow(m)
        * sqrtd(
            (2 * l + 1) * (2 * lambda_ + 1) / <double>(l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * cexp(1j * (m - mu) * phi)
    )
    cdef double complex res = 0
    cdef long p
    for p in range(l + lambda_ - 1, max(labs(lambda_ - l) + 1, labs(m - mu)) - 1, -2):
        res += (
            _tl_vsw_helper(l, m, lambda_, -mu, p, p - 1)
            * sqrtd(
                (l + lambda_ + 1 + p)
                * (l + lambda_ + 1 - p)
                * (p + lambda_ - l)
                * (p - lambda_ + l)
            )
            * spherical_hankel1(p, kr)
            * lpmv(m - mu, p, cos(theta))
        )
    return res * pref


cdef double complex tl_vsw_rB(
        long lambda_, long mu, long l, long m,
        number_t kr, number_t theta, double phi
) nogil:
    """Regular translation coefficient of vector spherical waves"""
    cdef double complex pref = (
        0.5
        * minusonepow(m)
        * sqrtd(
            (2 * l + 1) * (2 * lambda_ + 1) / <double>(l * (l + 1) * lambda_ * (lambda_ + 1))
        )
        * cexp(1j * (m - mu) * phi)
    )
    cdef double complex res = 0
    cdef long p
    for p in range(l + lambda_ - 1, max(labs(lambda_ - l) + 1, labs(m - mu)) - 1, -2):
        res += (
            _tl_vsw_helper(l, m, lambda_, -mu, p, p - 1)
            * sqrtd(
                (l + lambda_ + 1 + p)
                * (l + lambda_ + 1 - p)
                * (p + lambda_ - l)
                * (p - lambda_ + l)
            )
            * cs.spherical_jn(p, kr)
            * lpmv(m - mu, p, cos(theta))
        )
    return res * pref


cdef double complex tl_vcw(
        double kz1, long mu, double kz2, long m,
        double complex krr, double phi, double z
) nogil:
    """Singular translation coefficient of vector cylindrical waves"""
    if kz1 != kz2:
        return 0
    return cs.hankel1(m - mu, krr) * cexp(1j * ((m - mu) * phi + kz1 * z))


cdef double complex tl_vcw_r(
        double kz1, long mu, double kz2, long m,
        number_t krr, double phi, double z
) nogil:
    """Regular translation coefficient of vector cylindrical waves"""
    if kz1 != kz2:
        return 0
    return cs.jv(m - mu, krr) * cexp(1j * ((m - mu) * phi + kz1 * z))


cdef void vpw_M(
        number_t kx, number_t ky, number_t kz,
        double x, double y, double z,
        double complex *out, long i
) nogil:
    """Vector plane wave M"""
    cdef number_t k = sqrt(kx * kx + ky * ky + kz * kz)
    cdef number_t kpar = sqrt(kx * kx + ky * ky)
    cdef double complex phase = cexp(1j * (kx * x + ky * y + kz * z))
    if k == 0:
        out[0] = NAN
        out[i] = NAN
        out[2 * i] = NAN
    elif kpar == 0:
        out[0] = 0
        out[1] = -1j * phase
        out[2 * i] = 0
    else:
        out[0] = 1j * ky * phase / kpar
        out[1] = -1j * kx * phase / kpar
        out[2 * i] = 0


cdef void vpw_N(
        number_t kx, number_t ky, number_t kz,
        double x, double y, double z,
        double complex *out, long i
) nogil:
    """Vector plane wave N"""
    cdef number_t k = sqrt(kx * kx + ky * ky + kz * kz)
    cdef number_t kpar = sqrt(kx * kx + ky * ky)
    cdef double complex phase = cexp(1j * (kx * x + ky * y + kz * z))
    cdef double sign
    if k == 0:
        out[0] = NAN
        out[i] = NAN
        out[2 * i] = NAN
    elif kpar == 0:
        if cimag(kz) == 0:
            sign = 1 if creal(kz) >= 0 else -1
        else:
            sign = 1 if cimag(kz) >= 0 else -1
        out[0] = -phase * sign
        out[1] = 0
        out[2 * i] = 0
    else:
        out[0] = -kx * kz * phase / (k * kpar)
        out[1] = -ky * kz * phase / (k * kpar)
        out[2 * i] = kpar * phase / k


cdef void vpw_A(
        number_t kx, number_t ky, number_t kz,
        double x, double y, double z,
        long pol,
        double complex *out, long i,
) nogil:
    """Vector plane wave of well-defined helicity"""
    cdef double sign = 1 if pol > 0 else -1
    cdef double complex tmp[3]
    vpw_N(kx, ky, kz, x, y, z, out, i)
    vpw_M(kx, ky, kz, x, y, z, tmp, 1)
    out[0] = M_SQRT1_2 * (out[0] + sign * tmp[0])
    out[i] = M_SQRT1_2 * (out[i] + sign * tmp[1])
    out[2 * i] = M_SQRT1_2 * (out[2 * i] + sign * tmp[2])
