from libc.math cimport INFINITY, M_SQRT2, NAN, atan2
from libc.math cimport exp as expd
from libc.math cimport fabs, hypot, lgamma, log, pi
from libc.math cimport sqrt as sqrtd
from libc.stdlib cimport labs
from libc.string cimport memcpy
from numpy.math cimport EULER
from scipy.linalg.cython_blas cimport daxpy, ddot, dgemv, dscal

cimport treams.special.cython_special as sc
from treams.lattice cimport _misc
from treams.special._misc cimport (
    SQPI,
    array_zero,
    exp,
    ipow,
    min,
    minusonepow,
    pow,
    sqrt,
)


# The preprocessor directives correct the missing macro in mingw-w64 by
# substituting it with a corresponding function that is defined by cython anyway
cdef extern from "<complex.h>" nogil:
    """
    #ifndef CMPLX
    #define CMPLX __pyx_t_double_complex_from_parts
    #endif
    """
    double complex CMPLX(double, double)
    double cabs(double complex z)
    double complex cexp(double complex z)
    double cimag(double complex z)
    double complex cpow(double complex x, double complex y)
    double creal(double complex z)


cdef number_t _check_eta(number_t eta, number_t k, double *a, long ds, long dl) nogil:
    if eta != 0:
        return eta
    if ds == 2:
        if dl == 1:
            return sqrtd(2 * pi) / (k * fabs(a[0]))
        if dl == 2:
            return sqrtd(2 * pi / fabs(_misc.area(a, a + 2))) / k
    if ds == 3:
        if dl == 1:
            return sqrtd(2 * pi) / (k * fabs(a[0]))
        if dl == 2:
            return sqrtd(2 * pi / fabs(_misc.area(a, a + 2))) / k
        if dl == 3:
            return sqrtd(2 * pi) / (k * fabs(pow(_misc.volume(a, a + 3, a + 6), 1. / 3)))
    return NAN

cdef double complex _redincgamma(double n, double complex z) nogil:
    r"""
    Reduced incomplete gamma function

    This function returns the value of

    .. math::

        \frac{\Gamma(n, z)}{(-z)^n}

    for integer and half integer argument. The branch for negative values is taken below
    the real axis.

    For `z = 0` a singularity may arise that will be treated accordingly.

    Args:
        n (float): Integer or half integer order
        z (complex): Argument

    Returns:
        complex
    """
    cdef long twicen = <long>(2 * n)
    if twicen != 2 * n:
        raise ValueError("l is not integer of half-integer")
    cdef double singularity = 0.5e-3  # Value of the singularity: smaller is stronger
    cdef double complex res
    if creal(z) * creal(z) + cimag(z) * cimag(z) < 1e-12:
        if twicen > 2:
            raise ValueError("l must not be larger than one for z equal zero")
        if twicen == 0:
            return -<double>EULER - 2 * log(singularity) + 0.5j * pi
        if twicen == 1:
            res = CMPLX(1.0, -1.0) * SQPI / (2 * singularity)
        else:
            res = 0.0
        return res - cpow(-1.0j, 2 * n) / n
    if cimag(z) == 0:
        z = CMPLX(creal(z), -0.0)
    return sc.incgamma(n, z) / cpow(-z, n)


cdef double complex _redintkambe(double n, double complex val, double complex krzeta) nogil:
    r"""
    Reduced Kambe integral

    This function returns the value of

    .. math::

        2 \frac{I_{2n - 1}(\sqrt{-2 z w^2}, -\frac{\mathrm i}{w})}{w^{2n}}

    which reduces to :func:`redincgammma` for `z = 0` or `w = 0`.

    Args:
        n (float): Integer or half integer order
        z (complex): Argument
        w (complex): Second argument

    Returns:
        complex
    """
    if krzeta == 0 or val == 0:
        return _redincgamma(n, val)
    if creal(krzeta) < 0:  # todo: necessary?
        krzeta = -krzeta
    cdef double complex x = sqrt(-2 * val * krzeta * krzeta)
    if cimag(x) == 0:
        x = CMPLX(creal(x), 1e-100)
    if 2 * n != <long>(2 * n):
        raise ValueError("n not an integer or half integer")
    return (
        2
        * sc.intkambe((<long>(2 * n) - 1), x, -1j / krzeta)
        * pow(krzeta, 2 * n)
    )


cdef double complex zero2d(double complex eta) nogil:
    r"""
    Value to add for the zero point substraction in two-dimensional lattices:

    .. math::

        \frac{\mathrm i}{\pi} \Gamma\left(0, -\frac 1 {2\eta^2}\right)

    where `eta` is the cut between real and reciprocal space

    Args:
        eta (float or complex)

    Returns:
        complex
    """
    cdef double complex val = -0.5 / (eta * eta)
    if cimag(val) == 0:
        val = CMPLX(val.real, -0.0)
    return 1j * sc.incgamma(0, val) / pi


cdef double complex zero3d(double complex eta) nogil:
    r"""
    Value to add for the zero point substraction in two-dimensional lattices:

    .. math::

        \frac 1 {4\pi} \Gamma\left(-\frac 1 2, -\frac 1 {2\eta^2}\right)

    where `eta` is the cut between real and reciprocal space

    Args:
        eta (float or complex)

    Returns:
        complex
    """
    cdef double complex val = -0.5 / (eta * eta)
    if val.imag == 0:
        val = CMPLX(val.real, -0.0)
    return sc.incgamma(-0.5, val) / (4 * pi)


cdef double complex _recsw3d(long l, long m, number_t beta, double theta, double phi, number_t eta) nogil:
    r"""
    Summand of the reciprocal contribution in a 3d lattice using spherical solutions

    Computes

    .. math::

       2 Y_{lm}(\theta, \varphi) \beta^l \frac{\Gamma(1, -\frac{\gamma^2}{2\eta^2})}{\gamma^2}

    with :math:`\gamma = \sqrt{1 - \beta^2}`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        m (integer): Order :math:`|m| \leq l`
        beta (float): Tangential part of the wave vector :math:`\frac{|\boldsymbol k_\parallel + \boldsymbol G|}{k}`
        theta (float): Polar angle of the wave vector
        phi (float): Azimuthal angle of the wave vector
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    cdef number_t val = (beta * beta - 1) / (2 * eta * eta)
    if beta == 0:
        return exp(-val) / SQPI if l == 0 else 0
    return (
        sc.sph_harm(m, l, phi, theta)
        * pow(beta, l)
        * _redincgamma(1, val)
        / (eta * eta)
    )


cdef double complex _reccw2d(long l, number_t beta, double phi, number_t eta) nogil:
    r"""
    Summand of the reciprocal contribution in a 2d lattice using cylindrical solutions

    Computes

    .. math::

       2 \mathrm e^{\mathrm i l \varphi} \beta^{|l|} \frac{\Gamma(1, -\frac{\gamma^2}{2\eta^2})}{\gamma^2}

    with :math:`\gamma = \sqrt{1 - \beta^2}`.

    Args:
        l (integer): Degree
        beta (float): Tangential part of the wave vector :math:`\frac{|\boldsymbol k_\parallel + \boldsymbol G|}{k}`
        phi (float): Azimuthal angle of the wave vector
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    cdef number_t val = (beta * beta - 1) / (2 * eta * eta)
    if beta == 0:
        if l == 0:
            return 2 * exp(-val)
        return 0
    return (
        cexp(1j * l * phi)
        * pow(beta, labs(l))
        * _redincgamma(1, val)
        / (eta * eta)
    )


cdef double complex _recsw2d(long l, long m, number_t beta, double phi, number_t eta) nogil:
    r"""
    Summand of the reciprocal contribution in a 2d lattice using spherical solutions

    Computes

    .. math::

       \sqrt{2}
       \mathrm e^{\mathrm i m \varphi}
       \sum_{n = 0}^{\lfloor \frac{l - |m|}{2} \rfloor}
       \frac{\beta^{l - 2n}}{n! (\frac{l - m}{2} - n)! (\frac{l + m}{2} - n)!}
       \frac{\Gamma(\frac{1}{2} - n, -\frac{\gamma^2}{2\eta^2})}{\gamma^{1 - 2n}}

    with :math:`\gamma = \sqrt{1 - \beta^2}`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        m (integer): Order :math:`|m| \leq l` and :math:`l + m` even.
        beta (float): Tangential part of the wave vector :math:`\frac{|\boldsymbol k_\parallel + \boldsymbol G|}{k}`
        theta (float): Polar angle of the wave vector
        phi (float): Azimuthal angle of the wave vector
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    cdef double complex val = (beta * beta - 1) / (2 * eta * eta)
    if cimag(val) == 0:
        val = CMPLX(creal(val), -0.0)
    if beta == 0:
        if m == 0:
            return (
                M_SQRT2 * sc.incgamma(0.5 - l // 2, val) / expd(lgamma(l // 2 + 1))
            )  # l is an even number
        return 0
    cdef double complex res = 0
    cdef number_t macc = (
        pow(beta, l)
        / (eta * expd(lgamma((l + m) // 2 + 1) + lgamma((l - m) // 2 + 1)))
    )
    cdef number_t mult = 2 * eta * eta / (beta * beta)
    cdef long n
    for n in range((l - labs(m)) // 2 + 1):
        res += macc * _redincgamma(0.5 - n, val)
        macc *= mult * ((l + m) // 2 - n) * ((l - m) // 2 - n) / <double>(n + 1)
    return res * cexp(1j * m * phi)


cdef double complex _recsw1d(long l, number_t beta, number_t eta) nogil:
    r"""
    Summand of the reciprocal contribution in a 1d lattice using spherical solutions

    Computes

    .. math::

       \sum_{n = 0}^{\lfloor \frac{l}{2} \rfloor}
       \frac{\beta^{l - 2n}}{4^n n! (l - 2n)!}
       \frac{\Gamma(-n, -\frac{\gamma^2}{2\eta^2})}{\gamma^{-2n}}

    with :math:`\gamma = \sqrt{1 - \beta^2}`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        beta (float): Tangential part of the wave vector :math:`\frac{|k_\parallel + G|}{k}`
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    cdef double complex val = (beta * beta - 1) / (2 * eta * eta)
    if cimag(val) == 0:
        val = CMPLX(creal(val), -0.0)
    if beta == 0:
        if l % 2 == 0:
            return (
                sc.incgamma(- l // 2, val)
                / ipow(2, l)
                * expd(lgamma(l + 1) - lgamma(l // 2 + 1))
            )
        return 0
    cdef double complex res = 0
    cdef number_t macc = pow(beta, l)
    cdef number_t mult = eta * eta / (2 * beta * beta)
    cdef long n
    for n in range(l // 2 + 1):
        res += macc * _redincgamma(-n, val)
        macc *= mult * (l - 2 * n) * (l - 2 * n - 1) / <double>(n + 1)
    return res


cdef double complex _reccw1d(long l, number_t beta, number_t eta) nogil:
    r"""
    Summand of the reciprocal contribution in a 1d lattice using cylindrical solutions

    Computes

    .. math::

       \sqrt{2}
       \sum_{n = 0}^{\lfloor \frac{|l|}{2} \rfloor}
       \frac{\beta^{l - 2n}}{4^n n! (|l| - 2n)!}
       \frac{\Gamma(\frac{1}{2} - n, -\frac{\gamma^2}{2\eta^2})}{\gamma^{1 - 2n}}

    with :math:`\gamma = \sqrt{1 - \beta^2}`.

    Args:
        l (integer): Degree
        beta (float): Tangential part of the wave vector :math:`\frac{|k_\parallel + G|}{k}`
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    l = labs(l)
    cdef double complex val = (beta * beta - 1) / (2 * eta * eta)
    if cimag(val) == 0:
        val = CMPLX(creal(val), -0.0)
    if beta == 0:
        if l % 2 == 0:
            return (
                M_SQRT2
                * sc.incgamma(0.5 - l / 2, val)
                / ipow(2, l)
                * expd(lgamma(l + 1) - lgamma(l // 2 + 1))
            )
        return 0
    cdef double complex res = 0
    cdef number_t macc = pow(beta, l) / eta
    cdef number_t mult = eta * eta / (2 * beta * beta)
    cdef long n
    for n in range(l // 2 + 1):
        res += macc * _redincgamma(0.5 - n, val)
        macc *= mult * (l - 2 * n) * (l - 2 * n - 1) / <double>(n + 1)
    return res


cdef double complex _realsw(long l, long m, number_t kr, double theta, double phi, number_t eta) nogil:
    r"""
    Summand of the real contribution in a 3d lattice using spherical solutions

    Computes

    .. math::

       (k r)^l Y_{lm}(\theta, \varphi) I_{2l}(k r, \eta)

    with the Kambe integral :math:`I_{2l}`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        m (integer): Order :math:`|m| \leq l`
        kr (float or complex): Distance in units of the wave number
        theta (float): Polar angle
        phi (float): Azimuthal angle
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    return pow(kr, l) * sc.intkambe(2 * l, kr, eta) * sc.sph_harm(m, l, phi, theta)


cdef double complex _realcw(long l, number_t kr, double phi, number_t eta) nogil:
    r"""
    Summand of the real contribution in a 3d lattice using cylindrical solutions

    Computes

    .. math::

       (k r)^{|l|} \mathrm e^{\mathrm i l \varphi} I_{2|l| - 1}(k r, \eta)
       \begin{cases}
          (-1)^l & l < 0 \\
          1 & \text{otherwise}
       \end{cases}

    with the Kambe integral :math:`I_{2|l| - 1}`.

    Args:
        l (integer): Order
        kr (float or complex): Distance in units of the wave number
        phi (float): Azimuthal angle
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    cdef double complex pref = minusonepow(l) if l < 0 else 1
    return (
        pow(kr, labs(l))
        * sc.intkambe(2 * labs(l) - 1, kr, eta)
        * cexp(1j * l * phi)
        * pref
    )


cdef double complex _realsw1d(long l, number_t kr, number_t eta) nogil:
    r"""
    Summand of the real contribution in a 1d lattice using spherical solutions

    Computes

    .. math::

       (k r)^l I_{2l}(k r, \eta)

    with the Kambe integral :math:`I_{2l}`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        kr (float or complex): Distance in units of the wave number
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    return pow(kr, l) * sc.intkambe(2 * l, kr, eta)


cdef double complex _realsw2d(long l, long m, number_t kr, double phi, number_t eta) nogil:
    r"""
    Summand of the real contribution in a 2d lattice using spherical solutions

    Computes

    .. math::

       (k r)^l \mathrm e^{\mathrm i m \varphi} I_{2l}(k r, \eta)

    with the Kambe integral :math:`I_{2l}`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        m (integer): Order :math:`|m| \leq l`
        kr (float or complex): Distance in units of the wave number
        phi (float): Azimuthal angle
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    return pow(kr, l) * sc.intkambe(2 * l, kr, eta) * cexp(1j * m * phi)


cdef double complex _realcw1d(long l, number_t kr, number_t eta) nogil:
    r"""
    Summand of the real contribution in a 1d lattice using cylindrical solutions

    Computes

    .. math::

       (k r)^{|l|} I_{2|l| - 1}(k r, \eta)
       \begin{cases}
          (-1)^l & l < 0 \\
          1 & \text{otherwise}
       \end{cases}

    with the Kambe integral :math:`I_{2|l| - 1}`.

    Args:
        l (integer): Order
        kr (float or complex): Distance in units of the wave number
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    cdef double complex pref = minusonepow(l) if l < 0 else 1
    return pow(kr, labs(l)) * sc.intkambe(2 * labs(l) - 1, kr, eta) * pref


cdef double complex realsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    if array_zero(kpar, 2) and array_zero(r, 2) and l % 2 == 1:
        return 0
    eta = _check_eta(eta, k, a, 2, 2)
    cdef double complex realsum = 0, prev = INFINITY, pprev
    cdef long i, start = 1 if array_zero(r, 2) else 0
    cdef double vec[2]
    cdef double coord[2]
    cdef double pointf[2]
    cdef long point[2]
    cdef double mone = -1, zero = 0
    cdef int dim = 2, inc = 1
    cdef char c = b'N'
    for i in range(start, 50):
        pprev = prev
        prev = realsum
        point[0] = -i
        point[1] = -i
        while True:
            pointf[0] = <double>point[0]
            pointf[1] = <double>point[1]
            # vec = -1 * a @ point + 0 * vec
            dgemv(&c, &dim, &dim, &mone, a, &dim, pointf, &inc, &zero, vec, &inc)
            memcpy(coord, vec, dim * sizeof(double))  # coord = vec
            daxpy(&dim, &mone, r, &inc, coord, &inc)  # coord = coord - r
            sc.car2pol(coord, coord)
            realsum += _realcw(l, k * coord[0], coord[1], eta) * cexp(
                -1j * ddot(&dim, kpar, &inc, vec, &inc)
            )
            if not _misc.cubeedge_next(point, dim, i):
                break
        if cabs(realsum - pprev) < 1e-10:
            break
    return -2j * realsum / pi


cdef double complex recsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    if array_zero(kpar, 2) and array_zero(r, 2) and l % 2 == 1:
        return 0
    eta = _check_eta(eta, k, a, 2, 2)
    cdef double complex recsum = 0, prev = INFINITY, pprev
    cdef long i
    cdef double vec[2]
    cdef double coord[2]
    cdef double pointf[2]
    cdef long point[2]
    cdef double one = 1
    cdef int dim = 2, inc = 1
    cdef char c = b'N'
    cdef double b[4]
    _misc.recvec2(a, a + 2, b , b + 2)
    for i in range(50):
        pprev = prev
        prev = recsum
        point[0] = -i
        point[1] = -i
        while True:
            pointf[0] = <double>point[0]
            pointf[1] = <double>point[1]
            memcpy(vec, kpar, dim * sizeof(double))
            # vec = 1 * b @ point + 1 * vec
            dgemv(&c, &dim, &dim, &one, b, &dim, pointf, &inc, &one, vec, &inc)
            sc.car2pol(vec, coord)
            recsum += _reccw2d(l, coord[0] / k, coord[1], eta) * cexp(
                -1j * ddot(&dim, vec, &inc, r, &inc)
            )
            if not _misc.cubeedge_next(point, dim, i):
                break
        if cabs(recsum - pprev) < 1e-10:
            break
    recsum *= 2j * cpow(-1j, l) / (fabs(_misc.area(a, a + 2)) * k * k)
    if l == 0 and array_zero(r, 2):
        return recsum + zero2d(eta)
    return recsum


cdef double complex lsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumcw2d`"""
    eta = _check_eta(eta, k, a, 2, 2)
    return realsumcw2d(l, k, kpar, a, r, eta) + recsumcw2d(l, k, kpar, a, r, eta)


cdef double complex realsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    if l < labs(m):  # also catches l < 0
        return NAN
    if (l + m) % 2 == 1:
        return 0
    if array_zero(kpar, 2) and array_zero(r, 2) and (m % 2) == 1:
        return 0
    eta = _check_eta(eta, k, a, 3, 2)
    cdef double complex realsum = 0, prev = INFINITY, pprev
    cdef long i, start = 1 if array_zero(r, 2) else 0
    cdef double vec[2]
    cdef double coord[2]
    cdef double pointf[2]
    cdef long point[2]
    cdef double mone = -1, zero = 0
    cdef int dim = 2, inc = 1
    cdef char c = b'N'
    for i in range(start, 50):
        pprev = prev
        prev = realsum
        point[0] = -i
        point[1] = -i
        while True:
            pointf[0] = <double>point[0]
            pointf[1] = <double>point[1]
            # vec = -1 * a @ point + 0 * vec
            dgemv(&c, &dim, &dim, &mone, a, &dim, pointf, &inc, &zero, vec, &inc)
            memcpy(coord, vec, dim * sizeof(double))  # coord = vec
            daxpy(&dim, &mone, r, &inc, coord, &inc)  # coord = coord - r
            sc.car2pol(coord, coord)
            realsum += _realsw2d(l, m, k * coord[0], coord[1], eta) * cexp(
                -1j * ddot(&dim, kpar, &inc, vec, &inc)
            )
            if not _misc.cubeedge_next(point, dim, i):
                break
        if cabs(realsum - pprev) < 1e-10:
            break
    return (
        -1j
        * minusonepow((l + m) // 2)
        * sqrt((2 * l + 1) * 0.5)
        * realsum
        * expd(
            (lgamma(l + m + 1) + lgamma(l - m + 1)) * 0.5
            - lgamma((l + m) // 2 + 1)
            - lgamma((l - m) // 2 + 1)
        )
        / (pi * ipow(2, l))
    )


cdef double complex recsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    if l < labs(m):  # also catches l < 0
        return NAN
    if (l + m) % 2 == 1:
        return 0
    if array_zero(kpar, 2) and array_zero(r, 2) and (m % 2) == 1:
        return 0
    eta = _check_eta(eta, k, a, 3, 2)
    cdef double complex recsum = 0, prev = INFINITY, pprev
    cdef long i
    cdef double vec[2]
    cdef double coord[2]
    cdef double pointf[2]
    cdef long point[2]
    cdef double one = 1
    cdef int dim = 2, inc = 1
    cdef char c = b'N'
    cdef double b[4]
    _misc.recvec2(a, a + 2, b , b + 2)
    for i in range(50):
        pprev = prev
        prev = recsum
        point[0] = -i
        point[1] = -i
        while True:
            pointf[0] = <double>point[0]
            pointf[1] = <double>point[1]
            memcpy(vec, kpar, dim * sizeof(double))
            # vec = 1 * b @ point + 1 * vec
            dgemv(&c, &dim, &dim, &one, b, &dim, pointf, &inc, &one, vec, &inc)
            sc.car2pol(vec, coord)
            recsum += _recsw2d(l, m, coord[0] / k, coord[1], eta) * cexp(
                -1j * ddot(&dim, vec, &inc, r, &inc)
            )
            if not _misc.cubeedge_next(point, dim, i):
                break
        if cabs(recsum - pprev) < 1e-10:
            break
    recsum *= (
        sqrtd((2 * l + 1) * 0.5)
        * pow(1j, m)
        * expd((lgamma(l + m + 1) + lgamma(l - m + 1)) * 0.5)
        / (fabs(_misc.area(a, a + 2)) * k * k * ipow(2, l))
    )
    if l == 0 and array_zero(r, 2):
        return recsum + zero3d(eta)
    return recsum


cdef double complex lsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumsw2d`"""
    eta = _check_eta(eta, k, a, 3, 2)
    return realsumsw2d(l, m, k, kpar, a, r, eta) + recsumsw2d(l, m, k, kpar, a, r, eta)


cdef double complex realsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumsw1d`"""
    if l < 0:
        return NAN
    eta = _check_eta(eta, k, &a, 3, 1)
    # In these cases the opposite points of the lattice cancel each other
    if kpar == 0 and l % 2 == 1 and (r == 0 or fabs(a * 0.5) == fabs(r)):
        return 0
    cdef double complex realsum = 0, prev = INFINITY, pprev
    cdef number_t beta
    cdef long start = 1 if r == 0 else 0, i
    cdef long point[1]
    for i in range(start, 200):
        pprev = prev
        prev = realsum
        point[0] = -i
        while True:
            vec = a * point[0]
            realsum += _realsw1d(l, -k * (r + vec), eta) * cexp(1j * kpar * vec)
            if not _misc.cubeedge_next(point, 1, i):
                break
        if cabs(realsum - pprev) < 1e-10:
            break
    return -1j * sqrtd((2 * l + 1) * 0.5) * realsum / pi


cdef double complex recsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    if l < 0:
        return NAN
    eta = _check_eta(eta, k, &a, 3, 1)
    if kpar == 0 and l % 2 == 1 and (r == 0 or fabs(a * 0.5) == fabs(r)):
        return 0
    cdef double b = 2 * pi / a, vec
    cdef double complex recsum = 0, prev = INFINITY, pprev
    cdef number_t beta
    cdef long i
    cdef long point[1]
    for i in range(200):
        pprev = prev
        prev = recsum
        point[0] = -i
        while True:
            vec = b * point[0]
            beta = (kpar + vec) / k
            recsum += _recsw1d(l, beta, eta) * cexp(-1j * (kpar + vec) * r)
            if not _misc.cubeedge_next(point, 1, i):
                break
        if cabs(recsum - pprev) < 1e-10:
            break
    recsum *= pow(-1j, l + 1) * sqrtd((2 * l + 1) / pi) / (2 * fabs(a) * k)
    if l == 0 and r == 0:
        return recsum + zero3d(eta)
    return recsum


cdef double complex lsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumsw1d`"""
    if l < 0:
        return NAN
    eta = _check_eta(eta, k, &a, 3, 1)
    return realsumsw1d(l, k, kpar, a, r, eta) + recsumsw1d(l, k, kpar, a, r, eta)


cdef double complex realsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    eta = _check_eta(eta, k, &a, 2, 1)
    if kpar == 0 and l % 2 == 1 and (r == 0 or fabs(a * 0.5) == fabs(r)):
        return 0
    cdef double complex realsum = 0, prev = INFINITY, pprev
    cdef number_t beta
    cdef long start = 1 if r == 0 else 0, i
    cdef long point[1]
    for i in range(start, 200):
        pprev = prev
        prev = realsum
        point[0] = -i
        while True:
            vec = a * point[0]
            realsum += _realcw1d(l, -k * (r + vec), eta) * cexp(1j * kpar * vec)
            if not _misc.cubeedge_next(point, 1, i):
                break
        if cabs(realsum - pprev) < 1e-10:
            break
    return -2j * realsum / pi

cdef double complex recsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    eta = _check_eta(eta, k, &a, 2, 1)
    if kpar == 0 and l % 2 == 1 and (r == 0 or fabs(a * 0.5) == fabs(r)):
        return 0
    cdef double b = 2 * pi / a
    cdef double complex recsum = 0, prev = INFINITY, pprev
    cdef number_t beta
    cdef long i
    cdef long point[1]
    for i in range(200):
        pprev = prev
        prev = recsum
        point[0] = -i
        while True:
            vec = b * point[0]
            beta = (kpar + vec) / k
            recsum += _reccw1d(l, beta, eta) * cexp(-1j * (kpar + vec) * r)
            if not _misc.cubeedge_next(point, 1, i):
                break
        if cabs(recsum - pprev) < 1e-10:
            break
    recsum *= cpow(-1j, l) * sqrtd(2 / pi) / (fabs(a) * k)
    if l == 0 and r == 0:
        return recsum + zero2d(eta)
    return recsum

cdef double complex lsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumcw1d`"""
    eta = _check_eta(eta, k, &a, 2, 1)
    return realsumcw1d(l, k, kpar, a, r, eta) + recsumcw1d(l, k, kpar, a, r, eta)


cdef double complex realsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    if l < labs(m):  # also catches l < 0
        return NAN
    eta = _check_eta(eta, k, a, 3, 3)
    if array_zero(kpar, 3) and array_zero(r, 3) and l % 2 == 1:
        return 0

    cdef long i
    cdef double complex realsum = 0, prev = INFINITY, pprev
    cdef double vec[3]
    cdef double coord[3]
    cdef double pointf[3]
    cdef long point[3]
    cdef long start = 1 if array_zero(r, 3) else 0
    cdef double mone = -1, zero = 0
    cdef int dim = 3, inc = 1
    cdef char c = b'N'
    for i in range(start, 50):
        pprev = prev
        prev = realsum
        point[0] = -i
        point[1] = -i
        point[2] = -i
        while True:
            pointf[0] = <double>point[0]
            pointf[1] = <double>point[1]
            pointf[2] = <double>point[2]
            # vec = -1 * a @ point + 0 * vec
            dgemv(&c, &dim, &dim, &mone, a, &dim, pointf, &inc, &zero, vec, &inc)
            memcpy(coord, vec, dim * sizeof(double))  # coord = vec
            daxpy(&dim, &mone, r, &inc, coord, &inc)  # coord = coord - r
            sc.car2sph(coord, coord)
            realsum += _realsw(
                l, m, k * coord[0], coord[1], coord[2], eta
            ) * cexp(-1j * ddot(&dim, kpar, &inc, vec, &inc))
            if not _misc.cubeedge_next(point, dim, i):
                break
        if cabs(realsum - pprev) < 1e-10:
            break
    return -1j * sqrt(2 / pi) * realsum

cdef double complex recsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumsw3d`"""
    if l < labs(m):  # also catches l < 0
        return NAN
    eta = _check_eta(eta, k, a, 3, 3)
    if array_zero(kpar, 3) and array_zero(r, 3) and l % 2 == 1:
        return 0

    cdef double b[9]
    _misc.recvec3(a, a + 3, a + 6, b, b + 3, b + 6)
    cdef long i
    cdef double complex recsum = 0, prev = INFINITY, pprev
    cdef double vec[3]
    cdef double coord[3]
    cdef double pointf[3]
    cdef long point[3]
    cdef double one = 1
    cdef int dim = 3, inc = 1
    cdef char c = b'N'
    for i in range(50):
        pprev = prev
        prev = recsum
        point[0] = -i
        point[1] = -i
        point[2] = -i
        while True:
            pointf[0] = <double>point[0]
            pointf[1] = <double>point[1]
            pointf[2] = <double>point[2]
            memcpy(vec, kpar, dim * sizeof(double))
            dgemv(&c, &dim, &dim, &one, b, &dim, pointf, &inc, &one, vec, &inc)  # vec = 1 * b @ point + 1 * vec
            sc.car2sph(vec, coord)
            recsum += _recsw3d(l, m, coord[0] / k, coord[1], coord[2], eta) * cexp(
                -1j * ddot(&dim, vec, &inc, r, &inc)
            )
            if not _misc.cubeedge_next(point, dim, i):
                break
        if cabs(recsum - pprev) < 1e-10:
            break
    recsum *= (
        2j
        * pow(-1j, l)
        * pi
        / (fabs(_misc.volume(a, a + 3, a + 6)) * k * k * k)
    )
    if l == 0 and array_zero(r, 3):
        return recsum + zero3d(eta)
    return recsum


cdef double complex lsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumsw3d`"""
    if l < labs(m):  # also catches l < 0
        return NAN
    if array_zero(kpar, 3) and array_zero(r, 3) and l % 2 == 1:
        return 0
    eta = _check_eta(eta, k, a, 3, 3)
    return realsumsw3d(l, m, k, kpar, a, r, eta) + recsumsw3d(l, m, k, kpar, a, r, eta)


cdef number_t _s_sum_sw2d(long l, long absm, long n, number_t krz, number_t beta) nogil:
    r"""
    Inner sum of the reciprocal contribution in a 2d lattice using spherical solutions

    Computes

    .. math::

       S_{l m n}(k r_z, \beta)
       = \sum_{s = n}^{\min(l - |m|, 2n)}
       \frac{(-k r_z)^{2n - s} \beta^{l - s}}
       {(2n - s)!(s - n)!(\frac{l + |m| - s}{2})!(\frac{l - |m| - s}{2})!}

    Args:
        l (integer): Degree :math:`l \geq 0`
        absm (integer): Absolute value of the order :math:`|m| \leq l`
        n (integer): Outer sum parameter
        krz (float or complex): Shift value :math:`k r_z`
        beta (float): Tangential part of the wave vector :math:`\frac{|k_\parallel + G|}{k}`

    Returns:
        complex
    """
    cdef number_t res = 0.0
    if beta == 0:
        if absm == 0 and 2 * n >= l:
            res += pow(-krz, 2 * n - l) * expd(
                -lgamma(2 * n - l + 1) - lgamma(l - n + 1)
            )
        return res

    cdef long start = n if (l - absm - n) % 2 == 0 else n + 1
    if krz == 0:
        if l - absm >= 2 * n >= start and (l - absm) % 2 == 0:
            res += pow(beta, l - 2 * n) * expd(
                -lgamma(n + 1)
                - lgamma((l - absm) / 2 - n + 1)
                - lgamma((l + absm) / 2 - n + 1)
            )
        return res

    if start > min(l - absm, 2 * n):
        return 0
    cdef number_t mul = 1 / (krz * krz * beta * beta)
    cdef number_t macc = (
        pow(-krz, 2 * n - start)
        * pow(beta, l - start)
        * expd(
            -lgamma(2 * n - start + 1)
            - lgamma((l + absm - start) * 0.5 + 1)
            - lgamma((l - absm - start) * 0.5 + 1)
        )
    )
    cdef long s
    for s in range(start, min(l - absm, 2 * n) + 1, 2):
        res += macc
        macc *= (
            mul
            * (2 * n - s)
            * (2 * n - s - 1)
            * (l + absm - s)
            / 2.
            * (l - absm - s)
            / 2.
            / <double>((s - n + 1) * (s - n + 2))
        )
    return res


cdef double complex _n_sum_sw2d(long l, long m, number_t beta, number_t krz, double phi, number_t eta) nogil:
    r"""
    Outer sum of the reciprocal contribution in a 2d lattice using spherical solutions

    Computes

    .. math::

       \mathrm e^{\mathrm i m \varphi}
       \sum_{n = 0}^{l - |m|} S_{l m n}(k r_z, \beta)
       (2 \eta^2)^{n - \frac{1}{2}}
       \tilde{I}_{\frac{1}{2} - n}(-\frac{\gamma^2}{2 \eta^2}, k r_z \eta)

    with the inner sum :math:`S_{l m n}(k r_z, \beta)`
    (:func:`_s_sum_sw2d`) and :math:`\tilde{I}` (:func:`_redintkambe`), where
    :math:`\gamma = \sqrt{1 - \beta^2}`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        m (integer): Value of the order :math:`|m| \leq l`
        beta (float): Tangential part of the wave vector :math:`\frac{|k_\parallel + G|}{k}`
        krz (float or complex): Shift value :math:`k r_z`
        phi (float): Azimuthal angle
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    cdef double complex val = (beta * beta - 1) / (2 * eta * eta)
    if cimag(val) == 0:
        val = CMPLX(creal(val), -0.0)
    cdef number_t krzeta = krz * eta
    cdef double complex res = 0
    cdef number_t macc = 1 / (eta * M_SQRT2)
    cdef number_t mult = 2 * eta * eta
    cdef long n
    for n in range(l - labs(m) + 1):
        res += (
            macc * _redintkambe(0.5 - n, val, krzeta) * _s_sum_sw2d(l, labs(m), n, krz, beta)
        )
        macc *= mult
    return res * cexp(1j * m * phi)


cdef double complex realsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    eta = _check_eta(eta, k, a, 3, 2)
    if fabs(r[2]) < 1e-100:
        return realsumsw2d(l, m, k, kpar, a, r, eta)
    cdef double complex realsum = 0, prev = INFINITY, pprev
    cdef long i
    cdef double mone = -1, zero = 0
    cdef double vec[2]
    cdef double coord[3]
    cdef double pointf[2]
    cdef long point[3]
    cdef int dim = 2, inc = 1
    cdef char c = b'N'
    for i in range(50):
        pprev = prev
        prev = realsum
        point[0] = -i
        point[1] = -i
        while True:
            pointf[0] = <double>point[0]
            pointf[1] = <double>point[1]
            # vec = -1 * a @ point + 0 * vec
            dgemv(&c, &dim, &dim, &mone, a, &dim, pointf, &inc, &zero, vec, &inc)
            memcpy(coord, vec, dim * sizeof(double))
            coord[2] = -r[2]
            daxpy(&dim, &mone, r, &inc, coord, &inc)  # coord = coord - r
            sc.car2sph(coord, coord)
            realsum += _realsw(
                l, m, k * coord[0], coord[1], coord[2], eta
            ) * cexp(-1j * ddot(&dim, kpar, &inc, vec, &inc))
            if not _misc.cubeedge_next(point, dim, i):
                break
        if cabs(realsum - pprev) < 1e-10:
            break
    return -1j * sqrt(2 / pi) * realsum


cdef double complex recsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    r"""
    Reciprocal summation for spherical solutions in 2d exluding a common prefactor

    Computes

    .. math::

       \sum_{\boldsymbol G \in \Lambda_2^\ast}
       \mathrm e^{-\mathrm i (\boldsymbol k_\parallel + \boldsymbol G) \boldsymbol r}
       n(l, m, \frac{|\boldsymbol k_\parallel + \boldsymbol G|}{k}, k r_z, \varphi, \eta)

    where `n` is :func:`_n_sum_sw2d`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        m (integer): Order :math:`|m| \leq l`
        k (float or complex): Wave number
        kpar (float, (2,)-array): Tangential wave vector
        a (float, (2,2)-array): Lattice vectors
        r (float, (3,)-array): Shift vector
        eta (float or complex): Separation value

    Returns:
        complex
    """
    eta = _check_eta(eta, k, a, 3, 2)
    if fabs(r[2]) < 1e-100:
        return recsumsw2d(l, m, k, kpar, a, r, eta)
    cdef double complex recsum = 0, prev = INFINITY, pprev
    cdef long i
    cdef double vec[2]
    cdef double coord[2]
    cdef double pointf[2]
    cdef long point[2]
    cdef double one = 1
    cdef number_t krz = k * r[2]
    cdef int dim = 2, inc = 1
    cdef char c = b'N'
    cdef double b[4]
    _misc.recvec2(a, a + 2, b, b + 2)
    for i in range(50):
        pprev = prev
        prev = recsum
        point[0] = -i
        point[1] = -i
        while True:
            pointf[0] = <double>point[0]
            pointf[1] = <double>point[1]
            memcpy(vec, kpar, dim * sizeof(double))
            # vec = 1 * b @ point + 1 * vec
            dgemv(&c, &dim, &dim, &one, b, &dim, pointf, &inc, &one, vec, &inc)
            sc.car2pol(vec, coord)
            recsum += _n_sum_sw2d(
                l, m, coord[0] / k, krz, coord[1], eta
            ) * cexp(-1j * ddot(&dim, vec, &inc, r, &inc))
            if not _misc.cubeedge_next(point, dim, i):
                break
        if cabs(recsum - pprev) < 1e-10:
            break

    return (
        sqrtd(2 * l + 1) * cpow(-1j, m)
        * expd((lgamma(l + m + 1) + lgamma(l - m + 1)) * 0.5)
        * recsum
        / (fabs(_misc.area(a, a + 2)) * k * k * ipow(-2, l))
    )


cdef double complex lsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumsw2d_shift`"""
    eta = _check_eta(eta, k, a, 3, 2)
    if fabs(r[2]) < 1e-100:
        return lsumsw2d(l, m, k, kpar, a, r, eta)
    return (
        realsumsw2d_shift(l, m, k, kpar, a, r, eta)
        + recsumsw2d_shift(l, m, k, kpar, a, r, eta)
    )


cdef number_t _s_sum_sw1d(long l, long absm, long n, number_t krho, number_t beta) nogil:
    r"""
    Inner sum of the reciprocal contribution in a 1d lattice using spherical solutions

    Computes

    .. math::

       S_{l m n}(k \rho, \beta)
       = \sum_{s = n}^{\min(2n - |m|, l)}
       \frac{(k \rho)^{2n - s} \beta^{l - s}}
       {\left(n - \frac{s + m}{2}\right)!\left(n - \frac{s - m}{2}\right)!(l - s)!(s - n)!}

    Args:
        l (integer): Degree :math:`l \geq 0`
        absm (integer): Absolute value of the order :math:`|m| \leq l`
        n (integer): Outer sum parameter
        krho (float or complex): Shift value :math:`k \rho`
        beta (float): Tangential part of the wave vector :math:`\frac{k_\parallel + G}{k}`

    Returns:
        complex
    """
    cdef number_t res = 0.0
    if beta == 0:
        if (l + absm) % 2 == 0 and 2 * n - absm >= l >= n:
            res += pow(krho, 2 * n - l) * expd(
                -lgamma(n - (l + absm) / 2 + 1)
                - lgamma(n - (l - absm) / 2 + 1)
                - lgamma(l - n + 1)
            )
        return res

    cdef long start = n if (absm - n) % 2 == 0 else n + 1
    if krho == 0:
        # if m != 0, 2n - abs(m) < 2n
        # the sum runs from n to max(2n, l), so l >=  2n >= n
        if absm == 0 and l >= 2 * n >= start:
            res += pow(beta, l - 2 * n) * expd(
                -lgamma(n + 1) - lgamma(l - 2 * n + 1)
            )
        return res

    cdef number_t mul = 1 / (krho * krho * beta * beta)
    cdef number_t macc = (
        pow(krho, 2 * n - start)
        * pow(beta, l - start)
        * expd(
            -lgamma(n - (start + absm) * 0.5 + 1)
            - lgamma(n - (start - absm) * 0.5 + 1)
            - lgamma(start - n + 1)
            - lgamma(l - start + 1)
        )
    )
    cdef long s
    for s in range(start, min(2 * n - absm, l) + 1, 2):
        res += macc
        macc *= (
            mul
            * (n - (s + absm) * 0.5)
            * (n - (s - absm) * 0.5)
            * (l - s)
            * (l - s - 1)
            / <double>((s - n + 1) * (s - n + 2))
        )
    return res


cdef double complex _n_sum_sw1d(long l, long absm, number_t beta, number_t krho, number_t eta) nogil:
    r"""
    Outer sum of the reciprocal contribution in a 1d lattice using spherical solutions

    Computes

    .. math::

       \sum_{n = |m|}^{l} S_{l m n}(k \rho, \beta)
       \frac{\eta^{2n}}{2^n}
       \tilde{I}_{- n}(-\frac{\gamma^2}{2 \eta^2}, k r_z \eta)

    with the inner sum :math:`S_{l m n}(k r_z, \beta)`
    (:func:`_s_sum_sw1d`) and :math:`\tilde{I}` (:func:`_redintkambe`), where
    :math:`\gamma = \sqrt{1 - \beta^2}`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        absm (integer): Absolute value of the order :math:`|m| \leq l`
        beta (float): Tangential part of the wave vector :math:`\frac{k_\parallel + G}{k}`
        krho (float or complex): Shift value :math:`k \rho`
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    cdef double complex val = (beta * beta - 1) / (2 * eta * eta)
    if cimag(val) == 0:
        val = CMPLX(creal(val), -0.0)
    cdef number_t krhoeta = krho * eta
    cdef double complex res = 0
    cdef number_t macc = pow(0.5 * eta * eta, absm)
    cdef number_t mult = 0.5 * eta * eta
    cdef long stop = l if (l - absm) % 2 == 0 else l - 1, n
    for n in range(absm, stop + 1):
        res += macc * _redintkambe(-n, val, krhoeta) * _s_sum_sw1d(l, absm, n, krho, beta)
        macc *= mult
    return res


cdef double complex recsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    r"""
    Reciprocal summation for spherical solutions in 1d exluding a common prefactor

    Computes

    .. math::

       \sum_{G \in \Lambda_1^\ast}
       \mathrm e^{-\mathrm i (k_\parallel + G) \boldsymbol r_z}
       n(l, |m|, \frac{k_\parallel + G}{k}, k \rho, \eta)

    where `n` is :func:`_n_sum_sw1d`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        m (integer): Order :math:`|m| \leq l`
        k (float or complex): Wave number
        kpar (float): Tangential wave vector component
        a (float): Lattice pitch
        r (float, (3,)-array): Shift vector
        eta (float or complex): Separation value

    Returns:
        complex
    """
    if l < 0:
        return NAN
    eta = _check_eta(eta, k, &a, 3, 2)
    if fabs(r[0]) < 1e-100 and fabs(r[1]) < 1e-100:
        if m == 0:
            return recsumsw1d(l, k, kpar, a, r[2], eta)
        return 0
    cdef double complex recsum = 0, prev = INFINITY, pprev
    cdef long point[1]
    cdef long i
    cdef double vec
    cdef double b = 2 * pi / a
    cdef double rz = r[2], rho = hypot(r[0], r[1])
    cdef number_t krho = k * rho
    for i in range(200):
        pprev = prev
        prev = recsum
        point[0] = -i
        while True:
            vec = kpar + b * point[0]
            recsum += _n_sum_sw1d(l, labs(m), vec / k, krho, eta) * cexp(
                -1j * vec * rz
            )
            if not _misc.cubeedge_next(point, 1, i):
                break
        if cabs(recsum - pprev) < 1e-10:
            break
    return (
        -1j
        * (
            sqrtd((2 * l + 1) / pi)
            * cpow(-1j, l - m)
            * expd((lgamma(l + m + 1) + lgamma(l - m + 1)) * 0.5)
            / (2 * fabs(a) * k)
        )
        * cexp(1j * m * atan2(-r[1], -r[0]))
        * recsum
    )

cdef double complex realsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    if l < 0:
        return NAN
    eta = _check_eta(eta, k, &a, 3, 1)
    if fabs(r[0]) < 1e-100 and fabs(r[1]) < 1e-100:
        if m == 0:
            return realsumsw1d(l, k, kpar, a, r[2], eta)
        return 0
    cdef double complex realsum = 0, prev = INFINITY, pprev
    cdef double coord[3]
    cdef long point[1]
    cdef double vec
    cdef long i
    for i in range(200):
        pprev = prev
        prev = realsum
        point[0] = -i
        while True:
            vec = a * point[0]
            coord[0] = -r[0]
            coord[1] = -r[1]
            coord[2] = -r[2] - vec
            sc.car2sph(coord, coord)
            realsum += _realsw(
                l, m, k * coord[0], coord[1], coord[2], eta
            ) * cexp(1j * kpar * vec)
            if not _misc.cubeedge_next(point, 1, i):
                break
        if cabs(realsum - pprev) < 1e-10:
            break
    return -1j * sqrt(2 / pi) * realsum

cdef double complex lsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumsw1d_shift`"""
    eta = _check_eta(eta, k, &a, 3, 1)
    if l < 0:
        return NAN
    if fabs(r[0]) < 1e-100 and fabs(r[1]) < 1e-100:
        if m == 0:
            return lsumsw1d(l, k, kpar, a, r[2], eta)
        return 0
    return (
        realsumsw1d_shift(l, m, k, kpar, a, r, eta)
        + recsumsw1d_shift(l, m, k, kpar, a, r, eta)
    )


cdef number_t _s_sum_cw1d(long l, long n, number_t ky, number_t beta) nogil:
    r"""
    Inner sum of the reciprocal contribution in a 1d lattice using cylindrical solutions

    Computes

    .. math::

       S_{l n}(k r_y, \beta)
       = \sum_{s = n}^{\min(2n, |l|)}
       \frac{(\mp k y)^{2n - s} \beta^{|l| - s}}
       {2^s (2n - s)!(|l|-s)!(s-n)!}

    with the sign is determined by the sign of :math:`l`.

    Args:
        l (integer): Degree
        n (integer): Outer sum parameter
        ky (float or complex): Shift value :math:`k r_y`
        beta (float): Tangential part of the wave vector :math:`\frac{k_\parallel + G}{k}`

    Returns:
        complex
    """
    cdef number_t res = 0.0
    if beta == 0:
        if 2 * n >= l:
            res += (
                pow(-ky, 2 * n - l)
                * expd(lgamma(l + 1) - lgamma(2 * n - l + 1) - lgamma(l - n + 1))
                * pow(0.5, l)
            )
        return res

    if ky == 0:
        if l >= 2 * n:
            res += (
                pow(beta, l - 2 * n)
                * expd(lgamma(l + 1) - lgamma(n + 1) - lgamma(l - 2 * n + 1))
                * pow(0.25, n)
            )
        return res

    cdef number_t mul = -0.5 / (ky * beta)
    cdef number_t macc = (
        pow(-0.5 * ky, n)
        * pow(beta, l - n)
        * expd(lgamma(l + 1) - lgamma(n + 1) - lgamma(l - n + 1))
    )
    cdef long s
    for s in range(n, min(2 * n, l) + 1):
        res += macc
        macc *= mul * (2 * n - s) * (l - s) / <double>(s - n + 1)
    return res


cdef double complex _n_sum_cw1d(long l, number_t beta, number_t ky, number_t eta) nogil:
    r"""
    Outer sum of the reciprocal contribution in a 1d lattice using cylindrical solutions

    Computes

    .. math::

       \sum_{n = 0}^{|l|} S_{l n}(k r_y, \beta)
       (2 \eta^2)^{n - \frac{1}{2}}
       \tilde{I}_{\frac{1}{2} - n}(-\frac{\gamma^2}{2 \eta^2}, k r_z \eta)

    with the inner sum :math:`S_{l m n}(k r_z, \beta)`
    (:func:`_s_sum_cw1d`) and :math:`\tilde{I}` (:func:`_redintkambe`), where
    :math:`\gamma = \sqrt{1 - \beta^2}`.

    Args:
        l (integer): Degree :math:`l \geq 0`
        beta (float): Tangential part of the wave vector :math:`\frac{k_\parallel + G}{k}`
        ky (float or complex): Shift value :math:`k r_y`
        eta (float or complex): Cut between reciprocal and real space summation

    Returns:
        complex
    """
    cdef double complex val = (beta * beta - 1) / (2 * eta * eta)
    if cimag(val) == 0:
        val = CMPLX(creal(val), -0.0)
    cdef number_t kyeta = ky * eta
    cdef double complex res = 0
    cdef number_t macc = 1 / (M_SQRT2 * eta)
    cdef number_t mult = 2 * eta * eta
    cdef long n
    for n in range(l + 1):
        res += macc * _redintkambe(0.5 - n, val, kyeta) * _s_sum_cw1d(l, n, ky, beta)
        macc *= mult
    return res


cdef double complex recsumcw1d_shift(long l, number_t k, double kpar, double a, double* r, number_t eta) nogil:
    r"""
    Reciprocal summation for cylindrical solutions in 1d exluding a common prefactor

    Computes

    .. math::

       \sum_{G \in \Lambda_1^\ast}
       \mathrm e^{-\mathrm i (k_\parallel + G) \boldsymbol r_x}
       n(l, \frac{k_\parallel + G}{k}, k r_y, \eta)

    where `n` is :func:`_n_sum_cw1d`.

    Args:
        l (integer): Order
        k (float or complex): Wave number
        kpar (float): Tangential wave vector component
        a (float): Lattice pitch
        r (float, (2,)-array): Shift vector
        eta (float or complex): Separation value

    Returns:
        complex
    """
    eta = _check_eta(eta, k, &a, 2, 1)
    if fabs(r[1]) < 1e-100:
        return recsumcw1d(l, k, kpar, a, r[0], eta)
    cdef double vec, x = r[0], y = r[1], b = 2 * pi / a
    if l < 0:
        y = -y
    cdef long absl = labs(l)
    cdef double complex recsum = 0, prev = INFINITY, pprev
    cdef number_t ky = k * y
    cdef long i
    cdef long point[1]
    for i in range(200):
        pprev = prev
        prev = recsum
        point[0] = -i
        while True:
            vec = kpar + b * point[0]
            recsum += _n_sum_cw1d(absl, vec / k, ky, eta) * cexp(-1j * vec * x)
            if not _misc.cubeedge_next(point, 1, i):
                break
        if cabs(recsum - pprev) < 1e-10:
            break
    return 2 * cpow(-1j, l) * recsum / (sqrtd(pi) * fabs(a) * k)


cdef double complex realsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    eta = _check_eta(eta, k, &a, 2, 1)
    if fabs(r[1]) < 1e-100:
        return realsumcw1d(l, k, kpar, a, r[0], eta)
    cdef double complex realsum = 0, prev = INFINITY, pprev
    cdef long i
    cdef double coord[2]
    cdef long point[1]
    cdef double vec
    for i in range(200):
        pprev = prev
        prev = realsum
        point[0] = -i
        while True:
            vec = a * point[0]
            coord[0] = -r[0] - vec
            coord[1] = -r[1]
            sc.car2pol(coord, coord)
            realsum += _realcw(l, k * coord[0], coord[1], eta) * cexp(
                1j * kpar * vec
            )
            if not _misc.cubeedge_next(point, 1, i):
                break
        if cabs(realsum - pprev) < 1e-10:
            break
    return -2j * realsum / pi


cdef double complex lsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    """See the documentation of :func:`treams.lattice.lsumcw1d_shift`"""
    eta = _check_eta(eta, k, &a, 2, 1)
    if fabs(r[1]) < 1e-100:
        return lsumcw1d(l, k, kpar, a, r[0], eta)
    return (
        realsumcw1d_shift(l, k, kpar, a, r, eta)
        + recsumcw1d_shift(l, k, kpar, a, r, eta)
    )
