from libc.math cimport fabs, pi
from libc.math cimport sqrt as sqrtd
from scipy.linalg.cython_blas cimport dgemv
from scipy.special.cython_special cimport hankel1

from treams.lattice cimport _misc
from treams.special._misc cimport minusonepow
from treams.special.cython_special cimport car2pol, car2sph, sph_harm, spherical_hankel1


cdef extern from "<complex.h>" nogil:
    double complex cexp(double complex z)


cdef double complex _fsw3d(long l, long m, number_t k, double *kpar, double *a, double *r) nogil:
    """Summand of the direct summation in spherical solutions in 3d"""
    cdef double vec[3]
    vec[0] = -r[0] - a[0]
    vec[1] = -r[1] - a[1]
    vec[2] = -r[2] - a[2]
    car2sph(vec, vec)
    return (
        cexp(1j * (kpar[0] * a[0] + kpar[1] * a[1] + kpar[2] * a[2]))
        * spherical_hankel1(l, k * vec[0])
        * sph_harm(m, l, vec[2], vec[1])
    )


cdef double complex _fcw2d(long l, number_t k, double *kpar, double *a, double *r) nogil:
    """Summand of the direct summation in cylindrical solutions in 2d"""
    cdef double vec[2]
    vec[0] = -r[0] - a[0]
    vec[1] = -r[1] - a[1]
    car2pol(vec, vec)
    return cexp(1j * (kpar[0] * a[0] + kpar[1] * a[1] + l * vec[1])) * hankel1(l, k * vec[0])


cdef double complex _fsw2d(long l, long m, number_t k, double *kpar, double *a, double *r) nogil:
    """Summand of the direct summation in spherical solutions in 2d"""
    cdef double vec[2]
    vec[0] = -r[0] - a[0]
    vec[1] = -r[1] - a[1]
    car2pol(vec, vec)
    return (
        cexp(1j * (kpar[0] * a[0] + kpar[1] * a[1]))
        * spherical_hankel1(l, k * vec[0])
        * sph_harm(m, l, vec[1], pi * 0.5)
    )


cdef double complex _fsw1d(long l, number_t k, double kpar, double a, double r) nogil:
    """Summand of the direct summation in spherical solutions in 1d"""
    cdef double ra = r + a
    # P_l^0 (cos(theta(-r - vec))
    cdef double legp = minusonepow(l) if ra > 0 else 1
    return (
        cexp(1j * kpar * a)
        * spherical_hankel1(l, k * fabs(ra))
        * legp
        * sqrtd((2 * l + 1) / (4 * pi))
    )


cdef double complex _fcw1d(long l, number_t k, double kpar, double a, double r) nogil:
    """Summand of the direct summation in cylindrical solutions in 3d"""
    cdef double ra = r + a
    # exp(1j * l *phi(-r - a))
    cdef double expphi = minusonepow(l) if ra > 0 else 1
    return cexp(1j * kpar * a) * hankel1(l, k * fabs(ra)) * expphi


cdef double complex dsumcw2d(long l, number_t k, double *kpar, double *a, double *r, long i) nogil:
    """See the documentation of :func:`treams.lattice.dsumcw2d`"""
    cdef double complex res = 0
    if r[0] == 0 and r[1] == 0 and i == 0:
        return res
    cdef double pointf[2]
    cdef long point[2]
    cdef double vec[2]
    cdef char c = b'N'
    point[0] = -i
    point[1] = -i
    cdef double one = 1, zero = 0
    cdef int dim = 2, inc = 1
    while True:
        pointf[0] = <double>point[0]
        pointf[1] = <double>point[1]
        dgemv(&c, &dim, &dim, &one, a, &dim, pointf, &inc, &zero, vec, &inc)
        res += _fcw2d(l, k, kpar, vec, r)
        if not _misc.cubeedge_next(point, dim, i):
            break
    return res


cdef double complex dsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil:
    """See the documentation of :func:`treams.lattice.dsumsw2d`"""
    cdef double complex res = 0
    if (l + m) % 2 == 1 or (r[0] == 0 and r[1] == 0 and i == 0):
        return res
    cdef double pointf[2]
    cdef long point[2]
    cdef double vec[2]
    cdef char c = b'N'
    point[0] = -i
    point[1] = -i
    cdef double one = 1, zero = 0
    cdef int dim = 2, inc = 1
    while True:
        pointf[0] = <double>point[0]
        pointf[1] = <double>point[1]
        dgemv(&c, &dim, &dim, &one, a, &dim, pointf, &inc, &zero, vec, &inc)
        res += _fsw2d(l, m, k, kpar, vec, r)
        if not _misc.cubeedge_next(point, dim, i):
            break
    return res


cdef double complex dsumsw1d(long l, number_t k, double kpar, double a, double r, long i) nogil:
    """See the documentation of :func:`treams.lattice.dsumsw1d`"""
    if (r == 0 and i == 0):
        return 0
    if fabs(a * 0.5) == fabs(r):
        return _fsw1d(l, k, kpar, fabs(a) * i, fabs(r)) + _fsw1d(l, k, kpar, -fabs(a) * (i + 1), fabs(r))
    if i == 0:
        return _fsw1d(l, k, kpar, 0, r)
    return _fsw1d(l, k, kpar, a * i, r) + _fsw1d(l, k, kpar, -a * i, r)


cdef double complex dsumcw1d(long l, number_t k, double kpar, double a, double r, long i) nogil:
    """See the documentation of :func:`treams.lattice.dsumcw1d`"""
    if (r == 0 and i == 0):
        return 0
    if fabs(a * 0.5) == fabs(r):
        return _fcw1d(l, k, kpar, fabs(a) * i, fabs(r)) + _fcw1d(l, k, kpar, -fabs(a) * (i + 1), fabs(r))
    if i == 0:
        return _fcw1d(l, k, kpar, 0, r)
    return _fcw1d(l, k, kpar, a * i, r) + _fcw1d(l, k, kpar, -a * i, r)


cdef double complex dsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil:
    """See the documentation of :func:`treams.lattice.dsumsw3d`"""
    cdef double complex res = 0
    if r[0] == 0 and r[1] == 0 and r[2] == 0 and i == 0:
        return res
    cdef double pointf[3]
    cdef long point[3]
    cdef double vec[3]
    cdef char c = b'N'
    point[0] = -i
    point[1] = -i
    point[2] = -i
    cdef double one = 1, zero = 0
    cdef int dim = 3, inc = 1
    while True:
        pointf[0] = <double>point[0]
        pointf[1] = <double>point[1]
        pointf[2] = <double>point[2]
        dgemv(&c, &dim, &dim, &one, a, &dim, pointf, &inc, &zero, vec, &inc)
        res += _fsw3d(l, m, k, kpar, vec, r)
        if not _misc.cubeedge_next(point, dim, i):
            break
    return res


cdef double complex dsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil:
    """See the documentation of :func:`treams.lattice.dsumsw2d_shift`"""
    cdef double complex res = 0
    if r[0] == 0 and r[1] == 0 and r[2] == 0 and i == 0:
        return res
    cdef double pointf[2]
    cdef long point[2]
    cdef double vec[3]
    vec[2] = 0
    cdef char c = b'N'
    point[0] = -i
    point[1] = -i
    cdef double one = 1, zero = 0
    cdef int dim = 2, inc = 1
    while True:
        pointf[0] = <double>point[0]
        pointf[1] = <double>point[1]
        dgemv(&c, &dim, &dim, &one, a, &dim, pointf, &inc, &zero, vec, &inc)
        res += _fsw3d(l, m, k, kpar, vec, r)
        if not _misc.cubeedge_next(point, dim, i):
            break
    return res


cdef double complex dsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, long i) nogil:
    """See the documentation of :func:`treams.lattice.dsumsw1d_shift`"""
    if r[1] == 0 and r[2] == 0:
        if m != 0:
            return 0
        return dsumsw1d(l, k, kpar, a, r[0], i)
    cdef double complex res
    cdef double vec[3]
    vec[0] = 0
    vec[1] = 0
    cdef double kparvec[3]
    kparvec[0] = 0
    kparvec[1] = 0
    kparvec[2] = kpar
    vec[2] = a * i
    res = _fsw3d(l, m, k, kparvec, vec, r)
    if i == 0:
        return res
    vec[2] = -a * i
    res += _fsw3d(l, m, k, kparvec, vec, r)
    return res


cdef double complex dsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, long i) nogil:
    """See the documentation of :func:`treams.lattice.dsumcw1d_shift`"""
    if r[1] == 0:
        return dsumcw1d(l, k, kpar, a, r[0], i)
    cdef double complex res
    cdef double vec[2]
    vec[1] = 0
    cdef double kparvec[2]
    kparvec[0] = kpar
    kparvec[1] = 0
    vec[0] = a * i
    res = _fcw2d(l, k, kparvec, vec, r)
    if i == 0:
        return res
    vec[0] = -a * i
    res += _fcw2d(l, k, kparvec, vec, r)
    return res
