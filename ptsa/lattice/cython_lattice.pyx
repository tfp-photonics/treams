"""Cython versions of lattice functions.

This module defines the public interface of the implemented Cython functions that are
underlying the ufuncs of the module.
"""

from ptsa.lattice cimport _dsum
from ptsa.lattice cimport _esum
from ptsa.lattice cimport _misc


cdef real_t area(real_t *a, real_t *b) nogil:
    """See the documentation of :func:`ptsa.lattice._misc.area`."""
    return _misc.area(a, b)
cdef long cube_next(long *r, long d, long n) nogil:
    """See the documentation of :func:`ptsa.lattice._misc.cube_next`."""
    return _misc.cube_next(r, d, n)
cdef long cubeedge_next(long *r, long d, long n) nogil:
    """See the documentation of :func:`ptsa.lattice._misc.cubeedge_next`."""
    return _misc.cubeedge_next(r, d, n)
cpdef diffr_orders_circle(real_t[:, :] b, double rmax):
    """diffr_orders_circle(b, rmax)

    See the documentation of :func:`ptsa.lattice.diffr_orders_circle`.
    """
    return _misc.diffr_orders_circle(b, rmax)
cpdef long ncube(long d, long n) nogil:
    r"""ncube(d, n)

    Number of points in a d-dimensional cube with side length 2n.

    Computes :math:`(2n + 1)^d`.

    Args:
        d (int): Spatial dimensions
        n (int): Half size

    Returns:
        int
    """
    return _misc.ncube(d, n)
cpdef long nedge(long d, long n) nogil:
    r"""nedge(d, n)

    Number of points on the surface of a d-dimensional cube with side length 2n.

    Computes :math:`(2n + 1)^d - (2n - 1)^d` and for :math:`n = 0` returns :math:`1`.

    Args:
        d (int): Spatial dimensions
        n (int): Half size

    Returns:
        int
    """
    return _misc.nedge(d, n)
cdef void recvec2(double *a0, double *a1, double *b0, double *b1) nogil:
    """See the documentation of :func:`ptsa.lattice._misc.recvec2`."""
    _misc.recvec2(a0, a1, b0, b1)
cdef void recvec3(double *a0, double *a1, double *a2, double *b0, double *b1, double *b2) nogil:
    """See the documentation of :func:`ptsa.lattice._misc.recvec3`."""
    _misc.recvec3(a0, a1, a2, b0, b1, b2)
cdef real_t volume(real_t *a, real_t *b, real_t *c) nogil:
    """See the documentation of :func:`ptsa.lattice._misc.volume`."""
    return _misc.volume(a, b, c)


cpdef double complex lsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    """lsumcw1d(l, k, kpar, a, r, eta)

    See the documentation of :func:`ptsa.lattice.lsumcw1d`.
    """
    return _esum.lsumcw1d(l, k, kpar, a, r, eta)
cdef double complex lsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.lsumcw1d_shift`."""
    return _esum.lsumcw1d_shift(l, k, kpar, a, r, eta)
cdef double complex lsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.lsumcw2d`."""
    return _esum.lsumcw2d(l, k, kpar, a, r, eta)
cpdef double complex lsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    """lsumsw1d(l, k, kpar, a, r, eta)

    See the documentation of :func:`ptsa.lattice.lsumsw1d`.
    """
    return _esum.lsumsw1d(l, k, kpar, a, r, eta)
cdef double complex lsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.lsumsw1d_shift`."""
    return _esum.lsumsw1d_shift(l, m, k, kpar, a, r, eta)
cdef double complex lsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.lsumsw2d`."""
    return _esum.lsumsw2d(l, m, k, kpar, a, r, eta)
cdef double complex lsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.lsumsw2d_shift`."""
    return _esum.lsumsw2d_shift(l, m, k, kpar, a, r, eta)
cdef double complex lsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.lsumsw3d`."""
    return _esum.lsumsw3d(l, m, k, kpar, a, r, eta)


cpdef double complex realsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    """realsumcw1d(l, k, kpar, a, r, eta)

    See the documentation of :func:`ptsa.lattice.realsumcw1d`.
    """
    return _esum.realsumcw1d(l, k, kpar, a, r, eta)
cdef double complex realsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.realsumcw1d_shift`."""
    return _esum.realsumcw1d_shift(l, k, kpar, a, r, eta)
cdef double complex realsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.realsumcw2d`."""
    return _esum.realsumcw2d(l, k, kpar, a, r, eta)
cpdef double complex realsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    """realsumsw1d(l, k, kpar, a, r, eta)

    See the documentation of :func:`ptsa.lattice.realsumsw1d`.
    """
    return _esum.realsumsw1d(l, k, kpar, a, r, eta)
cdef double complex realsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.realsumsw1d_shift`."""
    return _esum.realsumsw1d_shift(l, m, k, kpar, a, r, eta)
cdef double complex realsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.realsumsw2d`."""
    return _esum.realsumsw2d(l, m, k, kpar, a, r, eta)
cdef double complex realsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.realsumsw2d_shift`."""
    return _esum.realsumsw2d_shift(l, m, k, kpar, a, r, eta)
cdef double complex realsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.realsumsw3d`."""
    return _esum.realsumsw3d(l, m, k, kpar, a, r, eta)


cpdef double complex recsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    """recsumcw1d(l, k, kpar, a, r, eta)

    See the documentation of :func:`ptsa.lattice.recsumcw1d`.
    """
    return _esum.recsumcw1d(l, k, kpar, a, r, eta)
cdef double complex recsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.recsumcw1d_shift`."""
    return _esum.recsumcw1d_shift(l, k, kpar, a, r, eta)
cdef double complex recsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.recsumcw2d`."""
    return _esum.recsumcw2d(l, k, kpar, a, r, eta)
cpdef double complex recsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil:
    """recsumsw1d(l, k, kpar, a, r, eta)

    See the documentation of :func:`ptsa.lattice.recsumsw1d`.
    """
    return _esum.recsumsw1d(l, k, kpar, a, r, eta)
cdef double complex recsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.recsumsw1d_shift`."""
    return _esum.recsumsw1d_shift(l, m, k, kpar, a, r, eta)
cdef double complex recsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.recsumsw2d`."""
    return _esum.recsumsw2d(l, m, k, kpar, a, r, eta)
cdef double complex recsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.recsumsw2d_shift`."""
    return _esum.recsumsw2d_shift(l, m, k, kpar, a, r, eta)
cdef double complex recsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil:
    """See the documentation of :func:`ptsa.lattice.recsumsw3d`."""
    return _esum.recsumsw3d(l, m, k, kpar, a, r, eta)


cpdef double complex zero3d(double complex eta) nogil:
    return _esum.zero3d(eta)
cpdef double complex zero2d(double complex eta) nogil:
    return _esum.zero2d(eta)

cpdef double complex dsumcw1d(long l, number_t k, double kpar, double a, double r, long i) nogil:
    """dsumcw1d(l, k, kpar, a, r, i)

    See the documentation of :func:`ptsa.lattice.dsumcw1d`.
    """
    return _dsum.dsumcw1d(l, k, kpar, a, r, i)
cdef double complex dsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, long i) nogil:
    """See the documentation of :func:`ptsa.lattice.dsumcw1d_shift`."""
    return _dsum.dsumcw1d_shift(l, k, kpar, a, r, i)
cdef double complex dsumcw2d(long l, number_t k, double *kpar, double *a, double *r, long i) nogil:
    """See the documentation of :func:`ptsa.lattice.dsumcw2d`."""
    return _dsum.dsumcw2d(l, k, kpar, a, r, i)
cpdef double complex dsumsw1d(long l, number_t k, double kpar, double a, double r, long i) nogil:
    """dsumsw1d(l, k, kpar, a, r, eta)

    See the documentation of :func:`ptsa.lattice.dsumsw1d`.
    """
    return _dsum.dsumsw1d(l, k, kpar, a, r, i)
cdef double complex dsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, long i) nogil:
    """See the documentation of :func:`ptsa.lattice.dsumsw1d_shift`."""
    return _dsum.dsumsw1d_shift(l, m, k, kpar, a, r, i)
cdef double complex dsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil:
    """See the documentation of :func:`ptsa.lattice.dsumsw2d`."""
    return _dsum.dsumsw2d(l, m, k, kpar, a, r, i)
cdef double complex dsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil:
    """See the documentation of :func:`ptsa.lattice.dsumsw2d_shift`."""
    return _dsum.dsumsw2d_shift(l, m, k, kpar, a, r, i)
cdef double complex dsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil:
    """See the documentation of :func:`ptsa.lattice.dsumsw3d`."""
    return _dsum.dsumsw3d(l, m, k, kpar, a, r, i)
