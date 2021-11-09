"""Miscellaneous functions for ptsa.lattice"""

import numpy as np
cimport numpy as np
from libc.math cimport pi

from ptsa.special._misc cimport ipow


cdef real_t area(real_t *a, real_t *b) nogil:
    """(Signed) area between two-vectors a and b"""
    return a[0] * b[1] - a[1] * b[0]


cdef real_t volume(real_t *a, real_t *b, real_t *c) nogil:
    """(Signed) volume between three-vectors a, b, and c"""
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        + a[1] * (b[2] * c[0] - b[0] * c[2])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


cdef long ncube(long d, long n) nogil:
    """Number of points in a d-dimensional cube with side length 2n"""
    if n < 0 or d < 0:
        raise ValueError("only positive values")
    return ipow(2 * n + 1, d)


cdef long nedge(long d, long n) nogil:
    """Number of points on the surface of a d-dimensional cube with side length 2n"""
    if n < 0 or d < 0:
        raise ValueError("only positive values")
    if n == 0:
        return 1
    return ipow(2 * n + 1, d) - ipow(2 * n - 1, d)


cdef long cube_next(long *r, long d, long n) nogil:
    """
    Next point in a d-dimensional cube of side length 2n

    The array `r` is expected to be `d` long, the initial point to iterate over the whole
    cube is `(-n, -n, ..., -n)``. Returns 0 once `(n, n, ..., n)` is reached.
    """
    if d < 1:
        return 0
    if r[d - 1] == n:
        r[d - 1] = -n
        return cube_next(r, d - 1, n)
    r[d - 1] += 1
    return 1


cdef long cubeedge_next(long *r, long d, long n) nogil:
    """
    Next point on the surface of a d-dimensional cube of side length 2n

    The array `r` is expected to be `d` long, the initial point to iterate over the whole
    cube is `(-n, -n, ..., -n)``. Returns 0 once `(n, n, ..., n)` is reached.
    """
    if d < 1:
        return 0
    if r[0] == n:
        if cube_next(r + 1, d - 1, n):
            return 1
        r[0] = -n
        return 0
    if d == 1 and r[0] == -n:
        r[0] = n
        return 1
    if r[0] == -n:
        if not cube_next(r + 1, d - 1, n):
            r[0] += 1
    elif not cubeedge_next(r + 1, d - 1, n):
        r[0] += 1
    return 1


cdef void recvec2(double *a0, double *a1, double *b0, double *b1) nogil:
    """Reciprocal vectors in a two-dimensional lattice"""
    cdef double ar = area(a0, a1)
    if ar == 0:
        raise ValueError("vectors are linearly dependent")
    ar = 2 * pi / ar
    cdef double b00 = a1[1] * ar, b01 = -a1[0] * ar
    b1[0] = -a0[1] * ar
    b1[1] = a0[0] * ar
    b0[0] = b00
    b0[1] = b01
    return


cdef void recvec3(double *a0, double *a1, double *a2, double *b0, double *b1, double *b2) nogil:
    """Reciprocal vectors in a three-dimensional lattice"""
    cdef double vol = volume(a0, a1, a2)
    if vol == 0:
        raise ValueError("vectors are linearly dependent")
    vol = 2 * pi / vol
    cdef double b00 = (a1[1] * a2[2] - a1[2] * a2[1]) * vol
    cdef double b01 = (a1[2] * a2[0] - a1[0] * a2[2]) * vol
    cdef double b02 = (a1[0] * a2[1] - a1[1] * a2[0]) * vol
    cdef double b10 = (a2[1] * a0[2] - a2[2] * a0[1]) * vol
    cdef double b11 = (a2[2] * a0[0] - a2[0] * a0[2]) * vol
    cdef double b12 = (a2[0] * a0[1] - a2[1] * a0[0]) * vol
    b2[0] = (a0[1] * a1[2] - a0[2] * a1[1]) * vol
    b2[1] = (a0[2] * a1[0] - a0[0] * a1[2]) * vol
    b2[2] = (a0[0] * a1[1] - a0[1] * a1[0]) * vol
    b0[0] = b00
    b0[1] = b01
    b0[2] = b02
    b1[0] = b10
    b1[1] = b11
    b1[2] = b12
    return


cdef long _next_diff_or(double b[2][2], double rmax, long *mn) nogil:
    """
    Next diffraction order within the circle

    Given the reciprocal lattice defined by the vectors that make up the rows of `b`,
    returns the next diffraction order within a circle of radius `rmax`. `mn` is
    expected to be at least of size 2. With `[0, 0]` it gives all points, returning `1`
    and if exhausted returns `0`.

    Args:
        b (float, (2, 2)-array): Reciprocal lattice vectors
        rmax (float): Maximal radius
        mn (float, 2-array): On entry, previous lattice point, on exit

    Returns:
        bool
    """
    if rmax < 0:
        return 0
    if mn[0] > 0 or (mn[0] == 0 and mn[1] > 0):
        mn[0] *= -1
        mn[1] *= -1
        return 1
    cdef double x, y
    cdef long m = -mn[0], n = -mn[1]
    if n >= 0:
        x = m * b[0][0] + (n + 1) * b[1][0]
        y = m * b[0][1] + (n + 1) * b[1][1]
        if x * x + y * y <= rmax * rmax:
            mn[0] = m
            mn[1] = n + 1
            return 1
        n = 0
    if m != 0:
        x = m * b[0][0] + (n - 1) * b[1][0]
        y = m * b[0][1] + (n - 1) * b[1][1]
        if x * x + y * y <= rmax * rmax:
            mn[0] = m
            mn[1] = n - 1
            return 1
    m += 1
    if m * m * (b[0][0] * b[0][0] + b[0][1] * b[0][1]) <= rmax * rmax:
        mn[0] = m
        mn[1] = 0
        return 1
    return 0


cpdef diffr_orders_circle(real_t[:, :] b, double rmax):
    """
    Diffraction orders in a circle

    Given the reciprocal lattice defined by the vectors that make up the rows of `b`,
    return all diffraction orders within a circle of radius `rmax`.

    Args:
        b (float, (2, 2)-array): Reciprocal lattice vectors
        rmax (float): Maximal radius

    Returns:
        float array
    """
    if not b.shape == [2, 2, 0, 0, 0, 0, 0, 0]:  # this is: if not b.shape = (2, 2)
        raise ValueError("Wrong shape")
    cdef double arr[2][2]
    arr[0][0] = b[0, 0]
    arr[0][1] = b[0, 1]
    arr[1][0] = b[1, 0]
    arr[1][1] = b[1, 1]
    cdef long mn[2]
    mn[:] = [0, 0]
    cdef np.ndarray res = np.empty((0, 2), np.int64)
    if rmax < 0:
        return res
    res = np.append(res, mn)
    while _next_diff_or(arr, rmax, mn):
        res = np.append(res, mn)
    return res.reshape((-1, 2))


def cube(long d, long n):
    """
    cube(d, n)

    All integer points in a d-dimensional cube of sidelength 2n

    Args:
        d (int): Space dimension
        n (int): Size of the cube

    Returns:
        integer (k, d)-array (k is defined by the size of the cube)
    """
    if d <= 0 or d > 3:
        raise ValueError(f"Dimension is {d} but must be 1, 2, or 3")
    if n < 0:
        raise ValueError(f"{n} is negative")
    cdef long r[3]
    cdef long[:] rview = r
    cdef long i = 0
    cdef np.ndarray[long, ndim=2] res = np.empty((ncube(d, n), d), dtype=long)
    for i in range(3):
        r[i] = -n
    res[0, :] = rview[:d]
    i = 1
    while cube_next(r, d, n):
        res[i, :] = rview[:d]
        i += 1
    return res


def cubeedge(long d, long n):
    """
    cubeedge(d, n)

    All integer points on the surface of a d-dimensional cube of sidelength 2n

    Args:
        d (int): Space dimension
        n (int): Size of the cube

    Returns:
        integer (k, d)-array (k is defined by the size of the cube)
    """
    if d <= 0 or d > 3:
        raise ValueError(f"Dimension is {d} but must be 1, 2, or 3")
    if n < 0:
        raise ValueError(f"{n} is negative")
    cdef long r[3]
    cdef long[:] rview = r
    cdef long i = 0
    cdef np.ndarray[long, ndim=2] res = np.empty((nedge(d, n), d), dtype=long)
    for i in range(3):
        r[i] = -n
    res[0, :] = rview[:d]
    i = 1
    while cubeedge_next(r, d, n):
        res[i, :] = rview[:d]
        i += 1
    return res
