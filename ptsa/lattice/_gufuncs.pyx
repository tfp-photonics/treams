"""Generalized universal functions for ptsa.lattice"""

cimport numpy as np

from ptsa.lattice cimport _misc
from ptsa.special._misc cimport double_complex
from ptsa.lattice cimport _esum
from ptsa.lattice cimport _dsum

__all__ = [
    'area',
    'dsumcw1d',
    'dsumcw1d_shift',
    'dsumcw2d',
    'dsumsw1d',
    'dsumsw1d_shift',
    'dsumsw2d',
    'dsumsw2d_shift',
    'dsumsw3d',
    'lsumcw1d',
    'lsumcw1d_shift',
    'lsumcw2d',
    'lsumsw1d',
    'lsumsw1d_shift',
    'lsumsw2d',
    'lsumsw2d_shift',
    'lsumsw3d',
    'realsumcw1d',
    'realsumcw1d_shift',
    'realsumcw2d',
    'realsumsw1d',
    'realsumsw1d_shift',
    'realsumsw2d',
    'realsumsw2d_shift',
    'realsumsw3d',
    'recsumcw1d',
    'recsumcw1d_shift',
    'recsumcw2d',
    'recsumsw1d',
    'recsumsw1d_shift',
    'recsumsw2d',
    'recsumsw2d_shift',
    'recsumsw3d',
    'reciprocal',
    'volume',
#    'zero2d',
#    'zero3d',
]

cdef void loop_volume_l(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    if dims[1] < 2 or dims[1] > 3:
        raise ValueError('Dimension out of range')  # todo: force error by numpy
    cdef long input[3][3]
    cdef void *area = (<void**>data)[0]
    cdef void *vol = (<void**>data)[1]
    cdef char *ip0 = args[0]
    cdef char *op0 = args[1]
    cdef long ov0
    for i in range(n):
        for j in range(dims[1]):
            for k in range(dims[1]):
                input[j][k] = (<long*>(ip0 + j * steps[2] + k * steps[3]))[0]
        if dims[1] == 2:
            ov0 = (<long (*)(long*, long*) nogil>area)(input[0], input[1])
        else:
            ov0 = (<long (*)(long*, long*, long*) nogil>vol)(input[0], input[1], input[2])
        (<long*>op0)[0] = <long>ov0
        ip0 += steps[0]
        op0 += steps[1]


cdef void loop_volume_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    if dims[1] < 2 or dims[1] > 3:
        raise ValueError('Dimension out of range')  # todo: force error by numpy
    cdef double input[3][3]
    cdef void *area = (<void**>data)[0]
    cdef void *vol = (<void**>data)[1]
    cdef char *ip0 = args[0]
    cdef char *op0 = args[1]
    cdef double ov0
    for i in range(n):
        for j in range(dims[1]):
            for k in range(dims[1]):
                input[j][k] = (<double*>(ip0 + j * steps[2] + k * steps[3]))[0]
        if dims[1] == 2:
            ov0 = (<double (*)(double*, double*) nogil>area)(input[0], input[1])
        else:
            ov0 = (<double (*)(double*, double*, double*) nogil>vol)(input[0], input[1], input[2])
        (<double*>op0)[0] = <double>ov0
        ip0 += steps[0]
        op0 += steps[1]


cdef void loop_reciprocal_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    if dims[1] < 2 or dims[1] > 3:
        raise ValueError('Dimension out of range')  # todo: force error by numpy
    cdef double input[3][3]
    cdef double output[3][3]
    cdef void *area = (<void**>data)[0]
    cdef void *vol = (<void**>data)[1]
    cdef char *ip0 = args[0]
    cdef char *op0 = args[1]
    for i in range(n):
        for j in range(dims[1]):
            for k in range(dims[1]):
                input[j][k] = (<double*>(ip0 + j * steps[2] + k * steps[3]))[0]
        if dims[1] == 2:
            (<void (*)(double*, double*, double*, double*) nogil>area)(input[0], input[1], output[0], output[1])
        else:
            (<void (*)(double*, double*, double*, double*, double*, double*) nogil>vol)(input[0], input[1], input[2], output[0], output[1], output[2])
        for j in range(dims[1]):
            for k in range(dims[1]):
                (<double*>(op0 + j * steps[4] + k * steps[5]))[0] = output[j][k]
        ip0 += steps[0]
        op0 += steps[1]


cdef void loop_lsumcw2d_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip2 + j * steps[7]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip3 + j * steps[8] + k * steps[9]))[0]
            r[j] = (<double*>(ip4 + j * steps[10]))[0]
        ov0 = (<double complex (*)(long, double, double*, double*, double*, double) nogil>func)(<long>(<long*>ip0)[0], <double>(<double*>ip1)[0], kpar, a, r, <double>(<double*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_lsumcw2d_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip2 + j * steps[7]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip3 + j * steps[8] + k * steps[9]))[0]
            r[j] = (<double*>(ip4 + j * steps[10]))[0]
        ov0 = (<double complex (*)(long, double complex, double*, double*, double*, double complex) nogil>func)(<long>(<long*>ip0)[0], <double complex>(<double complex*>ip1)[0], kpar, a, r, <double complex>(<double complex*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_lsumsw2d_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double, double*, double*, double*, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], kpar, a, r, <double>(<double*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_lsumsw2d_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double complex, double*, double*, double*, double complex) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], kpar, a, r, <double complex>(<double complex*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_lsumsw3d_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[3]
    cdef double a[9]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(3):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(3):
                a[3 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double, double*, double*, double*, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], kpar, a, r, <double>(<double*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_lsumsw3d_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[3]
    cdef double a[9]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(3):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(3):
                a[3 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double complex, double*, double*, double*, double complex) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], kpar, a, r, <double complex>(<double complex*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_lsumsw2d_shift_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        r[2] = (<double*>(ip5 + 2 * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double, double*, double*, double*, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], kpar, a, r, <double>(<double*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_lsumsw2d_shift_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        r[2] = (<double*>(ip5 + 2 * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double complex, double*, double*, double*, double complex) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], kpar, a, r, <double complex>(<double complex*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_lsumsw1d_shift_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(3):
            r[j] = (<double*>(ip5 + j * steps[8]))[0]
        ov0 = (<double complex (*)(long, long, double, double, double, double*, double) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], r, <double>(<double*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_lsumsw1d_shift_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(3):
            r[j] = (<double*>(ip5 + j * steps[8]))[0]
        ov0 = (<double complex (*)(long, long, double complex, double, double, double*, double complex) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], r, <double complex>(<double complex*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_lsumcw1d_shift_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            r[j] = (<double*>(ip4 + j * steps[7]))[0]
        ov0 = (<double complex (*)(long, double, double, double, double*, double) nogil>func)(<long>(<long*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], r, <double>(<double*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_lsumcw1d_shift_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            r[j] = (<double*>(ip4 + j * steps[7]))[0]
        ov0 = (<double complex (*)(long, double complex, double, double, double*, double complex) nogil>func)(<long>(<long*>ip0)[0], <double complex>(<double complex*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], r, <double complex>(<double complex*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_lsum1d_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex (*)(long, double, double, double, double, double) nogil>func)(<long>(<long*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double>(<double*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_lsum1d_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex (*)(long, double complex, double, double, double, double complex) nogil>func)(<long>(<long*>ip0)[0], <double complex>(<double complex*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double complex>(<double complex*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_dsumcw2d_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip2 + j * steps[7]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip3 + j * steps[8] + k * steps[9]))[0]
            r[j] = (<double*>(ip4 + j * steps[10]))[0]
        ov0 = (<double complex (*)(long, double, double*, double*, double*, long) nogil>func)(<long>(<long*>ip0)[0], <double>(<double*>ip1)[0], kpar, a, r, <long>(<long*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_dsumcw2d_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip2 + j * steps[7]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip3 + j * steps[8] + k * steps[9]))[0]
            r[j] = (<double*>(ip4 + j * steps[10]))[0]
        ov0 = (<double complex (*)(long, double complex, double*, double*, double*, long) nogil>func)(<long>(<long*>ip0)[0], <double complex>(<double complex*>ip1)[0], kpar, a, r, <long>(<long*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_dsumsw2d_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double, double*, double*, double*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], kpar, a, r, <long>(<long*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_dsumsw2d_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double complex, double*, double*, double*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], kpar, a, r, <long>(<long*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_dsumsw3d_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[3]
    cdef double a[9]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(3):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(3):
                a[3 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double, double*, double*, double*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], kpar, a, r, <long>(<long*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_dsumsw3d_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[3]
    cdef double a[9]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(3):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(3):
                a[3 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double complex, double*, double*, double*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], kpar, a, r, <long>(<long*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_dsumsw2d_shift_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        r[2] = (<double*>(ip5 + 2 * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double, double*, double*, double*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], kpar, a, r, <long>(<long*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_dsumsw2d_shift_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, k, n = dims[0]
    cdef double kpar[2]
    cdef double a[4]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            kpar[j] = (<double*>(ip3 + j * steps[8]))[0]
            for k in range(2):
                a[2 * j + k] = (<double*>(ip4 + j * steps[9] + k * steps[10]))[0]
            r[j] = (<double*>(ip5 + j * steps[11]))[0]
        r[2] = (<double*>(ip5 + 2 * steps[11]))[0]
        ov0 = (<double complex (*)(long, long, double complex, double*, double*, double*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], kpar, a, r, <long>(<long*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_dsumsw1d_shift_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(3):
            r[j] = (<double*>(ip5 + j * steps[8]))[0]
        ov0 = (<double complex (*)(long, long, double, double, double, double*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], r, <long>(<long*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_dsumsw1d_shift_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double r[3]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    cdef double complex ov0
    for i in range(n):
        for j in range(3):
            r[j] = (<double*>(ip5 + j * steps[8]))[0]
        ov0 = (<double complex (*)(long, long, double complex, double, double, double*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], r, <long>(<long*>ip6)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_dsumcw1d_shift_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            r[j] = (<double*>(ip4 + j * steps[7]))[0]
        ov0 = (<double complex (*)(long, double, double, double, double*, long) nogil>func)(<long>(<long*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], r, <long>(<long*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_dsumcw1d_shift_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, j, n = dims[0]
    cdef double r[2]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        for j in range(2):
            r[j] = (<double*>(ip4 + j * steps[7]))[0]
        ov0 = (<double complex (*)(long, double complex, double, double, double*, long) nogil>func)(<long>(<long*>ip0)[0], <double complex>(<double complex*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], r, <long>(<long*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_dsum1d_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex (*)(long, double, double, double, double, long) nogil>func)(<long>(<long*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <long>(<long*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_dsum1d_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex (*)(long, double complex, double, double, double, long) nogil>func)(<long>(<long*>ip0)[0], <double complex>(<double complex*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <long>(<long*>ip5)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


np.import_array()
np.import_ufunc()


cdef np.PyUFuncGenericFunction gufunc_reciprocal_loops[1]
cdef void *gufunc_reciprocal_ptr[2]
cdef void *gufunc_reciprocal_data[1]
cdef char gufunc_reciprocal_types[2]

gufunc_reciprocal_loops[0] = <np.PyUFuncGenericFunction>loop_reciprocal_d
gufunc_reciprocal_types[0] = <char>np.NPY_DOUBLE
gufunc_reciprocal_types[1] = <char>np.NPY_DOUBLE
gufunc_reciprocal_ptr[0] = <void*>_misc.recvec2
gufunc_reciprocal_ptr[1] = <void*>_misc.recvec3
gufunc_reciprocal_data[0] = &gufunc_reciprocal_ptr[0]


reciprocal = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_reciprocal_loops,
    gufunc_reciprocal_data,
    gufunc_reciprocal_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'reciprocal',  # function name
    """reciprocal(a)

Reciprocal vectors in two- and three-dimensional space

Calculate the reciprocal vectors to the lattice vectors given as rows of `a`.

Args:
    a (float, (2,2)- or (3,3)-array): Lattice vectors

Returns:
    float, (2,2)- or (3,3)-array
""",  # docstring
    0,  # unused
    '(i,i)->(i,i)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_volume_loops[2]
cdef void *gufunc_volume_ptr[2 * 2]
cdef void *gufunc_volume_data[2]
cdef char gufunc_volume_types[2 * 2]

gufunc_volume_loops[0] = <np.PyUFuncGenericFunction>loop_volume_l
gufunc_volume_loops[1] = <np.PyUFuncGenericFunction>loop_volume_d
gufunc_volume_types[0] = <char>np.NPY_LONG
gufunc_volume_types[1] = <char>np.NPY_LONG
gufunc_volume_types[2] = <char>np.NPY_DOUBLE
gufunc_volume_types[3] = <char>np.NPY_DOUBLE
gufunc_volume_ptr[0] = <void*>_misc.area[long]
gufunc_volume_ptr[1] = <void*>_misc.volume[long]
gufunc_volume_ptr[2] = <void*>_misc.area[double]
gufunc_volume_ptr[3] = <void*>_misc.volume[double]
gufunc_volume_data[0] = &gufunc_volume_ptr[0]
gufunc_volume_data[1] = &gufunc_volume_ptr[2]

volume = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_volume_loops,
    gufunc_volume_data,
    gufunc_volume_types,
    2,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'volume',  # function name
    """volume(a)

Calculate the signed volume (area)

From the vectors given in `a`, calculate the volume or area. This value is signed
showing the handedness of the system. For a two-dimensional lattice calculates the
area and the function has therefore as alias the name `area`.

Args:
    a (float, (2,2)- or (3,3)-array): Lattice vectors

Returns:
    float
""",  # docstring
    0,  # unused
    '(i,i)->()',  # signature
)

area = volume


cdef np.PyUFuncGenericFunction gufunc_lsumcw2d_loops[2]
cdef void *gufunc_lsumcw2d_data[2]
cdef void *gufunc_realsumcw2d_data[2]
cdef void *gufunc_recsumcw2d_data[2]
cdef char gufunc_lsumcw2d_types[2 * 7]

gufunc_lsumcw2d_loops[0] = <np.PyUFuncGenericFunction>loop_lsumcw2d_d
gufunc_lsumcw2d_loops[1] = <np.PyUFuncGenericFunction>loop_lsumcw2d_D
gufunc_lsumcw2d_types[0] = <char>np.NPY_LONG
gufunc_lsumcw2d_types[1] = <char>np.NPY_DOUBLE
gufunc_lsumcw2d_types[2] = <char>np.NPY_DOUBLE
gufunc_lsumcw2d_types[3] = <char>np.NPY_DOUBLE
gufunc_lsumcw2d_types[4] = <char>np.NPY_DOUBLE
gufunc_lsumcw2d_types[5] = <char>np.NPY_CDOUBLE
gufunc_lsumcw2d_types[6] = <char>np.NPY_CDOUBLE
gufunc_lsumcw2d_types[7] = <char>np.NPY_LONG
gufunc_lsumcw2d_types[8] = <char>np.NPY_CDOUBLE
gufunc_lsumcw2d_types[9] = <char>np.NPY_DOUBLE
gufunc_lsumcw2d_types[10] = <char>np.NPY_DOUBLE
gufunc_lsumcw2d_types[11] = <char>np.NPY_DOUBLE
gufunc_lsumcw2d_types[12] = <char>np.NPY_CDOUBLE
gufunc_lsumcw2d_types[13] = <char>np.NPY_CDOUBLE
gufunc_lsumcw2d_data[0] = <void*>_esum.lsumcw2d[double]
gufunc_lsumcw2d_data[1] = <void*>_esum.lsumcw2d[double_complex]
gufunc_realsumcw2d_data[0] = <void*>_esum.realsumcw2d[double]
gufunc_realsumcw2d_data[1] = <void*>_esum.realsumcw2d[double_complex]
gufunc_recsumcw2d_data[0] = <void*>_esum.recsumcw2d[double]
gufunc_recsumcw2d_data[1] = <void*>_esum.recsumcw2d[double_complex]

lsumcw2d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumcw2d_loops,
    gufunc_lsumcw2d_data,
    gufunc_lsumcw2d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'lsumcw2d',  # function name
    r"""lsumcw2d(l, k, kpar, a, r, eta)

Fast summation of cylindrical functions on a 2d lattice

Computes

.. math::

    D_{l}(k, \boldsymbol k_\parallel, \boldsymbol r, \Lambda_2)
    = \sum_{\boldsymbol R \in \Lambda_2}
    H_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    \mathrm e^{\mathrm i l \varphi_{-\boldsymbol r - \boldsymbol R}}
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

using the Ewald summation.

The cut between the real and reciprocal space summation is defined by `eta`. Larger
values increase the weight of the real sum. In `a` the lattice vectors are
given as rows.

Args:
    l (integer): Order
    k (float or complex): Wave number
    kpar (float, (2,)-array): Wave vector
    a (float, (2,2)-array): Lattice vectors
    r (float, (2,)-array): Shift vector
    eta (float or complex): Separation value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(2),(2,2),(2),()->()',  # signature
)
realsumcw2d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumcw2d_loops,
    gufunc_realsumcw2d_data,
    gufunc_lsumcw2d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'realsumcw2d',  # function name
    r"""realsumcw2d(l, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(2),(2,2),(2),()->()',  # signature
)
recsumcw2d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumcw2d_loops,
    gufunc_recsumcw2d_data,
    gufunc_lsumcw2d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'recsumcw2d',  # function name
    r"""recsumcw2d(l, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(2),(2,2),(2),()->()',  # signature
)

cdef np.PyUFuncGenericFunction gufunc_lsumsw3d_loops[2]
cdef void *gufunc_lsumsw3d_data[2]
cdef void *gufunc_realsumsw3d_data[2]
cdef void *gufunc_recsumsw3d_data[2]
cdef char gufunc_lsumsw3d_types[2 * 8]

gufunc_lsumsw3d_loops[0] = <np.PyUFuncGenericFunction>loop_lsumsw3d_d
gufunc_lsumsw3d_loops[1] = <np.PyUFuncGenericFunction>loop_lsumsw3d_D
gufunc_lsumsw3d_types[0] = <char>np.NPY_LONG
gufunc_lsumsw3d_types[1] = <char>np.NPY_LONG
gufunc_lsumsw3d_types[2] = <char>np.NPY_DOUBLE
gufunc_lsumsw3d_types[3] = <char>np.NPY_DOUBLE
gufunc_lsumsw3d_types[4] = <char>np.NPY_DOUBLE
gufunc_lsumsw3d_types[5] = <char>np.NPY_DOUBLE
gufunc_lsumsw3d_types[6] = <char>np.NPY_DOUBLE
gufunc_lsumsw3d_types[7] = <char>np.NPY_CDOUBLE
gufunc_lsumsw3d_types[8] = <char>np.NPY_LONG
gufunc_lsumsw3d_types[9] = <char>np.NPY_LONG
gufunc_lsumsw3d_types[10] = <char>np.NPY_CDOUBLE
gufunc_lsumsw3d_types[11] = <char>np.NPY_DOUBLE
gufunc_lsumsw3d_types[12] = <char>np.NPY_DOUBLE
gufunc_lsumsw3d_types[13] = <char>np.NPY_DOUBLE
gufunc_lsumsw3d_types[14] = <char>np.NPY_CDOUBLE
gufunc_lsumsw3d_types[15] = <char>np.NPY_CDOUBLE
gufunc_lsumsw3d_data[0] = <void*>_esum.lsumsw3d[double]
gufunc_lsumsw3d_data[1] = <void*>_esum.lsumsw3d[double_complex]
gufunc_realsumsw3d_data[0] = <void*>_esum.realsumsw3d[double]
gufunc_realsumsw3d_data[1] = <void*>_esum.realsumsw3d[double_complex]
gufunc_recsumsw3d_data[0] = <void*>_esum.recsumsw3d[double]
gufunc_recsumsw3d_data[1] = <void*>_esum.recsumsw3d[double_complex]

lsumsw3d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw3d_loops,
    gufunc_lsumsw3d_data,
    gufunc_lsumsw3d_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'lsumsw3d',  # function name
    r"""lsumsw3d(l, m, k, kpar, a, r, eta)

Fast summation of spherical functions on a 3d lattice

Computes

.. math::

    D_{lm}(k, \boldsymbol k_\parallel, \Lambda_3, \boldsymbol r)
    = \sum_{\boldsymbol R \in \Lambda_3}
    h_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    Y_{lm}(-\boldsymbol r - \boldsymbol R)
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

using the Ewald summation.

The cut between the real and reciprocal space summation is defined by `eta`. Larger
values increase the weight of the real sum. In `a` the lattice vectors are
given as rows.

Args:
    l (integer): Degree :math:`l \geq 0`
    m (integer): Order :math:`|m| \leq l`
    k (float or complex): Wave number
    kpar (float, (3,)-array): Wave vector
    a (float, (3,3)-array): Lattice vectors
    r (float, (3,)-array): Shift vector
    eta (float or complex): Separation value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(),(3),(3,3),(3),()->()',  # signature
)
realsumsw3d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw3d_loops,
    gufunc_realsumsw3d_data,
    gufunc_lsumsw3d_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'realsumsw3d',  # function name
    r"""realsumsw3d(l, m, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(3),(3,3),(3),()->()',  # signature
)
recsumsw3d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw3d_loops,
    gufunc_recsumsw3d_data,
    gufunc_lsumsw3d_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'recsumsw3d',  # function name
    r"""recsumsw3d(l, m, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(3),(3,3),(3),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_lsumsw2d_loops[2]
cdef void *gufunc_lsumsw2d_data[2]
cdef void *gufunc_realsumsw2d_data[2]
cdef void *gufunc_recsumsw2d_data[2]
cdef char gufunc_lsumsw2d_types[2 * 8]

gufunc_lsumsw2d_loops[0] = <np.PyUFuncGenericFunction>loop_lsumsw2d_d
gufunc_lsumsw2d_loops[1] = <np.PyUFuncGenericFunction>loop_lsumsw2d_D
gufunc_lsumsw2d_types[0] = <char>np.NPY_LONG
gufunc_lsumsw2d_types[1] = <char>np.NPY_LONG
gufunc_lsumsw2d_types[2] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_types[3] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_types[4] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_types[5] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_types[6] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_types[7] = <char>np.NPY_CDOUBLE
gufunc_lsumsw2d_types[8] = <char>np.NPY_LONG
gufunc_lsumsw2d_types[9] = <char>np.NPY_LONG
gufunc_lsumsw2d_types[10] = <char>np.NPY_CDOUBLE
gufunc_lsumsw2d_types[11] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_types[12] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_types[13] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_types[14] = <char>np.NPY_CDOUBLE
gufunc_lsumsw2d_types[15] = <char>np.NPY_CDOUBLE
gufunc_lsumsw2d_data[0] = <void*>_esum.lsumsw2d[double]
gufunc_lsumsw2d_data[1] = <void*>_esum.lsumsw2d[double_complex]
gufunc_realsumsw2d_data[0] = <void*>_esum.realsumsw2d[double]
gufunc_realsumsw2d_data[1] = <void*>_esum.realsumsw2d[double_complex]
gufunc_recsumsw2d_data[0] = <void*>_esum.recsumsw2d[double]
gufunc_recsumsw2d_data[1] = <void*>_esum.recsumsw2d[double_complex]

lsumsw2d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw2d_loops,
    gufunc_lsumsw2d_data,
    gufunc_lsumsw2d_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'lsumsw2d',  # function name
    r"""lsumsw2d(l, m, k, kpar, a, r, eta)

Fast summation of spherical functions on a 2d lattice

Computes

.. math::

    D_{lm}(k, \boldsymbol k_\parallel, \Lambda_2, \boldsymbol r)
    = \sum_{\boldsymbol R \in \Lambda_2}
    h_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    Y_{lm}(-\boldsymbol r - \boldsymbol R)
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

using the Ewald summation.

The cut between the real and reciprocal space summation is defined by `eta`. Larger
values increase the weight of the real sum. In `a` the lattice vectors are
given as rows. The lattice is in the x-y-plane.

Args:
    l (integer): Degree :math:`l \geq 0`
    m (integer): Order :math:`|m| \leq l`
    k (float or complex): Wave number
    kpar (float, (2,)-array): Tangential wave vector
    a (float, (2,2)-array): Lattice vectors
    r (float, (2,)-array): In-plane shift vector
    eta (float or complex): Separation value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(),(2),(2,2),(2),()->()',  # signature
)
realsumsw2d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw2d_loops,
    gufunc_realsumsw2d_data,
    gufunc_lsumsw2d_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'realsumsw2d',  # function name
    r"""realsumsw2d(l, m, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(2),(2,2),(2),()->()',  # signature
)
recsumsw2d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw2d_loops,
    gufunc_recsumsw2d_data,
    gufunc_lsumsw2d_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'recsumsw2d',  # function name
    r"""recsumsw2d(l, m, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(2),(2,2),(2),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_lsumsw2d_shift_loops[2]
cdef void *gufunc_lsumsw2d_shift_data[2]
cdef void *gufunc_realsumsw2d_shift_data[2]
cdef void *gufunc_recsumsw2d_shift_data[2]
cdef char gufunc_lsumsw2d_shift_types[2 * 8]

gufunc_lsumsw2d_shift_loops[0] = <np.PyUFuncGenericFunction>loop_lsumsw2d_shift_d
gufunc_lsumsw2d_shift_loops[1] = <np.PyUFuncGenericFunction>loop_lsumsw2d_shift_D
gufunc_lsumsw2d_shift_types[0] = <char>np.NPY_LONG
gufunc_lsumsw2d_shift_types[1] = <char>np.NPY_LONG
gufunc_lsumsw2d_shift_types[2] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_shift_types[3] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_shift_types[4] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_shift_types[5] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_shift_types[6] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_shift_types[7] = <char>np.NPY_CDOUBLE
gufunc_lsumsw2d_shift_types[8] = <char>np.NPY_LONG
gufunc_lsumsw2d_shift_types[9] = <char>np.NPY_LONG
gufunc_lsumsw2d_shift_types[10] = <char>np.NPY_CDOUBLE
gufunc_lsumsw2d_shift_types[11] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_shift_types[12] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_shift_types[13] = <char>np.NPY_DOUBLE
gufunc_lsumsw2d_shift_types[14] = <char>np.NPY_CDOUBLE
gufunc_lsumsw2d_shift_types[15] = <char>np.NPY_CDOUBLE
gufunc_lsumsw2d_shift_data[0] = <void*>_esum.lsumsw2d_shift[double]
gufunc_lsumsw2d_shift_data[1] = <void*>_esum.lsumsw2d_shift[double_complex]
gufunc_realsumsw2d_shift_data[0] = <void*>_esum.realsumsw2d_shift[double]
gufunc_realsumsw2d_shift_data[1] = <void*>_esum.realsumsw2d_shift[double_complex]
gufunc_recsumsw2d_shift_data[0] = <void*>_esum.recsumsw2d_shift[double]
gufunc_recsumsw2d_shift_data[1] = <void*>_esum.recsumsw2d_shift[double_complex]

lsumsw2d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw2d_shift_loops,
    gufunc_lsumsw2d_shift_data,
    gufunc_lsumsw2d_shift_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'lsumsw2d_shift',  # function name
    r"""lsumsw2d_shift(l, m, k, kpar, a, r, eta)

Fast summation of spherical functions on a 2d lattice with out of lattice shifts

Computes

.. math::

    D_{lm}(k, \boldsymbol k_\parallel, \Lambda_3, \boldsymbol r)
    = \sum_{\boldsymbol R \in \Lambda_3}
    h_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    Y_{lm}(-\boldsymbol r - \boldsymbol R)
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

using the Ewald summation.

The cut between the real and reciprocal space summation is defined by `eta`. Larger
values increase the weight of the real sum. In `a` the lattice vectors are
given as rows. The lattice is in the x-y-plane.

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
""",  # docstring
    0,  # unused
    '(),(),(),(2),(2,2),(3),()->()',  # signature
)
realsumsw2d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw2d_shift_loops,
    gufunc_realsumsw2d_shift_data,
    gufunc_lsumsw2d_shift_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'realsumsw2d_shift',  # function name
    r"""realsumsw2d_shift(l, m, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(2),(2,2),(3),()->()',  # signature
)
recsumsw2d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw2d_shift_loops,
    gufunc_recsumsw2d_shift_data,
    gufunc_lsumsw2d_shift_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'recsumsw2d_shift',  # function name
    r"""recsumsw2d_shift(l, m, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(2),(2,2),(3),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_lsumsw1d_shift_loops[2]
cdef void *gufunc_lsumsw1d_shift_data[2]
cdef void *gufunc_realsumsw1d_shift_data[2]
cdef void *gufunc_recsumsw1d_shift_data[2]
cdef char gufunc_lsumsw1d_shift_types[2 * 8]

gufunc_lsumsw1d_shift_loops[0] = <np.PyUFuncGenericFunction>loop_lsumsw1d_shift_d
gufunc_lsumsw1d_shift_loops[1] = <np.PyUFuncGenericFunction>loop_lsumsw1d_shift_D
gufunc_lsumsw1d_shift_types[0] = <char>np.NPY_LONG
gufunc_lsumsw1d_shift_types[1] = <char>np.NPY_LONG
gufunc_lsumsw1d_shift_types[2] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_shift_types[3] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_shift_types[4] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_shift_types[5] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_shift_types[6] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_shift_types[7] = <char>np.NPY_CDOUBLE
gufunc_lsumsw1d_shift_types[8] = <char>np.NPY_LONG
gufunc_lsumsw1d_shift_types[9] = <char>np.NPY_LONG
gufunc_lsumsw1d_shift_types[10] = <char>np.NPY_CDOUBLE
gufunc_lsumsw1d_shift_types[11] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_shift_types[12] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_shift_types[13] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_shift_types[14] = <char>np.NPY_CDOUBLE
gufunc_lsumsw1d_shift_types[15] = <char>np.NPY_CDOUBLE
gufunc_lsumsw1d_shift_data[0] = <void*>_esum.lsumsw1d_shift[double]
gufunc_lsumsw1d_shift_data[1] = <void*>_esum.lsumsw1d_shift[double_complex]
gufunc_realsumsw1d_shift_data[0] = <void*>_esum.realsumsw1d_shift[double]
gufunc_realsumsw1d_shift_data[1] = <void*>_esum.realsumsw1d_shift[double_complex]
gufunc_recsumsw1d_shift_data[0] = <void*>_esum.recsumsw1d_shift[double]
gufunc_recsumsw1d_shift_data[1] = <void*>_esum.recsumsw1d_shift[double_complex]

lsumsw1d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw1d_shift_loops,
    gufunc_lsumsw1d_shift_data,
    gufunc_lsumsw1d_shift_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'lsumsw1d_shift',  # function name
    r"""lsumsw1d_shift(l, m, k, kpar, a, r, eta)

Fast summation of spherical functions on a 1d lattice with out of lattice shifts

Computes

.. math::

    D_{lm}(k, \boldsymbol k_\parallel, \Lambda_1, \boldsymbol r)
    = \sum_{\boldsymbol R \in \Lambda_1}
    h_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    Y_{lm}(-\boldsymbol r - \boldsymbol R)
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

using the Ewald summation.

The cut between the real and reciprocal space summation is defined by `eta`. Larger
values increase the weight of the real sum. In `a` the lattice vectors are
given as rows. The lattice is along the z axis.

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
""",  # docstring
    0,  # unused
    '(),(),(),(),(),(3),()->()',  # signature
)
realsumsw1d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw1d_shift_loops,
    gufunc_realsumsw1d_shift_data,
    gufunc_lsumsw1d_shift_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'realsumsw1d_shift',  # function name
    r"""realsumsw1d_shift(l, m, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(),(),(3),()->()',  # signature
)
recsumsw1d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumsw1d_shift_loops,
    gufunc_recsumsw1d_shift_data,
    gufunc_lsumsw1d_shift_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'recsumsw1d_shift',  # function name
    r"""recsumsw1d_shift(l, m, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(),(),(3),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_lsumcw1d_shift_loops[2]
cdef void *gufunc_lsumcw1d_shift_data[2]
cdef void *gufunc_realsumcw1d_shift_data[2]
cdef void *gufunc_recsumcw1d_shift_data[2]
cdef char gufunc_lsumcw1d_shift_types[2 * 7]

gufunc_lsumcw1d_shift_loops[0] = <np.PyUFuncGenericFunction>loop_lsumcw1d_shift_d
gufunc_lsumcw1d_shift_loops[1] = <np.PyUFuncGenericFunction>loop_lsumcw1d_shift_D
gufunc_lsumcw1d_shift_types[0] = <char>np.NPY_LONG
gufunc_lsumcw1d_shift_types[1] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_shift_types[2] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_shift_types[3] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_shift_types[4] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_shift_types[5] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_shift_types[6] = <char>np.NPY_CDOUBLE
gufunc_lsumcw1d_shift_types[7] = <char>np.NPY_LONG
gufunc_lsumcw1d_shift_types[8] = <char>np.NPY_CDOUBLE
gufunc_lsumcw1d_shift_types[9] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_shift_types[10] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_shift_types[11] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_shift_types[12] = <char>np.NPY_CDOUBLE
gufunc_lsumcw1d_shift_types[13] = <char>np.NPY_CDOUBLE
gufunc_lsumcw1d_shift_data[0] = <void*>_esum.lsumcw1d_shift[double]
gufunc_lsumcw1d_shift_data[1] = <void*>_esum.lsumcw1d_shift[double_complex]
gufunc_realsumcw1d_shift_data[0] = <void*>_esum.realsumcw1d_shift[double]
gufunc_realsumcw1d_shift_data[1] = <void*>_esum.realsumcw1d_shift[double_complex]
gufunc_recsumcw1d_shift_data[0] = <void*>_esum.recsumcw1d_shift[double]
gufunc_recsumcw1d_shift_data[1] = <void*>_esum.recsumcw1d_shift[double_complex]

lsumcw1d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumcw1d_shift_loops,
    gufunc_lsumcw1d_shift_data,
    gufunc_lsumcw1d_shift_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'lsumcw1d_shift',  # function name
    r"""lsumcw1d_shift(l, k, kpar, a, r, eta)

Fast summation of cylindrical functions on a 1d lattice with out of lattice shifts

Computes

.. math::

    D_{l}(k, \boldsymbol k_\parallel, \boldsymbol r, \Lambda_1)
    = \sum_{\boldsymbol R \in \Lambda_1}
    H_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    \mathrm e^{\mathrm i l \varphi_{-\boldsymbol r - \boldsymbol R}}
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

using the Ewald summation.

The cut between the real and reciprocal space summation is defined by `eta`. Larger
values increase the weight of the real sum. In `a` the lattice vectors are
given as rows. The lattice is along the x axis.

Args:
    l (integer): Order
    k (float or complex): Wave number
    kpar (float): Tangential wave vector component
    a (float): Lattice pitch
    r (float, (2,)-array): Shift vector
    eta (float or complex): Separation value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(),(),(2),()->()',  # signature
)
realsumcw1d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumcw1d_shift_loops,
    gufunc_realsumcw1d_shift_data,
    gufunc_lsumcw1d_shift_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'realsumcw1d_shift',  # function name
    r"""realsumcw1d_shift(l, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(),(2),()->()',  # signature
)
recsumcw1d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_lsumcw1d_shift_loops,
    gufunc_recsumcw1d_shift_data,
    gufunc_lsumcw1d_shift_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'recsumcw1d_shift',  # function name
    r"""recsumcw1d_shift(l, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
    '(),(),(),(),(2),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_lsumcw1d_loops[2]
cdef void *gufunc_lsumcw1d_data[2]
cdef void *gufunc_realsumcw1d_data[2]
cdef void *gufunc_recsumcw1d_data[2]
cdef char gufunc_lsumcw1d_types[2 * 7]

gufunc_lsumcw1d_loops[0] = <np.PyUFuncGenericFunction>loop_lsum1d_d
gufunc_lsumcw1d_loops[1] = <np.PyUFuncGenericFunction>loop_lsum1d_D
gufunc_lsumcw1d_types[0] = <char>np.NPY_LONG
gufunc_lsumcw1d_types[1] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_types[2] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_types[3] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_types[4] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_types[5] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_types[6] = <char>np.NPY_CDOUBLE
gufunc_lsumcw1d_types[7] = <char>np.NPY_LONG
gufunc_lsumcw1d_types[8] = <char>np.NPY_CDOUBLE
gufunc_lsumcw1d_types[9] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_types[10] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_types[11] = <char>np.NPY_DOUBLE
gufunc_lsumcw1d_types[12] = <char>np.NPY_CDOUBLE
gufunc_lsumcw1d_types[13] = <char>np.NPY_CDOUBLE
gufunc_lsumcw1d_data[0] = <void*>_esum.lsumcw1d[double]
gufunc_lsumcw1d_data[1] = <void*>_esum.lsumcw1d[double_complex]
gufunc_realsumcw1d_data[0] = <void*>_esum.realsumcw1d[double]
gufunc_realsumcw1d_data[1] = <void*>_esum.realsumcw1d[double_complex]
gufunc_recsumcw1d_data[0] = <void*>_esum.recsumcw1d[double]
gufunc_recsumcw1d_data[1] = <void*>_esum.recsumcw1d[double_complex]

lsumcw1d = np.PyUFunc_FromFuncAndData(
    gufunc_lsumcw1d_loops,
    gufunc_lsumcw1d_data,
    gufunc_lsumcw1d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'lsumcw1d',  # function name
    r"""lsumcw1d(l, k, kpar, a, r, eta)

Fast summation of cylindrical functions on a 1d lattice

Computes

.. math::

    D_{l}(k, k_\parallel, r, \Lambda_1)
    = \sum_{R \in \Lambda_1}
    H_l^{(1)}(k |r + R|)
    (\mathrm{sign}(-r - R))^l
    \mathrm e^{\mathrm i k_\parallel R}

using the Ewald summation.

The cut between the real and reciprocal space summation is defined by `eta`. Larger
values increase the weight of the real sum. In `a` the lattice vectors are
given as rows. The lattice is along the x axis.

Args:
    l (integer): Order
    k (float or complex): Wave number
    kpar (float): Tangential wave vector component
    a (float): Lattice pitch
    r (float): In-line shift
    eta (float or complex): Separation value

Returns:
    complex
""",  # docstring
    0,  # unused
)
realsumcw1d = np.PyUFunc_FromFuncAndData(
    gufunc_lsumcw1d_loops,
    gufunc_realsumcw1d_data,
    gufunc_lsumcw1d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'realsumcw1d',  # function name
    r"""realsumcw1d(l, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
)
recsumcw1d = np.PyUFunc_FromFuncAndData(
    gufunc_lsumcw1d_loops,
    gufunc_recsumcw1d_data,
    gufunc_lsumcw1d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'recsumcw1d',  # function name
    r"""recsumcw1d(l, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction gufunc_lsumsw1d_loops[2]
cdef void *gufunc_lsumsw1d_data[2]
cdef void *gufunc_realsumsw1d_data[2]
cdef void *gufunc_recsumsw1d_data[2]
cdef char gufunc_lsumsw1d_types[2 * 7]

gufunc_lsumsw1d_loops[0] = <np.PyUFuncGenericFunction>loop_lsum1d_d
gufunc_lsumsw1d_loops[1] = <np.PyUFuncGenericFunction>loop_lsum1d_D
gufunc_lsumsw1d_types[0] = <char>np.NPY_LONG
gufunc_lsumsw1d_types[1] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_types[2] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_types[3] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_types[4] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_types[5] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_types[6] = <char>np.NPY_CDOUBLE
gufunc_lsumsw1d_types[7] = <char>np.NPY_LONG
gufunc_lsumsw1d_types[8] = <char>np.NPY_CDOUBLE
gufunc_lsumsw1d_types[9] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_types[10] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_types[11] = <char>np.NPY_DOUBLE
gufunc_lsumsw1d_types[12] = <char>np.NPY_CDOUBLE
gufunc_lsumsw1d_types[13] = <char>np.NPY_CDOUBLE
gufunc_lsumsw1d_data[0] = <void*>_esum.lsumsw1d[double]
gufunc_lsumsw1d_data[1] = <void*>_esum.lsumsw1d[double_complex]
gufunc_realsumsw1d_data[0] = <void*>_esum.realsumsw1d[double]
gufunc_realsumsw1d_data[1] = <void*>_esum.realsumsw1d[double_complex]
gufunc_recsumsw1d_data[0] = <void*>_esum.recsumsw1d[double]
gufunc_recsumsw1d_data[1] = <void*>_esum.recsumsw1d[double_complex]

lsumsw1d = np.PyUFunc_FromFuncAndData(
    gufunc_lsumsw1d_loops,
    gufunc_lsumsw1d_data,
    gufunc_lsumsw1d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'lsumsw1d',  # function name
    r"""lsumsw1d(l, k, kpar, a, r, eta)

Fast summation of spherical functions on a 1d lattice

Computes

.. math::

    D_{l0}(k, k_\parallel, \Lambda_1, r)
    = \sum_{R \in \Lambda_1}
    h_l^{(1)}(k |r + R|)
    Y_{l0}(-\boldsymbol{\hat z} (r + R))
    \mathrm e^{\mathrm i k_\parallel R}

using the Ewald summation.

The cut between the real and reciprocal space summation is defined by `eta`. Larger
values increase the weight of the real sum. The lattice is along the z-axis.
Therefore only :math:`m = 0` contributes.

Args:
    l (integer): Degree :math:`l \geq 0`
    k (float or complex): Wave number
    kpar (float): Tangential wave vector component
    a (float): Lattice pitch
    r (float): In-line shift
    eta (float or complex): Separation value

Returns:
    complex
""",  # docstring
    0,  # unused
)
realsumsw1d = np.PyUFunc_FromFuncAndData(
    gufunc_lsumsw1d_loops,
    gufunc_realsumsw1d_data,
    gufunc_lsumsw1d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'realsumsw1d',  # function name
    r"""realsumsw1d(l, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
)
recsumsw1d = np.PyUFunc_FromFuncAndData(
    gufunc_lsumsw1d_loops,
    gufunc_recsumsw1d_data,
    gufunc_lsumsw1d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'recsumsw1d',  # function name
    r"""recsumsw1d(l, k, kpar, a, r, eta)""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction gufunc_dsumcw2d_loops[2]
cdef void *gufunc_dsumcw2d_data[2]
cdef char gufunc_dsumcw2d_types[2 * 7]

gufunc_dsumcw2d_loops[0] = <np.PyUFuncGenericFunction>loop_dsumcw2d_d
gufunc_dsumcw2d_loops[1] = <np.PyUFuncGenericFunction>loop_dsumcw2d_D
gufunc_dsumcw2d_types[0] = <char>np.NPY_LONG
gufunc_dsumcw2d_types[1] = <char>np.NPY_DOUBLE
gufunc_dsumcw2d_types[2] = <char>np.NPY_DOUBLE
gufunc_dsumcw2d_types[3] = <char>np.NPY_DOUBLE
gufunc_dsumcw2d_types[4] = <char>np.NPY_DOUBLE
gufunc_dsumcw2d_types[5] = <char>np.NPY_LONG
gufunc_dsumcw2d_types[6] = <char>np.NPY_CDOUBLE
gufunc_dsumcw2d_types[7] = <char>np.NPY_LONG
gufunc_dsumcw2d_types[8] = <char>np.NPY_CDOUBLE
gufunc_dsumcw2d_types[9] = <char>np.NPY_DOUBLE
gufunc_dsumcw2d_types[10] = <char>np.NPY_DOUBLE
gufunc_dsumcw2d_types[11] = <char>np.NPY_DOUBLE
gufunc_dsumcw2d_types[12] = <char>np.NPY_LONG
gufunc_dsumcw2d_types[13] = <char>np.NPY_CDOUBLE
gufunc_dsumcw2d_data[0] = <void*>_dsum.dsumcw2d[double]
gufunc_dsumcw2d_data[1] = <void*>_dsum.dsumcw2d[double_complex]

dsumcw2d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_dsumcw2d_loops,
    gufunc_dsumcw2d_data,
    gufunc_dsumcw2d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'dsumcw2d',  # function name
    r"""dsumcw2d(l, k, kpar, a, r, i)

Direct summation of cylindrical functions on a 2d lattice

Computes

.. math::

    D_{l}(k, \boldsymbol k_\parallel, \boldsymbol r, \Lambda_2)
    = \sum_{\boldsymbol R \in \Lambda_2}
    H_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    \mathrm e^{\mathrm i l \varphi_{-\boldsymbol r - \boldsymbol R}}
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

directly for one expansion value `i`. Sum `i` from `0` to a large value to
obtain an approximation of the sum value. In `a` the lattice vectors are
given as rows.

Args:
    l (integer): Order
    k (float or complex): Wave number
    kpar (float, (2,)-array): Wave vector
    a (float, (2,2)-array): Lattice vectors
    r (float, (2,)-array): Shift vector
    i (integer): Expansion value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(2),(2,2),(2),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_dsumsw3d_loops[2]
cdef void *gufunc_dsumsw3d_data[2]
cdef char gufunc_dsumsw3d_types[2 * 8]

gufunc_dsumsw3d_loops[0] = <np.PyUFuncGenericFunction>loop_dsumsw3d_d
gufunc_dsumsw3d_loops[1] = <np.PyUFuncGenericFunction>loop_dsumsw3d_D
gufunc_dsumsw3d_types[0] = <char>np.NPY_LONG
gufunc_dsumsw3d_types[1] = <char>np.NPY_LONG
gufunc_dsumsw3d_types[2] = <char>np.NPY_DOUBLE
gufunc_dsumsw3d_types[3] = <char>np.NPY_DOUBLE
gufunc_dsumsw3d_types[4] = <char>np.NPY_DOUBLE
gufunc_dsumsw3d_types[5] = <char>np.NPY_DOUBLE
gufunc_dsumsw3d_types[6] = <char>np.NPY_LONG
gufunc_dsumsw3d_types[7] = <char>np.NPY_CDOUBLE
gufunc_dsumsw3d_types[8] = <char>np.NPY_LONG
gufunc_dsumsw3d_types[9] = <char>np.NPY_LONG
gufunc_dsumsw3d_types[10] = <char>np.NPY_CDOUBLE
gufunc_dsumsw3d_types[11] = <char>np.NPY_DOUBLE
gufunc_dsumsw3d_types[12] = <char>np.NPY_DOUBLE
gufunc_dsumsw3d_types[13] = <char>np.NPY_DOUBLE
gufunc_dsumsw3d_types[14] = <char>np.NPY_LONG
gufunc_dsumsw3d_types[15] = <char>np.NPY_CDOUBLE
gufunc_dsumsw3d_data[0] = <void*>_dsum.dsumsw3d[double]
gufunc_dsumsw3d_data[1] = <void*>_dsum.dsumsw3d[double_complex]

dsumsw3d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_dsumsw3d_loops,
    gufunc_dsumsw3d_data,
    gufunc_dsumsw3d_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'dsumsw3d',  # function name
    r"""dsumsw3d(l, m, k, kpar, a, r, i)

Direct summation of spherical functions on a 3d lattice

Computes

.. math::

    D_{lm}(k, \boldsymbol k_\parallel, \Lambda_3, \boldsymbol r)
    = \sum_{\boldsymbol R \in \Lambda_3}
    h_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    Y_{lm}(-\boldsymbol r - \boldsymbol R)
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

directly for one expansion value `i`. Sum `i` from `0` to a large value to
obtain an approximation of the sum value. In `a` the lattice vectors are
given as rows.

Args:
    l (integer): Degree :math:`l \geq 0`
    m (integer): Order :math:`|m| \leq l`
    k (float or complex): Wave number
    kpar (float, (3,)-array): Wave vector
    a (float, (3,3)-array): Lattice vectors
    r (float, (3,)-array): Shift vector
    i (integer): Expansion value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(),(3),(3,3),(3),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_dsumsw2d_loops[2]
cdef void *gufunc_dsumsw2d_data[2]
cdef char gufunc_dsumsw2d_types[2 * 8]

gufunc_dsumsw2d_loops[0] = <np.PyUFuncGenericFunction>loop_dsumsw2d_d
gufunc_dsumsw2d_loops[1] = <np.PyUFuncGenericFunction>loop_dsumsw2d_D
gufunc_dsumsw2d_types[0] = <char>np.NPY_LONG
gufunc_dsumsw2d_types[1] = <char>np.NPY_LONG
gufunc_dsumsw2d_types[2] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_types[3] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_types[4] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_types[5] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_types[6] = <char>np.NPY_LONG
gufunc_dsumsw2d_types[7] = <char>np.NPY_CDOUBLE
gufunc_dsumsw2d_types[8] = <char>np.NPY_LONG
gufunc_dsumsw2d_types[9] = <char>np.NPY_LONG
gufunc_dsumsw2d_types[10] = <char>np.NPY_CDOUBLE
gufunc_dsumsw2d_types[11] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_types[12] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_types[13] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_types[14] = <char>np.NPY_LONG
gufunc_dsumsw2d_types[15] = <char>np.NPY_CDOUBLE
gufunc_dsumsw2d_data[0] = <void*>_dsum.dsumsw2d[double]
gufunc_dsumsw2d_data[1] = <void*>_dsum.dsumsw2d[double_complex]

dsumsw2d = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_dsumsw2d_loops,
    gufunc_dsumsw2d_data,
    gufunc_dsumsw2d_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'dsumsw2d',  # function name
    r"""dsumsw2d(l, m, k, kpar, a, r, i)

Direct summation of spherical functions on a 2d lattice

Computes

.. math::

    D_{lm}(k, \boldsymbol k_\parallel, \Lambda_2, \boldsymbol r)
    = \sum_{\boldsymbol R \in \Lambda_2}
    h_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    Y_{lm}(-\boldsymbol r - \boldsymbol R)
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

directly for one expansion value `i`. Sum `i` from `0` to a large value to
obtain an approximation of the sum value. In `a` the lattice vectors are
given as rows. The lattice is in the x-y-plane.

Args:
    l (integer): Degree :math:`l \geq 0`
    m (integer): Order :math:`|m| \leq l`
    k (float or complex): Wave number
    kpar (float, (2,)-array): Tangential wave vector
    a (float, (2,2)-array): Lattice vectors
    r (float, (2,)-array): In-plane shift vector
    i (integer): Expansion value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(),(2),(2,2),(2),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_dsumsw2d_shift_loops[2]
cdef void *gufunc_dsumsw2d_shift_data[2]
cdef char gufunc_dsumsw2d_shift_types[2 * 8]

gufunc_dsumsw2d_shift_loops[0] = <np.PyUFuncGenericFunction>loop_dsumsw2d_shift_d
gufunc_dsumsw2d_shift_loops[1] = <np.PyUFuncGenericFunction>loop_dsumsw2d_shift_D
gufunc_dsumsw2d_shift_types[0] = <char>np.NPY_LONG
gufunc_dsumsw2d_shift_types[1] = <char>np.NPY_LONG
gufunc_dsumsw2d_shift_types[2] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_shift_types[3] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_shift_types[4] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_shift_types[5] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_shift_types[6] = <char>np.NPY_LONG
gufunc_dsumsw2d_shift_types[7] = <char>np.NPY_CDOUBLE
gufunc_dsumsw2d_shift_types[8] = <char>np.NPY_LONG
gufunc_dsumsw2d_shift_types[9] = <char>np.NPY_LONG
gufunc_dsumsw2d_shift_types[10] = <char>np.NPY_CDOUBLE
gufunc_dsumsw2d_shift_types[11] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_shift_types[12] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_shift_types[13] = <char>np.NPY_DOUBLE
gufunc_dsumsw2d_shift_types[14] = <char>np.NPY_LONG
gufunc_dsumsw2d_shift_types[15] = <char>np.NPY_CDOUBLE
gufunc_dsumsw2d_shift_data[0] = <void*>_dsum.dsumsw2d_shift[double]
gufunc_dsumsw2d_shift_data[1] = <void*>_dsum.dsumsw2d_shift[double_complex]

dsumsw2d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_dsumsw2d_shift_loops,
    gufunc_dsumsw2d_shift_data,
    gufunc_dsumsw2d_shift_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'dsumsw2d_shift',  # function name
    r"""dsumsw2d_shift(l, m, k, kpar, a, r, i)

Direct summation of spherical functions on a 2d lattice with out of lattice shifts

Computes

.. math::

    D_{lm}(k, \boldsymbol k_\parallel, \Lambda_3, \boldsymbol r)
    = \sum_{\boldsymbol R \in \Lambda_3}
    h_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    Y_{lm}(-\boldsymbol r - \boldsymbol R)
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

directly for one expansion value `i`. Sum `i` from `0` to a large value to
obtain an approximation of the sum value. In `a` the lattice vectors are
given as rows. The lattice is in the x-y-plane.

Args:
    l (integer): Degree :math:`l \geq 0`
    m (integer): Order :math:`|m| \leq l`
    k (float or complex): Wave number
    kpar (float, (2,)-array): Tangential wave vector
    a (float, (2,2)-array): Lattice vectors
    r (float, (3,)-array): Shift vector
    i (integer): Expansion value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(),(2),(2,2),(3),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_dsumsw1d_shift_loops[2]
cdef void *gufunc_dsumsw1d_shift_data[2]
cdef char gufunc_dsumsw1d_shift_types[2 * 8]

gufunc_dsumsw1d_shift_loops[0] = <np.PyUFuncGenericFunction>loop_dsumsw1d_shift_d
gufunc_dsumsw1d_shift_loops[1] = <np.PyUFuncGenericFunction>loop_dsumsw1d_shift_D
gufunc_dsumsw1d_shift_types[0] = <char>np.NPY_LONG
gufunc_dsumsw1d_shift_types[1] = <char>np.NPY_LONG
gufunc_dsumsw1d_shift_types[2] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_shift_types[3] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_shift_types[4] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_shift_types[5] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_shift_types[6] = <char>np.NPY_LONG
gufunc_dsumsw1d_shift_types[7] = <char>np.NPY_CDOUBLE
gufunc_dsumsw1d_shift_types[8] = <char>np.NPY_LONG
gufunc_dsumsw1d_shift_types[9] = <char>np.NPY_LONG
gufunc_dsumsw1d_shift_types[10] = <char>np.NPY_CDOUBLE
gufunc_dsumsw1d_shift_types[11] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_shift_types[12] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_shift_types[13] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_shift_types[14] = <char>np.NPY_LONG
gufunc_dsumsw1d_shift_types[15] = <char>np.NPY_CDOUBLE
gufunc_dsumsw1d_shift_data[0] = <void*>_dsum.dsumsw1d_shift[double]
gufunc_dsumsw1d_shift_data[1] = <void*>_dsum.dsumsw1d_shift[double_complex]

dsumsw1d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_dsumsw1d_shift_loops,
    gufunc_dsumsw1d_shift_data,
    gufunc_dsumsw1d_shift_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'dsumsw1d_shift',  # function name
    r"""dsumsw1d_shift(l, m, k, kpar, a, r, i)

Fast summation of spherical functions on a 1d lattice with out of lattice shifts

Computes

.. math::

    D_{lm}(k, \boldsymbol k_\parallel, \Lambda_1, \boldsymbol r)
    = \sum_{\boldsymbol R \in \Lambda_1}
    h_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    Y_{lm}(-\boldsymbol r - \boldsymbol R)
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

directly for one expansion value `i`. Sum `i` from `0` to a large value to
obtain an approximation of the sum value. In `a` the lattice vectors are
given as rows. The lattice is along the z axis.

Args:
    l (integer): Degree :math:`l \geq 0`
    m (integer): Order :math:`|m| \leq l`
    k (float or complex): Wave number
    kpar (float): Tangential wave vector component
    a (float): Lattice pitch
    r (float, (3,)-array): Shift vector
    i (integer): Expansion value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(),(),(),(3),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_dsumcw1d_shift_loops[2]
cdef void *gufunc_dsumcw1d_shift_data[2]
cdef char gufunc_dsumcw1d_shift_types[2 * 7]

gufunc_dsumcw1d_shift_loops[0] = <np.PyUFuncGenericFunction>loop_dsumcw1d_shift_d
gufunc_dsumcw1d_shift_loops[1] = <np.PyUFuncGenericFunction>loop_dsumcw1d_shift_D
gufunc_dsumcw1d_shift_types[0] = <char>np.NPY_LONG
gufunc_dsumcw1d_shift_types[1] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_shift_types[2] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_shift_types[3] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_shift_types[4] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_shift_types[5] = <char>np.NPY_LONG
gufunc_dsumcw1d_shift_types[6] = <char>np.NPY_CDOUBLE
gufunc_dsumcw1d_shift_types[7] = <char>np.NPY_LONG
gufunc_dsumcw1d_shift_types[8] = <char>np.NPY_CDOUBLE
gufunc_dsumcw1d_shift_types[9] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_shift_types[10] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_shift_types[11] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_shift_types[12] = <char>np.NPY_LONG
gufunc_dsumcw1d_shift_types[13] = <char>np.NPY_CDOUBLE
gufunc_dsumcw1d_shift_data[0] = <void*>_dsum.dsumcw1d_shift[double]
gufunc_dsumcw1d_shift_data[1] = <void*>_dsum.dsumcw1d_shift[double_complex]

dsumcw1d_shift = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_dsumcw1d_shift_loops,
    gufunc_dsumcw1d_shift_data,
    gufunc_dsumcw1d_shift_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'dsumcw1d_shift',  # function name
    r"""dsumcw1d_shift(l, k, kpar, a, r, i)

Direct summation of cylindrical functions on a 1d lattice with out of lattice shifts

Computes

.. math::

    D_{l}(k, \boldsymbol k_\parallel, \boldsymbol r, \Lambda_1)
    = \sum_{\boldsymbol R \in \Lambda_1}
    H_l^{(1)}(k |\boldsymbol r + \boldsymbol R|)
    \mathrm e^{\mathrm i l \varphi_{-\boldsymbol r - \boldsymbol R}}
    \mathrm e^{\mathrm i \boldsymbol k_\parallel \boldsymbol R}

directly for one expansion value `i`. Sum `i` from `0` to a large value to
obtain an approximation of the sum value. In `a` the lattice vectors are
given as rows. The lattice is along the x axis.

Args:
    l (integer): Order
    k (float or complex): Wave number
    kpar (float): Tangential wave vector component
    a (float): Lattice pitch
    r (float, (2,)-array): Shift vector
    i (integer): Expansion value

Returns:
    complex
""",  # docstring
    0,  # unused
    '(),(),(),(),(2),()->()',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_dsumcw1d_loops[2]
cdef void *gufunc_dsumcw1d_data[2]
cdef char gufunc_dsumcw1d_types[2 * 7]

gufunc_dsumcw1d_loops[0] = <np.PyUFuncGenericFunction>loop_dsum1d_d
gufunc_dsumcw1d_loops[1] = <np.PyUFuncGenericFunction>loop_dsum1d_D
gufunc_dsumcw1d_types[0] = <char>np.NPY_LONG
gufunc_dsumcw1d_types[1] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_types[2] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_types[3] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_types[4] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_types[5] = <char>np.NPY_LONG
gufunc_dsumcw1d_types[6] = <char>np.NPY_CDOUBLE
gufunc_dsumcw1d_types[7] = <char>np.NPY_LONG
gufunc_dsumcw1d_types[8] = <char>np.NPY_CDOUBLE
gufunc_dsumcw1d_types[9] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_types[10] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_types[11] = <char>np.NPY_DOUBLE
gufunc_dsumcw1d_types[12] = <char>np.NPY_LONG
gufunc_dsumcw1d_types[13] = <char>np.NPY_CDOUBLE
gufunc_dsumcw1d_data[0] = <void*>_dsum.dsumcw1d[double]
gufunc_dsumcw1d_data[1] = <void*>_dsum.dsumcw1d[double_complex]

dsumcw1d = np.PyUFunc_FromFuncAndData(
    gufunc_dsumcw1d_loops,
    gufunc_dsumcw1d_data,
    gufunc_dsumcw1d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'dsumcw1d',  # function name
    r"""dsumcw1d(l, k, kpar, a, r, i)

Direct summation of cylindrical functions on a 1d lattice

Computes

.. math::

    D_{l}(k, k_\parallel, r, \Lambda_1)
    = \sum_{R \in \Lambda_1}
    H_l^{(1)}(k |r + R|)
    (\mathrm{sign}(-r - R))^l
    \mathrm e^{\mathrm i k_\parallel R}

directly for one expansion value `i`. Sum `i` from `0` to a large value to
obtain an approximation of the sum value. In `a` the lattice vectors are
given as rows. The lattice is along the x axis.

Args:
    l (integer): Order
    k (float or complex): Wave number
    kpar (float): Tangential wave vector component
    a (float): Lattice pitch
    r (float): In-line shift
    i (integer): Expansion value

Returns:
    complex
""",  # docstring
    0,  # unused
)


cdef np.PyUFuncGenericFunction gufunc_dsumsw1d_loops[2]
cdef void *gufunc_dsumsw1d_data[2]
cdef char gufunc_dsumsw1d_types[2 * 7]

gufunc_dsumsw1d_loops[0] = <np.PyUFuncGenericFunction>loop_dsum1d_d
gufunc_dsumsw1d_loops[1] = <np.PyUFuncGenericFunction>loop_dsum1d_D
gufunc_dsumsw1d_types[0] = <char>np.NPY_LONG
gufunc_dsumsw1d_types[1] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_types[2] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_types[3] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_types[4] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_types[5] = <char>np.NPY_LONG
gufunc_dsumsw1d_types[6] = <char>np.NPY_CDOUBLE
gufunc_dsumsw1d_types[7] = <char>np.NPY_LONG
gufunc_dsumsw1d_types[8] = <char>np.NPY_CDOUBLE
gufunc_dsumsw1d_types[9] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_types[10] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_types[11] = <char>np.NPY_DOUBLE
gufunc_dsumsw1d_types[12] = <char>np.NPY_LONG
gufunc_dsumsw1d_types[13] = <char>np.NPY_CDOUBLE
gufunc_dsumsw1d_data[0] = <void*>_dsum.dsumsw1d[double]
gufunc_dsumsw1d_data[1] = <void*>_dsum.dsumsw1d[double_complex]

dsumsw1d = np.PyUFunc_FromFuncAndData(
    gufunc_dsumsw1d_loops,
    gufunc_dsumsw1d_data,
    gufunc_dsumsw1d_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'dsumsw1d',  # function name
    r"""dsumsw1d(l, k, kpar, a, r, i)

Direct summation of spherical functions on a 1d lattice

Computes

.. math::

    D_{l0}(k, k_\parallel, \Lambda_1, r)
    = \sum_{R \in \Lambda_1}
    h_l^{(1)}(k |r + R|)
    Y_{l0}(-\boldsymbol{\hat z} (r + R))
    \mathrm e^{\mathrm i k_\parallel R}

using the Ewald summation.

directly for one expansion value `i`. Sum `i` from `0` to a large value to
obtain an approximation of the sum value. The lattice is along the z-axis.
Therefore only :math:`m = 0` contributes.

Args:
    l (integer): Degree :math:`l \geq 0`
    k (float or complex): Wave number
    kpar (float): Tangential wave vector component
    a (float): Lattice pitch
    r (float): In-line shift
    i (integer): Expansion value

Returns:
    complex
""",  # docstring
    0,  # unused
)


cdef void loop_D_D(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *op0 = args[1]
    cdef double complex ov0
    for i in range(n):
        ov0 = (<double complex(*)(double complex) nogil>func)(<double complex>(<double complex*>ip0)[0])
        (<double complex*>op0)[0] = <double complex>ov0
        ip0 += steps[0]
        op0 += steps[1]


cdef np.PyUFuncGenericFunction ufunc_zero_loops[1]
cdef void *ufunc_zero3d_data[1]
cdef void *ufunc_zero2d_data[1]
cdef char ufunc_zero_types[2]

ufunc_zero_loops[0] = <np.PyUFuncGenericFunction>loop_D_D
ufunc_zero_types[0] = <char>np.NPY_CDOUBLE
ufunc_zero_types[1] = <char>np.NPY_CDOUBLE
ufunc_zero3d_data[0] = <void*>_esum.zero3d
ufunc_zero2d_data[0] = <void*>_esum.zero2d

zero3d = np.PyUFunc_FromFuncAndData(
    ufunc_zero_loops,
    ufunc_zero3d_data,
    ufunc_zero_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'zero3d',  # function name
    r"""zero3d(eta)""",  # docstring
    0,  # unused
)
zero2d = np.PyUFunc_FromFuncAndData(
    ufunc_zero_loops,
    ufunc_zero2d_data,
    ufunc_zero_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # `identity` element, never mind this
    'zero2d',  # function name
    r"""zero2d(eta)""",  # docstring
    0,  # unused
)

