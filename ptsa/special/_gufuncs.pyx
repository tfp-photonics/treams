"""Generalized universal functions for the special subpackage"""

cimport numpy as np

from ptsa.special cimport _coord, _waves
from ptsa.special._misc cimport double_complex

__all__ = [
    'car2cyl',
    'car2sph',
    'cyl2car',
    'cyl2sph',
    'sph2car',
    'sph2cyl',
    'vcar2cyl',
    'vcar2sph',
    'vcyl2car',
    'vcyl2sph',
    'vsph2car',
    'vsph2cyl',
    'car2pol',
    'pol2car',
    'vcar2pol',
    'vpol2car',
    'vsh_X',
    'vsh_Y',
    'vsh_Z',
    'vsw_M',
    'vsw_N',
    'vsw_A',
    'vsw_rM',
    'vsw_rN',
    'vsw_rA',
    'vcw_M',
    'vcw_N',
    'vcw_A',
    'vcw_rM',
    'vcw_rN',
    'vcw_rA',
    'vpw_M',
    'vpw_N',
    'vpw_A',
]


cdef void loop_d_d(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long istep = steps[2] // sizeof(double), ostep = steps[3] // sizeof(double)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *op0 = args[1]
    for i in range(n):
        (<void(*)(double*, double*, long, long) nogil>func)(<double*>ip0, <double*>op0, istep, ostep)
        ip0 += steps[0]
        op0 += steps[1]


cdef void loop_d_dd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long istep0 = steps[3] // sizeof(double), istep1 = steps[4] // sizeof(double), ostep = steps[5] // sizeof(double)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *op0 = args[2]
    for i in range(n):
        (<void(*)(double*, double*, double*, long, long, long) nogil>func)(<double*>ip0, <double*>ip1, <double*>op0, istep0, istep1, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        op0 += steps[2]


cdef void loop_D_Dd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long istep0 = steps[3] // sizeof(double complex), istep1 = steps[4] // sizeof(double), ostep = steps[5] // sizeof(double complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *op0 = args[2]
    for i in range(n):
        (<void(*)(double complex*, double*, double complex*, long, long, long) nogil>func)(<double complex*>ip0, <double*>ip1, <double complex*>op0, istep0, istep1, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        op0 += steps[2]


cdef void loop_D_lldd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[5] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *op0 = args[4]
    for i in range(n):
        (<void(*)(long, long, double, double, double complex*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        op0 += steps[4]


cdef void loop_D_llDd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[5] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *op0 = args[4]
    for i in range(n):
        (<void(*)(long, long, double complex, double, double complex*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        op0 += steps[4]


cdef void loop_D_llddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[6] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *op0 = args[5]
    for i in range(n):
        (<void(*)(long, long, double, double, double, double complex*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        op0 += steps[5]


cdef void loop_D_llDdd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[6] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *op0 = args[5]
    for i in range(n):
        (<void(*)(long, long, double complex, double, double, double complex*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        op0 += steps[5]


cdef void loop_D_llDDd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[6] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *op0 = args[5]
    for i in range(n):
        (<void(*)(long, long, double complex, double complex, double, double complex*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double complex>(<double complex*>ip3)[0], <double>(<double*>ip4)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        op0 += steps[5]


cdef void loop_D_lldddl(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[7] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    for i in range(n):
        (<void(*)(long, long, double, double, double, long, double complex*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <long>(<long*>ip5)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_llDddl(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[7] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    for i in range(n):
        (<void(*)(long, long, double complex, double, double, long, double complex*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <long>(<long*>ip5)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_llDDdl(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[7] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    for i in range(n):
        (<void(*)(long, long, double complex, double complex, double, long, double complex*, long) nogil>func)(<long>(<long*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double complex>(<double complex*>ip3)[0], <double>(<double*>ip4)[0], <long>(<long*>ip5)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_dlddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[6] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *op0 = args[5]
    for i in range(n):
        (<void(*)(double, long, double, double, double, double complex*, long) nogil>func)(<double>(<double*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        op0 += steps[5]


cdef void loop_D_dlDdd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[6] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *op0 = args[5]
    for i in range(n):
        (<void(*)(double, long, double complex, double, double, double complex*, long) nogil>func)(<double>(<double*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        op0 += steps[5]


cdef void loop_D_dldddD(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[7] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    for i in range(n):
        (<void(*)(double, long, double, double, double, double complex, double complex*, long) nogil>func)(<double>(<double*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double complex>(<double complex*>ip5)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_dlDddD(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[7] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    for i in range(n):
        (<void(*)(double, long, double complex, double, double, double complex, double complex*, long) nogil>func)(<double>(<double*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double complex>(<double complex*>ip5)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_dldddDl(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[8] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    for i in range(n):
        (<void(*)(double, long, double, double, double, double complex, long, double complex*, long) nogil>func)(<double>(<double*>ip0)[0], <long>(<long*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double complex>(<double complex*>ip5)[0], <long>(<long*>ip6)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_D_dlDddDl(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[8] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    for i in range(n):
        (<void(*)(double, long, double complex, double, double, double complex, long, double complex*, long) nogil>func)(<double>(<double*>ip0)[0], <long>(<long*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double complex>(<double complex*>ip5)[0], <long>(<long*>ip6)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_D_dddddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[7] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    for i in range(n):
        (<void(*)(double, double, double, double, double, double, double complex*, long) nogil>func)(<double>(<double*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double>(<double*>ip5)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_DDDddd(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[7] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *op0 = args[6]
    for i in range(n):
        (<void(*)(double complex, double complex, double complex, double, double, double, double complex*, long) nogil>func)(<double complex>(<double complex*>ip0)[0], <double complex>(<double complex*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double>(<double*>ip5)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        op0 += steps[6]


cdef void loop_D_ddddddl(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[8] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    for i in range(n):
        (<void(*)(double, double, double, double, double, double, long, double complex*, long) nogil>func)(<double>(<double*>ip0)[0], <double>(<double*>ip1)[0], <double>(<double*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double>(<double*>ip5)[0], <long>(<long*>ip6)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


cdef void loop_D_DDDdddl(char **args, np.npy_intp *dims, np.npy_intp *steps, void *data) nogil:
    cdef np.npy_intp i, n = dims[0]
    cdef long ostep = steps[8] // sizeof(double_complex)
    cdef void *func = <void*>data
    cdef char *ip0 = args[0]
    cdef char *ip1 = args[1]
    cdef char *ip2 = args[2]
    cdef char *ip3 = args[3]
    cdef char *ip4 = args[4]
    cdef char *ip5 = args[5]
    cdef char *ip6 = args[6]
    cdef char *op0 = args[7]
    for i in range(n):
        (<void(*)(double complex, double complex, double complex, double, double, double, long, double complex*, long) nogil>func)(<double complex>(<double complex*>ip0)[0], <double complex>(<double complex*>ip1)[0], <double complex>(<double complex*>ip2)[0], <double>(<double*>ip3)[0], <double>(<double*>ip4)[0], <double>(<double*>ip5)[0], <long>(<long*>ip6)[0], <double complex*>op0, ostep)
        ip0 += steps[0]
        ip1 += steps[1]
        ip2 += steps[2]
        ip3 += steps[3]
        ip4 += steps[4]
        ip5 += steps[5]
        ip6 += steps[6]
        op0 += steps[7]


np.import_array()
np.import_ufunc()


cdef np.PyUFuncGenericFunction gufunc_car2cyl_loops[1]
cdef void *gufunc_car2cyl_data[1]
cdef char gufunc_car2cyl_types[2]

gufunc_car2cyl_loops[0] = <np.PyUFuncGenericFunction>loop_d_d
gufunc_car2cyl_types[0] = <char>np.NPY_DOUBLE
gufunc_car2cyl_types[1] = <char>np.NPY_DOUBLE
gufunc_car2cyl_data[0] = <void*>_coord.car2cyl

car2cyl = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_car2cyl_loops,
    gufunc_car2cyl_data,
    gufunc_car2cyl_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # identity element
    'car2cyl',  # function name
    r"""car2cyl(r)

Convert cartesian to cylindrical coordinates

The cartesian coordinates :math:`(x, y, z)` are converted to
:math:`(\rho, \varphi, z)` with :math:`x = \rho \cos \varphi` and
:math:`y = \rho \sin \varphi`.

Args:
    r (float, 3-array): Cartesian coordinates in 3-dimensional space

Returns:
    float, 3-array
""",  # docstring
    0,  # unused
    '(3)->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_car2sph_loops[1]
cdef void *gufunc_car2sph_data[1]
cdef char gufunc_car2sph_types[2]

gufunc_car2sph_loops[0] = <np.PyUFuncGenericFunction>loop_d_d
gufunc_car2sph_types[0] = <char>np.NPY_DOUBLE
gufunc_car2sph_types[1] = <char>np.NPY_DOUBLE
gufunc_car2sph_data[0] = <void*>_coord.car2sph

car2sph = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_car2sph_loops,
    gufunc_car2sph_data,
    gufunc_car2sph_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # identity element
    'car2sph',  # function name
    r"""car2sph(r)

Convert cartesian to spherical coordinates

The cartesian coordinates :math:`(x, y, z)` are converted to
:math:`(r, \theta, \varphi)` with :math:`x = r \sin \theta \cos \varphi`,
:math:`y = r \sin \theta \sin \varphi`, and :math:`z = r \cos \theta`.

Args:
    r (float, 3-array): Cartesian coordinates in 3-dimensional space

Returns:
    float, 3-array
""",  # docstring
    0,  # unused
    '(3)->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_cyl2car_loops[1]
cdef void *gufunc_cyl2car_data[1]
cdef char gufunc_cyl2car_types[2]

gufunc_cyl2car_loops[0] = <np.PyUFuncGenericFunction>loop_d_d
gufunc_cyl2car_types[0] = <char>np.NPY_DOUBLE
gufunc_cyl2car_types[1] = <char>np.NPY_DOUBLE
gufunc_cyl2car_data[0] = <void*>_coord.cyl2car

cyl2car = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_cyl2car_loops,
    gufunc_cyl2car_data,
    gufunc_cyl2car_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # identity element
    'cyl2car',  # function name
    r"""cyl2car(r)

Convert cylindrical to cartesian coordinates

The cylindrical coordinates :math:`(\rho, \varphi, z)` are converted to
:math:`(x, y, z)` with :math:`x = \rho \cos \varphi` and
:math:`y = \rho \sin \varphi`.

Args:
    r (float, 3-array): Cylindrical coordinates in 3-dimensional space

Returns:
    float, 3-array
""",  # docstring
    0,  # unused
    '(3)->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_cyl2sph_loops[1]
cdef void *gufunc_cyl2sph_data[1]
cdef char gufunc_cyl2sph_types[2]

gufunc_cyl2sph_loops[0] = <np.PyUFuncGenericFunction>loop_d_d
gufunc_cyl2sph_types[0] = <char>np.NPY_DOUBLE
gufunc_cyl2sph_types[1] = <char>np.NPY_DOUBLE
gufunc_cyl2sph_data[0] = <void*>_coord.cyl2sph

cyl2sph = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_cyl2sph_loops,
    gufunc_cyl2sph_data,
    gufunc_cyl2sph_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # identity element
    'cyl2sph',  # function name
    r"""cyl2sph(r)

Convert cylindrical to spherical coordinates

The cylindrical coordinates :math:`(\rho, \varphi, z)` are converted to
:math:`(r, \theta, \varphi)` with :math:`\rho = r \sin \theta` and
:math:`z = \rho \cos \theta`.

Args:
    r (float, 3-array): Cylindrical coordinates in 3-dimensional space

Returns:
    float, 3-array
""",  # docstring
    0,  # unused
    '(3)->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_sph2car_loops[1]
cdef void *gufunc_sph2car_data[1]
cdef char gufunc_sph2car_types[2]

gufunc_sph2car_loops[0] = <np.PyUFuncGenericFunction>loop_d_d
gufunc_sph2car_types[0] = <char>np.NPY_DOUBLE
gufunc_sph2car_types[1] = <char>np.NPY_DOUBLE
gufunc_sph2car_data[0] = <void*>_coord.sph2car

sph2car = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_sph2car_loops,
    gufunc_sph2car_data,
    gufunc_sph2car_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # identity element
    'sph2car',  # function name
    r"""sph2car(r)

Convert spherical to cartesian coordinates

The spherical coordinates :math:`(r, \theta, \varphi)` are converted to
:math:`(x, y, z)` with :math:`x = r \sin \theta \cos \varphi`,
:math:`y = r \sin \theta \sin \varphi`, and :math:`z = r \cos \theta`.

Args:
    r (float, 3-array): Spherical coordinates in 3-dimensional space

Returns:
    float, 3-array
""",  # docstring
    0,  # unused
    '(3)->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_sph2cyl_loops[1]
cdef void *gufunc_sph2cyl_data[1]
cdef char gufunc_sph2cyl_types[2]

gufunc_sph2cyl_loops[0] = <np.PyUFuncGenericFunction>loop_d_d
gufunc_sph2cyl_types[0] = <char>np.NPY_DOUBLE
gufunc_sph2cyl_types[1] = <char>np.NPY_DOUBLE
gufunc_sph2cyl_data[0] = <void*>_coord.sph2cyl

sph2cyl = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_sph2cyl_loops,
    gufunc_sph2cyl_data,
    gufunc_sph2cyl_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # identity element
    'sph2cyl',  # function name
    r"""sph2cylr(r)

Convert spherical to cylindrical coordinates

The spherical coordinates :math:`(r, \theta, \varphi)` are converted to
:math:`(\rho, \varphi, z)` with :math:`\rho = r \sin \theta` and
:math:`z = \rho \cos \theta`.

Args:
    r (float, 3-array): Spherical coordinates in 3-dimensional space

Returns:
    float, 3-array
""",  # docstring
    0,  # unused
    '(3)->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_car2pol_loops[1]
cdef void *gufunc_car2pol_data[1]
cdef char gufunc_car2pol_types[2]

gufunc_car2pol_loops[0] = <np.PyUFuncGenericFunction>loop_d_d
gufunc_car2pol_types[0] = <char>np.NPY_DOUBLE
gufunc_car2pol_types[1] = <char>np.NPY_DOUBLE
gufunc_car2pol_data[0] = <void*>_coord.car2pol

car2pol = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_car2pol_loops,
    gufunc_car2pol_data,
    gufunc_car2pol_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # identity element
    'car2pol',  # function name
    r"""car2pol(r)

Convert cartesian to polar coordinates

The cartesian coordinates :math:`(x, y)` are converted to
:math:`(\rho, \varphi)` with :math:`x = \rho \cos \varphi` and
:math:`y = \rho \sin \varphi`.

Args:
    r (float, 2-array): Cartesian coordinates in 2-dimensional space

Returns:
    float, 2-array
""",  # docstring
    0,  # unused
    '(2)->(2)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_pol2car_loops[1]
cdef void *gufunc_pol2car_data[1]
cdef char gufunc_pol2car_types[2]

gufunc_pol2car_loops[0] = <np.PyUFuncGenericFunction>loop_d_d
gufunc_pol2car_types[0] = <char>np.NPY_DOUBLE
gufunc_pol2car_types[1] = <char>np.NPY_DOUBLE
gufunc_pol2car_data[0] = <void*>_coord.pol2car

pol2car = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_pol2car_loops,
    gufunc_pol2car_data,
    gufunc_pol2car_types,
    1,  # number of supported input types
    1,  # number of input args
    1,  # number of output args
    0,  # identity element
    'pol2car',  # function name
    r"""pol2car(r)

Convert polar to cartesian coordinates

The polar coordinates :math:`(x, y)` are converted to
:math:`(\rho, \varphi)` with :math:`x = \rho \cos \varphi` and
:math:`y = \rho \sin \varphi`.

Args:
    r (float, 2-array): Polar coordinates in 2-dimensional space

Returns:
    float, 2-array
""",  # docstring
    0,  # unused
    '(2)->(2)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vcoord_loops[2]
cdef void *gufunc_vcar2cyl_data[2]
cdef char gufunc_vcoord_types[2 * 3]

gufunc_vcoord_loops[0] = <np.PyUFuncGenericFunction>loop_d_dd
gufunc_vcoord_loops[1] = <np.PyUFuncGenericFunction>loop_D_Dd
gufunc_vcoord_types[0] = <char>np.NPY_DOUBLE
gufunc_vcoord_types[1] = <char>np.NPY_DOUBLE
gufunc_vcoord_types[2] = <char>np.NPY_DOUBLE
gufunc_vcoord_types[3] = <char>np.NPY_CDOUBLE
gufunc_vcoord_types[4] = <char>np.NPY_DOUBLE
gufunc_vcoord_types[5] = <char>np.NPY_CDOUBLE
gufunc_vcar2cyl_data[0] = <void*>_coord.vcar2cyl[double]
gufunc_vcar2cyl_data[1] = <void*>_coord.vcar2cyl[double_complex]

vcar2cyl = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcoord_loops,
    gufunc_vcar2cyl_data,
    gufunc_vcoord_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcar2cyl',  # function name
    r"""vcar2cyl(v, r)

Convert vector in cartesian coordinates to vector in cylindrical coordinates

The cartesian vector :math:`(v_x, v_y, v_z)`  at coordinates :math:`(x, y, z)` is
converted to :math:`(v_\rho, v_\varphi, v_z)` with

.. math::

    \begin{pmatrix} v_\rho \\ v_\varphi \\ v_z \end{pmatrix}
    =
    \begin{pmatrix}
        \cos \varphi & \sin \varphi & 0 \\
        -\sin \varphi & \cos \varphi & 0 \\
        0 & 0 & 1
    \end{pmatrix}
    \begin{pmatrix} v_x \\ v_y \\ v_z \end{pmatrix}

Args:
    v (float or complex, 3-array): Vector in 3-dimensional cartesian space
    r (float, 3-array): Cartesian coordinates in 3-dimensional space

Returns:
    float or complex, 3-array
""",  # docstring
    0,  # unused
    '(3),(3)->(3)',  # signature
)


cdef void *gufunc_vcar2sph_data[2]

gufunc_vcar2sph_data[0] = <void*>_coord.vcar2sph[double]
gufunc_vcar2sph_data[1] = <void*>_coord.vcar2sph[double_complex]

vcar2sph = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcoord_loops,
    gufunc_vcar2sph_data,
    gufunc_vcoord_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcar2sph',  # function name
    r"""vcar2cyl(v, r)

Convert vector in cartesian coordinates to vector in spherical coordinates

The cartesian vector :math:`(v_x, v_y, v_z)`  at coordinates :math:`(x, y, z)` is
converted to :math:`(v_r, v_\theta, v_\varphi)` with

.. math::

    \begin{pmatrix} v_r \\ v_\theta \\ v_\varphi \end{pmatrix}
    =
    \begin{pmatrix}
        \sin \theta \cos \varphi & \sin \theta \sin \varphi & \cos \theta \\
        \cos \theta \cos \varphi & \cos \theta \sin \varphi & \cos \theta \\
        -\sin \varphi & \cos \varphi & 0
    \end{pmatrix}
    \begin{pmatrix} v_x \\ v_y \\ v_z \end{pmatrix}

Args:
    v (float or complex, 3-array): Vector in 3-dimensional cartesian space
    r (float, 3-array): Cartesian coordinates in 3-dimensional space

Returns:
    float or complex, 3-array
""",  # docstring
    0,  # unused
    '(3),(3)->(3)',  # signature
)


cdef void *gufunc_vcyl2car_data[2]
gufunc_vcyl2car_data[0] = <void*>_coord.vcyl2car[double]
gufunc_vcyl2car_data[1] = <void*>_coord.vcyl2car[double_complex]

vcyl2car = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcoord_loops,
    gufunc_vcyl2car_data,
    gufunc_vcoord_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcyl2car',  # function name
    r"""vcyl2car(v, r)

Convert vector in cylindrical coordinates to vector in cartesian coordinates

The cylindrical vector :math:`(v_\rho, v_\varphi, v_z)`  at coordinates
:math:`(\rho, \varphi, z)` is converted to :math:`(v_x, v_y, v_z)` with

.. math::

    \begin{pmatrix} v_x \\ v_y \\ v_z \end{pmatrix}
    =
    \begin{pmatrix}
        \cos \varphi & -\sin \varphi & 0 \\
        \sin \varphi & \cos \varphi & 0 \\
        0 & 0 & 1
    \end{pmatrix}
    \begin{pmatrix} v_\rho \\ v_\varphi \\ v_z \end{pmatrix}

Args:
    v (float or complex, 3-array): Vector in 3-dimensional cylindrical space
    r (float, 3-array): Cylindrical coordinates in 3-dimensional space

Returns:
    float or complex, 3-array
""",  # docstring
    0,  # unused
    '(3),(3)->(3)',  # signature
)


cdef void *gufunc_vcyl2sph_data[2]
gufunc_vcyl2sph_data[0] = <void*>_coord.vcyl2sph[double]
gufunc_vcyl2sph_data[1] = <void*>_coord.vcyl2sph[double_complex]

vcyl2sph = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcoord_loops,
    gufunc_vcyl2sph_data,
    gufunc_vcoord_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcyl2sph',  # function name
    r"""vcyl2sph(v, r)

Convert vector in cylindrical coordinates to vector in spherical coordinates

The cylindrical vector :math:`(v_\rho, v_\varphi, v_z)`  at coordinates
:math:`(\rho, \varphi, z)` is converted to :math:`(v_r, v_\theta, v_\varphi)` with

.. math::

    \begin{pmatrix} v_r \\ v_\theta \\ v_\varphi \end{pmatrix}
    =
    \begin{pmatrix}
        \sin \theta & 0 & \cos \theta \\
        \cos \theta & 0 & -\sin \theta \\
        0 & 1 & 0
    \end{pmatrix}
    \begin{pmatrix} v_\rho \\ v_\varphi \\ v_z \end{pmatrix}

Args:
    v (float or complex, 3-array): Vector in 3-dimensional cylindrical space
    r (float, 3-array): Cylindrical coordinates in 3-dimensional space

Returns:
    float or complex, 3-array
""",  # docstring
    0,  # unused
    '(3),(3)->(3)',  # signature
)


cdef void *gufunc_vsph2car_data[2]
gufunc_vsph2car_data[0] = <void*>_coord.vsph2car[double]
gufunc_vsph2car_data[1] = <void*>_coord.vsph2car[double_complex]

vsph2car = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcoord_loops,
    gufunc_vsph2car_data,
    gufunc_vcoord_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsph2car',  # function name
    r"""vsph2car(v, r)

Convert vector in spherical coordinates to vector in cartesian coordinates

The spherical vector :math:`(v_r, v_\theta, v_\varphi)`  at coordinates
:math:`(r, \theta,  \varphi)` is converted to :math:`(v_x, v_y, v_z)` with

.. math::

    \begin{pmatrix} v_x \\ v_y \\ v_z \end{pmatrix}
    =
    \begin{pmatrix}
        \sin \theta \cos \varphi & \cos \theta \cos \varphi & -\sin \varphi \\
        \sin \theta \sin \varphi & \cos \theta \sin \varphi & \cos \varphi \\
        \cos \theta & -\sin \theta & 0
    \end{pmatrix}
    \begin{pmatrix} v_r \\ v_\theta \\ v_\varphi \end{pmatrix}

Args:
    v (float or complex, 3-array): Vector in 3-dimensional spherical space
    r (float, 3-array): Spherical coordinates in 3-dimensional space

Returns:
    float or complex, 3-array
""",  # docstring
    0,  # unused
    '(3),(3)->(3)',  # signature
)


cdef void *gufunc_vsph2cyl_data[2]
gufunc_vsph2cyl_data[0] = <void*>_coord.vsph2cyl[double]
gufunc_vsph2cyl_data[1] = <void*>_coord.vsph2cyl[double_complex]

vsph2cyl = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcoord_loops,
    gufunc_vsph2cyl_data,
    gufunc_vcoord_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsph2cyl',  # function name
    r"""vsph2cyl(v, r)

Convert vector in spherical coordinates to vector in cylindrical coordinates

The spherical vector :math:`(v_r, v_\theta, v_\varphi)`  at coordinates
:math:`(r, \theta,  \varphi)` is converted to :math:`(v_\rho, v_\varphi, v_z)` with

.. math::

    \begin{pmatrix} v_\rho \\ v_\varphi \\ v_z \end{pmatrix}
    =
    \begin{pmatrix}
        \sin \theta & \cos \theta & 0 \\
        0 & 0 & 1 \\
        \cos \theta & -\sin \theta & 0
    \end{pmatrix}
    \begin{pmatrix} v_r \\ v_\theta \\ v_\varphi \end{pmatrix}

Args:
    v (float or complex, 3-array): Vector in 3-dimensional spherical space
    r (float, 3-array): Spherical coordinates in 3-dimensional space

Returns:
    float or complex, 3-array
""",  # docstring
    0,  # unused
    '(3),(3)->(3)',  # signature
)


cdef void *gufunc_vcar2pol_data[2]
gufunc_vcar2pol_data[0] = <void*>_coord.vcar2pol[double]
gufunc_vcar2pol_data[1] = <void*>_coord.vcar2pol[double_complex]

vcar2pol = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcoord_loops,
    gufunc_vcar2pol_data,
    gufunc_vcoord_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcar2pol',  # function name
    r"""vcar2pol(v, r)

Convert vector in cartesian coordinates to vector in polar coordinates

The cartesian vector :math:`(v_x, v_y)`  at coordinates :math:`(x, y)` is
converted to :math:`(v_\rho, v_\varphi)` with

.. math::

    \begin{pmatrix} v_\rho \\ v_\varphi \end{pmatrix}
    =
    \begin{pmatrix}
        \cos \varphi & \sin \varphi \\
        -\sin \varphi & \cos \varphi
    \end{pmatrix}
    \begin{pmatrix} v_x \\ v_y \end{pmatrix}

Args:
    v (float or complex, 3-array): Vector in 2-dimensional cartesian space
    r (float, 2-array): Cartesian coordinates in 2-dimensional space

Returns:
    float or complex, 2-array
""",  # docstring
    0,  # unused
    '(2),(2)->(2)',  # signature
)


cdef void *gufunc_vpol2car_data[2]
gufunc_vpol2car_data[0] = <void*>_coord.vpol2car[double]
gufunc_vpol2car_data[1] = <void*>_coord.vpol2car[double_complex]

vpol2car = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcoord_loops,
    gufunc_vpol2car_data,
    gufunc_vcoord_types,
    2,  # number of supported input types
    2,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vpol2car',  # function name
    r"""vpol2car(v, r)

Convert vector in polar coordinates to vector in cartesian coordinates

The polar vector :math:`(v_\rho, v_\varphi)`  at coordinates
:math:`(\rho, \varphi)` is converted to :math:`(v_x, v_y)` with

.. math::

    \begin{pmatrix} v_x \\ v_y \end{pmatrix}
    =
    \begin{pmatrix}
        \cos \varphi & -\sin \varphi \\
        \sin \varphi & \cos \varphi
    \end{pmatrix}
    \begin{pmatrix} v_\rho \\ v_\varphi \end{pmatrix}

Args:
    v (float or complex, 3-array): Vector in 3-dimensional cylindrical space
    r (float, 3-array): Cylindrical coordinates in 3-dimensional space

Returns:
    float or complex, 3-array
""",  # docstring
    0,  # unused
    '(2),(2)->(2)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vsh_X_loops[2]
cdef void *gufunc_vsh_X_data[2]
cdef char gufunc_vsh_X_types[2 * 5]

gufunc_vsh_X_loops[0] = <np.PyUFuncGenericFunction>loop_D_lldd
gufunc_vsh_X_loops[1] = <np.PyUFuncGenericFunction>loop_D_llDd
gufunc_vsh_X_types[0] = <char>np.NPY_LONG
gufunc_vsh_X_types[1] = <char>np.NPY_LONG
gufunc_vsh_X_types[2] = <char>np.NPY_DOUBLE
gufunc_vsh_X_types[3] = <char>np.NPY_DOUBLE
gufunc_vsh_X_types[4] = <char>np.NPY_CDOUBLE
gufunc_vsh_X_types[5] = <char>np.NPY_LONG
gufunc_vsh_X_types[6] = <char>np.NPY_LONG
gufunc_vsh_X_types[7] = <char>np.NPY_CDOUBLE
gufunc_vsh_X_types[8] = <char>np.NPY_DOUBLE
gufunc_vsh_X_types[9] = <char>np.NPY_CDOUBLE
gufunc_vsh_X_data[0] = <void*>_waves.vsh_X[double]
gufunc_vsh_X_data[1] = <void*>_waves.vsh_X[double_complex]

vsh_X = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vsh_X_loops,
    gufunc_vsh_X_data,
    gufunc_vsh_X_types,
    2,  # number of supported input types
    4,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsh_X',  # function name
    r"""vsh_X(l, m, theta, phi)

Vector spherical harmonic X in spherical coordinates

The vector spherical harmonics can be defined via the (scalar) spherical harmonics
(:func:`ptsa.special.sph_harm`)

.. math::

    \boldsymbol X_{lm}(\theta, \varphi)
    = \frac{1}{\sqrt{l (l + 1)}} \nabla Y_{lm}(\theta, \varphi)

which can be expressed as

.. math::

    \boldsymbol X_{lm}(\theta, \varphi)
    = \mathrm i \sqrt{\frac{2 l + 1}{4 \pi l (l + 1)} \frac{(l - m)!}{(l + m)!}}
    \left(\mathrm i \pi_{lm}(\theta, \varphi) \boldsymbol{\hat\theta} - \tau_{lm}(\theta, \varphi) \boldsymbol{\hat\varphi}\right)

with the angular functions :func:`ptsa.special.pi_fun` and
:func:`ptsa.special.tau_fun`.

Args:
    l (int, array_like): Degree :math:`l \geq 0`
    m (int, array_like): Order :math:`|m| \leq l`
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angle

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vsh_Y_loops[2]
cdef void *gufunc_vsh_Y_data[2]
cdef char gufunc_vsh_Y_types[2 * 5]

gufunc_vsh_Y_loops[0] = <np.PyUFuncGenericFunction>loop_D_lldd
gufunc_vsh_Y_loops[1] = <np.PyUFuncGenericFunction>loop_D_llDd
gufunc_vsh_Y_types[0] = <char>np.NPY_LONG
gufunc_vsh_Y_types[1] = <char>np.NPY_LONG
gufunc_vsh_Y_types[2] = <char>np.NPY_DOUBLE
gufunc_vsh_Y_types[3] = <char>np.NPY_DOUBLE
gufunc_vsh_Y_types[4] = <char>np.NPY_CDOUBLE
gufunc_vsh_Y_types[5] = <char>np.NPY_LONG
gufunc_vsh_Y_types[6] = <char>np.NPY_LONG
gufunc_vsh_Y_types[7] = <char>np.NPY_CDOUBLE
gufunc_vsh_Y_types[8] = <char>np.NPY_DOUBLE
gufunc_vsh_Y_types[9] = <char>np.NPY_CDOUBLE
gufunc_vsh_Y_data[0] = <void*>_waves.vsh_Y[double]
gufunc_vsh_Y_data[1] = <void*>_waves.vsh_Y[double_complex]

vsh_Y = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vsh_Y_loops,
    gufunc_vsh_Y_data,
    gufunc_vsh_Y_types,
    2,  # number of supported input types
    4,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsh_Y',  # function name
    r"""vsh_Y(l, m, theta, phi)

Vector spherical harmonic Y in spherical coordinates

The vector spherical harmonics can be defined by
:math:`\boldsymbol Y_{lm}(\theta, \varphi) = \boldsymbol{\hat r} \times \boldsymbol X_{lm}(\theta, \varphi)`
using :func:`ptsa.special.vsh_X`. Alternatively, it can can be expressed as

.. math::

    \boldsymbol Y_{lm}(\theta, \varphi)
    = \mathrm i \sqrt{\frac{2 l + 1}{4 \pi l (l + 1)} \frac{(l - m)!}{(l + m)!}}
    \left(\tau_{lm}(\theta, \varphi) \boldsymbol{\hat\theta} + \mathrm i \pi_{lm}(\theta, \varphi) \boldsymbol{\hat\varphi}\right)

with the angular functions :func:`ptsa.special.pi_fun` and
:func:`ptsa.special.tau_fun`.

Args:
    l (int, array_like): Degree :math:`l \geq 0`
    m (int, array_like): Order :math:`|m| \leq l`
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angle

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vsh_Z_loops[2]
cdef void *gufunc_vsh_Z_data[2]
cdef char gufunc_vsh_Z_types[2 * 5]

gufunc_vsh_Z_loops[0] = <np.PyUFuncGenericFunction>loop_D_lldd
gufunc_vsh_Z_loops[1] = <np.PyUFuncGenericFunction>loop_D_llDd
gufunc_vsh_Z_types[0] = <char>np.NPY_LONG
gufunc_vsh_Z_types[1] = <char>np.NPY_LONG
gufunc_vsh_Z_types[2] = <char>np.NPY_DOUBLE
gufunc_vsh_Z_types[3] = <char>np.NPY_DOUBLE
gufunc_vsh_Z_types[4] = <char>np.NPY_CDOUBLE
gufunc_vsh_Z_types[5] = <char>np.NPY_LONG
gufunc_vsh_Z_types[6] = <char>np.NPY_LONG
gufunc_vsh_Z_types[7] = <char>np.NPY_CDOUBLE
gufunc_vsh_Z_types[8] = <char>np.NPY_DOUBLE
gufunc_vsh_Z_types[9] = <char>np.NPY_CDOUBLE
gufunc_vsh_Z_data[0] = <void*>_waves.vsh_Z[double]
gufunc_vsh_Z_data[1] = <void*>_waves.vsh_Z[double_complex]

vsh_Z = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vsh_Z_loops,
    gufunc_vsh_Z_data,
    gufunc_vsh_Z_types,
    2,  # number of supported input types
    4,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsh_Z',  # function name
    r"""vsh_Z(l, m, theta, phi)

Vector spherical harmonic Z in spherical coordinates

The vector spherical harmonics can be defined via the (scalar) spherical harmonics
(:func:`ptsa.special.sph_harm`) by
:math:`\boldsymbol Z_{lm}(\theta, \varphi) = \mathrm i Y_{lm}(\theta, \varphi) \boldsymbol{\hat r}`.

Args:
    l (int, array_like): Degree :math:`l \geq 0`
    m (int, array_like): Order :math:`|m| \leq l`
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angle

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vsw_M_loops[2]
cdef void *gufunc_vsw_M_data[2]
cdef char gufunc_vsw_M_types[2 * 6]

gufunc_vsw_M_loops[0] = <np.PyUFuncGenericFunction>loop_D_llDdd
gufunc_vsw_M_loops[1] = <np.PyUFuncGenericFunction>loop_D_llDDd
gufunc_vsw_M_types[0] = <char>np.NPY_LONG
gufunc_vsw_M_types[1] = <char>np.NPY_LONG
gufunc_vsw_M_types[2] = <char>np.NPY_CDOUBLE
gufunc_vsw_M_types[3] = <char>np.NPY_DOUBLE
gufunc_vsw_M_types[4] = <char>np.NPY_DOUBLE
gufunc_vsw_M_types[5] = <char>np.NPY_CDOUBLE
gufunc_vsw_M_types[6] = <char>np.NPY_LONG
gufunc_vsw_M_types[7] = <char>np.NPY_LONG
gufunc_vsw_M_types[8] = <char>np.NPY_CDOUBLE
gufunc_vsw_M_types[9] = <char>np.NPY_CDOUBLE
gufunc_vsw_M_types[10] = <char>np.NPY_DOUBLE
gufunc_vsw_M_types[11] = <char>np.NPY_CDOUBLE
gufunc_vsw_M_data[0] = <void*>_waves.vsw_M[double]
gufunc_vsw_M_data[1] = <void*>_waves.vsw_M[double_complex]

vsw_M = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vsw_M_loops,
    gufunc_vsw_M_data,
    gufunc_vsw_M_types,
    2,  # number of supported input types
    5,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsw_M',  # function name
    r"""vsw_M(l, m, x, theta, phi)

Singular vector spherical wave M

The vector spherical wave is defined by
:math:`\boldsymbol M_{lm}^{(3)}(x, \theta, \varphi) = h_l^{(1)} (x) \boldsymbol X_{lm}(\theta, \varphi)`
using :func:`ptsa.special.spherical_hankel1` and :func:`ptsa.special.vsh_X`.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation, that is also tangential on a spherical surface. This is often used to
describe a transverse electric (in spherical coordinates) (TE) wave. Additionally,
the term magnetic (refferring to the multipole) is used for this solution.

Args:
    l (int, array_like): Degree :math:`l \geq 0`
    m (int, array_like): Order :math:`|m| \leq l`
    x (float or complex, array_like): Distance in units of the wave number :math:`kr`
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angle

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vsw_N_loops[2]
cdef void *gufunc_vsw_N_data[2]
cdef char gufunc_vsw_N_types[2 * 6]

gufunc_vsw_N_loops[0] = <np.PyUFuncGenericFunction>loop_D_llDdd
gufunc_vsw_N_loops[1] = <np.PyUFuncGenericFunction>loop_D_llDDd
gufunc_vsw_N_types[0] = <char>np.NPY_LONG
gufunc_vsw_N_types[1] = <char>np.NPY_LONG
gufunc_vsw_N_types[2] = <char>np.NPY_CDOUBLE
gufunc_vsw_N_types[3] = <char>np.NPY_DOUBLE
gufunc_vsw_N_types[4] = <char>np.NPY_DOUBLE
gufunc_vsw_N_types[5] = <char>np.NPY_CDOUBLE
gufunc_vsw_N_types[6] = <char>np.NPY_LONG
gufunc_vsw_N_types[7] = <char>np.NPY_LONG
gufunc_vsw_N_types[8] = <char>np.NPY_CDOUBLE
gufunc_vsw_N_types[9] = <char>np.NPY_CDOUBLE
gufunc_vsw_N_types[10] = <char>np.NPY_DOUBLE
gufunc_vsw_N_types[11] = <char>np.NPY_CDOUBLE
gufunc_vsw_N_data[0] = <void*>_waves.vsw_N[double]
gufunc_vsw_N_data[1] = <void*>_waves.vsw_N[double_complex]

vsw_N = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vsw_N_loops,
    gufunc_vsw_N_data,
    gufunc_vsw_N_types,
    2,  # number of supported input types
    5,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsw_N',  # function name
    r"""vsw_N(l, m, x, theta, phi)

Singular vector spherical wave N

The vector spherical wave is defined by

.. math::

    \boldsymbol N_{lm}^{(3)}(x, \theta, \varphi)
    = \nabla \times \boldsymbol M_{lm}^{(3)}(x, \theta, \varphi) \\
    = \left(h_l^{(1)}'(x) + \frac{h_l^{(1)}(x)}{x}\right) \boldsymbol Y_{lm}(\theta, \varphi)
    + \sqrt{l (l + 1)} \frac{h_l^{(1)}(x)}{x} \boldsymbol Z_{lm}(\theta, \varphi)

with :func:`ptsa.special.vsw_M`, :func:`ptsa.special.vsh_Y`,
:func:`ptsa.special.vsh_Z`, and :func:`ptsa.special.spherical_hankel1`.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. This is often used to describe a transverse magnetic (in spherical
coordinates) (TM) wave. Additionally, the term electric (refferring to the
multipole) is used for this solution.

Args:
    l (int, array_like): Degree :math:`l \geq 0`
    m (int, array_like): Order :math:`|m| \leq l`
    x (float or complex, array_like): Distance in units of the wave number :math:`kr`
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angle

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vsw_rM_loops[2]
cdef void *gufunc_vsw_rM_data[2]
cdef char gufunc_vsw_rM_types[2 * 6]

gufunc_vsw_rM_loops[0] = <np.PyUFuncGenericFunction>loop_D_llddd
gufunc_vsw_rM_loops[1] = <np.PyUFuncGenericFunction>loop_D_llDDd
gufunc_vsw_rM_types[0] = <char>np.NPY_LONG
gufunc_vsw_rM_types[1] = <char>np.NPY_LONG
gufunc_vsw_rM_types[2] = <char>np.NPY_DOUBLE
gufunc_vsw_rM_types[3] = <char>np.NPY_DOUBLE
gufunc_vsw_rM_types[4] = <char>np.NPY_DOUBLE
gufunc_vsw_rM_types[5] = <char>np.NPY_CDOUBLE
gufunc_vsw_rM_types[6] = <char>np.NPY_LONG
gufunc_vsw_rM_types[7] = <char>np.NPY_LONG
gufunc_vsw_rM_types[8] = <char>np.NPY_CDOUBLE
gufunc_vsw_rM_types[9] = <char>np.NPY_CDOUBLE
gufunc_vsw_rM_types[10] = <char>np.NPY_DOUBLE
gufunc_vsw_rM_types[11] = <char>np.NPY_CDOUBLE
gufunc_vsw_rM_data[0] = <void*>_waves.vsw_rM[double]
gufunc_vsw_rM_data[1] = <void*>_waves.vsw_rM[double_complex]

vsw_rM = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vsw_rM_loops,
    gufunc_vsw_rM_data,
    gufunc_vsw_rM_types,
    2,  # number of supported input types
    5,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsw_rM',  # function name
    r"""vsw_rM(l, m, x, theta, phi)

Regular vector spherical wave M

The vector spherical wave is defined by
:math:`\boldsymbol M_{lm}^{(1)}(x, \theta, \varphi) = j_l(x) \boldsymbol X_{lm}(\theta, \varphi)`
using :func:`ptsa.special.spherical_jn` and :func:`ptsa.special.vsh_X`.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation, that is also tangential on a spherical surface. This is often used to
describe a transverse electric (in spherical coordinates) (TE) wave. Additionally,
the term magnetic (refferring to the multipole) is used for this solution.

Args:
    l (int, array_like): Degree :math:`l \geq 0`
    m (int, array_like): Order :math:`|m| \leq l`
    x (float or complex, array_like): Distance in units of the wave number :math:`kr`
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angle

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vsw_rN_loops[2]
cdef void *gufunc_vsw_rN_data[2]
cdef char gufunc_vsw_rN_types[2 * 6]

gufunc_vsw_rN_loops[0] = <np.PyUFuncGenericFunction>loop_D_llddd
gufunc_vsw_rN_loops[1] = <np.PyUFuncGenericFunction>loop_D_llDDd
gufunc_vsw_rN_types[0] = <char>np.NPY_LONG
gufunc_vsw_rN_types[1] = <char>np.NPY_LONG
gufunc_vsw_rN_types[2] = <char>np.NPY_DOUBLE
gufunc_vsw_rN_types[3] = <char>np.NPY_DOUBLE
gufunc_vsw_rN_types[4] = <char>np.NPY_DOUBLE
gufunc_vsw_rN_types[5] = <char>np.NPY_CDOUBLE
gufunc_vsw_rN_types[6] = <char>np.NPY_LONG
gufunc_vsw_rN_types[7] = <char>np.NPY_LONG
gufunc_vsw_rN_types[8] = <char>np.NPY_CDOUBLE
gufunc_vsw_rN_types[9] = <char>np.NPY_CDOUBLE
gufunc_vsw_rN_types[10] = <char>np.NPY_DOUBLE
gufunc_vsw_rN_types[11] = <char>np.NPY_CDOUBLE
gufunc_vsw_rN_data[0] = <void*>_waves.vsw_rN[double]
gufunc_vsw_rN_data[1] = <void*>_waves.vsw_rN[double_complex]

vsw_rN = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vsw_rN_loops,
    gufunc_vsw_rN_data,
    gufunc_vsw_rN_types,
    2,  # number of supported input types
    5,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsw_rN',  # function name
    r"""vsw_rN(l, m, x, theta, phi)

Regular vector spherical wave N

The vector spherical wave is defined by

.. math::

    \boldsymbol N_{lm}^{(1)}(x, \theta, \varphi)
    = \nabla \times \boldsymbol M_{lm}^{(1)}(x, \theta, \varphi) \\
    = \left(j_l'(x) + \frac{j_l(x)}{x}\right) \boldsymbol Y_{lm}(\theta, \varphi)
    + \sqrt{l (l + 1)} \frac{j_l(x)}{x} \boldsymbol Z_{lm}(\theta, \varphi)

with :func:`ptsa.special.vsw_M`, :func:`ptsa.special.vsh_Y`,
:func:`ptsa.special.vsh_Z`, and :func:`ptsa.special.spherical_jn`.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. This is often used to describe a transverse magnetic (in spherical
coordinates) (TM) wave. Additionally, the term electric (refferring to the
multipole) is used for this solution.

Args:
    l (int, array_like): Degree :math:`l \geq 0`
    m (int, array_like): Order :math:`|m| \leq l`
    x (float or complex, array_like): Distance in units of the wave number :math:`kr`
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angle

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vsw_rA_loops[2]
cdef void *gufunc_vsw_rA_data[2]
cdef char gufunc_vsw_rA_types[2 * 7]

gufunc_vsw_rA_loops[0] = <np.PyUFuncGenericFunction>loop_D_lldddl
gufunc_vsw_rA_loops[1] = <np.PyUFuncGenericFunction>loop_D_llDDdl
gufunc_vsw_rA_types[0] = <char>np.NPY_LONG
gufunc_vsw_rA_types[1] = <char>np.NPY_LONG
gufunc_vsw_rA_types[2] = <char>np.NPY_DOUBLE
gufunc_vsw_rA_types[3] = <char>np.NPY_DOUBLE
gufunc_vsw_rA_types[4] = <char>np.NPY_DOUBLE
gufunc_vsw_rA_types[5] = <char>np.NPY_LONG
gufunc_vsw_rA_types[6] = <char>np.NPY_CDOUBLE
gufunc_vsw_rA_types[7] = <char>np.NPY_LONG
gufunc_vsw_rA_types[8] = <char>np.NPY_LONG
gufunc_vsw_rA_types[9] = <char>np.NPY_CDOUBLE
gufunc_vsw_rA_types[10] = <char>np.NPY_CDOUBLE
gufunc_vsw_rA_types[11] = <char>np.NPY_DOUBLE
gufunc_vsw_rA_types[12] = <char>np.NPY_LONG
gufunc_vsw_rA_types[13] = <char>np.NPY_CDOUBLE
gufunc_vsw_rA_data[0] = <void*>_waves.vsw_rA[double]
gufunc_vsw_rA_data[1] = <void*>_waves.vsw_rA[double_complex]

vsw_rA = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vsw_rA_loops,
    gufunc_vsw_rA_data,
    gufunc_vsw_rA_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsw_rA',  # function name
    r"""vsw_rA(l, m, x, theta, phi, p)

Regular helical vector spherical wave

The vector spherical wave is defined by

.. math::

    \boldsymbol A_{lm\pm}^{(1)}(x, \theta, \varphi)
    = \frac{\boldsymbol N_{lm}^{(1)}(x, \theta, \varphi) \pm \boldsymbol M_{lm}^{(1)}(x, \theta, \varphi)}{\sqrt{2}}

with :func:`ptsa.special.vsw_rM` and :func:`ptsa.special.vsw_rN`. The sign is
determined by `p`, where `0` corresponds to negative and `1` to positive helicity.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. Additionally, it has a well-defined helicity.

Args:
    l (int, array_like): Degree :math:`l \geq 0`
    m (int, array_like): Order :math:`|m| \leq l`
    x (float or complex, array_like): Distance in units of the wave number :math:`kr`
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angle
    p (bool, array_like): Helicity

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vsw_A_loops[2]
cdef void *gufunc_vsw_A_data[2]
cdef char gufunc_vsw_A_types[2 * 7]

gufunc_vsw_A_loops[0] = <np.PyUFuncGenericFunction>loop_D_llDddl
gufunc_vsw_A_loops[1] = <np.PyUFuncGenericFunction>loop_D_llDDdl
gufunc_vsw_A_types[0] = <char>np.NPY_LONG
gufunc_vsw_A_types[1] = <char>np.NPY_LONG
gufunc_vsw_A_types[2] = <char>np.NPY_CDOUBLE
gufunc_vsw_A_types[3] = <char>np.NPY_DOUBLE
gufunc_vsw_A_types[4] = <char>np.NPY_DOUBLE
gufunc_vsw_A_types[5] = <char>np.NPY_LONG
gufunc_vsw_A_types[6] = <char>np.NPY_CDOUBLE
gufunc_vsw_A_types[7] = <char>np.NPY_LONG
gufunc_vsw_A_types[8] = <char>np.NPY_LONG
gufunc_vsw_A_types[9] = <char>np.NPY_CDOUBLE
gufunc_vsw_A_types[10] = <char>np.NPY_CDOUBLE
gufunc_vsw_A_types[11] = <char>np.NPY_DOUBLE
gufunc_vsw_A_types[12] = <char>np.NPY_LONG
gufunc_vsw_A_types[13] = <char>np.NPY_CDOUBLE
gufunc_vsw_A_data[0] = <void*>_waves.vsw_A[double]
gufunc_vsw_A_data[1] = <void*>_waves.vsw_A[double_complex]

vsw_A = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vsw_A_loops,
    gufunc_vsw_A_data,
    gufunc_vsw_A_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vsw_A',  # function name
    r"""vsw_A(l, m, x, theta, phi, p)

Singular helical vector spherical wave

The vector spherical wave is defined by

.. math::

    \boldsymbol A_{lm\pm}^{(3)}(x, \theta, \varphi)
    = \frac{\boldsymbol N_{lm}^{(3)}(x, \theta, \varphi) \pm \boldsymbol M_{lm}^{(3)}(x, \theta, \varphi)}{\sqrt{2}}

with :func:`ptsa.special.vsw_M` and :func:`ptsa.special.vsw_N`. The sign is
determined by `p`, where `0` corresponds to negative and `1` to positive helicity.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. Additionally, it has a well-defined helicity.

Args:
    l (int, array_like): Degree :math:`l \geq 0`
    m (int, array_like): Order :math:`|m| \leq l`
    x (float or complex, array_like): Distance in units of the wave number :math:`kr`
    theta (float or complex, array_like): Polar angle
    phi (float, array_like): Azimuthal angle
    p (bool, array_like): Helicity

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vcw_M_loops[1]
cdef void *gufunc_vcw_M_data[1]
cdef char gufunc_vcw_M_types[6]

gufunc_vcw_M_loops[0] = <np.PyUFuncGenericFunction>loop_D_dlDdd
gufunc_vcw_M_types[0] = <char>np.NPY_DOUBLE
gufunc_vcw_M_types[1] = <char>np.NPY_LONG
gufunc_vcw_M_types[2] = <char>np.NPY_CDOUBLE
gufunc_vcw_M_types[3] = <char>np.NPY_DOUBLE
gufunc_vcw_M_types[4] = <char>np.NPY_DOUBLE
gufunc_vcw_M_types[5] = <char>np.NPY_CDOUBLE
gufunc_vcw_M_data[0] = <void*>_waves.vcw_M

vcw_M = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcw_M_loops,
    gufunc_vcw_M_data,
    gufunc_vcw_M_types,
    1,  # number of supported input types
    5,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcw_M',  # function name
    r"""vcw_M(kz, m, xrho, phi, z)

Singular vector cylindrical wave M

The vector cylindrical wave is defined by

.. math::

    \boldsymbol M_{k_z m}^{(3)}(x_\rho, \varphi, z)
    = \left(\frac{\mathrm i m}{x_\rho} H_m^{(1)}(x_\rho)\boldsymbol{\hat{\rho}} - H_m^{(1)}'(x_\rho) \boldsymbol{\hat{\varphi}}\right)
    \mathrm e^{\mathrm i (k_z z + m \varphi)}

using :func:`ptsa.special.hankel1`.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation, that is also tangential on a cylindrical surface. This is often used to
describe a transverse electric (in cylindrical coordinates) (TE) wave.

Args:
    kz (float, array_like): Z component of the wave
    m (int, array_like): Order :math:`|m| \leq l`
    xrho (float or complex, array_like): Radial in units of the wave number :math:`k_\rho \rho`
    phi (float, array_like): Azimuthal angle
    z (float, array_like): Z coordinate

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vcw_rM_loops[2]
cdef void *gufunc_vcw_rM_data[2]
cdef char gufunc_vcw_rM_types[2 * 6]

gufunc_vcw_rM_loops[0] = <np.PyUFuncGenericFunction>loop_D_dlddd
gufunc_vcw_rM_loops[1] = <np.PyUFuncGenericFunction>loop_D_dlDdd
gufunc_vcw_rM_types[0] = <char>np.NPY_DOUBLE
gufunc_vcw_rM_types[1] = <char>np.NPY_LONG
gufunc_vcw_rM_types[2] = <char>np.NPY_DOUBLE
gufunc_vcw_rM_types[3] = <char>np.NPY_DOUBLE
gufunc_vcw_rM_types[4] = <char>np.NPY_DOUBLE
gufunc_vcw_rM_types[5] = <char>np.NPY_CDOUBLE
gufunc_vcw_rM_types[6] = <char>np.NPY_DOUBLE
gufunc_vcw_rM_types[7] = <char>np.NPY_LONG
gufunc_vcw_rM_types[8] = <char>np.NPY_CDOUBLE
gufunc_vcw_rM_types[9] = <char>np.NPY_DOUBLE
gufunc_vcw_rM_types[10] = <char>np.NPY_DOUBLE
gufunc_vcw_rM_types[11] = <char>np.NPY_CDOUBLE
gufunc_vcw_rM_data[0] = <void*>_waves.vcw_rM[double]
gufunc_vcw_rM_data[1] = <void*>_waves.vcw_rM[double_complex]

vcw_rM = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcw_rM_loops,
    gufunc_vcw_rM_data,
    gufunc_vcw_rM_types,
    2,  # number of supported input types
    5,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcw_rM',  # function name
    r"""vcw_rM(kz, m, xrho, phi, z)

Regular vector cylindrical wave M

The vector cylindrical wave is defined by

.. math::

    \boldsymbol M_{k_z m}^{(1)}(x_\rho, \varphi, z)
    = \left(\frac{\mathrm i m}{x_\rho} J_m(x_\rho)\boldsymbol{\hat{\rho}} - J_m'(x_\rho) \boldsymbol{\hat{\varphi}}\right)
    \mathrm e^{\mathrm i (k_z z + m \varphi)}

using :func:`ptsa.special.jn`.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation, that is also tangential on a cylindrical surface. This is often used to
describe a transverse electric (in cylindrical coordinates) (TE) wave.

Args:
    kz (float, array_like): Z component of the wave
    m (int, array_like): Order :math:`|m| \leq l`
    xrho (float or complex, array_like): Radial in units of the wave number :math:`k_\rho \rho`
    phi (float, array_like): Azimuthal angle
    z (float, array_like): Z coordinate

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vcw_N_loops[1]
cdef void *gufunc_vcw_N_data[1]
cdef char gufunc_vcw_N_types[7]

gufunc_vcw_N_loops[0] = <np.PyUFuncGenericFunction>loop_D_dlDddD
gufunc_vcw_N_types[0] = <char>np.NPY_DOUBLE
gufunc_vcw_N_types[1] = <char>np.NPY_LONG
gufunc_vcw_N_types[2] = <char>np.NPY_CDOUBLE
gufunc_vcw_N_types[3] = <char>np.NPY_DOUBLE
gufunc_vcw_N_types[4] = <char>np.NPY_DOUBLE
gufunc_vcw_N_types[5] = <char>np.NPY_CDOUBLE
gufunc_vcw_N_types[6] = <char>np.NPY_CDOUBLE
gufunc_vcw_N_data[0] = <void*>_waves.vcw_N

vcw_N = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcw_N_loops,
    gufunc_vcw_N_data,
    gufunc_vcw_N_types,
    1,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcw_N',  # function name
    r"""vcw_N(kz, m, xrho, phi, z, k)

Singular vector cylindrical wave N

The vector cylindrical wave is defined by

.. math::

    \boldsymbol N_{k_z m}^{(3)}(x_\rho, \varphi, z)
    = \left(\frac{\mathrm i k_z}{k} H_m^{(1)}'(x_\rho)\boldsymbol{\hat{\rho}} - \frac{m k_z}{k x_\rho} H_m^{(1)}(x_\rho) \boldsymbol{\hat{\varphi}} + \frac{k_\rho}{k} H_m^{(1)}(x_\rho)\right)
    \mathrm e^{\mathrm i (k_z z + m \varphi)}

using :func:`ptsa.special.hankel1`.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. This is often used to describe a transverse magnetic (in cylindrical
coordinates) (TM) wave.

Args:
    kz (float, array_like): Z component of the wave
    m (int, array_like): Order :math:`|m| \leq l`
    xrho (float or complex, array_like): Radial in units of the wave number :math:`k_\rho \rho`
    phi (float, array_like): Azimuthal angle
    z (float, array_like): Z coordinate
    k (float or complex): Wave number

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vcw_rN_loops[2]
cdef void *gufunc_vcw_rN_data[2]
cdef char gufunc_vcw_rN_types[2 * 7]

gufunc_vcw_rN_loops[0] = <np.PyUFuncGenericFunction>loop_D_dldddD
gufunc_vcw_rN_loops[1] = <np.PyUFuncGenericFunction>loop_D_dlDddD
gufunc_vcw_rN_types[0] = <char>np.NPY_DOUBLE
gufunc_vcw_rN_types[1] = <char>np.NPY_LONG
gufunc_vcw_rN_types[2] = <char>np.NPY_DOUBLE
gufunc_vcw_rN_types[3] = <char>np.NPY_DOUBLE
gufunc_vcw_rN_types[4] = <char>np.NPY_DOUBLE
gufunc_vcw_rN_types[5] = <char>np.NPY_CDOUBLE
gufunc_vcw_rN_types[6] = <char>np.NPY_CDOUBLE
gufunc_vcw_rN_types[7] = <char>np.NPY_DOUBLE
gufunc_vcw_rN_types[8] = <char>np.NPY_LONG
gufunc_vcw_rN_types[9] = <char>np.NPY_CDOUBLE
gufunc_vcw_rN_types[10] = <char>np.NPY_DOUBLE
gufunc_vcw_rN_types[11] = <char>np.NPY_DOUBLE
gufunc_vcw_rN_types[12] = <char>np.NPY_CDOUBLE
gufunc_vcw_rN_types[13] = <char>np.NPY_CDOUBLE
gufunc_vcw_rN_data[0] = <void*>_waves.vcw_rN[double]
gufunc_vcw_rN_data[1] = <void*>_waves.vcw_rN[double_complex]

vcw_rN = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcw_rN_loops,
    gufunc_vcw_rN_data,
    gufunc_vcw_rN_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcw_rN',  # function name
    r"""vcw_N(kz, m, xrho, phi, z, k)

Regular vector cylindrical wave N

The vector cylindrical wave is defined by

.. math::

    \boldsymbol N_{k_z m}^{(3)}(x_\rho, \varphi, z)
    = \left(\frac{\mathrm i k_z}{k} J_m'(x_\rho)\boldsymbol{\hat{\rho}} - \frac{m k_z}{k x_\rho} J_m(x_\rho) \boldsymbol{\hat{\varphi}} + \frac{k_\rho}{k} J_m(x_\rho)\right)
    \mathrm e^{\mathrm i (k_z z + m \varphi)}

using :func:`ptsa.special.jv`.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. This is often used to describe a transverse magnetic (in cylindrical
coordinates) (TM) wave.

Args:
    kz (float, array_like): Z component of the wave
    m (int, array_like): Order :math:`|m| \leq l`
    xrho (float or complex, array_like): Radial in units of the wave number :math:`k_\rho \rho`
    phi (float, array_like): Azimuthal angle
    z (float, array_like): Z coordinate
    k (float or complex): Wave number

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vcw_A_loops[1]
cdef void *gufunc_vcw_A_data[1]
cdef char gufunc_vcw_A_types[8]

gufunc_vcw_A_loops[0] = <np.PyUFuncGenericFunction>loop_D_dlDddDl
gufunc_vcw_A_types[0] = <char>np.NPY_DOUBLE
gufunc_vcw_A_types[1] = <char>np.NPY_LONG
gufunc_vcw_A_types[2] = <char>np.NPY_CDOUBLE
gufunc_vcw_A_types[3] = <char>np.NPY_DOUBLE
gufunc_vcw_A_types[4] = <char>np.NPY_DOUBLE
gufunc_vcw_A_types[5] = <char>np.NPY_CDOUBLE
gufunc_vcw_A_types[6] = <char>np.NPY_LONG
gufunc_vcw_A_types[7] = <char>np.NPY_CDOUBLE
gufunc_vcw_A_data[0] = <void*>_waves.vcw_A

vcw_A = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcw_A_loops,
    gufunc_vcw_A_data,
    gufunc_vcw_A_types,
    1,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcw_A',  # function name
    r"""vcw_A(kz, m, xrho, phi, z, k, pol)

Singular helical vector cylindrical wave

The vector spherical wave is defined by

.. math::

    \boldsymbol A_{k_z m \pm}^{(3)}(x_\rho, \varphi, z)
    = \frac{\boldsymbol N_{k_z m}^{(3)}(x_\rho, \varphi, z) \pm \boldsymbol M_{k_z m}^{(3)}(x_\rho, \varphi, z)}{\sqrt{2}}

with :func:`ptsa.special.vcw_M` and :func:`ptsa.special.vcw_N`. The sign is
determined by `p`, where `0` corresponds to negative and `1` to positive helicity.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. Additionally, it has a well-defined helicity.

Args:
    kz (float, array_like): Z component of the wave
    m (int, array_like): Order :math:`|m| \leq l`
    xrho (float or complex, array_like): Radial in units of the wave number :math:`k_\rho \rho`
    phi (float, array_like): Azimuthal angle
    z (float, array_like): Z coordinate
    k (float or complex): Wave number
    pol (bool, array_like): Polarization

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vcw_rA_loops[2]
cdef void *gufunc_vcw_rA_data[2]
cdef char gufunc_vcw_rA_types[2 * 8]

gufunc_vcw_rA_loops[0] = <np.PyUFuncGenericFunction>loop_D_dldddDl
gufunc_vcw_rA_loops[1] = <np.PyUFuncGenericFunction>loop_D_dlDddDl
gufunc_vcw_rA_types[0] = <char>np.NPY_DOUBLE
gufunc_vcw_rA_types[1] = <char>np.NPY_LONG
gufunc_vcw_rA_types[2] = <char>np.NPY_DOUBLE
gufunc_vcw_rA_types[3] = <char>np.NPY_DOUBLE
gufunc_vcw_rA_types[4] = <char>np.NPY_DOUBLE
gufunc_vcw_rA_types[5] = <char>np.NPY_CDOUBLE
gufunc_vcw_rA_types[6] = <char>np.NPY_LONG
gufunc_vcw_rA_types[7] = <char>np.NPY_CDOUBLE
gufunc_vcw_rA_types[8] = <char>np.NPY_DOUBLE
gufunc_vcw_rA_types[9] = <char>np.NPY_LONG
gufunc_vcw_rA_types[10] = <char>np.NPY_CDOUBLE
gufunc_vcw_rA_types[11] = <char>np.NPY_DOUBLE
gufunc_vcw_rA_types[12] = <char>np.NPY_DOUBLE
gufunc_vcw_rA_types[13] = <char>np.NPY_CDOUBLE
gufunc_vcw_rA_types[14] = <char>np.NPY_LONG
gufunc_vcw_rA_types[15] = <char>np.NPY_CDOUBLE
gufunc_vcw_rA_data[0] = <void*>_waves.vcw_rA[double]
gufunc_vcw_rA_data[1] = <void*>_waves.vcw_rA[double_complex]

vcw_rA = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vcw_rA_loops,
    gufunc_vcw_rA_data,
    gufunc_vcw_rA_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vcw_rA',  # function name
    r"""vcw_rA(kz, m, xrho, phi, z, k)

Regular helical vector cylindrical wave

The vector spherical wave is defined by

.. math::

    \boldsymbol A_{k_z m \pm}^{(1)}(x_\rho, \varphi, z)
    = \frac{\boldsymbol N_{k_z m}^{(1)}(x_\rho, \varphi, z) \pm \boldsymbol M_{k_z m}^{(1)}(x_\rho, \varphi, z)}{\sqrt{2}}

with :func:`ptsa.special.vcw_rM` and :func:`ptsa.special.vcw_rN`. The sign is
determined by `p`, where `0` corresponds to negative and `1` to positive helicity.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. Additionally, it has a well-defined helicity.

Args:
    kz (float, array_like): Z component of the wave
    m (int, array_like): Order :math:`|m| \leq l`
    xrho (float or complex, array_like): Radial in units of the wave number :math:`k_\rho \rho`
    phi (float, array_like): Azimuthal angle
    z (float, array_like): Z coordinate
    k (float or complex): Wave number
    pol (bool, array_like): Polarization

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vpw_M_loops[2]
cdef void *gufunc_vpw_M_data[2]
cdef char gufunc_vpw_M_types[2 * 7]

gufunc_vpw_M_loops[0] = <np.PyUFuncGenericFunction>loop_D_dddddd
gufunc_vpw_M_loops[1] = <np.PyUFuncGenericFunction>loop_D_DDDddd
gufunc_vpw_M_types[0] = <char>np.NPY_DOUBLE
gufunc_vpw_M_types[1] = <char>np.NPY_DOUBLE
gufunc_vpw_M_types[2] = <char>np.NPY_DOUBLE
gufunc_vpw_M_types[3] = <char>np.NPY_DOUBLE
gufunc_vpw_M_types[4] = <char>np.NPY_DOUBLE
gufunc_vpw_M_types[5] = <char>np.NPY_DOUBLE
gufunc_vpw_M_types[6] = <char>np.NPY_CDOUBLE
gufunc_vpw_M_types[7] = <char>np.NPY_CDOUBLE
gufunc_vpw_M_types[8] = <char>np.NPY_CDOUBLE
gufunc_vpw_M_types[9] = <char>np.NPY_CDOUBLE
gufunc_vpw_M_types[10] = <char>np.NPY_DOUBLE
gufunc_vpw_M_types[11] = <char>np.NPY_DOUBLE
gufunc_vpw_M_types[12] = <char>np.NPY_DOUBLE
gufunc_vpw_M_types[13] = <char>np.NPY_CDOUBLE
gufunc_vpw_M_data[0] = <void*>_waves.vpw_M[double]
gufunc_vpw_M_data[1] = <void*>_waves.vpw_M[double_complex]

vpw_M = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vpw_M_loops,
    gufunc_vpw_M_data,
    gufunc_vpw_M_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vpw_M',  # function name
    r"""vpw_M(kx, ky, kz, x, y, z)

Vector plane wave M

The vector plane wave is defined by

.. math::

    \boldsymbol M_{\boldsymbol k}(\boldsymbol r)
    = - \mathrm i \boldsymbol{\hat \varphi}_{\boldsymbol k} \mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation, that is also tangential on a planar surface. This is often used to
describe a transverse electric (TE) wave.

Args:
    kx (float or complex, array_like): X component of the wave vector
    ky (float or complex, array_like): Y component of the wave vector
    kz (float or complex, array_like): Z component of the wave vector
    x (float, array_like): Y coordinate
    y (float, array_like): X coordinate
    z (float, array_like): Z coordinate

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vpw_N_loops[2]
cdef void *gufunc_vpw_N_data[2]
cdef char gufunc_vpw_N_types[2 * 7]

gufunc_vpw_N_loops[0] = <np.PyUFuncGenericFunction>loop_D_dddddd
gufunc_vpw_N_loops[1] = <np.PyUFuncGenericFunction>loop_D_DDDddd
gufunc_vpw_N_types[0] = <char>np.NPY_DOUBLE
gufunc_vpw_N_types[1] = <char>np.NPY_DOUBLE
gufunc_vpw_N_types[2] = <char>np.NPY_DOUBLE
gufunc_vpw_N_types[3] = <char>np.NPY_DOUBLE
gufunc_vpw_N_types[4] = <char>np.NPY_DOUBLE
gufunc_vpw_N_types[5] = <char>np.NPY_DOUBLE
gufunc_vpw_N_types[6] = <char>np.NPY_CDOUBLE
gufunc_vpw_N_types[7] = <char>np.NPY_CDOUBLE
gufunc_vpw_N_types[8] = <char>np.NPY_CDOUBLE
gufunc_vpw_N_types[9] = <char>np.NPY_CDOUBLE
gufunc_vpw_N_types[10] = <char>np.NPY_DOUBLE
gufunc_vpw_N_types[11] = <char>np.NPY_DOUBLE
gufunc_vpw_N_types[12] = <char>np.NPY_DOUBLE
gufunc_vpw_N_types[13] = <char>np.NPY_CDOUBLE
gufunc_vpw_N_data[0] = <void*>_waves.vpw_N[double]
gufunc_vpw_N_data[1] = <void*>_waves.vpw_N[double_complex]

vpw_N = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vpw_N_loops,
    gufunc_vpw_N_data,
    gufunc_vpw_N_types,
    2,  # number of supported input types
    6,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vpw_N',  # function name
    r"""vpw_N(kx, ky, kz, x, y, z)

Vector plane wave N

The vector plane wave is defined by

.. math::

    \boldsymbol N_{\boldsymbol k}(\boldsymbol r)
    = - \boldsymbol{\hat \theta}_{\boldsymbol k} \mathrm e^{\mathrm i \boldsymbol k \boldsymbol r}.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. This is often used to describe a transverse magnetic (TM) wave.

Args:
    kx (float or complex, array_like): X component of the wave vector
    ky (float or complex, array_like): Y component of the wave vector
    kz (float or complex, array_like): Z component of the wave vector
    x (float, array_like): Y coordinate
    y (float, array_like): X coordinate
    z (float, array_like): Z coordinate

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),(),()->(3)',  # signature
)


cdef np.PyUFuncGenericFunction gufunc_vpw_A_loops[2]
cdef void *gufunc_vpw_A_data[2]
cdef char gufunc_vpw_A_types[2 * 8]

gufunc_vpw_A_loops[0] = <np.PyUFuncGenericFunction>loop_D_ddddddl
gufunc_vpw_A_loops[1] = <np.PyUFuncGenericFunction>loop_D_DDDdddl
gufunc_vpw_A_types[0] = <char>np.NPY_DOUBLE
gufunc_vpw_A_types[1] = <char>np.NPY_DOUBLE
gufunc_vpw_A_types[2] = <char>np.NPY_DOUBLE
gufunc_vpw_A_types[3] = <char>np.NPY_DOUBLE
gufunc_vpw_A_types[4] = <char>np.NPY_DOUBLE
gufunc_vpw_A_types[5] = <char>np.NPY_DOUBLE
gufunc_vpw_A_types[6] = <char>np.NPY_LONG
gufunc_vpw_A_types[7] = <char>np.NPY_CDOUBLE
gufunc_vpw_A_types[8] = <char>np.NPY_CDOUBLE
gufunc_vpw_A_types[9] = <char>np.NPY_CDOUBLE
gufunc_vpw_A_types[10] = <char>np.NPY_CDOUBLE
gufunc_vpw_A_types[11] = <char>np.NPY_DOUBLE
gufunc_vpw_A_types[12] = <char>np.NPY_DOUBLE
gufunc_vpw_A_types[13] = <char>np.NPY_DOUBLE
gufunc_vpw_A_types[14] = <char>np.NPY_LONG
gufunc_vpw_A_types[15] = <char>np.NPY_CDOUBLE
gufunc_vpw_A_data[0] = <void*>_waves.vpw_A[double]
gufunc_vpw_A_data[1] = <void*>_waves.vpw_A[double_complex]

vpw_A = np.PyUFunc_FromFuncAndDataAndSignature(
    gufunc_vpw_A_loops,
    gufunc_vpw_A_data,
    gufunc_vpw_A_types,
    2,  # number of supported input types
    7,  # number of input args
    1,  # number of output args
    0,  # identity element
    'vpw_A',  # function name
    r"""vpw_A(kx, ky, kz, x, y, z, p)

Vector plane wave of well-defined helicity

The vector plane wave is defined by

.. math::

    \boldsymbol A_{\boldsymbol k \pm}(\boldsymbol r)
    = \frac{\boldsymbol N_{\boldsymbol k \pm}(\boldsymbol r) \pm \boldsymbol M_{\boldsymbol k}(\boldsymbol r)}{\sqrt{2}}

with :func:`ptsa.special.vpw_M` and :func:`ptsa.special.vpw_N`. The sign is
determined by `p`, where `0` corresponds to negative and `1` to positive helicity.

This function is describing a transverse solution to the vectorial Helmholtz wave
equation. Additionally, it has a well-defined helicity.

Args:
    kx (float or complex, array_like): X component of the wave vector
    ky (float or complex, array_like): Y component of the wave vector
    kz (float or complex, array_like): Z component of the wave vector
    x (float, array_like): Y coordinate
    y (float, array_like): X coordinate
    z (float, array_like): Z coordinate
    p (bool, array_like): Helicity

Returns:
    complex, 3-array
""",  # docstring
    0,  # unused
    '(),(),(),(),(),(),()->(3)',  # signature
)
