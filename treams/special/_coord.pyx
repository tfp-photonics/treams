"""
Coordinate transformations

Intended to be used in a generalized Numpy ufunc. `input` and `output` are allowed to
be identical.
"""

from libc.math cimport atan2, cos, hypot, sin


cdef void car2cyl(double *input, double *output, long istep, long ostep) nogil:
    """Convert cartesian to cylindrical coordinate for gufunc"""
    cdef double phi = atan2(input[istep], input[0])
    output[0] = hypot(input[0], input[istep])
    output[ostep] = phi
    output[2 * ostep] = input[2 * istep]


cdef void car2pol(double *input, double *output, long istep, long ostep) nogil:
    """Convert cartesian to polar coordinate for gufunc"""
    cdef double phi = atan2(input[istep], input[0])
    output[0] = hypot(input[0], input[istep])
    output[ostep] = phi


cdef void car2sph(double *input, double *output, long istep, long ostep) nogil:
    """Convert cartesian to spherical coordinate for gufunc"""
    cdef double xy = hypot(input[0], input[istep])
    cdef double z = input[2 * istep]
    output[2 * ostep] = atan2(input[istep], input[0])
    output[0] = hypot(xy, z)
    output[ostep] = atan2(xy, z)


cdef void cyl2car(double *input, double *output, long istep, long ostep) nogil:
    """Convert cylindrical to cartesian coordinate for gufunc"""
    cdef double y = input[0] * sin(input[istep])
    output[0] = input[0] * cos(input[istep])
    output[ostep] = y
    output[2 * ostep] = input[2 * istep]


cdef void cyl2sph(double *input, double *output, long istep, long ostep) nogil:
    """Convert cylindrical to spherical coordinate for gufunc"""
    cdef double r = hypot(input[0], input[2 * istep])
    cdef double phi = input[istep]
    output[0] = r
    output[ostep] = atan2(input[0], input[2 * istep])
    output[2 * ostep] = phi


cdef void pol2car(double *input, double *output, long istep, long ostep) nogil:
    """Convert polar to cartesian coordinate for gufunc"""
    cdef double y = input[0] * sin(input[istep])
    output[0] = input[0] * cos(input[istep])
    output[ostep] = y


cdef void sph2car(double *input, double *output, long istep, long ostep) nogil:
    """Convert spherical to spherical coordinate for gufunc"""
    cdef double rs = input[0] * sin(input[istep])
    output[0] = rs * cos(input[2 * istep])
    output[ostep] = rs * sin(input[2 * istep])
    output[2 * ostep] = input[0] * cos(input[istep])


cdef void sph2cyl(double *input, double *output, long istep, long ostep) nogil:
    """Convert spherical to cylindrical coordinate for gufunc"""
    cdef double phi = input[2 * istep]
    output[0] = input[0] * sin(input[istep])
    output[2 * ostep] = input[0] * cos(input[istep])
    output[ostep] = phi


cdef void vcar2cyl(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil:
    """Convert vector from cartesian to cylindrical basis for gufunc"""
    vcar2pol(iv, ip, ov, ivs, ips, ovs)
    ov[2 * ovs] = iv[2 * ivs]


cdef void vcar2pol(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil:
    """Convert vector from cartesian to polar basis for gufunc"""
    cdef double op[2]
    car2pol(ip, op, ips, 1)
    cdef number_t vphi = cos(op[1]) * iv[ivs] - sin(op[1]) * iv[0]
    ov[0] = cos(op[1]) * iv[0] + sin(op[1]) * iv[ivs]
    ov[ovs] = vphi


cdef void vcar2sph(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil:
    """Convert vector from cartesian to spherical basis for gufunc"""
    cdef double op[3]
    # TODO: no actual need to call the coordinate version
    car2sph(ip, op, ips, 1)
    cdef number_t vtheta = cos(op[1]) * (iv[0] * cos(op[2]) + iv[ivs] * sin(op[2])) - iv[2 * ivs] * sin(op[1])
    cdef number_t vphi = iv[ivs] * cos(op[2]) - iv[0] * sin(op[2])
    ov[0] = sin(op[1]) * (iv[0] * cos(op[2]) + iv[ivs] * sin(op[2])) + iv[2 * ivs] * cos(op[1])
    ov[ovs] = vtheta
    ov[2 * ovs] = vphi


cdef void vcyl2car(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil:
    """Convert vector from cylindrical to cartesian basis for gufunc"""
    vpol2car(iv, ip, ov, ivs, ips, ovs)
    ov[2 * ovs] = iv[2 * ivs]


cdef void vcyl2sph(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil:
    """Convert vector from cylindrical to spherical basis for gufunc"""
    cdef double op[3]
    cyl2sph(ip, op, ips, 1)
    cdef number_t vphi = iv[ivs]
    ov[ovs] = cos(op[1]) * iv[0] - sin(op[1]) * iv[2 * ivs]
    ov[0] = sin(op[1]) * iv[0] + cos(op[1]) * iv[2 * ivs]
    ov[2 * ovs] = vphi


cdef void vpol2car(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil:
    """Convert vector from polar to cartesian basis for gufunc"""
    cdef number_t vy = cos(ip[ips]) * iv[ivs] + sin(ip[ips]) * iv[0]
    ov[0] = cos(ip[ips]) * iv[0] - sin(ip[ips]) * iv[ivs]
    ov[ovs] = vy


cdef void vsph2car(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil:
    """Convert vector from spherical to cartesian basis for gufunc"""
    cdef number_t vy = sin(ip[ips]) * sin(ip[2 * ips]) * iv[0] + cos(ip[ips]) * sin(ip[2 * ips]) * iv[ivs] + cos(ip[2 * ips]) * iv[2 * ivs]
    cdef number_t vz = cos(ip[ips]) * iv[0] - sin(ip[ips]) * iv[ivs]
    ov[0] = sin(ip[ips]) * cos(ip[2 * ips]) * iv[0] + cos(ip[ips]) * cos(ip[2 * ips]) * iv[ivs] - sin(ip[2 * ips]) * iv[2 * ivs]
    ov[ovs] = vy
    ov[2 * ovs] = vz


cdef void vsph2cyl(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil:
    """Convert vector from spherical to cylindrical basis for gufunc"""
    cdef number_t vphi = iv[2 * ivs]
    ov[2 * ovs] = cos(ip[ips]) * iv[0] - sin(ip[ips]) * iv[ivs]
    ov[0] = sin(ip[ips]) * iv[0] + cos(ip[ips]) * iv[ivs]
    ov[ovs] = vphi
