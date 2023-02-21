ctypedef fused number_t:
    double complex
    double

cdef void car2cyl(double *input, double *output, long istep, long ostep) nogil
cdef void car2pol(double *input, double *output, long istep, long ostep) nogil
cdef void car2sph(double *input, double *output, long istep, long ostep) nogil
cdef void cyl2car(double *input, double *output, long istep, long ostep) nogil
cdef void cyl2sph(double *input, double *output, long istep, long ostep) nogil
cdef void pol2car(double *input, double *output, long istep, long ostep) nogil
cdef void sph2cyl(double *input, double *output, long istep, long ostep) nogil
cdef void sph2car(double *input, double *output, long istep, long ostep) nogil

cdef void vcar2cyl(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil
cdef void vcar2pol(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil
cdef void vcar2sph(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil
cdef void vcyl2car(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil
cdef void vcyl2sph(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil
cdef void vpol2car(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil
cdef void vsph2car(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil
cdef void vsph2cyl(number_t *iv, double *ip, number_t *ov, long ivs, long ips, long ovs) nogil
