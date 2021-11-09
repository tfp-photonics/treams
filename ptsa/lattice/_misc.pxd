ctypedef fused real_t:
    double
    long

cdef real_t area(real_t *a, real_t *b) nogil
cdef long cube_next(long *r, long d, long n) nogil
cdef long cubeedge_next(long *r, long d, long n) nogil
cpdef diffr_orders_circle(real_t[:,:] b, double rmax)
cdef long ncube(long d, long n) nogil
cdef long nedge(long d, long n) nogil
cdef void recvec2(double *a0, double *a1, double *b0, double *b1) nogil
cdef void recvec3(double *a0, double *a1, double *a2, double *b0, double *b1, double *b2) nogil
cdef real_t volume(real_t *a, real_t *b, real_t *c) nogil
