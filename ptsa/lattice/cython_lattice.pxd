ctypedef fused real_t:
    double
    long

ctypedef fused number_t:
    double complex
    double

cdef real_t area(real_t *a, real_t *b) nogil
cdef long cube_next(long *r, long d, long n) nogil
cdef long cubeedge_next(long *r, long d, long n) nogil
cpdef diffr_orders_circle(real_t[:, :] b, double rmax)
cpdef long ncube(long d, long n) nogil
cpdef long nedge(long d, long n) nogil
cdef void recvec2(double *a0, double *a1, double *b0, double *b1) nogil
cdef void recvec3(double *a0, double *a1, double *a2, double *b0, double *b1, double *b2) nogil
cdef real_t volume(real_t *a, real_t *b, real_t *c) nogil

cpdef double complex lsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil
cdef double complex lsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, number_t eta) nogil
cdef double complex lsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cpdef double complex lsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil
cdef double complex lsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil
cdef double complex lsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex lsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex lsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil

cpdef double complex dsumcw1d(long l, number_t k, double kpar, double a, double r, long i) nogil
cdef double complex dsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, long i) nogil
cdef double complex dsumcw2d(long l, number_t k, double *kpar, double *a, double *r, long i) nogil
cpdef double complex dsumsw1d(long l, number_t k, double kpar, double a, double r, long i) nogil
cdef double complex dsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, long i) nogil
cdef double complex dsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil
cdef double complex dsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil
cdef double complex dsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil
