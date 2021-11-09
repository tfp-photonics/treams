ctypedef fused number_t:
    double complex
    double

cdef double complex lsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil
cdef double complex lsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, number_t eta) nogil
cdef double complex lsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex lsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil
cdef double complex lsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil
cdef double complex lsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex lsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex lsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
