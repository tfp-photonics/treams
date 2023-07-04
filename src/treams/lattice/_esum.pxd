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

cdef double complex recsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil
cdef double complex recsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, number_t eta) nogil
cdef double complex recsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex recsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil
cdef double complex recsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil
cdef double complex recsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex recsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex recsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil

cdef double complex realsumcw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil
cdef double complex realsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, number_t eta) nogil
cdef double complex realsumcw2d(long l, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex realsumsw1d(long l, number_t k, double kpar, double a, double r, number_t eta) nogil
cdef double complex realsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, number_t eta) nogil
cdef double complex realsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex realsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil
cdef double complex realsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, number_t eta) nogil

cdef double complex zero3d(double complex eta) nogil
cdef double complex zero2d(double complex eta) nogil
