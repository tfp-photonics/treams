ctypedef fused number_t:
    double complex
    double

cdef double complex dsumcw1d(long l, number_t k, double kpar, double a, double r, long i) nogil
cdef double complex dsumcw1d_shift(long l, number_t k, double kpar, double a, double *r, long i) nogil
cdef double complex dsumcw2d(long l, number_t k, double *kpar, double *a, double *r, long i) nogil
cdef double complex dsumsw1d(long l, number_t k, double kpar, double a, double r, long i) nogil
cdef double complex dsumsw1d_shift(long l, long m, number_t k, double kpar, double a, double *r, long i) nogil
cdef double complex dsumsw2d(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil
cdef double complex dsumsw2d_shift(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil
cdef double complex dsumsw3d(long l, long m, number_t k, double *kpar, double *a, double *r, long i) nogil
