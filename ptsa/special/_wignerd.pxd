ctypedef fused number_t:
    double complex
    double

cdef number_t wignerd(long l, long m, long k, number_t theta) nogil
cdef double complex wignerD(long l, long m, long k, double phi, number_t theta, double psi) nogil
