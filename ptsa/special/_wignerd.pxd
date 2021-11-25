ctypedef fused number_t:
    double complex
    double

cdef number_t wignersmalld(long l, long m, long k, number_t theta) nogil
cdef double complex wignerd(long l, long m, long k, double phi, number_t theta, double psi) nogil
