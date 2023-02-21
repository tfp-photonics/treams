ctypedef fused number_t:
    double complex
    double

cdef number_t incgamma(double n, number_t z) nogil
cdef number_t intkambe(long n, number_t z, number_t eta) nogil
