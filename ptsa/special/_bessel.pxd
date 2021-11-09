ctypedef fused number_t:
    double complex
    double


cdef double complex hankel1_d(double l, double complex z) nogil
cdef double complex hankel2_d(double l, double complex z) nogil
cdef number_t jv_d(double l, number_t z) nogil
cdef double complex spherical_hankel1(double n, double complex z) nogil
cdef double complex spherical_hankel1_d(double l, double complex z) nogil
cdef double complex spherical_hankel2(double n, double complex z) nogil
cdef double complex spherical_hankel2_d(double l, double complex z) nogil
cdef number_t yv_d(double l, number_t z) nogil
