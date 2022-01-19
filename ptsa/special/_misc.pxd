from cython cimport numeric

cdef double SQPI

ctypedef double complex double_complex

ctypedef fused number_t:
    double complex
    double


ctypedef fused real_t:
    double
    long


cdef double abs(number_t x) nogil
cdef number_t cos(number_t x) nogil
cdef number_t exp(number_t x) nogil
cdef number_t pow(number_t x, number_t y) nogil
cdef number_t sin(number_t x) nogil
cdef number_t sqrt(number_t x) nogil

cdef long minusonepow(long l) nogil
cdef long array_zero(numeric *n, long l) nogil
cdef long ipow(long base, long exponent) nogil

cdef inline real_t max(real_t a, real_t b) nogil:
    return (a if a > b else b)

cdef inline real_t min(real_t a, real_t b) nogil:
    return (a if a < b else b)
