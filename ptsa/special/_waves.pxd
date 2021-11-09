ctypedef fused number_t:
    double complex
    double

cdef double complex clpmv(double m, double l, double complex z) nogil
cdef double complex csph_harm(double m, double l, double phi, double complex theta) nogil
cdef number_t lpmv(double m, double l, number_t z) nogil
cdef number_t pi_fun(double l, double m, number_t theta) nogil
cdef double complex sph_harm(double m, double l, double phi, number_t theta) nogil
cdef number_t tau_fun(double l, double m, number_t theta) nogil

cdef double complex tl_vsw_A(long lambda_, long mu, long l, long m, double complex kr, number_t theta, double phi) nogil
cdef double complex tl_vsw_B(long lambda_, long mu, long l, long m, double complex kr, number_t theta, double phi) nogil
cdef double complex tl_vsw_rA(long lambda_, long mu, long l, long m, number_t kr, number_t theta, double phi) nogil
cdef double complex tl_vsw_rB(long lambda_, long mu, long l, long m, number_t kr, number_t theta, double phi) nogil

cdef double complex tl_vcw(double kz1, long mu, double kz2, long m, double complex krr, double phi, double z) nogil
cdef double complex tl_vcw_r(double kz1, long mu, double kz2, long m, number_t krr, double phi, double z) nogil

cdef void vcw_A(double kz, long m, double complex krr, double phi, double z, double complex k, long pol, double complex *out, long i) nogil
cdef void vcw_M(double kz, long m, double complex krr, double phi, double z, double complex *out, long i) nogil
cdef void vcw_N(double kz, long m, double complex krr, double phi, double z, double complex k, double complex *out, long i) nogil
cdef void vcw_rA(double kz, long m, number_t, double phi, double z, double complex k, long pol, double complex *out, long i) nogil
cdef void vcw_rM(double kz, long m, number_t krr, double phi, double z, double complex *out, long i) nogil
cdef void vcw_rN(double kz, long m, number_t krr, double phi, double z, double complex k, double complex *out, long i) nogil

cdef void vpw_A(number_t kx, number_t ky, number_t kz, double x, double y, double z, long pol, double complex *out, long i) nogil
cdef void vpw_M(number_t kx, number_t ky, number_t kz, double x, double y, double z, double complex *out, long i) nogil
cdef void vpw_N(number_t kx, number_t ky, number_t kz, double x, double y, double z, double complex *out, long i) nogil

cdef void vsh_X(long l, long m, number_t theta, double phi, double complex *out, long i) nogil
cdef void vsh_Y(long l, long m, number_t theta, double phi, double complex *out, long i) nogil
cdef void vsh_Z(long l, long m, number_t theta, double phi, double complex *out, long i) nogil

cdef void vsw_A(long l, long m, double complex kr, number_t theta, double phi, long pol, double complex *out, long i) nogil
cdef void vsw_M(long l, long m, double complex kr, number_t theta, double phi, double complex *out, long i) nogil
cdef void vsw_N(long l, long m, double complex kr, number_t theta, double phi, double complex *out, long i) nogil
cdef void vsw_rA(long l, long m, number_t kr, number_t theta, double phi, long pol, double complex *out, long i) nogil
cdef void vsw_rM(long l, long m, number_t kr, number_t theta, double phi, double complex *out, long i) nogil
cdef void vsw_rN(long l, long m, number_t kr, number_t theta, double phi, double complex *out, long i) nogil

# vsw_L gufunc
# vsw_rL gufunc
# vcw_L gufunc
# vcw_rL gufunc
