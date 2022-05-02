ctypedef fused number_t:
    double complex
    double


cpdef double complex hankel1_d(double l, double complex z) nogil
cpdef double complex hankel2_d(double l, double complex z) nogil
cpdef number_t jv_d(double l, number_t z) nogil
cpdef double complex spherical_hankel1(double n, number_t z) nogil
cpdef double complex spherical_hankel1_d(double l, double complex z) nogil
cpdef double complex spherical_hankel2(double n, number_t z) nogil
cpdef double complex spherical_hankel2_d(double l, double complex z) nogil
cpdef number_t yv_d(double l, number_t z) nogil


cdef void car2cyl(double *input, double *output) nogil
cdef void car2pol(double *input, double *output) nogil
cdef void car2sph(double *input, double *output) nogil
cdef void cyl2car(double *input, double *output) nogil
cdef void cyl2sph(double *input, double *output) nogil
cdef void pol2car(double *input, double *output) nogil
cdef void sph2car(double *input, double *output) nogil
cdef void sph2cyl(double *input, double *output) nogil

cdef void vcar2cyl(number_t *iv, double *ip, number_t *ov) nogil
cdef void vcar2pol(number_t *iv, double *ip, number_t *ov) nogil
cdef void vcar2sph(number_t *iv, double *ip, number_t *ov) nogil
cdef void vcyl2car(number_t *iv, double *ip, number_t *ov) nogil
cdef void vcyl2sph(number_t *iv, double *ip, number_t *ov) nogil
cdef void vpol2car(number_t *iv, double *ip, number_t *ov) nogil
cdef void vsph2car(number_t *iv, double *ip, number_t *ov) nogil
cdef void vsph2cyl(number_t *iv, double *ip, number_t *ov) nogil


cpdef number_t incgamma(double l, number_t z) nogil
cpdef number_t intkambe(long n, number_t z, number_t eta) nogil


cpdef number_t lpmv(double m, double l, number_t z) nogil
cpdef number_t pi_fun(double l, double m, number_t x) nogil
cpdef double complex sph_harm(double m, double l, double phi, number_t theta) nogil
cpdef number_t tau_fun(double l, double m, number_t x) nogil

cpdef double complex tl_vcw(double kz1, long mu, double kz2, long m, double complex krr, double phi, double z) nogil
cpdef double complex tl_vcw_r(double kz1, long mu, double kz2, long m, number_t krr, double phi, double z) nogil

cpdef double complex _tl_vsw_helper(long l, long m, long lambda_, long mu, long p, long q) nogil

cpdef double complex tl_vsw_A(long lambda_, long mu, long l, long m, double complex kr, number_t theta, double phi) nogil
cpdef double complex tl_vsw_B(long lambda_, long mu, long l, long m, double complex kr, number_t theta, double phi) nogil
cpdef double complex tl_vsw_rA(long lambda_, long mu, long l, long m, number_t kr, number_t theta, double phi) nogil
cpdef double complex tl_vsw_rB(long lambda_, long mu, long l, long m, number_t kr, number_t theta, double phi) nogil

cdef void vcw_A(double kz, long m, double complex krr, double phi, double z, double complex k, long pol, double complex *out) nogil
cdef void vcw_M(double kz, long m, double complex krr, double phi, double z, double complex *out) nogil
cdef void vcw_N(double kz, long m, double complex krr, double phi, double z, double complex k, double complex *out) nogil
cdef void vcw_rA(double kz, long m, number_t krr, double phi, double z, double complex k, long pol, double complex *out) nogil
cdef void vcw_rM(double kz, long m, number_t krr, double phi, double z, double complex *out, long i) nogil
cdef void vcw_rN(double kz, long m, number_t krr, double phi, double z, double complex k, double complex *out) nogil

cdef void vpw_A(number_t kx, number_t ky, number_t kz, double x, double y, double z, long pol, double complex *out) nogil
cdef void vpw_M(number_t kx, number_t ky, number_t kz, double x, double y, double z, double complex *out) nogil
cdef void vpw_N(number_t kx, number_t ky, number_t kz, double x, double y, double z, double complex *out) nogil

cdef void vsh_X(long l, long m, number_t theta, double phi, double complex *out) nogil
cdef void vsh_Y(long l, long m, number_t theta, double phi, double complex *out) nogil
cdef void vsh_Z(long l, long m, number_t theta, double phi, double complex *out) nogil

cdef void vsw_A(long l, long m, double complex kr, number_t theta, double phi, long pol, double complex *out) nogil
cdef void vsw_M(long l, long m, double complex kr, number_t theta, double phi, double complex *out) nogil
cdef void vsw_N(long l, long m, double complex kr, number_t theta, double phi, double complex *out) nogil
cdef void vsw_rA(long l, long m, number_t kr, number_t theta, double phi, long pol, double complex *out) nogil
cdef void vsw_rM(long l, long m, number_t kr, number_t theta, double phi, double complex *out) nogil
cdef void vsw_rN(long l, long m, number_t kr, number_t theta, double phi, double complex *out) nogil


cpdef number_t wignersmalld(long l, long m, long k, number_t theta) nogil
cpdef double complex wignerd(long l, long m, long k, double phi, number_t theta, double psi) nogil


cpdef double wigner3j(long j1, long j2, long j3, long m1, long m2, long m3) nogil
