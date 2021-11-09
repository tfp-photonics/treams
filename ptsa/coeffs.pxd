ctypedef fused number_t:
    double complex
    double

cdef void _mie(long l, double *x, number_t *epsilon, number_t *mu, number_t *kappa, long n, double complex res[2][2]) nogil
cdef void _mie_cyl(double kz, long m, double k0, double *radii, double complex *epsilon, double complex *mu, double complex *kappa, long n, double complex res[2][2]) nogil
cdef void _fresnel(number_t ks[2][2], number_t kzs[2][2], number_t zs[2], number_t res[2][2][2][2]) nogil
