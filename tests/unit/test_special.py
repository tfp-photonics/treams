import pytest

import numpy as np
import scipy.special as ssc

import treams.special as sc
import treams.special.cython_special as cs


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


EPS = 2e-7
EPSSQ = 4e-14


class TestIncgamma:
    def test_zero_real(self):
        assert isclose(sc.incgamma(0, 1.5), 0.10001958240663263, rel_tol=EPSSQ)

    def test_zero_complex(self):
        assert isclose(
            sc.incgamma(0, 2 + 4j),
            0.006575211740584215 + 0.0261438237000811j,
            rel_tol=EPSSQ,
        )

    def test_zero_negreal(self):
        assert isclose(
            sc.incgamma(0, -3 + 0.0j),
            -9.933832570625414 - 3.141592653589793j,
            rel_tol=EPSSQ,
        )

    def test_zero_negreal_branch(self):
        assert isclose(
            sc.incgamma(0, complex(-3, -0.0)),
            -9.933832570625414 + 3.141592653589793j,
            rel_tol=EPSSQ,
        )

    def test_half_real(self):
        assert isclose(sc.incgamma(0.5, 1.5), 0.1475825132040964, rel_tol=EPSSQ)

    def test_half_complex(self):
        assert isclose(
            sc.incgamma(0.5, 2 + 4j),
            -0.01415763494202471 + 0.058731665238669344j,
            rel_tol=EPSSQ,
        )

    def test_half_negreal(self):
        assert isclose(
            sc.incgamma(0.5, -3 + 0.0j),
            1.7724538509055152 - 14.626171384019093j,
            rel_tol=EPSSQ,
        )

    def test_half_negreal_branch(self):
        assert isclose(
            sc.incgamma(0.5, complex(-3, -0.0)),
            1.7724538509055152 + 14.626171384019093j,
            rel_tol=EPSSQ,
        )

    def test_one_real(self):
        assert isclose(sc.incgamma(1, 1.5), 0.22313016014842982, rel_tol=EPSSQ)

    def test_one_complex(self):
        assert isclose(
            sc.incgamma(1, 2 + 4j),
            -0.08846104456538201 + 0.10242208005667372j,
            rel_tol=EPSSQ,
        )

    def test_one_negreal(self):
        assert isclose(sc.incgamma(1, -3), 20.085536923187668, rel_tol=EPSSQ)

    def test_one_negreal_branch(self):
        assert isclose(
            sc.incgamma(1, complex(-3, -0.0)), 20.085536923187668, rel_tol=EPSSQ
        )

    def test_ten_real(self):
        assert isclose(sc.incgamma(10, 1.5), 362878.5130988457, rel_tol=EPSSQ)

    def test_ten_complex(self):
        assert isclose(
            sc.incgamma(10, 2 + 4j),
            345166.2033113096 - 45890.997544067905j,
            rel_tol=EPSSQ,
        )

    def test_ten_negreal(self):
        assert isclose(sc.incgamma(10, -3), 270070.1294691813, rel_tol=EPSSQ)

    def test_ten_negreal_branch(self):
        assert isclose(
            sc.incgamma(10, complex(-3, -0.0)), 270070.1294691813, rel_tol=EPSSQ
        )

    def test_neg_real(self):
        assert isclose(sc.incgamma(-10, 1.5), 0.0003324561166899859, rel_tol=EPSSQ)

    def test_neg_complex(self):
        assert isclose(
            sc.incgamma(-10, 2 + 4j),
            -3.109457703343637e-9 - 9.73849356067146e-10j,
            rel_tol=EPSSQ,
        )

    def test_neg_negreal(self):
        assert isclose(
            sc.incgamma(-10, -3 + 0.0j),
            0.00005342780082756921 - 8.657387162670285e-7j,
            rel_tol=EPSSQ,
        )

    def test_neg_negreal_branch(self):
        assert isclose(
            sc.incgamma(-10, complex(-3, -0.0)),
            0.00005342780082756921 + 8.657387162670285e-7j,
            rel_tol=EPSSQ,
        )

    def test_huge(self):
        assert isclose(
            sc.incgamma(100, 120 - 42j),
            -4.4658781836612545e156 + 1.1325309029755172e155j,
            rel_tol=EPS,
        )

    def test_tiny(self):
        assert isclose(
            sc.incgamma(100, 1e-12 + 1e-3j),
            9.332621544394404e155 + 9.901988480274498e-306j,
            rel_tol=EPSSQ,
        )

    def test_huge_neg(self):
        assert isclose(
            sc.incgamma(-100, 42 - 12j),
            -2.2632633469219943e-185 + 3.010799952817908e-185j,
            rel_tol=EPS,
        )

    def test_tiny_neg(self):
        assert isclose(
            sc.incgamma(-100, 1e-2 + 1e-2j),
            -8.792072241883211e182 + 8.881173950746141e180j,
            rel_tol=EPS,
        )

    def test_huge_half(self):
        assert isclose(
            sc.incgamma(99.5, 120 - 42j),
            -3.9035577320280324e155 - 5.255135904002826e154j,
            rel_tol=EPSSQ,
        )

    def test_tiny_half(self):
        assert isclose(
            sc.incgamma(99.5, 1e-8 + 1e-3j),
            9.367802114655592e154 + 2.249528564769392e-301j,
            rel_tol=EPS,
        )

    def test_huge_neg_half(self):
        assert isclose(
            sc.incgamma(-99.5, 42 - 12j),
            -1.2102237586206635e-184 + 2.1853661691833773e-184j,
            rel_tol=EPS,
        )

    def test_tiny_neg_half(self):
        assert isclose(
            sc.incgamma(-99.5, 1e-2 + 1e-2j),
            -9.748868790182211e181 - 3.9232159891109815e181j,
            rel_tol=EPSSQ,
        )

    @pytest.mark.skip
    def test_error(self):
        with pytest.raises(ValueError):
            sc.incgamma(0.4, 1)

    def test_zero_zero(self):
        result = sc.incgamma(0, 0)
        assert np.isinf(result.real) and result.imag == 0

    def test_neg_zero(self):
        result = sc.incgamma(-1, 0)
        assert np.isinf(result.real) and result.imag == 0

    def test_gamma(self):
        assert sc.incgamma(4, 0) == 6


class TestWigner3j:
    def test_zeros(self):
        assert isclose(sc.wigner3j(0, 0, 0, 0, 0, 0), 1)

    def test_small(self):
        assert isclose(sc.wigner3j(1, 0, 1, 0, 0, 0), -1 / np.sqrt(3))

    def test_medium(self):
        assert isclose(sc.wigner3j(5, 2, 6, 3, -2, -1), np.sqrt(8 / 1001))

    def test_large(self):
        assert isclose(sc.wigner3j(123, 60, 95, -64, 32, 32), -0.007933910368778899)

    def test_large_forward(self):
        assert isclose(sc.wigner3j(123, 60, 66, -64, 32, 32), -0.01069952664596778)

    def test_more(self):
        assert isclose(sc.wigner3j(43, 61, 21, -1, 12, -11), -0.02786405482897469)

    def test_even_more(self):
        assert isclose(sc.wigner3j(43, 61, 22, -9, -12, 21), -0.00005702392146902575)

    @pytest.mark.skip
    def test_error(self):
        with pytest.raises(ValueError):
            sc.wigner3j(-2, 1, 3, 0, 0, 0)

    def test_triangular(self):
        assert sc.wigner3j(3, 1, 1, 0, 0, 0) == 0.0

    def test_physical_jm(self):
        assert sc.wigner3j(1, 1, 1, 0, -2, 2) == 0.0

    def test_physical_m(self):
        assert sc.wigner3j(1, 1, 1, 0, 0, 1) == 0.0

    def test_zero_return(self):
        assert sc.wigner3j(2, 2, 1, 0, 0, 0) == 0.0

    def test_divide_by_zero(self):
        assert isclose(sc.wigner3j(4, 4, 1, 1, -1, 0), -1 / (6 * np.sqrt(5)))

    def test_last_init(self):
        assert isclose(sc.wigner3j(4, 7, 4, 1, 3, -4), np.sqrt(5 / 858))


class TestHankel1:
    def test(self):
        assert sc.hankel1(1, 3 + 1j) == ssc.hankel1(1, 3 + 1j)


class TestHankel2:
    def test(self):
        assert sc.hankel2(1, 3 + 1j) == ssc.hankel2(1, 3 + 1j)


class TestJv:
    def test_real_type(self):
        assert isinstance(sc.jv(1, 3), float)

    def test_real(self):
        assert sc.jv(1, 3) == ssc.jv(1, 3)

    def test_complex_type(self):
        assert isinstance(sc.jv(1, 3 + 0j), complex)

    def test_complex(self):
        assert sc.jv(1, 3 + 1j) == ssc.jv(1, 3 + 1j)


class TestSpericalJn:
    def test_real_type(self):
        assert isinstance(sc.spherical_jn(1, 3), float)

    def test_real(self):
        assert sc.spherical_jn(1, 3) == ssc.spherical_jn(1, 3)

    def test_complex_type(self):
        assert isinstance(sc.spherical_jn(1, 3 + 0j), complex)

    def test_complex(self):
        assert sc.spherical_jn(1, 3 + 1j) == ssc.spherical_jn(1, 3 + 1j)


class TestSpericalYn:
    def test_real_type(self):
        assert isinstance(sc.spherical_yn(1, 3), float)

    def test_real(self):
        assert sc.spherical_yn(1, 3) == ssc.spherical_yn(1, 3)

    def test_complex_type(self):
        assert isinstance(sc.spherical_yn(1, 3 + 0j), complex)

    def test_complex(self):
        assert sc.spherical_yn(1, 3 + 1j) == ssc.spherical_yn(1, 3 + 1j)


class TestYv:
    def test_real_type(self):
        assert isinstance(sc.yv(1, 3), float)

    def test_real(self):
        assert sc.yv(1, 3) == ssc.yv(1, 3)

    def test_complex_type(self):
        assert isinstance(sc.yv(1, 3 + 0j), complex)

    def test_complex(self):
        assert sc.yv(1, 3 + 1j) == ssc.yv(1, 3 + 1j)


class TestLpmv:
    def test_real_type(self):
        assert isinstance(sc.lpmv(1, 2, 0.3), float)

    def test_real(self):
        assert sc.lpmv(1, 2, 0.3) == ssc.lpmv(1, 2, 0.3)

    def test_complex_type(self):
        assert isinstance(sc.lpmv(1, 2, 0.3 + 0j), complex)

    @pytest.mark.skip
    def test_complex_error(self):
        with pytest.raises(ValueError):
            sc.lpmv(0, -1, 0j)

    def test_non_physical(self):
        assert sc.lpmv(2, 1, 0j) == 0

    def test_complex_00(self):
        assert sc.lpmv(0, 0, 4j) == 1

    def test_complex_01(self):
        assert sc.lpmv(0, 1, 3 + 0j) == 3

    def test_complex_even(self):
        assert isclose(
            sc.lpmv(0, 14, 1 / np.sqrt(2) + 0j), ssc.lpmv(0, 14, 1 / np.sqrt(2))
        )

    def test_complex_odd(self):
        assert isclose(
            sc.lpmv(0, 13, -1.2 + 0j), ssc.clpmn(0, 13, -1.2, type=2)[0][0, 13]
        )

    def test_complex_odd_c(self):
        assert isclose(
            sc.lpmv(0, 13, -1.2 - 1j), ssc.clpmn(0, 13, -1.2 - 1j, type=2)[0][0, 13]
        )

    def test_complex_asso(self):
        assert isclose(
            sc.lpmv(3, 3, 0.1 + 0j), ssc.clpmn(3, 3, -0.1 + 0j, type=2)[0][3, 3,]
        )

    def test_complex_asso_above(self):
        assert isclose(
            sc.lpmv(-5, 12, -1.3 + 0j),
            ssc.clpmn(-5, 12, -1.3 + 1e-16j, type=2)[0][5, 12],
        )

    def test_complex_asso_below(self):
        assert isclose(
            sc.lpmv(-5, 12, complex(-1.3, -0.0)),
            ssc.clpmn(-5, 12, complex(-1.3, 0), type=2)[0][5, 12],
        )

    def test_complex_asso_odd(self):
        assert isclose(
            sc.lpmv(-4, 13, 1.3 - 3j), ssc.clpmn(-4, 13, 1.3 - 3j, type=2)[0][4, 13]
        )

    def test_complex_asso_pos(self):
        assert isclose(
            sc.lpmv(5, 12, 0.4 + 0j), ssc.clpmn(5, 12, 0.4 + 0j, type=2)[0][5, 12]
        )

    def test_complex_asso_pos_c(self):
        assert isclose(
            sc.lpmv(5, 12, 1.3 - 1j), ssc.clpmn(5, 12, 1.3 - 1j, type=2)[0][5, 12]
        )

    def test_nan_l(self):
        assert np.isnan(sc.lpmv(0, 0.5, 1j))

    def test_nan_m(self):
        assert np.isnan(sc.lpmv(0.5, 1, 1j))


class TestSphHarm:
    def test_real(self):
        assert sc.sph_harm(1, 2, 3, 4) == ssc.sph_harm(1, 2, 3, 4)

    def test_complex(self):
        assert isclose(
            sc.sph_harm(1, 2, 3, 4j),
            np.sqrt(5 / (24 * np.pi))
            * ssc.clpmn(1, 2, np.cos(4j), type=2)[0][1, 2]
            * np.exp(3j),
        )

    def test_nan_l(self):
        assert np.isnan(sc.sph_harm(0, 0.5, 0, 1j))

    def test_nan_m(self):
        assert np.isnan(sc.sph_harm(0.5, 1, 0, 1j))


class TestSphHankel1:
    def test(self):
        assert isclose(
            sc.spherical_hankel1(1, 2 + 1j), 0.0589704984384257 - 0.1995739736279250j
        )


class TestSphHankel2:
    def test(self):
        assert isclose(
            sc.spherical_hankel2(1, 2 + 1j), 1.0624415871431773 + 0.2312289984789762j
        )


class TestJvD:
    def test_zero(self):
        assert isclose(sc.jv_d(0, 3), ssc.jvp(0, 3))

    def test_non_zero(self):
        assert isclose(sc.jv_d(2, 3 + 4j), ssc.jvp(2, 3 + 4j))


class TestYvD:
    def test_zero(self):
        assert isclose(sc.yv_d(0, 3), ssc.yvp(0, 3))

    def test_non_zero(self):
        assert isclose(sc.yv_d(2, 3 + 4j), ssc.yvp(2, 3 + 4j))


class TestHankel1D:
    def test_zero(self):
        assert isclose(sc.hankel1_d(0, 3), ssc.h1vp(0, 3))

    def test_non_zero(self):
        assert isclose(sc.hankel1_d(2, 3 + 4j), ssc.h1vp(2, 3 + 4j))


class TestHankel2D:
    def test_zero(self):
        assert isclose(sc.hankel2_d(0, 3), ssc.h2vp(0, 3))

    def test_non_zero(self):
        assert isclose(sc.hankel2_d(2, 3 + 4j), ssc.h2vp(2, 3 + 4j))


class TestSphericalJnD:
    def test_zero(self):
        assert isclose(sc.spherical_jn_d(0, 3), ssc.spherical_jn(0, 3, True))

    def test_non_zero(self):
        assert isclose(sc.spherical_jn_d(2, 3 + 4j), ssc.spherical_jn(2, 3 + 4j, True))

    def test_zero_arg(self):
        assert isclose(sc.spherical_jn_d(3, 0), ssc.spherical_jn(3, 0, True))

    def test_zero_arg_2(self):
        assert isclose(sc.spherical_jn_d(1, 0), ssc.spherical_jn(1, 0, True))


class TestSphericalYnD:
    def test_zero(self):
        assert isclose(sc.spherical_yn_d(0, 3), ssc.spherical_yn(0, 3, True))

    def test_non_zero(self):
        assert isclose(sc.spherical_yn_d(2, 3 + 4j), ssc.spherical_yn(2, 3 + 4j, True))


class TestSphericalHankel1D:
    def test_zero(self):
        assert isclose(
            sc.spherical_hankel1_d(0, 3), -0.3456774997623560 - 0.0629591636023160j
        )

    def test_non_zero(self):
        assert isclose(
            sc.spherical_hankel1_d(2, 3 + 4j),
            0.005890898262660627 - 0.003935166138594885j,
        )


class TestSphericalHankel2D:
    def test_zero(self):
        assert isclose(
            sc.spherical_hankel2_d(0, 3), -0.3456774997623560 + 0.0629591636023160j
        )

    def test_non_zero(self):
        assert isclose(
            sc.spherical_hankel2_d(2, 3 + 4j), 2.193962713943190 - 5.312458889454144j
        )


class TestIntkambe:
    def test_m3_1(self):
        assert isclose(sc.intkambe(-3, 0.8, 1.2), 0.1536931539507005)

    def test_m3_2(self):
        assert isclose(sc.intkambe(-3, 0.3, 0.3), 256.2213077991314, rel_tol=1e-5)

    def test_m3_3(self):
        assert isclose(sc.intkambe(-3, 5, 4), 2.208658257800406e-91)

    def test_m3_complex(self):
        assert isclose(
            sc.intkambe(-3, 1 + 1j, 2 - 1.1j),
            -0.00009809435152495933 - 0.00015139494926761342j,
        )

    def test_m2_1(self):
        assert isclose(sc.intkambe(-2, 0.8, 1.2), 0.2340296298080376)

    def test_m2_2(self):
        assert isclose(sc.intkambe(-2, 0.3, 0.3), 87.3244612483742, rel_tol=1e-5)

    def test_m2_3(self):
        assert isclose(sc.intkambe(-2, 5, 4), 8.8564460016641e-91)

    def test_m2_complex(self):
        assert isclose(
            sc.intkambe(-2, 1 + 1j, 2 - 1.1j),
            -0.00039153718178222677 - 0.00019654173190154122j,
        )

    def test_m1_1(self):
        assert isclose(sc.intkambe(-1, 0.8, 1.2), 0.373562480945606)

    def test_m1_2(self):
        assert isclose(sc.intkambe(-1, 0.3, 0.3), 31.60318167885612, rel_tol=1e-5)

    def test_m1_3(self):
        assert isclose(sc.intkambe(-1, 5, 4), 3.55134657134584e-90)

    def test_m1_complex(self):
        assert isclose(
            sc.intkambe(-1, 1 + 1j, 2 - 1.1j),
            -0.001064442613991556 + 0.00007284003446857335j,
        )

    def test_0_1(self):
        assert isclose(sc.intkambe(0, 0.8, 1.2), 0.6329973232058413)

    def test_0_2(self):
        assert isclose(sc.intkambe(0, 0.3, 0.3), 14.50548439475395)

    def test_0_3(self):
        assert isclose(sc.intkambe(0, 5, 4), 1.424063216552536e-89)

    def test_0_complex(self):
        assert isclose(
            sc.intkambe(0, 1 + 1j, 2 - 1.1j),
            -0.002143348723888108 + 0.0014826182600382856j,
        )

    def test_fodd_1(self):
        assert isclose(sc.intkambe(9, 0.8, 1.2), 3722.587631513942)

    def test_fodd_2(self):
        assert isclose(sc.intkambe(9, 0.4, 0.4), 3.698976253172273e6, rel_tol=1e-5)

    def test_fodd_r(self):
        assert isclose(sc.intkambe(9, 5, 4), 3.818351328744268e-84)

    def test_fodd_complex(self):
        assert isclose(
            sc.intkambe(9, 1.4 + 0.01j, 3 - 0.01j),
            0.8489489550578334 - 0.09535673649786904j,
            rel_tol=1e-7,
        )

    def test_feven_1(self):
        assert isclose(sc.intkambe(10, 0.8, 1.2), 14289.0024493579)

    def test_feven_2(self):
        assert isclose(sc.intkambe(10, 0.4, 0.4), 2.849030924828978e7)

    def test_feven_3(self):
        assert isclose(sc.intkambe(10, 5, 4), 1.53122554488494e-83)

    def test_feven_complex(self):
        assert isclose(
            sc.intkambe(10, 1.4 + 0.01j, 3 - 0.01j),
            2.7511557962581623 - 0.3203862171927444j,
            rel_tol=1e-5,
        )

    @pytest.mark.skip
    def test_error(self):
        with pytest.raises(ValueError):
            sc.intkambe(0.3, 1, 1)

    def test_inf(self):
        assert np.isinf(sc.intkambe(1, 1, 0))

    def test_m3_zero(self):
        assert isclose(
            sc.intkambe(-3, 0, 1 + 1j), -0.031087578289355378 - 0.24740395925452308j
        )

    def test_m2_zero(self):
        assert isclose(
            sc.intkambe(-2, 0, 1 + 1j), 0.4554030049462477 - 0.5383650534833417j
        )

    def test_inf_2(self):
        assert np.isinf(sc.intkambe(-1, 0, 1))

    def test_inf_3(self):
        assert np.isinf(sc.intkambe(-1, 1, 0))

    def test_bodd_1(self):
        assert isclose(sc.intkambe(-9, 0.8, 1.2), 0.021334939311674394170463156)

    def test_bodd_2(self):
        assert isclose(sc.intkambe(-9, 0.4, 0.4), 2552.5567756521395093849138)

    def test_bodd_r(self):
        assert isclose(sc.intkambe(-9, 3, 2), 1.471110396630751e-12)

    def test_bodd_complex(self):
        assert isclose(
            sc.intkambe(-9, 1.4 + 0.01j, 3 - 0.01j),
            8.77953e-10 - 3.97705e-11j,
            rel_tol=1e-5,
        )

    def test_beven_1(self):
        assert isclose(sc.intkambe(-10, 0.8, 1.2), 0.0161459, rel_tol=1e-5)

    def test_beven_2(self):
        assert isclose(sc.intkambe(-10, 0.4, 0.4), 5915.31, rel_tol=1e-5)

    def test_beven_3(self):
        assert isclose(sc.intkambe(-10, 3, 2), 7.203365170219268e-13)

    def test_beven_complex(self):
        assert isclose(
            sc.intkambe(-10, 1.4 + 0.01j, 3 - 0.01j),
            2.82715e-10 - 1.18144e-11j,
            rel_tol=1e-5,
        )

    def test_m4_zzero(self):
        assert isclose(sc.intkambe(-4, 0, 3), 0.01276548573157296882)

    def test_m6_negz(self):
        assert isclose(
            sc.intkambe(-6, -2 - 1j, 3), 1.20189537847431e-10 + 3.80731298663362e-12j
        )

    def test_6_negz(self):
        assert isclose(sc.intkambe(6, -1, 3), 4.959558569294379)

    def test_m5_negz(self):
        assert isclose(
            sc.intkambe(-5, -2 + 1j, 3), 3.65682124179e-10 - 6.38779831117841e-12j
        )


class TestWignerD:
    @pytest.mark.skip
    def test_error(self):
        with pytest.raises(ValueError):
            sc.wignersmalld(-1, 0, 0, 0)

    def test_warning(self):
        assert sc.wignersmalld(1, 0, 2, 0) == 0

    def test_1(self):
        assert sc.wignersmalld(1, 0, 0, 0) == 1

    def test_2(self):
        assert sc.wignersmalld(1, 1, 0, 2 * np.pi) == 0

    def test_3(self):
        assert sc.wignersmalld(2, 1, -1, np.pi) == -1

    def test_4(self):
        assert sc.wignersmalld(2, 1, 0, 2 * np.pi) == 0

    def test_5(self):
        assert isclose(sc.wignersmalld(4, 3, 2, 1), -0.07526360176530718)

    def test_6(self):
        assert isclose(sc.wignersmalld(16, 8, -4, 2), -0.06370185806824848)

    def test_7(self):
        assert isclose(sc.wignersmalld(15, -7, -4, 4), -0.3126274668164052)

    def test_8(self):
        assert isclose(sc.wignersmalld(2, -1, 1, 6), 0.05815816395893696)

    def test_9(self):
        assert isclose(
            sc.wignerd(4, 2, 3, 1.0, 2.0, 3.0),
            -0.0011756123083512 - 0.2656305961739311j,
        )

    def test_9_complex(self):
        assert isclose(
            sc.wignerd(4, 2, 3, 1.0, 2j, 3.0), -250.9892374159303 + 1.1108134417492j
        )

    def test_10(self):
        assert isclose(
            sc.wignersmalld(2, 0, -2, 7 + 1j), 0.14867417641112 + 1.1000641894940703j
        )

    def test_11(self):
        assert isclose(sc.wignersmalld(2, 0, -2, np.pi + 1e-18), 0)


class TestPiFun:
    def test_1_m1_0(self):
        assert sc.pi_fun(1, -1, 1) == -0.5

    def test_2_1_pi(self):
        assert sc.pi_fun(3, 1, -1) == -6

    def test_3_3_0(self):
        assert sc.pi_fun(3, 3, 1) == 0

    def test_4_2_complex(self):
        assert isclose(
            sc.pi_fun(4, 2, np.cos(1 + 1j)), 51.9022970181194 - 253.1942074716320j
        )

    def test_nan_l(self):
        assert np.isnan(sc.pi_fun(0.5, 0, 1j))

    def test_nan_m(self):
        assert np.isnan(sc.pi_fun(1, 0.5, 1j))


class TestTauFun:
    def test_1_m1_0(self):
        assert sc.tau_fun(1, -1, 1) == 0.5

    def test_2_1_pi(self):
        assert isclose(sc.tau_fun(3, 3, np.cos(0.2)), -1.740723332980088)

    def test_4_2_complex(self):
        assert isclose(
            sc.tau_fun(4, 2, np.cos(1 + 1j)), -568.1643037179911 - 456.9245589672897j
        )

    def test_nan_l(self):
        assert np.isnan(sc.tau_fun(0.5, 0, 1j))

    def test_nan_m(self):
        assert np.isnan(sc.tau_fun(1, 0.5, 1j))


class TestVshZ:
    def test_0(self):
        assert np.array_equal(sc.vsh_Z(0, 0, 0, 0), [1j * np.sqrt(0.25 / np.pi), 0, 0])

    def test_1(self):
        assert np.array_equal(
            sc.vsh_Z(4, 3, 2j, 1), [1j * sc.sph_harm(3, 4, 1, 2j), 0, 0]
        )


class TestVshX:
    def test_0(self):
        assert np.array_equal(sc.vsh_X(0, 0, 0, 0), [0, 0, 0])

    def test_1(self):
        expect = (
            -1j
            * np.sqrt(15 / (2 * 3 * 8 * np.pi))
            * np.exp(1j)
            * np.array([0, 1j * np.cos(2j), -np.cos(4j)])
        )
        assert np.sum(np.abs(sc.vsh_X(2, 1, 2j, 1) - expect)) < EPS


class TestVshY:
    def test_0(self):
        assert np.array_equal(sc.vsh_Y(0, 0, 0, 0), [0, 0, 0])

    def test_1(self):
        expect = (
            -1j
            * np.sqrt(15 / (2 * 3 * 8 * np.pi))
            * np.exp(1j)
            * np.array([0, np.cos(4j), 1j * np.cos(2j)])
        )
        assert np.sum(np.abs(sc.vsh_Y(2, 1, 2j, 1) - expect)) < EPS


class TestVswM:
    def test_0(self):
        expect = (
            -1j
            * (ssc.spherical_jn(2, 4 + 1j) + 1j * ssc.spherical_yn(2, 4 + 1j))
            * np.sqrt(15 / (2 * 3 * 8 * np.pi))
            * np.exp(1j)
            * np.array([0, 1j * np.cos(2), -np.cos(4)])
        )
        assert np.sum(np.abs(sc.vsw_M(2, 1, 4 + 1j, 2, 1) - expect)) < EPS


class TestVswrM:
    def test_0(self):
        assert np.array_equal(sc.vsw_rM(1, 0, 0, 0, 0), [0, 0, 0])

    def test_1(self):
        expect = (
            -1j
            * ssc.spherical_jn(2, 4 + 1j)
            * np.sqrt(15 / (2 * 3 * 8 * np.pi))
            * np.exp(1j)
            * np.array([0, 1j * np.cos(2), -np.cos(4)])
        )
        assert np.sum(np.abs(sc.vsw_rM(2, 1, 4 + 1j, 2, 1) - expect)) < EPS


class TestVswN:
    def test_0(self):
        expect = (
            -1j
            * np.sqrt(5 / (16 * np.pi))
            * np.exp(1j)
            * np.array(
                [
                    6
                    * sc.spherical_hankel1(2, 4 + 1j)
                    / (4 + 1j)
                    * np.sin(2)
                    * np.cos(2),
                    np.cos(4)
                    * (
                        sc.spherical_hankel1_d(2, 4 + 1j)
                        + sc.spherical_hankel1(2, 4 + 1j) / (4 + 1j)
                    ),
                    1j
                    * np.cos(2)
                    * (
                        sc.spherical_hankel1_d(2, 4 + 1j)
                        + sc.spherical_hankel1(2, 4 + 1j) / (4 + 1j)
                    ),
                ]
            )
        )
        assert np.sum(np.abs(sc.vsw_N(2, 1, 4 + 1j, 2, 1) - expect)) < EPS


class TestVswrN:
    def test_0(self):
        assert (
            np.sum(
                np.abs(
                    sc.vsw_rN(1, 0, 0, 1, 0)
                    - [
                        1j / np.sqrt(6 * np.pi) * np.cos(1),
                        -1j / np.sqrt(6 * np.pi) * np.sin(1),
                        0,
                    ]
                )
            )
            < EPSSQ
        )

    def test_1(self):
        expect = (
            -1j
            * np.sqrt(5 / (16 * np.pi))
            * np.exp(1j)
            * np.array(
                [
                    6 * ssc.spherical_jn(2, 4 + 1j) / (4 + 1j) * np.sin(2) * np.cos(2),
                    np.cos(4)
                    * (
                        ssc.spherical_jn(2, 4 + 1j, 1)
                        + ssc.spherical_jn(2, 4 + 1j) / (4 + 1j)
                    ),
                    1j
                    * np.cos(2)
                    * (
                        ssc.spherical_jn(2, 4 + 1j, 1)
                        + ssc.spherical_jn(2, 4 + 1j) / (4 + 1j)
                    ),
                ]
            )
        )
        assert np.sum(np.abs(sc.vsw_rN(2, 1, 4 + 1j, 2, 1) - expect)) < EPS


class TestVswA:
    def test_p(self):
        assert np.array_equal(
            sc.vsw_A(5, 4, 3, 2, 1, 1),
            (sc.vsw_N(5, 4, 3, 2, 1) + sc.vsw_M(5, 4, 3, 2, 1)) * np.sqrt(0.5),
        )

    def test_m(self):
        assert np.array_equal(
            sc.vsw_A(5, 4, 3, 2, 1, 0),
            (sc.vsw_N(5, 4, 3, 2, 1) - sc.vsw_M(5, 4, 3, 2, 1)) * np.sqrt(0.5),
        )


class TestVswrA:
    def test_p(self):
        assert np.array_equal(
            sc.vsw_rA(5, 4, 3, 2, 1, 1),
            (sc.vsw_rN(5, 4, 3, 2, 1) + sc.vsw_rM(5, 4, 3, 2, 1)) * np.sqrt(0.5),
        )

    def test_m(self):
        assert np.array_equal(
            sc.vsw_rA(5, 4, 3j, 2, 1, 0),
            (sc.vsw_rN(5, 4, 3j, 2, 1) - sc.vsw_rM(5, 4, 3j, 2, 1)) * np.sqrt(0.5),
        )


class TestTlVswA:
    def test_0(self):
        assert isclose(
            sc.tl_vsw_A(1, 0, 1, 0, 1, 0, 0), 0.903506036819270 - 4.145319872028107j
        )

    def test_1(self):
        assert isclose(
            sc.tl_vsw_A(14, 13, 11, -3, 12, 2, 1),
            -6.265680341371548e02 - 2.084168034177037e03j,
        )


class TestTlVswrA:
    def test_0(self):
        assert isclose(sc.tl_vsw_rA(1, 0, 1, 0, 1, 0, 0), 0.903506036819270)

    def test_1(self):
        assert isclose(
            sc.tl_vsw_rA(14, 13, 11, -3, 12 + 0j, 2, 1),
            6.853762595651612e-05 - 2.060462015430312e-05j,
        )


class TestTlVswB:
    def test_0(self):
        assert isclose(sc.tl_vsw_B(1, 0, 1, 0, 1, 0, 0), 0)

    def test_1(self):
        assert isclose(
            sc.tl_vsw_B(14, 13, 11, -3, 12, 2, 1),
            -2.087226903652359e02 + 6.274791216231195e01j,
        )


class TestTlVswrB:
    def test_0(self):
        assert isclose(sc.tl_vsw_rB(1, 0, 1, 0, 1, 0, 0), 0)

    def test_1(self):
        assert isclose(
            sc.tl_vsw_rB(14, 13, 11, -3, 12, 2, 1),
            -2.366181401015062e-04 - 7.870684079277587e-04j,
        )


class TestVcwM:
    def test_0(self):
        assert np.array_equal(sc.vcw_M(0, 0, 1, 0, 0), [0, -sc.hankel1_d(0, 1), 0])

    def test_1(self):
        assert (
            np.sum(
                np.abs(
                    sc.vcw_M(1, -2, 3, 4, 5)
                    - np.exp(-3j)
                    * np.array([-2j / 3 * sc.hankel1(-2, 3), -sc.hankel1_d(-2, 3), 0])
                )
            )
            < EPSSQ
        )


class TestVcwrM:
    def test_0(self):
        assert np.array_equal(sc.vcw_rM(0, 0, 1, 0, 0), [0, -sc.jv_d(0, 1), 0])

    def test_1(self):
        assert (
            np.sum(
                np.abs(
                    sc.vcw_rM(1, -2, 3, 4, 5)
                    - np.exp(-3j)
                    * np.array([-2j / 3 * sc.jv(-2, 3), -sc.jv_d(-2, 3), 0])
                )
            )
            < EPSSQ
        )

    def test_origin_m0(self):
        assert np.array_equal(sc.vcw_rM(0, 0, 0, 0, 0), [0, 0, 0])

    def test_origin_m1(self):
        assert np.array_equal(sc.vcw_rM(0, 1, 0, 0, 0), [0.5j, -0.5, 0])

    def test_origin_m4(self):
        assert np.array_equal(sc.vcw_rM(0, 4, 0, 0, 0), [0, 0, 0])


class TestVcwN:
    def test_0(self):
        assert np.array_equal(sc.vcw_N(0, 0, 1, 0, 0, 1), [0, 0, sc.hankel1(0, 1)])

    def test_1(self):
        assert (
            np.sum(
                np.abs(
                    sc.vcw_N(4, -2, 3, 1, 2, 5)
                    - np.exp(6j)
                    * np.array(
                        [
                            4j / 5 * sc.hankel1_d(-2, 3),
                            2 * 4 / (3 * 5) * sc.hankel1(-2, 3),
                            3 / 5 * sc.hankel1(-2, 3),
                        ]
                    )
                )
            )
            < EPSSQ
        )


class TestVcwrN:
    def test_0(self):
        assert np.array_equal(sc.vcw_rN(0, 0, 1, 0, 0, 1), [0, 0, sc.jv(0, 1)])

    def test_1(self):
        assert (
            np.sum(
                np.abs(
                    sc.vcw_rN(4, -2, 3, 1, 2, 5)
                    - np.exp(6j)
                    * np.array(
                        [
                            4j / 5 * sc.jv_d(-2, 3),
                            2 * 4 / (3 * 5) * sc.jv(-2, 3),
                            3 / 5 * sc.jv(-2, 3),
                        ]
                    )
                )
            )
            < EPSSQ
        )


class TestVcwA:
    def test_p(self):
        assert np.array_equal(
            sc.vcw_A(4, -2, 3, 1, 2, 5, 1),
            (sc.vcw_N(4, -2, 3, 1, 2, 5) + sc.vcw_M(4, -2, 3, 1, 2)) * np.sqrt(0.5),
        )

    def test_m(self):
        assert np.array_equal(
            sc.vcw_A(4, -2, 3, 1, 2, 5, 0),
            (sc.vcw_N(4, -2, 3, 1, 2, 5) - sc.vcw_M(4, -2, 3, 1, 2)) * np.sqrt(0.5),
        )


class TestVcwrA:
    def test_p(self):
        assert np.array_equal(
            sc.vcw_rA(4, -2, 3, 1, 2, 5, 1),
            (sc.vcw_rN(4, -2, 3, 1, 2, 5) + sc.vcw_rM(4, -2, 3, 1, 2)) * np.sqrt(0.5),
        )

    def test_m(self):
        assert np.array_equal(
            sc.vcw_rA(4, -2, 3, 1, 2, 5, 0),
            (sc.vcw_rN(4, -2, 3, 1, 2, 5) - sc.vcw_rM(4, -2, 3, 1, 2)) * np.sqrt(0.5),
        )


class TestTlVcw:
    def test_0(self):
        assert sc.tl_vcw(0, 0, 1, 0, 0, 0, 0) == 0

    def test_1(self):
        assert sc.tl_vcw(1, 2, 1, 3, 4, 5, 6) == np.exp(11j) * sc.hankel1(1, 4)


class TestTlVcwr:
    def test_0(self):
        assert sc.tl_vcw_r(0, 0, 1, 0, 0, 0, 0) == 0

    def test_1(self):
        assert sc.tl_vcw_r(1, 2, 1, 3, 4, 5, 6) == np.exp(11j) * sc.jv(1, 4)


class TestVpwM:
    def test_k0(self):
        res = sc.vpw_M(0, 0, 0j, 0, 0, 0)
        assert np.all([np.isnan(i) for i in res])

    def test_kpar0(self):
        assert np.array_equal(sc.vpw_M(0, 0, 3, 3, 2, 1), [0, -1j * np.exp(3j), 0])

    def test_k_general(self):
        k = np.array([3, 4, -3])
        r = np.array([4, 0.2, 3])
        res = sc.vpw_M(*k, *r)
        assert np.all(
            np.abs(
                res
                - [
                    1j * k[1] * np.exp(1j * k @ r) / 5,
                    -1j * k[0] * np.exp(1j * k @ r) / 5,
                    0,
                ]
            )
            < EPSSQ
        )


class TestVpwN:
    def test_k0(self):
        res = sc.vpw_N(0, 0, 0j, 0, 0, 0)
        assert np.all([np.isnan(i) for i in res])

    def test_kpar0(self):
        assert np.array_equal(sc.vpw_N(0, 0, 3, 3, 2, 1), [-np.exp(3j), 0, 0])

    def test_kpar0_imagkz(self):
        assert np.array_equal(sc.vpw_N(0, 0, -3j, 3, 2, 1), [np.exp(3), 0, 0])

    def test_k_general(self):
        k = np.array([3, 4, -3])
        r = np.array([4, 0.2, 3])
        res = sc.vpw_N(*k, *r)
        assert np.all(
            np.abs(
                res
                - [
                    -0.2441697182761140 - 0.1888789726889204j,
                    -0.3255596243681520 - 0.2518386302518939j,
                    -0.6782492174336500 - 0.5246638130247790j,
                ]
            )
            < EPSSQ
        )


class TestVpwA:
    def test(self):
        k = np.array([-3, 1, -6])
        r = np.array([2, 0.2, 0.5])
        assert np.all(
            np.abs(
                np.sqrt(2) * sc.vpw_A(*k, *r, 0) - sc.vpw_N(*k, *r) + sc.vpw_M(*k, *r)
            )
            < EPSSQ
        )

    def test_complex(self):
        k = np.array([-3, 1j, -6])
        r = np.array([2, 0.2, 0.5])
        assert np.all(
            np.abs(
                np.sqrt(2) * sc.vpw_A(*k, *r, 0) - sc.vpw_N(*k, *r) + sc.vpw_M(*k, *r)
            )
            < EPSSQ
        )


class TestCar2Cyl:
    def test_0(self):
        assert np.array_equal(sc.car2cyl([0, 0, 0]), [0, 0, 0])

    def test_1(self):
        assert np.array_equal(sc.car2cyl([3, -4, 1]), [5, -0.9272952180016122, 1])


class TestCar2Sph:
    def test_0(self):
        assert np.array_equal(sc.car2sph([0, 0, 0]), [0, 0, 0])

    def test_1(self):
        assert np.array_equal(
            sc.car2sph([3, -4, 1]), [np.sqrt(26), np.arctan2(5, 1), -0.9272952180016122]
        )


class TestCyl2Car:
    def test_0(self):
        assert np.array_equal(sc.cyl2car([0, 1, 0]), [0, 0, 0])

    def test_1(self):
        assert (
            np.sum(np.abs(sc.cyl2car([5, -0.9272952180016122, 1]) - [3, -4, 1])) < EPSSQ
        )


class TestCyl2Sph:
    def test_0(self):
        assert np.array_equal(sc.cyl2sph([0, 1, 0]), [0, 0, 1])

    def test_1(self):
        assert (
            np.sum(np.abs(sc.cyl2sph([3, -4, 1]) - [np.sqrt(10), np.arctan2(3, 1), -4]))
            < EPSSQ
        )


class TestSph2Car:
    def test_0(self):
        assert np.array_equal(sc.sph2car([0, 1, 2]), [0, 0, 0])

    def test_1(self):
        assert (
            np.sum(
                np.abs(
                    sc.sph2car([np.sqrt(26), np.arctan2(5, 1), -0.9272952180016122])
                    - [3, -4, 1]
                )
            )
            < EPSSQ
        )


class TestSph2Cyl:
    def test_0(self):
        assert np.array_equal(sc.sph2cyl([0, 0, 1]), [0, 1, 0])

    def test_1(self):
        assert (
            np.sum(np.abs(sc.sph2cyl([np.sqrt(10), np.arctan2(3, 1), -4]) - [3, -4, 1]))
            < EPSSQ
        )


class TestCar2Pol:
    def test_0(self):
        assert np.array_equal(sc.car2pol([0, 0]), [0, 0])

    def test_1(self):
        assert np.array_equal(sc.car2pol([3, -4]), [5, -0.9272952180016122])


class TestPol2Car:
    def test_0(self):
        assert np.array_equal(sc.pol2car([0, 1]), [0, 0])

    def test_1(self):
        assert np.sum(np.abs(sc.pol2car([5, -0.9272952180016122]) - [3, -4])) < EPSSQ


class TestVCarCyl:
    def test_0(self):
        assert np.array_equal(sc.vcar2cyl([0, 0, 0j], [0, 0, 0]), [0, 0, 0])

    def test_1(self):
        assert np.array_equal(sc.vcar2cyl([1, 0, 0], [0, 0, 0]), [1, 0, 0])

    def test_2(self):
        assert np.array_equal(sc.vcyl2car([0, 0, 0], [0, 0, 0]), [0, 0, 0])

    def test_3(self):
        assert np.array_equal(sc.vcyl2car([1, 0, 0], [0, 0, 0]), [1, 0, 0])

    def test_4(self):
        pcar = [3, -4, 1]
        vcar = [1, 2, 3]
        pcyl = sc.car2cyl(pcar)
        vcyl = sc.vcar2cyl(vcar, pcar)
        assert np.sum(np.abs(sc.vcyl2car(vcyl, pcyl) - vcar)) < EPSSQ


class TestVCarSph:
    def test_0(self):
        assert np.array_equal(sc.vcar2sph([0, 0, 0], [0, 0, 0]), [0, 0, 0])

    def test_1(self):
        assert np.array_equal(sc.vcar2sph([1, 0, 0], [0, 0, 0]), [0, 1, 0])

    def test_2(self):
        assert np.array_equal(sc.vsph2car([0, 0, 0], [0, 0, 0]), [0, 0, 0])

    def test_3(self):
        assert np.array_equal(sc.vsph2car([0, 1, 0], [0, 0, 0]), [1, 0, 0])

    def test_4(self):
        pcar = [3, -4, 1]
        vcar = [1, 2, 3]
        psph = sc.car2sph(pcar)
        vsph = sc.vcar2sph(vcar, pcar)
        assert np.sum(np.abs(sc.vsph2car(vsph, psph) - vcar)) < EPSSQ


class TestVCylSph:
    def test_0(self):
        assert np.array_equal(sc.vcyl2sph([0, 0, 0], [0, 0, 0]), [0, 0, 0])

    def test_1(self):
        assert np.array_equal(sc.vcyl2sph([1, 0, 0], [0, 0, 0]), [0, 1, 0])

    def test_2(self):
        assert np.array_equal(sc.vsph2cyl([0, 0, 0], [0, 0, 0]), [0, 0, 0])

    def test_3(self):
        assert np.array_equal(sc.vsph2cyl([0, 1, 0], [0, 0, 0]), [1, 0, 0])

    def test_4(self):
        pcar = [3, -4, 1]
        vcar = [1, 2, 3]
        psph = sc.cyl2sph(pcar)
        vsph = sc.vcyl2sph(vcar, pcar)
        assert np.sum(np.abs(sc.vsph2cyl(vsph, psph) - vcar)) < EPSSQ


class TestVCarPol:
    def test_0(self):
        assert np.array_equal(sc.vcar2pol([0, 0], [0, 0]), [0, 0])

    def test_1(self):
        assert np.array_equal(sc.vcar2pol([1, 0], [0, 0]), [1, 0])

    def test_2(self):
        assert np.array_equal(sc.vpol2car([0, 0], [0, 0]), [0, 0])

    def test_3(self):
        assert np.array_equal(sc.vpol2car([1, 0], [0, 0]), [1, 0])

    def test_4(self):
        pcar = [3, -4]
        vcar = [1, 2]
        pcyl = sc.car2pol(pcar)
        vcyl = sc.vcar2pol(vcar, pcar)
        assert np.sum(np.abs(sc.vpol2car(vcyl, pcyl) - vcar)) < EPSSQ


class TestCythonSpecial:
    def test_hankel1_d(self):
        assert sc.hankel1_d(0.4, 0.3j) == cs.hankel1_d(0.4, 0.3j)

    def test_hankel2_d(self):
        assert sc.hankel2_d(0.4, 0.3j) == cs.hankel2_d(0.4, 0.3j)

    def test_jv_d(self):
        assert sc.jv_d(0.4, 0.3j) == cs.jv_d(0.4, 0.3j)

    def test_spherical_hankel1(self):
        assert sc.spherical_hankel1(4, 0.3j) == cs.spherical_hankel1(4, 0.3j)

    def test_spherical_hankel2(self):
        assert sc.spherical_hankel2(4, 0.3j) == cs.spherical_hankel2(4, 0.3j)

    def test_spherical_hankel1_d(self):
        assert sc.spherical_hankel1_d(4, 0.3j) == cs.spherical_hankel1_d(4, 0.3j)

    def test_spherical_hankel2_d(self):
        assert sc.spherical_hankel2_d(4, 0.3j) == cs.spherical_hankel2_d(4, 0.3j)

    def test_yv_d(self):
        assert sc.yv_d(0.4, 0.3j) == cs.yv_d(0.4, 0.3j)

    def test_incgamma(self):
        assert sc.incgamma(1.5, 0.3j) == cs.incgamma(1.5, 0.3j)

    def test_intkambe(self):
        assert sc.intkambe(4, 0.3j, 0.2 + 1j) == cs.intkambe(4, 0.3j, 0.2 + 1j)

    def test_lpmv(self):
        assert sc.lpmv(3, 4, 0.2 + 1j) == cs.lpmv(3, 4, 0.2 + 1j)

    def test_pi_fun(self):
        assert sc.pi_fun(4, 3, 0.2 + 1j) == cs.pi_fun(4, 3, 0.2 + 1j)

    def test_sph_harm(self):
        assert sc.sph_harm(3, 4, 1, 0.2 + 1j) == cs.sph_harm(3, 4, 1, 0.2 + 1j)

    def test_tau_fun(self):
        assert sc.tau_fun(4, 3, 0.2 + 1j) == cs.tau_fun(4, 3, 0.2 + 1j)

    def test_tl_vcw(self):
        assert sc.tl_vcw(6, 5, 4, 3, 2, 1, 0) == cs.tl_vcw(6, 5, 4, 3, 2, 1, 0)

    def test_tl_vcw_r(self):
        assert sc.tl_vcw_r(6, 5, 4, 3, 2.0, 1, 0) == cs.tl_vcw_r(6, 5, 4, 3, 2.0, 1, 0)

    def test_tl_vsw_A(self):
        assert sc.tl_vsw_A(6, 5, 4, 3, 2, 1.0, 0) == cs.tl_vsw_A(6, 5, 4, 3, 2, 1.0, 0)

    def test_tl_vsw_rA(self):
        assert sc.tl_vsw_rA(6, 5, 4, 3, 2, 1.0, 0) == cs.tl_vsw_rA(
            6, 5, 4, 3, 2.0, 1.0, 0
        )

    def test_tl_vsw_B(self):
        assert sc.tl_vsw_B(6, 5, 4, 3, 2, 1.0, 0) == cs.tl_vsw_B(6, 5, 4, 3, 2, 1.0, 0)

    def test_tl_vsw_rB(self):
        assert sc.tl_vsw_rB(6, 5, 4, 3, 2, 1.0, 0) == cs.tl_vsw_rB(
            6, 5, 4, 3, 2.0, 1.0, 0
        )

    def test_wigner3j(self):
        assert sc.wigner3j(5, 4, 3, -3, 2, 1) == cs.wigner3j(5, 4, 3, -3, 2, 1)

    def test_wigner3j(self):
        assert sc.wigner3j(5, 4, 3, -3, 2, 1) == cs.wigner3j(5, 4, 3, -3, 2, 1)

    def test_wignersmalld(self):
        assert sc.wignersmalld(5, 4, 3, 2) == cs.wignersmalld(5, 4, 3, 2.0)

    def test_wignerd(self):
        assert sc.wignerd(5, 4, 3, 2, 1, 0) == cs.wignerd(5, 4, 3, 2, 1.0, 0)
