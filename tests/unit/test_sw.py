import pytest

from ptsa import sw


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestPeriodicToCw:
    def test_h(self):
        assert sw.periodic_to_cw(1, 2, 0, 3, 2, 0, 3, 2) == -1.1890388773999045e-17+0.19418478507072567j
    def test_h_zero(self):
        assert sw.periodic_to_cw(1, 2, 0, 3, -2, 0, 3, 2) == 0
    def test_h_pos(self):
        assert sw.periodic_to_cw(1, 2, 0, 3, 2, 0, 3, 2, 0, 1) == 0
    def test_p_same(self):
        assert isclose(sw.periodic_to_cw(1, 1, 0, 3, 1, 0, 3, 2, helicity=False), -0.15197363286501506)
    def test_p_opposite(self):
        assert isclose(sw.periodic_to_cw(1, 1, 1, 3, 1, 0, 3, 2, helicity=False), -0.02171051898071644)
    def test_p_zero(self):
        assert sw.periodic_to_cw(1, 2, 0, 3, -2, 0, 3, 2, helicity=False) == 0

class TestPeriodicToPw:
    def test_h(self):
        assert isclose(sw.periodic_to_pw(1, 2, 3 + 4j, 0, 5, -4, 0, 2), -0.021612980340836 + 0.015904360513006j)
    def test_h_zero(self):
        assert sw.periodic_to_pw(1, 2, 3 + 4j, 0, 5, -4, 1, 2) == 0
    def test_h_kz_zero(self):
        x = sw.periodic_to_pw(1, 2, 0, 0, 5, -4, 0, 2)
        assert abs(x) > 1e16 and isclose(x.real / x.imag, -1.823529411764707)
    def test_h_kz_neg(self):
        assert isclose(sw.periodic_to_pw(1, 2, -3, 0, 5, -4, 0, 2), -0.015256789234245 - 0.004449896859988j)
    def test_h_pos(self):
        assert sw.periodic_to_pw(1, 2, -3, 0, 5, -4, 0, 2, 1, 0) == 0
    def test_p_same(self):
        assert isclose(sw.periodic_to_pw(1, 2, 3, 1, 5, -4, 1, 2, helicity=False), 0.034026205422735 + 0.009924309914964j)
    def test_p_opposite(self):
        assert isclose(sw.periodic_to_pw(1, 2, 3, 1, 5, -4, 0, 2, helicity=False), -0.049282994656980 - 0.014374206774952j)
    def test_p_kz_zero(self):
        x = sw.periodic_to_pw(1, 2, 0, 1, 5, -4, 1, 2, helicity=False)
        assert abs(x) > 1e16 and isclose(x.real / x.imag, -1.823529411764707)
    def test_p_kz_neg(self):
        assert isclose(sw.periodic_to_pw(1, 2, -3j, 1, 5, -4, 1, 2, helicity=False), -0.562764396508645 + 1.929477930886781j)
    def test_p_pos(self):
        assert sw.periodic_to_pw(1, 2, -3j, 1, 5, -4, 1, 2, 0, 1, helicity=False) == 0


class TestRotate:
    def test(self):
        assert isclose(sw.rotate(6, 5, 0, 6, 4, 0, 3, 2, 1), 0.049742012840172 - 0.007540365393637j)
    def test_zero(self):
        assert sw.rotate(8, 7, 1, 6, 5, 0, 4, 3, 2) == 0


class TestTranslate:
    def test_sh_real(self):
        assert isclose(sw.translate(5, 4, 1, 3, 2, 1, 8, 7, 6), 0.085624539859945 - 0.149999063169118j)
    def test_rh_real(self):
        assert isclose(sw.translate(5, 4, 0, 3, 2, 0, 8, 7, 6, singular=False), -0.117145153538869 + 0.042172438021060j)
    def test_sh_zero(self):
        assert sw.translate(5, 4, 1, 3, 2, 0, 8, 7, 6) == 0
    def test_rh_zero(self):
        assert sw.translate(5, 4, 1, 3, 2, 0, 8, 7, 6, singular=False) == 0
    def test_sh_kr_zero(self):
        assert isclose(sw.translate(5, 4, 1, 3, 2, 1, 1e-30j, 7, 6), 0)
    def test_sp_real_a(self):
        assert isclose(sw.translate(5, 4, 1, 3, 2, 1, 8, 7, 6, False), -0.024574996806028 - 0.103410185698746j)
    def test_sp_real_b(self):
        assert isclose(sw.translate(5, 4, 1, 3, 2, 0, 8, 7, 6, False), 0.110199536665973 - 0.046588877470371j)
    def test_rp_real_a(self):
        assert isclose(sw.translate(5, 4, 0, 3, 2, 0, 8, 7, 6, False, False), -0.064322610568195 - 0.040900170567219j)
    def test_rp_real_b(self):
        assert isclose(sw.translate(5, 4, 1, 3, 2, 0, 8, 7, 6, False, False), 0.052822542970675 - 0.083072608588279j)
    def test_sp_kr_zero(self):
        assert isclose(sw.translate(5, 4, 1, 3, 2, 1, 1e-30j, 7, 6, False), 0)


class TestTranslatePeriodic:
    def test_0(self):
        assert sw.translate_periodic(1, 0, 1, [0, 0, 0], ([1], [0], [1]), in_=([1], [0], [0])) == 0
    def test_1(self):
        assert isclose(sw.translate_periodic([1, 2], [0, 0], [[1, 0], [0, 1]], [0, 0, 0], ([2], [-1], [0]))[0, 0], 14.70796326794897 - 258.1708025505043j)
    def test_2(self):
        assert isclose(sw.translate_periodic(1, [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0], ([3], [3], [0]), helicity=False), -1 + 668.66039240j)
    def test_3(self):
        assert sw.translate_periodic([1, 1], [0, 0], [[1, 0], [0, 1]], [0, 0, 0], ([1], [0], [0]), in_=([2], [0], [1]), helicity=False) == 0
