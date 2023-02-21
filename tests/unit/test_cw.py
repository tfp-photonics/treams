import numpy as np
import scipy.special as sc

from treams import cw


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestPeriodicToPw:
    def test(self):
        assert isclose(
            cw.periodic_to_pw(6, -5, 4, 1, 4, 3, 1, 2),
            2 * np.power((-5 - 6j) / np.sqrt(61), 3) / 10,
        )

    def test_ky_zero(self):
        assert isclose(cw.periodic_to_pw(6, 0, 4, 1, 4, 3, 1, 2), 5e19 + 5e19j)

    def test_pol(self):
        assert cw.periodic_to_pw(6, 5j, 4, 1, 4, 3, 0, 2) == 0


class TestRotate:
    def test(self):
        assert isclose(cw.rotate(3, 2, 1, 3, 2, 1, 4), np.exp(8j))

    def test_zero(self):
        assert cw.rotate(3, 2, 1, 2, 2, 1, 4) == 0


class TestToSw:
    def test_h(self):
        assert isclose(cw.to_sw(4, 3, 1, 2, 3, 1, 5), -2.481668635232552j)

    def test_h_zero(self):
        assert cw.to_sw(4, 3, 1, 2, 2, 1, 5) == 0

    def test_h_pol(self):
        assert cw.to_sw(4, 3, 1, 2, 3, 0, 5) == 0

    def test_p_same(self):
        assert isclose(
            cw.to_sw(4, 3, 1, 2, 3, 1, 5, poltype="parity"), 1.063572272242522j
        )

    def test_p_same_zero(self):
        assert cw.to_sw(4, 3, 1, 2, 2, 1, 5, poltype="parity") == 0

    def test_p_opposite(self):
        assert isclose(
            cw.to_sw(4, 3, 1, 2, 3, 0, 5, poltype="parity"), -3.545240907475074j
        )

    def test_p_opposite_zero(self):
        assert cw.to_sw(4, 3, 1, 2, 2, 0, 5, poltype="parity") == 0


class TestTranslate:
    def test_s(self):
        assert isclose(
            cw.translate(3, 2, 1, 3, -2, 1, 4, 5, 6), sc.hankel1(-4, 4) * np.exp(-2j)
        )

    def test_s_opposite(self):
        assert cw.translate(3, 2, 0, 0, 2, 1, 4 + 1j, 5, 6) == 0

    def test_s_zero(self):
        assert cw.translate(3, 2, 0, 0, 2, 1, 0, 5, 0) == 0

    def test_r(self):
        assert isclose(
            cw.translate(3, 2, 1, 3, -2, 1, 4, 5, 6, singular=False),
            sc.jv(-4, 4) * np.exp(-2j),
        )

    def test_r_opposite(self):
        assert cw.translate(3, 2, 0, 0, 2, 1, 4 + 1j, 5, 6, singular=False) == 0


class TestTranslatePeriodic:
    def test(self):
        assert isclose(
            cw.translate_periodic(1, 0, 2, [0, 0, 0], ([0], [0], [0]))[0, 0],
            0.76102358j,
        )

    def test_ks(self):
        assert isclose(
            cw.translate_periodic([1, 2], 0, 2, [0, 0, 0], ([1], [3], [1]))[0, 0],
            -0.42264973081037416 + 0.30599871466756257j,
        )

    def test_ks_same(self):
        assert isclose(
            cw.translate_periodic(
                [1, 1], [0, 1], [[2, 0], [0, 2]], [0, 0, 0], ([0], [0], [0])
            )[0, 0],
            -1.0 + 0.28364898j,
        )
