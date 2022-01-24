import numpy as np

from ptsa import pw


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestToCw:
    def test(self):
        assert isclose(pw.to_cw(1, 4, 0, 3, 2, 1, 0), np.power((2 + 3j), 4) / 169)
    def test_complex(self):
        assert isclose(pw.to_cw(1, 4, 0, 3, 2j, 1, 0), 25)
    def test_one(self):
        assert pw.to_cw(1, 0, 0, 3, 2, 1, 0) == 1
    def test_pol(self):
        assert pw.to_cw(5, 4, 1, 3, 2, 1, 0) == 0
    def test_pos(self):
        assert pw.to_cw(5, 4, 0, 3, 2, 1, 0, 1, 0) == 0
    def test_pos_complex(self):
        assert pw.to_cw(5, 4, 0, 3, 2j, 1, 0, 1, 0) == 0


class TestToSw:
    def test_h(self):
        assert isclose(pw.to_sw(5, 4, 0, 3, 2, 1, 0), -3.6625822219731057+3.632060703456664j)
    def test_h_zero(self):
        assert pw.to_sw(5, 4, 1, 3, 2, 1j, 0) == 0
    def test_h_pos(self):
        assert pw.to_sw(5, 4, 0, 3, 2, 1, 0, 1, 0) == 0
    def test_p_same(self):
        assert isclose(pw.to_sw(5, 4, 0, 3, 2, 1, 0, helicity=False), -1.3753807114589616+1.3639192055301372j)
    def test_p_opposite(self):
        assert isclose(pw.to_sw(5, 4, 1, 3, 2, 1j, 0, helicity=False), 3.087192594399995+3.1131353893109184j)
    def test_p_pos(self):
        assert pw.to_sw(5, 4, 0, 3, 2, 1j, 0, 0, 100, False) == 0


class TestTranslate:
    def test(self):
        assert pw.translate(1, 2, 3, 4, 5, 6) == np.exp(32j)


class TestPermuteXyz:
    def test_h(self):
        assert pw.permute_xyz(1, 2, 3, 0, 1, 2, 3, 0) == (-3 - 2j * np.sqrt(14)) / np.sqrt(65)
    def test_h_inv(self):
        assert pw.permute_xyz(1, 2, 1j, 0, 1, 2, 1j, 0, inverse=True) == 3j / np.sqrt(15)
    def test_h_opposite(self):
        assert pw.permute_xyz(1, 2, 3, 0, 1, 2, 3, 1) == 0
    def test_h_zero(self):
        assert pw.permute_xyz(1, 2, 3, 0, 1, 2, 2, 0) == 0
    def test_h_kxy_zero(self):
        assert pw.permute_xyz(0, 0, -1, 0, 0, 0, -1, 0) == 1
    def test_p(self):
        assert pw.permute_xyz(1, 2, 3, 0, 1, 2, 3, 0, helicity=False) == -3 / np.sqrt(65)
    def test_p_inv(self):
        assert pw.permute_xyz(1, 2, 1j, 0, 1, 2, 1j, 0, helicity=False, inverse=True) == -1j / np.sqrt(15)
    def test_p_opposite(self):
        assert pw.permute_xyz(1, 2, 3, 0, 1, 2, 3, 1, helicity=False) == 2j * np.sqrt(14) / np.sqrt(65)
    def test_p_inv_opposite(self):
        assert pw.permute_xyz(1, 2, 1j, 0, 1, 2, 1j, 1, helicity=False, inverse=True) == -4j / np.sqrt(15)
    def test_p_zero(self):
        assert pw.permute_xyz(1, 2, 3, 0, 1, 2, 2j, 0, helicity=False) == 0
    def test_p_kxy_zero(self):
        assert pw.permute_xyz(0, 0, -1, 0, 0, 0, -1, 0, helicity=False) == 1
    def test_p_kxy_zero_opposite(self):
        assert pw.permute_xyz(0, 0, -1, 0, 0, 0, -1, 1, helicity=False) == 0
