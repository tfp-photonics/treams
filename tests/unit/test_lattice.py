import numpy as np

import ptsa.lattice as la


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

EPS = 2e-7
EPSSQ = 4e-14


class TestLSumSW1d:
    def test(self):
        assert isclose(la.lsumsw1d(7, 2, -.2, 1.1, .5, 0), 7933.055204332383 + 118032.50658349611j)
    def test_origin(self):
        assert isclose(la.lsumsw1d(0, 1 + .1j, -.2, 1.1, 0, 0), float('nan'))
    def test_zero(self):
        assert la.lsumsw1d(1, 2, 0, 1, .5, 0) == 0
    def test_beta_zero_even(self):
        assert isclose(la.lsumsw1d(4, 2, 0, 1.1, .5, 0), float('nan'))
    def test_beta_zero_odd(self):
        assert isclose(la.lsumsw1d(3, 2, 0, 1.1, .5, 0), float('nan'))


class TestLSumCW1d:
    def test(self):
        assert isclose(la.lsumcw1d(-7, 2, -.2, 1.1, .5, 0), float('nan'))
    def test_origin(self):
        assert isclose(la.lsumcw1d(0, 1 + .1j, -.2, 1.1, 0, 0), float('nan'))
    def test_zero(self):
        assert la.lsumcw1d(1, 2, 0, 1, .5, 0) == 0
    def test_beta_zero_even(self):
        assert isclose(la.lsumcw1d(4, 2, 0, 1.1, .5, 0), float('nan'))
    def test_beta_zero_odd(self):
        assert isclose(la.lsumcw1d(3, 2, 0, 1.1, .5, 0), float('nan'))


class TestLSumSW2d:
    def test(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumsw2d(7, 3, 2, [.1, .2], a, [.2, 0], 0), float('nan'))
    def test_origin(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumsw2d(0, 0, 1 + .1j, [.1, .2], a, [0, 0], 0), float('nan'))
    def test_zero(self):
        a = 1.1 * np.eye(2)
        assert la.lsumsw2d(7, 4, 2, [.1, .2], a, [0, 0], 0) == 0
    def test_zero_2(self):
        a = 1.1 * np.eye(2)
        assert la.lsumsw2d(7, 3, 2, [0, 0], a, [0, 0], 0) == 0
    def test_beta_zero(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumsw2d(6, 0, 2, [0, 0], a, [.2, 0], 0), float('nan'))
    def test_beta_zero_odd(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumsw2d(7, 3, 2, [0, 0], a, [.2, 0], 0), float('nan'))


class TestLSumCW2d:
    def test(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumcw2d(-7, 2, [.1, .2], a, [.2, 0], 0), float('nan'))
    def test_origin(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumcw2d(7, 1 + .1j, [.1, .2], a, [0, 0], 0), float('nan'))
    def test_zero(self):
        a = 1.1 * np.eye(2)
        assert la.lsumcw2d(7, 2, [0, 0], a, [0, 0], 0) == 0
    def test_beta_zero(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumcw2d(0, 2, [0, 0], a, [.2, 0], 0), float('nan'))
    def test_beta_zero_odd(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumcw2d(7, 2, [0, 0], a, [.2, 0], 0), float('nan'))


class TestLSumSW3d:
    def test(self):
        a = 1.1 * np.eye(3)
        assert isclose(la.lsumsw3d(7, 3, 2, [.1, .2, -.3], a, [.2, -.3, .4], 0), float('nan'))
    def test_origin(self):
        a = 1.1 * np.eye(3)
        assert isclose(la.lsumsw3d(0, 0, 1 + .1j, [.1, .2, -.3], a, [0, 0, 0], 0), float('nan'))
    def test_zero(self):
        a = 1.1 * np.eye(3)
        assert la.lsumsw3d(1, 0, 2, [0, 0, 0], a, [0, 0, 0], 0) == 0
    def test_beta_zero(self):
        a = 1.1 * np.eye(3)
        assert isclose(la.lsumsw3d(6, 3, 2, [0, 0, 0], a, [.2, -.3, .4], 0), float('nan'))


class TestLSumSW2dShift:
    def test(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumsw2d_shift(7, 4, 2, [.1, .2], a, [.2, .2, .1], 0), float('nan'))
    def test2(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumsw2d_shift(6, 2, 1 + .1j, [.1, .2], a, [.2, .2, .1], 0), float('nan'))
    def test_singular(self):
        a = np.eye(2)
        assert isclose(la.lsumsw2d_shift(6, 0, 2 * np.pi, [0, 0], a, [.2, .2, .1], 0), float('nan'))
    def test_z_zero(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumsw2d_shift(0, 0, 2, [.1, .2], a, [.2, 0, 0], 0), float('nan'))
    def test_beta_zero(self):
        a = 1.1 * np.eye(2)
        assert isclose(la.lsumsw2d_shift(6, 0, 2, [0, 0], a, [.2, .2, .1], 0), float('nan'))


class TestLSumSW1dShift:
    def test(self):
        assert isclose(la.lsumsw1d_shift(7, 4, 2, .1, 1, [.2, .2, .1], 0), float('nan'))
    def test2(self):
        assert isclose(la.lsumsw1d_shift(6, 2, 1 + .1j, .1, 1.1, [.2, .2, .1], 0), float('nan'))
    def test_singular(self):
        assert isclose(la.lsumsw1d_shift(6, 0, 2 * np.pi, 0, 1, [.2, .2, .1], 0), float('nan'))
    def test_xy_zero(self):
        assert isclose(la.lsumsw1d_shift(0, 0, 2, .1, 1.1, [0, 0, .2], 0), float('nan'))
    def test_xy_zero_2(self):
        assert la.lsumsw1d_shift(3, 1, 2, .1, 1.1, [0, 0, .2], 0) == 0
    def test_beta_zero(self):
        assert isclose(la.lsumsw1d_shift(6, 0, 2, 0, 1.1, [.2, .2, .1], 0), float('nan'))


class TestLSumCW1dShift:
    def test(self):
        assert isclose(la.lsumcw1d_shift(-7, 2, .1, 1, [.2, .1], 0), float('nan'))
    def test2(self):
        assert isclose(la.lsumcw1d_shift(6, 1 + .1j, .1, 1.1, [.2, .1], 0), float('nan'))
    def test_singular(self):
        assert isclose(la.lsumcw1d_shift(6, 2 * np.pi, 0, 1, [.2, .1], 0), float('nan'))
    def test_xy_zero(self):
        assert isclose(la.lsumcw1d_shift(0, 2, .1, 1.1, [.2, 0], 0), float('nan'))
    # def test_xy_zero_2(self):
    #     assert la.lsumcw1d_shift(3, 1, 2, .1, 1.1, [0, 0, .2], 0) == 0
    def test_beta_zero(self):
        assert isclose(la.lsumcw1d_shift(6, 2, 0, 1.1, [.2, .1], 0), float('nan'))


class TestArea:
    def test(self):
        assert la.area([[1, 2], [3, 4.]]) == -2


class TestVolume:
    def test_2d(self):
        assert la.volume([[1, 2], [3, 4]]) == -2
    def test_3d(self):
        assert la.volume([[1, 2, 3], [4, 5, 6], [7, 8, 10]]) == -3
    def test_3d_2(self):
        assert la.volume(np.eye(3)) == 1


class TestReciprocal:
    def test_2d(self):
        res = la.reciprocal(3 * np.eye(2)).flatten()
        expect = (np.eye(2) * 2 * np.pi / 3).flatten()
        assert np.all(np.abs(res - expect) < EPSSQ)
    def test_3d(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        res = la.reciprocal(a)
        expect = 2 * np.pi / (-3) * np.array([[2, 2, -3], [4, -11, 6], [-3, 6, -3]])
        assert np.all(np.abs(res - expect) < EPSSQ)


class TestDiffrOrdersCircle:
    def test_0(self):
        assert np.array_equal(la.diffr_orders_circle(np.eye(2), 0), [[0, 0]])
    def test_square(self):
        assert np.array_equal(
            la.diffr_orders_circle(np.eye(2), 1.5),
            [
                [0, 0],
                [0, 1],
                [0, -1],
                [1, 0],
                [-1, 0],
                [1, 1],
                [-1, -1],
                [1, -1],
                [-1, 1],
            ],
        )
    def test_hex(self):
        assert np.array_equal(
            la.diffr_orders_circle([[1, 0], [.5, np.sqrt(0.75)]], 1.1),
            [
                [0, 0],
                [0, 1],
                [0, -1],
                [1, 0],
                [-1, 0],
                [1, -1],
                [-1, 1],
            ],
        )
