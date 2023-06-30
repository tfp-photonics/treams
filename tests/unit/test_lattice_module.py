import numpy as np

import treams.lattice as la


def isclose(a, b, rel_tol=1e-06, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


EPS = 2e-7
EPSSQ = 4e-14


class TestLSumSW1d:
    def test(self):
        assert isclose(
            la.lsumsw1d(7, 2, -0.2, 1.1, 0.5, 0),
            7933.055199668467 + 118032.50658666379j,
        )

    def test_origin(self):
        assert isclose(
            la.lsumsw1d(0, 1 + 0.1j, -0.2, 1.1, 0, 0),
            0.46956307915218665 - 0.06082398908132698j,
        )

    def test_zero(self):
        assert la.lsumsw1d(1, 2, 0, 1, 0.5, 0) == 0

    def test_beta_zero_even(self):
        assert isclose(
            la.lsumsw1d(4, 2, 0, 1.1, 0.5, 0), 0.453184295682506 - 136.37405323742743j
        )

    def test_beta_zero_odd(self):
        assert isclose(
            la.lsumsw1d(3, 2, 0, 1.1, 0.5, 0),
            9.949637075241129e-08 + 6.188903717006807j,
        )


class TestDSumSW1d:
    def test(self):
        assert isclose(
            la.dsumsw1d(7, 2, -0.2, 1.1, 0.5, 1), 7926.666849705289 - 35407.6409574394j
        )

    def test_zero(self):
        assert la.dsumsw1d(7, 2j, -0.2, 1.1, 0, 0) == 0

    def test_i0(self):
        assert isclose(la.dsumsw1d(7, 2j, -0.2, 1.1, 0.5, 0), 142089.72088031314j)

    def test_edge(self):
        assert isclose(
            la.dsumsw1d(7, 2, -0.2, 1, 0.5, 0), 30486.115441831644 + 3058.814396120404j
        )


class TestLSumCW1d:
    def test(self):
        assert isclose(
            la.lsumcw1d(-7, 2, -0.2, 1.1, 0.5, 0),
            -1905.5519170004113 - 22107.5787204855j,
        )

    def test_origin(self):
        assert isclose(
            la.lsumcw1d(0, 1 + 0.1j, -0.2, 1.1, 0, 0),
            0.9012865303927504 + 0.9760354794296119j,
        )

    def test_zero(self):
        assert la.lsumcw1d(1, 2, 0, 1, 0.5, 0) == 0

    def test_beta_zero_even(self):
        assert isclose(
            la.lsumcw1d(4, 2, 0, 1.1, 0.5, 0), 0.9090909089382382 - 51.211404621440536j
        )

    def test_beta_zero_odd(self):
        assert isclose(
            la.lsumcw1d(3, 2, 0, 1.1, 0.5, 0),
            -2.377282713998203e-10 + 2.306034974211807j,
        )


class TestDSumCW1d:
    def test(self):
        assert isclose(
            la.dsumcw1d(-7, 2, -0.2, 1.1, 0.5, 1),
            -1900.7331947651744 + 8473.777703044347j,
        )

    def test_zero(self):
        assert la.dsumcw1d(-7, 2j, -0.2, 1.1, 0, 0) == 0

    def test_i0(self):
        assert isclose(la.dsumcw1d(-7, 2j, -0.2, 1.1, 0.5, 0), 28143.06322075269)

    def test_edge(self):
        assert isclose(
            la.dsumcw1d(-7, 2, -0.2, 1, 0.5, 0), -6077.087627234397 - 609.742594614585j
        )


class TestLSumSW2d:
    def test(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumsw2d(7, 3, 2, [0.1, 0.2], a, [0.2, 0], 0),
            51.04558205 - 68859783.71895242j,
        )

    def test_origin(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumsw2d(0, 0, 1 + 0.1j, [0.1, 0.2], a, [0, 0], 0),
            1.2763907 + 0.65148623j,
        )

    def test_zero(self):
        a = 1.1 * np.eye(2)
        assert la.lsumsw2d(7, 4, 2, [0.1, 0.2], a, [0, 0], 0) == 0

    def test_zero_2(self):
        a = 1.1 * np.eye(2)
        assert la.lsumsw2d(7, 3, 2, [0, 0], a, [0, 0], 0) == 0

    def test_beta_zero(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumsw2d(6, 0, 2, [0, 0], a, [0.2, 0], 0), -1.33422509 + 2031448.8128476j
        )

    def test_beta_zero_odd(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumsw2d(7, 3, 2, [0, 0], a, [0.2, 0], 0),
            0.00296384 - 68859750.95204805j,
        )


class TestDSumSW2d:
    def test(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.dsumsw2d(7, 3, 2, [0.1, 0.2], a, [0.2, 0], 1),
            50.744595676527425 + 469.33788946355355j,
        )

    def test_zero(self):
        a = 1.1 * np.eye(2)
        assert la.dsumsw2d(7, 3, 2j, [0.1, 0.2], a, [0, 0], 0) == 0


class TestLSumCW2d:
    def test(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumcw2d(-7, 2, [0.1, 0.2], a, [0.2, 0], 0),
            72.4265408 - 18023963.86399937j,
        )

    def test_origin(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumcw2d(7, 1 + 0.1j, [0.1, 0.2], a, [0, 0], 0),
            1856.54740513 + 8265.31298836j,
        )

    def test_zero(self):
        a = 1.1 * np.eye(2)
        assert la.lsumcw2d(7, 2, [0, 0], a, [0, 0], 0) == 0

    def test_beta_zero(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumcw2d(0, 2, [0, 0], a, [0.2, 0], 0),
            3.4057753716089e-7 + 0.4974331386852614j,
        )

    def test_beta_zero_odd(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumcw2d(7, 2, [0, 0], a, [0.2, 0], 0),
            0.00013173070560507455 + 18023975.612865075j,
        )


class TestDSumCW2d:
    def test(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.dsumcw2d(-7, 2, [0.1, 0.2], a, [0.2, 0], 1),
            71.68432125868046 + 811.2490404245293j,
        )

    def test_zero(self):
        a = 1.1 * np.eye(2)
        assert la.dsumcw2d(-7, 2j, [0.1, 0.2], a, [0, 0], 0) == 0


class TestLSumSW3d:
    def test(self):
        a = 1.1 * np.eye(3)
        assert isclose(
            la.lsumsw3d(7, 3, 2, [0.1, 0.2, -0.3], a, [0.2, -0.3, 0.4], 0),
            -5410.987286636962 + 26890.112935162313j,
        )

    def test_origin(self):
        a = 1.1 * np.eye(3)
        assert isclose(
            la.lsumsw3d(0, 0, 1 + 0.1j, [0.1, 0.2, -0.3], a, [0, 0, 0], 0),
            0.7797387103128192 + 3.546723626318176j,
        )

    def test_zero(self):
        a = 1.1 * np.eye(3)
        assert la.lsumsw3d(1, 0, 2, [0, 0, 0], a, [0, 0, 0], 0) == 0

    def test_beta_zero(self):
        a = 1.1 * np.eye(3)
        assert isclose(
            la.lsumsw3d(6, 3, 2, [0, 0, 0], a, [0.2, -0.3, 0.4], 0),
            475.3946724930312 - 2658.101604143323j,
        )


class TestDSumSW3d:
    def test(self):
        a = 1.1 * np.eye(3)
        assert isclose(
            la.dsumsw3d(7, 3, 2, [0.1, 0.2, -0.3], a, [0.2, -0.3, 0.4], 1),
            -376.09928123585945 + 1161.1103743114297j,
        )

    def test_zero(self):
        a = 1.1 * np.eye(3)
        assert la.dsumsw3d(7, 3, 2j, [0.1, 0.2, -0.3], a, [0, 0, 0], 0) == 0


class TestLSumSW2dShift:
    def test(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumsw2d_shift(7, 4, 2, [0.1, 0.2], a, [0.2, 0.2, 0.1], 0),
            -46.2822843 + 2452401.72386332j,
        )

    def test2(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumsw2d_shift(6, 2, 1 + 0.1j, [0.1, 0.2], a, [0.2, 0.2, 0.1], 0),
            -6065646.19932015 + 5077087.337158j,
        )

    def test_singular(self):
        a = np.eye(2)
        assert isclose(
            la.lsumsw2d_shift(6, 0, 2 * np.pi, [0, 0], a, [0.2, 0.2, 0.1], 0),
            110.69823590056764 - 138.74329901305353j,
        )

    def test_z_zero(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumsw2d_shift(0, 0, 2, [0.1, 0.2], a, [0.2, 0, 0], 0),
            0.37395298646674273 - 0.2501854057015295j,
        )

    def test_beta_zero(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.lsumsw2d_shift(6, 0, 2, [0, 0], a, [0.2, 0.2, 0.1], 0),
            -1.30733661 - 74158.23878748j,
        )


class TestDSumSW2dShift:
    def test(self):
        a = 1.1 * np.eye(2)
        assert isclose(
            la.dsumsw2d_shift(7, 4, 2, [0.1, 0.2], a, [0.2, 0.2, 0.1], 1),
            -46.14178891555638 - 327.2420554824938j,
        )

    def test_zero(self):
        a = 1.1 * np.eye(2)
        assert la.dsumsw2d_shift(7, 3, 2j, [0.1, 0.2], a, [0, 0, 0], 0) == 0


class TestLSumSW1dShift:
    def test(self):
        assert isclose(
            la.lsumsw1d_shift(7, 4, 2, 0.1, 1, [0.2, 0.2, 0.1], 0),
            5.410572060609522 + 2452771.8911033254j,
        )

    def test2(self):
        assert isclose(
            la.lsumsw1d_shift(6, 2, 1 + 0.1j, 0.1, 1.1, [0.2, 0.2, 0.1], 0),
            -6061553.746083904 + 5072912.632902036j,
        )

    def test_singular(self):
        assert isclose(
            la.lsumsw1d_shift(6, 0, 2 * np.pi, 0, 1, [0.2, 0.2, 0.1], 0),
            -0.3552853450806117 - 25.183882160999745j,
        )

    def test_xy_zero(self):
        assert isclose(
            la.lsumsw1d_shift(0, 0, 2, 0.1, 1.1, [0, 0, 0.2], 0),
            0.4025419802107214 - 0.5283600560367392j,
        )

    def test_xy_zero_2(self):
        assert la.lsumsw1d_shift(3, 1, 2, 0.1, 1.1, [0, 0, 0.2], 0) == 0

    def test_beta_zero(self):
        assert isclose(
            la.lsumsw1d_shift(6, 0, 2, 0, 1.1, [0.2, 0.2, 0.1], 0),
            0.41829203130234643 - 74299.44379903575j,
        )


class TestDSumSW1dShift:
    def test(self):
        assert isclose(
            la.dsumsw1d_shift(7, 4, 2, 0.1, 1, [0.2, 0.2, 0.1], 1),
            5.405752406639203 + 42.20701547695025j,
        )

    def test_zero(self):
        assert la.dsumsw1d_shift(7, 4, 2j, 0.1, 1, [0, 0, 0], 0) == 0

    def test_dispatch(self):
        assert isclose(
            la.dsumsw1d_shift(7, 0, 2j, -0.2, 1.1, [0.5, 0, 0], 0), 142089.72088031314j
        )

    def test_i0(self):
        assert isclose(
            la.dsumsw1d_shift(7, 3, 2, -0.2, 1, [0.1, 0.2, 0.5], 0),
            -4348.28631675554 + 23915.57474036112j,
        )


class TestLSumCW1dShift:
    def test(self):
        assert isclose(
            la.lsumcw1d_shift(-7, 2, 0.1, 1, [0.2, 0.1], 0.5),
            857009.7602939741 + 8224316.447124034j,
        )

    def test2(self):
        assert isclose(
            la.lsumcw1d_shift(6, 1 + 0.1j, 0.1, 1.1, [0.2, 0.1], 0),
            15553417.521558769 + 10955244.470110876j,
        )

    def test_singular(self):
        assert isclose(
            la.lsumcw1d_shift(6, 2 * np.pi, 0, 1, [0.2, 0.1], 0),
            -224.73515198832922 + 675.8601437840592j,
            1e-5,
        )

    def test_xy_zero(self):
        assert isclose(
            la.lsumcw1d_shift(0, 2, 0.1, 1.1, [0.2, 0], 0),
            0.9243271013171527 + 0.017729454351493944j,
        )

    def test_beta_zero(self):
        assert isclose(
            la.lsumcw1d_shift(6, 2, 0, 1.1, [0.2, 0.1], 0),
            108600.73717056998 + 288821.87794970983j,
        )


class TestDSumCW1dShift:
    def test(self):
        assert isclose(
            la.dsumcw1d_shift(-7, 2, 0.1, 1, [0.2, 0.1], 1),
            -841.490999741377 + 757.8694926475844j,
        )

    def test_zero(self):
        assert la.dsumcw1d_shift(-7, 2j, 0.1, 1, [0, 0], 0) == 0

    def test_i0(self):
        assert isclose(la.dsumcw1d_shift(-7, 2, -0.2, 1, [0, 0.5], 0), 30588.957052124)


class TestArea:
    def test(self):
        assert la.area([[1, 2], [3, 4.0]]) == -2


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
            la.diffr_orders_circle([[1, 0], [0.5, np.sqrt(0.75)]], 1.1),
            [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [1, -1], [-1, 1],],
        )

    def test_neg(self):
        assert la.diffr_orders_circle(np.eye(2), -1).size == 0


class TestCube:
    def test(self):
        assert np.all(
            la.cube(2, 1)
            == [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]
        )


class TestCubeEdge:
    def test(self):
        assert np.all(
            la.cubeedge(2, 1)
            == [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        )
