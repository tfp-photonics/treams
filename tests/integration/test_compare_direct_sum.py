import numpy as np
import pytest

import treams.lattice as la


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def osscilation_avg(res):
    avg_len = len(res) // 2
    return np.mean(np.cumsum(res)[-avg_len:])


EPS = 2e-7
EPSSQ = 4e-14


class TestLSumSW1d:
    @pytest.mark.parametrize("l", [0, 1, 2, 7, 8])
    @pytest.mark.parametrize("k", [2, 1.1 + 0.1j])
    @pytest.mark.parametrize("kpar", [-0.2, 0, 0.7])
    @pytest.mark.parametrize("a", [1, 1.1])
    @pytest.mark.parametrize("r", [-0.2, 0, 0.5])
    def test_regular(self, l, k, kpar, a, r):  # noqa: E741
        assert isclose(
            la.lsumsw1d(l, k, kpar, a, r, 0),
            np.sum(la.dsumsw1d(l, k, kpar, a, r, np.arange(100_000))),
            rel_tol=0.02,
            abs_tol=EPSSQ,
        )

    @pytest.mark.parametrize("l", [0, 1, 2, 7, 8])
    @pytest.mark.parametrize("r", [-0.2, 0, 0.5])
    @pytest.mark.slow
    def test_singular(self, l, r):  # noqa: E741
        a = 2 * np.pi
        k = 1
        kpar = 0
        assert isclose(
            la.lsumsw1d(l, k, kpar, a, r, 0),
            np.sum(la.dsumsw1d(l, k, kpar, a, r, np.arange(1_100_000))),
            rel_tol=0.05,
            abs_tol=EPSSQ,
        )


class TestLSumCW1d:
    @pytest.mark.parametrize("l", [-2, -1, 0, 7, 8])
    @pytest.mark.parametrize("k", [2, 1.1 + 0.1j, 2j])
    @pytest.mark.parametrize("kpar", [-0.2, 0, 0.7])
    @pytest.mark.parametrize("a", [1, 1.1])
    @pytest.mark.parametrize("r", [-0.2, 0, 0.5])
    def test_regular(self, l, k, kpar, a, r):  # noqa: E741
        assert isclose(
            la.lsumcw1d(l, k, kpar, a, r, 0),
            np.sum(la.dsumcw1d(l, k, kpar, a, r, np.arange(100_000))),
            rel_tol=0.02,
            abs_tol=EPSSQ,
        )

    @pytest.mark.parametrize("l", [-2, -1, 0, 7, 8])
    @pytest.mark.parametrize("r", [-0.2, 0, 0.5])
    def test_singular(self, l, r):  # noqa: E741
        a = 2 * np.pi
        k = 1
        kpar = 0
        assert isclose(
            la.lsumcw1d(l, k, kpar, a, r, 0),
            np.sum(la.dsumcw1d(l, k, kpar, a, r, np.arange(1_600_000))),
            rel_tol=0.05,
            abs_tol=EPSSQ,
        )


class TestLSumCW2d:
    @pytest.mark.parametrize("l", [-2, -1, 0, 7, 8])
    @pytest.mark.parametrize("k", [2, 1.1 + 0.1j])
    @pytest.mark.parametrize("kpar", [np.zeros(2), np.array([0.1, 0.2])])
    @pytest.mark.parametrize(
        "a", [np.array([[1, 0], [0, 1.2]]), np.array([[1.1, 0.2], [-0.1, 1]])]
    )
    @pytest.mark.parametrize("r", [np.zeros(2), np.array([0.2, 0])])
    @pytest.mark.slow
    def test_regular(self, l, k, kpar, a, r):  # noqa: E741
        assert isclose(
            la.lsumcw2d(l, k, kpar, a, r, 0),
            osscilation_avg(la.dsumcw2d(l, k, kpar, a, r, np.arange(800))),
            rel_tol=1e-1,
            abs_tol=EPS,
        )

    @pytest.mark.parametrize("l", [-2, -1, 0, 7, 8])
    @pytest.mark.parametrize("r", [np.zeros(2), np.array([0.2, 0])])
    @pytest.mark.slow
    def test_singular(self, l, r):  # noqa: E741
        k = 1
        kpar = np.zeros(2)
        a = np.array([[2 * np.pi, 0], [0, 1]])
        # Here the direct result is too oscillatory
        if l == -2:  # noqa: E741
            if np.all(r == 0):
                assert isclose(
                    np.imag(la.lsumcw2d(l, k, kpar, a, r, 0)), 2.9105962449216736
                )
        elif l == 0:  # noqa: E741
            if np.all(r == 0):
                assert isclose(
                    np.imag(la.lsumcw2d(l, k, kpar, a, r, 0)), 0.9156690769853802
                )
            else:
                assert isclose(
                    np.imag(la.lsumcw2d(l, k, kpar, a, r, 0)), -0.15877332685629297
                )
        else:
            assert isclose(
                np.imag(la.lsumcw2d(l, k, kpar, a, r, 0)),
                np.imag(np.sum(la.dsumcw2d(l, k, kpar, a, r, np.arange(1000)))),
                rel_tol=1e-1,
                abs_tol=EPS,
            )


class TestLSumSW2d:
    @pytest.mark.parametrize(
        "lm",
        [
            (0, 0),
            (1, -1),
            (1, 0),
            (2, -2),
            (2, 0),
            (2, 1),
            (7, -7),
            (7, 0),
            (7, 3),
            (8, -8),
            (8, 0),
            (8, 4),
        ],
    )
    @pytest.mark.parametrize("k", [2, 1.1 + 0.1j])
    @pytest.mark.parametrize("kpar", [np.zeros(2), np.array([0.1, 0.2])])
    @pytest.mark.parametrize(
        "a", [np.array([[1, 0], [0, 1.2]]), np.array([[1.1, 0.2], [-0.1, 1]])]
    )
    @pytest.mark.parametrize("r", [np.zeros(2), np.array([0.2, 0])])
    @pytest.mark.slow
    def test_regular(self, lm, k, kpar, a, r):
        assert isclose(
            la.lsumsw2d(*lm, k, kpar, a, r, 0),
            osscilation_avg(la.dsumsw2d(*lm, k, kpar, a, r, np.arange(800))),
            rel_tol=1e-1,
            abs_tol=EPS,
        )

    @pytest.mark.parametrize("lm", [(0, 0), (2, -2), (2, 0), (8, -8), (8, 0), (8, 4)])
    @pytest.mark.slow
    def test_singular_r0(self, lm):
        k = 1
        kpar = np.zeros(2)
        a = np.array([[2 * np.pi, 0], [0, 1]])
        r = np.zeros(2)
        dirres = np.cumsum(la.dsumsw2d(*lm, k, kpar, a, r, np.arange(800)))[[11, -1]]
        assert isclose(
            np.angle(la.lsumsw2d(*lm, k, kpar, a, r, 0) - dirres[0]),
            np.angle(dirres[1] - dirres[0]),
            rel_tol=1e-1,
            abs_tol=EPS,
        )

    @pytest.mark.parametrize("lm", [(0, 0), (2, -2), (2, 0), (7, -7), (7, 3), (8, 0)])
    @pytest.mark.slow
    def test_singular_r1(self, lm):
        k = 1
        kpar = np.zeros(2)
        a = np.array([[2 * np.pi, 0], [0, 1]])
        r = np.array([0.2, 0])
        dirres = np.cumsum(la.dsumsw2d(*lm, k, kpar, a, r, np.arange(800)))[[11, -1]]
        assert isclose(
            np.angle(la.lsumsw2d(*lm, k, kpar, a, r, 0) - dirres[0]),
            np.angle(dirres[1] - dirres[0]),
            rel_tol=1e-1,
            abs_tol=EPS,
        )


class TestLSumSW3d:
    @pytest.mark.parametrize(
        "lm",
        [
            (0, 0),
            (1, -1),
            (1, 0),
            (2, -2),
            (2, 1),
            (7, -7),
            (7, 0),
            (7, 3),
            (8, -8),
            (8, 0),
            (8, 4),
        ],
    )
    @pytest.mark.parametrize("k", [2, 1.1 + 0.1j])
    @pytest.mark.parametrize("kpar", [np.zeros(3), np.array([0.1, 0.2, -0.3])])
    @pytest.mark.parametrize(
        "a",
        [
            np.array([[1, 0, 0], [0, 2.0, 0], [0, 0, 3]]),
            np.array([[1.3, 0.2, 0.1], [-0.1, 1.9, 0.2], [-0.1, 0.3, 2.2]]),
        ],
    )
    @pytest.mark.parametrize("r", [np.zeros(3), np.array([0.2, -0.3, 0.4])])
    @pytest.mark.slow
    def test_regular(self, lm, k, kpar, a, r):
        if lm[0] in (0, 2) and lm[1] == 0:
            rtol = 0.2
            if (
                lm[0] == 2
                and np.array_equal(kpar, [0.1, 0.2, -0.3])
                and np.array_equal(
                    a, np.array([[1.3, 0.2, 0.1], [-0.1, 1.9, 0.2], [-0.1, 0.3, 2.2]])
                )
            ):
                rtol = 0.3
        else:
            rtol = 0.1
        assert isclose(
            la.lsumsw3d(*lm, k, kpar, a, r, 0),
            osscilation_avg(la.dsumsw3d(*lm, k, kpar, a, r, np.arange(120))),
            rel_tol=rtol,
            abs_tol=1e-8,
        )

    # Only the imaginary part fits
    @pytest.mark.parametrize(
        "lm",
        [
            (0, 0),
            (1, -1),
            (1, 0),
            (2, -2),
            (2, 1),
            (5, -5),
            (5, 0),
            (5, 2),
            (6, -6),
            (6, 0),
            (6, 3),
        ],
    )
    def test_singular_r0(self, lm):
        k = 1
        kpar = np.zeros(3)
        a = np.array([[2 * np.pi, 0, 0], [0, 2 * np.pi, 0], [0, 0, 3]])
        r = np.zeros(3)
        assert isclose(
            np.imag(la.lsumsw3d(*lm, k, kpar, a, r, 0)),
            np.imag(np.sum(la.dsumsw3d(*lm, k, kpar, a, r, np.arange(2)))),
            rel_tol=0.2,
            abs_tol=1e-8,
        )

    # Only the imaginary part fits
    @pytest.mark.parametrize(
        "lm",
        [
            (0, 0),
            (1, -1),
            (1, 0),
            (2, -2),
            (2, 0),
            (2, 1),
            (5, -5),
            (5, 0),
            (5, 2),
            (6, -6),
            (6, 0),
            (6, 3),
        ],
    )
    def test_singular_r1(self, lm):
        k = 1
        kpar = np.zeros(3)
        a = np.array([[2 * np.pi, 0, 0], [0, 2 * np.pi, 0], [0, 0, 3]])
        r = np.array([0.2, -0.3, 0.4])
        assert isclose(
            np.imag(la.lsumsw3d(*lm, k, kpar, a, r, 0)),
            np.imag(np.sum(la.dsumsw3d(*lm, k, kpar, a, r, np.arange(2)))),
            rel_tol=0.2,
            abs_tol=1e-8,
        )


class TestLSumSW2dShift:
    @pytest.mark.parametrize(
        "lm",
        [
            (0, 0),
            (1, -1),
            (1, 0),
            (2, -2),
            (2, 0),
            (2, 1),
            (7, -7),
            (7, 0),
            (7, 3),
            (8, -8),
            (8, 0),
            (8, 4),
        ],
    )
    @pytest.mark.parametrize("k", [2, 1.1 + 0.1j])
    @pytest.mark.parametrize("kpar", [np.zeros(2), np.array([0.1, 0.2])])
    @pytest.mark.parametrize(
        "a", [np.array([[1, 0], [0, 1.2]]), np.array([[1.1, 0.2], [-0.1, 1]])]
    )
    @pytest.mark.parametrize("r", [np.array([0, 0, 0.1]), np.array([0.2, 0.2, 1.1])])
    @pytest.mark.slow
    def test_regular(self, lm, k, kpar, a, r):
        assert isclose(
            la.lsumsw2d_shift(*lm, k, kpar, a, r, 0),
            osscilation_avg(la.dsumsw2d_shift(*lm, k, kpar, a, r, np.arange(800))),
            rel_tol=5e-2,
            abs_tol=EPS,
        )

    @pytest.mark.parametrize(
        "lm",
        [(0, 0), (1, 0), (2, -2), (2, 0), (2, 1), (7, 0), (8, -8), (8, 0), (8, 4)],
    )
    @pytest.mark.slow
    def test_singular_r0(self, lm):
        r = np.array([0, 0, 0.1])
        k = 1
        kpar = np.zeros(2)
        a = np.array([[2 * np.pi, 0], [0, 1]])
        if (lm[0] + lm[1]) % 2 == 1:
            assert isclose(
                la.lsumsw2d_shift(*lm, k, kpar, a, r, 0),
                la.dsumsw2d_shift(*lm, k, kpar, a, r, np.arange(800)).sum(),
                rel_tol=1e-2,
                abs_tol=EPS,
            )
        else:
            dirres = np.cumsum(la.dsumsw2d_shift(*lm, k, kpar, a, r, np.arange(800)))[
                [1, -1]
            ]
            assert isclose(
                np.angle(la.lsumsw2d_shift(*lm, k, kpar, a, r, 0) - dirres[0]),
                np.angle(dirres[1] - dirres[0]),
                rel_tol=1e-1,
                abs_tol=EPS,
            )

    @pytest.mark.parametrize(
        "lm",
        [
            (0, 0),
            (1, -1),
            (1, 0),
            (2, -2),
            (2, 0),
            (2, 1),
            (7, -7),
            (7, 0),
            (7, 3),
            (8, -8),
            (8, 0),
            (8, 4),
        ],
    )
    @pytest.mark.slow
    def test_singular_r1(self, lm):
        r = np.array([0.2, 0.2, 1.1])
        k = 1
        kpar = np.zeros(2)
        a = np.array([[2 * np.pi, 0], [0, 1]])
        if (lm[0] + lm[1]) % 2 == 1:
            assert isclose(
                la.lsumsw2d_shift(*lm, k, kpar, a, r, 0),
                la.dsumsw2d_shift(*lm, k, kpar, a, r, np.arange(1000)).sum(),
                rel_tol=1e-1,
                abs_tol=EPS,
            )
        else:
            dirres = np.cumsum(la.dsumsw2d_shift(*lm, k, kpar, a, r, np.arange(1000)))[
                [11, -1]
            ]
            assert isclose(
                np.angle(la.lsumsw2d_shift(*lm, k, kpar, a, r, 0) - dirres[0]),
                np.angle(dirres[1] - dirres[0]),
                rel_tol=1e-1,
                abs_tol=EPS,
            )


class TestLSumSW1dShift:
    @pytest.mark.parametrize(
        "lm",
        [
            (0, 0),
            (1, -1),
            (1, 0),
            (2, -2),
            (2, 0),
            (2, 1),
            (7, -7),
            (7, 0),
            (7, 3),
            (8, -8),
            (8, 0),
            (8, 4),
        ],
    )
    @pytest.mark.parametrize("k", [2, 1.1 + 0.1j])
    @pytest.mark.parametrize("kpar", [-0.2, 0, 0.7])
    @pytest.mark.parametrize("a", [1, 1.1])
    @pytest.mark.parametrize("r", [np.array([0, 0.1, 0]), np.array([0.2, 1.1, 0.2])])
    def test_regular(self, lm, k, kpar, a, r):
        assert isclose(
            la.lsumsw1d_shift(*lm, k, kpar, a, r, 0),
            la.dsumsw1d_shift(*lm, k, kpar, a, r, np.arange(100_000)).sum(),
            rel_tol=1e-2,
            abs_tol=EPS,
        )

    @pytest.mark.parametrize(
        "lm",
        [
            (0, 0),
            (1, -1),
            (1, 0),
            (2, -2),
            (2, 0),
            (2, 1),
            (7, -7),
            (7, 0),
            (7, 3),
            (8, -8),
            (8, 0),
            (8, 4),
        ],
    )
    @pytest.mark.parametrize("r", [np.array([0, 0.1, 0]), np.array([0.2, 0.2, 1.1])])
    @pytest.mark.slow
    def test_singular(self, lm, r):
        k = 1
        kpar = 0
        a = 2 * np.pi
        if lm == (2, 0) and np.array_equal(r, [0.2, 0.2, 1.1]):
            assert isclose(
                la.lsumsw1d_shift(*lm, k, kpar, a, r, 0),
                la.dsumsw1d_shift(*lm, k, kpar, a, r, np.arange(2_200_000)).sum(),
                rel_tol=1e-1,
                abs_tol=EPS,
            )
        else:
            assert isclose(
                la.lsumsw1d_shift(*lm, k, kpar, a, r, 0),
                la.dsumsw1d_shift(*lm, k, kpar, a, r, np.arange(800_000)).sum(),
                rel_tol=1e-1,
                abs_tol=EPS,
            )


class TestLSumCW1dShift:
    @pytest.mark.parametrize("l", [-2, -1, 0, 7, 8])
    @pytest.mark.parametrize("k", [2, 1.1 + 0.1j])
    @pytest.mark.parametrize("kpar", [-0.2, 0, 0.7])
    @pytest.mark.parametrize("a", [1, 1.1])
    @pytest.mark.parametrize("r", [np.array([0, 0.1]), np.array([0.2, 1.1])])
    def test_regular(self, l, k, kpar, a, r):  # noqa: E741
        assert isclose(
            la.lsumcw1d_shift(l, k, kpar, a, r, 0),
            la.dsumcw1d_shift(l, k, kpar, a, r, np.arange(100_000)).sum(),
            rel_tol=5e-2,
            abs_tol=EPS,
        )

    @pytest.mark.parametrize("l", [-2, -1, 0, 7, 8])
    @pytest.mark.parametrize("r", [np.array([0, 0.1]), np.array([0.2, 1.1])])
    @pytest.mark.slow
    def test_singular(self, l, r):  # noqa: E741
        k = 1
        kpar = 0
        a = 2 * np.pi
        assert isclose(
            la.lsumcw1d_shift(l, k, kpar, a, r, 0),
            la.dsumcw1d_shift(l, k, kpar, a, r, np.arange(1_800_000)).sum(),
            rel_tol=1e-1,
            abs_tol=EPS,
        )
