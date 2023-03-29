import numpy as np

import treams
from treams import TMatrixC


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestInit:
    def test_simple(self):
        tm = TMatrixC(np.eye(6), k0=1)
        assert (
            np.all(tm == np.eye(6))
            and tm.k0 == 1
            and tm.material.epsilon == 1
            and tm.material.mu == 1
            and tm.material.kappa == 0
            and np.all(tm.basis.positions == [[0, 0, 0]])
            and tm.poltype == "helicity"
            and np.all(tm.basis.kz == 6 * [0])
            and np.all(tm.basis.m == [-1, -1, 0, 0, 1, 1])
            and np.all(tm.basis.pol == [1, 0, 1, 0, 1, 0])
            and np.all(tm.basis.pidx == [0, 0, 0, 0, 0, 0])
            and np.all(tm.ks == [1, 1])
        )

    def test_complex(self):
        tm = TMatrixC(
            np.diag([1, 2]),
            k0=3,
            material=treams.Material(2, 8, 1),
            basis=treams.CylindricalWaveBasis([[1, 0, 0], [1, 0, 1]], [1, 0, 0]),
        )
        assert (
            np.all(tm == np.diag([1, 2]))
            and tm.k0 == 3
            and tm.material.epsilon == 2
            and tm.material.mu == 8
            and tm.material.kappa == 1
            and np.all(tm.basis.positions == [[1, 0, 0]])
            and tm.poltype == "helicity"
            and np.all(tm.basis.kz == [1, 1])
            and np.all(tm.basis.m == [0, 0])
            and np.all(tm.basis.pol == [0, 1])
            and np.all(tm.basis.pidx == [0, 0])
            and np.all(tm.ks == [9, 15])
        )


class TestCylinder:
    def test(self):
        tm = TMatrixC.cylinder([1], 2, 3, 4, [(2, 1, 1), (9, 1, 2)])
        m = treams.coeffs.mie_cyl(1, [-2, -1, 0, 1, 2], 3, [4], [2, 9], [1, 1], [1, 2])
        assert (
            tm[0, 0] == m[0, 1, 1]
            and tm[1, 1] == m[0, 0, 0]
            and tm[0, 1] == m[0, 1, 0]
            and tm[1, 0] == m[0, 0, 1]
            and tm[2, 2] == m[1, 1, 1]
            and tm[3, 3] == m[1, 0, 0]
            and tm[2, 3] == m[1, 1, 0]
            and tm[3, 2] == m[1, 0, 1]
            and tm[4, 4] == m[2, 1, 1]
            and tm[5, 5] == m[2, 0, 0]
            and tm[4, 5] == m[2, 1, 0]
            and tm[5, 4] == m[2, 0, 1]
            and tm[6, 6] == m[3, 1, 1]
            and tm[7, 7] == m[3, 0, 0]
            and tm[6, 7] == m[3, 1, 0]
            and tm[7, 6] == m[3, 0, 1]
            and tm[8, 8] == m[4, 1, 1]
            and tm[9, 9] == m[4, 0, 0]
            and tm[8, 9] == m[4, 1, 0]
            and tm[9, 8] == m[4, 0, 1]
            and tm.k0 == 3
            and tm.material.epsilon == 9
            and tm.material.mu == 1
            and tm.material.kappa == 2
            and np.all(tm.basis.positions == [[0, 0, 0]])
            and tm.poltype == "helicity"
            and np.all(tm.basis.kz == 10 * [1])
            and np.all(tm.basis.m == [-2, -2, -1, -1, 0, 0, 1, 1, 2, 2])
            and np.all(tm.basis.pol == [int((i + 1) % 2) for i in range(10)])
            and np.all(tm.basis.pidx == 10 * [0])
            and np.all(tm.ks == [3, 15])
        )


class TestProperties:
    def test_xw_ext_avg(self):
        tm = TMatrixC.cylinder([-1, 1], 1, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        assert isclose(tm.xw_ext_avg, 1.2047234457348013)

    def test_xw_ext_avg_kappa_zero(self):
        tm = TMatrixC.cylinder([-1, 1], 1, 3, [4], [(2 + 1j, 3), (9,)])
        assert isclose(tm.xw_ext_avg, 0.6661748147466017)

    def test_xw_sca_avg(self):
        tm = TMatrixC.cylinder([-1, 1], 1, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        assert isclose(tm.xw_sca_avg, 0.6847986332271352)

    def test_xw_sca_avg_kappa_zero(self):
        tm = TMatrixC.cylinder([-1, 1], 1, 3, [4], [(2 + 1j, 3), (9, 4)])
        assert isclose(tm.xw_sca_avg, 0.18067830829683562)

    def test_krho(self):
        tm = TMatrixC.cylinder([0, 5], 1, 3, [1], [(2 + 1j,), ()])
        assert np.all(
            np.abs(tm.krhos - [3, 3, 3, 3, 3, 3, 4j, 4j, 4j, 4j, 4j, 4j]) < 1e-16
        )

    def test_modes(self):
        tm = TMatrixC.cylinder([-1, 1], 1, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        kz, m, pol = tm.basis["kzmp"]
        assert (
            np.all(kz == 6 * [-1] + 6 * [1])
            and np.all(m == [-1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1])
            and np.all(pol == [int((i + 1) % 2) for i in range(12)])
        )

    def test_fullmodes(self):
        tm = TMatrixC.cylinder([-1, 1], 1, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        pidx, kz, m, pol = tm.basis[()]
        assert (
            np.all(kz == 6 * [-1] + 6 * [1])
            and np.all(m == [-1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1])
            and np.all(pol == [int((i + 1) % 2) for i in range(12)])
            and np.all(pidx == 12 * [0])
        )
