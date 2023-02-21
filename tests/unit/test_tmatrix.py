import copy

import numpy as np

import treams
from treams import TMatrix


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestInit:
    def test_simple(self):
        tm = TMatrix(np.eye(6), k0=1)
        assert (
            np.all(tm == np.eye(6))
            and tm.k0 == 1
            and tm.material == (1, 1, 0)
            and tm.basis == treams.SphericalWaveBasis.default(1)
            and tm.poltype == treams.config.POLTYPE
        )

    def test_complex(self):
        tm = TMatrix(
            np.diag([1, 2]),
            k0=3,
            material=[2, 8, 1],
            basis=treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]], [1, 0, 0]),
        )
        assert (
            np.all(tm == np.diag([1, 2]))
            and tm.k0 == 3
            and tm.material == (2, 8, 1)
            and tm.basis == treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]], [1, 0, 0])
        )


class TestSphere:
    def test(self):
        tm = TMatrix.sphere(2, 3, 4, [(2, 1, 1), (9, 1, 2)])
        m = treams.coeffs.mie([1, 2], [12], [2, 9], [1, 1], [1, 2])
        assert (
            np.all(np.diag(tm)[:6:2] == m[0, 1, 1])
            and np.all(np.diag(tm)[1:6:2] == m[0, 0, 0])
            and np.all(np.diag(tm)[6::2] == m[1, 1, 1])
            and np.all(np.diag(tm)[7::2] == m[1, 0, 0])
            and np.all(np.diag(tm, -1)[:6:2] == m[0, 0, 1])
            and np.all(np.diag(tm, 1)[:6:2] == m[0, 1, 0])
            and np.all(np.diag(tm, -1)[6::2] == m[1, 0, 1])
            and np.all(np.diag(tm, 1)[6::2] == m[1, 1, 0])
            and np.all(np.diag(tm, -1)[1:6:2] == 0)
            and np.all(np.diag(tm, 1)[1:6:2] == 0)
            and np.all(np.diag(tm, -1)[7::2] == 0)
            and np.all(np.diag(tm, 1)[7::2] == 0)
            and np.all(np.triu(tm, 2) == 0)
            and np.all(np.tril(tm, -2) == 0)
            and tm.k0 == 3
            and tm.material == (9, 1, 2)
            and tm.poltype == "helicity"
            and tm.basis == treams.SphericalWaveBasis.default(2)
        )


class TestProperties:
    def test_xs_ext_avg(self):
        tm = TMatrix.sphere(2, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        assert isclose(tm.xs_ext_avg, 2.9294930236877077)

    def test_xs_ext_avg_kappa_zero(self):
        tm = TMatrix.sphere(2, 3, [4], [(2 + 1j, 3), (9, 4)])
        assert isclose(tm.xs_ext_avg, 0.15523595021864234)

    def test_xs_sca_avg(self):
        tm = TMatrix.sphere(2, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        assert isclose(tm.xs_sca_avg, 1.6603264386283758)

    def test_xs_sca_avg_kappa_zero(self):
        tm = TMatrix.sphere(2, 3, [4], [(2 + 1j, 3), (9, 4)])
        assert isclose(tm.xs_sca_avg, 0.08434021223849283)

    def test_cd(self):
        tm = TMatrix.sphere(2, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        assert isclose(tm.cd, -0.9230263013362784)

    def test_chi(self):
        tm = TMatrix.sphere(2, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        assert isclose(tm.chi, 0.7483463517622965)

    def test_db(self):
        tm = TMatrix.sphere(2, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        assert isclose(tm.db, 0.6085814346536764)

    def test_modes(self):
        tm = TMatrix.sphere(2, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        l, m, pol = tm.basis["lmp"]
        assert (
            np.all(l == 6 * [1] + 10 * [2])
            and np.all(m == [-1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2])
            and np.all(pol == [int((i + 1) % 2) for i in range(16)])
        )


class TestXs:
    def test(self):
        tm = TMatrix.sphere(2, 3, [4], [(2 + 1j, 1, 1), (9, 1, 2)])
        illu = treams.PhysicsArray(
            [[0.5]], basis=treams.PlaneWaveBasis([[0, 0, tm.ks[0], 0]])
        )
        illu = illu.expand(tm.basis, k0=tm.k0, material=tm.material) @ illu
        xs = tm.xs(illu, 0.125)
        assert isclose(xs[0][0], 3.194830855171616,) and isclose(xs[1][0], 5.63547158)


class TestTranslate:
    def test(self):
        tm = TMatrix.sphere(3, 0.1, [0.2], [(2 + 1j, 1.1, 1), (9, 1, 2)])
        m = copy.deepcopy(tm)
        rs = np.array([[0.1, 0.2, 0.3], [-0.4, -0.5, -0.4]])
        tm = tm.translate(rs[0])
        tm = tm.translate(rs[1])
        tm = tm.translate(-rs[0] - rs[1])
        assert np.all(np.abs(tm - m) < 1e-8)

    def test_kappa_zero(self):
        tm = TMatrix.sphere(3, 0.1, [0.2], [(2 + 1j, 1.1), (9, 1)])
        m = copy.deepcopy(tm)
        rs = np.array([[0.1, 0.2, 0.3], [-0.4, -0.5, -0.4]])
        tm.translate(rs[0])
        tm.translate(rs[1])
        tm.translate(-rs[0] - rs[1])
        assert np.all(np.abs(tm - m) < 1e-8)


class TestClusterRotate:
    def test(self):
        tms = [TMatrix.sphere(3, 0.1, [0.1], [i * i, 1]) for i in range(1, 5)]
        rs1 = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]])
        tm1 = TMatrix.cluster(tms, rs1)
        tm1 = tm1.interact().globalmat()
        tm1 = tm1.rotate(1, 2, 3) @ tm1 @ tm1.rotate.inv(1, 2, 3)
        a = np.array([[np.cos(1), -np.sin(1), 0], [np.sin(1), np.cos(1), 0], [0, 0, 1]])
        b = np.array([[np.cos(2), 0, np.sin(2)], [0, 1, 0], [-np.sin(2), 0, np.cos(2)]])
        c = np.array([[np.cos(3), -np.sin(3), 0], [np.sin(3), np.cos(3), 0], [0, 0, 1]])
        rs2 = (a @ b @ c @ rs1.T).T
        tm2 = TMatrix.cluster(tms, rs2)
        tm2 = tm2.interact().globalmat()
        assert np.all(np.abs(tm1 - tm2) < 1e-16)
