import copy

import numpy as np

import ptsa
from ptsa import TMatrix


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestInit:
    def test_simple(self):
        tm = TMatrix(np.eye(6), k0=1)
        assert (
            np.all(tm == np.eye(6))
            and tm.k0 == 1
            and tm.material == (1, 1, 0)
            and tm.basis == ptsa.SphericalWaveBasis.default(1)
            and tm.poltype == ptsa.config.POLTYPE
        )

    def test_complex(self):
        tm = TMatrix(
            np.diag([1, 2]),
            k0=3,
            material=[2, 8, 1],
            basis=ptsa.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]], [1, 0, 0]),
        )
        assert (
            np.all(tm == np.diag([1, 2]))
            and tm.k0 == 3
            and tm.material == (2, 8, 1)
            and tm.basis == ptsa.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]], [1, 0, 0])
        )


class TestSphere:
    def test(self):
        tm = TMatrix.sphere(2, 3, 4, [(2, 1, 1), (9, 1, 2)])
        m = ptsa.coeffs.mie([1, 2], [12], [2, 9], [1, 1], [1, 2])
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
            and tm.basis == ptsa.SphericalWaveBasis.default(2)
        )


class TestProperties:
    def test_xs_ext_avg(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], kappa=[1, 2])
        assert isclose(tm.xs_ext_avg, 2.9294930236877077)

    def test_xs_ext_avg_kappa_zero(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], [3, 4])
        assert isclose(tm.xs_ext_avg, 0.15523595021864234)

    def test_xs_sca_avg(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], kappa=[1, 2])
        assert isclose(tm.xs_sca_avg, 1.6603264386283758)

    def test_xs_sca_avg_kappa_zero(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], [3, 4])
        assert isclose(tm.xs_sca_avg, 0.08434021223849283)

    def test_cd(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], kappa=[1, 2])
        assert isclose(tm.cd, 78.6846845551069)

    def test_chi(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], kappa=[1, 2])
        assert isclose(tm.chi, 0.7483463517622965)

    def test_db(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], kappa=[1, 2])
        assert isclose(tm.db, 1.9550276337620118)

    def test_modes(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], kappa=[1, 2])
        l, m, pol = tm.modes
        assert (
            np.all(l == 6 * [1] + 10 * [2])
            and np.all(m == [-1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2])
            and np.all(pol == [int((i + 1) % 2) for i in range(16)])
        )

    def test_fullmodes(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], kappa=[1, 2])
        pidx, l, m, pol = tm.fullmodes
        assert (
            np.all(l == 6 * [1] + 10 * [2])
            and np.all(m == [-1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2])
            and np.all(pol == [int((i + 1) % 2) for i in range(16)])
            and np.all(pidx == 16 * [0])
        )


class TestXs:
    def test(self):
        tm = TMatrix.sphere(2, 3, [4], [2 + 1j, 9], kappa=[1, 2])
        illu = 0.5 * tm.illuminate_pw(0, 0, tm.ks[0], 0)
        xs = tm.xs(illu, 0.125)
        assert isclose(xs[0][0], 3.194830855171616,) and isclose(xs[1][0], 5.63547158)


class TestTranslate:
    def test(self):
        tm = TMatrix.sphere(3, 0.1, [0.2], [2 + 1j, 9], [1.1, 1], [1, 2])
        m = copy.deepcopy(tm.t)
        rs = np.array([[0.1, 0.2, 0.3], [-0.4, -0.5, -0.4]])
        tm.translate(rs[0])
        tm.translate(rs[1])
        tm.translate(-rs[0] - rs[1])
        assert np.all(np.abs(tm.t - m) < 1e-8)

    def test_kappa_zero(self):
        tm = TMatrix.sphere(3, 0.1, [0.2], [2 + 1j, 9], [1.1, 1])
        m = copy.deepcopy(tm.t)
        rs = np.array([[0.1, 0.2, 0.3], [-0.4, -0.5, -0.4]])
        tm.translate(rs[0])
        tm.translate(rs[1])
        tm.translate(-rs[0] - rs[1])
        assert np.all(np.abs(tm.t - m) < 1e-8)


class TestClusterRotate:
    def test(self):
        tms = [TMatrix.sphere(3, 0.1, [0.1], [i * i, 1]) for i in range(1, 5)]
        rs1 = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]])
        tm1 = TMatrix.cluster(tms, rs1)
        tm1.interact().globalmat()
        tm1.rotate(1, 2, 3)
        a = np.array([[np.cos(1), -np.sin(1), 0], [np.sin(1), np.cos(1), 0], [0, 0, 1]])
        b = np.array([[np.cos(2), 0, np.sin(2)], [0, 1, 0], [-np.sin(2), 0, np.cos(2)]])
        c = np.array([[np.cos(3), -np.sin(3), 0], [np.sin(3), np.cos(3), 0], [0, 0, 1]])
        rs2 = (a @ b @ c @ rs1.T).T
        tm2 = TMatrix.cluster(tms, rs2)
        tm2.interact().globalmat()
        assert np.all(np.abs(tm1.t - tm2.t) < 1e-16)


class TestField:
    def test_sh(self):
        tm = TMatrix(np.eye(2), 1, modes=([4, 5], [-4, -3], [0, 1]))
        r = [1, 2, 3]
        r_sph = ptsa.special.car2sph(r)
        expect = ptsa.special.vsph2car(
            ptsa.special.vsw_A([4, 5], [-4, -3], tm.ks * r_sph[0], *r_sph[1:], [0, 1]),
            r_sph,
        )
        assert np.all(tm.field(r) == expect)

    def test_rh(self):
        tm = TMatrix(np.eye(2), 1, modes=([4, 5], [-4, -3], [0, 1]))
        r = [1, 2, 3]
        r_sph = ptsa.special.car2sph(r)
        expect = ptsa.special.vsph2car(
            ptsa.special.vsw_rA([4, 5], [-4, -3], tm.ks * r_sph[0], *r_sph[1:], [0, 1]),
            r_sph,
        )
        assert np.all(tm.field(r, scattered=False) == expect)

    def test_sp(self):
        tm = TMatrix(np.eye(2), 1, modes=([4, 5], [-4, -3], [0, 1]), helicity=False)
        r = [1, 2, 3]
        r_sph = ptsa.special.car2sph(r)
        expect = ptsa.special.vsph2car(
            np.stack(
                (
                    ptsa.special.vsw_M(4, -4, tm.ks[0] * r_sph[0], *r_sph[1:]),
                    ptsa.special.vsw_N(5, -3, tm.ks[1] * r_sph[0], *r_sph[1:]),
                ),
            ),
            r_sph,
        )
        assert np.all(tm.field(r) == expect)

    def test_rp(self):
        tm = TMatrix(np.eye(2), 1, modes=([4, 5], [-4, -3], [0, 1]), helicity=False)
        r = [1, 2, 3]
        r_sph = ptsa.special.car2sph(r)
        expect = ptsa.special.vsph2car(
            np.stack(
                (
                    ptsa.special.vsw_rM(4, -4, tm.ks[0] * r_sph[0], *r_sph[1:]),
                    ptsa.special.vsw_rN(5, -3, tm.ks[1] * r_sph[0], *r_sph[1:]),
                ),
            ),
            r_sph,
        )
        assert np.all(tm.field(r, scattered=False) == expect)


class TestPick:
    def test(self):
        tm = TMatrix([[1, 2], [3, 4]], 1, modes=([1, 1], [0, 0], [0, 1]))
        tm.pick(([1, 2], [0, 0], [0, 0]))
        assert np.array_equal(tm.t, [[1, 0], [0, 0]])
