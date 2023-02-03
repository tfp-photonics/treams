import numpy as np
import pytest

import ptsa
import ptsa.special as sc


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestRotate:
    def test_sw_invalid(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(IndexError):
            ptsa.rotate(1, basis=(b, a))

    def test_sw(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        x = ptsa.rotate(1, 2, 3, basis=(a, b), where=where)
        y = ptsa.PhysicsArray(
            [[ptsa.sw.rotate(1, 0, 0, 1, -1, 0, 1, 2, 3), 0]], basis=(a, b)
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_cw(self):
        a = ptsa.CylindricalWaveBasis([[0.1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[0.1, 0, 0], [0.1, 1, 0]])
        where = [True, False]
        x = ptsa.rotate(2, basis=(a, b), where=where)
        y = ptsa.PhysicsArray(
            [[ptsa.cw.rotate(0.1, 0, 0, 0.1, 0, 0, 2), 0]], basis=(a, b)
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_cw_invalid(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(IndexError):
            ptsa.rotate(1, basis=(a, b))

    def test_pw(self):
        b = ptsa.PlaneWaveBasis([[1, 0, 0, 0], [0, 1, 0, 0]])
        a = ptsa.PlaneWaveBasis(
            [[np.cos(2), np.sin(2), 0, 0], [-np.sin(2), np.cos(2), 0, 0]]
        )
        where = [True, False]
        x = ptsa.rotate(2, basis=b, where=where)
        y = ptsa.PhysicsArray([[1, 0], [0, 0]], basis=(a, b))
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_pwp(self):
        b = ptsa.PlaneWaveBasisPartial([[1, 0, 0], [0, 1, 0]])
        a = ptsa.PlaneWaveBasisPartial(
            [[np.cos(2), np.sin(2), 0], [-np.sin(2), np.cos(2), 0]]
        )
        where = [True, False]
        x = ptsa.rotate(2, basis=b, where=where)
        y = ptsa.PhysicsArray([[1, 0], [0, 0]], basis=(a, b))
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann


class TestTranslate:
    def test_sw(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        x = ptsa.translate(
            [[0, 0, 0], [0, 1, 1]], k0=3, basis=(a, b), material=(1, 2, 0), where=where
        )
        y = ptsa.PhysicsArray(
            [
                [[ptsa.sw.translate(1, 0, 0, 1, -1, 0, 0, 0, 0, singular=False), 0]],
                [
                    [
                        ptsa.sw.translate(
                            1, 0, 0, 1, -1, 0, 6, np.pi / 4, np.pi / 2, singular=False
                        ),
                        0,
                    ]
                ],
            ],
            basis=(None, a, b),
            poltype=(None, "helicity", "helicity"),
            k0=(None, 3, 3),
            material=(None,) + 2 * (ptsa.Material(1, 2, 0),),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_sw_invalid(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(IndexError):
            ptsa.translate([1, 0, 0], k0=1, basis=(b, a))

    def test_invalid_r(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            ptsa.translate([1, 0], basis=b, k0=1)

    def test_cw(self):
        a = ptsa.CylindricalWaveBasis([[0.1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[0.1, -1, 0], [0.1, 1, 0]])
        where = [True, False]
        x = ptsa.translate(
            [[0, 0, 0], [0, 1, 1]], k0=3, basis=(a, b), material=(1, 2, 0), where=where
        )
        y = ptsa.PhysicsArray(
            [
                [
                    [
                        ptsa.cw.translate(
                            0.1, 0, 0, 0.1, -1, 0, 0, 0, 0, singular=False
                        ),
                        0,
                    ]
                ],
                [
                    [
                        ptsa.cw.translate(
                            0.1,
                            0,
                            0,
                            0.1,
                            -1,
                            0,
                            np.sqrt(18 - 0.01),
                            np.pi / 2,
                            1,
                            singular=False,
                        ),
                        0,
                    ]
                ],
            ],
            basis=(None, a, b),
            k0=(None, 3, 3),
            material=(None,) + 2 * (ptsa.Material(1, 2, 0),),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_pw(self):
        a = ptsa.PlaneWaveBasis([[0.1, 0, 0, 0]])
        b = ptsa.PlaneWaveBasis([[0.1, 0, 0, 0], [0.1, 1, 0, 0]])
        where = [True, False]
        x = ptsa.translate([[0, 0, 0], [0, 1, 1]], basis=(a, b), where=where)
        y = ptsa.PhysicsArray(
            [
                [[ptsa.pw.translate(0.1, 0, 0, 0, 0, 0), 0]],
                [[ptsa.pw.translate(0.1, 0, 0, 0, 1, 1), 0]],
            ],
            basis=(None, a, b),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_pwp(self):
        a = ptsa.PlaneWaveBasisPartial([[4, 0, 0]])
        b = ptsa.PlaneWaveBasisPartial([[4, 0, 0], [4, 1, 0]])
        where = [True, False]
        x = ptsa.translate([[0, 0, 0], [0, 1, 1]], k0=5, basis=(a, b), where=where)
        y = ptsa.PhysicsArray(
            [
                [[ptsa.pw.translate(4, 0, 3, 0, 0, 0), 0]],
                [[ptsa.pw.translate(4, 0, 3, 0, 1, 1), 0]],
            ],
            basis=(None, a, b),
            k0=(None, 5, 5),
            material=(None, ptsa.Material(), ptsa.Material()),
            modetype=(None, "up", "up"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann


class TestExpand:
    def test_sw_sw_sing(self):
        a = ptsa.SphericalWaveBasis([[2, 0, 0]], [0, 1, 1])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        x = ptsa.expand(
            (a, b), ("regular", "singular"), k0=3, material=(1, 2, 0), where=where
        )
        y = ptsa.PhysicsArray(
            [[ptsa.sw.translate(2, 0, 0, 1, -1, 0, 6, 0.25 * np.pi, 0.5 * np.pi), 0]],
            basis=(a, b),
            poltype="helicity",
            k0=3,
            material=ptsa.Material(1, 2, 0),
            modetype=("regular", "singular"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_cw_cw_sing(self):
        a = ptsa.CylindricalWaveBasis([[0.2, 0, 0]], [0, 1, 1])
        b = ptsa.CylindricalWaveBasis([[0.2, -1, 0], [0.2, 1, 0]])
        where = [True, False]
        x = ptsa.expand(
            (a, b), ("regular", "singular"), k0=3, material=(1, 2, 0), where=where
        )
        y = ptsa.PhysicsArray(
            [
                [
                    ptsa.cw.translate(
                        0.2, 0, 0, 0.2, -1, 0, np.sqrt(18 - 0.04), np.pi / 2, 1
                    ),
                    0,
                ]
            ],
            basis=(a, b),
            k0=3,
            material=ptsa.Material(1, 2, 0),
            modetype=("regular", "singular"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_sw_cw(self):
        a = ptsa.SphericalWaveBasis([[1, 1, 0]])
        b = ptsa.CylindricalWaveBasis([[0.3, 1, 0], [0.1, 1, 0]])
        where = [True, False]
        x = ptsa.expand((a, b), k0=3, material=(1, 4, 0), where=where)
        y = ptsa.PhysicsArray(
            [[ptsa.cw.to_sw(1, 1, 0, 0.3, 1, 0, 6), 0]],
            basis=(a, b),
            poltype="helicity",
            k0=3,
            material=ptsa.Material(1, 4, 0),
            modetype="regular",
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_pw_pw(self):
        a = ptsa.PlaneWaveBasis([[3, 0, 4, 0], [0, 0, 5, 0]])
        b = ptsa.PlaneWaveBasis([[3, 0, 4, 0], [0, 5, 0, 0]])
        where = [True, False]
        x = ptsa.expand((a, b), k0=2.5, material=(2, 2, 0), where=where)
        y = ptsa.PhysicsArray(
            [[1, 0], [0, 0]], basis=(a, b), k0=2.5, material=ptsa.Material(2, 2, 0),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_cw_pw(self):
        a = ptsa.CylindricalWaveBasis([[3, 1, 0]], [1, 2, 3])
        b = ptsa.PlaneWaveBasis([[0, 4, 3, 0], [0, 4, 3, 1]])
        where = [True, False]
        x = ptsa.expand((a, b), k0=5, where=where)
        y = ptsa.PhysicsArray(
            [
                [
                    ptsa.pw.to_cw(3, 1, 0, 0, 4, 3, 0)
                    * ptsa.pw.translate(0, 4, 3, 1, 2, 3),
                    0,
                ]
            ],
            basis=(a, b),
            k0=5,
            material=ptsa.Material(),
            modetype=("regular", None),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_sw_pw(self):
        a = ptsa.SphericalWaveBasis([[3, 1, 0]], [1, 2, 3])
        b = ptsa.PlaneWaveBasis([[0, 4, 3, 0], [0, 4, 3, 1]])
        where = [True, False]
        x = ptsa.expand((a, b), k0=5, where=where)
        y = ptsa.PhysicsArray(
            [
                [
                    ptsa.pw.to_sw(3, 1, 0, 0, 4, 3, 0)
                    * ptsa.pw.translate(0, 4, 3, 1, 2, 3),
                    0,
                ]
            ],
            basis=(a, b),
            poltype="helicity",
            k0=5,
            material=ptsa.Material(),
            modetype=("regular", None),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann


class TestExpandLattice:
    def test_sw_1d(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice(1)
        x = ptsa.expandlattice(lattice, basis=b, k0=3, where=where)
        y = ptsa.PhysicsArray(
            [
                [
                    ptsa.sw.translate_periodic(
                        3, 0, lattice[...], [0, 0, 0], [[1], [-1], [0]]
                    )[0, 0],
                    0,
                ],
                [0, 0],
            ],
            basis=b,
            poltype="helicity",
            k0=3,
            material=ptsa._material.Material(),
            modetype=("regular", "singular"),
            kpar=[np.nan, np.nan, 0],
            lattice=lattice,
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_sw_2d(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice([[1, 0], [0, 1]])
        x = ptsa.expandlattice(lattice, basis=b, k0=3, where=where)
        y = ptsa.PhysicsArray(
            [
                [
                    ptsa.sw.translate_periodic(
                        3, [0, 0], lattice[...], [0, 0, 0], [[1], [-1], [0]]
                    )[0, 0],
                    0,
                ],
                [0, 0],
            ],
            basis=b,
            poltype="helicity",
            k0=3,
            material=ptsa._material.Material(),
            modetype=("regular", "singular"),
            kpar=[0, 0, np.nan],
            lattice=lattice,
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_sw_3d(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice(np.eye(3))
        x = ptsa.expandlattice(lattice, basis=b, k0=3, where=where)
        y = ptsa.PhysicsArray(
            [
                [
                    ptsa.sw.translate_periodic(
                        3, [0, 0, 0], lattice[...], [0, 0, 0], [[1], [-1], [0]]
                    )[0, 0],
                    0,
                ],
                [0, 0],
            ],
            basis=b,
            poltype="helicity",
            k0=3,
            material=ptsa._material.Material(),
            modetype=("regular", "singular"),
            kpar=[0, 0, 0],
            lattice=lattice,
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_sw_kpar(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            ptsa.expandlattice([[1, 0], [0, 1]], 0, basis=b, k0=1)

    def test_cw_sw(self):
        a = ptsa.CylindricalWaveBasis([[0.3, 2, 0]])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        x = ptsa.expandlattice(2, basis=(a, b), k0=3, material=(1, 4, 0), where=where)
        y = ptsa.PhysicsArray(
            [[ptsa.sw.periodic_to_cw(2, 0, 0, 1, -1, 0, 6, 2), 0]],
            basis=(a, b),
            poltype="helicity",
            k0=3,
            material=ptsa.Material(1, 4, 0),
            modetype="singular",
            lattice=ptsa.Lattice(2),
            kpar=[np.nan, np.nan, 0.3],
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_pw_sw(self):
        a = ptsa.PlaneWaveBasis([[3, 0, 4, 0]])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        lattice = ptsa.Lattice([[2, 0], [0, 2]])
        x = ptsa.expandlattice(
            lattice, [3, 0], basis=(a, b), k0=2.5, material=(1, 4, 0), where=where
        )
        y = ptsa.PhysicsArray(
            [[ptsa.sw.periodic_to_pw(3, 0, 4, 0, 1, -1, 0, 4), 0]],
            basis=(a, b),
            k0=2.5,
            kpar=[3, 0, np.nan],
            lattice=lattice,
            material=ptsa.Material(1, 4, 0),
            modetype=(None, "singular"),
            poltype="helicity",
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_cw_1d(self):
        b = ptsa.CylindricalWaveBasis([[0.1, -1, 0], [0.1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice(1, "x")
        x = ptsa.expandlattice(lattice, basis=b, k0=3, where=where)
        y = ptsa.PhysicsArray(
            [
                [
                    ptsa.cw.translate_periodic(
                        3, 0, lattice[...], [0, 0, 0], [[0.1], [-1], [0]]
                    )[0, 0],
                    0,
                ],
                [0, 0],
            ],
            basis=b,
            k0=3,
            material=ptsa._material.Material(),
            modetype=("regular", "singular"),
            kpar=[0, np.nan, 0.1],
            lattice=lattice,
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_cw_2d(self):
        b = ptsa.CylindricalWaveBasis([[0.1, -1, 0], [0.1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice([[1, 0], [0, 1]])
        x = ptsa.expandlattice(lattice, basis=b, k0=3, where=where)
        y = ptsa.PhysicsArray(
            [
                [
                    ptsa.cw.translate_periodic(
                        3, [0, 0], lattice[...], [0, 0, 0], [[0.1], [-1], [0]]
                    )[0, 0],
                    0,
                ],
                [0, 0],
            ],
            basis=b,
            k0=3,
            material=ptsa._material.Material(),
            modetype=("regular", "singular"),
            kpar=[0, 0, 0.1],
            lattice=lattice,
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_cw_kpar(self):
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            ptsa.expandlattice(lattice=[[1, 0], [0, 1]], basis=b, kpar=0, k0=1)

    def test_pw_cw(self):
        a = ptsa.PlaneWaveBasis([[3, 0, 4, 0]])
        b = ptsa.CylindricalWaveBasis([[4, -1, 0], [4, 1, 0]])
        where = [True, False]
        lattice = ptsa.Lattice(2, "x")
        x = ptsa.expandlattice(
            lattice, 3, basis=(a, b), k0=2.5, material=(1, 4, 0), where=where
        )
        y = ptsa.PhysicsArray(
            [[ptsa.cw.periodic_to_pw(3, 0, 4, 0, 4, -1, 0, 2), 0]],
            basis=(a, b),
            k0=2.5,
            material=ptsa.Material(1, 4, 0),
            modetype=(None, "singular"),
            lattice=lattice,
            kpar=[3, np.nan, np.nan],
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann


class TestChangePoltype:
    def test_sw(self):
        b = ptsa.CylindricalWaveBasis([[2, 0, 0], [1, 0, 1]])
        where = [True, False]
        x = ptsa.changepoltype(basis=b, where=where)
        y = ptsa.PhysicsArray(
            [[-np.sqrt(0.5), 0], [0, 0]], basis=(b, b), poltype=("helicity", "parity"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_cw(self):
        a = ptsa.SphericalWaveBasis([[2, 0, 0], [1, 0, 1]])
        b = ptsa.SphericalWaveBasis([[2, 0, 0], [1, 0, 1]])
        where = [True, False]
        x = ptsa.changepoltype(basis=(a, b), where=where)
        y = ptsa.PhysicsArray(
            [[-np.sqrt(0.5), 0], [0, 0]], basis=(a, b), poltype=("helicity", "parity"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_pw(self):
        b = ptsa.PlaneWaveBasis([[2, 0, 0, 0], [1, 0, 0, 1]])
        where = [True, False]
        x = ptsa.changepoltype(basis=b, where=where)
        y = ptsa.PhysicsArray(
            [[-np.sqrt(0.5), 0], [0, 0]], basis=(b, b), poltype=("helicity", "parity"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_pwp(self):
        b = ptsa.PlaneWaveBasisPartial([[2, 0, 0], [1, 0, 1]])
        where = [True, False]
        x = ptsa.changepoltype(basis=b, where=where)
        y = ptsa.PhysicsArray(
            [[-np.sqrt(0.5), 0], [0, 0]], basis=(b, b), poltype=("helicity", "parity"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann


class TestPermute:
    def test_pw(self):
        a = ptsa.PlaneWaveBasis([[1, 2, 3, 1], [1, 2, 3, 0]])
        b = ptsa.PlaneWaveBasis([[2, 3, 1, 1], [2, 3, 1, 0]])
        print(repr(ptsa.permute(basis=b)))
        assert ptsa.permute(basis=b).basis[0] == a

    def test_pwp(self):
        a = ptsa.PlaneWaveBasisPartial([[1, 2, 1], [1, 2, 0]], "yz")
        b = ptsa.PlaneWaveBasisPartial([[1, 2, 1], [1, 2, 0]])
        assert ptsa.permute(basis=b).basis[0] == a


class TestEField:
    def test_sw_rh(self):
        modes = [[0, 3, -2, 0], [1, 1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1, 1)
        x = ptsa.efield(r, basis=b, k0=k0, poltype="helicity", material=material)
        rsph = sc.car2sph(r[:, None] - positions)
        y = sc.vsw_rA(
            [3, 1],
            [-2, 1],
            k0 * rsph[..., 0] * [1, 3],
            rsph[..., 1],
            rsph[..., 2],
            [0, 1],
        )
        assert np.all(sc.vsph2car(y, rsph) == x)

    def test_sw_sh(self):
        modes = [[0, 3, -2, 0], [1, 1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1, 1)
        x = ptsa.efield(
            r,
            basis=b,
            k0=k0,
            poltype="helicity",
            material=material,
            modetype="singular",
        )
        rsph = sc.car2sph(r[:, None] - positions)
        y = sc.vsw_A(
            [3, 1],
            [-2, 1],
            k0 * rsph[..., 0] * [1, 3],
            rsph[..., 1],
            rsph[..., 2],
            [0, 1],
        )
        assert np.all(sc.vsph2car(y, rsph) == x)

    def test_sw_rp(self):
        modes = [[0, 3, -2, 0], [1, 1, 1, 0]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1)
        x = ptsa.efield(r, basis=b, k0=k0, poltype="parity", material=material)
        rsph = sc.car2sph(r[:, None] - positions)
        y = sc.vsw_rM(
            [3, 1], [-2, 1], 2 * k0 * rsph[..., 0], rsph[..., 1], rsph[..., 2],
        )
        assert np.all(sc.vsph2car(y, rsph) == x)

    def test_sw_sp(self):
        modes = [[0, 3, -2, 1], [1, 1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(modes, positions)
        r = np.array([3, 4, 5])
        k0 = 4
        material = (4, 1)
        x = ptsa.efield(
            r, basis=b, k0=k0, poltype="parity", material=material, modetype="singular",
        )
        rsph = sc.car2sph(r[None] - positions)
        y = sc.vsw_N(
            [3, 1], [-2, 1], 2 * k0 * rsph[..., 0], rsph[..., 1], rsph[..., 2],
        )
        assert np.all(sc.vsph2car(y, rsph) == x)

    def test_cw_rh(self):
        modes = [[0, 0.3, -2, 0], [1, 0.1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.CylindricalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1, 1)
        x = ptsa.efield(r, basis=b, k0=k0, poltype="helicity", material=material)
        rcyl = sc.car2cyl(r[:, None] - positions)
        y = sc.vcw_rA(
            [0.3, 0.1],
            [-2, 1],
            rcyl[..., 0] * [np.sqrt(16 - 0.3 ** 2), np.sqrt(144 - 0.1 ** 2)],
            rcyl[..., 1],
            rcyl[..., 2],
            [k0, 3 * k0],
            [0, 1],
        )
        assert np.all(sc.vcyl2car(y, rcyl) == x)

    def test_cw_sh(self):
        modes = [[0, 0.3, -2, 0], [1, 0.1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.CylindricalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1, 1)
        x = ptsa.efield(
            r,
            basis=b,
            k0=k0,
            poltype="helicity",
            material=material,
            modetype="singular",
        )
        rcyl = sc.car2cyl(r[:, None] - positions)
        y = sc.vcw_A(
            [0.3, 0.1],
            [-2, 1],
            rcyl[..., 0] * [np.sqrt(16 - 0.3 ** 2), np.sqrt(144 - 0.1 ** 2)],
            rcyl[..., 1],
            rcyl[..., 2],
            [k0, 3 * k0],
            [0, 1],
        )
        assert np.all(sc.vcyl2car(y, rcyl) == x)

    def test_cw_rp(self):
        modes = [[0, 0.3, -2, 0], [1, 0.1, 1, 0]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.CylindricalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1)
        x = ptsa.efield(r, basis=b, k0=k0, poltype="parity", material=material)
        rcyl = sc.car2cyl(r[:, None] - positions)
        y = sc.vcw_rM(
            [0.3, 0.1],
            [-2, 1],
            rcyl[..., 0] * [np.sqrt(64 - 0.3 ** 2), np.sqrt(64 - 0.1 ** 2)],
            rcyl[..., 1],
            rcyl[..., 2],
        )
        assert np.all(sc.vcyl2car(y, rcyl) == x)

    def test_cw_sp(self):
        modes = [[0, 0.3, -2, 1], [1, 0.1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.CylindricalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1)
        x = ptsa.efield(
            r, basis=b, k0=k0, poltype="parity", material=material, modetype="singular",
        )
        rcyl = sc.car2cyl(r[:, None] - positions)
        y = sc.vcw_N(
            [0.3, 0.1],
            [-2, 1],
            rcyl[..., 0] * [np.sqrt(64 - 0.3 ** 2), np.sqrt(64 - 0.1 ** 2)],
            rcyl[..., 1],
            rcyl[..., 2],
            8,
        )
        assert np.all(sc.vcyl2car(y, rcyl) == x)

    def test_pw_h(self):
        modes = [[0, 3, 4, 0], [0, 4, 3, 1]]
        b = ptsa.PlaneWaveBasis(modes)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        x = ptsa.efield(r, basis=b, poltype="helicity")
        y = sc.vpw_A(
            [0, 0],
            [3, 4],
            [4, 3],
            r[..., None, 0],
            r[..., None, 1],
            r[..., None, 2],
            [0, 1],
        )
        assert np.all(y == x)

    def test_pw_p(self):
        modes = [[0, 3, 4, 1], [0, 4, 3, 1]]
        b = ptsa.PlaneWaveBasis(modes)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        x = ptsa.efield(r, basis=b, poltype="parity")
        y = sc.vpw_N(
            [0, 0], [3, 4], [4, 3], r[..., None, 0], r[..., None, 1], r[..., None, 2]
        )
        assert np.all(y == x)

    def test_pwp_h(self):
        modes = [[0, 3, 0], [0, 4, 1]]
        b = ptsa.PlaneWaveBasisPartial(modes)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        x = ptsa.efield(r, basis=b, k0=5, poltype="helicity")
        y = sc.vpw_A(
            [0, 0],
            [3, 4],
            [4, 3],
            r[..., None, 0],
            r[..., None, 1],
            r[..., None, 2],
            [0, 1],
        )
        assert np.all(y == x)

    def test_pwp_p(self):
        modes = [[0, 3, 1], [0, 4, 1]]
        b = ptsa.PlaneWaveBasisPartial(modes)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        x = ptsa.efield(
            r, basis=b, k0=1, poltype="parity", material=(5, 5), modetype="down"
        )
        y = sc.vpw_N(
            [0, 0], [3, 4], [-4, -3], r[..., None, 0], r[..., None, 1], r[..., None, 2]
        )
        assert np.all(y == x)
