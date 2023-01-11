import numpy as np
import pytest

import ptsa
import ptsa.special as sc


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestSWB:
    def test_init_empty(self):
        b = ptsa.SphericalWaveBasis([])
        assert b.l.size == 0 and b.m.size == 0 and b.pol.size == 0 and b.pidx.size == 0

    def test_init_numpy(self):
        b = ptsa.SphericalWaveBasis(np.array([[1, 0, 0]]), [0, 1, 0])
        assert (
            np.all(b.l == [1])
            and np.all(b.m == [0])
            and np.all(b.pol == [0])
            and np.all(b.pidx == [0])
        )

    def test_init_duplicate(self):
        b = ptsa.SphericalWaveBasis([[0, 1, 0, 0], [0, 1, 0, 0]])
        assert (
            np.all(b.l == [1])
            and np.all(b.m == [0])
            and np.all(b.pol == [0])
            and np.all(b.pidx == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            ptsa.SphericalWaveBasis([[0, 0], [0, 0]])

    def test_init_invalid_positions(self):
        with pytest.raises(ValueError):
            ptsa.SphericalWaveBasis([[1, 0, 0]], [1, 2])

    def test_init_non_int_value(self):
        with pytest.raises(ValueError):
            ptsa.SphericalWaveBasis([[1.1, 0, 0]])

    def test_init_non_natural_l(self):
        with pytest.raises(ValueError):
            ptsa.SphericalWaveBasis([[0, 0, 0]])

    def test_init_non_too_large_m(self):
        with pytest.raises(ValueError):
            ptsa.SphericalWaveBasis([[1, -2, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            ptsa.SphericalWaveBasis([[1, 0, 2]])

    def test_init_unsecified_positions(self):
        with pytest.raises(ValueError):
            ptsa.SphericalWaveBasis([[1, 1, 0, 0]])

    def test_property_positions(self):
        a = np.array([[1, 2, 3]])
        b = ptsa.SphericalWaveBasis([[1, 0, 0]], a)
        assert (a == b.positions).all()

    def test_getitem_plmp(self):
        b = ptsa.SphericalWaveBasis([[0, 2, -1, 1]])
        assert b["plmp"] == ([0], [2], [-1], [1])

    def test_getitem_lmp(self):
        b = ptsa.SphericalWaveBasis([[2, -1, 1]])
        assert b["lmp"] == ([2], [-1], [1])

    def test_getitem_lm(self):
        b = ptsa.SphericalWaveBasis([[2, -1, 1]])
        assert b["LM"] == ([2], [-1])

    def test_getitem_invalid_index(self):
        b = ptsa.SphericalWaveBasis([[1, 0, 0]])
        with pytest.raises(IndexError):
            b["l"]

    def test_getitem_int(self):
        b = ptsa.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert b[1] == (0, 1, 0, 1)

    def test_getitem_tuple(self):
        b = ptsa.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert (np.array(b[()]) == ([0, 0], [1, 1], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert a == b[:1]

    def test_default(self):
        a = ptsa.SphericalWaveBasis.default(2, 2, [[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(
            zip(
                16 * [0] + 16 * [1],
                2 * (6 * [1] + 10 * [2]),
                2 * [-1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2],
                16 * [1, 0],
            ),
            [[0, 0, 0], [1, 0, 0]],
        )
        assert a == b

    def test_ebcm(self):
        a = ptsa.SphericalWaveBasis.ebcm(2, 2, positions=[[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(
            zip(
                16 * [0] + 16 * [1],
                2 * ([2, 2] + 3 * [1, 1, 2, 2] + [2, 2]),
                2 * [-2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
                16 * [1, 0],
            ),
            [[0, 0, 0], [1, 0, 0]],
        )
        assert a == b

    def test_ebcm_mmax(self):
        a = ptsa.SphericalWaveBasis.ebcm(2, mmax=1)
        b = ptsa.SphericalWaveBasis(
            zip(
                3 * [1, 1, 2, 2], [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1], 6 * [1, 0],
            ),
        )
        assert a == b

    def test_defaultlmax(self):
        assert ptsa.SphericalWaveBasis.defaultlmax(60, 2) == 3

    def test_defaultlmax_fail(self):
        with pytest.raises(ValueError):
            ptsa.SphericalWaveBasis.defaultlmax(1)

    def test_defaultdim(self):
        assert ptsa.SphericalWaveBasis.defaultdim(3, 2) == 60

    def test_defaultdim_fail(self):
        with pytest.raises(ValueError):
            ptsa.SphericalWaveBasis.defaultdim(1, -1)

    def test_property_isglobal_true(self):
        b = ptsa.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert b.isglobal

    def test_property_isglobal_false(self):
        b = ptsa.SphericalWaveBasis(
            [[0, 1, 0, 0], [1, 1, 0, 0]], [[0, 0, 0], [1, 0, 0]]
        )
        assert not b.isglobal

    def test_call_rh(self):
        modes = [[0, 3, -2, 0], [1, 1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1, 1)
        x = b(r, k0, "helicity", material)
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

    def test_call_sh(self):
        modes = [[0, 3, -2, 0], [1, 1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1, 1)
        x = b(r, k0, "helicity", material, "singular")
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

    def test_call_rp(self):
        modes = [[0, 3, -2, 0], [1, 1, 1, 0]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1)
        x = b(r, k0, "parity", material)
        rsph = sc.car2sph(r[:, None] - positions)
        y = sc.vsw_rM(
            [3, 1], [-2, 1], 2 * k0 * rsph[..., 0], rsph[..., 1], rsph[..., 2],
        )
        assert np.all(sc.vsph2car(y, rsph) == x)

    def test_call_sp(self):
        modes = [[0, 3, -2, 1], [1, 1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.SphericalWaveBasis(modes, positions)
        r = np.array([3, 4, 5])
        k0 = 4
        material = (4, 1)
        x = b(r, k0, "parity", material, "singular")
        rsph = sc.car2sph(r[None] - positions)
        y = sc.vsw_N(
            [3, 1], [-2, 1], 2 * k0 * rsph[..., 0], rsph[..., 1], rsph[..., 2],
        )
        assert np.all(sc.vsph2car(y, rsph) == x)

    def test_call_invalid(self):
        b = ptsa.SphericalWaveBasis([[1, 0, 0]])
        with pytest.raises(ValueError):
            b([1, 0, 0], 1, "asdf")

    def test_rotate(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        x = b.rotate(1, 2, 3, a, where=where)
        y = ptsa.PhysicsArray(
            [[ptsa.sw.rotate(1, 0, 0, 1, -1, 0, 1, 2, 3), 0]], basis=(a, b)
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_rotate_invalid(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            a.rotate(1, basis=b)

    def test_translate(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        x = b.translate(3, [[0, 0, 0], [0, 1, 1]], a, material=(1, 2, 0), where=where)
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

    def test_translate_invalid(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            a.translate(1, [1, 0, 0], basis=b)

    def test_translate_invalid_r(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            b.translate(1, [1, 0])

    def test_expand_sw(self):
        a = ptsa.SphericalWaveBasis([[2, 0, 0]], [0, 1, 1])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        x = b.expand(3, a, material=(1, 2, 0), modetype="singular", where=where)
        y = ptsa.PhysicsArray(
            [[ptsa.sw.translate(2, 0, 0, 1, -1, 0, 6, 0.25 * np.pi, 0.5 * np.pi), 0]],
            basis=(a, b),
            poltype="helicity",
            k0=3,
            material=ptsa.Material(1, 2, 0),
            modetype=("regular", "singular"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_expand_invalid_modetype(self):
        b = ptsa.SphericalWaveBasis([[1, 0, 0]])
        with pytest.raises(ValueError):
            b.expand(1, modetype="asdf")

    def test_expand_sw_lattice_1d(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice(1)
        x = b.expand(3, lattice=lattice, where=where)
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
            material=ptsa._material.vacuum,
            modetype=("regular", "singular"),
            kpar=[np.nan, np.nan, 0],
            lattice=lattice,
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_expand_sw_lattice_2d(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice([[1, 0], [0, 1]])
        x = b.expand(3, lattice=lattice, where=where)
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
            material=ptsa._material.vacuum,
            modetype=("regular", "singular"),
            kpar=[0, 0, np.nan],
            lattice=lattice,
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_expand_sw_lattice_3d(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice(np.eye(3))
        x = b.expand(3, lattice=lattice, where=where)
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
            material=ptsa._material.vacuum,
            modetype=("regular", "singular"),
            kpar=[0, 0, 0],
            lattice=lattice,
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_expand_sw_lattice_kpar(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            b.expand(1, lattice=[[1, 0], [0, 1]], kpar=0)

    def test_expand_cw(self):
        a = ptsa.CylindricalWaveBasis([[0.3, 2, 0]])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        x = b.expand(3, a, lattice=2, material=(1, 4, 0), where=where)
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

    def test_expand_pw(self):
        a = ptsa.PlaneWaveBasis([[0.3, 0.2, 1, 0]])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        lattice = ptsa.Lattice([[2, 0], [0, 2]])
        x = b.expand(
            3, a, lattice=lattice, kpar=[0.3, 0.2], material=(1, 4, 0), where=where
        )
        y = ptsa.PhysicsArray(
            [[ptsa.sw.periodic_to_pw(0.3, 0.2, 1, 0, 1, -1, 0, 4), 0]],
            basis=(a, b),
            poltype="helicity",
            k0=3,
            material=ptsa.Material(1, 4, 0),
            modetype=(None, "singular"),
            lattice=lattice,
            kpar=[0.3, 0.2, np.nan],
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_basischange(self):
        a = ptsa.SphericalWaveBasis([[2, 0, 0], [1, 0, 1]])
        b = ptsa.SphericalWaveBasis([[2, 0, 0], [1, 0, 1]])
        where = [True, False]
        x = b.basischange(a, where=where)
        y = ptsa.PhysicsArray(
            [[-np.sqrt(0.5), 0], [0, 0]], basis=(a, b), poltype=("helicity", "parity"),
        )
        print(x, y)
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann


class TestCWB:
    def test_init_empty(self):
        b = ptsa.CylindricalWaveBasis([])
        assert b.kz.size == 0 and b.m.size == 0 and b.pol.size == 0 and b.pidx.size == 0

    def test_init_numpy(self):
        b = ptsa.CylindricalWaveBasis(np.array([[0.5, 0, 0]]), [0, 1, 0])
        assert (
            np.all(b.kz == [0.5])
            and np.all(b.m == [0])
            and np.all(b.pol == [0])
            and np.all(b.pidx == [0])
        )

    def test_init_duplicate(self):
        b = ptsa.CylindricalWaveBasis([[0, 1, 0, 0], [0, 1, 0, 0]])
        assert (
            np.all(b.kz == [1])
            and np.all(b.m == [0])
            and np.all(b.pol == [0])
            and np.all(b.pidx == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            ptsa.CylindricalWaveBasis([[0, 0], [0, 0]])

    def test_init_invalid_positions(self):
        with pytest.raises(ValueError):
            ptsa.CylindricalWaveBasis([[1, 0, 0]], [1, 2])

    def test_init_non_int_value(self):
        with pytest.raises(ValueError):
            ptsa.CylindricalWaveBasis([[1, 0.2, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            ptsa.CylindricalWaveBasis([[1, 0, 2]])

    def test_init_unsecified_positions(self):
        with pytest.raises(ValueError):
            ptsa.CylindricalWaveBasis([[1, 1, 0, 0]])

    def test_property_positions(self):
        a = np.array([[1, 2, 3]])
        b = ptsa.CylindricalWaveBasis([[1, 0, 0]], a)
        assert (a == b.positions).all()

    def test_getitem_pkzmp(self):
        b = ptsa.CylindricalWaveBasis([[0, 2, -1, 1]])
        assert b["pkzmp"] == ([0], [2], [-1], [1])

    def test_getitem_kzmp(self):
        b = ptsa.CylindricalWaveBasis([[2, -1, 1]])
        assert b["kzmp"] == ([2], [-1], [1])

    def test_getitem_kzm(self):
        b = ptsa.CylindricalWaveBasis([[2, -1, 1]])
        assert b["KZM"] == ([2], [-1])

    def test_getitem_invalid_index(self):
        b = ptsa.CylindricalWaveBasis([[1, 0, 0]])
        with pytest.raises(IndexError):
            b["l"]

    def test_getitem_int(self):
        b = ptsa.CylindricalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert b[1] == (0, 1, 0, 1)

    def test_getitem_tuple(self):
        b = ptsa.CylindricalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert (np.array(b[()]) == ([0, 0], [1, 1], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = ptsa.CylindricalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert a == b[:1]

    def test_default(self):
        a = ptsa.CylindricalWaveBasis.default([0.3, -0.2], 2, 2, [[0, 0, 0], [1, 0, 0]])
        b = ptsa.CylindricalWaveBasis(
            zip(
                20 * [0] + 20 * [1],
                2 * (10 * [0.3] + 10 * [-0.2]),
                4 * [-2, -2, -1, -1, 0, 0, 1, 1, 2, 2],
                20 * [1, 0],
            ),
            [[0, 0, 0], [1, 0, 0]],
        )
        assert a == b

    def test_defaultlmax(self):
        assert ptsa.CylindricalWaveBasis.defaultmmax(112, 4, 2) == 3

    def test_defaultlmax_fail(self):
        with pytest.raises(ValueError):
            ptsa.CylindricalWaveBasis.defaultmmax(1)

    def test_defaultdim(self):
        assert ptsa.CylindricalWaveBasis.defaultdim(3, 2, 4) == 120

    def test_defaultdim_fail(self):
        with pytest.raises(ValueError):
            ptsa.CylindricalWaveBasis.defaultdim(1, -1)

    def test_property_isglobal_true(self):
        b = ptsa.CylindricalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert b.isglobal

    def test_property_isglobal_false(self):
        b = ptsa.CylindricalWaveBasis(
            [[0, 1, 0, 0], [1, 1, 0, 0]], [[0, 0, 0], [1, 0, 0]]
        )
        assert not b.isglobal

    def test_call_rh(self):
        modes = [[0, 0.3, -2, 0], [1, 0.1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.CylindricalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1, 1)
        x = b(r, k0, "helicity", material)
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

    def test_call_sh(self):
        modes = [[0, 0.3, -2, 0], [1, 0.1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.CylindricalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1, 1)
        x = b(r, k0, "helicity", material, "singular")
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

    def test_call_rp(self):
        modes = [[0, 0.3, -2, 0], [1, 0.1, 1, 0]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.CylindricalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1)
        x = b(r, k0, "parity", material)
        rcyl = sc.car2cyl(r[:, None] - positions)
        y = sc.vcw_rM(
            [0.3, 0.1],
            [-2, 1],
            rcyl[..., 0] * [np.sqrt(64 - 0.3 ** 2), np.sqrt(64 - 0.1 ** 2)],
            rcyl[..., 1],
            rcyl[..., 2],
        )
        print(x)
        print(sc.vcyl2car(y, rcyl))
        assert np.all(sc.vcyl2car(y, rcyl) == x)

    def test_call_sp(self):
        modes = [[0, 0.3, -2, 1], [1, 0.1, 1, 1]]
        positions = np.array([[0, 0, 0], [1, 0, 0]])
        b = ptsa.CylindricalWaveBasis(modes, positions)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        k0 = 4
        material = (4, 1)
        x = b(r, k0, "parity", material, "singular")
        rcyl = sc.car2cyl(r[:, None] - positions)
        y = sc.vcw_N(
            [0.3, 0.1],
            [-2, 1],
            rcyl[..., 0] * [np.sqrt(64 - 0.3 ** 2), np.sqrt(64 - 0.1 ** 2)],
            rcyl[..., 1],
            rcyl[..., 2],
            8,
        )
        print(x)
        print(sc.vcyl2car(y, rcyl))
        assert np.all(sc.vcyl2car(y, rcyl) == x)

    def test_call_invalid(self):
        b = ptsa.CylindricalWaveBasis([[1, 0, 0]])
        with pytest.raises(ValueError):
            b([1, 0, 0], 1, "asdf")

    def test_rotate(self):
        a = ptsa.CylindricalWaveBasis([[0.1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[0.1, 0, 0], [0.1, 1, 0]])
        where = [True, False]
        x = b.rotate(2, a, where=where)
        y = ptsa.PhysicsArray(
            [[ptsa.cw.rotate(0.1, 0, 0, 0.1, 0, 0, 2), 0]], basis=(a, b)
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_rotate_invalid(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            b.rotate(1, basis=a)

    def test_translate(self):
        a = ptsa.CylindricalWaveBasis([[0.1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[0.1, -1, 0], [0.1, 1, 0]])
        where = [True, False]
        x = b.translate(3, [[0, 0, 0], [0, 1, 1]], a, material=(1, 2, 0), where=where)
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
        print(np.asarray(x))
        print(np.asarray(y))
        print(x.ann)
        print(y.ann)
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_translate_invalid(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            b.translate(1, [1, 0, 0], basis=a)

    def test_translate_invalid_r(self):
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            b.translate(1, [1, 0])

    # def test_expand_sw(self):
    #     a = ptsa.SphericalWaveBasis([[2, 0, 0]], [0, 1, 1])
    #     b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
    #     where = [True, False]
    #     x = b.expand(3, a, material=(1, 2, 0), modetype="singular", where=where)
    #     y = ptsa.PhysicsArray(
    #         [[ptsa.sw.translate(2, 0, 0, 1, -1, 0, 6, 0.25 * np.pi, 0.5 * np.pi), 0]],
    #         basis=(a, b),
    #         poltype="helicity",
    #         k0=3,
    #         material=ptsa.Material(1, 2, 0),
    #         modetype=("regular", "singular"),
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    # def test_expand_invalid_modetype(self):
    #     b = ptsa.SphericalWaveBasis([[1, 0, 0]])
    #     with pytest.raises(ValueError):
    #         b.expand(1, modetype="asdf")

    # def test_expand_sw_lattice_1d(self):
    #     b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
    #     where = [[True, False], [False, False]]
    #     lattice = ptsa.Lattice(1)
    #     x = b.expand(3, lattice=lattice, where=where)
    #     y = ptsa.PhysicsArray(
    #         [
    #             [
    #                 ptsa.sw.translate_periodic(
    #                     3, 0, lattice[...], [0, 0, 0], [[1], [-1], [0]]
    #                 )[0, 0],
    #                 0,
    #             ],
    #             [0, 0],
    #         ],
    #         basis=b,
    #         poltype="helicity",
    #         k0=3,
    #         material=ptsa._material.vacuum,
    #         modetype=("regular", "singular"),
    #         kpar=[np.nan, np.nan, 0],
    #         lattice=lattice,
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    # def test_expand_sw_lattice_2d(self):
    #     b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
    #     where = [[True, False], [False, False]]
    #     lattice = ptsa.Lattice([[1, 0], [0, 1]])
    #     x = b.expand(3, lattice=lattice, where=where)
    #     y = ptsa.PhysicsArray(
    #         [
    #             [
    #                 ptsa.sw.translate_periodic(
    #                     3, [0, 0], lattice[...], [0, 0, 0], [[1], [-1], [0]]
    #                 )[0, 0],
    #                 0,
    #             ],
    #             [0, 0],
    #         ],
    #         basis=b,
    #         poltype="helicity",
    #         k0=3,
    #         material=ptsa._material.vacuum,
    #         modetype=("regular", "singular"),
    #         kpar=[0, 0, np.nan],
    #         lattice=lattice,
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    # def test_expand_sw_lattice_3d(self):
    #     b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
    #     where = [[True, False], [False, False]]
    #     lattice = ptsa.Lattice(np.eye(3))
    #     x = b.expand(3, lattice=lattice, where=where)
    #     y = ptsa.PhysicsArray(
    #         [
    #             [
    #                 ptsa.sw.translate_periodic(
    #                     3, [0, 0, 0], lattice[...], [0, 0, 0], [[1], [-1], [0]]
    #                 )[0, 0],
    #                 0,
    #             ],
    #             [0, 0],
    #         ],
    #         basis=b,
    #         poltype="helicity",
    #         k0=3,
    #         material=ptsa._material.vacuum,
    #         modetype=("regular", "singular"),
    #         kpar=[0, 0, 0],
    #         lattice=lattice,
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    # def test_expand_sw_lattice_kpar(self):
    #     b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
    #     with pytest.raises(ValueError):
    #         b.expand(1, lattice=[[1, 0], [0, 1]], kpar=0)

    # def test_expand_cw(self):
    #     a = ptsa.CylindricalWaveBasis([[0.3, 2, 0]])
    #     b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
    #     where = [True, False]
    #     x = b.expand(3, a, lattice=2, material=(1, 4, 0), where=where)
    #     y = ptsa.PhysicsArray(
    #         [[ptsa.sw.periodic_to_cw(2, 0, 0, 1, -1, 0, 6, 2), 0]],
    #         basis=(a, b),
    #         poltype="helicity",
    #         k0=3,
    #         material=ptsa.Material(1, 4, 0),
    #         modetype="singular",
    #         lattice=ptsa.Lattice(2),
    #         kpar=[np.nan, np.nan, 0.3],
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    # def test_expand_pw(self):
    #     a = ptsa.PlaneWaveBasis([[0.3, 0.2, 1, 0]])
    #     b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
    #     where = [True, False]
    #     lattice = ptsa.Lattice([[2, 0], [0, 2]])
    #     x = b.expand(
    #         3, a, lattice=lattice, kpar=[0.3, 0.2], material=(1, 4, 0), where=where
    #     )
    #     y = ptsa.PhysicsArray(
    #         [[ptsa.sw.periodic_to_pw(.3, .2, 1, 0, 1, -1, 0, 4), 0]],
    #         basis=(a, b),
    #         poltype="helicity",
    #         k0=3,
    #         material=ptsa.Material(1, 4, 0),
    #         modetype=(None, "singular"),
    #         lattice=lattice,
    #         kpar=[0.3, 0.2, np.nan],
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    # def test_basischange(self):
    #     a = ptsa.SphericalWaveBasis([[2, 0, 0], [1, 0, 1]])
    #     b = ptsa.SphericalWaveBasis([[2, 0, 0], [1, 0, 1]])
    #     where = [True, False]
    #     x = b.basischange(a, where=where)
    #     y = ptsa.PhysicsArray(
    #         [[-np.sqrt(.5), 0], [0, 0]],
    #         basis=(a, b),
    #         poltype=("helicity", "parity"),
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann
