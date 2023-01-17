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

    def test_repr(self):
        b = ptsa.SphericalWaveBasis(
            [[0, 1, 0, 0], [1, 1, 0, 0]], [[0.0, 0, 0], [1, 0, 0]]
        )
        assert (
            repr(b)
            == """SphericalWaveBasis(
    pidx=[0 1],
    l=[1 1],
    m=[0 0],
    pol=[0 0],
    positions=[[0. 0. 0.], [1. 0. 0.]],
)"""
        )

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
        x = b.rotate(1, 2, 3, basis=a, where=where)
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
        x = b.translate(
            [[0, 0, 0], [0, 1, 1]], 3, basis=a, material=(1, 2, 0), where=where
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

    def test_translate_invalid(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            a.translate([1, 0, 0], 1, basis=b)

    def test_translate_invalid_r(self):
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            b.translate([1, 0], 1)

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
            material=ptsa._material.Material(),
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
            material=ptsa._material.Material(),
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
            material=ptsa._material.Material(),
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
        a = ptsa.PlaneWaveBasis([[3, 0, 4, 0]])
        b = ptsa.SphericalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        lattice = ptsa.Lattice([[2, 0], [0, 2]])
        x = b.expand(
            2.5, a, lattice=lattice, kpar=[0.3, 0.2], material=(1, 4, 0), where=where
        )
        y = ptsa.PhysicsArray(
            [[ptsa.sw.periodic_to_pw(3, 0, 4, 0, 1, -1, 0, 4), 0]],
            basis=(a, b),
            poltype="helicity",
            k0=2.5,
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
        x = b.basischange(basis=a, where=where)
        y = ptsa.PhysicsArray(
            [[-np.sqrt(0.5), 0], [0, 0]], basis=(a, b), poltype=("helicity", "parity"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_from_iterable(self):
        a = ptsa.SphericalWaveBasis.default(1)
        b = ptsa.SphericalWaveBasis.default(2)
        assert a & b == a

    def test_neq(self):
        b = ptsa.SphericalWaveBasis.default(1)
        assert not b == []


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

    def test_repr(self):
        b = ptsa.CylindricalWaveBasis(
            [[0, 1, 0, 0], [1, 1, 0, 0]], [[0.0, 0, 0], [1, 0, 0]]
        )
        assert (
            repr(b)
            == """CylindricalWaveBasis(
    pidx=[0 1],
    kz=[1. 1.],
    m=[0 0],
    pol=[0 0],
    positions=[[0. 0. 0.], [1. 0. 0.]],
)"""
        )

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

    def test_defaultmmax(self):
        assert ptsa.CylindricalWaveBasis.defaultmmax(112, 4, 2) == 3

    def test_defaultmmax_fail(self):
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
        assert np.all(sc.vcyl2car(y, rcyl) == x)

    def test_call_invalid(self):
        b = ptsa.CylindricalWaveBasis([[1, 0, 0]])
        with pytest.raises(ValueError):
            b([1, 0, 0], 1, "asdf")

    def test_rotate(self):
        a = ptsa.CylindricalWaveBasis([[0.1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[0.1, 0, 0], [0.1, 1, 0]])
        where = [True, False]
        x = b.rotate(2, basis=a, where=where)
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
        x = b.translate(
            [[0, 0, 0], [0, 1, 1]], 3, basis=a, material=(1, 2, 0), where=where
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

    def test_translate_invalid(self):
        a = ptsa.SphericalWaveBasis([[1, 0, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            b.translate([1, 0, 0], 1, basis=a)

    def test_translate_invalid_r(self):
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            b.translate([1, 0], 1)

    def test_expand_cw(self):
        a = ptsa.CylindricalWaveBasis([[0.2, 0, 0]], [0, 1, 1])
        b = ptsa.CylindricalWaveBasis([[0.2, -1, 0], [0.2, 1, 0]])
        where = [True, False]
        x = b.expand(3, a, material=(1, 2, 0), modetype="singular", where=where)
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

    def test_expand_invalid_modetype(self):
        b = ptsa.CylindricalWaveBasis([[1, 0, 0]])
        with pytest.raises(ValueError):
            b.expand(1, modetype="asdf")

    def test_expand_cw_lattice_1d(self):
        b = ptsa.CylindricalWaveBasis([[0.1, -1, 0], [0.1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice(1, "x")
        x = b.expand(3, lattice=lattice, where=where)
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

    def test_expand_cw_lattice_2d(self):
        b = ptsa.CylindricalWaveBasis([[0.1, -1, 0], [0.1, 1, 0]])
        where = [[True, False], [False, False]]
        lattice = ptsa.Lattice([[1, 0], [0, 1]])
        x = b.expand(3, lattice=lattice, where=where)
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

    def test_expand_cw_lattice_kpar(self):
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        with pytest.raises(ValueError):
            b.expand(1, lattice=[[1, 0], [0, 1]], kpar=0)

    def test_expand_sw(self):
        a = ptsa.SphericalWaveBasis([[1, 1, 0]])
        b = ptsa.CylindricalWaveBasis([[0.3, 1, 0], [0.1, 1, 0]])
        where = [True, False]
        x = b.expand(3, a, material=(1, 4, 0), where=where)
        y = ptsa.PhysicsArray(
            [[ptsa.cw.to_sw(1, 1, 0, 0.3, 1, 0, 6), 0]],
            basis=(a, b),
            poltype="helicity",
            k0=3,
            material=ptsa.Material(1, 4, 0),
            modetype="regular",
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_expand_pw(self):
        a = ptsa.PlaneWaveBasis([[3, 0, 4, 0]])
        b = ptsa.CylindricalWaveBasis([[1, -1, 0], [1, 1, 0]])
        where = [True, False]
        lattice = ptsa.Lattice(2, "x")
        x = b.expand(
            2.5, a, lattice=lattice, kpar=[3, 0], material=(1, 4, 0), where=where
        )
        y = ptsa.PhysicsArray(
            [[ptsa.cw.periodic_to_pw(3, 0, 4, 0, 1, -1, 0, 2), 0]],
            basis=(a, b),
            k0=2.5,
            material=ptsa.Material(1, 4, 0),
            modetype=(None, "singular"),
            lattice=lattice,
            kpar=[3, 0, np.nan],
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_basischange(self):
        b = ptsa.CylindricalWaveBasis([[2, 0, 0], [1, 0, 1]])
        where = [True, False]
        x = b.basischange(where=where)
        y = ptsa.PhysicsArray(
            [[-np.sqrt(0.5), 0], [0, 0]], basis=(b, b), poltype=("helicity", "parity"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_from_iterable(self):
        a = ptsa.CylindricalWaveBasis.default(0, 1)
        b = ptsa.CylindricalWaveBasis.default(0, 2)
        assert a & b == a

    def test_neq(self):
        b = ptsa.CylindricalWaveBasis.default(0, 1)
        assert not b == []

    def test_with_periodicity(self):
        a = ptsa.CylindricalWaveBasis.with_periodicity(0.1, 1, 2 * np.pi, 1.5)
        b = ptsa.CylindricalWaveBasis(
            zip(
                *[
                    6 * [-0.9] + 6 * [0.1] + 6 * [1.1],
                    3 * [-1, -1, 0, 0, 1, 1],
                    18 * [1, 0],
                ]
            )
        )
        assert (
            a == b
            and a.hints["lattice"] == ptsa.Lattice(2 * np.pi)
            and a.hints["kpar"] == [np.nan, np.nan, 0.1]
        )


class TestPWB:
    def test_init_empty(self):
        b = ptsa.PlaneWaveBasis([])
        assert b.kx.size == 0 and b.ky.size == 0 and b.kz.size == 0 and b.pol.size == 0

    def test_init_numpy(self):
        b = ptsa.PlaneWaveBasis(np.array([[0.4, 0.2, 0.1, 0]]))
        assert (
            np.all(b.kx == [0.4])
            and np.all(b.ky == [0.2])
            and np.all(b.kz == [0.1])
            and np.all(b.pol == [0])
        )

    def test_init_duplicate(self):
        b = ptsa.PlaneWaveBasis([[0.4, 0.2, 0.1, 0], [0.4, 0.2, 0.1, 0]])
        assert (
            np.all(b.kx == [0.4])
            and np.all(b.ky == [0.2])
            and np.all(b.kz == [0.1])
            and np.all(b.pol == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            ptsa.PlaneWaveBasis([[0, 0], [0, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            ptsa.PlaneWaveBasis([[1, 0, 0, 2]])

    def test_repr(self):
        b = ptsa.PlaneWaveBasis([[0.0, 1.0, 0.0, 0], [1, 1, 0, 0]])
        assert (
            repr(b)
            == """PlaneWaveBasis(
    kx=[0. 1.],
    ky=[1. 1.],
    kz=[0. 0.],
    pol=[0 0],
)"""
        )

    def test_getitem_xyzp(self):
        b = ptsa.PlaneWaveBasis([[0, 2, -1, 1]])
        assert b["xyzp"] == ([0], [2], [-1], [1])

    def test_getitem_xyp(self):
        b = ptsa.PlaneWaveBasis([[0, 2, -1, 1]])
        assert b["xyp"] == ([0], [2], [1])

    def test_getitem_zp(self):
        b = ptsa.PlaneWaveBasis([[0, 2, -1, 1]])
        assert b["ZP"] == ([-1], [1])

    def test_getitem_invalid_index(self):
        b = ptsa.PlaneWaveBasis([[1, 0, 0, 0]])
        with pytest.raises(IndexError):
            b["l"]

    def test_getitem_int(self):
        b = ptsa.PlaneWaveBasis([[1, 0, 0, 0], [1, 0, 0, 1]])
        assert b[1] == (1, 0, 0, 1)

    def test_getitem_tuple(self):
        b = ptsa.PlaneWaveBasis([[1, 0, 0, 0], [1, 0, 0, 1]])
        assert (np.array(b[()]) == ([1, 1], [0, 0], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = ptsa.PlaneWaveBasis([[1, 0, 0, 0]])
        b = ptsa.PlaneWaveBasis([[1, 0, 0, 0], [1, 0, 0, 1]])
        assert a == b[:1]

    def test_default(self):
        a = ptsa.PlaneWaveBasis.default([0.3, -0.2, 0.1])
        b = ptsa.PlaneWaveBasis([[0.3, -0.2, 0.1, 1], [0.3, -0.2, 0.1, 0]])
        assert a == b

    def test_property_isglobal_true(self):
        b = ptsa.PlaneWaveBasis([])
        assert b.isglobal

    def test_call_h(self):
        modes = [[0, 3, 4, 0], [0, 4, 3, 1]]
        b = ptsa.PlaneWaveBasis(modes)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        x = b(r, "helicity")
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

    def test_call_p(self):
        modes = [[0, 3, 4, 1], [0, 4, 3, 1]]
        b = ptsa.PlaneWaveBasis(modes)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        x = b(r, "parity")
        y = sc.vpw_N(
            [0, 0], [3, 4], [4, 3], r[..., None, 0], r[..., None, 1], r[..., None, 2]
        )
        assert np.all(y == x)

    def test_call_invalid(self):
        b = ptsa.PlaneWaveBasis([[1, 0, 0, 0]])
        with pytest.raises(ValueError):
            b([1, 0, 0], "asdf")

    def test_rotate(self):
        b = ptsa.PlaneWaveBasis([[1, 0, 0, 0], [0, 1, 0, 0]])
        a = ptsa.PlaneWaveBasis(
            [[np.cos(2), np.sin(2), 0, 0], [-np.sin(2), np.cos(2), 0, 0]]
        )
        where = [True, False]
        x = b.rotate(2, where=where)
        y = ptsa.PhysicsArray([[1, 0], [0, 0]], basis=(a, b))
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_translate(self):
        a = ptsa.PlaneWaveBasis([[0.1, 0, 0, 0]])
        b = ptsa.PlaneWaveBasis([[0.1, 0, 0, 0], [0.1, 1, 0, 0]])
        where = [True, False]
        x = b.translate([[0, 0, 0], [0, 1, 1]], basis=a, where=where)
        y = ptsa.PhysicsArray(
            [
                [[ptsa.pw.translate(0.1, 0, 0, 0, 0, 0), 0]],
                [[ptsa.pw.translate(0.1, 0, 0, 0, 1, 1), 0]],
            ],
            basis=(None, a, b),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_translate_invalid_r(self):
        b = ptsa.PlaneWaveBasis([[1, -1, 0, 0], [1, 1, 0, 0]])
        with pytest.raises(ValueError):
            b.translate([1, 0])

    def test_expand_pw(self):
        a = ptsa.PlaneWaveBasis([[3, 0, 4, 0], [0, 0, 5, 0]])
        b = ptsa.PlaneWaveBasis([[3, 0, 4, 0], [0, 5, 0, 0]])
        where = [True, False]
        x = b.expand(2.5, a, material=(2, 2, 0), where=where)
        y = ptsa.PhysicsArray(
            [[1, 0], [0, 0]], basis=(a, b), k0=2.5, material=ptsa.Material(2, 2, 0),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_expand_cw(self):
        a = ptsa.CylindricalWaveBasis([[3, 1, 0]], [1, 2, 3])
        b = ptsa.PlaneWaveBasis([[0, 4, 3, 0], [0, 4, 3, 1]])
        where = [True, False]
        x = b.expand(5, a, where=where)
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

    def test_expand_sw(self):
        a = ptsa.SphericalWaveBasis([[3, 1, 0]], [1, 2, 3])
        b = ptsa.PlaneWaveBasis([[0, 4, 3, 0], [0, 4, 3, 1]])
        where = [True, False]
        x = b.expand(5, a, where=where)
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

    def test_basischange(self):
        b = ptsa.PlaneWaveBasis([[2, 0, 0, 0], [1, 0, 0, 1]])
        where = [True, False]
        x = b.basischange(where=where)
        y = ptsa.PhysicsArray(
            [[-np.sqrt(0.5), 0], [0, 0]], basis=(b, b), poltype=("helicity", "parity"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_permute(self):
        a = ptsa.PlaneWaveBasis([[1, 2, 3, 1], [1, 2, 3, 0]])
        b = ptsa.PlaneWaveBasis([[2, 3, 1, 1], [2, 3, 1, 0]])
        assert b.permute() == a

    def test_from_iterable(self):
        a = ptsa.PlaneWaveBasis.default([0, 0, 1])
        b = ptsa.PlaneWaveBasis.default([[0, 0, 1], [0, 1, 0]])
        assert a & b == a

    def test_partial(self):
        a = ptsa.PlaneWaveBasisPartial.default([0, 1], "yz")
        b = ptsa.PlaneWaveBasis.default([0, 0, 1])
        assert b.partial("yz", 1) == a


class TestPWBP:
    def test_init_empty(self):
        b = ptsa.PlaneWaveBasisPartial([])
        assert b.kx.size == 0 and b.ky.size == 0 and b.kz is None and b.pol.size == 0

    def test_init_numpy(self):
        b = ptsa.PlaneWaveBasisPartial(np.array([[0.4, 0.2, 0]]), "zx")
        assert (
            np.all(b.kz == [0.4])
            and np.all(b.kx == [0.2])
            and np.all(b.ky is None)
            and np.all(b.pol == [0])
        )

    def test_init_duplicate(self):
        b = ptsa.PlaneWaveBasisPartial([[0.4, 0.2, 0], [0.4, 0.2, 0]])
        assert (
            np.all(b.kx == [0.4])
            and np.all(b.ky == [0.2])
            and np.all(b.kz is None)
            and np.all(b.pol == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            ptsa.PlaneWaveBasisPartial([[0, 0], [0, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            ptsa.PlaneWaveBasisPartial([[1, 0, 2]])

    def test_repr(self):
        b = ptsa.PlaneWaveBasisPartial([[0.0, 1.0, 0], [1, 1, 0]], "yz")
        assert (
            repr(b)
            == """PlaneWaveBasisPartial(
    ky=[0. 1.],
    kz=[1. 1.],
    pol=[0 0],
)"""
        )

    def test_from_iterable(self):
        a = ptsa.PlaneWaveBasisPartial.default([0, 0])
        b = ptsa.PlaneWaveBasisPartial.default([[0, 0], [0, 1]])
        assert a & b == a

    def test_getitem_int(self):
        b = ptsa.PlaneWaveBasisPartial([[1, 0, 0], [1, 0, 1]])
        assert b[1] == (1, 0, 1)

    def test_getitem_tuple(self):
        b = ptsa.PlaneWaveBasisPartial([[1, 0, 0], [1, 0, 1]])
        assert (np.array(b[()]) == ([1, 1], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = ptsa.PlaneWaveBasisPartial([[1, 0, 0]])
        b = ptsa.PlaneWaveBasisPartial([[1, 0, 0], [1, 0, 0, 1]])
        assert a == b[:1]

    def test_complete(self):
        a = ptsa.PlaneWaveBasis.default([0, 0, 1])
        b = ptsa.PlaneWaveBasisPartial.default([0, 1], "yz")
        assert b.complete(1) == a

    def test_call_h(self):
        modes = [[0, 3, 0], [0, 4, 1]]
        b = ptsa.PlaneWaveBasisPartial(modes)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        x = b(r, 5, "helicity")
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

    def test_call_p(self):
        modes = [[0, 3, 1], [0, 4, 1]]
        b = ptsa.PlaneWaveBasisPartial(modes)
        r = np.array([[0, 1, 2], [3, 4, 5]])
        x = b(r, 1, "parity", material=(5, 5), modetype="down")
        y = sc.vpw_N(
            [0, 0], [3, 4], [-4, -3], r[..., None, 0], r[..., None, 1], r[..., None, 2]
        )
        assert np.all(y == x)

    def test_call_invalid(self):
        b = ptsa.PlaneWaveBasisPartial([[1, 0, 0]])
        with pytest.raises(ValueError):
            b([1, 0, 0], 1, modetype="asdf")

    def test_permute(self):
        a = ptsa.PlaneWaveBasisPartial([[1, 2, 1], [1, 2, 0]], "yz")
        b = ptsa.PlaneWaveBasisPartial([[1, 2, 1], [1, 2, 0]])
        assert b.permute() == a

    def test_diffr_orders(self):
        lattice = ptsa.Lattice(2 * np.pi * np.eye(2))
        b = ptsa.PlaneWaveBasisPartial.diffr_orders([0, 0], lattice, 1)
        a = ptsa.PlaneWaveBasisPartial.default(
            [[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]]
        )
        assert (
            a <= b
            and b <= a
            and b.hints["lattice"] == lattice
            and b.hints["kpar"] == [0, 0, np.nan]
        )

    def test_rotate(self):
        b = ptsa.PlaneWaveBasisPartial([[1, 0, 0], [0, 1, 0]])
        a = ptsa.PlaneWaveBasisPartial(
            [[np.cos(2), np.sin(2), 0], [-np.sin(2), np.cos(2), 0]]
        )
        where = [True, False]
        x = b.rotate(2, where=where)
        y = ptsa.PhysicsArray([[1, 0], [0, 0]], basis=(a, b))
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_translate(self):
        a = ptsa.PlaneWaveBasisPartial([[4, 0, 0]])
        b = ptsa.PlaneWaveBasisPartial([[4, 0, 0], [4, 1, 0]])
        where = [True, False]
        x = b.translate([[0, 0, 0], [0, 1, 1]], 5, basis=a, where=where)
        y = ptsa.PhysicsArray(
            [
                [[ptsa.pw.translate(4, 0, 3, 0, 0, 0), 0]],
                [[ptsa.pw.translate(4, 0, 3, 0, 1, 1), 0]],
            ],
            basis=(None, a, b),
            material=(None, ptsa.Material(), ptsa.Material()),
            k0=(None, 5, 5),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    # def test_expand_pw(self):
    #     a = ptsa.PlaneWaveBasis([[0.2, 0, 0, 0], [0, 1, 1, 0]])
    #     b = ptsa.PlaneWaveBasis([[0.2, 0, 0, 0], [0.2, 1, 0, 0]])
    #     where = [True, False]
    #     x = b.expand(3, a, material=(1, 2, 0), modetype="singular", where=where)
    #     y = ptsa.PhysicsArray(
    #         [[1, 0], [0, 0]], basis=(a, b), k0=3, material=ptsa.Material(1, 2, 0),
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    # def test_expand_cw(self):
    #     a = ptsa.CylindricalWaveBasis([[3, 1, 0]], [1, 2, 3])
    #     b = ptsa.PlaneWaveBasis([[0, 4, 3, 0], [0, 4, 3, 1]])
    #     where = [True, False]
    #     x = b.expand(5, a, where=where)
    #     y = ptsa.PhysicsArray(
    #         [
    #             [
    #                 ptsa.pw.to_cw(3, 1, 0, 0, 4, 3, 0)
    #                 * ptsa.pw.translate(0, 4, 3, 1, 2, 3),
    #                 0,
    #             ]
    #         ],
    #         basis=(a, b),
    #         poltype="helicity",
    #         k0=5,
    #         material=ptsa.Material(),
    #         modetype=("regular", None),
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    # def test_expand_sw(self):
    #     a = ptsa.SphericalWaveBasis([[3, 1, 0]], [1, 2, 3])
    #     b = ptsa.PlaneWaveBasis([[0, 4, 3, 0], [0, 4, 3, 1]])
    #     where = [True, False]
    #     x = b.expand(5, a, where=where)
    #     y = ptsa.PhysicsArray(
    #         [
    #             [
    #                 ptsa.pw.to_sw(3, 1, 0, 0, 4, 3, 0)
    #                 * ptsa.pw.translate(0, 4, 3, 1, 2, 3),
    #                 0,
    #             ]
    #         ],
    #         basis=(a, b),
    #         poltype="helicity",
    #         k0=5,
    #         material=ptsa.Material(),
    #         modetype=("regular", None),
    #     )
    #     assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann

    def test_basischange(self):
        b = ptsa.PlaneWaveBasisPartial([[2, 0, 0], [1, 0, 1]])
        where = [True, False]
        x = b.basischange(where=where)
        y = ptsa.PhysicsArray(
            [[-np.sqrt(0.5), 0], [0, 0]], basis=(b, b), poltype=("helicity", "parity"),
        )
        assert np.all(np.abs(x - y) < 1e-14) and x.ann == y.ann
