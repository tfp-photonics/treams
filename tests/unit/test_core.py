import numpy as np
import pytest

import treams


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class TestSWB:
    def test_init_empty(self):
        b = treams.SphericalWaveBasis([])
        assert b.l.size == 0 and b.m.size == 0 and b.pol.size == 0 and b.pidx.size == 0

    def test_init_numpy(self):
        b = treams.SphericalWaveBasis(np.array([[1, 0, 0]]), [0, 1, 0])
        assert (
            np.all(b.l == [1])
            and np.all(b.m == [0])
            and np.all(b.pol == [0])
            and np.all(b.pidx == [0])
        )

    def test_init_duplicate(self):
        b = treams.SphericalWaveBasis([[0, 1, 0, 0], [0, 1, 0, 0]])
        assert (
            np.all(b.l == [1])
            and np.all(b.m == [0])
            and np.all(b.pol == [0])
            and np.all(b.pidx == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            treams.SphericalWaveBasis([[0, 0], [0, 0]])

    def test_init_invalid_positions(self):
        with pytest.raises(ValueError):
            treams.SphericalWaveBasis([[1, 0, 0]], [1, 2])

    def test_init_non_int_value(self):
        with pytest.raises(ValueError):
            treams.SphericalWaveBasis([[1.1, 0, 0]])

    def test_init_non_natural_l(self):
        with pytest.raises(ValueError):
            treams.SphericalWaveBasis([[0, 0, 0]])

    def test_init_non_too_large_m(self):
        with pytest.raises(ValueError):
            treams.SphericalWaveBasis([[1, -2, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            treams.SphericalWaveBasis([[1, 0, 2]])

    def test_init_unsecified_positions(self):
        with pytest.raises(ValueError):
            treams.SphericalWaveBasis([[1, 1, 0, 0]])

    def test_property_positions(self):
        a = np.array([[1, 2, 3]])
        b = treams.SphericalWaveBasis([[1, 0, 0]], a)
        assert (a == b.positions).all()

    def test_repr(self):
        b = treams.SphericalWaveBasis(
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
        b = treams.SphericalWaveBasis([[0, 2, -1, 1]])
        assert b["plmp"] == ([0], [2], [-1], [1])

    def test_getitem_lmp(self):
        b = treams.SphericalWaveBasis([[2, -1, 1]])
        assert b["lmp"] == ([2], [-1], [1])

    def test_getitem_lm(self):
        b = treams.SphericalWaveBasis([[2, -1, 1]])
        assert b["LM"] == ([2], [-1])

    def test_getitem_invalid_index(self):
        b = treams.SphericalWaveBasis([[1, 0, 0]])
        with pytest.raises(IndexError):
            b["l"]

    def test_getitem_int(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert b[1] == (0, 1, 0, 1)

    def test_getitem_tuple(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert (np.array(b[()]) == ([0, 0], [1, 1], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = treams.SphericalWaveBasis([[1, 0, 0]])
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert a == b[:1]

    def test_default(self):
        a = treams.SphericalWaveBasis.default(2, 2, [[0, 0, 0], [1, 0, 0]])
        b = treams.SphericalWaveBasis(
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
        a = treams.SphericalWaveBasis.ebcm(2, 2, positions=[[0, 0, 0], [1, 0, 0]])
        b = treams.SphericalWaveBasis(
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
        a = treams.SphericalWaveBasis.ebcm(2, mmax=1)
        b = treams.SphericalWaveBasis(
            zip(
                3 * [1, 1, 2, 2],
                [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1],
                6 * [1, 0],
            ),
        )
        assert a == b

    def test_defaultlmax(self):
        assert treams.SphericalWaveBasis.defaultlmax(60, 2) == 3

    def test_defaultlmax_fail(self):
        with pytest.raises(ValueError):
            treams.SphericalWaveBasis.defaultlmax(1)

    def test_defaultdim(self):
        assert treams.SphericalWaveBasis.defaultdim(3, 2) == 60

    def test_defaultdim_fail(self):
        with pytest.raises(ValueError):
            treams.SphericalWaveBasis.defaultdim(1, -1)

    def test_property_isglobal_true(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert b.isglobal

    def test_property_isglobal_false(self):
        b = treams.SphericalWaveBasis(
            [[0, 1, 0, 0], [1, 1, 0, 0]], [[0, 0, 0], [1, 0, 0]]
        )
        assert not b.isglobal

    def test_from_iterable(self):
        a = treams.SphericalWaveBasis.default(1)
        b = treams.SphericalWaveBasis.default(2)
        assert a & b == a

    def test_neq(self):
        b = treams.SphericalWaveBasis.default(1)
        assert not b == []


class TestCWB:
    def test_init_empty(self):
        b = treams.CylindricalWaveBasis([])
        assert b.kz.size == 0 and b.m.size == 0 and b.pol.size == 0 and b.pidx.size == 0

    def test_init_numpy(self):
        b = treams.CylindricalWaveBasis(np.array([[0.5, 0, 0]]), [0, 1, 0])
        assert (
            np.all(b.kz == [0.5])
            and np.all(b.m == [0])
            and np.all(b.pol == [0])
            and np.all(b.pidx == [0])
        )

    def test_init_duplicate(self):
        b = treams.CylindricalWaveBasis([[0, 1, 0, 0], [0, 1, 0, 0]])
        assert (
            np.all(b.kz == [1])
            and np.all(b.m == [0])
            and np.all(b.pol == [0])
            and np.all(b.pidx == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            treams.CylindricalWaveBasis([[0, 0], [0, 0]])

    def test_init_invalid_positions(self):
        with pytest.raises(ValueError):
            treams.CylindricalWaveBasis([[1, 0, 0]], [1, 2])

    def test_init_non_int_value(self):
        with pytest.raises(ValueError):
            treams.CylindricalWaveBasis([[1, 0.2, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            treams.CylindricalWaveBasis([[1, 0, 2]])

    def test_init_unsecified_positions(self):
        with pytest.raises(ValueError):
            treams.CylindricalWaveBasis([[1, 1, 0, 0]])

    def test_property_positions(self):
        a = np.array([[1, 2, 3]])
        b = treams.CylindricalWaveBasis([[1, 0, 0]], a)
        assert (a == b.positions).all()

    def test_repr(self):
        b = treams.CylindricalWaveBasis(
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
        b = treams.CylindricalWaveBasis([[0, 2, -1, 1]])
        assert b["pkzmp"] == ([0], [2], [-1], [1])

    def test_getitem_kzmp(self):
        b = treams.CylindricalWaveBasis([[2, -1, 1]])
        assert b["kzmp"] == ([2], [-1], [1])

    def test_getitem_kzm(self):
        b = treams.CylindricalWaveBasis([[2, -1, 1]])
        assert b["KZM"] == ([2], [-1])

    def test_getitem_invalid_index(self):
        b = treams.CylindricalWaveBasis([[1, 0, 0]])
        with pytest.raises(IndexError):
            b["l"]

    def test_getitem_int(self):
        b = treams.CylindricalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert b[1] == (0, 1, 0, 1)

    def test_getitem_tuple(self):
        b = treams.CylindricalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert (np.array(b[()]) == ([0, 0], [1, 1], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = treams.CylindricalWaveBasis([[1, 0, 0]])
        b = treams.CylindricalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert a == b[:1]

    def test_default(self):
        a = treams.CylindricalWaveBasis.default(
            [0.3, -0.2], 2, 2, [[0, 0, 0], [1, 0, 0]]
        )
        b = treams.CylindricalWaveBasis(
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
        assert treams.CylindricalWaveBasis.defaultmmax(112, 4, 2) == 3

    def test_defaultmmax_fail(self):
        with pytest.raises(ValueError):
            treams.CylindricalWaveBasis.defaultmmax(1)

    def test_defaultdim(self):
        assert treams.CylindricalWaveBasis.defaultdim(3, 2, 4) == 120

    def test_defaultdim_fail(self):
        with pytest.raises(ValueError):
            treams.CylindricalWaveBasis.defaultdim(1, -1)

    def test_property_isglobal_true(self):
        b = treams.CylindricalWaveBasis([[1, 0, 0], [1, 0, 1]])
        assert b.isglobal

    def test_property_isglobal_false(self):
        b = treams.CylindricalWaveBasis(
            [[0, 1, 0, 0], [1, 1, 0, 0]], [[0, 0, 0], [1, 0, 0]]
        )
        assert not b.isglobal

    def test_from_iterable(self):
        a = treams.CylindricalWaveBasis.default(0, 1)
        b = treams.CylindricalWaveBasis.default(0, 2)
        assert a & b == a

    def test_neq(self):
        b = treams.CylindricalWaveBasis.default(0, 1)
        assert not b == []

    def test_diffr_orders(self):
        a = treams.CylindricalWaveBasis.diffr_orders(0.1, 1, 2 * np.pi, 1.5)
        b = treams.CylindricalWaveBasis(
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
            and a.hints["lattice"] == treams.Lattice(2 * np.pi)
            and a.hints["kpar"] == [np.nan, np.nan, 0.1]
        )


class TestPWB:
    def test_init_empty(self):
        b = treams.PlaneWaveBasis([])
        assert b.kx.size == 0 and b.ky.size == 0 and b.kz.size == 0 and b.pol.size == 0

    def test_init_numpy(self):
        b = treams.PlaneWaveBasis(np.array([[0.4, 0.2, 0.1, 0]]))
        assert (
            np.all(b.kx == [0.4])
            and np.all(b.ky == [0.2])
            and np.all(b.kz == [0.1])
            and np.all(b.pol == [0])
        )

    def test_init_duplicate(self):
        b = treams.PlaneWaveBasis([[0.4, 0.2, 0.1, 0], [0.4, 0.2, 0.1, 0]])
        assert (
            np.all(b.kx == [0.4])
            and np.all(b.ky == [0.2])
            and np.all(b.kz == [0.1])
            and np.all(b.pol == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            treams.PlaneWaveBasis([[0, 0], [0, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            treams.PlaneWaveBasis([[1, 0, 0, 2]])

    def test_repr(self):
        b = treams.PlaneWaveBasis([[0.0, 1.0, 0.0, 0], [1, 1, 0, 0]])
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
        b = treams.PlaneWaveBasis([[0, 2, -1, 1]])
        assert b["xyzp"] == ([0], [2], [-1], [1])

    def test_getitem_xyp(self):
        b = treams.PlaneWaveBasis([[0, 2, -1, 1]])
        assert b["xyp"] == ([0], [2], [1])

    def test_getitem_zp(self):
        b = treams.PlaneWaveBasis([[0, 2, -1, 1]])
        assert b["ZP"] == ([-1], [1])

    def test_getitem_invalid_index(self):
        b = treams.PlaneWaveBasis([[1, 0, 0, 0]])
        with pytest.raises(IndexError):
            b["l"]

    def test_getitem_int(self):
        b = treams.PlaneWaveBasis([[1, 0, 0, 0], [1, 0, 0, 1]])
        assert b[1] == (1, 0, 0, 1)

    def test_getitem_tuple(self):
        b = treams.PlaneWaveBasis([[1, 0, 0, 0], [1, 0, 0, 1]])
        assert (np.array(b[()]) == ([1, 1], [0, 0], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = treams.PlaneWaveBasis([[1, 0, 0, 0]])
        b = treams.PlaneWaveBasis([[1, 0, 0, 0], [1, 0, 0, 1]])
        assert a == b[:1]

    def test_default(self):
        a = treams.PlaneWaveBasis.default([0.3, -0.2, 0.1])
        b = treams.PlaneWaveBasis([[0.3, -0.2, 0.1, 1], [0.3, -0.2, 0.1, 0]])
        assert a == b

    def test_property_isglobal_true(self):
        b = treams.PlaneWaveBasis([])
        assert b.isglobal

    def test_from_iterable(self):
        a = treams.PlaneWaveBasis.default([0, 0, 1])
        b = treams.PlaneWaveBasis.default([[0, 0, 1], [0, 1, 0]])
        assert a & b == a

    def test_partial(self):
        a = treams.PlaneWaveBasisPartial.default([0, 1], "yz")
        b = treams.PlaneWaveBasis.default([0, 0, 1])
        assert b.partial("yz", 1) == a


class TestPWBP:
    def test_init_empty(self):
        b = treams.PlaneWaveBasisPartial([])
        assert b.kx.size == 0 and b.ky.size == 0 and b.kz is None and b.pol.size == 0

    def test_init_numpy(self):
        b = treams.PlaneWaveBasisPartial(np.array([[0.4, 0.2, 0]]), "zx")
        assert (
            np.all(b.kz == [0.4])
            and np.all(b.kx == [0.2])
            and np.all(b.ky is None)
            and np.all(b.pol == [0])
        )

    def test_init_duplicate(self):
        b = treams.PlaneWaveBasisPartial([[0.4, 0.2, 0], [0.4, 0.2, 0]])
        assert (
            np.all(b.kx == [0.4])
            and np.all(b.ky == [0.2])
            and np.all(b.kz is None)
            and np.all(b.pol == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            treams.PlaneWaveBasisPartial([[0, 0], [0, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            treams.PlaneWaveBasisPartial([[1, 0, 2]])

    def test_repr(self):
        b = treams.PlaneWaveBasisPartial([[0.0, 1.0, 0], [1, 1, 0]], "yz")
        assert (
            repr(b)
            == """PlaneWaveBasisPartial(
    ky=[0. 1.],
    kz=[1. 1.],
    pol=[0 0],
)"""
        )

    def test_from_iterable(self):
        a = treams.PlaneWaveBasisPartial.default([0, 0])
        b = treams.PlaneWaveBasisPartial.default([[0, 0], [0, 1]])
        assert a & b == a

    def test_getitem_int(self):
        b = treams.PlaneWaveBasisPartial([[1, 0, 0], [1, 0, 1]])
        assert b[1] == (1, 0, 1)

    def test_getitem_tuple(self):
        b = treams.PlaneWaveBasisPartial([[1, 0, 0], [1, 0, 1]])
        assert (np.array(b[()]) == ([1, 1], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = treams.PlaneWaveBasisPartial([[1, 0, 0]])
        b = treams.PlaneWaveBasisPartial([[1, 0, 0], [1, 0, 0, 1]])
        assert a == b[:1]

    def test_complete(self):
        a = treams.PlaneWaveBasis.default([0, 0, 1])
        b = treams.PlaneWaveBasisPartial.default([0, 1], "yz")
        assert b.complete(1) == a

    def test_diffr_orders(self):
        lattice = treams.Lattice(2 * np.pi * np.eye(2))
        b = treams.PlaneWaveBasisPartial.diffr_orders([0, 0], lattice, 1)
        a = treams.PlaneWaveBasisPartial.default(
            [[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]]
        )
        assert (
            a <= b
            and b <= a
            and b.hints["lattice"] == lattice
            and b.hints["kpar"] == [0, 0, np.nan]
        )


class TestPhysicsArray:
    def test_init(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray(np.eye(2), basis=b)
        assert (p == np.eye(2)).all() and p.basis == b

    def test_type_error(self):
        with pytest.raises(TypeError):
            treams.PhysicsArray(np.eye(2), basis="fail")

    def test_lattice(self):
        b = treams.PlaneWaveBasisPartial.diffr_orders([0, 0], np.eye(2), 4)
        p = treams.PhysicsArray([1, 2], lattice=treams.Lattice(1, "x"), basis=b)
        assert p.lattice == treams.Lattice(1, "x")

    def test_lattice_fail(self):
        b = treams.PlaneWaveBasisPartial.diffr_orders([0, 0], np.eye(2), 4)
        with pytest.raises(ValueError):
            treams.PhysicsArray([1, 2], lattice=treams.Lattice(2, "x"), basis=b)

    def test_matmul(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([[1, 1], [2, 2]], basis=b)
        x = p @ [1, 2]
        assert (x == [3, 6]).all() and x.basis == b

    def test_rmatmul(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([[1, 1], [2, 2]], basis=b)
        x = [1, 2] @ p
        assert (x == [5, 5]).all() and x.basis == b

    def test_changepoltype(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([[1, 0], [0, 1]], basis=b, poltype="helicity")
        x = p.changepoltype()
        assert (
            x == np.sqrt(0.5) * np.array([[-1, 1], [1, 1]])
        ).all() and x.poltype == ("parity", "helicity")

    def test_changepoltype_inv(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([[1, 0], [0, 1]], basis=b, poltype="helicity")
        x = p.changepoltype.inv()
        assert (
            x == np.sqrt(0.5) * np.array([[-1, 1], [1, 1]])
        ).all() and x.poltype == ("helicity", "parity")

    def test_efield(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1, poltype="helicity")
        x = p.efield([1, 0, 0])
        assert (
            x
            == (
                treams.special.vsph2car(
                    treams.special.vsw_rA(1, 0, 1, np.pi / 2, 0, [0, 1]),
                    [1, np.pi / 2, 0],
                )
            )
        ).all()

    def test_efield_inv(self):
        with pytest.raises(NotImplementedError):
            treams.PhysicsArray([0]).efield.inv([0, 0, 0])

    def test_expand(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1, poltype="helicity")
        x = p.expand()
        assert (np.abs(x - np.eye(2)) < 1e-14).all()

    def test_expand_inv(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1, poltype="helicity")
        x = p.expand.inv()
        assert (np.abs(x - np.eye(2)) < 1e-14).all()

    def test_expandlattice(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1, poltype="helicity")
        x = p.expandlattice(lattice=[[1, 0], [0, 1]])
        assert x.modetype == ("regular", "singular")

    def test_expandlattice_inv(self):
        with pytest.raises(NotImplementedError):
            treams.PhysicsArray([0]).expandlattice.inv(lattice=1)

    def test_permute(self):
        b = treams.PlaneWaveBasis([[1, 0, 0, 0], [1, 0, 0, 1]])
        c = treams.PlaneWaveBasis([[0, 1, 0, 0], [0, 1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b)
        assert p.permute().basis == (c, b)

    def test_permute_inv(self):
        b = treams.PlaneWaveBasis([[1, 0, 0, 0], [1, 0, 0, 1]])
        c = treams.PlaneWaveBasis([[0, 1, 0, 0], [0, 1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b)
        assert p.permute.inv().basis == (b, c)

    def test_rotate(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b)
        x = p.rotate(1, 2, 3)
        y = np.diag([treams.special.wignerd(1, 0, 0, 1, 2, 3)] * 2)
        assert (np.abs(x - y) < 1e-14).all()

    def test_rotate_inv(self):
        b = treams.SphericalWaveBasis([[1, -1, 0], [1, 0, 0], [1, 1, 0]])
        p = treams.PhysicsArray([1, 0, 0], basis=b)
        x = p.rotate.inv(1, 2, 3)
        y = treams.special.wignerd(1, [[-1], [0], [1]], [-1, 0, 1], 1, 2, 3)
        assert (np.abs(x - y.conj().T) < 1e-14).all()

    def test_translate(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1)
        x = p.translate([0, 0, 0])
        assert (np.abs(x - np.eye(2)) < 1e-14).all()

    def test_translate_inv(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1)
        x = p.translate.inv([0, 0, 0])
        assert (np.abs(x - np.eye(2)) < 1e-14).all()
