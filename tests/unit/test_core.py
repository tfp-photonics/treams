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

    def test_getitem_plms(self):
        b = treams.SphericalWaveBasis([[0, 2, -1, 1]])
        assert b.plms == ([0], [2], [-1], [1])

    def test_getitem_lms(self):
        b = treams.SphericalWaveBasis([[2, -1, 1]])
        assert b.lms == ([2], [-1], [1])

    def test_getitem_invalid_index(self):
        b = treams.SphericalWaveBasis([[1, 0, 0]])
        with pytest.raises(AttributeError):
            b.fail

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

    def test_getitem_pzms(self):
        b = treams.CylindricalWaveBasis([[0, 2, -1, 1]])
        assert b.pzms == ([0], [2], [-1], [1])

    def test_getitem_zms(self):
        b = treams.CylindricalWaveBasis([[2, -1, 1]])
        assert b.zms == ([2], [-1], [1])

    def test_getitem_invalid_index(self):
        b = treams.CylindricalWaveBasis([[1, 0, 0]])
        with pytest.raises(AttributeError):
            b.fail

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
            and a.lattice == treams.Lattice(2 * np.pi)
            and a.kpar == [np.nan, np.nan, 0.1]
        )


class TestPWBUV:
    def test_init_empty(self):
        b = treams.PlaneWaveBasisByUnitVector([])
        assert b.qx.size == 0 and b.qy.size == 0 and b.qz.size == 0 and b.pol.size == 0

    def test_init_numpy(self):
        b = treams.PlaneWaveBasisByUnitVector(np.array([[0.1, 0.2, 0.2, 0]]))
        assert (
            np.all(b.qx == [1 / 3])
            and np.all(b.qy == [2 / 3])
            and np.all(b.qz == [2 / 3])
            and np.all(b.pol == [0])
        )

    def test_init_duplicate(self):
        b = treams.PlaneWaveBasisByUnitVector([[0.1, 0.2, 0.2, 0], [0.1, 0.2, 0.2, 0]])
        assert (
            np.all(b.qx == [1 / 3])
            and np.all(b.qy == [2 / 3])
            and np.all(b.qz == [2 / 3])
            and np.all(b.pol == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            treams.PlaneWaveBasisByUnitVector([[0, 0], [0, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            treams.PlaneWaveBasisByUnitVector([[1, 0, 0, 2]])

    def test_repr(self):
        b = treams.PlaneWaveBasisByUnitVector([[0.0, 1.0, 0.0, 0], [1, 0, 0, 0]])
        assert (
            repr(b)
            == """PlaneWaveBasisByUnitVector(
    qx=[0. 1.],
    qy=[1. 0.],
    qz=[0. 0.],
    pol=[0 0],
)"""
        )

    def test_getitem_xyzs(self):
        b = treams.PlaneWaveBasisByUnitVector([[0, 4, -3, 1]])
        assert b.xyzs == ([0], [0.8], [-0.6], [1])

    def test_getitem_xys(self):
        b = treams.PlaneWaveBasisByUnitVector([[0, 4, -3, 1]])
        assert b.xys == ([0], [0.8], [1])

    def test_getitem_invalid_index(self):
        b = treams.PlaneWaveBasisByUnitVector([[1, 0, 0, 0]])
        with pytest.raises(AttributeError):
            b.fail

    def test_getitem_int(self):
        b = treams.PlaneWaveBasisByUnitVector([[1, 0, 0, 0], [1, 0, 0, 1]])
        assert b[1] == (1, 0, 0, 1)

    def test_getitem_tuple(self):
        b = treams.PlaneWaveBasisByUnitVector([[1, 0, 0, 0], [1, 0, 0, 1]])
        assert (np.array(b[()]) == ([1, 1], [0, 0], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = treams.PlaneWaveBasisByUnitVector([[1, 0, 0, 0]])
        b = treams.PlaneWaveBasisByUnitVector([[1, 0, 0, 0], [1, 0, 0, 1]])
        assert a == b[:1]

    def test_default(self):
        a = treams.PlaneWaveBasisByUnitVector.default([0.3, -0.2, 0.1])
        b = treams.PlaneWaveBasisByUnitVector(
            [[0.3, -0.2, 0.1, 1], [0.3, -0.2, 0.1, 0]]
        )
        assert a == b

    def test_property_isglobal_true(self):
        b = treams.PlaneWaveBasisByUnitVector([])
        assert b.isglobal

    def test_from_iterable(self):
        a = treams.PlaneWaveBasisByUnitVector.default([0, 0, 1])
        b = treams.PlaneWaveBasisByUnitVector.default([[0, 0, 1], [0, 1, 0]])
        assert a & b == a

    def test_bycomp(self):
        a = treams.PlaneWaveBasisByComp.default([0, 1], "yz")
        b = treams.PlaneWaveBasisByUnitVector.default([0, 0, 1])
        assert b.bycomp(1, "yz") == a


class TestPWBC:
    def test_init_empty(self):
        b = treams.PlaneWaveBasisByComp([])
        assert b.kx.size == 0 and b.ky.size == 0 and b.kz is None and b.pol.size == 0

    def test_init_numpy(self):
        b = treams.PlaneWaveBasisByComp(np.array([[0.4, 0.2, 0]]), "zx")
        assert (
            np.all(b.kz == [0.4])
            and np.all(b.kx == [0.2])
            and np.all(b.ky is None)
            and np.all(b.pol == [0])
        )

    def test_init_duplicate(self):
        b = treams.PlaneWaveBasisByComp([[0.4, 0.2, 0], [0.4, 0.2, 0]])
        assert (
            np.all(b.kx == [0.4])
            and np.all(b.ky == [0.2])
            and np.all(b.kz is None)
            and np.all(b.pol == [0])
        )

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            treams.PlaneWaveBasisByComp([[0, 0], [0, 0]])

    def test_init_invalid_pol(self):
        with pytest.raises(ValueError):
            treams.PlaneWaveBasisByComp([[1, 0, 2]])

    def test_repr(self):
        b = treams.PlaneWaveBasisByComp([[0.0, 1.0, 0], [1, 1, 0]], "yz")
        assert (
            repr(b)
            == """PlaneWaveBasisByComp(
    ky=[0. 1.],
    kz=[1. 1.],
    pol=[0 0],
)"""
        )

    def test_from_iterable(self):
        a = treams.PlaneWaveBasisByComp.default([0, 0])
        b = treams.PlaneWaveBasisByComp.default([[0, 0], [0, 1]])
        assert a & b == a

    def test_getitem_int(self):
        b = treams.PlaneWaveBasisByComp([[1, 0, 0], [1, 0, 1]])
        assert b[1] == (1, 0, 1)

    def test_getitem_tuple(self):
        b = treams.PlaneWaveBasisByComp([[1, 0, 0], [1, 0, 1]])
        assert (np.array(b[()]) == ([1, 1], [0, 0], [0, 1])).all()

    def test_getitem_slice(self):
        a = treams.PlaneWaveBasisByComp([[1, 0, 0]])
        b = treams.PlaneWaveBasisByComp([[1, 0, 0], [1, 0, 0, 1]])
        assert a == b[:1]

    def test_byunitvector(self):
        a = treams.PlaneWaveBasisByUnitVector.default([0, 0, 1])
        b = treams.PlaneWaveBasisByComp.default([0, 1], "yz")
        assert b.byunitvector(1) == a

    def test_diffr_orders(self):
        lattice = treams.Lattice(2 * np.pi * np.eye(2))
        b = treams.PlaneWaveBasisByComp.diffr_orders([0, 0], lattice, 1)
        a = treams.PlaneWaveBasisByComp.default(
            [[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]]
        )
        assert a <= b and b <= a and b.lattice == lattice and b.kpar == [0, 0, np.nan]


class TestPhysicsArray:
    def test_init(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray(np.eye(2), basis=b)
        assert (p == np.eye(2)).all() and p.basis == b

    def test_type_error(self):
        with pytest.raises(TypeError):
            treams.PhysicsArray(np.eye(2), basis="fail")

    def test_lattice(self):
        b = treams.PlaneWaveBasisByComp.diffr_orders([0, 0], np.eye(2), 4)
        p = treams.PhysicsArray([1, 2], lattice=treams.Lattice(1, "x"), basis=b)
        assert p.lattice == treams.Lattice(1, "x")

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
        x = p.changepoltype.apply_left()
        assert (
            x == np.sqrt(0.5) * np.array([[-1, 1], [1, 1]])
        ).all() and x.poltype == (
            "parity",
            "helicity",
        )

    def test_changepoltype_inv(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([[1, 0], [0, 1]], basis=b, poltype="helicity")
        x = p.changepoltype.apply_right()
        assert (
            x == np.sqrt(0.5) * np.array([[-1, 1], [1, 1]])
        ).all() and x.poltype == (
            "helicity",
            "parity",
        )

    def test_efield(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1, poltype="helicity")
        x = p.efield([1, 0, 0])
        assert (
            np.abs(
                x
                - (
                    treams.special.vsph2car(
                        treams.special.vsw_rA(1, 0, 1, np.pi / 2, 0, [0, 1]),
                        [1, np.pi / 2, 0],
                    ).sum(0)
                )
                < 1e-15
            )
        ).all()

    def test_efield_inv(self):
        with pytest.raises(NotImplementedError):
            treams.PhysicsArray([0]).efield.apply_right([0, 0, 0])

    def test_expand(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1, poltype="helicity")
        x = p.expand(b)
        assert (np.abs(x - [1, 0]) < 1e-14).all()

    def test_expand_inv(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1, poltype="helicity")
        x = p.expand.eval_inv(b)
        assert (np.abs(x - np.eye(2)) < 1e-14).all()

    def test_expandlattice(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1, poltype="helicity")
        x = p.expandlattice.eval(lattice=[[1, 0], [0, 1]], kpar=[0, 0])
        assert x.modetype == ("regular", "singular")

    def test_expandlattice_inv(self):
        with pytest.raises(NotImplementedError):
            treams.PhysicsArray([0]).expandlattice.eval_inv(lattice=1)

    def test_permute(self):
        b = treams.PlaneWaveBasisByUnitVector([[1, 0, 0, 0], [1, 0, 0, 1]])
        c = treams.PlaneWaveBasisByUnitVector([[0, 1, 0, 0], [0, 1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b)
        assert p.permute.eval().basis == (c, b)

    def test_permute_inv(self):
        b = treams.PlaneWaveBasisByUnitVector([[1, 0, 0, 0], [1, 0, 0, 1]])
        c = treams.PlaneWaveBasisByUnitVector([[0, 1, 0, 0], [0, 1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b)
        assert p.permute.eval_inv().basis == (b, c)

    def test_rotate(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b)
        x = p.rotate.eval(1, 2, 3)
        y = np.diag([treams.special.wignerd(1, 0, 0, 1, 2, 3)] * 2)
        assert (np.abs(x - y) < 1e-14).all()

    def test_rotate_inv(self):
        b = treams.SphericalWaveBasis([[1, -1, 0], [1, 0, 0], [1, 1, 0]])
        p = treams.PhysicsArray([1, 0, 0], basis=b)
        x = p.rotate.eval_inv(1, 2, 3)
        y = treams.special.wignerd(1, [[-1], [0], [1]], [-1, 0, 1], 1, 2, 3)
        assert (np.abs(x - y.conj().T) < 1e-14).all()

    def test_translate(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1)
        x = p.translate.eval([0, 0, 0])
        assert (np.abs(x - np.eye(2)) < 1e-14).all()

    def test_translate_inv(self):
        b = treams.SphericalWaveBasis([[1, 0, 0], [1, 0, 1]])
        p = treams.PhysicsArray([1, 0], basis=b, k0=1)
        x = p.translate.eval_inv([0, 0, 0])
        assert (np.abs(x - np.eye(2)) < 1e-14).all()
