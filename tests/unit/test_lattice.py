import numpy as np
import pytest

from treams import Lattice, WaveVector


class TestLattice:
    def test_init(self):
        assert Lattice([1]) == Lattice(1, alignment="z")

    def test_init_lattice(self):
        assert Lattice(Lattice([1, 2])) == Lattice([[1, 0], [0, 2]])

    def test_init_invalid_shape(self):
        with pytest.raises(ValueError):
            Lattice([0, 1, 2, 3])

    def test_init_invalid_alignment(self):
        with pytest.raises(ValueError):
            Lattice(1, "a")

    def test_init_volume(self):
        with pytest.raises(ValueError):
            Lattice([[1, 2], [2, 4]])

    def test_square(self):
        assert Lattice.square(1) == Lattice([1, 1])

    def test_cubic(self):
        assert Lattice.cubic(1) == Lattice([1, 1, 1])

    def test_rectangular(self):
        assert Lattice.rectangular(1, 2) == Lattice([1, 2])

    def test_orthorhombic(self):
        assert Lattice.orthorhombic(1, 2, 3) == Lattice([1, 2, 3])

    def test_hexagonal_2d(self):
        assert Lattice.hexagonal(1) == Lattice([[1, 0], [0.5, np.sqrt(0.75)]])

    def test_hexagonal_3d(self):
        assert Lattice.hexagonal(1, 2) == Lattice(
            [[1, 0, 0], [0.5, 0.75**0.5, 0], [0, 0, 2]]
        )

    def test_reciprocal(self):
        assert (Lattice([np.pi, 2 * np.pi]).reciprocal == [[2, 0], [0, 1]]).all()

    def test_str(self):
        assert str(Lattice(1)) == "1.0"

    def test_repr(self):
        assert (
            repr(Lattice([1, 2]))
            == """Lattice([[1. 0.]
         [0. 2.]], alignment='xy')"""
        )

    def test_sublattice(self):
        assert Lattice(Lattice([1, 2, 3]), "xy") == Lattice([1, 2])

    def test_sublattice_1d(self):
        assert Lattice(Lattice(1), "z") == Lattice(1)

    def test_sublattice_1d_error(self):
        with pytest.raises(ValueError):
            Lattice(Lattice(1), "x")

    def test_sublattice_2d_error(self):
        with pytest.raises(ValueError):
            Lattice(Lattice([1, 2], "zx"), "xy")

    def test_sublattice_find_error(self):
        with pytest.raises(ValueError):
            Lattice(Lattice([[1, 1], [0, 1]]), "x")

    def test_permute(self):
        assert Lattice([1, 2]).permute(4) == Lattice([1, 2], "yz")

    def test_permute_3d(self):
        assert Lattice([1, 2, 3]).permute(2.0) == Lattice(
            [[0, 0, 1], [2, 0, 0], [0, 3, 0]]
        )

    def test_permute_error(self):
        with pytest.raises(ValueError):
            Lattice(1).permute(0.2)

    def test_bool(self):
        assert bool(Lattice(1))

    def test_or(self):
        assert Lattice(1) | Lattice([2, 3]) == Lattice([2, 3, 1])

    def test_or_none(self):
        assert Lattice(1) | None == Lattice(1)

    def test_or_1d(self):
        assert Lattice(1) | Lattice(2, "x") == Lattice([1, 2], "zx")

    def test_or_1d_swapped(self):
        assert Lattice(1, "x") | Lattice(2) == Lattice([2, 1], "zx")

    def test_or_2d(self):
        assert Lattice([1, 2]) | Lattice(1, "x") == Lattice([1, 2])

    def test_or_2d_swapped(self):
        assert Lattice(1, "x") | Lattice([1, 2]) == Lattice([1, 2])

    def test_or_3d(self):
        assert Lattice([1, 2, 3]) | Lattice(1, "x") == Lattice([1, 2, 3])

    def test_or_3d_swapped(self):
        assert Lattice(2, "y") | Lattice([1, 2, 3]) == Lattice([1, 2, 3])

    def test_or_3d_2d(self):
        assert Lattice([1, 2]) | Lattice([2, 3], "yz") == Lattice([1, 2, 3])

    def test_or_3d_2d_error(self):
        with pytest.raises(ValueError):
            Lattice([1, 2]) | Lattice([3, 3], "yz")

    def test_or_3d_x(self):
        assert Lattice(1, "x") | Lattice([2, 3], "yz") == Lattice([1, 2, 3])

    def test_or_3d_y(self):
        assert Lattice(2, "y") | Lattice([3, 1], "zx") == Lattice([1, 2, 3])

    def test_or_error(self):
        with pytest.raises(ValueError):
            Lattice(1) | Lattice(2)

    def test_and_none(self):
        assert Lattice(1) & None is None

    def test_and_same(self):
        assert Lattice(1) & Lattice(1) == Lattice(1)

    def test_and_empty(self):
        with pytest.raises(ValueError):
            Lattice(1) & Lattice(2)

    def test_and_error(self):
        with pytest.raises(ValueError):
            Lattice(1) & Lattice(1, "x")

    def test_and(self):
        assert Lattice([1, 2, 3]) & Lattice([[0, 1], [3, 0]], "zx") == Lattice(
            [[0, 1], [3, 0]], "zx"
        )

    def test_le(self):
        assert Lattice(3) <= Lattice([1, 2, 3])

    def test_le_false(self):
        assert not Lattice(3) >= Lattice([1, 2, 3])

    def test_isdisjoint(self):
        assert Lattice(1).isdisjoint(Lattice([1, 1]))

    def test_isdisjoint_false(self):
        assert not Lattice(1).isdisjoint(Lattice([1, 1, 1]))
