import numpy as np

from treams import misc


class TestRefractiveIndex:
    def test_single(self):
        assert np.all(misc.refractive_index() == [1, 1])

    def test_multiple(self):
        expect = [
            [[0.5, 1.5], [-np.sqrt(2) + 1j, np.sqrt(2) + 1j]],
            [[np.sqrt(3) - 0.5, np.sqrt(3) + 0.5], [1j, 3j]],
        ]
        assert np.all(
            misc.refractive_index(mu=[[1, 2], [3, -4 + 0j]], kappa=[0.5, 1j]) == expect
        )


class TestBasisChange:
    def test(self):
        assert np.all(
            misc.basischange(([1, 1], [0, 1]))
            == np.sqrt(0.5) * np.array([[-1, 1], [1, 1]])
        )


class TestPickModes:
    def test(self):
        assert np.all(misc.pickmodes(([1, 1], [0, 1]), ([1], [0])) == [[1], [0]])


class TestWaveVecZ:
    def test(self):
        assert misc.wave_vec_z(3, 4, 5) == 0

    def test_neg_single(self):
        assert misc.wave_vec_z(0, 0, 2 - 2j) == -2 + 2j

    def test_multiple(self):
        assert np.all(misc.wave_vec_z([3, 3], [0, 5], 5) == [4, 3j])


class TestFirstBrillouin1d:
    def test(self):
        assert np.abs(misc.firstbrillouin1d(0.81, 0.5) + 0.19) < 1e-16

    def test_edge(self):
        assert misc.firstbrillouin1d(-0.25, 0.5) == 0.25


class TestFirstBrillouin2d:
    def test(self):
        assert np.all(
            np.abs(
                misc.firstbrillouin2d([1.81, -0.25], 0.5 * np.eye(2)) - [-0.19, -0.25]
            )
            < 1e-16
        )


class TestFirstBrillouin3d:
    def test(self):
        assert np.all(
            np.abs(
                misc.firstbrillouin3d([1.81, -0.25, 0], 0.5 * np.eye(3))
                - [-0.19, -0.25, 0]
            )
            < 1e-16
        )
