import pytest
import numpy as np

import ptsa.coeffs as cf

EPS = 2e-7
EPSSQ = 4e-14


class TestMie:
    def test_real(self):
        expect = np.array(
            [
                [
                    -0.207488775205865 - 0.398855023857451j,
                    0.047334571416857 + 0.044794841521540j,
                ],
                [
                    0.059646659631622 + 0.056446326342700j,
                    -0.117865783454384 - 0.314040741356843j,
                ],
            ]
        )
        assert np.all(
            np.abs(
                cf.mie(1, [1, 2, 4], [1, 1.3, 3, 2], [1, 2, 1, 1.5], [0.1, 0, 0.2, 0.1])
                - expect
            )
            < EPSSQ
        )

    def test_complex(self):
        expect = np.array(
            [
                [
                    -0.233064414843298 + 0.218720052849983j,
                    0.047852376140501 + 0.048918498716278j,
                ],
                [
                    0.084573700327288 + 0.086457952239276j,
                    -0.619693739232935 + 0.081746714390217j,
                ],
            ]
        )
        assert np.all(
            np.abs(
                cf.mie(
                    3,
                    [1, 2, 3],
                    [1, 2, 3 + 1j, 4],
                    [4, 3 + 0.1j, 1, 2],
                    [0.3, -0.1 + 0.1j, 1 + 0.1j, 0.4],
                )
                - expect
            )
            < EPSSQ
        )


class TestFresnel:
    def test_real(self):
        expect = [
            [
                [[1.042449234640745, 0], [0, 1.005929062176551],],
                [[-0.042449234640745, 0], [0, -0.005929062176551],],
            ],
            [
                [[0.042449234640745, 0], [0, 0.005929062176551],],
                [[0.957550765359255, 0], [0, 0.994070937823449],],
            ],
        ]
        assert np.all(
            np.abs(
                cf.fresnel(
                    [[3, 5], [2, 4]],
                    [[np.sqrt(8), np.sqrt(24)], [np.sqrt(3), np.sqrt(15)]],
                    [1, 1],
                )
                - expect
            )
            < EPSSQ
        )

    def test_evanescent(self):
        expect = [
            [
                [
                    [1.6, 0],
                    [-0.3525 - 0.892941067484299j, 1.0125 - 1.488235112473832j],
                ],
                [[0, 0.6], [0.6, 0.235 + 0.595294044989533j],],
            ],
            [
                [
                    [0.1321875 + 0.334852900306612j, -0.3796875 + 0.558088167177687j],
                    [-0.8203125 - 0.558088167177687j, -0.3671875 - 0.930146945296145j],
                ],
                [
                    [0.4, -0.088125 - 0.223235266871075j],
                    [0, 0.546875 + 0.372058778118458j],
                ],
            ],
        ]
        assert np.all(
            np.abs(
                cf.fresnel(
                    [[3, 5], [3, 3]],
                    [[1j * np.sqrt(7), 3], [1j * np.sqrt(7), 1j * np.sqrt(7)]],
                    [1 / 4, 1],
                )
                - expect
            )
            < EPSSQ
        )

    def test_complex(self):
        expect = [
            [
                [
                    [
                        1.097225776644201 + 0.417220660903015j,
                        -0.091009961428571 + 0.126135520878490j,
                    ],
                    [
                        -0.001231951759500 + 0.002873123884810j,
                        1.344486739106106 + 0.035486557500712j,
                    ],
                ],
                [
                    [
                        0.150107671767365 - 0.265167179806910j,
                        0.338343409840136 + 0.025917960217615j,
                    ],
                    [
                        0.341187323894434 + 0.025964969567879j,
                        -0.004531366971172 - 0.006648464048023j,
                    ],
                ],
            ],
            [
                [
                    [
                        -0.174619475651097 + 0.295135397879729j,
                        -0.406453529679773 + 0.067329209352982j,
                    ],
                    [
                        -0.273077204054797 - 0.119212139138476j,
                        0.029043170854903 - 0.023319754024796j,
                    ],
                ],
                [
                    [
                        0.768675971525422 - 0.225752788034603j,
                        0.000510025554099 + 0.002053400492144j,
                    ],
                    [
                        -0.040244376347509 + 0.065379361340185j,
                        0.657635248742791 - 0.030513023773495j,
                    ],
                ],
            ],
        ]
        epsilon = np.array([4 + 1j, 3 + 0.1j])
        mu = np.array([1 + 0.1j, 3 + 0.01j])
        kappa = np.array([1 + 0.1j, -0.1])
        ks = np.stack(
            (np.sqrt(epsilon * mu) - kappa, np.sqrt(epsilon * mu) + kappa), axis=-1
        )
        kzs = np.sqrt(ks * ks - 1)
        zs = np.sqrt(mu / epsilon)
        assert np.all(np.abs(cf.fresnel(ks, kzs, zs) - expect) < EPSSQ)


class TestMieCyl:
    def test_real(self):
        # Calculated with comsol
        expect = [
            [-0.87100 - 0.069690j, -0.14510 - 0.2492j],
            [-0.18286 - 0.30517j, -0.62281 + 0.37137j],
        ]
        m = -2
        epsilon = [1, 4, 2]
        mu = [4, 3, 2]
        kappa = [-0.5, 0.1, 0.2]
        kz = 2 * np.pi / 600
        k0 = 2 * np.pi / 500
        radii = [50, 100]
        assert np.all(
            np.abs(cf.mie_cyl(kz, m, k0, radii, epsilon, mu, kappa) - expect)
            / np.max(np.abs(expect))
            < 1.1e-2
        )

    def test_evanescent(self):
        # Calculated with comsol
        expect = [
            [-3.2138e-4 - 29.992j, -0.0031377 + 43.380j],
            [6.7389e-4 + 35.637j, -0.0090938 - 74.594j],
        ]
        m = 1
        epsilon = [3, 2, 2]
        mu = [6, 1, 2]
        kappa = [1, 0, -0.2]
        kz = 2 * np.pi / 150
        k0 = 2 * np.pi / 500
        radii = [30, 100]
        assert np.all(
            np.abs(cf.mie_cyl(kz, m, k0, radii, epsilon, mu, kappa) - expect)
            / np.max(np.abs(expect))
            < 1e-2
        )

    def test_complex(self):
        # Calculated with comsol
        expect = [
            [-0.88746 - 0.15313j, 0.12955 + 0.086434j],
            [0.103112 + 0.081574j, -0.098055 + 0.017689j],
        ]
        m = -1
        epsilon = [4, 2 + 1j, 2 + 0.02j]
        mu = [1 + 0.5j, 1, 2 + 0.1j]
        kappa = [-0.5, 0.1 + 0.1j, -0.2 + 0.001j]
        kz = 2 * np.pi / 250
        k0 = 2 * np.pi / 500
        radii = [50, 100]
        assert np.all(
            np.abs(cf.mie_cyl(kz, m, k0, radii, epsilon, mu, kappa) - expect)
            / np.max(np.abs(expect))
            < 1e-2
        )
