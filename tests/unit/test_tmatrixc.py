import copy

import numpy as np

import ptsa
from ptsa import TMatrixC

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class TestInit:
    def test_simple(self):
        tm = TMatrixC(np.eye(6), 1)
        assert (
            np.all(tm.t == np.eye(6))
            and tm.k0 == 1
            and tm.epsilon == 1
            and tm.mu == 1
            and tm.kappa == 0
            and np.all(tm.positions == [[0, 0, 0]])
            and tm.helicity
            and np.all(tm.kz == 6 * [0])
            and np.all(tm.m == [-1, -1, 0, 0, 1, 1])
            and np.all(tm.pol == [1, 0, 1, 0, 1, 0])
            and np.all(tm.pidx == [0, 0, 0, 0, 0, 0])
            and np.all(tm.ks == [1, 1])
        )
    def test_complex(self):
        tm = TMatrixC(
            np.diag([1, 2]),
            3,
            epsilon=2,
            mu=8,
            kappa=1,
            positions=[1, 0, 0],
            modes=([1, 1], [0, 0], [0, 1]),
        )
        assert (
            np.all(tm.t == np.diag([1, 2]))
            and tm.k0 == 3
            and tm.epsilon == 2
            and tm.mu == 8
            and tm.kappa == 1
            and np.all(tm.positions == [[1, 0, 0]])
            and tm.helicity
            and np.all(tm.kz == [1, 1])
            and np.all(tm.m == [0, 0])
            and np.all(tm.pol == [0, 1])
            and np.all(tm.pidx == [0, 0])
            and np.all(tm.ks == [9, 15])
        )



class TestCylinder:
    def test(self):
        tm = TMatrixC.cylinder([1], 2, 3, [4], [2, 9], kappa=[1, 2])
        m = ptsa.coeffs.mie_cyl(1, [-2, -1, 0, 1, 2], 3, [4], [2, 9], [1, 1], [1, 2])
        assert  (
            tm.t[0, 0] == m[0, 1, 1]
            and tm.t[1, 1] == m[0, 0, 0]
            and tm.t[0, 1] == m[0, 1, 0]
            and tm.t[1, 0] == m[0, 0, 1]
            and tm.t[2, 2] == m[1, 1, 1]
            and tm.t[3, 3] == m[1, 0, 0]
            and tm.t[2, 3] == m[1, 1, 0]
            and tm.t[3, 2] == m[1, 0, 1]
            and tm.t[4, 4] == m[2, 1, 1]
            and tm.t[5, 5] == m[2, 0, 0]
            and tm.t[4, 5] == m[2, 1, 0]
            and tm.t[5, 4] == m[2, 0, 1]
            and tm.t[6, 6] == m[3, 1, 1]
            and tm.t[7, 7] == m[3, 0, 0]
            and tm.t[6, 7] == m[3, 1, 0]
            and tm.t[7, 6] == m[3, 0, 1]
            and tm.t[8, 8] == m[4, 1, 1]
            and tm.t[9, 9] == m[4, 0, 0]
            and tm.t[8, 9] == m[4, 1, 0]
            and tm.t[9, 8] == m[4, 0, 1]
            and tm.k0 == 3
            and tm.epsilon == 9
            and tm.mu == 1
            and tm.kappa == 2
            and np.all(tm.positions == [[0, 0, 0]])
            and tm.helicity
            and np.all(tm.kz == 10 * [1])
            and np.all(tm.m == [-2, -2, -1, -1, 0, 0, 1, 1, 2, 2])
            and np.all(tm.pol == [int((i + 1) % 2) for i in range(10)])
            and np.all(tm.pidx == 10 * [0])
            and np.all(tm.ks == [3, 15])
        )
