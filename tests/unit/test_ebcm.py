import numpy as np

import treams
import treams.ebcm


def _r(theta):
    return 0.3 * (1 + 0.23 * np.cos(theta) ** 2)


def _dr(theta):
    return -0.138 * np.cos(theta) * np.sin(theta)


def _tmatrix(lmax, inside, outside, r=_r, dr=_dr, k0=1.3):
    basis = treams.SphericalWaveBasis.default(lmax)
    ks = [inside.ks(k0), outside.ks(k0)]
    zs = [inside.impedance, outside.impedance]
    modes = (basis.l, basis.m, basis.pol)
    q = treams.ebcm.qmat(r, dr, ks, zs, modes)
    qreg = treams.ebcm.qmat(r, dr, ks, zs, modes, singular=False)
    return -np.linalg.solve(q, qreg)


def test_sphere():
    materials = [treams.Material(3.1, 1, 0.07), treams.Material()]
    t = _tmatrix(2, *materials, r=lambda theta: 0.3, dr=lambda theta: 0)
    expect = treams.TMatrix.sphere(2, 1.3, 0.3, materials, poltype="helicity")
    assert np.allclose(t, np.asarray(expect))


def test_zero_contrast():
    t = _tmatrix(2, treams.Material(), treams.Material())
    assert np.abs(t).max() < 1e-14


def test_unitary():
    t = _tmatrix(4, treams.Material(3.1, 1, 0.07), treams.Material())
    s = np.eye(len(t)) + 2 * t
    assert np.abs(s.conj().T @ s - np.eye(len(t))).max() < 1e-5
