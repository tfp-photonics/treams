import itertools

import numpy as np
import scipy.integrate

import ptsa.special as sc


def _j_real(l0, l1, m, r, dr, pol0, pol1, ks, k, zs, z):
    def fun(theta):
        return np.real(
            np.sin(theta)
            * ((2 * pol0 - 1) * z + (2 * pol1 - 1) * zs)
            * np.dot(
                [r(theta), -dr(theta), 0],
                np.cross(
                    sc.vsw_rA(l0, -m, ks[pol0] * r(theta), theta, 0, pol0),
                    sc.vsw_A(l1, m, k[pol1] * r(theta), theta, 0, pol1),
                ),
            )
        )

    return fun


def _j_imag(l0, l1, m, r, dr, pol0, pol1, ks, k, zs, z):
    def fun(theta):
        return np.imag(
            np.sin(theta)
            * ((2 * pol0 - 1) * z + (2 * pol1 - 1) * zs)
            * np.dot(
                [r(theta), -dr(theta), 0],
                np.cross(
                    sc.vsw_rA(l0, -m, ks[pol0] * r(theta), theta, 0, pol0),
                    sc.vsw_A(l1, m, k[pol1] * r(theta), theta, 0, pol1),
                ),
            )
        )

    return fun


def _rj_real(l0, l1, m, r, dr, pol0, pol1, ks, k, zs, z):
    def fun(theta):
        return np.real(
            np.sin(theta)
            * ((2 * pol0 - 1) * z + (2 * pol1 - 1) * zs)
            * np.dot(
                [r(theta), -dr(theta), 0],
                np.cross(
                    sc.vsw_rA(l0, -m, ks[pol0] * r(theta), theta, 0, pol0),
                    sc.vsw_rA(l1, m, k[pol1] * r(theta), theta, 0, pol1),
                ),
            )
        )

    return fun


def _rj_imag(l0, l1, m, r, dr, pol0, pol1, ks, k, zs, z):
    def fun(theta):
        return np.imag(
            np.sin(theta)
            * ((2 * pol0 - 1) * z + (2 * pol1 - 1) * zs)
            * np.dot(
                [r(theta), -dr(theta), 0],
                np.cross(
                    sc.vsw_rA(l0, -m, ks[pol0] * r(theta), theta, 0, pol0),
                    sc.vsw_rA(l1, m, k[pol1] * r(theta), theta, 0, pol1),
                ),
            )
        )

    return fun


def qmat(r, dr, ks, zs, out, in_=None, singular=True):
    l_out, m_out, pol_out = out
    in_ = out if in_ is None else in_
    l_in, m_in, pol_in = in_
    if singular:
        fr = _j_real
        fi = _j_imag
    else:
        fr = _rj_real
        fi = _rj_imag
    size_out = len(l_out)
    size_in = len(l_in)
    res = np.zeros((size_out, size_in), complex)
    for i, j in itertools.product(range(size_out), range(size_in)):
        if m_out[i] != m_in[j]:
            continue
        res[i, j] = (
            scipy.integrate.quad(
                fr(l_out[i], l_in[j], m_out[i], r, dr, pol_out[i], pol_in[j], *ks, *zs),
                0,
                np.pi,
            )[0]
            + 1j
            * scipy.integrate.quad(
                fi(l_out[i], l_in[j], m_out[i], r, dr, pol_out[i], pol_in[j], *ks, *zs),
                0,
                np.pi,
            )[0]
        )
    return res
