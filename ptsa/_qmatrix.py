import copy

import numpy as np

import ptsa.special as sc
from ptsa import misc, pw
from ptsa.coeffs import fresnel


class QMatrix:
    """Q-Matrix"""

    def __init__(
        self, qmats, k0, modes, epsilon=1, mu=1, kappa=0, helicity=True, kz=None,
    ):
        if np.any(kappa != 0) and not helicity:
            raise ValueError("chiral medium requires helicity modes")
        qmats = np.array(qmats)
        epsilon = np.array(epsilon)
        mu = np.array(mu)
        kappa = np.array(kappa)
        if epsilon.ndim == 0 or (epsilon.ndim == 1 and epsilon.shape[0] == 1):
            epsilon = np.stack((epsilon, epsilon), axis=-1)
        else:
            ValueError(f"shape of epsilon f{epsilon.shape} not supported")
        if mu.ndim == 0 or (mu.ndim == 1 and mu.shape[0] == 1):
            mu = np.stack((mu, mu), axis=-1)
        else:
            ValueError(f"shape of mu f{mu.shape} not supported")
        if kappa.ndim == 0 or (kappa.ndim == 1 and kappa.shape[0] == 1):
            kappa = np.stack((kappa, kappa), axis=-1)
        else:
            ValueError(f"shape of kappa f{kappa.shape} not supported")
        modes = self._check_modes(modes)
        self.kx, self.ky, self.pol = modes  # ((a,), (a,), (a,))
        if kz is None:
            kz = misc.wave_vec_z(self.kx, self.ky, self.ks[:, self.pol])
        kz = np.array(kz)
        if kz.shape != (2, self.kx.shape[0]):
            ValueError(f"shape of kz f{kz.shape} not supported")
        self.q = qmats  # (2, 2, a, a)
        self.k0 = k0  # ()
        self.epsilon = epsilon  # (2,)
        self.mu = mu  # (2,)
        self.kappa = kappa  # (2,)
        self.helicity = helicity
        self.kz = kz  # (2, a)
        self.ks = k0 * misc.refractive_index(epsilon, mu, kappa)

    @staticmethod
    def defaultmodes(kpars):
        """Default modes"""
        kpars = np.array(kpars)
        res = np.empty((2 * kpars.shape[0], 2), np.float64)
        res[::2, :] = kpars
        res[1::2, :] = kpars
        pols = np.empty(2 * kpars.shape[0], int)
        pols[::2] = 1
        pols[1::2] = 0
        return (*res.T, pols)

    @classmethod
    def interface(cls, k0, kpars, epsilon, mu=1, kappa=0):
        """
        Interface

        Args:
            k0 (float): Wave number in vacuum
            kpars (float, array): Tangential part of the wave vector
            epsilon (float or complex): Relative permittivity on both sides of the interface
            mu (float or complex, optional): Relative permittivity on both sides of the interface
            kappa (float or complex, optional): Relative permittivity on both sides of the interface

        Returns:
            QMatrix
        """
        epsilon = np.array(epsilon, complex)
        mu = np.array(mu, complex)
        kappa = np.array(kappa, complex)
        if epsilon.ndim == 0:
            epsilon = np.array([epsilon, epsilon])
        if mu.ndim == 0:
            mu = np.array([mu, mu])
        if kappa.ndim == 0:
            kappa = np.array([kappa, kappa])
        kpars = np.array(kpars)
        if kpars.ndim == 1:
            kpars = kpars[None, :]
        ks = k0 * misc.refractive_index(epsilon, mu, kappa)
        zs = np.sqrt(mu / epsilon)
        modes = QMatrix.defaultmodes(kpars)
        kzs = misc.wave_vec_z(kpars[:, 0, None, None], kpars[:, 1, None, None], ks)
        qs = np.zeros((2, 2, modes[0].shape[0], modes[0].shape[0]), complex)
        vals = fresnel(ks, kzs, zs)
        for i in range(kpars.shape[0]):
            qs[:, :, 2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = vals[i, :, :, :, :]
        return cls(qs, k0, modes, epsilon, mu, kappa, True, kzs)

    @classmethod
    def slab(cls, k0, kpars, thickness, epsilon, mu=1, kappa=0):
        """
        Slab

        Args:
            k0 (float): Wave number in vacuum
            kpars (float, array): Tangential part of the wave vector
            epsilon (float or complex): Relative permittivity on both sides of the interface
            mu (float or complex, optional): Relative permittivity on both sides of the interface
            kappa (float or complex, optional): Relative permittivity on both sides of the interface

        Returns:
            QMatrix
        """
        raise NotImplementedError

    @classmethod
    def stack(cls, items, check_materials=True, check_modes=False):
        """Stack"""
        acc = copy.deepcopy(items[0])
        for item in items[1:]:
            acc.add(item, check_materials, check_modes)
        return acc

    @classmethod
    def propagation(cls, r, k0, kpars, epsilon=1, mu=1, kappa=0):
        """Propagation"""
        ks = k0 * misc.refractive_index(epsilon, mu, kappa)
        kpars = np.array(kpars)
        if kpars.ndim == 1:
            kpars = kpars[None, :]
        modes = QMatrix.defaultmodes(kpars)
        kzs = misc.wave_vec_z(*modes[:2], ks[modes[2]])
        qs = np.zeros((2, 2, modes[0].shape[0], modes[0].shape[0]), complex)
        tangential = np.exp(1j * (r[0] * modes[0] + r[1] * modes[1]))
        zdir = np.exp(1j * r[2] * kzs)
        qs[0, 0, :, :] = np.diag(tangential * zdir)
        qs[1, 1, :, :] = np.diag(tangential.conjugate() * zdir)
        return cls(
            qs,
            k0,
            modes,
            np.array([epsilon, epsilon], complex),
            np.array([mu, mu], complex),
            np.array([kappa, kappa], complex),
            True,
            np.stack((kzs, kzs), axis=-1),
        )

    @classmethod
    def array(cls, tmat, kpars, a, eta=0):
        """Array"""
        modes = QMatrix.defaultmodes(kpars)
        kzs = misc.wave_vec_z(*modes[:2], tmat.ks[modes[2]])
        a = np.array(a)
        allpw = (
            np.hstack((modes[0], modes[0])),
            np.hstack((modes[1], modes[1])),
            np.hstack((kzs, -kzs)),
            np.hstack((modes[2], modes[2])),
        )
        if a.ndim == 0 or (a.ndim == 1 and a.shape[0] == 1):
            yz = pw.xyz_to_zxy(
                *(m[:, None] for m in allpw), *allpw, helicity=tmat.helicity,
            )
            zy = pw.xyz_to_zxy(
                *(m[:, None] for m in allpw),
                *allpw,
                helicity=tmat.helicity,
                inverse=True,
            )
            res = (
                zy
                @ tmat.array1d(allpw[1], allpw[2], allpw[0], allpw[3], a, eta=eta)
                @ yz
            )
        else:
            res = tmat.array2d(*allpw, a, eta=eta)
        dim = modes[0].shape[0]
        qs = np.empty((2, 2, dim, dim), complex)
        qs[0, 0, :, :] = res[0:dim, 0:dim] + np.eye(dim)
        qs[0, 1, :, :] = res[0:dim, dim : 2 * dim]
        qs[1, 0, :, :] = res[dim : 2 * dim, 0:dim]
        qs[1, 1, :, :] = res[dim : 2 * dim, dim : 2 * dim] + np.eye(dim)
        epsilon = np.array([tmat.epsilon, tmat.epsilon], complex)
        mu = np.array([tmat.mu, tmat.mu], complex)
        kappa = np.array([tmat.kappa, tmat.kappa], complex)
        return cls(
            qs,
            tmat.k0,
            modes,
            epsilon,
            mu,
            kappa,
            tmat.helicity,
            np.stack((kzs, kzs), axis=-1),
        )

    @property
    def material(self):
        """
        Material parameters of the embedding medium

        The relative permittivity, relative permeability, and the chirality parameter
        specify the embedding medium.

        Returns:
            tuple
        """
        return self.epsilon, self.mu, self.kappa

    def add(self, qmat, check_materials=True, check_modes=False):
        """Add"""
        if check_materials and (
            self.epsilon[1] != qmat.epsilon[0]
            or self.mu[1] != qmat.mu[0]
            or self.kappa[1] != qmat.kappa[0]
        ):
            ValueError("materials do not match")
        if check_modes and (
            np.any(self.kx != qmat.kx)
            or np.any(self.ky != qmat.ky)
            or np.any(self.kz != qmat.kz)
        ):
            ValueError("modes do not match")
        dim = self.q.shape[2]
        qnew = np.empty_like(self.q)
        q_tmp = np.linalg.solve(
            np.eye(dim) - self.q[0, 1, :, :] @ qmat.q[1, 0, :, :], self.q[0, 0, :, :]
        )
        qnew[0, 0, :, :] = qmat.q[0, 0, :, :] @ q_tmp
        qnew[1, 0, :, :] = (
            self.q[1, 0, :, :] + self.q[1, 1, :, :] @ qmat.q[1, 0, :, :] @ q_tmp
        )
        q_tmp = np.linalg.solve(
            np.eye(dim) - qmat.q[1, 0, :, :] @ self.q[0, 1, :, :], qmat.q[1, 1, :, :]
        )
        qnew[1, 1, :, :] = self.q[1, 1, :, :] @ q_tmp
        qnew[0, 1, :, :] = (
            qmat.q[0, 1, :, :] + qmat.q[0, 0, :, :] @ self.q[0, 1, :, :] @ q_tmp
        )
        self.q = qnew
        self.epsilon[1] = qmat.epsilon[1]
        self.mu[1] = qmat.mu[1]
        self.kappa[1] = qmat.kappa[1]
        return self

    def double(self, times=1):
        """Double"""
        for _ in range(times):
            self.coupling(self)
        return self

    def changebasis(self, modes=None):
        """Change the basis"""
        if modes is None:
            modes = self.modes
        mat = misc.basischange(self.modes, modes)
        self.q[0, 0, :, :] = mat.T @ self.q[0, 0, :, :] @ mat
        self.q[0, 1, :, :] = mat.T @ self.q[0, 1, :, :] @ mat
        self.q[1, 0, :, :] = mat.T @ self.q[1, 0, :, :] @ mat
        self.q[1, 1, :, :] = mat.T @ self.q[1, 1, :, :] @ mat
        self.modes = modes
        self.helicity = not self.helicity
        self.kzs = np.zeros((modes.shape[0], 2))
        self.kzs[0, :] = misc.wave_vec_z(modes[0], modes[1], self.ks[0, modes[2]])
        self.kzs[1, :] = misc.wave_vec_z(modes[0], modes[1], self.ks[1, modes[2]])
        return self

    def helicitybasis(self, modes=None):
        """Helicity basis"""
        if not self.helicity:
            return self.changebasis(modes)
        if modes is None:
            return self
        return self.pick(modes)

    def paritybasis(self, modes=None):
        """Parity basis"""
        if self.helicity:
            return self.changebasis(modes)
        if modes is None:
            return self
        return self.pick(modes)

    def pick(self, modes):
        """Pick modes"""
        modes = self._check_modes(modes)
        mat = misc.pickmodes(self.modes, modes)
        self.q[0, 0, :, :] = mat.T @ self.q[0, 0, :, :] @ mat
        self.q[0, 1, :, :] = mat.T @ self.q[0, 1, :, :] @ mat
        self.q[1, 0, :, :] = mat.T @ self.q[1, 0, :, :] @ mat
        self.q[1, 1, :, :] = mat.T @ self.q[1, 1, :, :] @ mat
        self.kx, self.ky, self.pol = modes
        self.kz = misc.wave_vec_z(self.kx, self.ky, self.ks[:, self.pol])
        return self

    def field_outside(self, illu, illu_above=None):
        if illu_above is None:
            illu_above = np.zeros_like(illu)
        field_above = self.q[0, 0, :, :] @ illu + self.q[0, 1] @ illu_above
        field_below = self.q[1, 0, :, :] @ illu + self.q[1, 1] @ illu_above
        return field_above, field_below

    def field_inside(self, illu, q_above, illu_above=None):
        if illu_above is None:
            illu_above = np.zeros_like(illu)
        qtmp_up = np.eye(self.q.shape[2]) - self.q[0, 1, :, :] @ q_above.q[1, 0, :, :]
        qtmp_down = np.eye(self.q.shape[2]) - q_above.q[1, 0, :, :] @ self.q[0, 1, :, :]
        field_up = np.linalg.solve(qtmp_up, self.q[0, 0, :, :] @ illu) + self.q[
            0, 1
        ] @ np.linalg.solve(qtmp_down, q_above.q[1, 1, :, :] @ illu_above)
        field_down = np.linalg.solve(
            qtmp_down, q_above.q[1, 1, :, :] @ illu_above
        ) + q_above.q[1, 0] @ np.linalg.solve(qtmp_up, self.q[0, 0, :, :] @ illu)
        return field_up, field_down

    def field(self, r, field_above=True, outgoing=True):
        choice = np.int(field_above)
        pref = (2 * choice - 1) * (2 * outgoing - 1)
        if self.helicity:
            return sc.vpw_A(
                self.kx,
                self.ky,
                pref * self.kz[:, choice],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
                self.pol,
            )
        else:
            return (1 - self.pol[:, None]) * sc.vpw_M(
                self.kx,
                self.ky,
                pref * self.kz[:, choice],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
            ) - self.pol[:, None] * sc.vpw_M(
                self.kx,
                self.ky,
                pref * self.kz[:, choice],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
            )

    def poynting_avg(self, coeffs, thickness=0, above=True):
        if not self.helicity:
            raise NotImplementedError
        choice = np.int(above)
        coeffs = np.array(coeffs)
        res = (
            coeffs[:, None]
            @ coeffs.T.conjugate()
            * (
                self.pol[:, None]
                * self.pol
                * self.kz[:, choice, None]
                / self.ks[self.pol, None]
                + self.kz[:, choice].conjugate() / self.ks[self.pol]
            )
            / np.conjugate(np.sqrt(self.mu[choice] / self.epsilon[choice]))
        )
        if thickness != 0:
            res *= np.exp(1j * (self.kz[:, None] - self.kz.conjugate()) * thickness)
        return 0.25 * np.real(np.sum(res))

    def chirality_density(self, coeffs, thickness=0):
        if not self.helicity:
            raise NotImplementedError
        raise NotImplementedError

    def transmittance(self, illu, from_above=False):
        """Transmittance"""
        raise NotImplementedError

    def reflectance(self, illu, from_above=False):
        """Reflectance"""
        raise NotImplementedError

    def tr(self, illu, from_above=False):
        """Transmittance and reflectance"""
        return self.transmittance, self.reflectance

    def cd(self, illu, from_above=False):
        """Circular dichroism"""
        raise NotImplementedError

    def optrot(self, illu, from_above=False):
        """Optical rotation"""
        raise NotImplementedError

    def periodic(self):
        """Periodic arrangement"""
        dim = self.q.shape[2]
        res = np.empty((2 * dim, 2 * dim), dtype=complex)
        res[0:dim, 0:dim] = self.q[0, 0, :, :]
        res[0:dim, dim : 2 * dim] = self.q[0, 1, :, :]
        res[dim : 2 * dim, 0:dim] = -np.linalg.solve(
            self.q[1, 1, :, :], self.q[1, 0, :, :] @ self.q[0, 0, :, :]
        )
        res[dim : 2 * dim, dim : 2 * dim] = np.linalg.solve(
            self.q[1, 1, :, :], np.eye(dim) - self.q[1, 0, :, :] @ self.q[0, 1, :, :]
        )
        return res

    def bands_kz(self, a3):
        """Band structure calculation"""
        w, v = np.linalg.eig(self.periodic())
        return -1j * np.log(w) / a3, v

    def _check_modes(self, modes):
        """_check_modes"""
        if len(modes) != 3:
            ValueError(f"invalid length of variable modes {len(modes)}, must be 3 or 4")
        modes = (*(np.array(a) for a in modes),)
        if not np.all([m.size == modes[0].size for m in modes[1:]]):
            raise ValueError("all modes need equal size")
        return modes
