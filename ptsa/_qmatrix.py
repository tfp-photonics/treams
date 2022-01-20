import copy

import numpy as np

import ptsa.special as sc
from ptsa import misc, pw
from ptsa.coeffs import fresnel


class QMatrix:
    r"""
    Q-Matrix (plane wave basis)

    The Q-matrix describes the scattering of incoming into outgoing modes using a plane
    wave basis, with functions :func:`ptsa.special.vsw_A`, :func:`ptsa.special.vsw_M`,
    and :func:`ptsa.special.vsw_N`. The primary direction of propagation is parallel or
    anti-parallel to the z-axis. The scattering object itself is infinitely extended in
    the x- and y-directions. The Q-matrix is divided into four submatrices
    :math:`Q_{\uparrow \uparrow}`, :math:`Q_{\uparrow \downarrow}`,
    :math:`Q_{\downarrow \uparrow}`, and :math:`Q_{\downarrow \downarrow}`:

    .. math::

        Q = \begin{pmatrix}
            Q_{\uparrow \uparrow} & Q_{\uparrow \downarrow} \\
            Q_{\downarrow \uparrow} & Q_{\downarrow \downarrow}
        \end{pmatrix}\,.

    These matrices describe the transmission of plane waves propagating into positive
    z-direction, reflection of plane waves into the positive z-direction, reflection
    of plane waves into negative z-direction, and the transmission of plane waves
    propagating in negative z-direction, respectively. Each of those for matrices
    contain different diffraction orders and different polarizations. The polarizations
    can be in either helicity of parity basis.

    The embedding medium is described by permittivity, permeability, and the chirality
    parameter. The two sides of the Q-matrix can have different materials.

    Args:
        qmats (float or complex, array): Q-matrices
        k0 (float): Wave number in vacuum
        modes (iterable, optional): The modes corresponding to the Q-matrices rows and
            columns. It must contain three arrays of equal length for the wave vectors'
            x- and y-component, and the polarization.
        epsilon (float or complex, optional): Relative permittivity of the medium
        mu (float or complex, optional): Relative permeability of the medium
        kappa (float or complex, optional): Chirality parameter of the medium
        helicity (bool, optional): Helicity or parity modes
        kz (float or complex, (N, 2)-array, optional): The z-component of the wave
            vector on both sides of the Q-matrix
    """

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
        self.ks = k0 * misc.refractive_index(epsilon, mu, kappa)
        if kz is None:
            kz = misc.wave_vec_z(self.kx[:, None], self.ky[:, None], self.ks[:, self.pol])
        kz = np.array(kz)
        if kz.shape != (2, self.kx.shape[0]):
            ValueError(f"shape of kz f{kz.shape} not supported")
        self.q = qmats  # (2, 2, a, a)
        self.k0 = np.array(k0).item()  # ()
        self.epsilon = epsilon  # (2,)
        self.mu = mu  # (2,)
        self.kappa = kappa  # (2,)
        self.helicity = helicity
        self.kz = kz  # (2, a)

    @property
    def modes(self):
        return self.kx, self.ky, self.pol

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
        thickness = np.atleast_1d(thickness)
        epsilon = np.atleast_1d(epsilon)
        mu = np.atleast_1d(mu)
        kappa = np.atleast_1d(kappa)
        nthickness = thickness.shape[0]
        for name, param in [
                ("thickness", thickness),
                ("epsilon", epsilon),
                ("kappa", kappa),
                ("mu", mu)
        ]:
            if param.ndim > 1:
                raise ValueError(f"'{name}' must be scalar or 1D, but was {param.ndim}D")
            if name == "thickness":
                continue
            if len(param) == 1:
                param = np.repeat(param, nthickness + 2)
            elif len(param) != nthickness + 2:
                raise ValueError(f"'{name}' must be scalar or of length {thickness}, but was {len(param)}")
        items == 0
        # Use the truncation of the material parameters by zip due to 'thickness'
        for d, epsilon_below, epsilon_above, mu_below, mu_above, k_below, k_above in zip(thickness, epsilon, epsilon[1:], mu, mu[1:], kappa, kappa[1:]):
            items.append(QMatrix.interface(k0, kpars, [epsilon_below, epsilon_above], [mu_below, mu_above], [kappa_below, kappa_above]))
            items.append(QMatrix.propagation([0, 0, d], k0, kpars, epsilon_above, mu_above, kappa_above))
        items.append(QMatrix.interface(k0, kpars, epsilon[-2:], mu[-2:], kappa[-2:]))
        return QMatrix.stack(items, check_materials=False, check_modes=False)



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
        self.kz = np.zeros((2, modes.shape[0]))
        self.kz[0, :] = misc.wave_vec_z(modes[0], modes[1], self.ks[0, modes[2]])
        self.kz[1, :] = misc.wave_vec_z(modes[0], modes[1], self.ks[1, modes[2]])
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

    def field_outside(self, illu):
        illu = (np.zeros_like(self.kx) if i is None else i for i in illu)
        field_above = self.q[0, 0, :, :] @ illu[0] + self.q[0, 1] @ illu[1]
        field_below = self.q[1, 0, :, :] @ illu[0] + self.q[1, 1] @ illu[1]
        return field_above, field_below

    def field_inside(self, illu, q_above):
        illu = (np.zeros_like(self.kx) if i is None else i for i in illu)
        qtmp_up = np.eye(self.q.shape[2]) - self.q[0, 1, :, :] @ q_above.q[1, 0, :, :]
        qtmp_down = np.eye(self.q.shape[2]) - q_above.q[1, 0, :, :] @ self.q[0, 1, :, :]
        field_up = np.linalg.solve(qtmp_up, self.q[0, 0, :, :] @ illu[0]) + self.q[
            0, 1
        ] @ np.linalg.solve(qtmp_down, q_above.q[1, 1, :, :] @ illu[1])
        field_down = np.linalg.solve(
            qtmp_down, q_above.q[1, 1, :, :] @ illu[1]
        ) + q_above.q[1, 0] @ np.linalg.solve(qtmp_up, self.q[0, 0, :, :] @ illu[0])
        return field_up, field_down

    def field(self, r, direction=1):
        if direction not in (-1, 1):
            raise ValueError(f"direction must be '-1' or '1', but is '{direction}''")
        if self.helicity:
            return sc.vpw_A(
                self.kx,
                self.ky,
                direction * self.kz[:, choice],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
                self.pol,
            )
        else:
            return (1 - self.pol[:, None]) * sc.vpw_M(
                self.kx,
                self.ky,
                direction * self.kz[:, choice],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
            ) - self.pol[:, None] * sc.vpw_M(
                self.kx,
                self.ky,
                direction * self.kz[:, choice],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
            )

    def poynting_avg(self, coeffs, above=True):
        choice = bool(above)
        ky, ky, pol = self.modes
        selections = pol == 0, pol == 1
        pref = (
            self.kz[pol == 0, choice] / self.ks[choice, 0],
            self.kz[pol == 1, choice] / self.ks[choice, 1],
        )
        coeffs = [np.zeros_like(self.kx) if c is None else c for c in coeffs]
        allcoeffs = [
            (1, -1, coeffs[0][selections[0]]),
            (1, 1, coeffs[0][selections[1]]),
            (-1, -1, coeffs[1][selections[0]]),
            (-1, 1, coeffs[1][selections[1]]),
        ]
        res = 0
        if self.helicity:
            for (dira, pola, a), (dirb, polb, b) in itertools.product(allcoeffs, repeat=2):
                res += a @ (b.conjugate() * (pola * polb * pref[(polb + 1) // 2].conjugate() * dirb + pref[(pola + 1) // 2] * dira))
            res *= 0.25
        else:
            for (dira, _, a), (dirb, _, b) in itertools.product(allcoeffs[::2], repeat=2):
                res += a @ (b.conjugate() * pref[0].conjugate() * dirb)
            for (dira, _, a), (dirb, _, b) in itertools.product(allcoeffs[1::2], repeat=2):
                res += a @ (b.conjugate() * pref[1] * dira)
            res *= 0.5
        return np.real(res / np.conjugate(np.sqrt(self.mu[choice] / self.epsilon[choice])))

    def chirality_density(self, coeffs, thickness=0):
        if not self.helicity:
            raise NotImplementedError
        raise NotImplementedError

    def tr(self, illu, direction=1):
        """Transmittance and reflectance"""
        if direction not in (-1, 1):
            raise ValueError(f"direction must be '-1' or '1', but is '{direction}''")
        illu = (illu, None) if direction == 1 else (None, illu)
        pow_i = self.poynting_avg(illu, above=direction == -1)
        field_above, field_below = self.field_outside(illu)
        a = self.poynting_avg((field_above, None)) / pow_i
        b = self.poynting_avg((None, field_below)) / pow_i
        return (a, -b) if direction == 1 else (b, -a)

    def cd(self, illu):
        """Circular dichroism"""
        if not self.helicity:
            raise NotImplementedError
        raise NotImplementedError

    def optrot(self, illu):
        """Optical rotation"""
        if not self.helicity:
            raise NotImplementedError
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
