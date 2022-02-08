import copy
import itertools

import numpy as np

import ptsa.special as sc
from ptsa import misc, pw
from ptsa.coeffs import fresnel


class QMatrix:
    r"""
    Q-matrix (plane wave basis)

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

    Attributes:
        q (complex, (2, 2, N, N)-array): Q-matrices for the layer. The first dimension
            corresponds to the output side, the second to the input side, the third one
            to the outgoing modes and the last one to the incoming modes.
        k0 (float): Wave number in vacuum
        epsilon (float or complex, optional): Relative permittivity of the embedding medium
        mu (complex): Relative permeability of the embedding medium
        kappa (complex): Chirality parameter of the embedding medium
        helicity (bool): Helicity or parity modes
        kx (float, (N,)-array): X-component of the wave vector for each column/row of the Q-matrix
        ky (float, (N,)-array): Y-component of the wave vector for each column/row of the Q-matrix
        kz (float, (2, N)-array): Z-component of the wave vector for each column/row of the Q-matrix
            and for both sides
        pol (int, (N,)-array): Polarization of the mode for each column/row of the Q-matrix
        ks (float or complex (2, 2)-array): Wave numbers in the medium for both sides
            and both polarizations
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
        elif epsilon.ndim == 1 and epsilon.shape[0] == 2:
            pass
        else:
            raise ValueError(f"shape of epsilon f{epsilon.shape} not supported")
        if mu.ndim == 0 or (mu.ndim == 1 and mu.shape[0] == 1):
            mu = np.stack((mu, mu), axis=-1)
        elif mu.ndim == 1 and mu.shape[0] == 2:
            pass
        else:
            raise ValueError(f"shape of mu f{mu.shape} not supported")
        if kappa.ndim == 0 or (kappa.ndim == 1 and kappa.shape[0] == 1):
            kappa = np.stack((kappa, kappa), axis=-1)
        elif kappa.ndim == 1 and kappa.shape[0] == 2:
            pass
        else:
            raise ValueError(f"shape of kappa f{kappa.shape} not supported")
        modes = self._check_modes(modes)
        self.kx, self.ky, self.pol = modes
        self.ks = k0 * misc.refractive_index(epsilon, mu, kappa)  # (2, 2) -> side, pol
        if kz is None:
            kz = misc.wave_vec_z(self.kx, self.ky, self.ks[:, self.pol])
        kz = np.array(kz)
        if kz.shape != (2, self.kx.shape[0]):
            raise ValueError(f"shape of kz f{kz.shape} not supported")
        self.q = qmats
        self.k0 = np.array(k0).item()
        self.epsilon = epsilon
        self.mu = mu
        self.kappa = kappa
        self.helicity = helicity
        self.kz = kz

    @property
    def modes(self):
        """
        Modes of the Q-matrix

        X- and Y-components and polarization of each row/column of the Q-matrix

        Returns:
            3-tuple
        """
        return self.kx, self.ky, self.pol

    @staticmethod
    def defaultmodes(kpars):
        """
        Default sortation of modes

        Default sortation of the Q-matrix entries, including degree `kx`, order `ky` and
        polarization `p`.

        Args:
            kpars (float, (N, 2)-array): Tangential components of the wave vector

        Returns:
            tuple
        """
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
        Planar interface between two media

        Args:
            k0 (float): Wave number in vacuum
            kpars (float, (N, 2)-array): Tangential components of the wave vector
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
            qs[:, :, 2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = vals[i, :, :, ::-1, ::-1]

        reshaped_kzs = np.zeros((2, 2 * kzs.shape[0]), kzs.dtype)
        reshaped_kzs[:, modes[2] == 0] = kzs[:, :, 0].T
        reshaped_kzs[:, modes[2] == 1] = kzs[:, :, 1].T
        return cls(qs, k0, modes, epsilon, mu, kappa, True, reshaped_kzs)

    @classmethod
    def slab(cls, k0, kpars, thickness, epsilon, mu=1, kappa=0):
        """
        Slab of material

        A slab of material, defined by a thickness and three materials. Consecutive
        slabs of material can be defined by `n` thicknesses and and `n + 2` material
        parameters.

        Args:
            k0 (float): Wave number in vacuum
            kpars (float, (N, 2)-array): Tangential components of the wave vector
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
            ("mu", mu),
        ]:
            if param.ndim > 1:
                raise ValueError(
                    f"'{name}' must be scalar or 1D, but was {param.ndim}D"
                )
        if len(epsilon) == 1:
            epsilon = np.repeat(epsilon, nthickness + 2)
        elif len(epsilon) != nthickness + 2:
            raise ValueError(
                f"'epsilon' must be scalar or of length {nthickness}, but was {len(epsilon)}"
            )
        if len(mu) == 1:
            mu = np.repeat(mu, nthickness + 2)
        elif len(mu) != nthickness + 2:
            raise ValueError(
                f"'mu' must be scalar or of length {nthickness}, but was {len(mu)}"
            )
        if len(kappa) == 1:
            kappa = np.repeat(kappa, nthickness + 2)
        elif len(kappa) != nthickness + 2:
            raise ValueError(
                f"'kappa' must be scalar or of length {nthickness}, but was {len(kappa)}"
            )
        items = []
        # Use the truncation of the material parameters by zip due to 'thickness'
        for (
            d,
            epsilon_below,
            epsilon_above,
            mu_below,
            mu_above,
            kappa_below,
            kappa_above,
        ) in zip(thickness, epsilon, epsilon[1:], mu, mu[1:], kappa, kappa[1:]):
            items.append(
                QMatrix.interface(
                    k0,
                    kpars,
                    [epsilon_below, epsilon_above],
                    [mu_below, mu_above],
                    [kappa_below, kappa_above],
                )
            )
            items.append(
                QMatrix.propagation(
                    [0, 0, d], k0, kpars, epsilon_above, mu_above, kappa_above
                )
            )
        items.append(QMatrix.interface(k0, kpars, epsilon[-2:], mu[-2:], kappa[-2:]))
        return QMatrix.stack(items, check_materials=False, check_modes=False)

    @classmethod
    def stack(cls, items, check_materials=True, check_modes=False):
        """
        Stack of Q-matrices

        Electromagnetically couple multiple Q-matrices in the order given. Before
        coupling it can be checked for matching materials and modes.

        Args:
            items (QMatrix, array-like): An array of Q-matrices in their intended order
            check_materials (bool, optional): Check for matching material parameters
                at each Q-matrix
            check_materials (bool, optional): Check for matching modes at each Q-matrix

        Returns:
            QMatrix
        """
        acc = copy.deepcopy(items[0])
        for item in items[1:]:
            acc.add(item, check_materials, check_modes)
        return acc

    @classmethod
    def propagation(cls, r, k0, kpars, epsilon=1, mu=1, kappa=0):
        """
        Q-matrix for the propagation along a distance

        This Q-matrix translates the reference origin along `r`.

        Args:
            r (float, (3,)-array): Translation vector
            k0 (float): Wave number in vacuum
            kpars (float, (N, 2)-array): Tangential components of the wave vector
            epsilon (float or complex, optional): Relative permittivity of the medium
            mu (float or complex, optional): Relative permeability of the medium
            kappa (float or complex, optional): Chirality parameter of the medium

        Returns:
            QMatrix
        """
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
            np.stack((kzs, kzs), axis=0),
        )

    @classmethod
    def array(cls, tmat, kpars, a, eta=0):
        """
        Q-matrix from an array of (cylindrical) T-matrices

        Create a Q-matrix for a two-dimensional array of objects described by the
        T-Matrix or an one-dimensional array of objects described by a cylindrical
        T-matrix.

        Args:
            tmat (TMatrix or TMatrixC): (Cylindrical) T-matrix to put in the arry
            kpars (float, (N, 2)-array): Tangential components of the wave vector
            a (array): Definition of the lattice
            eta (float or complex, optional): Splitting parameter in the lattice summation

        Returns:
            QMatrix
        """
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
            yz = pw.permute_xyz(
                *(m[:, None] for m in allpw), *allpw, helicity=tmat.helicity,
            )
            zy = pw.permute_xyz(
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
            np.stack((kzs, kzs), axis=0),
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
        """
        Couple another Q-matrix on top of the current one

        See also :func:`ptsa.QMatrix.stack` for a function that does not change the
        current Q-matrix but creates a new one.

        Args:
            items (QMatrix): Q-matrices in their intended order
            check_materials (bool, optional): Check for matching material parameters
                at each Q-matrix
            check_materials (bool, optional): Check for matching modes at each Q-matrix

        Returns:
            QMatrix
        """
        if check_materials and (
            self.epsilon[1] != qmat.epsilon[0]
            or self.mu[1] != qmat.mu[0]
            or self.kappa[1] != qmat.kappa[0]
        ):
            raise ValueError("materials do not match")
        if check_modes and (
            np.any(self.kx != qmat.kx)
            or np.any(self.ky != qmat.ky)
            or np.any(self.kz != qmat.kz)
        ):
            raise ValueError("modes do not match")
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
        self.kz[1, :] = qmat.kz[1, :]
        self.ks[1, :] = qmat.ks[1, :]
        return self

    def double(self, times=1):
        """
        Double the Q-matrix

        By default this function doubles the Q-matrix but it can also create a
        :math:`2^n`-fold repetition of itself:

        Args:
            times (int, optional): Number of times to double itself. Defaults to 1.

        Returns:
            QMatrix
        """
        for _ in range(times):
            self.add(self)
        return self

    def changebasis(self, modes=None):
        """
        Swap between helicity and parity basis

        Args:
            modes (array, optional): Change the number of modes while changing the basis

        Returns:
            QMatrix
        """
        if modes is None:
            modes = self.modes
        mat = misc.basischange(self.modes, modes)
        self.q[0, 0, :, :] = mat.T @ self.q[0, 0, :, :] @ mat
        self.q[0, 1, :, :] = mat.T @ self.q[0, 1, :, :] @ mat
        self.q[1, 0, :, :] = mat.T @ self.q[1, 0, :, :] @ mat
        self.q[1, 1, :, :] = mat.T @ self.q[1, 1, :, :] @ mat
        self.kx, self.ky, self.pol = modes
        self.helicity = not self.helicity
        self.kz = misc.wave_vec_z(*modes[:2], self.ks[:, modes[2]])
        return self

    def helicitybasis(self, modes=None):
        """
        Change to helicity basis

        Args:
            modes (array, optional): Change the number of modes while changing the basis

        Returns:
            QMatrix
        """
        if not self.helicity:
            return self.changebasis(modes)
        if modes is None:
            return self
        return self.pick(modes)

    def paritybasis(self, modes=None):
        """
        Change to parity basis

        Args:
            modes (array, optional): Change the number of modes while changing the basis

        Returns:
            QMatrix
        """
        if self.helicity:
            return self.changebasis(modes)
        if modes is None:
            return self
        return self.pick(modes)

    def pick(self, modes):
        """
        Pick modes from the Q-matrix

        Args:
            modes (array): Modes of the new Q-matrix

        Returns:
            QMatrix
        """
        modes = self._check_modes(modes)
        mat = misc.pickmodes(self.modes, modes)
        q = np.zeros((2, 2, mat.shape[-1], mat.shape[-1]), self.q.dtype)
        q[0, 0, :, :] = mat.T @ self.q[0, 0, :, :] @ mat
        q[0, 1, :, :] = mat.T @ self.q[0, 1, :, :] @ mat
        q[1, 0, :, :] = mat.T @ self.q[1, 0, :, :] @ mat
        q[1, 1, :, :] = mat.T @ self.q[1, 1, :, :] @ mat
        self.q = q
        self.kx, self.ky, self.pol = modes
        self.kz = misc.wave_vec_z(self.kx, self.ky, self.ks[:, self.pol])
        return self

    def field_outside(self, illu):
        """
        Field coefficients above and below the Q-matrix

        Given an illumination defined by the coefficients of each incoming mode
        calculate the coefficients for the outgoing field above and below the Q-matrix.

        Args:
            illu (tuple): A 2-tuple of arrays, with the entries corresponding to
                upwards and downwards incoming modes.

        Returns:
            tuple
        """
        illu = [np.zeros_like(self.kx) if i is None else i for i in illu]
        field_above = self.q[0, 0, :, :] @ illu[0] + self.q[0, 1] @ illu[1]
        field_below = self.q[1, 0, :, :] @ illu[0] + self.q[1, 1] @ illu[1]
        return field_above, field_below

    def field_inside(self, illu, q_above):
        """
        Field coefficients between two Q-matrices

        Given an illumination defined by the coefficients of each incoming mode
        calculate the coefficients for the field between the two Q-matrices. The
        coefficients are separated into upwards and downwards propagating modes.

        Args:
            illu (tuple): A 2-tuple of arrays, with the entries corresponding to
                upwards and downwards incoming modes.
            q_above (QMatrix): Q-matrix above the current one

        Returns:
            tuple
        """
        illu = [np.zeros_like(self.kx) if i is None else i for i in illu]
        qtmp_up = np.eye(self.q.shape[2]) - self.q[0, 1, :, :] @ q_above.q[1, 0, :, :]
        qtmp_down = np.eye(self.q.shape[2]) - q_above.q[1, 0, :, :] @ self.q[0, 1, :, :]
        field_up = np.linalg.solve(qtmp_up, self.q[0, 0, :, :] @ illu[0]) + self.q[
            0, 1
        ] @ np.linalg.solve(qtmp_down, q_above.q[1, 1, :, :] @ illu[1])
        field_down = np.linalg.solve(
            qtmp_down, q_above.q[1, 1, :, :] @ illu[1]
        ) + q_above.q[1, 0] @ np.linalg.solve(qtmp_up, self.q[0, 0, :, :] @ illu[0])
        return field_up, field_down

    def field(self, r, direction=1, above=True):
        """
        Calculate the field at specified points

        The mode expansion of the Q-matrix is used. The field direction and the side of
        the Q-matrix can be specified

        Args:
            r (float, array_like): Array of the positions to probe
            direction (int, optional): The direction of the field, options are `-1` and `1`
            above (bool, optional): Take the field above or below the Q-matrix

        Returns
            complex
        """
        if direction not in (-1, 1):
            raise ValueError(f"direction must be '-1' or '1', but is '{direction}''")
        choice = int(bool(above))
        r = np.array(r)
        if r.ndim == 1:
            r = np.reshape(r, (1, -1))
        if self.helicity:
            return sc.vpw_A(
                self.kx,
                self.ky,
                direction * self.kz[choice, :],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
                self.pol,
            )
        else:
            return (1 - self.pol[:, None]) * sc.vpw_M(
                self.kx,
                self.ky,
                direction * self.kz[choice, :],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
            ) + self.pol[:, None] * sc.vpw_N(
                self.kx,
                self.ky,
                direction * self.kz[choice, :],
                r[..., None, 0],
                r[..., None, 1],
                r[..., None, 2],
            )

    def poynting_avg(self, coeffs, above=True):
        r"""
        Time-averaged z-component of the Poynting vector

        Calculate the time-averaged Poynting vector's z-component

        .. math::

            \langle S_z \rangle = \frac{1}{2} \Re (\boldsymbol E \times \boldsymbol H^\ast) \boldsymbol{\hat{z}}

        on one side of the Q-matrix with the given coefficients.

        Args:
            coeffs (2-tuple): The first entry are the upwards propagating modes the
                second one the downwards propagating modes
            above (bool, optional): Calculate the Poynting vector above or below the
                Q-matrix

        Returns:
            float
        """
        choice = int(bool(above))
        ky, ky, pol = self.modes
        selections = pol == 0, pol == 1
        pref = (
            self.kz[choice, selections[0]] / self.ks[choice, 0],
            self.kz[choice, selections[1]] / self.ks[choice, 1],
        )
        coeffs = [np.zeros_like(self.kx) if c is None else np.array(c) for c in coeffs]
        allcoeffs = [
            (1, -1, coeffs[0][selections[0]]),
            (1, 1, coeffs[0][selections[1]]),
            (-1, -1, coeffs[1][selections[0]]),
            (-1, 1, coeffs[1][selections[1]]),
        ]
        res = 0
        if self.helicity:
            for (dira, pola, a), (dirb, polb, b) in itertools.product(
                allcoeffs, repeat=2
            ):
                res += a @ (
                    b.conjugate()
                    * (
                        pola * polb * pref[(polb + 1) // 2].conjugate() * dirb
                        + pref[(pola + 1) // 2] * dira
                    )
                )
            res *= 0.25
        else:
            for (dira, _, a), (dirb, _, b) in itertools.product(
                allcoeffs[::2], repeat=2
            ):
                res += a @ (b.conjugate() * pref[0].conjugate() * dirb)
            for (dira, _, a), (dirb, _, b) in itertools.product(
                allcoeffs[1::2], repeat=2
            ):
                res += a @ (b.conjugate() * pref[1] * dira)
            res *= 0.5
        return np.real(
            res / np.conjugate(np.sqrt(self.mu[choice] / self.epsilon[choice]))
        )

    def chirality_density(self, coeffs, z=None, above=True):
        r"""
        Volume-averaged chirality density

        Calculate the volume-averaged chirality density

        .. math::

            \int_{A_\text{unit cell}} \mathrm d A \int_{z_0}^{z_1} \mathrm d z |G_+(\boldsymbol r)|^2 - |G_-(\boldsymbol r)|^2

        on one side of the Q-matrix with the given coefficients. The calculation can
        also be done for an infinitely thin sheet. The Riemann-Silberstein vectors are
        :math:`\sqrt{2} \boldsymbol G_\pm(\boldsymbol r) = \boldsymbol E(\boldsymbol r) + \mathrm i Z_0 Z \boldsymbol H(\boldsymbol r)`.

        Args:
            coeffs (2-tuple): The first entry are the upwards propagating modes the
                second one the downwards propagating modes
            above (bool, optional): Calculate the chirality density above or below the
                Q-matrix

        Returns:
            float
        """
        choice = int(bool(above))
        z = (0, 0) if z is None else z
        selections = self.pol == 0, self.pol == 1
        kzs = (self.kz[choice, selections[0]], self.kz[choice, selections[1]])
        coeffs = [np.zeros_like(self.kx) if c is None else np.array(c) for c in coeffs]
        allcoeffs = [
            (1, 0, coeffs[0][selections[0]]),
            (1, 1, coeffs[0][selections[1]]),
            (-1, 0, coeffs[1][selections[0]]),
            (-1, 1, coeffs[1][selections[1]]),
        ]
        res = 0
        if self.helicity:
            for (dira, pola, a), (dirb, polb, b) in itertools.product(
                allcoeffs, repeat=2
            ):
                if pola != polb:
                    continue
                res += (2 * pola - 1) * _coeff_chirality_density(
                    self.ks[choice, pola], kzs[pola], a, dira, b, dirb, z
                )
        else:
            for (dira, _, a), (dirb, _, b) in itertools.product(
                allcoeffs[::2], allcoeffs[1::2]
            ):
                res += _coeff_chirality_density(
                    self.ks[choice, 0], kzs[0], a, dira, b, dirb, z, False
                )
        return 0.5 * np.real(res)

    def tr(self, illu, direction=1):
        """
        Transmittance and reflectance

        Calculate the transmittance and reflectance for one Q-matrix with the given
        illumination and direction.

        Args:
            illu (complex, array_like): Expansion coefficients for the incoming light
            direction (int, optional): The direction of the field, options are `-1` and `1`

        Returns:
            tuple
        """
        if direction not in (-1, 1):
            raise ValueError(f"direction must be '-1' or '1', but is '{direction}''")
        illu_full = (illu, None) if direction == 1 else (None, illu)
        field_above, field_below = self.field_outside(illu_full)
        r = (None, field_below) if direction == 1 else (field_above, None)
        ir = (illu, field_below) if direction == 1 else (field_above, illu)
        t = (field_above, None) if direction == 1 else (None, field_below)

        s_r = self.poynting_avg(r, above=direction == -1)
        s_ir = self.poynting_avg(ir, above=direction == -1)
        s_t = self.poynting_avg(t, above=direction == 1)
        return s_t / (s_ir - s_r), s_r / (s_r - s_ir)

    def cd(self, illu, direction=1):
        """
        Transmission and absorption circular dichroism

        Calculate the transmission and absoption CD for one Q-matrix with the given
        illumination and direction.

        Args:
            illu (complex, array_like): Expansion coefficients for the incoming light
            direction (int, optional): The direction of the field, options are `-1` and `1`

        Returns:
            tuple
        """
        minus, plus = self.pol == 0, self.pol == 1
        illu = np.array(illu)
        if self.helicity:
            illuopposite = np.zeros_like(illu)
            illuopposite[minus] = illu[plus]
            illuopposite[plus] = illu[minus]
        else:
            illuopposite = copy.deepcopy(illu)
            illuopposite[minus] *= -1
        tm, rm = self.tr(illu, direction=direction)
        tp, rp = self.tr(illuopposite, direction=direction)
        return (tp - tm) / (tp + tm), (tp + rp - tm - rm) / (tp + rp + tm + rm)

    def periodic(self):
        r"""
        Periodic repetition of the Q-Matrix

        Transform the Q-matrix to an infinite periodic arrangement of itself defined
        by

        .. math::

            \begin{pmatrix}
                Q_{\uparrow \uparrow} & Q_{\uparrow \downarrow} \\
                -Q_{\downarrow \downarrow}^{-1} Q_{\downarrow \uparrow} Q_{\uparrow \uparrow} &
                Q_{\downarrow \downarrow}^{-1} (\mathbb{1} - Q_{\downarrow \uparrow} Q_{\uparrow \downarrow})
            \end{pmatrix}\,.

        Returns:
            complex, array_like

        """
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

    def bands_kz(self, az):
        r"""
        Band structure calculation

        Calculate the band structure for the given Q-matrix, assuming it is periodically
        repeated along the z-axis. The function returns the z-components of the wave
        vector :math:`k_z` and the corresponding eigenvectors :math:`v` of

        .. math::

            \begin{pmatrix}
                Q_{\uparrow \uparrow} & Q_{\uparrow \downarrow} \\
                -Q_{\downarrow \downarrow}^{-1} Q_{\downarrow \uparrow} Q_{\uparrow \uparrow} &
                Q_{\downarrow \downarrow}^{-1} (\mathbb{1} - Q_{\downarrow \uparrow} Q_{\uparrow \downarrow})
            \end{pmatrix}
            \boldsymbol v
            =
            \mathrm{e}^{\mathrm{i}k_z a_z}
            \boldsymbol v\,.

        Args:
            az (float): Lattice pitch along the z direction

        Returns:
            tuple

        """
        w, v = np.linalg.eig(self.periodic())
        return -1j * np.log(w) / az, v

    def _check_modes(self, modes):
        """_check_modes"""
        if len(modes) != 3:
            raise ValueError(
                f"invalid length of variable modes {len(modes)}, must be 3 or 4"
            )
        modes = (*(np.array(a) for a in modes),)
        if not np.all([m.size == modes[0].size for m in modes[1:]]):
            raise ValueError("all modes need equal size")
        return modes


def _coeff_chirality_density(k, kz, a, dira, b, dirb, z=None, helicity=True):
    z = (0, 0) if z is None else z
    tmp = (dira * kz - dirb * kz.conjugate()) * 0.5
    pref = np.exp(1j * tmp * (z[1] + z[0]))
    if np.abs(z[1] - z[0]) > 1e-16:
        pref *= np.sin(tmp * (z[1] - z[0])) / (tmp * (z[1] - z[0]))
    pref *= 1 + (k * k - kz * (kz - dira * dirb * kz.conjugate())) / (k * k.conjugate())
    if helicity:
        return np.sum(a * b.conjugate() * pref)
    return 2 * np.sum(np.real(a * b.conjugate()) * pref)
