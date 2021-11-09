import warnings

import numpy as np

import ptsa.lattice as la
import ptsa.special as sc
from ptsa import cw, misc, pw, sw
from ptsa._tmatrix_base import TMatrixBase
from ptsa.coeffs import mie


class TMatrix(TMatrixBase):
    """
    T-matrix for spherical modes

    The T-matrix is square, with the modes defined in the corresponding fields. The
    order of the T-matrix can be arbitrary, but the normalization is fixed to that of
    the modes defined in :func:`ptsa.special.vsw_A`, :func:`ptsa.special.vsw_M`, and
    :func:`ptsa.special.vsw_N`. A default order according to :func:`TMatrix.defaultmodes`
    is assumed if not specified. Helicity and parity modes are possible, but not mixed.

    The embedding medium is described by permittivity, permeability and the chirality
    parameter.

    The T-matrix can be global or local. For a local T-matrix multiple positions have to
    be specified. Also modes must have as first element a position index.

    Args:
        tmat (float or complex, array): T-matrix itself
        k0 (float): Wave number in vacuum
        epsilon (float or complex, optional): Relative permittivity of the embedding medium
        mu (complex): Relative permeability of the embedding medium
        kappa (complex): Chirality parameter of the embedding medium
        positions (float, (3,)- or (M,3)-array): Positions for a local T-matrix
        helicity (bool): Helicity or parity modes
        modes (iterable): Sorting of the T-matrix entries. Either four entries for
            local T-Matrices, with the first specifying the corresponding position
            or three entries only, specifying degree, order, and polarization.

    Attributes:
        t (float or complex, (N,N)-array): T-matrix
        k0 (float): Wave number in vacuum
        positions (float, (3,)- or (M,3)-array): Positions for a local T-matrix
        epsilon (float or complex, optional): Relative permittivity of the embedding medium
        mu (complex): Relative permeability of the embedding medium
        kappa (complex): Chirality parameter of the embedding medium
        helicity (bool): Helicity or parity modes
        pidx (int, (N,)-array): Position index for each column/row of the T-matrix
        l (int, (N,)-array): Degree of the mode for each column/row of the T-matrix
        m (int, (N,)-array): Order of the mode for each column/row of the T-matrix
        pol (int, (N,)-array): Polarization of the mode for each column/row of the T-matrix
        ks (float or complex (2,)-array): Wave numbers in the medium for both polarizations
    """

    def __init__(
        self,
        tmat,
        k0,
        epsilon=1,
        mu=1,
        kappa=0,
        positions=None,
        helicity=True,
        modes=None,
    ):
        super().__init__(tmat, k0, epsilon, mu, kappa, positions, helicity, modes)
        if modes is None:
            modes = TMatrix.defaultmodes(
                TMatrix.defaultlmax(self.t.shape[0], self.positions.shape[0]),
                self.positions.shape[0],
            )
        modes = self._check_modes(modes)
        if modes[0].size != self.t.shape[0]:
            raise ValueError("dimensions of modes and T-matrix do not match")
        self.pidx, self.l, self.m, self.pol = modes

    @classmethod
    def sphere(cls, lmax, k0, radii, epsilon, mu=None, kappa=None):
        """
        T-Matrix of a sphere

        Args:
            lmax (int): Positive integer for the maximum degree of the T-matrix
            k0 (int): Wave number in vacuum
            radii (float array): Radii from inside to outside of the sphere and
                multiple shells
            epsilon, mu, kappa (complex arrays): Relative permittivity, permeability
                and chirality parameter from inside to outside

        Returns:
            TMatrix
        """
        epsilon = np.array(epsilon)
        if mu is None:
            mu = np.ones_like(epsilon)
        if kappa is None:
            kappa = np.zeros_like(epsilon)
        mu = np.array(mu)
        kappa = np.array(kappa)
        radii = np.array(radii)
        if (
            np.any([i.ndim != 1 for i in (epsilon, mu, kappa, radii)])
            or not epsilon.size == mu.size == kappa.size == radii.size + 1
        ):
            ValueError(
                f"dimensions of radii and material parameters do not match, got {radii.size}, {epsilon.size}, {mu.size}, and {kappa.size}"
            )
        dim = TMatrix.defaultdim(lmax)
        tmat = np.zeros((dim, dim), np.complex128)
        for l in range(1, lmax + 1):
            miecoeffs = mie(l, k0 * radii, epsilon, mu, kappa)
            pos = TMatrix.defaultdim(l - 1)
            for i in range(2 * l + 1):
                tmat[
                    pos + 2 * i : pos + 2 * i + 2, pos + 2 * i : pos + 2 * i + 2
                ] = miecoeffs
        return cls(
            tmat, k0, epsilon[-1], mu[-1], kappa[-1], modes=TMatrix.defaultmodes(lmax)
        )

    @property
    def xsext_avg(self):
        """
        Rotational average of the extinction cross section

        Only implemented for global T-matrices and in achiral media.

        Returns:
            float or complex
        """
        if self.positions.shape[0] > 1:
            raise NotImplementedError
        if not self.kappa == 0:
            raise NotImplementedError(
                "Cross section in chiral embedding is not implemented"
            )
        res = (
            -2
            * np.pi
            * np.real(np.trace(self.t))
            / (self.k0 * self.k0 * self.epsilon * self.mu)
        )
        if res.imag == 0:
            return res.real
        return res

    @property
    def xssca_avg(self):
        """
        Rotational average of the scattering cross section

        Only implemented for global T-matrices and in achiral media.

        Returns:
            float or complex
        """
        if self.positions.shape[0] > 1:
            raise NotImplementedError
        if not self.kappa == 0:
            raise NotImplementedError(
                "Cross section in chiral embedding is not implemented"
            )
        res = (
            2
            * np.pi
            * np.sum(self.t.real * self.t.real + self.t.imag * self.t.imag)
            / (self.k0 * self.k0 * self.epsilon * self.mu)
        )
        if res.imag == 0:
            return res.real
        return res

    @property
    def cd(self):
        """
        Circular dichroism

        Only implemented for global T-matrices.
        """
        if self.positions.shape[0] > 1:
            raise NotImplementedError
        selections = self.pol == 0, self.pol == 1
        tpp = self.t[np.ix_(selections[1], selections[1])]
        tpm = self.t[np.ix_(selections[1], selections[0])]
        tmp = self.t[np.ix_(selections[0], selections[1])]
        tmm = self.t[np.ix_(selections[0], selections[0])]
        plus = tpp + tpp.T.conj() + 2 * tpp.T.conj() @ tpp + 2 * tmp.T.conj() @ tmp
        minus = tmm + tmm.T.conj() + 2 * tmm.T.conj() @ tmm + 2 * tpm.T.conj() @ tpm
        return 2 * np.pi * np.real(np.trace(minus) - np.trace(plus))

    @property
    def chi(self):
        """
        Electromagnetic chirality

        Only implemented for global T-matrices.

        Returns:
            float
        """
        if self.positions.shape[0] > 1:
            raise NotImplementedError
        selections = self.pol == 0, self.pol == 1
        _, spp, _ = np.linalg.svd(self.t[np.ix_(selections[1], selections[1])])
        _, spm, _ = np.linalg.svd(self.t[np.ix_(selections[1], selections[0])])
        _, smp, _ = np.linalg.svd(self.t[np.ix_(selections[0], selections[1])])
        _, smm, _ = np.linalg.svd(self.t[np.ix_(selections[0], selections[0])])
        plus = np.concatenate((spp, spm))
        minus = np.concatenate((smm, smp))
        return np.linalg.norm(plus - minus) / np.sqrt(
            np.sum(np.power(np.abs(self.t), 2))
        )

    @property
    def db(self):
        """
        Duality breaking

        Only implemented for global T-matrices.

        Returns:
            float
        """
        if self.positions.shape[0] > 1:
            raise NotImplementedError
        selections = self.pol == 0, self.pol == 1
        tpm = self.t[np.ix_(selections[1], selections[0])]
        tmp = self.t[np.ix_(selections[0], selections[1])]
        return np.sum(
            tpm.real * tpm.real
            + tpm.imag * tpm.imag
            + tmp.real * tmp.real
            + tmp.imag * tmp.imag
        ) / np.sqrt(np.sum(np.power(np.abs(self.t), 2)))

    @property
    def modes(self):
        """
        Modes of the T-matrix

        Degree, order, and polarization of each row/column in the T-matrix

        Returns:
            3-tuple
        """
        return self.l, self.m, self.pol

    def xs(self, illu, flux=1):
        """
        Scattering and extinction cross section

        Possible for all T-matrices (global and local) in non-absorbing embedding.

        Args:
            illu (complex, array): Illumination coefficients
            flux (optional): Ingoing flux correspondin to the illumination. Used for
                the result's normalization.

        Returns:
            float, (2,)-tuple
        """
        if np.any(np.imag(self.ks)) != 0:
            raise NotImplementedError
        p = self.t @ illu
        rs = sc.car2sph(self.positions[:, None, :] - self.positions)
        m = (
            sw.translate(
                *(m[:, None] for m in self.modes),
                *self.modes,
                self.ks[self.pol] * rs[self.pidx[:, None], self.pidx, 0],
                rs[self.pidx[:, None], self.pidx, 1],
                rs[self.pidx[:, None], self.pidx, 2],
                self.helicity,
                singular=False,
            )
            / (np.power(self.ks[self.pol], 2) * flux)
        )
        return (
            np.real(np.diag(p.conjugate().T @ m @ p)),
            -np.real(np.diag(illu.conjugate().T @ m @ p)),
        )

    def rotate(self, phi, theta, psi, modes=None):
        """
        Rotate the T-Matrix by the euler angles

        Rotation is done in-place. If you need the original T-Matrix make a deepcopy first.
        Rotations can only be applied to global T-Matrices. The angles are given in the
        zyz-convention. In the intrinsic (object fixed coordinate system) convention the
        rotations are applied in the order phi first, theta second, psi third. In the
        extrinsic (global or reference frame fixed coordinate system) the rotations are
        applied psi first, theta second, phi third.

        Args:
            phi, theta, psi (float): Euler angles
            modes (array): While rotating also take only specific output modes into
            account

        Returns:
            TMatrix
        """
        if modes is None:
            modes = self.modes
        modes = self._check_modes(modes)
        if self.positions.shape[0] > 1 and np.any(modes[0] != 0):
            raise NotImplementedError
        mat = sw.rotate(*(m[:, None] for m in self.modes), *modes[1:], phi, theta, psi)
        self.t = mat @ self.t @ mat.conjugate().T
        self.pidx, self.l, self.m, self.pol = modes
        return self

    def translate(self, rvec, modes=None):
        """
        Translate the origin of the T-Matrix

        Translation is done in-place. If you need the original T-Matrix make a copy first.
        Translations can only be applied to global T-Matrices.

        Args:
            rvec (float array): Translation vector
            modes (array): While translating also take only specific output modes into
            account

        Returns:
            TMatrix
        """
        if modes is None:
            modes = self.modes
        modes = self._check_modes(modes)
        if self.positions.shape[0] > 1 or np.any(modes[0] != 0):
            raise NotImplementedError
        rvec = np.array(rvec)
        if rvec.ndim == 1:
            rvec = np.reshape(rvec, (1, -1))
        rs = sc.car2sph(rvec - self.positions)
        if self.kappa == 0:
            kr = np.array([self.ks[0] * rs[0, 0]])
        else:
            kr = self.ks[self.pol] * rs[0, 0]
        matin = sw.translate(
            *(m[:, None] for m in self.modes),
            *modes[1:],
            kr[:, None],
            rs[0, 1],
            rs[0, 2],
            self.helicity,
            False,
        )
        matout = sw.translate(
            *(m[:, None] for m in modes[1:]),
            *self.modes,
            kr,
            np.pi - rs[0, 1],
            rs[0, 2] + np.pi,
            self.helicity,
            False,
        )
        self.t = matout @ self.t @ matin
        self.pidx, self.l, self.m, self.pol = modes
        return self

    def coupling(self):
        """
        Calculate the coupling term of a blockdiagonal T-matrix

        Returns:
            TMatrix
        """
        rs = sc.car2sph(self.positions[:, None, :] - self.positions)
        m = sw.translate(
            *(m[:, None] for m in self.modes),
            *self.modes,
            self.ks[self.pol] * rs[self.pidx[:, None], self.pidx, 0],
            rs[self.pidx[:, None], self.pidx, 1],
            rs[self.pidx[:, None], self.pidx, 2],
            self.helicity,
        )
        return np.eye(self.t.shape[0]) - self.t @ m

    def globalmat(self, origin=None, modes=None, interacted=True):
        """
        Global T-matrix

        Calculate the global T-matrix starting from a local one. This changes the
        T-matrix.

        Args:
            origin (array, optional): The origin of the new T-matrix
            modes (array, optional): The modes that are considered for the global T-matrix
            interacted (bool, optional): If set to `False` the interaction is calulated first.

        Returns
            TMatrix
        """
        if origin is None:
            origin = np.zeros((3,))
        origin = np.reshape(origin, (1, 3))
        if modes is None:
            zeroth = self.pidx == 0
            modes = self.l[zeroth], self.m[zeroth], self.pol[zeroth]
        modes = self._check_modes(modes)
        rs = sc.car2sph(self.positions - origin)
        ain = sw.translate(
            *(m[:, None] for m in self.modes),
            *modes[1:],
            self.ks[self.pol[:, None]] * rs[self.pidx, :1],
            rs[self.pidx, 1:2],
            rs[self.pidx, 2:],
            self.helicity,
            False,
        )
        pout = sw.translate(
            *(m[:, None] for m in modes[1:]),
            *self.modes,
            self.ks[self.pol] * rs[self.pidx, 0],
            np.pi - rs[self.pidx, 1],
            rs[self.pidx, 2] + np.pi,
            self.helicity,
            False,
        )
        if interacted:
            self.t = pout @ self.t @ ain
        else:
            self.t = pout @ np.linalg.solve(self.coupling(), self.t @ ain)
        self.pidx, self.l, self.m, self.pol = modes
        self.positions = origin
        return self

    def illuminate_pw(self, kx, ky, kz, pol):
        """
        Illuminate with a plane wave

        Args:
            kx (float, scalar or (N,)-array): X component of the wave vector
            ky (float, scalar or (N,)-array): Y component of the wave vector
            kz (float or complex, scalar or (N,)-array): Z component of the wave vector
            pol (int, scalar or (N,)-array): Polarization of wave, corresponding to
                the attribute `TMatrix.helicity`
        """
        pos = (*(i[:, None] for i in self.positions[self.pidx, :].T),)
        return pw.to_sw(
            *(m[:, None] for m in self.modes), kx, ky, kz, pol, helicity=self.helicity
        ) * pw.translate(kx, ky, kz, *pos)

    def illuminate_cw(self, kz, m, pol, l=None):
        """Illuminate with a cylindrical wave"""
        # The question is how to implement this. Do you specify one cw at the origin,
        # that is translated to the different places, or do you specify "local" cws.
        # In the former case, how do you choose the order.
        # --> Split in local and global part
        raise NotImplementedError

    def illuminate_cw_local(self, pidx, kz, m, pol):
        """Illuminate with a local cylindrical wave"""
        return cw.to_sw(
            *(m[:, None] for m in self.modes),
            kz,
            m,
            pol,
            self.ks[pol],
            posout=self.pidx[:, None],
            posin=pidx,
            helicity=self.helicity,
        )

    def field(self, r, scattered=True):
        """
        Calculate the scattered or incident field at a specified points

        The mode expansion of the T-matrix is used

        Args:
            r (float, array_like): Array of the positions to probe
            scattered (bool, optional): Select the scattered (default) or incident field

        Returns
            complex
        """
        r = np.array(r)
        if r.ndim == 1:
            r = np.reshape(r, (1, -1))
        r_sph = sc.car2sph(r[..., None, :] - self.positions)
        if scattered:
            wave_A = sc.vsw_A
            wave_M = sc.vsw_M
            wave_N = sc.vsw_N
        else:
            wave_A = sc.vsw_rA
            wave_M = sc.vsw_rM
            wave_N = sc.vsw_rN
        if self.helicity:
            res = wave_A(
                *self.modes[:2],
                self.ks[self.pol] * r_sph[..., self.pidx, 0],
                r_sph[..., self.pidx, 1],
                r_sph[..., self.pidx, 2],
                self.pol,
            )
        else:
            res = (1 - self.pol[:, None]) * wave_M(
                *self.modes[:2],
                self.ks[self.pol] * r_sph[..., self.pidx, 0],
                r_sph[..., self.pidx, 1],
                r_sph[..., self.pidx, 2],
            ) + self.pol[:, None] * wave_N(
                *self.modes[:2],
                self.ks[self.pol] * r_sph[..., self.pidx, 0],
                r_sph[..., self.pidx, 1],
                r_sph[..., self.pidx, 2],
            )
        return sc.vsph2car(res, r_sph[..., self.pidx, :])

    def latticecoupling(self, kpar, a, eta=0):
        """
        The coupling of the T-matrix in a lattice

        Returns

        .. math::

            \mathbb 1 - T C

        The inverse of this multiplied to the T-matrix in `latticeinteract`. The lattice
        type is inferred from `kpar`.

        Args:
            kpar (float): The parallel component of the T-matrix
            a (array): Definition of the lattice
            eta (float or complex, optional): Splitting parameter in the lattice summation

        Returns:
            complex, array
        """
        m = sw.translate_periodic(
            self.ks,
            kpar,
            a,
            self.positions,
            self.fullmodes,
            helicity=self.helicity,
            eta=eta,
        )
        return np.eye(self.t.shape[0]) - self.t @ m

    def lattice_field(self, r, modes, kpar, a, eta=0):
        """
        Field expansion at a specified point in a lattice

        Args:
            r (float, array_like): Positions
            modes (tuple): Modes of the expansion
            kpar (float, array_like: Parallel component of the wave vector
            a (float, array_like): Lattice vectors
            eta (float or complex, optional): Splitting parameter in the lattice summation

        Returns:
            complex
        """
        r = np.array(r)
        if r.ndim == 1:
            r = r.reshape((1, -1))
        return sw.translate_periodic(
            self.ks,
            kpar,
            a,
            r,
            modes,
            in_=self.fullmodes,
            rsin=self.positions,
            helicity=self.helicity,
            eta=eta,
        )

    def array1d(self, modes, a, eta=0):
        """
        Convert a one-dimensional array of T-matrices into a (cylindrical) 2D-T-matrix

        Args:
            modes (tuple): Cylindrical wave modes
            a (float): Lattice pitch
            eta (float or complex, optional): Splitting parameter in the lattice summation

        Returns:
            complex, array
        """
        pidx, kz, ms, pol = self._check_modes(modes)
        minkz = misc.firstbrillouin1d(kz[0], 2 * np.pi / a)
        interaction = np.linalg.solve(self.latticecoupling(minkz, a, eta), self.t)
        ain = self.illuminate_cw_local(pidx, kz, ms, pol)
        pout = sw.periodic_to_cw(
            kz[:, None],
            ms[:, None],
            pol[:, None],
            *self.modes,
            self.ks[self.pol],
            a,
            pidx[:, None],
            self.pidx,
            self.helicity,
        )
        return pout @ interaction @ ain

    def array2d(self, kx, ky, kz, pwpol, a, origin=None, eta=0):
        """
        Convert a two-dimensional array of T-matrices into a Q-matrix

        Unlike for the 1d-case there is no local Q-matrix used, so the result is taken
        with respect to the reference origin.

        Args:
            kx (float, array_like): X component of the plane wave
            ky (float, array_like): Y component of the plane wave
            kz (float, array_like): Z component of the plane wave
            pwpol (int, array_like): Plane wave polarizations
            a (float, (2,2)-array): Lattice vectors
            origin (float, (3,)-array, optional): Reference origin of the result
            eta (float or complex, optional): Splitting parameter in the lattice summation

        Returns:
            complex, array
        """
        kpar = misc.firstbrillouin2d([kx[0], ky[0]], la.reciprocal(a))
        interaction = np.linalg.solve(self.latticecoupling(kpar, a, eta), self.t)
        if origin is None:
            origin = np.zeros((3,))
        posdiff = self.positions - origin
        tout = pw.translate(
            kx[:, None],
            ky[:, None],
            kz[:, None],
            -posdiff[self.pidx, 0],
            -posdiff[self.pidx, 1],
            -posdiff[self.pidx, 2],
        )
        ain = self.illuminate_pw(kx, ky, kz, pwpol)
        pout = sw.periodic_to_pw(
            kx[:, None],
            ky[:, None],
            kz[:, None],
            pwpol[:, None],
            *self.modes,
            la.area(a),
            helicity=self.helicity,
        )
        return (tout * pout) @ interaction @ ain

    @staticmethod
    def defaultmodes(lmax, nmax=1):
        """
        Default sortation of modes

        Default sortation of the T-Matrix entries, including degree `l`, order `m` and
        polarization `p`.

        Args:
            lmax (int): Maximal value of `l`
            nmax (int, optional): Number of particles, defaults to `1`

        Returns:
            tuple
        """
        return (
            *np.array(
                [
                    [n, l, m, p]
                    for n in range(0, nmax)
                    for l in range(1, lmax + 1)
                    for m in range(-l, l + 1)
                    for p in range(1, -1, -1)
                ]
            ).T,
        )

    @staticmethod
    def defaultlmax(dim, nmax=1):
        """
        Default maximal degree

        Given the dimension of the T-matrix return the estimated maximal value of `l`. This
        is the inverse of defaultdim. A value of zero is allowed for empty T-matrices.

        Args:
            dim (int): Dimension of the T-matrix, respectively number of modes
            nmax (int, optional): Number of particles, defaults to `1`

        Returns:
            int
        """
        res = np.sqrt(1 + dim * 0.5 / nmax) - 1
        res_int = np.int(np.rint(res))
        if np.abs(res - res_int) > 1e-8 * np.maximum(np.abs(res), np.abs(res_int)):
            raise ValueError("cannot estimate the default lmax")
        return res_int

    @staticmethod
    def defaultdim(lmax, nmax=1):
        """
        Default dimension

        Given the maximal value of `l` return the size of the corresponding T-matrix. This
        is the inverse of defaultlmax. A value of zero is allowed.

        Args:
            lmax (int): Maximal value of `l`
            nmax (int, optional): Number of particles, defaults to `1`

        Returns:
            int
        """
        # zero is allowed and won't give an error
        if lmax < 0 or nmax < 0:
            raise ValueError("maximal order must be positive")
        return 2 * lmax * (lmax + 2) * nmax

    def to_clsT(self):
        """
        Convert T-matrix to an object of class clsT (Xavi's code)

        Returns:
            object of class clsCluster
        """

        from clsT import clsT

        if self.positions.shape[0] > 1:
            warnings.warn(
                """
                    This T-matrix can not be converted into an object of class clsT as
                    it contains more than one scatterer. An object of class clsCluster
                    will be returned instead
                """
            )
            return self.to_clsCluster()
        if self.helicity:
            basis = "helicity"
        else:
            basis = "electric-magnetic"
        n_max = np.max(self.modes[0])
        modes = (
            *np.array(
                [
                    [n, m, p]
                    for p in range(1, -1, -1)
                    for n in range(1, n_max + 1)
                    for m in range(-n, n + 1)
                ]
            ).T,
        )
        mat = misc.pickmodes(self.modes, modes)
        tmatrix = mat.T @ self.t @ mat
        return clsT(
            tmatrix,
            basis,
            wavelength=2 * np.pi / self.k0,
            refr_index=np.sqrt(self.epsilon * self.mu),
        )

    def to_clsCluster(self):
        """
        Convert T-matrix to an object of class clsCluster (Xavi's code)

        Returns:
            object of class clsCluster
        """

        from clsCluster import clsCluster
        from clsT import clsT

        if self.helicity:
            basis = "helicity"
        else:
            basis = "electric-magnetic"
        number_particles = self.positions.shape[0]
        max_n = np.zeros((number_particles,), int)
        for i_particle in range(number_particles):
            max_n[i_particle] = np.max(self.l[self.pidx == i_particle])
        modes = (
            *np.array(
                [
                    [np, n, m, p]
                    for np in range(number_particles)
                    for p in range(1, -1, -1)
                    for n in range(1, max_n[np] + 1)
                    for m in range(-n, n + 1)
                ]
            ).T,
        )
        mat = misc.pickmodes(self.fullmodes, modes)
        tmatrix = mat.T @ self.t @ mat
        ts = []
        for i_particle in range(number_particles):
            entries = modes[0] == i_particle
            ts.append(
                clsT(
                    tmatrix[entries, :][:, entries],
                    basis,
                    wavelength=2 * np.pi / self.k0,
                    refr_index=np.sqrt(self.epsilon * self.mu),
                )
            )
        cluster = clsCluster(self.positions, ts)
        return cluster
