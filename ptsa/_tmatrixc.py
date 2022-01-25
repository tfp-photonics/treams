import numpy as np

import ptsa.special as sc
from ptsa import cw, misc, pw
from ptsa._tmatrix_base import TMatrixBase
from ptsa.coeffs import mie_cyl


class TMatrixC(TMatrixBase):
    """
    T-matrix for cylindrical modes

    The T-matrix is square, with the modes defined in the corresponding fields. The
    order of the T-matrix can be arbitrary, but the normalization is fixed to that of
    the modes defined in :func:`ptsa.special.vcw_A`, :func:`ptsa.special.vcw_M`, and
    :func:`ptsa.special.vcw_N`. A default order according to :func:`TMatrixC.defaultmodes`
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
            modes = TMatrixC.defaultmodes(
                TMatrixC.defaultmmax(self.t.shape[0], self.positions.shape[0]),
                self.positions.shape[0],
            )
        modes = self._check_modes(modes)
        if modes[0].size != self.t.shape[0]:
            raise ValueError("dimensions of modes and T-matrix do not match")
        self.pidx, self.kz, self.m, self.pol = modes

    @classmethod
    def cylinder(cls, kzs, mmax, k0, radii, epsilon, mu=None, kappa=None):
        """
        T-Matrix of a cylinder

        Args:
            kzs (float, array_like): Z component of the cylindrical wave
            mmax (int): Positive integer for the maximum order of the T-matrix
            k0 (int): Wave number in vacuum
            radii (float array): Radii from inside to outside of the sphere and
                multiple shells
            epsilon, mu, kappa (complex arrays): Relative permittivity, permeability
                and chirality parameter from inside to outside

        Returns:
            T-Matrix object
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
            raise ValueError(
                f"dimensions of radii and material parameters do not match, got {radii.size}, {epsilon.size}, {mu.size}, and {kappa.size}"
            )
        dim = TMatrixC.defaultdim(len(kzs), mmax)
        tmat = np.zeros((dim, dim), np.complex128)
        idx = 0
        for kz in kzs:
            for m in range(-mmax, mmax + 1):
                miecoeffs = mie_cyl(kz, m, k0, radii, epsilon, mu, kappa)
                tmat[idx : idx + 2, idx : idx + 2] = miecoeffs
                idx += 2
        return cls(
            tmat,
            k0,
            epsilon[-1],
            mu[-1],
            kappa[-1],
            modes=TMatrixC.defaultmodes(kzs, mmax),
        )

    @classmethod
    def array(cls, tmat, kzs, a, eta=0):
        """1d array of spherical T-matrices"""
        allmodes = ([], [], [], [])
        for i in range(tmat.positions.shape[0]):
            mmax = np.max(np.abs(tmat.m[tmat.pidx == i]))
            modes = TMatrixC.defaultmodes(kzs, mmax)
            modes[0][:] = i
            for j in range(4):
                allmodes[j].extend(modes[j])
        tm = tmat.array1d(allmodes, a, eta)
        return cls(
            tm,
            tmat.k0,
            tmat.epsilon,
            tmat.mu,
            tmat.kappa,
            tmat.positions,
            tmat.helicity,
            allmodes,
        )

    @property
    def modes(self):
        """
        Modes of the T-matrix

        Z component of the wave, order, and polarization for each row/column in the T-matrix.

        Returns:
            3-tuple
        """
        return self.kz, self.m, self.pol

    @property
    def krho(self):
        r"""
        Radial part of the wave

        Calculate :math:`\sqrt{k^2 - k_z^2}`.

        Returns:
            float or complex, array
        """
        k_square = self.ks * self.ks
        arg = k_square[self.pol] - self.kz * self.kz
        if not np.iscomplexobj(arg) and np.any(arg < 0):
            arg = np.array(arg, complex)
        krhos = np.sqrt(arg)
        krhos[krhos.imag < 0] *= -1
        return krhos

    @property
    def xl_ext(self):
        """Extinction cross length"""
        raise NotImplementedError

    @property
    def xl_sca(self):
        """Scattering cross length"""
        raise NotImplementedError

    def rotate(self, phi, modes=None):
        """
        Rotate the T-Matrix around the z-axis

        Rotation is done in-place. If you need the original T-Matrix make a deepcopy first.
        Rotations can only be applied to global T-Matrices.

        Args:
            phi (float): Rotation angle
            modes (array): While rotating also take only specific output modes into
            account

        Returns:
            T-MatrixC
        """
        if modes is None:
            modes = self.modes
        modes = self._check_modes(modes)
        if self.positions.shape[0] > 1 and np.any(modes[0] != 0):
            raise NotImplementedError
        mat = cw.rotate(*(m[:, None] for m in self.modes), *modes[1:], phi)
        self.t = mat.conjugate().T @ self.t @ mat
        self.pidx, self.kz, self.m, self.pol = modes
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
            TMatrixC
        """
        if modes is None:
            modes = self.modes
        modes = self._check_modes(modes)
        if self.positions.shape[0] > 1 or np.any(modes[0] != 0):
            raise NotImplementedError
        rvec = np.array(rvec)
        rs = sc.car2cyl(rvec - self.positions)
        if rvec.ndim == 1:
            rvec = np.reshape(rvec, (1, -1))
        matin = cw.translate(
            *(m[:, None] for m in self.modes),
            *modes[1:],
            self.krho[:, None] * rs[0, 0],
            rs[0, 1],
            rs[0, 2],
            False,
        )
        matout = cw.translate(
            *(m[:, None] for m in modes[1:]),
            *self.modes,
            self.krho * rs[0, 0],
            rs[0, 1] + np.pi,
            -rs[0, 2],
            False,
        )
        self.t = matout @ self.t @ matin
        self.pidx, self.kz, self.m, self.pol = modes
        return self

    def coupling(self):
        """
        Calculate the coupling term of a blockdiagonal T-matrix

        Returns:
            TMatrixC
        """
        rs = sc.car2cyl(self.positions[:, None, :] - self.positions)
        m = cw.translate(
            *(m[:, None] for m in self.modes),
            *self.modes,
            self.krho * rs[self.pidx[:, None], self.pidx, 0],
            rs[self.pidx[:, None], self.pidx, 1],
            rs[self.pidx[:, None], self.pidx, 2],
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
            modes = self.kz[zeroth], self.m[zeroth], self.pol[zeroth]
        modes = self._check_modes(modes)
        rs = sc.car2cyl(self.positions - origin)
        ain = cw.translate(
            *(m[:, None] for m in self.modes),
            *modes[1:],
            self.krho[:, None] * rs[self.pidx, :1],
            rs[self.pidx, 1:2],
            rs[self.pidx, 2:],
            False,
        )
        pout = cw.translate(
            *(m[:, None] for m in modes[1:]),
            *self.modes,
            self.krho * rs[self.pidx, 0],
            rs[self.pidx, 1] + np.pi,
            -rs[self.pidx, 2],
            False,
        )
        if interacted:
            self.t = pout @ self.t @ ain
        else:
            self.t = pout @ np.linalg.solve(self.coupling(), self.t @ ain)
        self.pidx, self.kz, self.m, self.pol = modes
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
        return pw.to_cw(
            *(m[:, None] for m in self.modes), kx, ky, kz, pol,
        ) * pw.translate(kx, ky, kz, *pos)

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
        r_cyl = sc.car2cyl(r[..., None, :] - self.positions)
        if scattered:
            wave_A = sc.vcw_A
            wave_M = sc.vcw_M
            wave_N = sc.vcw_N
        else:
            wave_A = sc.vcw_rA
            wave_M = sc.vcw_rM
            wave_N = sc.vcw_rN
        if self.helicity:
            res = wave_A(
                *self.modes[:2],
                self.krho * r_cyl[..., self.pidx, 0],
                r_cyl[..., self.pidx, 1],
                r_cyl[..., self.pidx, 2],
                self.ks[self.pol],
                self.pol,
            )
        else:
            res = (1 - self.pol[:, None]) * wave_M(
                *self.modes[:2],
                self.krho * r_cyl[..., self.pidx, 0],
                r_cyl[..., self.pidx, 1],
                r_cyl[..., self.pidx, 2],
            ) + self.pol[:, None] * wave_N(
                *self.modes[:2],
                self.krho * r_cyl[..., self.pidx, 0],
                r_cyl[..., self.pidx, 1],
                r_cyl[..., self.pidx, 2],
                self.ks[self.pol],
            )
        return sc.vcyl2car(res, r_cyl[..., self.pidx, :])

    def latticecoupling(self, kpar, a, eta=0):
        r"""
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
        m = cw.translate_periodic(
            self.ks, kpar, a, self.positions, self.fullmodes, eta=eta,
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
        return cw.translate_periodic(
            self.ks,
            kpar,
            a,
            r,
            modes,
            in_=self.fullmodes,
            rsin=self.positions,
            eta=eta,
        )

    def array1d(self, kx, ky, kz, pwpol, a, origin=None, eta=0):
        """
        Convert a one-dimensional array of T-matrices into a Q-matrix

        There is no local Q-matrix used, so the result is taken with respect to the
        reference origin.

        Args:
            kx (float, array_like): X component of the plane wave
            ky (float, array_like): Y component of the plane wave
            kz (float, array_like): Z component of the plane wave
            pwpol (int, array_like): Plane wave polarizations
            a (float): Lattice pitch
            origin (float, (3,)-array, optional): Reference origin of the result
            eta (float or complex, optional): Splitting parameter in the lattice summation

        Returns:
            complex, array
        """
        minkx = misc.firstbrillouin1d(kx[0], 2 * np.pi / a)
        interaction = np.linalg.solve(self.latticecoupling(minkx, a, eta), self.t)
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
        pout = cw.periodic_to_pw(
            kx[:, None], ky[:, None], kz[:, None], pwpol[:, None], *self.modes, a,
        )
        return (tout * pout) @ interaction @ ain

    @staticmethod
    def defaultdim(nkz, mmax, nmax=1):
        """
        Default dimension

        Given the maximal value of `l` return the size of the corresponding T-matrix. This
        is the inverse of defaultmmax. A value of zero is allowed.

        Args:
            nkz (int): Number of z components of the wave
            mmax (int): Maximal value of `m`
            nmax (int, optional): Number of particles, defaults to `1`

        Returns:
            int
        """
        if nkz < 0 or mmax < 0:
            raise ValueError("maximal order must be positive")
        return (2 * mmax + 1) * nkz * 2 * nmax

    @staticmethod
    def defaultmmax(dim, nkz=1, nmax=1):
        """
        Default maximal order

        Given the dimension of the T-matrix return the estimated maximal value of `m`. This
        is the inverse of defaultdim. A value of zero is allowed for empty T-matrices.

        Args:
            dim (int): Dimension of the T-matrix, respectively number of modes
            nkz (int, optional): Number of z components of the wave
            nmax (int, optional): Number of particles, defaults to `1`

        Returns:
            int
        """
        if dim % (2 * nkz * nmax) != 0:
            raise ValueError("cannot estimate the default mmax")
        dim = dim // (2 * nkz * nmax)
        if (dim - 1) % 2 != 0:
            raise ValueError("cannot estimate the default mmax")
        return (dim - 1) // 2

    @staticmethod
    def defaultmodes(kzs, mmax, nmax=1):
        """
        Default sortation of modes

        Default sortation of the T-Matrix entries, including degree `l`, order `m` and
        polarization `p`.

        Args:
            kzs (float, array_like): Z components of the waves
            mmax (int): Maximal value of `m`
            nmax (int, optional): Number of particles, defaults to `1`

        Returns:
            tuple
        """
        res = np.array(
            [
                [n, kz, m, p]
                for n in range(nmax)
                for kz in kzs
                for m in range(-mmax, mmax + 1)
                for p in range(1, -1, -1)
            ]
        )
        return (
            res[:, 0].astype(int),
            res[:, 1],
            res[:, 2].astype(int),
            res[:, 3].astype(int),
        )
