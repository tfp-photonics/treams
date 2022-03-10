import warnings

import numpy as np

class BasisSet:
    def modes(self):
        raise NotImplementedError

    @classmethod
    def default(self):
        raise NotImplementedError

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, lattice):
        if lattice is None:
            self._periodic = lattice
            return
        lattice = np.array(lattice)
        shape = lattice.shape
        lattice = np.squeeze(lattice) if lattice.size = 1 else lattice
        if (
                lattice.ndim not in (2, 0)
                or (lattice.size == 1 and lattice.ndim > 0)
                or (lattice.ndim == 2 and (lattice.shape[0] != lattice.shape[1] or lattice.shape[0] not in (2, 3))
        ):
            raise ValueError(f"invalid shape of lattice '{shape}'")
        if self._periodic is not None:
            warnings.warn(f"overwriting value for periodicity '{self._periodic}'")
        self._periodic = lattice

class SphericalWaveBasis(BasisSet):
    def __init__(self, pidx, l, m, pol=None, /, positions=None, helicity=True, wavetype=None):
        if pol is None:
            l, m, pol = pidx, l, m
            pidx = np.zeros_like(l)
        if positions is None:
            positions = np.zeros((3, 1))
        if positions.ndim == 1:
            positions = positions[:, None]
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(f"invalid shape of positions {positions.shape}")
        if wavetype not in (None, 'incident', 'scattered'):
            raise ValueError(f"invalid wavetype '{wavetype}'")
        pidx, l, m, pol = [np.atleast_1d(i) for i in (pidx, l, m, pol)]
        for i in (pidx, l, m, pol):
            if i.ndim > 1:
                raise ValueError("invalid shape of parameters")
        pidx, l, m, pol = np.broadcast_arrays(pidx, l, m, pol)
        self.pidx, self.l, self.m, self.pol = [np.array(i, int) for i in (pidx, l, m, pol)]
        for i, j in ((self.pidx, pidx), (self.l, l), (self.m, m), (self.pol, pol)):
            if np.any(i != j):
                raise ValueError("parameters must be integer")
        if np.any(self.l < 1):
            raise ValueError("'l' must be a strictly positive integer")
        if np.any(self.l < np.abs(self.m)):
            raise ValueError("'|m|' cannot be larger than 'l'")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")
        self.wavetype = wavetype
        self.helicity = bool(helicity)
        self.positions = positions
        self._periodic = None

    @classmethod
    def diffr_orders(cls, kx, ky, lattice, rmax):
        raise NotImplementedError

    @classmethod
    def diffr_orders_grid(cls, kx, ky, lattice, n):
        raise NotImplementedError

    @property
    def lm(self):
        return self.l, self.m

    @property
    def lmp(self):
        return self.l, self.m, self.pol

    @property
    def plmp(self):
        return self.pidx, self.l, self.m, self.pol

    modes = plmp

    @classmethod
    def default(cls, lmax, nmax=1, *args, **kwargs):
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
        pidx, l, m, p = np.array(
            [
                [n, l, m, p]
                for n in range(0, nmax)
                for l in range(1, lmax + 1)
                for m in range(-l, l + 1)
                for p in range(1, -1, -1)
            ]
        ).T
        return cls(pidx, l, m, p, *args, **kwargs)


    @classmethod
    def ebcm(cls, lmax, nmax=1, mmax=None):
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
        mmax = lmax if mmax is None else mmax
        pidx, l, m, p = np.array(
            [
                [n, l, m, p]
                for n in range(0, nmax)
                for m in range(-mmax, mmax + 1)
                for l in range(max(abs(m), 1), lmax + 1)
                for p in range(1, -1, -1)
            ]
        ).T
        return cls(pidx, l, m, p, *args)

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
        res_int = int(np.rint(res))
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


class PlaneWaveBasis(BasisSet):
    def __init__(self, kx=None, ky=None, kz=None, pol=None, helicity=True, wavetype=None):
        if wavetype not in (None, 'up', 'down'):
            raise ValueError(f"invalid wavetype '{wavetype}'")
        if sum([i is None for i in (kx, ky, kz)]) > int(wavetype is not None):
            raise ValueError("insufficient wave vector definition")
        if pol is None:
            raise ValueError("polarizations undefined")
        kx, ky, kz, pol = [i if i is None else np.atleast_1d(i) for i in (kx, ky, kz, pol)]
        bc = np.broadcast_arrays(kx, ky, kz, pol)
        kx, ky, kz, pol = [i if i is None else j for i, j in zip([kx, ky, kz, pol], bc)]
        self._kx, self._ky, self._kz = kx, ky, kz
        self.pol = np.array(pol, int)
        if np.any(self.pol != pol):
            raise ValueError("polarizations must be integer")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")
        self.wavetype = wavetype
        self.helicity = bool(helicity)
        self._periodic = None

    @property
    def kx(self):
        if self._kx is None:
            try:
                ks = np.array(self.ks[self.pol], complex)
            except AttributeError:
                raise AttributeError("undefined wave number cannot determine 'kx'")
            res = np.sqrt(ks * ks - self._ky * self._ky - self._kz * self._kz)
            res[np.imag(res) < 0] *= -1
            if self.wavetype == 'down':
                res = -res
            return res
        return self._kx

    @property
    def ky(self):
        if self._ky is None:
            try:
                ks = np.array(self.ks[self.pol], complex)
            except AttributeError:
                raise AttributeError("undefined wave number cannot determine 'ky'")
            res = np.sqrt(ks * ks - self._kx * self._kx - self._kz * self._kz)
            res[np.imag(res) < 0] *= -1
            if self.wavetype == 'down':
                res = -res
            return res
        return self._kx

    @property
    def kz(self):
        if self._kz is None:
            try:
                ks = np.array(self.ks[self.pol], complex)
            except AttributeError:
                raise AttributeError("undefined wave number cannot determine 'kz'")
            res = np.sqrt(ks * ks - self._kx * self._kx - self._ky * self._ky)
            res[np.imag(res) < 0] *= -1
            if self.wavetype == 'down':
                res = -res
            return res
        return self._kz

    @property
    def kvec(self):
        return np.stack((self.kx, self.ky, self.kz), axis=-1)

    @property
    def modes(self):
        return kx, ky, kz, pol
