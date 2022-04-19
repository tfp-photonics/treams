import abc
from collections.abc import Sequence, Set
import copy

import numpy as np

import ptsa.lattice as la
from ptsa import misc


class OrderedSet(Sequence, Set):
    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError


# class PolarizationType:
#     _allowed_poltypes = {"helicity", "parity"}
#
#     @property
#     def poltype(self):
#         return self._poltype
#
#     @poltype.setter
#     def poltype(self, val):
#         if val in self._allowed_poltypes:
#             self._poltype = val
#         else:
#             raise ValueError(f"illegal polarization '{val}'")
#
#     def __eq__(self, other):
#         return super().__eq__(other) and self.poltype == other.poltype


class BasisSet(OrderedSet, metaclass=abc.ABCMeta):
    _names = ()

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def default(cls):
        raise NotImplementedError

    def __str__(self):
        return ",\n".join(f"{name}={i}" for name, i in zip(self._names, self()))

    def __repr__(self):
        whitespace = "\n" + " " * (len(self.__class__.__name__) + 1)
        string = str(self).replace("\n", whitespace)
        return (
            f"{self.__class__.__name__}({string})"
        )

    def __len__(self):
        return len(self.pol)


class SphericalWaveBasis(BasisSet):
    _names = ("pidx", "l", "m", "pol")

    def __repr__(self):
        positions = ",\n" + " " * (len(self.__class__.__name__) + 1) + "positions=" + ", ".join(str(self.positions).split()) + ")"
        return super().__repr__()[:-1] + positions

    def __init__(self, modes, positions=None):
        tmp = []
        if isinstance(modes, np.ndarray):
            modes = modes.tolist()
        for m in modes:
            if m not in tmp:
                tmp.append(m)
        modes = tmp
        if len(modes) == 0:
            pidx = []
            l = []
            m = []
            pol = []
        elif len(modes[0]) == 3:
            l, m, pol = (*zip(*modes),)
            pidx = np.zeros_like(l)
        elif len(modes[0]) == 4:
            pidx, l, m, pol = (*zip(*modes),)
        else:
            raise ValueError("invalid shape of modes")

        if positions is None:
            positions = np.zeros((1, 3))
        positions = np.array(positions)
        if positions.ndim == 1:
            positions = positions[None, :]
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"invalid shape of positions {positions.shape}")

        self.pidx, self.l, self.m, self.pol = [
            np.array(i, int) for i in (pidx, l, m, pol)
        ]
        for i, j in ((self.pidx, pidx), (self.l, l), (self.m, m), (self.pol, pol)):
            i.flags.writeable = False
            if np.any(i != j):
                raise ValueError("parameters must be integer")
        if np.any(self.l < 1):
            raise ValueError("'l' must be a strictly positive integer")
        if np.any(self.l < np.abs(self.m)):
            raise ValueError("'|m|' cannot be larger than 'l'")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")
        if np.any(self.pidx >= len(positions)):
            raise ValueError("undefined position is indexed")

        self._positions = positions
        self._positions.flags.writeable = False
        self.hints = {}

    @property
    def positions(self):
        return self._positions

    def __getitem__(self, idx):
        res = self.pidx[idx], self.l[idx], self.m[idx], self.pol[idx]
        if isinstance(idx, int):
            return res
        return type(self)(zip(*res), self.positions)

    def __eq__(self, other):
        return (
            other is not None
            and np.array_equal(self.pidx, other.pidx)
            and np.array_equal(self.l, other.l)
            and np.array_equal(self.m, other.m)
            and np.array_equal(self.pol, other.pol)
            and np.array_equal(self.positions, other.positions)
        )

    def __call__(self, key=None):
        key = "plmp" if key is None else key.lower().strip()
        if key == "plmp":
            return self.pidx, self.l, self.m, self.pol
        elif key == "lmp":
            return self.l, self.m, self.pol
        elif key == "lm":
            return self.l, self.m
        raise ValueError(f"unrecognized key '{key}'")

    @classmethod
    def default(cls, lmax, nmax=1, **kwargs):
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
        modes = [
            [n, l, m, p]
            for n in range(0, nmax)
            for l in range(1, lmax + 1)
            for m in range(-l, l + 1)
            for p in range(1, -1, -1)
        ]
        return cls(modes, **kwargs)

    @classmethod
    def ebcm(cls, lmax, nmax=1, mmax=-1, **kwargs):
        """
        Default sortation of modes

        Default sortation of the T-Matrix entries, including degree `l`, order `m` and
        polarization `p`.

        Args:
            lmax (int): Maximal value of `l`
            nmax (int, optional): Number of particles, defaults to `1`
            mmax (int, optional): Maximal value of `|m|`

        Returns:
            tuple
        """
        mmax = lmax if mmax == -1 else mmax
        modes = [
            [n, l, m, p]
            for n in range(0, nmax)
            for m in range(-mmax, mmax + 1)
            for l in range(max(abs(m), 1), lmax + 1)
            for p in range(1, -1, -1)
        ]
        return cls(modes, **kwargs)

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

    def _from_iterable(cls, it, positions=None):
        hints = {}
        if isinstance(cls, SphericalWaveBasis):
            positions = cls.positions if positions is None else positions
            hints = cls.hints
            cls = type(cls)
        obj = cls(it, positions=positions)
        obj.hints = copy.deepcopy(hints)
        return

    @property
    def isglobal(self):
        return len(self) == 0 or np.all(self.pidx == self.pidx[0])


class CylindricalWaveBasis(BasisSet):
    _names = ("pidx", "kz", "m", "pol")

    def __repr__(self):
        positions = ",\n" + " " * (len(self.__class__.__name__) + 1) + "positions=" + ", ".join(str(self.positions).split()) + ")"
        return super().__repr__()[:-1] + positions

    @property
    def isglobal(self):
        return len(self) == 0 or np.all(self.pidx == self.pidx[0])

    def __init__(self, modes, positions=None):
        tmp = []
        if isinstance(modes, np.ndarray):
            modes = modes.tolist()
        for m in modes:
            if m not in tmp:
                tmp.append(m)
        modes = tmp
        if len(modes) == 0:
            pidx = []
            kz = []
            m = []
            pol = []
        elif len(modes[0]) == 3:
            kz, m, pol = (*zip(*modes),)
            pidx = np.zeros_like(m)
        elif len(modes[0]) == 4:
            pidx, kz, m, pol = (*zip(*modes),)
        else:
            raise ValueError("invalid shape of modes")

        if positions is None:
            positions = np.zeros((1, 3))
        if positions.ndim == 1:
            positions = positions[None, :]
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"invalid shape of positions {positions.shape}")

        pidx, kz, m, pol = [np.array(i) for i in (pidx, kz, m, pol)]
        for i in (pidx, kz, m, pol):
            i.flags.writeable = False
            if i.ndim > 1:
                raise ValueError("invalid shape of parameters")
        self.pidx, self.m, self.pol = [np.array(i, int) for i in (pidx, m, pol)]
        self.kz = np.array(kz, float)
        for i, j in ((self.pidx, pidx), (self.m, m), (self.pol, pol)):
            if np.any(i != j):
                raise ValueError("parameters must be integer")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")
        if np.any(self.pidx >= len(positions)):
            raise ValueError("undefined position is indexed")

        self._positions = positions
        self._positions.flags.writeable = False
        self.hints = {}

    @property
    def positions(self):
        return self._positions


    def __getitem__(self, idx):
        res = self.pidx[idx], self.kz[idx], self.m[idx], self.pol[idx]
        if isinstance(idx, int):
            return res
        return type(self)(*res, self.positions)

    def __eq__(self, other):
        return (
            other is not None
            and np.array_equal(self.pidx, other.pidx)
            and np.array_equal(self.kz, other.kz)
            and np.array_equal(self.m, other.m)
            and np.array_equal(self.pol, other.pol)
            and np.array_equal(self.positions, other.positions)
        )
    def __call__(self, key=None):
        key = "pkzmp" if key is None else key.lower().strip()
        if key == "pkzmp":
            return self.pidx, self.kz, self.m, self.pol
        elif key == "kzmp":
            return self.kz, self.m, self.pol
        elif key == "kzm":
            return self.kz, self.m
        raise ValueError(f"unrecognized key '{key}'")

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

    @classmethod
    def default(cls, kzs, mmax, nmax=1, **kwargs):
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
        kzs = np.atleast_1d(kzs)
        if kzs.ndim > 1:
            raise ValueError(f"kzs has dimension larger than one: '{kzs.ndim}'")
        modes = [
            [n, kz, m, p]
            for n in range(nmax)
            for kz in kzs
            for m in range(-mmax, mmax + 1)
            for p in range(1, -1, -1)
        ]
        return cls(modes, **kwargs)

    def _from_iterable(cls, it, positions=None):
        hints = {}
        if isinstance(cls, CylindricalWaveBasis):
            positions = cls.positions if positions is None else positions
            hints = cls.hints
            cls = type(cls)
        obj = cls(it, positions=positions)
        obj.hints = copy.deepcopy(hints)
        return obj


class PlaneWaveBasis(BasisSet):
    _names = ("kx", "ky", "kz", "pol")

    def __init__(self, modes):
        if isinstance(modes, np.ndarray):
            modes = modes.tolist()
        tmp = []
        for m in modes:
            if m not in tmp:
                tmp.append(m)
        modes = tmp
        if len(modes) == 0:
            pidx = []
            kx = []
            ky = []
            kz = []
        elif len(modes[0]) == 4:
            kx, ky, kz, pol = (*zip(*modes),)
        else:
            raise ValueError("invalid shape of modes")

        kx, ky, kz, pol = [np.array(i) for i in (kx, ky, kz, pol)]
        for i in (kx, ky, kz, pol):
            i.flags.writeable = False
            if i.ndim > 1:
                raise ValueError("invalid shape of parameters")
        self.kx, self.ky, self.kz = kx, ky, kz
        self.pol = np.array(pol.real, int)
        if np.any(self.pol != pol):
            raise ValueError("polarizations must be integer")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")

        self.hints = {}

    def __getitem__(self, idx):
        res = tuple((i[idx] for i in (self.kx, self.ky, self.kz, self.pol)))
        if isinstance(idx, int):
            return res
        return type(self)(*res)

    def permute(self, n=1):
        if n != int(n):
            raise ValueError("'n' must be integer")
        n = n % 3
        kx, ky, kz = self.kx, self.ky, self.kz
        kpar = self.hints.get("kpar")
        while n > 0:
            kx, ky, kz = kz, kx, ky
            kpar = kpar[[2, 0, 1]]
            n -= 1
        obj = type(self)(kx, ky, kz, self.pol)
        if "lattice" in self.hints:
            obj.hints["lattice"] = self.hints["lattice"].permute(n)
        if kpar is not None:
            obj.hints["kpar"] = kpar
        return obj

    def __call__(self, key=None):
        key = "kxyzp" if key is None else key.lower().strip()
        if key == "kxyzp":
            return self.kx, self.ky, self.kz, self.pol
        elif key == "kxyp":
            return self.kx, self.ky, self.pol
        elif key == "kzp":
            return self.kz, self.pol
        return ValueError(f"unrecognized key '{key}'")

    @classmethod
    def default(cls, kvecs):
        kvecs = np.atleast_2d(kvecs)
        modes = np.empty((2 * kvecs.shape[0], 4), kvecs.dtype)
        modes[::2, :3] = kvecs
        modes[1::2, :3] = kvecs
        modes[::2, 3] = 1
        modes[1::2, 3] = 0
        return cls(modes)

    def _from_iterable(cls, it):
        hints = {}
        if isinstance(cls, PlaneWaveBasis):
            hints = cls.hints
            cls = type(cls)
        obj = cls(it)
        obj.hints = hints
        return obj

    def __eq__(self, other):
        return (
            other is not None
            and np.array_equal(self.kx, other.ky)
            and np.array_equal(self.ky, other.ky)
            and np.array_equal(self.kz, other.kz)
            and np.array_equal(self.pol, other.pol)
        )

    def partial(self, alignment=None):
        alignment = "xy" if alignment is None else alignment
        if alignment in ("xy", "yz", "zx"):
            kpar = [getattr(self, "k" + s) for s in alignment]
        else:
            raise ValueError(f"invalid alignment '{alignment}'")
        obj = PlaneWaveBasisPartial(zip(*kpar, self.pol), alignment)
        obj.hints = copy.deepcopy(self.hints)
        return obj


class PlaneWaveBasisPartial(PlaneWaveBasis):
    def __init__(self, modes, alignment=None):
        alignment = "xy" if alignment is None else alignment
        if isinstance(modes, np.ndarray):
            modes = modes.tolist()
        tmp = []
        for m in modes:
            if m not in tmp:
                tmp.append(m)
        modes = tmp
        if len(modes) == 0:
            kx = []
            ky = []
            pol = []
        elif len(modes[0]) == 3:
            kx, ky, pol = (*zip(*modes),)
        else:
            raise ValueError("invalid shape of modes")

        self._kx, self._ky = [np.real(i) for i in (kx, ky)]
        self.pol = np.array(np.real(pol), int)
        for i, j in [(self._kx, kx), (self._ky, ky), (self.pol, pol)]:
            i.flags.writeable = False
            if i.ndim > 1:
                raise ValueError("invalid shape of parameters")
            if np.any(i != j):
                raise ValueError("ivalid value for parameter, must be float or integer")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")

        if alignment in ("xy", "yz", "zx"):
            self._names = (*("k" + i for i in alignment),) + ("pol",)
        else:
            raise ValueError(f"invalid alignment '{alignment}'")

        self.alignment = alignment
        self.hints = {}

    def _from_iterable(cls, it, alignment=None):
        hints = {}
        if isinstance(cls, PlaneWaveBasis):
            hints = cls.hints
            alignment = cls.alignment if alignment is None else alignment
        obj = super()._from_iterable(it)
        obj.alignment = obj.alignment if alignment is None else alignment
        obj.hints = hints
        return obj

    def __getitem__(self, idx):
        res = tuple((i[idx] for i in (self._kx, self._ky, self.pol)))
        if isinstance(idx, int):
            return res
        return type(self)(*res, self.alignment)

    @property
    def kx(self):
        if self.alignment == "xy":
            return self._kx
        elif self.alignment == "zx":
            return self._ky
        return None

    @property
    def ky(self):
        if self.alignment == "xy":
            return self._ky
        elif self.alignment == "yz":
            return self._kx
        return None

    @property
    def kz(self):
        if self.alignment == "yz":
            return self._ky
        elif self.alignment == "zx":
            return self._kx
        return None

    def complete(self, ks, direction=None):
        direction = "up" if direction is None else direction
        direction = direction.lower()
        if direction not in ("up", "down"):
            raise ValueError("direction not recognized")
        try:
            ks = np.array(ks[self.pol], complex)
        except TypeError:
            pass
        kx = self._kx
        ky = self._ky
        kz = misc.wave_vec_z(kx, ky, ks) * (2 * (direction == "up") - 1)
        if self.alignment == "yz":
            kx, ky, kz = kz, kx, ky
        elif self.alignment == "zy":
            kx, ky, kz = ky, kz, kx
        obj = PlaneWaveBasis(zip(kx, ky, kz, self.pol))
        obj.hints = copy.deepcopy(self.hints)
        return obj

    def __call__(self):
        return self._kx, self._ky, self.pol

    def __repr__(self):
        whitespace = "\n" + " " * (len(self.__class__.__name__) + 1)
        string = str(self).replace("\n", whitespace)
        return f"{self.__class__.__name__}({string},{whitespace}alignment='{self.alignment}')"

    def permute(self, n=1):
        if n != int(n):
            raise ValueError("'n' must be integer")
        n = n % 3
        alignment = self.alignment
        dct = {"xy": "yz", "yz": "zx", "zx": "xy"}
        kpar = self.hints.get("kpar", [0, 0, 0])
        while n > 0:
            alignment = dct[alignment]
            kpar = kpar[[2, 0, 1]]
            n -= 1
        obj = type(self)(kx, ky, kz, self.pol)
        if "lattice" in self.hints:
            obj.hints["lattice"] = self.hints["lattice"].permute(n)
        if "kpar" in self.hints:
            obj.hints["kpar"] = kpar
        return obj

    def __eq__(self, other):
        return (
            other is not None
            and np.array_equal(self.kx, other.kx)
            and np.array_equal(self.ky, other.ky)
            and np.array_equal(self.kz, other.kz)
            and np.array_equal(self.pol, other.pol)
        )

    @classmethod
    def default(cls, kpars, *args, **kwargs):
        kpars = np.atleast_2d(kpars)
        modes = np.empty((2 * kpars.shape[0], 3), float)
        modes[::2, :2] = kpars
        modes[1::2, :2] = kpars
        modes[::2, 2] = 1
        modes[1::2, 2] = 0
        return cls(modes, *args, **kwargs)

    @classmethod
    def diffr_orders(kpar, lattice, bmax):
        raise NotImplementedError
