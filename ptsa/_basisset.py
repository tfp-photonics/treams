import abc
import copy
import warnings
from collections.abc import Sequence, Set

import numpy as np

import ptsa.special as sc
from ptsa import config, cw, misc, pw, sw
from ptsa._lattice import Lattice
from ptsa._material import Material
from ptsa.numpy import AnnotatedArray, AnnotatedArrayWarning


class OrderedSet(Sequence, Set):
    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError


class BasisSet(OrderedSet, metaclass=abc.ABCMeta):
    _names = ()

    def __str__(self):
        return ",\n".join(f"{name}={i}" for name, i in zip(self._names, self[()]))

    def __repr__(self):
        whitespace = "\n" + " " * (len(self.__class__.__name__) + 1)
        string = str(self).replace("\n", whitespace)
        return f"{self.__class__.__name__}({string})"

    def __len__(self):
        return len(self.pol)

    @abc.abstractmethod
    def __call__(self, r, k0, poltype=None, material=None, modetype=None):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def default(cls):
        raise NotImplementedError

    @abc.abstractmethod
    def rotate(self, phi, theta=0, psi=0, basis=None, *, where=True):
        raise NotImplementedError

    @abc.abstractmethod
    def translate(self, k0, r, basis=None, poltype=None, material=None, *, where=True):
        raise NotImplementedError

    @abc.abstractmethod
    def expandin(
        self, k0, basis=None, material=None, *, where=True
    ):  # poltype, lattice, kpar, mode
        raise NotImplementedError

    @abc.abstractmethod
    def basischange(self, basis=None, poltype=None, *, where=True):
        raise NotImplementedError


class SphericalWaveBasis(BasisSet):
    _names = ("pidx", "l", "m", "pol")

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

    @property
    def positions(self):
        return self._positions

    def __repr__(self):
        whitespace = ",\n" + " " * (len(self.__class__.__name__) + 1)
        positions = whitespace + "positions=" + ", ".join(str(self.positions).split())
        return super().__repr__()[:-1] + positions + ")"

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = idx.lower().strip()
            if idx == "plmp":
                return self.pidx, self.l, self.m, self.pol
            elif idx == "lmp":
                return self.l, self.m, self.pol
            elif idx == "lm":
                return self.l, self.m
            raise ValueError(f"unrecognized key '{idx}'")
        res = self.pidx[idx], self.l[idx], self.m[idx], self.pol[idx]
        if isinstance(idx, int) or idx == ():
            return res
        return type(self)(zip(*res), self.positions)

    def __eq__(self, other):
        return self is other or (
            other is not None
            and np.array_equal(self.pidx, other.pidx)
            and np.array_equal(self.l, other.l)
            and np.array_equal(self.m, other.m)
            and np.array_equal(self.pol, other.pol)
            and np.array_equal(self.positions, other.positions)
        )

    def __call__(self, r, k0, poltype=None, material=None, modetype="regular"):
        material = Material() if material is None else Material(material)
        ks = k0 * material.nmp
        poltype = config.POLTYPE if poltype is None else poltype
        r = np.asanyarray(r)
        r = r[..., None, :]
        rsph = sc.car2sph(r - self.positions)
        res = None
        if poltype == "helicity":
            if modetype == "regular":
                res = sc.vsw_rA(
                    self.l,
                    self.m,
                    ks[self.pol] * rsph[..., self.pidx, 0],
                    rsph[..., self.pidx, 1],
                    rsph[..., self.pidx, 2],
                    self.pol,
                )
            elif modetype == "singular":
                res = sc.vsw_A(
                    self.l,
                    self.m,
                    ks[self.pol] * rsph[..., self.pidx, 0],
                    rsph[..., self.pidx, 1],
                    rsph[..., self.pidx, 2],
                    self.pol,
                )
        elif poltype == "parity":
            if modetype == "regular":
                res = (
                    self.pol[:, None]
                    * sc.vsw_rM(
                        self.l,
                        self.m,
                        ks[self.pol] * rsph[..., self.pidx, 0],
                        rsph[..., self.pidx, 1],
                        rsph[..., self.pidx, 2],
                    )
                    * (1 - self.pol[:, None])
                    * sc.vsw_rN(
                        self.l,
                        self.m,
                        ks[self.pol] * rsph[..., self.pidx, 0],
                        rsph[..., self.pidx, 1],
                        rsph[..., self.pidx, 2],
                    )
                )
            elif modetype == "singular":
                res = (
                    self.pol[:, None]
                    * sc.vsw_M(
                        self.l,
                        self.m,
                        ks[self.pol] * rsph[..., self.pidx, 0],
                        rsph[..., self.pidx, 1],
                        rsph[..., self.pidx, 2],
                    )
                    * (1 - self.pol[:, None])
                    * sc.vsw_N(
                        self.l,
                        self.m,
                        ks[self.pol] * rsph[..., self.pidx, 0],
                        rsph[..., self.pidx, 1],
                        rsph[..., self.pidx, 2],
                    )
                )
        if res is None:
            raise ValueError("invalid parameters")
        res = AnnotatedArray(sc.vsph2car(res, rsph[..., self.pidx, :]))
        res.annotations[-2, "basis"] = self
        res.annotations[-2, "material"] = material
        res.annotations[-2, "poltype"] = poltype
        res.annotations[-2, "modetype"] = modetype
        return res

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
        modes = [
            [n, l, m, p]
            for n in range(0, nmax)
            for l in range(1, lmax + 1)
            for m in range(-l, l + 1)
            for p in range(1, -1, -1)
        ]
        return cls(modes, *args, **kwargs)

    @classmethod
    def ebcm(cls, lmax, nmax=1, mmax=-1, *args, **kwargs):
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
        return cls(modes, *args, **kwargs)

    @staticmethod
    def defaultlmax(dim, nmax=1):
        """
        Default maximal degree

        Given the dimension of the T-matrix return the estimated maximal value of `l`.
        This is the inverse of defaultdim. A value of zero is allowed for empty
        T-matrices.

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

        Given the maximal value of `l` return the size of the corresponding T-matrix.
        This is the inverse of defaultlmax. A value of zero is allowed.

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
        if isinstance(cls, SphericalWaveBasis):
            positions = cls.positions if positions is None else positions
            cls = type(cls)
        obj = cls(it, positions=positions)
        return obj

    @property
    def isglobal(self):
        return len(self) == 0 or np.all(self.pidx == self.pidx[0])

    def rotate(self, phi, theta=0, psi=0, basis=None, *, where=True):
        basis = self if basis is None else basis
        if not isinstance(basis, SphericalWaveBasis):
            raise ValueError("can only rotate within SphericalWaveBasis")
        where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
        res = sw.rotate(
            *(m[:, None] for m in basis["lmp"]),
            *self["lmp"],
            phi,
            theta,
            psi,
            where=where,
        )
        res[np.logical_not(where)] = 0
        return PhysicsArray(res, basis=(basis, self))

    def translate(self, k0, r, basis=None, poltype=None, material=None, *, where=True):
        basis = self if basis is None else basis
        poltype = config.POLTYPE if poltype is None else poltype
        material = Material() if material is None else Material(material)
        if not isinstance(basis, SphericalWaveBasis):
            raise ValueError("can only translate within SphericalWaveBasis")
        where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
        r = np.asanyarray(r).squeeze()
        if r.shape != (3,):
            raise ValueError(f"invalid 'r'")
        ks = k0 * material.nmp
        r = sc.car2sph(r)
        res = sw.translate(
            *(m[:, None] for m in basis["lmp"]),
            *self["lmp"],
            ks[self.pol] * r[0],
            r[1],
            r[2],
            poltype=poltype,
            where=where,
        )
        res[np.logical_not(where)] = 0
        return PhysicsArray(
            res, k0=k0, basis=(basis, self), poltype=poltype, material=material
        )

    def expandin(
        self,
        k0,
        basis=None,
        poltype=None,
        material=None,
        lattice=None,
        kpar=None,
        modetype=None,
        *,
        where=True,
    ):
        basis = self if basis is None else basis
        poltype = config.POLTYPE if poltype is None else poltype
        material = Material() if material is None else Material(material)
        if lattice is not None:
            lattice = Lattice(lattice)
            kpar = np.zeros(lattice.dim) if kpar is None else kpar
            if len(kpar) != lattice.dim:
                raise ValueError("incompatible dimensions of 'lattice' and 'kpar'")
        elif kpar is not None:
            raise ValueError("missing definition for 'lattice'")
        if modetype not in (None, "same", "regular", "singular"):
            raise ValueError(f"invalid mode '{mode}'")

        ks = k0 * material.nmp
        if isinstance(basis, SphericalWaveBasis):
            rs = sc.car2sph(basis.positions[:, None, :] - self.positions)
            if lattice is None:
                res = sw.translate(
                    *[m[:, None] for m in basis["lmp"]],
                    *self["lmp"],
                    ks[self.pol] * rs[basis.pidx[:, None], self.pidx, 0],
                    rs[basis.pidx[:, None], self.pidx, 1],
                    rs[basis.pidx[:, None], self.pidx, 2],
                    poltype=poltype,
                    singular=modetype == "singular",
                    where=where,
                )
                res[np.logical_not(where)] = 0
                res = PhysicsArray(
                    res, k0=k0, basis=(basis, self), poltype=poltype, material=material
                )
                if modetype == "singular":
                    res.modetype = ("regular", "singular")
                return res
            kpar = np.asarray(kpar).squeeze()
            if kpar.size == 1:
                x = kpar
                kpar = np.zeros(3)
                kpar[2] = x
            elif kpar.size == 2:
                x = kpar
                kpar = np.zero(3)
                kpar[:2] = x
            elif kpar.size == 3:
                kpar = np.array(kpar, float)
            res = sw.translate_periodic(
                ks,
                kpar,
                lattice[...],
                basis.positions,
                basis["lmp"],
                self["lmp"],
                self.positions,
                poltype=poltype,
            )
            res[np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                lattice=lattice,
                modetype=("regular", "singular"),
                kpar=kpar,
            )

        elif isinstance(basis, CylindricalWaveBasis) and lattice is not None:
            where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
            res = sw.periodic_to_cw(
                *[m[:, None] for m in basis["kzmp"]],
                *self["lmp"],
                ks[self.pol],
                lattice.volume,
                poltype=None,
                where=where,
            )
            res[np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                lattice=lattice[...],
                kpar=kpar,
            )
        elif isinstance(basis, PlaneWaveBasis) and lattice is not None:
            if isinstance(basis, PlaneWaveBasisPartial):
                basis = basis.complete(k0)
            basis.check_lattice(lattice, kpar)
            res = sw.periodic_to_sw(
                *[m[:, None] for m in basis],
                *self,
                lattice.volume,
                poltype=poltype,
                where=where,
            )
            res[np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                lattice=lattice,
                kpar=kpar,
            )
        raise ValueError("invalid basis or lattice definitions")

    def basischange(self, basis=None, poltype=None, *, where=True):
        basis = self if basis is None else basis
        poltype = config.POLTYPE if poltype is None else poltype
        if poltye == "helicity":
            poltype = ("helicity", "parity")
        elif poltype == "parity":
            poltype = ("parity", "helicity")
        if poltype != ("helicity", "parity") and poltype != ("parity", "helicity"):
            raise ValueError("todo")
        res = np.logical_and(
            np.logical_and(basis.l[:, None] == self.l, basis.m[:, None] == self.m),
            basis.pidx[:, None] == self.pidx,
        )
        res[np.logical_and(res, basis.pol[:, None] == self.pol)] = -1
        return PhysicsArray(res, basis=(basis, self), poltype=poltype)


class CylindricalWaveBasis(BasisSet):
    _names = ("pidx", "kz", "m", "pol")

    def __repr__(self):
        whitespace = ",\n" + " " * (len(self.__class__.__name__) + 1)
        positions = whitespace + "positions=" + ", ".join(str(self.positions).split())
        return super().__repr__()[:-1] + positions + ")"

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
        positions = np.array(positions)
        if positions.ndim == 1:
            positions = positions[None, :]
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"invalid shape of positions {positions.shape}")

        self.pidx, self.m, self.pol = [np.array(i, int) for i in (pidx, m, pol)]
        self.kz = np.array(kz, float)
        self.kz.writeable = False
        for i, j in ((self.pidx, pidx), (self.m, m), (self.pol, pol)):
            i.flags.writeable = False
            if np.any(i != j):
                raise ValueError("parameters must be integer")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")
        if np.any(self.pidx >= len(positions)):
            raise ValueError("undefined position is indexed")

        self._positions = positions
        self._positions.flags.writeable = False

    @property
    def positions(self):
        return self._positions

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = idx.lower().strip()
            if idx == "pkzmp":
                return self.pidx, self.kz, self.m, self.pol
            elif idx == "kzmp":
                return self.kz, self.m, self.pol
            elif idx == "kzm":
                return self.kz, self.m
            raise ValueError(f"unrecognized key '{idx}'")
        res = self.pidx[idx], self.kz[idx], self.m[idx], self.pol[idx]
        if isinstance(idx, int) or idx == ():
            return res
        return type(self)(zip(*res), self.positions)

    def __eq__(self, other):
        return self is other or (
            other is not None
            and np.array_equal(self.pidx, other.pidx)
            and np.array_equal(self.kz, other.kz)
            and np.array_equal(self.m, other.m)
            and np.array_equal(self.pol, other.pol)
            and np.array_equal(self.positions, other.positions)
        )

    def __call__(self, r, k0, poltype=None, material=None, modetype=None):
        material = Material() if material is None else Material(material)
        ks = (k0 * material.nmp)[self.pol]
        krhos = np.sqrt(ks * ks - self.kz * self.kz)
        krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
        poltype = config.POLTYPE if poltype is None else poltype
        if r.ndim == 1:
            r = r[None, :]
        rcyl = sc.car2cyl(r - self.positions)
        res = None
        if poltype == "helicity":
            if modetype == "regular":
                res = sc.vsw_rA(
                    self.kz,
                    self.m,
                    krhos * rcyl[..., self.pidx, 0],
                    rcyl[..., self.pidx, 1],
                    rcyl[..., self.pidx, 2],
                    ks,
                    self.pol,
                )
            elif modetype == "singular":
                res = sc.vsw_A(
                    self.kz,
                    self.m,
                    krhos * rcyl[..., self.pidx, 0],
                    rcyl[..., self.pidx, 1],
                    rcyl[..., self.pidx, 2],
                    ks,
                    self.pol,
                )
        elif poltype == "parity":
            if modetype == "regular":
                res = (
                    self.pol[:, None]
                    * sc.vsw_rM(
                        self.kz,
                        self.m,
                        krhos * rcyl[..., self.pidx, 0],
                        rcyl[..., self.pidx, 1],
                        rcyl[..., self.pidx, 2],
                    )
                    * (1 - self.pol[:, None])
                    * sc.vsw_rN(
                        self.kz,
                        self.m,
                        krhos * rcyl[..., self.pidx, 0],
                        rcyl[..., self.pidx, 1],
                        rcyl[..., self.pidx, 2],
                        ks,
                    )
                )
            elif modetype == "singular":
                res = (
                    self.pol[:, None]
                    * sc.vsw_M(
                        self.kz,
                        self.m,
                        krhos * rcyl[..., self.pidx, 0],
                        rcyl[..., self.pidx, 1],
                        rcyl[..., self.pidx, 2],
                    )
                    * (1 - self.pol[:, None])
                    * sc.vsw_N(
                        self.kz,
                        self.m,
                        krhos * rcyl[..., self.pidx, 0],
                        rcyl[..., self.pidx, 1],
                        rcyl[..., self.pidx, 2],
                        ks,
                    )
                )
        if res is None:
            raise ValueError("invalid parameters")
        res = AnnotatedArray(sc.vcyl2car(res, rsph[..., self.pidx, :]))
        res.annotations[-1, "basis"] = self
        res.annotations[-1, "material"] = material
        res.annotations[-1, "poltype"] = poltype
        res.annotations[-1, "modetype"] = modetype
        return res

    @staticmethod
    def defaultdim(nkz, mmax, nmax=1):
        """
        Default dimension

        Given the maximal value of `l` return the size of the corresponding T-matrix.
        This is the inverse of defaultmmax. A value of zero is allowed.

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

        Given the dimension of the T-matrix return the estimated maximal value of `m`.
        This is the inverse of defaultdim. A value of zero is allowed for empty
        T-matrices.

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
        if isinstance(cls, CylindricalWaveBasis):
            positions = cls.positions if positions is None else positions
            cls = type(cls)
        obj = cls(it, positions=positions)
        return obj


    def rotate(self, phi, basis=None, *, where=True):
        basis = self if basis is None else basis
        if not isinstance(basis, CylindricalWaveBasis):
            raise ValueError("can only rotate within CylindricalWaveBasis")
        where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
        res = cw.rotate(
            *(m[:, None] for m in basis["kzmp"]),
            *self["kzmp"],
            phi,
            where=where,
        )
        res[np.logical_not(where)] = 0
        return PhysicsArray(res, basis=(basis, self))

    def translate(self, k0, r, basis=None, poltype=None, material=None, *, where=True):
        basis = self if basis is None else basis
        material = Material() if material is None else Material(material)
        if not isinstance(basis, CylindricalWaveBasis):
            raise ValueError("can only translate within CylindricalWaveBasis")
        where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
        r = np.asanyarray(r)
        if r.shape[-1] != 3:
            raise ValueError(f"invalid shape at dimension {r.ndim - 1}: {r.shape[-1]}")
        ks = (k0 * material.nmp)[self.pol]
        krhos = np.sqrt(ks * ks - self.kz * self.kz)
        krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
        r = sc.car2cyl(r)
        res = cw.translate(
            *(m[:, None] for m in basis["kzmp"]),
            *self["kzmp"],
            krhos * r[..., 0],
            r[..., 1],
            r[..., 2],
            where=where,
        )
        res[np.logical_not(where)] = 0
        return PhysicsArray(
            res, k0=k0, basis=(basis, self), poltype=poltype, material=material
        )

    def expandin(
        self,
        k0,
        basis=None,
        poltype=None,
        material=None,
        lattice=None,
        kpar=None,
        modetype=None,
        *,
        where=True,
    ):
        basis = self if basis is None else basis
        material = Material() if material is None else Material(material)
        if lattice is not None:
            lattice = Lattice(lattice)
            kpar = np.zeros(lattice.dim) if kpar is None else kpar
            if len(kpar) != lattice.dim:
                raise ValueError("incompatible dimensions of 'lattice' and 'kpar'")
        elif kpar is not None:
            raise ValueError("missing definition for 'lattice'")
        if modetype not in (None, "same", "regular", "singular"):
            raise ValueError(f"invalid mode '{mode}'")


        ks = (k0 * material.nmp)[self.pol]
        krhos = np.sqrt(ks * ks - self.kz * self.kz)
        krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
        if isinstance(basis, CylindricalWaveBasis):
            rs = sc.car2cyl(basis.positions[:, None, :] - self.positions)
            if lattice is None:
                res = cw.translate(
                    *(m[:, None] for m in basis["kzmp"]),
                    *self["kzmp"],
                    krhos * r[..., 0],
                    r[..., 1],
                    r[..., 2],
                    poltype=poltype,
                    singular=modetype == "singular",
                    where=where,
                )
                res[np.logical_not(where)] = 0
                res = PhysicsArray(
                    res, k0=k0, basis=(basis, self), poltype=poltype, material=material
                )
                if modetype == "singular":
                    res.modetype = ("regular", "singular")
                return res
            res = cw.translate_periodic(
                ks,
                kpar,
                lattice,
                basis.positions,
                basis["lmp"],
                self["lmp"],
                self.positions,
                poltype=poltype,
                where=where,
            )
            res[np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                lattice=lattice,
                modetype=("regular", "singular"),
                kpar=kpar,
            )

        elif isinstance(basis, SphericalWaveBasis) and lattice is None:
            where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
            res = cw.to_sw(
                *[m[:, None] for m in basis["lmp"]],
                *self["kzmp"],
                ks,
                poltype=None,
                where=where,
            )
            res[np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
            )
        elif isinstance(basis, PlaneWaveBasis) and lattice is not None:
            if isinstance(basis, PlaneWaveBasisPartial):
                basis = basis.complete(k0)
            basis.check_lattice(lattice, kpar)
            res = sw.periodic_to_sw(
                *[m[:, None] for m in basis],
                *self["kzmp"],
                lattice.volume,
                poltype=poltype,
                where=where,
            )
            res[np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                lattice=lattice,
                kpar=kpar,
            )
        raise ValueError("invalid basis or lattice definitions")

    def basischange(self, basis=None, poltype=None, *, where=True):
        basis = self if basis is None else basis
        poltype = config.POLTYPE if poltype is None else poltype
        if poltye == "helicity":
            poltype = ("helicity", "parity")
        elif poltype == "parity":
            poltype = ("parity", "helicity")
        if poltype != ("helicity", "parity") and poltype != ("parity", "helicity"):
            raise ValueError("todo")
        res = np.logical_and(
            np.logical_and(basis.l[:, None] == self.l, basis.m[:, None] == self.m),
            basis.pidx[:, None] == self.pidx,
        )
        res[np.logical_and(res, basis.pol[:, None] == self.pol)] = -1
        return PhysicsArray(res, basis=(basis, self), poltype=poltype)


class PlaneWaveBasis(BasisSet):
    _names = ("kx", "ky", "kz", "pol")

    is_global = True

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


class PhysicsArrayWarning(AnnotatedArrayWarning):
    pass


class PhysicsArray(AnnotatedArray):
    _scales = {"basis"}

    def __new__(
        cls,
        arr,
        k0=None,
        basis=None,
        poltype=None,
        material=None,
        modetype=None,
        lattice=None,
        kpar=None,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PhysicsArrayWarning)
            obj = super().__new__(cls, arr)
        obj.k0 = k0
        obj.basis = basis
        obj.poltype = poltype
        obj.material = material
        obj.modetype = modetype
        obj.lattice = lattice
        obj.kpar = kpar
        obj._get_hidden_values()
        obj._check()
        return obj

    def _get_hidden_values(self):
        for a in self.annotations[::-1]:
            if isinstance(a.get("basis"), PlaneWaveBasis):
                lat = getattr(a.get("basis"), "hints", {}).get("lattice", None)
                hidden_lat = a.get("hidden_lattice", None)
                if lat is not None and hidden_lat is not None:
                    a["hidden_lattice"] = hidden_lat & lat

    def index(self, item):
        if len(item) == 3:
            item = (0,) + item
        return super().index(item)

    def indices(self, basis):
        return [self.index(i) for i in basis]

    def _check(self):
        self.k0
        self.basis
        self.poltype
        self.material
        self.modetype
        self.lattice
        self.kpar
        self.hidden_lattice
        self.hidden_kpar
        for a in self.annotations[::-1]:
            lat = a.get("lattice")
            hidden_lat = a.get("hidden_lattice")
            if lat is not None and hidden_lat is not None and not (lat <= hidden_lat):
                raise ValueError("incompatible lattices")
            elif hidden_lat is None and isinstance(lat, Lattice):
                a["hidden_lattice"] = lat
            kpar = a.get("kpar")
            hidden_kpar = a.get("hidden_kpar")
            if None not in (kpar, hidden_kpar) and np.any(np.logical_or(kpar != hidden_kpar, kpar == 0)):
                raise ValueError("incompatible kpar")
            elif hidden_kpar is None and kpar is not None:
                a["hidden_kpar"] = kpar

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)
        self._get_basis_hints()
        self._check()

    k0 = AnnotatedArray.register_property("k0", (int, float, np.floating, np.integer))
    basis = AnnotatedArray.register_property("basis", BasisSet)
    poltype = AnnotatedArray.register_property("poltype", str)
    modetype = AnnotatedArray.register_property("modetype", str)
    material = AnnotatedArray.register_property("material", Material)
    lattice = AnnotatedArray.register_property("lattice", Lattice)
    kpar = AnnotatedArray.register_property(
        "kpar", (int, float, list, np.floating, np.integer, np.ndarray)
    )
    hidden_lattice = AnnotatedArray.register_property("hidden_lattice", Lattice)
    hidden_kpar = AnnotatedArray.register_property(
        "hidden_kpar", (int, float, list, np.floating, np.integer, np.ndarray)
    )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [np.asanyarray(i) for i in inputs]
        if np.any([i.ndim > 2 for i in inputs]):
            inputs = [i.view(AnnotatedArray) for i in inputs]
            return self.view(AnnotatedArray).__array_ufunc__(
                ufunc, method, *inputs, **kwargs
            )
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def __matmul__(self, other, *args, **kwargs):
        other = np.asanyarray(other)
        # if other.ndim > 2:
        #     return np.matmul(self.view(AnnotatedArray), other, *args, **kwargs)
        res = super().__matmul__(other, *args, **kwargs)
        other_ann = getattr(other, "annotations", [{}])
        other_ndim = np.ndim(other)
        for name in (
            "k0",
            "poltype",
            "modetype",
            "material",
            "lattice",
            "kpar",
            "hidden_lattice",
            "hidden_kpar",
        ):
            if self.ndim > 1 and self.annotations[-1].get(name) is None:
                if other_ndim == 1:
                    res.annotations[-1] = other_ann[-1].get(name)
                else:
                    res.annotations[-2] = other_ann[-2].get(name)
            if other_ndim > 1 and other_ann[-2].get(name) is None:
                res.annotations[-1] = self.annotations[-1].get(name)
        return res

    def __rmatmul__(self, other, *args, **kwargs):
        other = np.asanyarray(other)
        # if other.ndim > 2:
        #     return np.matmul(other, self.view(AnnotatedArray), *args, **kwargs)
        res = super().__rmatmul__(other, *args, **kwargs)
        other_ann = getattr(other, "annotations", [{}])
        other_ndim = np.ndim(other)
        for name in (
            "k0",
            "poltype",
            "modetype",
            "material",
            "lattice",
            "kpar",
            "hidden_lattice",
            "hidden_kpar",
        ):
            if other_ndim > 1 and other_ann[-1].get(name) is None:
                if self.ndim == 1:
                    res.annotations[-1] = self.annotations[-1].get(name)
                else:
                    res.annotations[-2] = self.annotations[-2].get(name)
            if self.ndim > 1 and self.annotations[-2].get(name) is None:
                res.annotations[-1] = other_ann[-1].get(name)
        return res
