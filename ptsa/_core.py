"""Basis sets and core array functionalities"""

import abc
import copy
import warnings
from collections.abc import Sequence, Set
from functools import partial

import numpy as np

import ptsa.lattice as la
import ptsa.special as sc
from ptsa import config, cw, misc, pw, sw
from ptsa._lattice import Lattice
from ptsa._material import Material, vacuum
from ptsa.numpy import AnnotatedArray, AnnotatedArrayWarning


class OrderedSet(Sequence, Set):
    """
    Ordered set

    A abstract base class that combines a sequence and set. In contrast to regular sets
    it is expected that the equality comparison only returns `True` if all entries are
    in the same order.
    """

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError


class BasisSet(OrderedSet, metaclass=abc.ABCMeta):
    """
    BasisSet

    BasisSet is the base class for all basis sets used. They are expected to be an
    ordered sequence of the modes, that are included in a expansion. Some methods
    are required to rotate or translate the basis set, to change the polarization types,
    and to expand into a different basis, when possible.
    """

    _names = ()

    def __repr__(self):
        string = ",\n    ".join(f"{name}={i}" for name, i in zip(self._names, self[()]))
        return f"{self.__class__.__name__}(\n    {string},\n)"

    def __len__(self):
        return len(self.pol)

    @abc.abstractmethod
    def __call__(self, r, k0, poltype=None, material=None, modetype=None):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def default(cls, *args, **kwargs):
        """
        default(cls, *args, **kwargs)

        Construct a basis set in a default order by giving few parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rotate(self, phi, theta=0, psi=0, basis=None, *, where=True):
        """
        rotate(self, phi, theta=0, psi=0, basis=None, *, where=True)

        Rotation coefficients for the specified Euler angles.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def translate(self, k0, r, basis=None, poltype=None, material=None, *, where=True):
        """
        translate(self, k0, r, basis=None, poltype=None, material=None, *, where=True)

        Translation coefficients for for angular wave number `k0`, along the
        Cartesian vector `r`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def expand(
        self, k0, basis=None, material=None, *, where=True
    ):  # poltype, lattice, kpar, mode
        """
        expand(self, k0, basis=None, material=None, *, where=True)

        Expansion coefficients for the transformation to a different basis.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def basischange(self, basis=None, poltype=None, *, where=True):
        """
        basischange(self, basis=None, poltype=None, *, where=True)

        Transformation coefficents for changing the polarization type.
        """
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
            l = []  # noqa: E741
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
        positions = "positions=" + str(self.positions).replace("\n", ",")
        return f"{super().__repr__()[:-1]}    {positions},\n)"

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = idx.lower().strip()
            if idx == "plmp":
                return self.pidx, self.l, self.m, self.pol
            elif idx == "lmp":
                return self.l, self.m, self.pol
            elif idx == "lm":
                return self.l, self.m
            raise IndexError(f"unrecognized key '{idx}'")
        res = self.pidx[idx], self.l[idx], self.m[idx], self.pol[idx]
        if isinstance(idx, int) or idx == ():
            return res
        return type(self)(zip(*res), self.positions)

    def __eq__(self, other):
        try:
            return self is other or (
                np.array_equal(self.pidx, other.pidx)
                and np.array_equal(self.l, other.l)
                and np.array_equal(self.m, other.m)
                and np.array_equal(self.pol, other.pol)
                and np.array_equal(self.positions, other.positions)
            )
        except AttributeError:
            return False

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
                res = (1 - self.pol[:, None]) * sc.vsw_rM(
                    self.l,
                    self.m,
                    ks[self.pol] * rsph[..., self.pidx, 0],
                    rsph[..., self.pidx, 1],
                    rsph[..., self.pidx, 2],
                ) + self.pol[:, None] * sc.vsw_rN(
                    self.l,
                    self.m,
                    ks[self.pol] * rsph[..., self.pidx, 0],
                    rsph[..., self.pidx, 1],
                    rsph[..., self.pidx, 2],
                )
            elif modetype == "singular":
                res = (1 - self.pol[:, None]) * sc.vsw_M(
                    self.l,
                    self.m,
                    ks[self.pol] * rsph[..., self.pidx, 0],
                    rsph[..., self.pidx, 1],
                    rsph[..., self.pidx, 2],
                ) + self.pol[:, None] * sc.vsw_N(
                    self.l,
                    self.m,
                    ks[self.pol] * rsph[..., self.pidx, 0],
                    rsph[..., self.pidx, 1],
                    rsph[..., self.pidx, 2],
                )
        if res is None:
            raise ValueError("invalid parameters")
        res = AnnotatedArray(sc.vsph2car(res, rsph[..., self.pidx, :]))
        res.ann[-2, "basis"] = self
        res.ann[-2, "k0"] = k0
        res.ann[-2, "material"] = material
        res.ann[-2, "poltype"] = poltype
        res.ann[-2, "modetype"] = modetype
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
            for l in range(1, lmax + 1)  # noqa: E741
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
            for l in range(max(abs(m), 1), lmax + 1)  # noqa: E741
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
        res[..., np.logical_not(where)] = 0
        return PhysicsArray(res, basis=(basis, self))

    def translate(self, k0, r, basis=None, poltype=None, material=None, *, where=True):
        basis = self if basis is None else basis
        poltype = config.POLTYPE if poltype is None else poltype
        material = Material() if material is None else Material(material)
        if not isinstance(basis, SphericalWaveBasis):
            raise ValueError("can only translate within SphericalWaveBasis")
        where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
        r = np.asanyarray(r)
        if r.shape[-1] != 3:
            raise ValueError("invalid 'r'")
        ks = k0 * material.nmp
        r = sc.car2sph(r)
        res = sw.translate(
            *(m[:, None] for m in basis["lmp"]),
            *self["lmp"],
            ks[self.pol] * r[..., None, None, 0],
            r[..., None, None, 1],
            r[..., None, None, 2],
            singular=False,
            poltype=poltype,
            where=where,
        )
        res[..., np.logical_not(where)] = 0
        prepend = (r.ndim - 1) * (None,)
        return PhysicsArray(
            res,
            k0=prepend + (k0, k0),
            basis=prepend + (basis, self),
            poltype=prepend + 2 * (poltype,),
            material=prepend + 2 * (material,),
        )

    def expand(
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
        if modetype not in (None, "same", "regular", "singular"):
            raise ValueError(f"invalid mode '{modetype}'")

        ks = k0 * material.nmp
        if isinstance(basis, SphericalWaveBasis):
            rs = sc.car2sph(basis.positions[:, None, :] - self.positions)
            if lattice is None:
                res = sw.translate(
                    *(m[:, None] for m in basis["lmp"]),
                    *self["lmp"],
                    ks[self.pol] * rs[basis.pidx[:, None], self.pidx, 0],
                    rs[basis.pidx[:, None], self.pidx, 1],
                    rs[basis.pidx[:, None], self.pidx, 2],
                    poltype=poltype,
                    singular=modetype == "singular",
                    where=where,
                )
                res[..., np.logical_not(where)] = 0
                res = PhysicsArray(
                    res, k0=k0, basis=(basis, self), poltype=poltype, material=material
                )
                if modetype == "singular":
                    res.modetype = ("regular", "singular")
                return res
            lattice = Lattice(lattice)
            try:
                kpar = list(kpar)
            except TypeError:
                kpar = [0] * lattice.dim if kpar is None else [kpar]
            if len(kpar) != lattice.dim:
                raise ValueError("incompatible dimensions of 'lattice' and 'kpar'")
            if len(kpar) == 1:
                x = kpar[0]
                kpar = [np.nan, np.nan, x]
            elif len(kpar) == 2:
                x = kpar
                kpar = kpar + [np.nan]
            elif len(kpar) == 3:
                x = kpar
            res = sw.translate_periodic(
                ks,
                x,
                lattice[...],
                basis.positions,
                basis[()],
                self[()],
                self.positions,
                poltype=poltype,
            )
            res[..., np.logical_not(where)] = 0
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

        if isinstance(basis, CylindricalWaveBasis):
            if lattice is None:
                lattice = basis.hints["lattice"]
            lattice = Lattice(lattice)
            if kpar is None:
                kpar = basis.hints["kpar"]
            if len(kpar) == 1:
                kpar = [np.nan, np.nan, kpar]
            if 3 - np.isnan(kpar).sum() != lattice.dim:
                raise ValueError("incompatible dimensions of 'lattice' and 'kpar'")
            where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
            res = sw.periodic_to_cw(
                *(m[:, None] for m in basis["kzmp"]),
                *self["lmp"],
                ks[self.pol],
                lattice.volume,
                poltype=poltype,
                where=where,
            )
            res[..., np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                modetype="singular",
                lattice=lattice,
                kpar=kpar,
            )
        elif isinstance(basis, PlaneWaveBasis):
            if lattice is None:
                lattice = basis.hints["lattice"]
            if kpar is None:
                kpar = basis.hints["kpar"]
            if len(kpar) == 2:
                kpar = [kpar[0], kpar[1], np.nan]
            if isinstance(basis, PlaneWaveBasisPartial):
                basis = basis.complete(k0)
            res = sw.periodic_to_pw(
                *(m[:, None] for m in basis["xyzp"]),
                *self["lmp"],
                lattice.volume,
                poltype=poltype,
                where=where,
            )
            res[..., np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                modetype=(None, "singular"),
                lattice=lattice,
                kpar=kpar,
            )
        raise ValueError("invalid basis or lattice definitions")

    def basischange(self, basis=None, poltype=None, *, where=True):
        basis = self if basis is None else basis
        poltype = config.POLTYPE if poltype is None else poltype
        if poltype == "helicity":
            poltype = ("helicity", "parity")
        elif poltype == "parity":
            poltype = ("parity", "helicity")
        if poltype != ("helicity", "parity") and poltype != ("parity", "helicity"):
            raise ValueError("todo")
        where = (
            (basis.l[:, None] == self.l)
            & (basis.m[:, None] == self.m)
            & (basis.pidx[:, None] == self.pidx)
            & where
        )
        res = np.zeros_like(where, float)
        res[where] = np.sqrt(0.5)
        res[where & (basis.pol[:, None] == self.pol) & (self.pol == 0)] = -np.sqrt(0.5)
        return PhysicsArray(res, basis=(basis, self), poltype=poltype)


class CylindricalWaveBasis(BasisSet):
    _names = ("pidx", "kz", "m", "pol")

    def __repr__(self):
        positions = "positions=" + str(self.positions).replace("\n", ",")
        return f"{super().__repr__()[:-1]}    {positions},\n)"

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
        self.kz.flags.writeable = False
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

        self.hints = {}
        if len(self.kz) > 0 and np.all(self.kz == self.kz[0]):
            self.hints["kpar"] = [np.nan, np.nan, self.kz[0]]

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
            raise IndexError(f"unrecognized key '{idx}'")
        res = self.pidx[idx], self.kz[idx], self.m[idx], self.pol[idx]
        if isinstance(idx, int) or idx == ():
            return res
        return type(self)(zip(*res), self.positions)

    def __eq__(self, other):
        try:
            return self is other or (
                np.array_equal(self.pidx, other.pidx)
                and np.array_equal(self.kz, other.kz)
                and np.array_equal(self.m, other.m)
                and np.array_equal(self.pol, other.pol)
                and np.array_equal(self.positions, other.positions)
            )
        except AttributeError:
            return False

    def __call__(self, r, k0, poltype=None, material=None, modetype="regular"):
        material = Material() if material is None else Material(material)
        ks = (k0 * material.nmp)[self.pol]
        krhos = np.sqrt(ks * ks - self.kz * self.kz)
        krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
        poltype = config.POLTYPE if poltype is None else poltype
        r = np.asanyarray(r)
        r = r[..., None, :]
        rcyl = sc.car2cyl(r - self.positions)
        res = None
        if poltype == "helicity":
            if modetype == "regular":
                res = sc.vcw_rA(
                    self.kz,
                    self.m,
                    krhos * rcyl[..., self.pidx, 0],
                    rcyl[..., self.pidx, 1],
                    rcyl[..., self.pidx, 2],
                    ks,
                    self.pol,
                )
            elif modetype == "singular":
                res = sc.vcw_A(
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
                res = (1 - self.pol[:, None]) * sc.vcw_rM(
                    self.kz,
                    self.m,
                    krhos * rcyl[..., self.pidx, 0],
                    rcyl[..., self.pidx, 1],
                    rcyl[..., self.pidx, 2],
                ) + self.pol[:, None] * sc.vcw_rN(
                    self.kz,
                    self.m,
                    krhos * rcyl[..., self.pidx, 0],
                    rcyl[..., self.pidx, 1],
                    rcyl[..., self.pidx, 2],
                    ks,
                )
            elif modetype == "singular":
                res = (1 - self.pol[:, None]) * sc.vcw_M(
                    self.kz,
                    self.m,
                    krhos * rcyl[..., self.pidx, 0],
                    rcyl[..., self.pidx, 1],
                    rcyl[..., self.pidx, 2],
                ) + self.pol[:, None] * sc.vcw_N(
                    self.kz,
                    self.m,
                    krhos * rcyl[..., self.pidx, 0],
                    rcyl[..., self.pidx, 1],
                    rcyl[..., self.pidx, 2],
                    ks,
                )
        if res is None:
            raise ValueError("invalid parameters")
        res = AnnotatedArray(sc.vcyl2car(res, rcyl[..., self.pidx, :]))
        res.ann[-2, "basis"] = self
        res.ann[-2, "k0"] = k0
        res.ann[-2, "material"] = material
        res.ann[-2, "poltype"] = poltype
        res.ann[-2, "modetype"] = modetype
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
    def default(cls, kzs, mmax, nmax=1, *args, **kwargs):
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
        return cls(modes, *args, **kwargs)

    @classmethod
    def with_periodicity(cls, kz, mmax, lattice, rmax, nmax=1, **kwargs):
        lattice = Lattice(lattice)
        lattice_z = Lattice(lattice, "z")
        nkz = np.floor(np.abs(rmax / lattice_z.reciprocal))
        kzs = kz + np.arange(-nkz, nkz + 1) * lattice_z.reciprocal
        res = cls.default(kzs, mmax, nmax, **kwargs)
        res.hints["lattice"] = lattice
        res.hints["kpar"] = [np.nan, np.nan, kz]
        return res

    def _from_iterable(cls, it, positions=None):
        hints = {}
        if isinstance(cls, CylindricalWaveBasis):
            positions = cls.positions if positions is None else positions
            hints = copy.deepcopy(cls.hints)
            cls = type(cls)
        obj = cls(it, positions=positions)
        obj.hints = hints
        return obj

    def rotate(self, phi, basis=None, *, where=True):
        basis = self if basis is None else basis
        if not isinstance(basis, CylindricalWaveBasis):
            raise ValueError("can only rotate within CylindricalWaveBasis")
        where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
        res = cw.rotate(
            *(m[:, None] for m in basis["kzmp"]), *self["kzmp"], phi, where=where,
        )
        res[..., np.logical_not(where)] = 0
        return PhysicsArray(res, basis=(basis, self))

    def translate(self, k0, r, basis=None, material=None, *, where=True):
        basis = self if basis is None else basis
        material = Material() if material is None else Material(material)
        if not isinstance(basis, CylindricalWaveBasis):
            raise ValueError("can only translate within CylindricalWaveBasis")
        where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
        r = np.asanyarray(r)
        if r.shape[-1] != 3:
            raise ValueError("invalid 'r'")
        ks = (k0 * material.nmp)[self.pol]
        krhos = np.sqrt(ks * ks - self.kz * self.kz + 0j)
        krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
        r = sc.car2cyl(r)
        res = cw.translate(
            *(m[:, None] for m in basis["kzmp"]),
            *self["kzmp"],
            krhos * r[..., None, None, 0],
            r[..., None, None, 1],
            r[..., None, None, 2],
            singular=False,
            where=where,
        )
        res[..., np.logical_not(where)] = 0
        prepend = (r.ndim - 1) * (None,)
        return PhysicsArray(
            res,
            k0=prepend + (k0, k0),
            basis=prepend + (basis, self),
            material=prepend + 2 * (material,),
        )

    def expand(
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
        if modetype not in (None, "same", "regular", "singular"):
            raise ValueError(f"invalid mode '{modetype}'")

        ks = (k0 * material.nmp)[self.pol]
        krhos = np.sqrt(ks * ks - self.kz * self.kz + 0j)
        krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
        if isinstance(basis, CylindricalWaveBasis):
            rs = sc.car2cyl(basis.positions[:, None, :] - self.positions)
            if lattice is None:
                res = cw.translate(
                    *(m[:, None] for m in basis["kzmp"]),
                    *self["kzmp"],
                    krhos * rs[basis.pidx[:, None], self.pidx, 0],
                    rs[basis.pidx[:, None], self.pidx, 1],
                    rs[basis.pidx[:, None], self.pidx, 2],
                    singular=modetype == "singular",
                    where=where,
                )
                res[..., np.logical_not(where)] = 0
                res = PhysicsArray(res, k0=k0, basis=(basis, self), material=material)
                if modetype == "singular":
                    res.modetype = ("regular", "singular")
                return res
            alignment = (
                "x"
                if not isinstance(lattice, Lattice) and np.size(lattice) == 1
                else None
            )
            lattice = Lattice(lattice, alignment)
            if "z" in lattice.alignment:
                raise ValueError("invalid lattice")
            try:
                kpar = list(kpar)
            except TypeError:
                kpar = [0] * lattice.dim if kpar is None else [kpar]
            if len(kpar) != lattice.dim:
                raise ValueError("incompatible dimensions of 'lattice' and 'kpar'")
            if len(kpar) == 1:
                x = kpar[0]
                kpar = [x, np.nan, basis.hints["kpar"][2]]
            elif len(kpar) == 2:
                x = kpar
                kpar = kpar + [basis.hints["kpar"][2]]
            res = cw.translate_periodic(
                ks,
                x,
                lattice[...],
                basis.positions,
                basis[()],
                self[()],
                self.positions,
            )
            res[..., np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                material=material,
                lattice=lattice,
                modetype=("regular", "singular"),
                kpar=kpar,
            )

        elif isinstance(basis, SphericalWaveBasis) and lattice is None:
            where = np.logical_and(where, basis.pidx[:, None] == self.pidx)
            res = cw.to_sw(
                *(m[:, None] for m in basis["lmp"]),
                *self["kzmp"],
                ks,
                poltype=poltype,
                where=where,
            )
            res[..., np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                modetype="regular",
            )
        elif isinstance(basis, PlaneWaveBasis):
            if lattice is None:
                lattice = basis.hints["lattice"]
            if kpar is None:
                kpar = basis.hints["kpar"]
            if len(kpar) == 2:
                kpar = [kpar[0], kpar[1], np.nan]
            if isinstance(basis, PlaneWaveBasisPartial):
                basis = basis.complete(k0)
            res = cw.periodic_to_pw(
                *(m[:, None] for m in basis["xyzp"]),
                *self["kzmp"],
                lattice.volume,
                where=where,
            )
            res[..., np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                material=material,
                modetype=(None, "singular"),
                lattice=lattice,
                kpar=kpar,
            )
        raise ValueError("invalid basis or lattice definitions")

    def basischange(self, basis=None, poltype=None, *, where=True):
        basis = self if basis is None else basis
        poltype = config.POLTYPE if poltype is None else poltype
        if poltype == "helicity":
            poltype = ("helicity", "parity")
        elif poltype == "parity":
            poltype = ("parity", "helicity")
        if poltype != ("helicity", "parity") and poltype != ("parity", "helicity"):
            raise ValueError("todo")
        where = (
            (basis.kz[:, None] == self.kz)
            & (basis.m[:, None] == self.m)
            & (basis.pidx[:, None] == self.pidx)
            & where
        )
        res = np.zeros_like(where, float)
        res[where] = np.sqrt(0.5)
        res[where & (basis.pol[:, None] == self.pol) & (self.pol == 0)] = -np.sqrt(0.5)
        return PhysicsArray(res, basis=(basis, self), poltype=poltype)


class PlaneWaveBasis(BasisSet):
    _names = ("kx", "ky", "kz", "pol")
    isglobal = True

    def __init__(self, modes):
        tmp = []
        if isinstance(modes, np.ndarray):
            modes = modes.tolist()
        for m in modes:
            if m not in tmp:
                tmp.append(m)
        modes = tmp
        if len(modes) == 0:
            kx = []
            ky = []
            kz = []
            pol = []
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
        if isinstance(idx, str):
            idx = idx.lower().strip()
            if idx == "xyzp":
                return self.kx, self.ky, self.kz, self.pol
            elif idx == "xyp":
                return self.kx, self.ky, self.pol
            elif idx == "zp":
                return self.kz, self.pol
            raise IndexError(f"unrecognized key '{idx}'")
        res = self.kx[idx], self.ky[idx], self.kz[idx], self.pol[idx]
        if isinstance(idx, int) or idx == ():
            return res
        return type(self)(zip(*res))

    def permute(self, n=1):
        if n != int(n):
            raise ValueError("'n' must be integer")
        n = n % 3
        kx, ky, kz = self.kx, self.ky, self.kz
        kpar = self.hints.get("kpar")
        while n > 0:
            kx, ky, kz = kz, kx, ky
            kpar = kpar[[2, 0, 1]] if kpar is not None else None
            n -= 1
        obj = type(self)(kx, ky, kz, self.pol)
        if "lattice" in self.hints:
            obj.hints["lattice"] = self.hints["lattice"].permute(n)
        if kpar is not None:
            obj.hints["kpar"] = kpar
        return obj

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
            hints = copy.deepcopy(cls.hints)
            cls = type(cls)
        obj = cls(it)
        obj.hints = hints
        return obj

    def __eq__(self, other):
        try:
            return self is other or (
                np.array_equal(self.kx, other.kx)
                and np.array_equal(self.ky, other.ky)
                and np.array_equal(self.kz, other.kz)
                and np.array_equal(self.pol, other.pol)
            )
        except AttributeError:
            return False

    def partial(self, alignment=None, k0=None, material=vacuum):
        if k0 is not None:
            ks = (k0 * material.nmp)[self.pol]
            test = self.kx * self.kx + self.ky * self.ky + self.kz * self.kz - ks * ks
            if np.any(
                np.logical_or(np.abs(test.real) > 1e-10, np.abs(test.imag) > 1e-10)
            ):
                raise ValueError("dispersion relation violated")
        alignment = "xy" if alignment is None else alignment
        if alignment in ("xy", "yz", "zx"):
            kpar = [getattr(self, "k" + s) for s in alignment]
        else:
            raise ValueError(f"invalid alignment '{alignment}'")
        obj = PlaneWaveBasisPartial(zip(*kpar, self.pol), alignment)
        obj.hints = copy.deepcopy(self.hints)
        return obj

    def __call__(self, r, poltype=None):
        poltype = config.POLTYPE if poltype is None else poltype
        r = np.asanyarray(r)
        r = r[..., None, :]
        res = None
        if poltype == "helicity":
            res = sc.vpw_A(
                self.kx, self.ky, self.kz, r[..., 0], r[..., 1], r[..., 2], self.pol,
            )
        elif poltype == "parity":
            res = (1 - self.pol[:, None]) * sc.vpw_M(
                self.kx, self.ky, self.kz, r[..., 0], r[..., 1], r[..., 2],
            ) + self.pol[:, None] * sc.vpw_N(
                self.kx, self.ky, self.kz, r[..., 0], r[..., 1], r[..., 2],
            )
        if res is None:
            raise ValueError("invalid parameters")
        res = AnnotatedArray(res)
        res.ann[-2, "basis"] = self
        res.ann[-2, "poltype"] = poltype
        return res

    def rotate(self, phi, *, where=True):
        c1, s1 = np.cos(phi), np.sin(phi)
        r = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        kvecs = r @ np.array([self.kx, self.ky, self.kz])
        modes = zip(*kvecs, self.pol)
        res = np.eye(len(self))
        res[..., np.logical_not(where)] = 0
        return PhysicsArray(res, basis=(PlaneWaveBasis(modes), self))

    def translate(self, r, basis=None, *, where=True):
        basis = self if basis is None else basis
        if not isinstance(basis, PlaneWaveBasis):
            raise ValueError("can only translate within PlaneWaveBasis")
        elif isinstance(basis, PlaneWaveBasisPartial):
            warnings.warn(
                "casting PlaneWavePartial to PlaneWave can cause floating-point errors"
            )
            basis = basis.complete(self.k0, self.material)  # TODO
        r = np.asanyarray(r)
        if r.shape[-1] != 3:
            raise ValueError("invalid 'r'")
        where = (
            where
            & (basis.kx[:, None] == self.kx)
            & (basis.ky[:, None] == self.ky)
            & (basis.kz[:, None] == self.kz)
            & (basis.pol[:, None] == self.pol)
        )
        res = pw.translate(
            self.kx,
            self.ky,
            self.kz,
            r[..., None, None, 0],
            r[..., None, None, 1],
            r[..., None, None, 2],
            where=where,
        )
        res[..., np.logical_not(where)] = 0
        return PhysicsArray(res, basis=(r.ndim - 1) * (None,) + (basis, self))

    def expand(
        self,
        k0,
        basis=None,
        poltype=None,
        material=vacuum,
        modetype="up",
        *,
        where=True,
    ):
        basis = self if basis is None else basis
        poltype = config.POLTYPE if poltype is None else poltype
        if isinstance(self, PlaneWaveBasisPartial):
            self_c = self.complete(k0, material, modetype)
        else:
            self_c = self
            modetype = None
        if isinstance(basis, PlaneWaveBasis):
            if isinstance(basis, PlaneWaveBasisPartial):
                basis_c = basis.complete(k0, material, modetype)
            else:
                basis_c = basis
                modetype = None
            res = np.array(
                where
                & (basis_c.kx[:, None] == self_c.kx)
                & (basis_c.ky[:, None] == self_c.ky)
                & (basis_c.kz[:, None] == self_c.kz)
                & (basis_c.pol[:, None] == self_c.pol),
                int,
            )
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                material=Material(material),
                modetype=modetype,
            )
        elif isinstance(basis, CylindricalWaveBasis):
            res = pw.to_cw(
                *(m[:, None] for m in basis["kzmp"]), *self[()], where=where,
            ) * pw.translate(
                self.kx, self.ky, self.kz,
                basis.positions[basis.pidx, None, 0],
                basis.positions[basis.pidx, None, 1],
                basis.positions[basis.pidx, None, 2],
            )
            res[..., np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                modetype=("regular", modetype),
            )
        elif isinstance(basis, SphericalWaveBasis):
            res = pw.to_sw(
                *(m[:, None] for m in basis["lmp"]),
                *self[()],
                poltype=poltype,
                where=where,
            ) * pw.translate(
                self.kx, self.ky, self.kz,
                basis.positions[basis.pidx, None, 0],
                basis.positions[basis.pidx, None, 1],
                basis.positions[basis.pidx, None, 2],
            )
            res[..., np.logical_not(where)] = 0
            return PhysicsArray(
                res,
                k0=k0,
                basis=(basis, self),
                poltype=poltype,
                material=material,
                modetype=("regular", modetype),
            )
        raise ValueError("invalid basis definition")

    def basischange(self, basis=None, poltype=None, *, where=True):
        basis = self if basis is None else basis
        poltype = config.POLTYPE if poltype is None else poltype
        if poltype == "helicity":
            poltype = ("helicity", "parity")
        elif poltype == "parity":
            poltype = ("parity", "helicity")
        if poltype != ("helicity", "parity") and poltype != ("parity", "helicity"):
            raise ValueError("todo")
        where = (
            (basis.kx[:, None] == self.kx)
            & (basis.ky[:, None] == self.ky)
            & (basis.kz[:, None] == self.kz)
            & where
        )
        res = np.zeros_like(where, float)
        res[where] = np.sqrt(0.5)
        res[where & (basis.pol[:, None] == self.pol) & (self.pol == 0)] = -np.sqrt(0.5)
        return PhysicsArray(res, basis=(basis, self), poltype=poltype)


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
                raise ValueError("invalid value for parameter, must be real")
        if np.any(self.pol > 1) or np.any(self.pol < 0):
            raise ValueError("polarization must be '0' or '1'")

        if alignment in ("xy", "yz", "zx"):
            self._names = (*("k" + i for i in alignment),) + ("pol",)
        else:
            raise ValueError(f"invalid alignment '{alignment}'")

        self.alignment = alignment
        self.hints = {}

    def _from_iterable(cls, it, alignment=None):
        if isinstance(cls, PlaneWaveBasisPartial):
            alignment = cls.alignment if alignment is None else alignment
        obj = super()._from_iterable(it)
        obj.alignment = obj.alignment if alignment is None else alignment
        return obj

    def __getitem__(self, idx):
        res = tuple((i[idx] for i in (self._kx, self._ky, self.pol)))
        if isinstance(idx, int) or idx == ():
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

    def complete(self, k0, material=vacuum, modetype="up"):
        modetype = modetype.lower()
        if modetype not in ("up", "down"):
            raise ValueError("modetype not recognized")
        try:
            ks = np.array(self.ks[self.pol], complex)
        except TypeError:
            pass
        kx = self._kx
        ky = self._ky
        kz = misc.wave_vec_z(kx, ky, ks) * (2 * (modetype == "up") - 1)
        if self.alignment == "yz":
            kx, ky, kz = kz, kx, ky
        elif self.alignment == "zy":
            kx, ky, kz = ky, kz, kx
        obj = PlaneWaveBasis(zip(kx, ky, kz, self.pol))
        obj.hints = copy.deepcopy(self.hints)
        return obj

    def __call__(self, r, k0, poltype=None, material=vacuum, modetype="up"):
        basis = self.complete(k0, material, modetype)
        res = basis(r, poltype)
        res[-2, "basis"] = self
        res[-2, "k0"] = k0
        res[-2, "material"] = material
        res[-2, "modetype"] = modetype
        return res

    def permute(self, n=1):
        if n != int(n):
            raise ValueError("'n' must be integer")
        n = n % 3
        alignment = self.alignment
        dct = {"xy": "yz", "yz": "zx", "zx": "xy"}
        kpar = self.hints.get("kpar")
        while n > 0:
            alignment = dct[alignment]
            kpar = kpar[[2, 0, 1]] if kpar is not None else kpar
            n -= 1
        obj = type(self)(self.kx, self.ky, self.kz, self.pol)
        if "lattice" in self.hints:
            obj.hints["lattice"] = self.hints["lattice"].permute(n)
        if kpar is not None:
            obj.hints["kpar"] = kpar
        return obj

    def __eq__(self, other):
        try:
            return self is other or (
                np.array_equal(self.kx, other.kx)
                and np.array_equal(self.ky, other.ky)
                and np.array_equal(self.kz, other.kz)
                and np.array_equal(self.pol, other.pol)
            )
        except AttributeError:
            return False

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
    def diffr_orders(cls, kpar, lattice, bmax):
        lattice = Lattice(lattice)
        if lattice.dim != 2:
            raise ValueError("todo")
        latrec = lattice.reciprocal
        kpars = kpar + la.diffr_orders_circle(latrec, bmax) @ latrec
        obj = cls(kpars, alignment=lattice.alignment)
        obj.hints["lattice"] = lattice
        if lattice.alignment == "xy":
            kpar = list(kpar) + [np.nan]
        elif lattice.alignment == "yz":
            kpar = [np.nan] + list(kpar)
        else:
            kpar = [kpar[1], np.nan, kpar[0]]
        obj.hints["kpar"] = kpar
        return obj

    def rotate(self, phi, *, where=True):
        if self.alignment != "xy":
            raise ValueError("todo")
        c1, s1 = np.cos(phi), np.sin(phi)
        r = np.array([[c1, -s1], [s1, c1]])
        kvecs = r @ np.array([self.kx, self.ky])
        modes = zip(*kvecs, self.pol)
        res = np.eye(len(self))
        res[..., np.logical_not(where)] = 0
        return PhysicsArray(res, basis=(PlaneWaveBasisPartial(modes), self))

    def translate(
        self, k0, r, basis=None, poltype=None, material=vacuum, *, where=True
    ):
        self_c = self.complete(k0, material)
        if isinstance(basis, PlaneWaveBasisPartial):
            basis_c = basis.complete(k0, material)
        elif isinstance(basis, PlaneWaveBasis):
            basis_c = basis
        elif basis is None:
            basis_c = basis
        res = self_c.translate(r, basis_c, poltype, where=where)
        res.basis = (basis, self)
        res.material = material
        res.k0 = k0
        return res

    def basischange(self, basis=None, poltype=None, *, where=True):
        basis = self if basis is None else basis
        if not isinstance(basis, PlaneWaveBasisPartial):
            raise ValueError("todo")
        poltype = config.POLTYPE if poltype is None else poltype
        if poltype == "helicity":
            poltype = ("helicity", "parity")
        elif poltype == "parity":
            poltype = ("parity", "helicity")
        if poltype != ("helicity", "parity") and poltype != ("parity", "helicity"):
            raise ValueError("todo")
        bkx, bky, bpol = basis[()]
        res = np.array(
            np.logical_and(
                np.logical_and(bkx[:, None] == self._kx, bky[:, None] == self._ky),
            ),
            int,
        )
        res[np.logical_and(res, bpol[:, None] == self.pol)] = -1
        return PhysicsArray(res, basis=(basis, self), poltype=poltype)


def _pget(name, arr):
    if arr.ndim == 0:
        warnings.warn(
            f"'{name}' undefined for PhysicsArray of dimension 0", PhysicsArrayWarning,
        )
        return None
    val = [a.get(name) for a in arr.ann]
    if all(v == val[0] for v in val[1:]):
        return val[0]
    return tuple(val)


def _pset(name, arr, val, vtype=object):
    if not isinstance(val, tuple):
        val = (val,) * arr.ndim
    if len(val) != arr.ndim:
        warnings.warn("non-matching property size", PhysicsArrayWarning)
    for a, v in zip(arr.ann, val):
        if v is None:
            a.pop(name, None)
        elif isinstance(v, vtype):
            a[name] = v
        else:
            warnings.warn(
                f"invalid type for '{name}': {type(v).__name__}", PhysicsArrayWarning,
            )


def _pdel(name, arr):
    for a in arr.ann:
        a.pop(name, None)


def _physicsarray_property(name, vtype=(int, float, np.floating, np.integer)):
    return property(
        partial(_pget, name), partial(_pset, name, vtype=vtype), partial(_pdel, name)
    )


class PhysicsArrayWarning(AnnotatedArrayWarning):
    pass


class PhysicsArrayError(Exception):
    pass


class PhysicsArray(AnnotatedArray):
    _scales = {"basis"}

    def __init__(
        self,
        arr,
        k0=None,
        basis=None,
        poltype=None,
        material=None,
        modetype=None,
        lattice=None,
        kpar=None,
    ):
        super().__init__(arr)
        self.k0 = k0
        self.basis = basis
        self.poltype = poltype
        self.material = material
        self.modetype = modetype
        self.lattice = lattice
        self.kpar = kpar
        self._check()

    def __repr__(self):
        repr_arr = "    " + repr(self._array)[6:-1].replace("\n  ", "\n")
        repr_kwargs = ",\n    ".join(
            f"{key}={repr(getattr(self, key))}"
            for key in (
                "k0",
                "basis",
                "poltype",
                "material",
                "modetype",
                "lattice",
                "kpar",
            )
            if getattr(self, key) is not None
        )
        if repr_kwargs != "":
            repr_arr += ",\n    " + repr_kwargs
        return f"{self.__class__.__name__}(\n{repr_arr}\n)"

    def index(self, item):
        if len(item) == 3:
            item = (0,) + item
        return super().index(item)

    def indices(self, basis):
        return [self.index(i) for i in basis]

    def _check(self):
        total_lat = None
        total_kpar = [np.nan] * 3
        for a in self.ann[-2:]:
            for lat in (
                a.get("lattice"),
                getattr(a.get("basis"), "hints", {}).get("lattice"),
            ):
                if lat is not None:
                    total_lat = lat | total_lat
            for kpar in (
                a.get("kpar"),
                getattr(a.get("basis"), "hints", {}).get("kpar"),
            ):
                if kpar is not None:
                    for i, (x, y) in enumerate(zip(total_kpar, kpar)):
                        if np.isnan(x):
                            total_kpar[i] = y
                        elif not np.isnan(y) and x != y:
                            raise ValueError("imcompatible kpar")

    k0 = _physicsarray_property("k0", (int, float, np.floating, np.integer))
    basis = _physicsarray_property("basis", BasisSet)
    poltype = _physicsarray_property("poltype", str)
    modetype = _physicsarray_property("modetype", str)
    material = _physicsarray_property("material", Material)
    lattice = _physicsarray_property("lattice", Lattice)
    kpar = _physicsarray_property("kpar", list)

    def __matmul__(self, other, *args, **kwargs):
        res = super().__matmul__(other, *args, **kwargs)
        other_ann = getattr(other, "ann", [{}])
        other_ndim = np.ndim(other)
        for name in (
            "k0",
            "poltype",
            "modetype",
            "material",
            "lattice",
            "kpar",
        ):
            if self.ndim > 1 and self.ann[-1].get(name) is None:
                if other_ndim == 1:
                    res.ann[-1][name] = other_ann[-1].get(name)
                else:
                    res.ann[-2][name] = other_ann[-2].get(name)
            if other_ndim > 1 and other_ann[-2].get(name) is None:
                res.ann[-1][name] = self.ann[-1].get(name)
        return res

    def __rmatmul__(self, other, *args, **kwargs):
        res = super().__rmatmul__(other, *args, **kwargs)
        other_ann = getattr(other, "ann", [{}])
        other_ndim = np.ndim(other)
        for name in (
            "k0",
            "poltype",
            "modetype",
            "material",
            "lattice",
            "kpar",
        ):
            if other_ndim > 1 and other_ann[-1].get(name) is None:
                if self.ndim == 1:
                    res.ann[-1][name] = self.ann[-1].get(name)
                else:
                    res.ann[-2][name] = self.ann[-2].get(name)
            if self.ndim > 1 and self.ann[-2].get(name) is None:
                res.ann[-1][name] = other_ann[-1].get(name)
        return res
