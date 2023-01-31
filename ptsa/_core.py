"""Basis sets and core array functionalities"""

import abc
import copy

import numpy as np

import ptsa._operators as op
import ptsa.lattice as la
from ptsa._lattice import Lattice
from ptsa._material import Material
from ptsa.util import AnnotatedArray, OrderedSet, register_properties


class BasisSet(OrderedSet, metaclass=abc.ABCMeta):
    """
    BasisSet

    BasisSet is the base class for all basis sets used. They are expected to be an
    ordered sequence of the modes, that are included in a expansion.
    """

    _names = ()

    def __repr__(self):
        string = ",\n    ".join(f"{name}={i}" for name, i in zip(self._names, self[()]))
        return f"{self.__class__.__name__}(\n    {string},\n)"

    def __len__(self):
        return len(self.pol)

    @classmethod
    @abc.abstractmethod
    def default(cls, *args, **kwargs):
        """
        default(cls, *args, **kwargs)

        Construct a basis set in a default order by giving few parameters.
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
        self.hints = {}

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
        if isinstance(idx, int) or (isinstance(idx, tuple) and len(idx) == 0):
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

    def partial(self, alignment=None, k0=None, material=Material()):
        if k0 is not None:
            ks = material.ks(k0)[self.pol]
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

    def complete(self, k0, material=Material(), modetype=None):
        # TODO: check kz depending on modetype (alignment?)
        ks = (k0 * Material(material).nmp)[self.pol]
        if (np.abs(self.kx ** 2 + self.ky ** 2 + self.kz ** 2 - ks * ks) > 1e-14).any():
            raise ValueError("incompatible k0 and/or material")
        return self


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
        res = self._kx[idx], self._ky[idx], self.pol[idx]
        if isinstance(idx, int) or idx == ():
            return res
        return type(self)(zip(*res))

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

    def complete(self, k0, material=Material(), modetype="up"):
        if modetype not in ("up", "down"):
            raise ValueError("modetype not recognized")
        material = Material(material)
        kx = self._kx
        ky = self._ky
        kz = material.kzs(k0, kx, ky, self.pol) * (2 * (modetype == "up") - 1)
        if self.alignment == "yz":
            kx, ky, kz = kz, kx, ky
        elif self.alignment == "zy":
            kx, ky, kz = ky, kz, kx
        obj = PlaneWaveBasis(zip(kx, ky, kz, self.pol))
        obj.hints = copy.deepcopy(self.hints)
        return obj

    def __eq__(self, other):
        try:
            skx, sky, skz = self.kx, self.ky, self.kz
            okx, oky, okz = other.kx, other.ky, other.kz
            return self is other or (
                (np.array_equal(skx, okx) or (skx is None and okx is None))
                and (np.array_equal(sky, oky) or (sky is None and oky is None))
                and (np.array_equal(skz, okz) or (skz is None and okz is None))
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
            raise ValueError("invalid lattice dimensions")
        latrec = lattice.reciprocal
        kpars = kpar + la.diffr_orders_circle(latrec, bmax) @ latrec
        obj = cls.default(kpars, alignment=lattice.alignment)
        obj.hints["lattice"] = lattice
        if lattice.alignment == "xy":
            kpar = list(kpar) + [np.nan]
        elif lattice.alignment == "yz":
            kpar = [np.nan] + list(kpar)
        else:
            kpar = [kpar[1], np.nan, kpar[0]]
        obj.hints["kpar"] = kpar
        return obj


CAST = {
    "k0": (lambda x: isinstance(x, float), float),
    "basis": (lambda x: isinstance(x, BasisSet), None),
    "poltype": (lambda x: isinstance(x, str), str),
    "modetype": (lambda x: isinstance(x, str), str),
    "material": (lambda x: isinstance(x, Material), Material),
    "lattice": (lambda x: isinstance(x, Lattice), Lattice),
    "kpar": (lambda x: isinstance(x, list), list),
}


@register_properties
class PhysicsArray(AnnotatedArray):
    _scales = {"basis"}
    _cast = CAST

    changepoltype = op.ChangePoltype()
    efield = op.EField()
    expand = op.Expand()
    expandlattice = op.ExpandLattice()
    permute = op.Permute()
    rotate = op.Rotate()
    translate = op.Translate()

    def __init__(self, arr, ann=(), **kwargs):
        super().__init__(arr, ann)
        for key, val in kwargs.items():
            setattr(self, key, val)
        self._check()

    def __repr__(self):
        repr_arr = "    " + repr(self._array)[6:-1].replace("\n  ", "\n")
        repr_kwargs = ",\n    ".join(
            f"{key}={repr(getattr(self, key))}"
            for key in self._cast
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
            k0 = a.get("k0")
            material = a.get("material")
            modetype = a.get("modetype")
            poltype = a.get("poltype")
            basis = a.get("basis")
            lattice = a.get("lattice")
            for lat in (
                lattice,
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
                            raise ValueError("incompatible kpar")
            if type(basis) == PlaneWaveBasis and None not in (k0, material):
                basis.complete(k0, material, modetype)
            if poltype == "parity" and getattr(material, "ischiral", False):
                raise ValueError("poltype 'parity' not possible for chiral material")

    def __matmul__(self, other, *args, **kwargs):
        res = super().__matmul__(other, *args, **kwargs)
        other_ann = getattr(other, "ann", ({},))
        other_ndim = np.ndim(other)
        for name in self._cast:
            if self.ndim > 1 and self.ann[-1].get(name) is None:
                dim = -1 - (other_ndim != 1)
                val = other_ann[dim].get(name)
                if val is not None:
                    res.ann[dim][name] = val
            val = self.ann[-1].get(name)
            if other_ndim > 1 and other_ann[-2].get(name) is None and val is not None:
                res.ann[-1][name] = val
        return res

    def __rmatmul__(self, other, *args, **kwargs):
        res = super().__rmatmul__(other, *args, **kwargs)
        other_ann = getattr(other, "ann", ({},))
        other_ndim = np.ndim(other)
        for name in self._cast:
            if other_ndim > 1 and other_ann[-1].get(name) is None:
                dim = -1 - (self.ndim != 1)
                val = self.ann[dim].get(name)
                if val is not None:
                    res.ann[dim][name] = val
            val = other_ann[-1].get(name)
            if self.ndim > 1 and self.ann[-2].get(name) is None and val is not None:
                res.ann[-1][name] = val
        return res
