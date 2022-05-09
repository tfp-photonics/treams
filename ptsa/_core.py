import abc, collections, warnings

import ptsa.special as sc
from ptsa._material import Material
from ptsa._basisset import BasisSet, SphericalWaveBasis
from ptsa.numpy import AnnotatedArray, AnnotatedArrayWarning

import numpy as np

class PhysicsArrayWarning(AnnotatedArrayWarning): pass

class PhysicsArray(AnnotatedArray):
    def __new__(cls, arr, k0=None, basis=None, polarization=None, material=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PhysicsArrayWarning)
            obj = super().__new__(cls, arr)
        polarization = polarization if polarization is None else polarization.strip().lower()
        obj.k0 = k0
        if not isinstance(basis, tuple):
            basis = (basis,) * obj.ndim
        if not isinstance(polarization, tuple):
            polarization = (polarization,) * obj.ndim
        if not isinstance(material, tuple):
            material = (material,) * obj.ndim
        for a, b, p, m in zip(obj.annotations, basis, polarization, material):
            if b is None:
                a.setdefault("basis", None)
            else:
                a["basis"] = b
            if p is None:
                a.setdefault("polarization", "helicity")
            else:
                a["polarization"] = p
            if m is None:
                a.setdefault("material", Material())
            else:
                a["material"] = m
        obj._check()
        return obj

    def _check(self):
        if self.ndim > 2:
            warnings.warn(f"invalid number of dimension '{self.ndim}' > 2", PhysicsArrayWarning)
        self.k0
        self.basis
        self.polarization
        self.material


    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)
        self._check()

    @property
    def k0(self):
        res = self.annotations["all"].get("k0")
        if res is None:
            warnings.warn("missing 'k0' for PhysicsArray", PhysicsArrayWarning)
        return res

    @k0.setter
    def k0(self, val):
        if val is None:
            self.annotations["all"].setdefault("k0", float("nan"))
        else:
            self.annotations["all", "k0"] = val

    @property
    def basis(self):
        if self.ndim == 0:
            warnings.warn("no basis defined for PhysicsArray of dimension 0", PhysicsArrayWarning)
            return
        basis = [a.get("basis") for a in self.annotations]
        if None in basis:
            warnings.warn("missing 'basis' for PhysicsArray", PhysicsArrayWarning)
        elif any([not isinstance(b, BasisSet) for b in basis]):
            warnings.warn("invalid 'basis' for PhysicsArray", PhysicsArrayWarning)
        if np.all([b == basis[0] for b in basis[1:]]):
            return basis[0]
        return tuple(basis)

    @property
    def polarization(self):
        if self.ndim == 0:
            warnings.warn("no polarization defined for PhysicsArray of dimension 0", PhysicsArrayWarning)
            return
        polarizations = np.array([a.get("polarization") for a in self.annotations])
        if None in polarizations:
            warnings.warn("missing polarization for PhysicsArray", PhysicsArrayWarning)
        elif np.any((polarizations != "helicity") & (polarizations != "parity")):
            warnings.warn("invalid polarization")
        if np.all(polarizations == polarizations[0]):
            return polarizations[0]
        return "mixed"

    @property
    def material(self):
        if self.ndim == 0:
            warnings.warn("no material defined for PhysicsArray of dimension 0", PhysicsArrayWarning)
            return
        material = np.array([a.get("material") for a in self.annotations])
        if None in material:
            warnings.warn("missing 'material' for PhysicsArray", PhysicsArrayWarning)
        elif any([not isinstance(m, Material) for m in material]):
            warnings.warn("invalid 'material' for PhysicsArray", PhysicsArrayWarning)
        if np.all(material == material[0]):
            return material[0]
        return tuple(material)

    @property
    def ks(self):
        mat = self.material
        if isinstance(mat, tuple):
            mat = mat[0]
        return mat.nmp * self.k0

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [np.asanyarray(i) for i in inputs]
        if np.any([i.ndim > 2 for i in inputs]):
            inputs = [i.view(AnnotatedArray) for i in inputs]
            return self.view(AnnotatedArray).__array_ufunc__(ufunc, method, *inputs, **kwargs)
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    # def __matmul__(self, other, *args, **kwargs):
    #     print("__matmul__")
    #     other = np.asanyarray(other)
    #     if other.ndim > 2:
    #         return np.matmul(self.view(AnnotatedArray), other, *args, **kwargs)
    #     return super().__matmul__(other, *args, **kwargs)
    #
    # def __rmatmul__(self, other, *args, **kwargs):
    #     print("__rmatmul__")
    #     if isinstance(other, PhysicsArray):
    #         return other.__matmul__(self, *args, **kwargs)
    #     return np.matmul(other, self.view(AnnotatedArray), *args, **kwargs)


class Field(metaclass=abc.ABCMeta):

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.matmul and not isinstance(inputs[0], Field):
            inputs = [i.view(PhysicsArray) for i in inputs]
            return self.view(PhysicsArray).__array_ufunc__(ufunc, method, *inputs, **kwargs)
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)


    def __call__(self, r, **kwargs):
        return self.efield(r, **kwargs)

    @abc.abstractmethod
    def efield(self, r, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def hfield(self, r, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def dfield(self, r, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def bfield(self, r, **kwargs):
        raise NotImplementedError

    def gfield(self, r, sign, **kwargs):
        sign = sign > 0
        return np.sqrt(0.5) * (self.efield(r, **kwargs) + 1j * self.material.z * sign * self.hfield(r, **kwargs))


class SphericalWaveGeneral(PhysicsArray, Field):
    def _check(self):
        if self.ndim == 0:
            warnings.warn("invalid number of dimensions 0 < 1", PhysicsArrayWarning)
            super()._check()
            return
        basis = self.annotations[0].get("basis")
        if basis is None:
            dim = self.shape[0]
            try:
                lmax = SphericalWaveBasis.defaultlmax(dim)
            except ValueError:
                super()._check()
                return
            self.annotations[0, "basis"] = SphericalWaveBasis.default(lmax)
        elif not isinstance(basis, SphericalWaveBasis):
            warnings.warn("spherical wave needs 'SphericalWaveBasis' at dimension '0'", PhysicsArrayWarning)
        super()._check()


class SphericalWave(SphericalWaveGeneral):
    def efield(self, r):
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        if self.annotations[0, "polarization"] == "helicity":
            res = sc.vsw_rA(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
                basis.pol,
            )
        else:
            res = (1 - basis.pol[:, None]) * sc.vsw_rM(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
            ) + basis.pol[:, None] * sc.vsw_rN(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
            )
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        if self.ndim == 2:
            return np.matmul(res, self, axes=[(-1, -2), (-2, -1), (-1, -2)])
        return np.matmul(res, self, axes=[(-1, -2), -1, -1])

    def hfield(self, r):
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        if self.annotations[0, "polarization"] == "helicity":
            res = sc.vsw_rA(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
                basis.pol,
            ) * (2 * basis.pol[:, None] - 1)
        else:
            res = basis.pol[:, None] * sc.vsw_rM(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
            ) + (1 - basis.pol[:, None]) * sc.vsw_rN(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
            )
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        factor = 1j * self.material.z
        if self.ndim == 2:
            return np.matmul(res, self, axes=[(-1, -2), (-2, -1), (-1, -2)]) / factor
        return np.matmul(res, self, axes=[(-1, -2), -1, -1]) / factor

    def dfield(self, r):
        if self.annotations[0, "polarization"] == "parity":
            return self.efield(r) * self.material.epsilon
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        res = sc.vsw_rA(
            *basis("lm"),
            self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
            r_sph[..., basis.pidx, 1],
            r_sph[..., basis.pidx, 2],
            basis.pol,
        ) * self.material.nmp[basis.pol]
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        if self.ndim == 2:
            return np.matmul(res, self, axes=[(-1, -2), (-2, -1), (-1, -2)])
        return np.matmul(res, self, axes=[(-1, -2), -1, -1])

    def bfield(self, r):
        if self.annotations[0, "polarization"] == "parity":
            return self.hfield(r) * self.material.mu
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        res = sc.vsw_rA(
            *basis("lm"),
            self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
            r_sph[..., basis.pidx, 1],
            r_sph[..., basis.pidx, 2],
            basis.pol,
        ) * (2 * basis.pol[:, None] - 1) * self.material.nmp[basis.pol]
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        if self.ndim == 2:
            return np.matmul(res, self, axes=[(-1, -2), (-2, -1), (-1, -2)])
        return np.matmul(res, self, axes=[(-1, -2), -1, -1])

    def gfield(self, r, sign):
        if self.annotations[0, "polarization"] == "parity":
            return super().gfield(r, sign)
        sign = sign > 0
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        basis = basis[basis.pol == sign]
        res = sc.vsw_rA(
            *basis("lm"),
            self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
            r_sph[..., basis.pidx, 1],
            r_sph[..., basis.pidx, 2],
            basis.pol,
        ) * (2 * basis.pol[:, None] - 1) * self.material.nmp[basis.pol]
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        if self.ndim == 2:
            return np.matmul(res, self[choice, :], axes=[(-1, -2), (-2, -1), (-1, -2)])
        return np.matmul(res, self[choice], axes=[(-1, -2), -1, -1])


class SphericalWaveSingular(SphericalWaveGeneral):
    def efield(self, r):
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        if self.annotations[0, "polarization"] == "helicity":
            res = sc.vsw_A(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
                basis.pol,
            )
        else:
            res = (1 - basis.pol[:, None]) * sc.vsw_M(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
            ) + basis.pol[:, None] * sc.vsw_N(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
            )
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        if self.ndim == 2:
            return np.matmul(res, self, axes=[(-1, -2), (-2, -1), (-1, -2)])
        return np.matmul(res, self, axes=[(-1, -2), -1, -1])

    def hfield(self, r):
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        if self.annotations[0, "polarization"] == "helicity":
            res = sc.vsw_A(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
                basis.pol,
            ) * (2 * basis.pol[:, None] - 1)
        else:
            res = basis.pol[:, None] * sc.vsw_M(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
            ) + (1 - basis.pol[:, None]) * sc.vsw_N(
                *basis("lm"),
                self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
                r_sph[..., basis.pidx, 1],
                r_sph[..., basis.pidx, 2],
            )
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        factor = 1j * self.material.z
        if self.ndim == 2:
            return np.matmul(res, self, axes=[(-1, -2), (-2, -1), (-1, -2)]) / factor
        return np.matmul(res, self, axes=[(-1, -2), -1, -1]) / factor

    def dfield(self, r):
        if self.annotations[0, "polarization"] == "parity":
            return self.efield(r) * self.material.epsilon
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        res = sc.vsw_A(
            *basis("lm"),
            self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
            r_sph[..., basis.pidx, 1],
            r_sph[..., basis.pidx, 2],
            basis.pol,
        ) * self.material.nmp[basis.pol]
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        if self.ndim == 2:
            return np.matmul(res, self, axes=[(-1, -2), (-2, -1), (-1, -2)])
        return np.matmul(res, self, axes=[(-1, -2), -1, -1])

    def bfield(self, r):
        if self.annotations[0, "polarization"] == "parity":
            return self.hfield(r) * self.material.mu
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        res = sc.vsw_A(
            *basis("lm"),
            self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
            r_sph[..., basis.pidx, 1],
            r_sph[..., basis.pidx, 2],
            basis.pol,
        ) * (2 * basis.pol[:, None] - 1) * self.material.nmp[basis.pol]
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        if self.ndim == 2:
            return np.matmul(res, self, axes=[(-1, -2), (-2, -1), (-1, -2)])
        return np.matmul(res, self, axes=[(-1, -2), -1, -1])

    def gfield(self, r, sign):
        if self.annotations[0, "polarization"] == "parity":
            return super().gfield(r, sign)
        sign = sign > 0
        r = np.asarray(r)
        basis = self.annotations[0, "basis"]
        if r.ndim == 1:
            r = np.reshape(r, (1, 3))
        r_sph = sc.car2sph(r[..., None, :] - basis.positions)
        basis = basis[basis.pol == sign]
        res = sc.vsw_A(
            *basis("lm"),
            self.ks[basis.pol] * r_sph[..., basis.pidx, 0],
            r_sph[..., basis.pidx, 1],
            r_sph[..., basis.pidx, 2],
            basis.pol,
        ) * (2 * basis.pol[:, None] - 1) * self.material.nmp[basis.pol]
        res = sc.vsph2car(res, r_sph[..., basis.pidx, :])
        if self.ndim == 2:
            return np.matmul(res, self[choice, :], axes=[(-1, -2), (-2, -1), (-1, -2)])
        return np.matmul(res, self[choice], axes=[(-1, -2), -1, -1])


class SphericalWaveLattice(SphericalWaveSingular):
    def _check(self):
        super()._check()
        if self.ndim == 0:
            return
        lattice = self.annotations[0].get("lattice")
        if lattice is None:
            warnings.warn("wave on latttice needs 'Lattice' at dimension '0'", PhysicsArrayWarning)
        elif not isinstance(lattice, Lattice):
            try:
                self.lattice = lattice
            except ValueError:
                warnings.warn("wave on latttice needs 'Lattice' at dimension '0'", PhysicsArrayWarning)

    @property
    def lattice(self):
        return self.annotations[0, "lattice"]

    @lattice.setter
    def lattice(self, lat):
        self.annotations[0, "lattice"] = lattice(Lat)

    def efield(self, r, **kwargs):
        raise NotImplementedError("try re-expanding in a 'SphericalWave'")

    def hfield(self, r, **kwargs):
        raise NotImplementedError("try re-expanding in a 'SphericalWave'")

    def dfield(self, r, **kwargs):
        raise NotImplementedError("try re-expanding in a 'SphericalWave'")

    def bfield(self, r, **kwargs):
        raise NotImplementedError("try re-expanding in a 'SphericalWave'")


class TMatrixGeneral(PhysicsArray):
    def _check(self):
        super()._check()
        polarization = self.polarization
        if polarization is None or polarization == "mixed":
            warnings.warn("invalid polarization", PhysicsArrayWarning)
        if self.ndim != 2:
            warnings.warn(f"invalid number of dimensions {self.ndim} != 2", PhysicsArrayWarning)
            return
        basis = self.annotations[1].get("basis")
        if basis is None:
            self.annotations[1, "basis"] = self.annotations[0].get("basis")
        elif not isinstance(basis, SphericalWaveBasis):
            warnings.warn("T-Matrix needs 'SphericalWaveBasis' at dimension '1'", PhysicsArrayWarning)

    def __new__(cls, arr, k0=None, basis=None, polarization=None, material=None, interacted=None):
        obj = super().__new__(cls, arr, k0, basis, polarization)
        obj.interacted = getattr(arr, "interacted", True) if interacted is None else interacted
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.interacted = getattr(obj, "interacted", True)
        super().__array_finalize__(obj)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [np.asanyarray(i) for i in inputs]
        if np.any([i.ndim != 2 for i in inputs]) or method == "reduce":
            inputs = [i.view(SphericalWaveSingular) for i in inputs]
            return self.view(SphericalWaveSingular).__array_ufunc__(ufunc, method, *inputs, **kwargs)
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    @property
    def interacted(self):
        return self._interacted or self.basis.isglobal

    @interacted.setter
    def interacted(self, val):
        self._interacted = bool(val)

    @classmethod
    def cluster(cls, tmats, positions):
        positions = np.asanyarray(positions)
        if len(tmats) < positions.shape[0]:
            warnings.warn("specified more positions than T-matrices")
        elif len(tmats) > positions.shape[0]:
            raise ValueError(
                f"got '{len(tmats)}' T-matrices and only '{positions.shape[0]}' positions"
            )
        material = tmats[0].material
        k0 = tmats[0].k0
        polarization = tmats[0].polarization
        dim = sum([tmat.shape[0] for tmat in tmats])
        modes = np.zeros((dim, 4), int)
        tlocal = np.zeros((dim, dim), complex)

        if polarization not in ("helicity", "parity"):
            raise ValueError(f"invalid polarization '{polarization}'")

        i = 0
        for j, tmat in enumerate(tmats):
            if tmat.material != material:
                warnings.warn(f"materials with values '{material}' and '{tmat.material}' do not match")
            if tmat.k0 != k0:
                warnings.warn(f"vacuum wave numbers '{k0}' and '{tmat.k0}' do not match")
            if tmat.polarization != polarization:
                warnings.warn(f"polarizations '{polarization}' and '{tmat.polarization}' do not match")
            if not tmat.basis.isglobal:
                raise ValueError("non-global T-matrix given")
            dim = tmat.shape[0]
            for k in range(1, 4):
                modes[i : i + dim, k] = tmat.basis()[k]
            modes[i : i + dim, 0] = j
            tlocal[i : i + dim, i : i + dim] = tmat
            i += dim
        return cls(tlocal, k0, SphericalWaveBasis(modes, positions), polarization, material, False)

    @classmethod
    def sphere(cls, lmax, k0, radii, epsilon, mu=None, kappa=None):
        pass

    @classmethod
    def ebcm(cls, lmax, k0, r, dr, epsilon, mu=1, kappa=0, lint=None):
        pass

    @property
    def xs_ext_avg(self):
        return NotImplemented

    @property
    def xs_sca_avg(self):
        return NotImplemented

    @property
    def cd(self):
        return NotImplemented

    @property
    def chi(self):
        return NotImplemented

    @property
    def db(self):
        return NotImplemented

    def xs(self):
        return NotImplemented

    def coupling(self):
        pass

    def interact(self, out=None):
        pass

    def basischange(self, out=None):
        pass

    def helicitybasis(self, out=None):
        pass

    def paritybasis(self, out=None):
        pass

    def __getitem__(self, key):
        return super().__getitem__(self, key)

    def rotate(self, phi, theta, phi):
        pass

    def translate(self, rvec):
        pass

    def globalmat(self):
        pass



class TMatrix(SphericalWaveSingular, TMatrixGeneral): pass


class TMatrixLattice(SphericalWaveLattice, TMatrix):
    @property
    def interacted(self):
        # a global basis doesn't mean the matrix is interacted as for the 'normal' case
        return self._interacted

    def coupling(self):
        pass


class TMatrixCLattice(TMatrixGeneral): pass

class CylindricalWaveGeneral(Field): pass
class CylindricalWave(CylindricalWaveGeneral): pass
class CylindricalWaveSingular(CylindricalWaveGeneral): pass
class CylindricalWaveLattice(CylindricalWaveSingular): pass

class QMatrix(PhysicsArray): pass

class PlaneWave(Field): pass


def expand(dst, src=None, ks=float("nan"), pol=None, lattice=None, kpar=None):
    """
    The source basis is expanded into a different basis, assuming the given polarization.
    If the lattice is defined and the source is spherical or cylindrical, this basis
    is assumed to be periodic.
    """
    pol = "helicity" if pol is None else pol
    if pol not in ("helicity", "parity"):
        raise ValueError(f"invalid polarization '{pol}'")
    pol = pol == "helicity"
    if lattice is None and kpar is not None:
        raise ValueError(f"'kpar' given but 'lattice' is undefined")
    src = dst if src is None else src
    ks = np.atleast_1d(ks)
    if ks.ndim != 1 or ks.size > 2:
        raise ValueError("invalid size of 'ks'")
    if ks.size = 1:
        ks = np.array([ks[0], ks[0]])
    if (not isinstance(src, BasisSet) and not isinstance(dst, BasisSet)) or isinstance(src, PlaneWaveBasisPartial) or isinstance(src, PlaneWaveBasisPartial):
        raise ValueError("unrecognized BasisSet in 'src' or 'dst'")
    if lattice is None:
        if isinstance(src, SphericalWaveBasis):
            if isinstance(dst, SphericalWaveBasis):
                rvec = sc.car2sph(dst.positions[:, None, :] - src.positions)
                return sw.translate(*(m[:, None] for m in dst("lmp")), src("lmp"), ks[src.pol] * rvec[:, :, 0], rvec[:, :, 1], rvec[:, :, 2], helicity=pol)
        elif isinstance(src, CylindricalWaveBasis):
            if isinstance(dst, SphericalWaveBasis):
                mask = dst.pidx[:, None] == src.pidx
                res = np.zeros_like(mask, complex)
                cw.to_sw(*(m[:, None] for m in dst("lmp")), src("kzmp"), where=mask, out=res)
                return res
            elif isinstance(dst CylindricalWaveBasis):
                rvec = sc.car2cyl(dst.positions[:, None, :] - src.positions)
                krho = src.krho(ks)
                return cw.translate(*(m[:, None] for m in dst("kzmp")), src("kzmp"), krho * rvec[:, :, 0], rvec[:, 1], rvec[:, 2])
        elif isinstance(src, PlaneWaveBasis):
            if isinstance(src, PlaneWaveBasisPartial):
                src = src.complete(ks)
            if isinstance(dst, SphericalWaveBasis):
                mask = dst.pidx[:, None] == src.pidx
                res = np.zeros_like(mask, complex)
                return pw.to_sw(*(m[:, None] for m in dst("lmp")), src(), ks[src.pol] * rvec[:, :, 0], rvec[:, :, 1], rvec[:, :, 2], helicity=pol)
            elif isinstance(src, CylindricalWaveBasis):
                return pw.to_cw(*(m[:, None] for m in dst("lmp")), src("lmp"), ks[src.pol] * rvec[:, :, 0], rvec[:, :, 1], rvec[:, :, 2], helicity=pol)
    elif isinstance(src, SphericalWaveBasis):
    elif isinstance(src, CylindricalWaveBasis):
    elif isinstance(src, PlaneWaveBasis):
    raise ValueError("invalid expansion")

def expand_global(dst, src=None): pass
def translate(r, dst, src=None, pol=None): pass
def rotate(angles, dst, src=None): pass
def permute(dst, src=None, inverse=True): pass
