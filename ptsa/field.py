import abc

import numpy as np

from ptsa import special as sc
from ptsa import cw, sw
from ptsa._basisset import SphericalWaveBasis, CylindricalWaveBasis, PlaneWaveBasis, PlaneWaveBasisPartial
from ptsa import _material


class Field(np.ndarray, metaclass=abc.ABCMeta):
    _basistype = SphericalWaveBasis
    def __new__(cls, arr, k0=float("nan"), material=None, basis=None):
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        obj = arr.view(Field)
        obj.k0 = getattr(arr, "k0", k0) if np.isnan(k0) else k0
        obj.material = getattr(arr, "material", ()) if material is None else material
        obj.basis = getattr(arr, "basis", None) if basis is None else basis
        obj.hints = getattr(arr, "hints", {})
        return obj

    def __array_finalize__(self, obj):
        if self.ndim > 2:
            raise ValueError(f"invalid shape: '{self.shape}'")
        if obj is None:
            return
        self.k0 = getattr(obj, "k0", float("nan"))
        self.material = getattr(obj, "material", ())
        self.basis = getattr(obj, "basis", None)
        self.hints = getattr(obj, "hints", {})

    def __getitem__(self, item):
        if isinstance(item, tuple) and (len(item) > 2 or item[0] is None):
            raise KeyError("invalid index")
        res = super().__getitem__(item)
        if isinstance(res, type(self)):
            if isinstance(item, tuple) and len(item) > 1:
                item = item[0]
            res.basis = None if self.basis is None else self.basis[item]
        return res

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, val):
        self._material = None if val is None else _material.Material(*val)

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, val):
        if not val is None and not isinstance(val, self._basistype):
            raise ValueError("unrecognized basis set")
        self._basis = val

    @property
    def k0(self):
        return self._k0

    @k0.setter
    def k0(self, val):
        self._k0 = float(val)

    # @abc.abstractmethod
    # def efield(self, r):
    #     raise NotImplementedError

    def match(self, k0, material, basis):
        return (
            self.k0 == k0
            and self.material == material
            and self.basis.match(basis)
        )

    @property
    def ks(self):
        return self.k0 * self.material.nmp

class FieldArray(Field):
    def __new__(cls, arr, lattice, k0=float("nan"), material=None, basis=None):
        obj = super().__new__(cls, arr, k0, material, basis)
        obj.lattice = lattice
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.lattice = getattr(obj, "lattice")

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, val):
        self._lattice = None if val is None else Lattice(val)


class SphericalWaveGeneral(Field):
    _basistype = SphericalWaveBasis
    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return
        if isinstance(obj, CylindricalWaveGeneral) or isinstance(obj, PlaneWaveGeneral):
            raise ValueError(f"cannot cast '{obj.__class__.__name__}' to '{self.__class__.__name__}'")


class SphericalWave(SphericalWaveGeneral):
    def __new__(cls, arr, k0=float("nan"), material=None, basis=None):
        if isinstance(arr, SphericalWaveGeneral) and basis is None:
            basis = arr.basis
        if isinstance(arr, Field):
            if basis is None:
                raise ValueError("unspecified basis")
            k0 = arr.k0 if np.isnan(k0) else k0
            material = arr.material if material is None else _material.Material(*material)
            if k0 != arr.k0 or material != arr.material:
                raise ValueError("incompatible parameters")
            print(arr.basis)
            print(basis)
            if not (isinstance(arr, SphericalWave) and arr.basis.exact(basis)):
                print("here")
                arr = arr.to_sw(basis)
            else:
                print("there")
        obj = super().__new__(cls, arr, k0, material, basis)
        obj = obj.view(cls)
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return
        if np.isnan(self.k0):
            raise ValueError("'k0' undefined")
        if self.basis is None:
            self.basis = SphericalWaveBasis.default(
                SphericalWaveBasis.defaultlmax(self.shape[0])
            )

    def to_sw(self, basis):
        if self.basis.poltype != basis.poltype or self.basis.hints != basis.hints:
            raise ValueError("non-matching basis")
        rdiff = basis.positions[basis.pidx, None, :] - self.basis.positions[self.basis.pidx, :]
        rdiff = sc.car2sph(rdiff)
        if self.basis.isglobal:
            mat = sw.translate(
                *(m[:, None] for m in basis('lmp')),
                *self.basis('lmp'),
                self.ks[self.basis.pol] * rdiff[:, :, 0], rdiff[:, :, 1], rdiff[:, :, 2],
                helicity=self.basis.poltype == "helicity",
                singular=False
            )
        else:
            raise ValueError("cannot translate non-global field")
        print(mat.shape, self.shape)
        arr = mat @ self
        print(type(arr))
        return SphericalWave(arr, self.k0, self.material, basis)


    # def efield(self, r):
    #     r = np.array(r)
    #     if r.ndim == 1:
    #         r = np.reshape(r, (1, -1))
    #     r_sph = sc.car2sph(r[..., None, :] - self.basis.positions)
    #     res = self.basis.eval(
    #         self.ks[self.basis.pol] * r_sph[..., self.basis.pidx, 0],
    #         r_sph[..., self.basis.pidx, 1],
    #         r_sph[..., self.basis.pidx, 2],
    #     )
    #     res = sc.vsph2car(res, r_sph[..., self.basis.pidx, :])
    #     if self.ndim == 2:
    #         res = res[..., None]
    #     return np.array(np.sum(res * self[:, None, ...], axis=-max(self.ndim, 1)))


SphericalWaveRegular = SphericalWave
class SphericalWaveSingular(SphericalWaveGeneral): pass

class SphericalWaveArray(SphericalWaveSingular, FieldArray): pass
class SphericalWave1D(SphericalWaveArray): pass
class SphericalWave2D(SphericalWaveArray): pass
class SphericalWave3D(SphericalWaveArray): pass

class CylindricalWaveGeneral(Field):
    _basistype = CylindricalWaveBasis
class CylindricalWave(CylindricalWaveGeneral): pass
CylindricalWaveRegular = CylindricalWave
class CylindricalWaveSingular(CylindricalWaveGeneral): pass
class CylindricalWaveArray(CylindricalWaveSingular, FieldArray): pass
class CylindricalWave1D(CylindricalWaveArray): pass
class CylindricalWave2D(CylindricalWaveArray): pass
    # def __new__(cls, array, k0=float("nan"), material=None, basis=None, lattice=None):
    #     obj = super().__new__(cls, array, k0, basis, material)
    #     if isinstance(array, SphericalWave) or isinstance(array, PlaneWave):
    #         if lattice is not None:
    #             raise ValueError("cannot cast to periodic field")
    #         if obj.basis is None:
    #             raise ValueError("unspecified basis")
    #         if not all([x == y for x, y in [(obj.k0, array.k0), (obj.material, array.material)]]):
    #             raise ValueError("incompatible parameters")
    #         array = array.to_sw(basis)
    #     obj = obj.view(cls)
    #     obj.lattice = getattr(obj, "lattice", None) if lattice is None else lattice
    #     return obj
    #
    # def __array_finalize__(self, obj):
    #     super().__array_finalize__(obj)
    #     if obj is None:
    #         return
    #     if isinstance(obj, SphericalWave) or isinstance(obj, PlaneWave):
    #         raise ValueError(f"cannot cast '{obj.__class__.__name__}' to 'SphericalWave'")
    #     if np.isnan(self.k0):
    #         raise ValueError("'k0' undefined")
    #     if self.basis is None:
    #         self.basis = CylindricalWaveBasis.default(
    #             [0], CylindricalWaveBasis.defaultmmax(self.shape[0])
    #         )
    #         self.basis.hints["kpar"] = [0, 0, 0]
    #     self.hints = getattr(obj, "hints", {}).update(getattr(self.basis, "hints"), {})
    #     self.lattice = getattr(obj, "lattice", None)
    #
    # def efield(self, r):
    #     r = np.array(r)
    #     if r.ndim == 1:
    #         r = np.reshape(r, (1, -1))
    #     r_sph = sc.car2sph(r[..., None, :] - self.basis.positions)
    #     res = self.basis.eval(
    #         self.ks[self.basis.pol] * r_sph[..., self.basis.pidx, 0],
    #         r_sph[..., self.basis.pidx, 1],
    #         r_sph[..., self.basis.pidx, 2],
    #     )
    #     res = sc.vsph2car(res, r_sph[..., self.basis.pidx, :])
    #     if self.ndim == 2:
    #         res = res[..., None]
    #     return np.array(np.sum(res * self[:, None, ...], axis=-max(self.ndim, 1)))
    #
    # def to_sw(self, basis):
    #     if self.basis.wavetype == "regular" and self.lattice is not None:
    #         raise NotImplementedError("conversion from regular periodic CylindricalWave to SphericalWave")
    # @abc.abstractmethod
    # def hfield(self, r):
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def dfield(self, r):
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def bfield(self, r):
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def gfield(self, r, pol):
    #     raise NotImplementedError


class PlaneWaveGeneral(Field):
    _basistype = PlaneWaveBasis
class PlaneWaveUp(PlaneWaveGeneral):
    _basistype = PlaneWaveBasisPartial
class PlaneWaveDown(PlaneWaveGeneral):
    _basistype = PlaneWaveBasisPartial
class PlaneWave(PlaneWaveGeneral):
    _basistype = PlaneWaveBasis
#     def __init__(self, coeff, k0=float('nan'), basis=None, material=None):
#         # Todo: test
#         self.kx = np.atleast_1d(kx)
#         self.ky = np.atleast_1d(ky)
#         self.kz = np.atleast_1d(kz)
#         self.pol = np.atleast_1d(pol)
#         self.k0 = k0
#         self.material = material
#         self.helicity = helicity
#         self.coeff = coeff
#
#     @classmethod
#     def by_efield(cls, kx, ky, kz, pol, evec, helicity=True, k0=float('nan'), material=None):
#         if helicity:
#             v = sc.vpw_A(kx, ky, kz, 0, 0, 0, 1 - pol)
#         else:
#             v = (1 - pol) * sc.vpw_N(kx, ky, kz, 0, 0, 0) - pol * sc.vpw_M(kx, ky, kz, 0, 0, 0)
#         coeff = np.sum(evec * v, axis=-1)
#         return cls(kx, ky, kz, pol, coeff, helicity, k0, material)
#
#     @classmethod
#     def by_diffr_orders_circle(cls, kx, ky, pol, ks, a, rmax, helicity=True):
#         pass
#
#     @classmethod
#     def by_diffr_orders_grid(cls, kx, ky, pol, ks, a, n, helicity=True):
#         pass
#
#     @classmethod
#     def from(cls, source, kx, ky, kz, pol):
#         if isinstance(source, PlaneWave):
#             pass
#         elif isinstance(source, SphericalWave):
#             pass
#         elif isinstance(source, CylindricalWave):
#             pass
#         else:
#             raise ValueError
#
#     @property
#     def modes(self):
#         """
#         Modes of the Q-matrix
#
#         X- and Y-components and polarization of each row/column of the Q-matrix
#
#         Returns:
#             3-tuple
#         """
#         return self.kx, self.ky, self.pol
#
#     def translate(self, modes):
#         pass
#
#     def flip(self, modes):
#         pass
#
#     def pick(self, modes):
#         pass
#
#     def field(r):
#         r = np.array(r)
#         if r.ndim == 1:
#             r = np.reshape(r, (1, -1))
#         if self.helicity:
#             return sc.vpw_A(
#                 self.kx,
#                 self.ky,
#                 self.kz,
#                 r[..., None, 0],
#                 r[..., None, 1],
#                 r[..., None, 2],
#                 self.pol,
#             )
#         else:
#             return (1 - self.pol[:, None]) * sc.vpw_M(
#                 self.kx,
#                 self.ky,
#                 self.kz,
#                 r[..., None, 0],
#                 r[..., None, 1],
#                 r[..., None, 2],
#             ) + self.pol[:, None] * sc.vpw_N(
#                 self.kx,
#                 self.ky,
#                 self.kz[choice, :],
#                 r[..., None, 0],
#                 r[..., None, 1],
#                 r[..., None, 2],
#             )
#     @classmethod
#     def from(cls, source, basis):
#         pidx, l, m, pol = modes
#         positions = np.zeros((3, 1)) if positions is None else positions
#         if positions.ndim == 1:
#             positions = positions[:, None]
#         elif positions.ndim != 2:
#             raise ValueError
#         pos = (*(i[:, None] for i in positions[pidx, :].T),)
#         if isinstance(source, PlaneWave):
#             m = pw.to_sw(
#                 l, m, pol, source.kx, source.ky, source.kz, source.pol, source.helicity
#             ) * pw.translate(source.kx, source.ky, source.kz, *pos)
#         elif isinstance(source, CylindricalWave):
#             m = cw.to_sw(
#                 *(m[:, None] for m in self.modes),
#                 source.kz,
#                 source.m,
#                 source.pol,
#                 self.ks[pol],
#                 posout=self.pidx[:, None],
#                 posin=pidx,
#                 helicity=self.helicity,
#             )
#         elif isinstance(source, SphericalWave):
#             pass
#         else:
#             raise ValueError
#
#     def pick(self, modes):
#         pass
#
#
# class CylindricalWave(Field):
#     def __init__(self, k0, l, m, pol, material=None, coeff=None, helicity=True):
#         self.k0, self.l, self.m, self.pol =
#         self.material = material
#         self.helicity = helicity
#         coeff=np.zeros_like(kx, complex)
#
#     @classmethod
#     def from(cls, source, kx, ky, kz, pol):
#         if isinstance(source, PlaneWave):
#             pass
#         elif isinstance(source, SphericalWave):
#             pass
#         elif isinstance(source, CylindricalWave):
#             pass
#         else:
#             raise ValueError
#
#     def pick(self, modes):
#         pass
#
#
#     def field(self, r):
#         r = np.array(r)
#         if r.ndim == 1:
#             r = np.reshape(r, (1, -1))
#         r_cyl = sc.car2cyl(r[..., None, :] - self.positions)
#         if self.scattered:
#             wave_A = sc.vcw_A
#             wave_M = sc.vcw_M
#             wave_N = sc.vcw_N
#         else:
#             wave_A = sc.vcw_rA
#             wave_M = sc.vcw_rM
#             wave_N = sc.vcw_rN
#         if self.helicity:
#             res = wave_A(
#                 *self.modes[:2],
#                 self.krho * r_cyl[..., self.pidx, 0],
#                 r_cyl[..., self.pidx, 1],
#                 r_cyl[..., self.pidx, 2],
#                 self.ks[self.pol],
#                 self.pol,
#             )
#         else:
#             res = (1 - self.pol[:, None]) * wave_M(
#                 *self.modes[:2],
#                 self.krho * r_cyl[..., self.pidx, 0],
#                 r_cyl[..., self.pidx, 1],
#                 r_cyl[..., self.pidx, 2],
#             ) + self.pol[:, None] * wave_N(
#                 *self.modes[:2],
#                 self.krho * r_cyl[..., self.pidx, 0],
#                 r_cyl[..., self.pidx, 1],
#                 r_cyl[..., self.pidx, 2],
#                 self.ks[self.pol],
#             )
#         return sc.vcyl2car(res, r_cyl[..., self.pidx, :])
