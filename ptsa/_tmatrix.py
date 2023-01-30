import warnings

import numpy as np

import ptsa.lattice as la
import ptsa.special as sc
from ptsa import config, io, misc, pw, sw
from ptsa._core import PhysicsArray
from ptsa._core import SphericalWaveBasis as SWB
from ptsa._material import Material
from ptsa.coeffs import mie
from ptsa.util import AnnotationError


class TMatrix(PhysicsArray):
    """
    T-matrix for spherical modes

    The T-matrix is square, with the modes defined in the corresponding fields. The
    order of the T-matrix can be arbitrary, but the normalization is fixed to that of
    the modes defined in :func:`ptsa.special.vsw_A`, :func:`ptsa.special.vsw_M`, and
    :func:`ptsa.special.vsw_N`. A default order according to
    :func:`TMatrix.defaultmodes` is assumed if not specified. Helicity and parity modes
    are possible, but not mixed.

    The embedding medium is described by permittivity, permeability, and the chirality
    parameter.

    The T-matrix can be global or local. For a local T-matrix multiple positions have to
    be specified. Also modes must have as first element a position index.

    Args:
        tmat (float or complex, array): T-matrix itself
        k0 (float): Wave number in vacuum
        epsilon (float or complex, optional): Relative permittivity of the embedding
            medium
        mu (float or complex, optional): Relative permeability of the embedding medium
        kappa (float or complex, optional): Chirality parameter of the embedding medium
        positions (float, (3,)- or (M,3)-array, optional): Positions for a local
            T-matrix
        helicity (bool, optional): Helicity or parity modes
        modes (iterable, optional): Sorting of the T-matrix entries. Either four entries
            for local T-Matrices, with the first specifying the corresponding position
            or three entries only, specifying degree, order, and polarization.

    Attributes:
        t (float or complex, (N,N)-array): T-matrix
        k0 (float): Wave number in vacuum
        positions (float, (3,)- or (M,3)-array): Positions for a local T-matrix
        epsilon (float or complex, optional): Relative permittivity of the embedding
            medium
        mu (complex): Relative permeability of the embedding medium
        kappa (complex): Chirality parameter of the embedding medium
        helicity (bool): Helicity or parity modes
        pidx (int, (N,)-array): Position index for each column/row of the T-matrix
        l (int, (N,)-array): Degree of the mode for each column/row of the T-matrix
        m (int, (N,)-array): Order of the mode for each column/row of the T-matrix
        pol (int, (N,)-array): Polarization of the mode for each column/row of the
            T-matrix
        ks (float or complex (2,)-array): Wave numbers in the medium for both
            polarizations
    """
    def __str__(self):
        return str(self._array)

    def _check(self):
        super()._check()
        if not isinstance(self.k0, (int, float, np.floating, np.integer)):
            raise AnnotationError("invalid k0")
        if self.poltype is None:
            self.poltype = config.POLTYPE
        if self.poltype not in ("parity", "helicity"):
            raise AnnotationError("invalid poltype")
        shape = np.shape(self)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise AnnotationError(f"invalid shape: '{shape}'")
        if self.basis is None:
            self.basis = SWB.default(SWB.defaultlmax(shape[0]))
        if self.material is None:
            self.material = Material()

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    #     if (
    #         ufunc.signature is not None
    #         and any(map(lambda x: x not in " (),->", ufunc.signature))
    #     ) or (method not in ("__call__", "accumulate", "at")):
    #         inputs = [PhysicsArray(a) if isinstance(a, TMatrix) else a for a in inputs]
    #         return PhysicsArray(self).__array_ufunc__(ufunc, method, *inputs, **kwargs)
    #     return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    # def __array_function__(self, func, types, args, kwargs):
    #     if func in DECAY_FUNCTIONS:
    #         args = [PhysicsArray(a) if isinstance(a, TMatrix) else a for a in args]
    #         kwargs = {
    #             k: PhysicsArray(a) if isinstance(a, TMatrix) else a
    #             for k, a in kwargs.items()
    #         }
    #     return super().__array_function__(func, types, args, kwargs)

    @property
    def ks(self):
        return self.material.ks(self.k0)

    # @decays(np.trace)
    # def trace(self, *args, **kwargs):
    #     return PhysicsArray(self).trace(*args, **kwargs)

    # @decays(np.sum)
    # def sum(self, *args, **kwargs):
    #     return PhysicsArray(self).sum(*args, **kwargs)

    @classmethod
    def sphere(cls, lmax, k0, radii, materials):
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
        materials = [Material(m) for m in materials]
        radii = np.atleast_1d(radii)
        if radii.size != len(materials) - 1:
            raise ValueError("incompatible lengths of radii and materials")
        dim = SWB.defaultdim(lmax)
        tmat = np.zeros((dim, dim), np.complex128)
        for l in range(1, lmax + 1):  # noqa: E741
            miecoeffs = mie(l, k0 * radii, *zip(*materials))
            pos = SWB.defaultdim(l - 1)
            for i in range(2 * l + 1):
                tmat[
                    pos + 2 * i : pos + 2 * i + 2, pos + 2 * i : pos + 2 * i + 2
                ] = miecoeffs[::-1, ::-1]
        return cls(tmat, k0=k0, basis=SWB.default(lmax), material=materials[-1])

    @classmethod
    def cluster(cls, tmats, positions):
        for tm in tmats:
            if not tm.basis.isglobal:
                raise ValueError("global basis required")
        positions = np.array(positions)
        if len(tmats) < positions.shape[0]:
            warnings.warn("specified more positions than T-matrices")
        elif len(tmats) > positions.shape[0]:
            raise ValueError(
                f"'{len(tmats)}' T-matrices "
                f"but only '{positions.shape[0]}' positions given"
            )
        mat = tmats[0].material
        k0 = tmats[0].k0
        poltype = tmats[0].poltype
        modes = [], [], []
        pidx = []
        dim = sum([tmat.shape[0] for tmat in tmats])
        tres = np.zeros((dim, dim), complex)
        i = 0
        for j, tm in enumerate(tmats):
            if tm.material != mat:
                raise ValueError(f"incompatible materials: '{mat}' and '{tm.material}'")
            if tm.k0 != k0:
                raise ValueError(f"incompatible k0: '{k0}' and '{tm.k0}'")
            if tm.poltype != poltype:
                raise ValueError(f"incompatible modetypes: '{poltype}', '{tm.poltype}'")
            dim = tm.shape[0]
            for m, n in zip(modes, tm.basis["lmp"]):
                m.extend(list(n))
            pidx += [j] * dim
            tres[i : i + dim, i : i + dim] = tm
            i += dim
        basis = SWB(zip(pidx, *modes), positions)
        obj = cls(tres, k0=k0, material=mat, basis=basis, poltype=poltype)
        obj.interacted = False

    # @classmethod
    # def ebcm(cls, lmax, k0, r, dr, epsilon, mu=1, kappa=0, lint=None):
    #     ks = k0 * misc.refractive_index(epsilon, mu, kappa)
    #     zs = np.sqrt(np.array(mu) / epsilon)
    #     modes = cls.ebcmmodes(lmax)
    #     modes_int = modes if lint is None else cls.ebcmmodes(lint, mmax=lmax)
    #     qm = ebcm.qmat(r, dr, ks, zs, modes_int[1:], modes[1:])
    #     rqm = ebcm.qmat(r, dr, ks, zs, modes_int[1:], modes[1:], singular=False)
    #     tm = -np.linalg.lstsq(qm, rqm, rcond=None)[0]
    #     return cls(tm, k0, epsilon[-1], mu[-1], kappa[-1], modes=modes)

    @property
    def isglobal(self):
        return self.basis.isglobal and self.lattice is None and self.kpar is None

    @property
    def xs_ext_avg(self):
        """
        Rotational average of the extinction cross section

        Only implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not self.isglobal or not self.material.isreal:
            raise NotImplementedError
        if not self.material.ischiral:
            k = self.ks[0]
            res = -2 * np.pi * self.trace().real / (k * k)
        else:
            res = 0
            diag = np.diag(self)
            for pol in [0, 1]:
                choice = self.basis.pol == pol
                k = self.ks[pol]
                res += -2 * np.pi * diag[choice].sum().real / (k * k)
        if res.imag == 0:
            return res.real
        return res

    @property
    def xs_sca_avg(self):
        """
        Rotational average of the scattering cross section

        Only implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not self.isglobal or not self.material.isreal:
            raise NotImplementedError
        re, im = self.real, self.imag
        if not self.material.ischiral:
            ks = self.ks[0]
        else:
            ks = self.ks[self.basis.pol, None]
        res = 2 * np.pi * np.sum((re * re + im * im) / (ks * ks))
        return res.real

    @property
    def cd(self):
        """
        Circular dichroism

        Only implemented for global T-matrices.
        """
        if not (self.isglobal and self.poltype == "helicity" and self.material.isreal):
            raise NotImplementedError
        sel = np.array(self.basis.pol, bool)
        re, im = self.real, self.imag
        plus = -np.sum(re[sel, sel]) / (self.ks[1] * self.ks[1])
        re_part = re[:, sel] / self.ks[self.basis.pol, None]
        im_part = im[:, sel] / self.ks[self.basis.pol, None]
        plus -= np.sum(re_part * re_part + im_part * im_part)
        sel = ~sel
        minus = -np.sum(re[sel, sel]) / (self.ks[0] * self.ks[0])
        re_part = re[:, sel] / self.ks[self.basis.pol, None]
        im_part = im[:, sel] / self.ks[self.basis.pol, None]
        minus -= np.sum(re_part * re_part + im_part * im_part)
        return np.real((plus - minus) / (plus + minus))

    @property
    def chi(self):
        """
        Electromagnetic chirality

        Only implemented for global T-matrices.

        Returns:
            float
        """
        if not (self.isglobal and self.poltype == "helicity"):
            raise NotImplementedError
        sel = self.basis.pol == 0, self.basis.pol == 1
        spp = np.linalg.svd(self[np.ix_(sel[1], sel[1])], compute_uv=False)
        spm = np.linalg.svd(self[np.ix_(sel[1], sel[0])], compute_uv=False)
        smp = np.linalg.svd(self[np.ix_(sel[0], sel[1])], compute_uv=False)
        smm = np.linalg.svd(self[np.ix_(sel[0], sel[0])], compute_uv=False)
        plus = np.concatenate((np.asarray(spp), np.asarray(spm)))
        minus = np.concatenate((np.asarray(smm), np.asarray(smp)))
        return np.linalg.norm(plus - minus) / np.sqrt(np.sum(np.power(np.abs(self), 2)))

    @property
    def db(self):
        """
        Duality breaking

        Only implemented for global T-matrices.

        Returns:
            float
        """
        if not (self.isglobal and self.poltype == "helicity"):
            raise NotImplementedError
        sel = self.basis.pol == 0, self.basis.pol == 1
        tpm = np.asarray(self[np.ix_(sel[1], sel[0])])
        tmp = np.asarray(self[np.ix_(sel[0], sel[1])])
        return np.sum(
            tpm.real * tpm.real
            + tpm.imag * tpm.imag
            + tmp.real * tmp.real
            + tmp.imag * tmp.imag
        ) / (np.sum(np.power(np.abs(self), 2)))

    def xs(self, illu, flux=0.5):
        r"""
        Scattering and extinction cross section

        Possible for all T-matrices (global and local) in non-absorbing embedding.

        Args:
            illu (complex, array): Illumination coefficients
            flux (optional): Ingoing flux corresponding to the illumination. Used for
                the result's normalization. The flux is given in units of
                :math:`\frac{\text{V}^2}{{l^2}} \frac{1}{Z_0 Z}` where :math:`l` is the
                unit of length used in the wave number (and positions). A plane wave
                has the flux `0.5` in this normalization, which is used as default.

        Returns:
            float, (2,)-tuple
        """
        if not self.material.isreal:
            raise NotImplementedError
        illu = PhysicsArray(illu)
        p = self @ illu
        m = self.basis.expand(
            self.k0, self.basis, poltype=self.poltype, material=self.material
        ) / np.power(self.ks[self.basis.pol], 2)
        return (
            0.5 * np.sum(np.real(p.conjugate() * (m @ p)), axis=-2) / flux,
            -0.5 * np.sum(np.real(illu.conjugate() * (m @ p)), axis=-2) / flux,
        )

    # def rotate(self, phi, theta=0, psi=0, *, basis=None):
    #     """
    #     Rotate the T-Matrix by the euler angles

    #     Rotation is done in-place. If you need the original T-Matrix make a deepcopy
    #     first. Rotations can only be applied to global T-Matrices. The angles are given
    #     in the zyz-convention. In the intrinsic (object fixed coordinate system)
    #     convention the rotations are applied in the order phi first, theta second, psi
    #     third. In the extrinsic (global or reference frame fixed coordinate system) the
    #     rotations are applied psi first, theta second, phi third.

    #     Args:
    #         phi, theta, psi (float): Euler angles
    #         modes (array): While rotating also take only specific output modes into
    #         account

    #     Returns:
    #         TMatrix
    #     """
    #     mat = self.basis.rotate(phi, theta, psi, basis=basis)
    #     return TMatrix(mat @ self @ mat.conjugate().T)

    # def translate(self, rvec, basis=None):
    #     """
    #     Translate the origin of the T-Matrix

    #     Translation is done in-place. If you need the original T-Matrix make a copy
    #     first. Translations can only be applied to global T-Matrices.

    #     Args:
    #         rvec (float array): Translation vector
    #         modes (array): While translating also take only specific output modes into
    #         account

    #     Returns:
    #         TMatrix
    #     """
    #     if basis is None:
    #         basis = self.basis
    #     if not (self.basis.isglobal and basis.isglobal):
    #         raise NotImplementedError
    #     matin = basis.translate(
    #         rvec,
    #         self.k0,
    #         basis=self.basis,
    #         material=self.material,
    #         poltype=self.poltype,
    #     )
    #     matout = self.basis.translate(
    #         np.negative(rvec),
    #         self.k0,
    #         basis=basis,
    #         material=self.material,
    #         poltype=self.poltype,
    #     )
    #     return TMatrix(matout @ self @ matin)

    def coupling(self, *, lattice=None, kpar=None):
        """
        Calculate the coupling term of a blockdiagonal T-matrix

        Returns:
            TMatrix
        """
        if self.lattice is not None:
            raise NotImplementedError
        m = self.basis.expand(
            self.k0,
            basis=self.basis,
            poltype=self.poltype,
            material=self.material,
            modetype="singular",
            lattice=lattice,
            kpar=kpar,
        )
        return np.eye(self.shape[0]) - self @ m

    def globalmat(self, basis=None):
        """
        Global T-matrix

        Calculate the global T-matrix starting from a local one. This changes the
        T-matrix.

        Args:
            origin (array, optional): The origin of the new T-matrix
            modes (array, optional): The modes that are considered for the global
                T-matrix
            interacted (bool, optional): If set to `False` the interaction is calulated
                first.

        Returns
            TMatrix
        """
        basis = SWB.default(max(self.basis.l)) if basis is None else basis
        if not basis.isglobal:
            raise ValueError("Global basis required")
        ain = basis.expand(k0=self.k0, basis=self.basis, poltype=self.poltype)
        pout = self.basis.expand(k0=self.k0, basis=basis, poltype=self.poltype)
        if self.interacted:
            return TMatrix(pout @ self @ ain)
        return TMatrix(pout @ np.linalg.solve(self.coupling(), self @ ain))

    # def field(self, r, scattered=True):
    #     """
    #     Calculate the scattered or incident field at specified points

    #     The mode expansion of the T-matrix is used

    #     Args:
    #         r (float, array_like): Array of the positions to probe
    #         scattered (bool, optional): Select the scattered (default) or incident field

    #     Returns
    #         complex
    #     """
    #     r = np.array(r)
    #     if r.ndim == 1:
    #         r = np.reshape(r, (1, -1))
    #     r_sph = sc.car2sph(r[..., None, :] - self.positions)
    #     if scattered:
    #         wave_A = sc.vsw_A
    #         wave_M = sc.vsw_M
    #         wave_N = sc.vsw_N
    #     else:
    #         wave_A = sc.vsw_rA
    #         wave_M = sc.vsw_rM
    #         wave_N = sc.vsw_rN
    #     if self.helicity:
    #         res = wave_A(
    #             *self.modes[:2],
    #             self.ks[self.pol] * r_sph[..., self.pidx, 0],
    #             r_sph[..., self.pidx, 1],
    #             r_sph[..., self.pidx, 2],
    #             self.pol,
    #         )
    #     else:
    #         res = (1 - self.pol[:, None]) * wave_M(
    #             *self.modes[:2],
    #             self.ks[self.pol] * r_sph[..., self.pidx, 0],
    #             r_sph[..., self.pidx, 1],
    #             r_sph[..., self.pidx, 2],
    #         ) + self.pol[:, None] * wave_N(
    #             *self.modes[:2],
    #             self.ks[self.pol] * r_sph[..., self.pidx, 0],
    #             r_sph[..., self.pidx, 1],
    #             r_sph[..., self.pidx, 2],
    #         )
    #     return sc.vsph2car(res, r_sph[..., self.pidx, :])

    # def latticecoupling(self, kpar, a, eta=0):
    #     r"""
    #     The coupling of the T-matrix in a lattice

    #     Returns

    #     .. math::

    #         \mathbb 1 - T C

    #     The inverse of this multiplied to the T-matrix in `latticeinteract`. The lattice
    #     type is inferred from `kpar`.

    #     Args:
    #         kpar (float): The parallel component of the T-matrix
    #         a (array): Definition of the lattice
    #         eta (float or complex, optional): Splitting parameter in the lattice sum

    #     Returns:
    #         complex, array
    #     """
    #     m = sw.translate_periodic(
    #         self.ks,
    #         kpar,
    #         a,
    #         self.positions,
    #         self.fullmodes,
    #         helicity=self.helicity,
    #         eta=eta,
    #     )
    #     return np.eye(self.t.shape[0]) - self.t @ m

    # def lattice_field(self, r, modes, kpar, a, eta=0):
    #     """
    #     Field expansion at a specified point in a lattice

    #     Args:
    #         r (float, array_like): Positions
    #         modes (tuple): Modes of the expansion
    #         kpar (float, array_like: Parallel component of the wave vector
    #         a (float, array_like): Lattice vectors
    #         eta (float or complex, optional): Splitting parameter in the lattice sum

    #     Returns:
    #         complex
    #     """
    #     r = np.array(r)
    #     if r.ndim == 1:
    #         r = r.reshape((1, -1))
    #     return sw.translate_periodic(
    #         self.ks,
    #         kpar,
    #         a,
    #         r,
    #         modes,
    #         in_=self.fullmodes,
    #         rsin=self.positions,
    #         helicity=self.helicity,
    #         eta=eta,
    #     )

    def array1d(self, basis, lattice=None, kpar=None, eta=0):
        """
        Convert a one-dimensional array of T-matrices into a (cylindrical) 2D-T-matrix

        Args:
            modes (tuple): Cylindrical wave modes
            a (float): Lattice pitch
            eta (float or complex, optional): Splitting parameter in the lattice sum

        Returns:
            complex, array
        """
        lattice = basis.hints["lattice"] if lattice is None else lattice
        kpar = basis.hints["kpar"] if kpar is None else lattice
        interaction = self.interact(lattice, kpar, eta=eta)
        ain = basis.expand(self.k0, self.basis, material=self.material, poltype=self.poltype)
        pout = self.basis.expand(self.k0, basis, material=self.material, poltype=self.poltype,)
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
            eta (float or complex, optional): Splitting parameter in the lattice sum

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
