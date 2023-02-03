import warnings

import numpy as np

from ptsa import config
from ptsa._core import CylindricalWaveBasis as CWB
from ptsa._core import PhysicsArray
from ptsa._core import SphericalWaveBasis as SWB
from ptsa._core import PlaneWaveBasis as PWB
from ptsa._core import PlaneWaveBasisPartial as PWBP
from ptsa._material import Material
from ptsa.coeffs import mie, mie_cyl
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
        arr (float or complex, array): T-matrix itself
        k0 (float): Wave number in vacuum
        basis (SphericalWaveBasis, optional): Basis definition
        material (Material, optional): Embedding material, defaults to vacuum
        poltype (str, optional): Helicity or parity basis, defauts to config.POLTYPE

    Attributes:
        t (float or complex, (N,N)-array): T-matrix
        k0 (float): Wave number in vacuum
        basis (SphericalWaveBasis): Basis modes
        material (Material): Material definition
        poltype (str): Helicity or parity basis
    """

    def _check(self):
        super()._check()
        if not isinstance(self.k0, (int, float, np.floating, np.integer)):
            raise AnnotationError("invalid k0")
        if self.poltype is None:
            self.poltype = config.POLTYPE
        if self.poltype not in ("parity", "helicity"):
            raise AnnotationError("invalid poltype")
        modetype = self.modetype
        if modetype is None or (
            modetype[0] in (None, "singular") and modetype[1] in (None, "regular")
        ):
            self.modetype = ("singular", "regular")
        else:
            raise AnnotationError("invalid modetype")
        shape = np.shape(self)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise AnnotationError(f"invalid shape: '{shape}'")
        if self.basis is None:
            self.basis = SWB.default(SWB.defaultlmax(shape[0]))
        if self.material is None:
            self.material = Material()

    @property
    def ks(self):
        return self.material.ks(self.k0)

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
        return cls(tres, k0=k0, material=mat, basis=basis, poltype=poltype)

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
        if not self.material.ischiral:
            ks = self.ks[0]
        else:
            ks = self.ks[self.basis.pol, None]
        re, im = self.real, self.imag
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
        m = self.expand(modetype="regular") / np.power(self.ks[self.basis.pol], 2)
        return (
            0.5 * np.sum(np.real(p.conjugate() * (m @ p)), axis=-2) / flux,
            -0.5 * np.sum(np.real(illu.conjugate() * (m @ p)), axis=-2) / flux,
        )

    def rotated(self, phi, theta=0, psi=0, **kwargs):
        """
        Rotate the T-Matrix by the euler angles

        Rotation is done in-place. If you need the original T-Matrix make a deepcopy
        first. Rotations can only be applied to global T-Matrices. The angles are given
        in the zyz-convention. In the intrinsic (object fixed coordinate system)
        convention the rotations are applied in the order phi first, theta second, psi
        third. In the extrinsic (global or reference frame fixed coordinate system) the
        rotations are applied psi first, theta second, phi third.

        Args:
            phi, theta, psi (float): Euler angles
            modes (array): While rotating also take only specific output modes into
            account

        Returns:
            TMatrix
        """
        r = self.rotate(phi, theta, psi, **kwargs)
        return TMatrix(r @ self @ r.conjugate().T)

    def translated(self, r, **kwargs):
        """
        Translate the origin of the T-Matrix

        Translation is done in-place. If you need the original T-Matrix make a copy
        first. Translations can only be applied to global T-Matrices.

        Args:
            rvec (float array): Translation vector
            modes (array): While translating also take only specific output modes into
            account

        Returns:
            TMatrix
        """
        return TMatrix(self.expand(r, **kwargs) @ self @ self.inv.expand(r, **kwargs))

    def couple(self):
        """
        Calculate the coupling term of a blockdiagonal T-matrix
        """
        return np.eye(self.shape[0]) - self @ self.expand(modetype="regular")

    def interacted(self):
        return TMatrix(np.linalg.solve(self.couple(), self))

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
        return TMatrix(self.expand(basis) @ self @ self.expand.inv(basis))

    def couple_lattice(self, lattice, kpar, *, eta=0):
        """
        Calculate the coupling term of a blockdiagonal T-matrix
        """
        return np.eye(self.shape[0]) - self @ self.expandlattice(lattice, kpar, eta=eta)

    def interacted_lattice(self, lattice, kpar, *, eta=0):
        return TMatrix(
            np.linalg.solve(self.couple_lattice(lattice, kpar, eta=eta), self)
        )

    def grid(self, grid, radii):
        grid = np.asarray(grid)
        if grid.shape[-1] != 3:
            raise ValueError("invalid grid")
        if len(radii) != len(self.basis.positions):
            raise ValueError("invalid length of 'radii'")
        res = np.ones_like(grid, bool)
        for r, p in zip(radii, self.positions):
            res &= np.sum(np.power(grid - p, 2), axis=-1) > r * r
        return res


class TMatrixC(PhysicsArray):
    """
    T-matrix for cylindrical modes

    The T-matrix is square, with the modes defined in the corresponding fields. The
    order of the T-matrix can be arbitrary, but the normalization is fixed to that of
    the modes defined in :func:`ptsa.special.vcw_A`, :func:`ptsa.special.vcw_M`, and
    :func:`ptsa.special.vcw_N`. Helicity and parity modes are possible, but not mixed.

    The embedding medium is described by permittivity, permeability and the chirality
    parameter.

    The T-matrix can be global or local. For a local T-matrix multiple positions have to
    be specified. Also modes must have as first element a position index.

    Args:
        tmat (float or complex, array): T-matrix itself
        k0 (float): Wave number in vacuum
        epsilon (float or complex, optional): Relative permittivity of the embedding
            medium
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

    def _check(self):
        super()._check()
        if not isinstance(self.k0, (int, float, np.floating, np.integer)):
            raise AnnotationError("invalid k0")
        if self.poltype is None:
            self.poltype = config.POLTYPE
        if self.poltype not in ("parity", "helicity"):
            raise AnnotationError("invalid poltype")
        modetype = self.modetype
        if modetype is None or (
            modetype[0] in (None, "singular") and modetype[1] in (None, "regular")
        ):
            self.modetype = ("singular", "regular")
        else:
            raise AnnotationError("invalid modetype")
        shape = np.shape(self)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise AnnotationError(f"invalid shape: '{shape}'")
        if self.basis is None:
            self.basis = CWB.default([0], CWB.defaultmmax(shape[0]))
        if self.material is None:
            self.material = Material()

    @property
    def ks(self):
        return self.material.ks(self.k0)

    @property
    def krhos(self):
        r"""
        Radial part of the wave

        Calculate :math:`\sqrt{k^2 - k_z^2}`.

        Returns:
            float or complex, array
        """
        return self.material.krhos(self.k0, self.basis.kz, self.basis.pol)

    @classmethod
    def cylinder(cls, kzs, mmax, k0, radii, materials):
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
        materials = [Material(m) for m in materials]
        kzs = np.atleast_1d(kzs)
        radii = np.atleast_1d(radii)
        if radii.size != len(materials) - 1:
            raise ValueError("incompatible lengths of radii and materials")
        dim = CWB.defaultdim(len(kzs), mmax)
        tmat = np.zeros((dim, dim), np.complex128)
        idx = 0
        for kz in kzs:
            for m in range(-mmax, mmax + 1):
                miecoeffs = mie_cyl(kz, m, k0, radii, *zip(*materials))
                tmat[idx : idx + 2, idx : idx + 2] = miecoeffs[::-1, ::-1]
                idx += 2
        return cls(tmat, k0=k0, basis=CWB.default(kzs, mmax), material=materials[-1])

    @classmethod
    def from_array(cls, tm, basis, eta=0):
        """1d array of spherical T-matrices"""
        if tm.lattice is None:
            lattice = basis.hints["lattice"]
            kpar = basis.hints["kpar"]
            tm = tm.interacted_lattice(lattice, kpar, eta=eta)
        p = tm.expandlattice(basis=basis)
        a = tm.expand.inv(basis=basis)
        return cls(p @ tm @ a)

    @property
    def xw_ext_avg(self):
        """
        Rotational average of the extinction cross width

        Only implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not self.isglobal or not self.material.isreal:
            raise NotImplementedError
        nk = np.unique(self.kz).size
        if not self.material.ischiral:
            res = -2 * np.real(np.trace(self.t)) / (self.ks[0] * nk)
        else:
            res = 0
            diag = np.diag(self)
            for pol in [0, 1]:
                choice = self.pol == pol
                res += -2 * np.real(diag[choice].sum()) / (self.ks[pol] * nk)
        if res.imag == 0:
            return res.real
        return res

    @property
    def xw_sca_avg(self):
        """
        Rotational average of the scattering cross width

        Only implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not self.isglobal or not self.material.isreal:
            raise NotImplementedError
        if not self.material.ischiral:
            ks = self.ks[0]
        else:
            ks = self.ks[self.basis.pol, None]
        re, im = self.real, self.imag
        nk = np.unique(self.kz).size
        res = 2 * np.sum(re * re + im * im) / (ks * nk)
        return res.real

    def xw(self, illu, flux=0.5):
        r"""
        Scattering and extinction cross width

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
        m = self.expand(modetype="regular") / self.ks[self.basis.pol]
        return (
            2 * np.sum(np.real(p.conjugate() * (m @ p)), axis=-2) / flux,
            -2 * np.sum(np.real(illu.conjugate() * (m @ p)), axis=-2) / flux,
        )

    def rotated(self, phi, **kwargs):
        """
        Rotate the T-Matrix around the z-axis

        Rotation is done in-place. If you need the original T-Matrix make a deepcopy
        first. Rotations can only be applied to global T-Matrices.

        Args:
            phi (float): Rotation angle
            modes (array): While rotating also take only specific output modes into
            account

        Returns:
            T-MatrixC
        """
        r = self.rotate(phi, **kwargs)
        return TMatrixC(r @ self @ r.conjugate().T)

    def translated(self, r, **kwargs):
        """
        Translate the origin of the T-Matrix

        Translation is done in-place. If you need the original T-Matrix make a copy
        first. Translations can only be applied to global T-Matrices.

        Args:
            rvec (float array): Translation vector
            modes (array): While translating also take only specific output modes into
            account

        Returns:
            TMatrixC
        """
        return TMatrixC(self.expand(r, **kwargs) @ self @ self.inv.expand(r, **kwargs))

    def couple(self):
        """
        Calculate the coupling term of a blockdiagonal T-matrix
        """
        return np.eye(self.shape[0]) - self @ self.expand(modetype="regular")

    def interacted(self):
        return TMatrixC(np.linalg.solve(self.couple(), self))

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

        """
        if not basis.isglobal:
            raise ValueError("global basis required")
        if basis is None:
            basis = CWB.default(np.unique(self.kz), max(self.basis.l))
        return TMatrix(self.expand(basis) @ self @ self.expand.inv(basis))

    def couple_lattice(self, lattice, kpar, *, eta=0):
        """
        Calculate the coupling term of a blockdiagonal T-matrix
        """
        return np.eye(self.shape[0]) - self @ self.expandlattice(lattice, kpar, eta=eta)

    def interacted_lattice(self, lattice, kpar, eta=0):
        return TMatrix(
            np.linalg.solve(self.couple_lattice(lattice, kpar, eta=eta), self)
        )

    def grid(self, grid, radii):
        grid = np.asarray(grid)
        if grid.shape[-1] not in (2, 3):
            raise ValueError("invalid grid")
        if len(radii) != len(self.basis.positions):
            raise ValueError("invalid length of 'radii'")
        res = np.ones_like(grid, bool)
        for r, p in zip(radii, self.positions):
            res &= np.sum(np.power(grid[..., :2] - p[:2], 2), axis=-1) > r * r
        return res


def plane_wave(*args, basis=None, k0=None, material=None, modetype=None, poltype=None):
    *args, amp = args
    if basis is None:
        if len(args) == 4:
            basis = PWB.default([args])
        else:
            basis = PWBP.default([args])
    if isinstance(basis, PWBP):
        modetype = "up" if modetype is None else modetype
        material = Material() if material is None else Material(material)
        if len(args) == 4:
            basis_c = basis.complete(k0, material, modetype)
        else:
            basis_c = basis
    elif k0 is not None:
        material = Material() if material is None else Material(material)
        basis_c = basis.complete(k0, material, modetype)
    else:
        basis_c = basis
    args = np.array(args)
    res = [np.abs(args - x) < 1e-14 for x in basis_c]
    if sum(res) != 1:
        raise ValueError("cannot find matching mode in basis")
    res = np.array(res, complex) * amp
    return PhysicsArray(
        res, basis=basis, k0=k0, material=material, modetype=modetype, poltype=poltype
    )


def spherical_wave(
    l,  # noqa: E741
    m,
    pol,
    amp,
    *,
    k0,
    basis=None,
    material=None,
    modetype=None,
    poltype=None,
):
    if basis is None:
        basis = SWB.default(l)
    if not basis.isglobal:
        raise ValueError("basis must be global")
    res = [0] * len(basis)
    res[basis.index((0, l, m, pol))] = amp
    return PhysicsArray(
        res, basis=basis, k0=k0, material=material, modetype=modetype, poltype=poltype
    )


def cylindrical_wave(
    kz, m, pol, amp, *, k0, basis=None, material=None, modetype=None, poltype=None
):
    if basis is None:
        basis = CWB.default([kz], abs(m))
    if not basis.isglobal:
        raise ValueError("basis must be global")
    res = [0] * len(basis)
    res[basis.index((0, kz, m, pol))] = amp
    return PhysicsArray(
        res, basis=basis, k0=k0, material=material, modetype=modetype, poltype=poltype
    )
