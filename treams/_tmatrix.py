import warnings

import numpy as np

import treams.special as sc
from treams import config
from treams._core import CylindricalWaveBasis as CWB
from treams._core import PhysicsArray
from treams._core import PlaneWaveBasisByUnitVector as PWBUV
from treams._core import PlaneWaveBasisByComp as PWBC
from treams._core import SphericalWaveBasis as SWB
import treams._operators as op
from treams._material import Material
from treams.coeffs import mie, mie_cyl
from treams.util import AnnotationError


class _Interact:
    def __init__(self):
        self._obj = self._objtype = None

    def __get__(self, obj, objtype=None):
        self._obj = obj
        self._objtype = objtype

    def __call__(self):
        basis = self._obj.basis
        return np.eye(self._obj.shape[-1]) - self._obj @ op.Expand(basis, "singular")

    def solve(self):
        return np.linalg.solve(self(), self._obj)


class _InteractLattice:
    def __init__(self):
        self._obj = self._objtype = None

    def __get__(self, obj, objtype=None):
        self._obj = obj
        self._objtype = objtype

    def __call__(self, lattice, kpar):
        return np.eye(self._obj.shape[-1]) - self._obj @ op.ExpandLattice(
            lattice=lattice, kpar=kpar
        )

    def solve(self, lattice, kpar):
        return np.linalg.solve(self(), self._obj)


class TMatrix(PhysicsArray):
    """T-matrix with a spherical basis.

    The T-matrix is square relating incident (regular) fields
    :func:`treams.special.vsw_rA` (helical polarizations) or
    :func:`treams.special.vsw_rN` and :func:`treams.special.vsw_rM` (parity
    polarizations) to the corresponding scattered fields :func:`treams.special.vsw_A` or
    :func:`treams.special.vsw_N` and :func:`treams.special.vsw_M`. The modes themselves
    are defined in :attr:`basis`, the polarization type in :attr:`poltype`. Also, the
    wave number :attr:`k0` and, if not vacuum, the material :attr:`material` are
    specified.

    Args:
        arr (float or complex, array-like): T-matrix itself.
        k0 (float): Wave number in vacuum.
        basis (SphericalWaveBasis, optional): Basis definition.
        material (Material, optional): Embedding material, defaults to vacuum.
        poltype (str, optional): Polarization type (:ref:`params:Polarizations`).
        lattice (Lattice, optional): Lattice definition. If specified the T-Matrix is
            assumed to be periodically repeated in the defined lattice.
        kpar (list, optional): Phase factor for the periodic T-Matrix.
    """

    interact = _Interact()
    interactlattice = _InteractLattice()

    def _check(self):
        """Fill in default values or raise errors for missing attributes."""
        super()._check()
        shape = np.shape(self)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise AnnotationError(f"invalid shape: '{shape}'")
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
        if self.basis is None:
            self.basis = SWB.default(SWB.defaultlmax(shape[0]))
        if self.material is None:
            self.material = Material()

    @property
    def ks(self):
        """Wave numbers (in medium).

        The wave numbers for both polarizations.
        """
        return self.material.ks(self.k0)

    @classmethod
    def sphere(cls, lmax, k0, radii, materials):
        """T-Matrix of a (multi-layered) sphere.

        Construct the T-matrix of the given order and material for a sphere. The object
        can also consist of multiple concentric spherical shells with an arbitrary
        number of layers. The calculation is always done in helicity basis.

        Args:
            lmax (int): Positive integer for the maximum degree of the T-matrix.
            k0 (float): Wave number in vacuum.
            radii (float or array): Radii from inside to outside of the sphere. For a
                simple sphere the radius can be given as a single number, for a multi-
                layered sphere it is a list of increasing radii for all shells.
            material (list[Material]): The material parameters from the inside to the
                outside. The last material in the list specifies the embedding medium.

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
        r"""Block-diagonal T-matrix of multiple objects.

        Construct the initial block-diagonal T-matrix for a cluster of objects. The
        T-matrices in the list are placed together into a block-diagonal matrix and the
        complete (local) basis is defined based on the individual T-matrices and their
        bases together with the defined positions. In mathematical terms the matrix

        .. math::

            \begin{pmatrix}
                T_0 & 0 & \dots & 0 \\
                0 & T_1 & \ddots & \vdots \\
                \vdots & \ddots & \ddots & 0 \\
                0 & \dots & 0 & T_{N-1} \\
            \end{pmatrix}

        is created from the list of T-matrices :math:`(T_0, \dots, T_{N-1})`. Only
        T-matrices of the same wave number, embedding material, and polarization type
        can be combined.

        Args:
            tmats (Sequence): List of T-matrices.
            positions (array): The positions of all individual objects in the cluster.

        Returns:
            TMatrix
        """
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
            for m, n in zip(modes, tm.basis.lmp):
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
        """Test if a T-matrix is global.

        A T-matrix is considered global, when its basis refers to only a single point
        and it is not placed periodically in a lattice.
        """
        return self.basis.isglobal and self.lattice is None and self.kpar is None

    @property
    def xs_ext_avg(self):
        r"""Rotation and polarization averaged extinction cross section.

        The average is calculated as

        .. math::

            \langle \sigma_\mathrm{ext} \rangle
            = -2 \pi \sum_{slm} \frac{\Re(T_{slm,slm})}{k_s^2}

        where :math:`k_s` is the wave number in the embedding medium for the
        polarization :math:`s`. It is only implemented for global T-matrices.

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
            modetype = self.modetype
            del self.modetype
            diag = np.diag(self)
            self.modetype = modetype
            for pol in [0, 1]:
                choice = self.basis.pol == pol
                k = self.ks[pol]
                res += -2 * np.pi * diag[choice].sum().real / (k * k)
        if res.imag == 0:
            return res.real
        return res

    @property
    def xs_sca_avg(self):
        r"""Rotation and polarization averaged scattering cross section.

        The average is calculated as

        .. math::

            \langle \sigma_\mathrm{sca} \rangle
            = 2 \pi \sum_{slm} \sum_{s'l'm'}
            \frac{|T_{slm,s'l'm'}|^2}{k_s^2}

        where :math:`k_s` is the wave number in the embedding medium for the
        polarization :math:`s`. It is only implemented for global T-matrices.

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
        r"""Circular dichroism (CD).

        The CD is calculated as

        .. math::

            CD
            = \frac{\langle \sigma_\mathrm{abs} \rangle_+
            - \langle \sigma_\mathrm{abs} \rangle_-}
            {\langle \sigma_\mathrm{abs} \rangle_+
            + \langle \sigma_\mathrm{abs} \rangle_-}

        where :math:`\langle \sigma \rangle_s` is the rotationally averaged absorption
        cross section under the illumination with the polarization :math:`s`.
        It is only implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not (self.isglobal and self.poltype == "helicity" and self.material.isreal):
            raise NotImplementedError
        sel = np.array(self.basis.pol, bool)
        re, im = self.real, self.imag
        plus = -np.sum(re[sel[:, None] & sel]) / (self.ks[1] * self.ks[1])
        re_part = re[:, sel] / self.ks[self.basis.pol, None]
        im_part = im[:, sel] / self.ks[self.basis.pol, None]
        plus -= np.sum(re_part * re_part + im_part * im_part)
        sel = ~sel
        minus = -np.sum(re[sel[:, None] & sel]) / (self.ks[0] * self.ks[0])
        re_part = re[:, sel] / self.ks[self.basis.pol, None]
        im_part = im[:, sel] / self.ks[self.basis.pol, None]
        minus -= np.sum(re_part * re_part + im_part * im_part)
        return np.real((plus - minus) / (plus + minus))

    @property
    def chi(self):
        """Electromagnetic chirality.

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
        """Duality breaking.

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
        r"""Scattering and extinction cross section.

        Possible for all T-matrices (global and local) in non-absorbing embedding. The
        values are calculated by

        .. math::

            \sigma_\mathrm{sca}
            = \frac{1}{2 I}
            a_{slm}^\ast T_{s'l'm',slm}^\ast k_{s'}^{-2} C_{s'l'm',s''l''m''}^{(1)}
            T_{s''l''m'',s'''l'''m'''} a_{s'''l'''m'''} \\
            \sigma_\mathrm{ext}
            = \frac{1}{2 I}
            a_{slm}^\ast k_s^{-2} T_{slm,s'l'm'} a_{s'l'm'}

        where :math:`a_{slm}` are the expansion coefficients of the illumination,
        :math:`T` is the T-matrix, :math:`C^{(1)}` is the (regular) translation
        matrix and :math:`k_s` are the wave numbers in the medium. All repeated indices
        are summed over. The incoming flux is :math:`I`.

        Args:
            illu (complex, array): Illumination coefficients
            flux (optional): Ingoing flux corresponding to the illumination. Used for
                the result's normalization. The flux is given in units of
                :math:`\frac{\text{V}^2}{{l^2}} \frac{1}{Z_0 Z}` where :math:`l` is the
                unit of length used in the wave number (and positions). A plane wave
                has the flux `0.5` in this normalization, which is used as default.

        Returns:
            tuple[float]
        """
        if not self.material.isreal:
            raise NotImplementedError
        illu = PhysicsArray(illu)
        illu_basis = illu.basis
        illu_basis = illu_basis[-2] if isinstance(illu_basis, tuple) else illu_basis
        if not isinstance(illu_basis, SWB):
            illu = illu.expand(self.basis) @ illu
        p = self @ illu
        invksq = np.power(self.ks[self.basis.pol], -2)
        m = self.expand() * invksq
        del illu.modetype
        return (
            0.5 * np.real(p.conjugate().T @ (m @ p)) / flux,
            -0.5 * np.real(illu.conjugate().T @ (p * invksq)) / flux,
        )

    # def rotated(self, phi, theta=0, psi=0, **kwargs):
    #     """Rotated T-Matrix.

    #     Rotation is done in-place. If you need the original T-Matrix make a deepcopy
    #     first. Rotations can only be applied to global T-Matrices. The angles are given
    #     in the zyz-convention. In the intrinsic (object fixed coordinate system)
    #     convention the rotations are applied in the order phi first, theta second, psi
    #     third. In the extrinsic (global or reference frame fixed coordinate system) the
    #     rotations are applied psi first, theta second, phi third.

    #     Args:
    #         phi, theta, psi (float): Euler angles

    #     Returns:
    #         TMatrix
    #     """
    #     r = self.rotate(phi, theta, psi, **kwargs)
    #     return TMatrix(r @ self @ r.conjugate().T)

    # def translated(self, r, **kwargs):
    #     """Translated T-Matrix.

    #     Translation is done in-place.

    #     Args:
    #         r (float array): Translation vector
    #         modes (array): While translating also take only specific output modes into
    #         account

    #     Returns:
    #         TMatrix
    #     """
    #     return TMatrix(
    #         self.translate(r, **kwargs) @ self @ self.translate.inv(r, **kwargs)
    #     )

    # def couple(self):
    #     """Calculate the coupling term of a block-diagonal T-matrix."""
    #     return np.eye(self.shape[0]) - self @ self.expand(modetype="regular")

    # def interacted(self):
    #     """T-matrix with the self-interaction included."""
    #     return TMatrix(np.linalg.solve(self.couple(), self))

    def globalmat(self, basis=None):
        """Global T-matrix.

        Calculate the global T-matrix starting from a local one. This changes the
        T-matrix.

        Args:
            basis (SphericalWaveBasis, optional): The basis of the global T-matrix. By
                default the default basis for the maximal multipolar order found is
                used.

        Returns
            TMatrix
        """
        basis = SWB.default(max(self.basis.l)) if basis is None else basis
        return TMatrix(self.expand(basis) @ self @ self.expand.inv(basis))

    # def couple_lattice(self, lattice, kpar, *, eta=0):
    #     """Calculate the coupling term of a block-diagonal T-matrix."""
    #     return np.eye(self.shape[0]) - self @ self.expandlattice(lattice, kpar, eta=eta)

    # def interacted_lattice(self, lattice, kpar, *, eta=0):
    #     """T-matrix with lattice interaction included."""
    #     obj = TMatrix(
    #         np.linalg.solve(self.couple_lattice(lattice, kpar, eta=eta), self)
    #     )
    #     lattice, _ = obj.lattice
    #     obj.lattice = lattice
    #     kpar, _ = obj.kpar
    #     obj.kpar = kpar
    #     return obj

    def valid_points(self, grid, radii):
        """Points on the grid where the expansion is valid.

        The expansion of the electromagnetic field is valid outside of the
        circumscribing spheres of each object. From a given set of coordinates mark
        those that are outside of the given radii.

        Args:
            grid (array-like): Points to assess. The last dimension needs length three
                and corresponds to the Cartesian coordinates.
            radii (Sequence[float]): Radii of the circumscribing spheres. Each radius
                corresponds to a position of the basis.

        Returns:
            array
        """
        grid = np.asarray(grid)
        if grid.shape[-1] != 3:
            raise ValueError("invalid grid")
        if len(radii) != len(self.basis.positions):
            raise ValueError("invalid length of 'radii'")
        res = np.ones(grid.shape[:-1], bool)
        for r, p in zip(radii, self.basis.positions):
            res &= np.sum(np.power(grid - p, 2), axis=-1) > r * r
        return res


class TMatrixC(PhysicsArray):
    """T-matrix with a cylindrical basis.

    The T-matrix is square relating incident (regular) fields
    :func:`treams.special.vcw_rA` (helical polarizations) or
    :func:`treams.special.vcw_rN` and :func:`treams.special.vcw_rM` (parity
    polarizations) to the corresponding scattered fields :func:`treams.special.vcw_A` or
    :func:`treams.special.vcw_N` and :func:`treams.special.vcw_M`. The modes themselves
    are defined in :attr:`basis`, the polarization type in :attr:`poltype`. Also, the
    wave number :attr:`k0` and, if not vacuum, the material :attr:`material` are
    specified.

    Args:
        arr (float or complex, array-like): T-matrix itself.
        k0 (float): Wave number in vacuum.
        basis (SphericalWaveBasis, optional): Basis definition.
        material (Material, optional): Embedding material, defaults to vacuum.
        poltype (str, optional): Polarization type (:ref:`params:Polarizations`).
        lattice (Lattice, optional): Lattice definition. If specified the T-Matrix is
            assumed to be periodically repeated in the defined lattice.
        kpar (list, optional): Phase factor for the periodic T-Matrix.
    """

    interact = _Interact()
    interactlattice = _InteractLattice()

    def _check(self):
        """Fill in default values or raise errors for missing attributes."""
        super()._check()
        shape = np.shape(self)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise AnnotationError(f"invalid shape: '{shape}'")
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
        if self.basis is None:
            self.basis = CWB.default([0], CWB.defaultmmax(shape[0]))
        if self.material is None:
            self.material = Material()

    @property
    def ks(self):
        """Wave numbers (in medium).

        The wave numbers for both polarizations.
        """
        return self.material.ks(self.k0)

    @property
    def krhos(self):
        r"""Radial part of the wave.

        Calculate :math:`\sqrt{k^2 - k_z^2}`, where :math:`k` is the wave number in the
        medium for each illumination

        Returns:
            Sequence[complex]
        """
        return self.material.krhos(self.k0, self.basis.kz, self.basis.pol)

    @classmethod
    def cylinder(cls, kzs, mmax, k0, radii, materials):
        """T-Matrix of a (multi-layered) cylinder.

        Construct the T-matrix of the given order and material for an infinitely
        extended cylinder. The object can also consist of multiple concentric
        cylindrical shells with an arbitrary number of layers. The calculation is always
        done in helicity basis.

        Args:
            kzs (float, array_like): Z component of the cylindrical wave.
            mmax (int): Positive integer for the maximum order of the T-matrix.
            k0 (float): Wave number in vacuum.
            radii (float or array): Radii from inside to outside of the cylinder. For a
                simple cylinder the radius can be given as a single number, for a multi-
                layered cylinder it is a list of increasing radii for all shells.
            material (list[Material]): The material parameters from the inside to the
                outside. The last material in the list specifies the embedding medium.

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
    def cluster(cls, tmats, positions):
        r"""Block-diagonal T-matrix of multiple objects.

        Construct the initial block-diagonal T-matrix for a cluster of objects. The
        T-matrices in the list are placed together into a block-diagonal matrix and the
        complete (local) basis is defined based on the individual T-matrices and their
        bases together with the defined positions. In mathematical terms the matrix

        .. math::

            \begin{pmatrix}
                T_0 & 0 & \dots & 0 \\
                0 & T_1 & \ddots & \vdots \\
                \vdots & \ddots & \ddots & 0 \\
                0 & \dots & 0 & T_{N-1} \\
            \end{pmatrix}

        is created from the list of T-matrices :math:`(T_0, \dots, T_{N-1})`. Only
        T-matrices of the same wave number, embedding material, and polarization type
        can be combined.

        Args:
            tmats (Sequence): List of T-matrices.
            positions (array): The positions of all individual objects in the cluster.

        Returns:
            TMatrix
        """
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
            for m, n in zip(modes, tm.basis.zmp):
                m.extend(list(n))
            pidx += [j] * dim
            tres[i : i + dim, i : i + dim] = tm
            i += dim
        basis = CWB(zip(pidx, *modes), positions)
        return cls(tres, k0=k0, material=mat, basis=basis, poltype=poltype)

    @classmethod
    def from_array(cls, tm, basis, eta=0):
        """1d array of spherical T-matrices."""
        if tm.lattice is None:
            tm = tm.interacted_lattice(basis.lattice, basis.kpar, eta=eta)
        p = tm.expandlattice(basis=basis)
        a = tm.expand.inv(basis=basis)
        return cls(p @ tm @ a, lattice=tm.lattice, kpar=tm.kpar)

    @property
    def xw_ext_avg(self):
        r"""Rotation and polarization averaged extinction cross width.

        The average is calculated as

        .. math::

            \langle \lambda_\mathrm{ext} \rangle
            = -\frac{2 \pi}{n_{k_z}} \sum_{sk_zm} \frac{\Re(T_{sk_zm,sk_zm})}{k_s}

        where :math:`k_s` is the wave number in the embedding medium for the
        polarization :math:`s` and :math:`n_{k_z}` is the number of wave components
        :math:`k_z` included in the T-matrix. The average is taken over all given
        z-components of the wave vector and rotations around the z-axis. It is only
        implemented for global T-matrices.

        Returns:
            float or complex
        """
        if not self.isglobal or not self.material.isreal:
            raise NotImplementedError
        nk = np.unique(self.basis.kz).size
        if not self.material.ischiral:
            res = -2 * np.real(np.trace(self)) / (self.ks[0] * nk)
        else:
            res = 0
            modetype = self.modetype
            del self.modetype
            diag = np.diag(self)
            self.modetype = modetype
            for pol in [0, 1]:
                choice = self.basis.pol == pol
                res += -2 * np.real(diag[choice].sum()) / (self.ks[pol] * nk)
        if res.imag == 0:
            return res.real
        return res

    @property
    def xw_sca_avg(self):
        r"""Rotation and polarization averaged scattering cross width.

        The average is calculated as

        .. math::

            \langle \lambda_\mathrm{sca} \rangle
            = \frac{2 \pi}{n_{k_z}} \sum_{sk_zm} \sum_{s'{k_z}'m'}
            \frac{|T_{sk_zm,s'{k_z}'m'}|^2}{k_s}

        where :math:`k_s` is the wave number in the embedding medium for the
        polarization :math:`s`. and :math:`n_{k_z}` is the number of wave components
        :math:`k_z` included in the T-matrix. The average is taken over all given
        z-components of the wave vector and rotations around the z-axis. It is only
        implemented for global T-matrices.

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
        nk = np.unique(self.basis.kz).size
        res = 2 * np.sum((re * re + im * im) / (ks * nk))
        return res.real

    @property
    def isglobal(self):
        """Test if a T-matrix is global.

        A T-matrix is considered global, when its basis refers to only a single point
        and it is not placed periodically in a lattice.
        """
        return self.basis.isglobal and self.lattice is None and self.kpar is None

    def xw(self, illu, flux=0.5):
        r"""Scattering and extinction cross width.

        Possible for all T-matrices (global and local) in non-absorbing embedding. The
        values are calculated by

        .. math::

            \lambda_\mathrm{sca}
            = \frac{1}{2 I}
            a_{sk_zm}^\ast T_{s'{k_z}'m',sk_zm}^\ast k_{s'}^{-2}
            C_{s'l'm',s''l''m''}^{(1)}
            T_{s''l''m'',s'''l'''m'''} a_{s'''l'''m'''} \\
            \sigma_\mathrm{ext}
            = \frac{1}{2 I}
            a_{slm}^\ast k_s^{-2} T_{slm,s'l'm'} a_{s'l'm'}

        where :math:`a_{slm}` are the expansion coefficients of the illumination,
        :math:`T` is the T-matrix, :math:`C^{(1)}` is the (regular) translation
        matrix and :math:`k_s` are the wave numbers in the medium. All repeated indices
        are summed over. The incoming flux is :math:`I`.

        Args:
            illu (complex, array): Illumination coefficients
            flux (optional): Ingoing flux corresponding to the illumination. Used for
                the result's normalization. The flux is given in units of
                :math:`\frac{\text{V}^2}{{l^2}} \frac{1}{Z_0 Z}` where :math:`l` is the
                unit of length used in the wave number (and positions). A plane wave
                has the flux `0.5` in this normalization, which is used as default.

        Returns:
            tuple[float]
        """
        if not self.material.isreal:
            raise NotImplementedError
        illu = PhysicsArray(illu)
        illu_basis = illu.basis
        illu_basis = illu_basis[-2] if isinstance(illu_basis, tuple) else illu_basis
        if not isinstance(illu_basis, CWB):
            illu = illu.expand(self.basis) @ illu
        p = self @ illu
        m = self.expand() / self.ks[self.basis.pol]
        del illu.modetype
        return (
            2 * np.real(p.conjugate().T @ (m @ p)) / flux,
            -2 * np.real(illu.conjugate().T @ (p / self.ks[self.basis.pol])) / flux,
        )

    # def rotated(self, phi, **kwargs):
    #     """Rotated T-Matrix around the z-axis.

    #     Rotation is done in-place. If you need the original T-Matrix make a deepcopy
    #     first. Rotations can only be applied to global T-Matrices.

    #     Args:
    #         phi (float): Rotation angle
    #         modes (array): While rotating also take only specific output modes into
    #         account

    #     Returns:
    #         T-MatrixC
    #     """
    #     r = self.rotate(phi, **kwargs)
    #     return TMatrixC(r @ self @ r.conjugate().T)

    # def translated(self, r, **kwargs):
    #     """Translated T-Matrix.

    #     Translation is done in-place. If you need the original T-Matrix make a copy
    #     first. Translations can only be applied to global T-Matrices.

    #     Args:
    #         r (float array): Translation vector
    #         modes (array): While translating also take only specific output modes into
    #         account

    #     Returns:
    #         TMatrixC
    #     """
    #     return TMatrixC(
    #         self.translate(r, **kwargs) @ self @ self.inv.translate(r, **kwargs)
    #     )

    # def couple(self):
    #     """Calculate the coupling term of a blockdiagonal T-matrix."""
    #     return np.eye(self.shape[0]) - self @ self.expand(modetype="regular")

    # def interacted(self):
    #     return TMatrixC(np.linalg.solve(self.couple(), self))

    def globalmat(self, basis=None):
        """Global T-matrix.

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

    # def couple_lattice(self, lattice, kpar, *, eta=0):
    #     """Calculate the coupling term of a blockdiagonal T-matrix."""
    #     return np.eye(self.shape[0]) - self @ self.expandlattice(lattice, kpar, eta=eta)

    # def interacted_lattice(self, lattice, kpar, eta=0):
    #     obj = TMatrix(
    #         np.linalg.solve(self.couple_lattice(lattice, kpar, eta=eta), self)
    #     )
    #     lattice, _ = obj.lattice
    #     obj.lattice = lattice
    #     kpar, _ = obj.kpar
    #     obj.kpar = kpar
    #     return obj

    def valid_points(self, grid, radii):
        grid = np.asarray(grid)
        if grid.shape[-1] not in (2, 3):
            raise ValueError("invalid grid")
        if len(radii) != len(self.basis.positions):
            raise ValueError("invalid length of 'radii'")
        res = np.ones(grid.shape[:-1], bool)
        for r, p in zip(radii, self.basis.positions):
            res &= np.sum(np.power(grid[..., :2] - p[:2], 2), axis=-1) > r * r
        return res


def _plane_wave_partial(
    kpar,
    pol,
    *,
    k0=None,
    basis=None,
    material=None,
    modetype=None,
    poltype=None,
):
    if basis is None:
        basis = PWBC.default([kpar])
    if pol == 0 or pol == -1:
        pol = [1, 0]
    elif pol == 1:
        pol = [0, 1]
    elif len(pol) == 3:
        modetype = "up" if modetype is None else modetype
        if modetype not in ("up", "down"):
            raise ValueError(f"invalid 'modetype': {modetype}")
        kzs = Material(material).kzs(k0, *kpar, [0, 1])
        poltype = config.POLTYPE if poltype is None else poltype
        if poltype == "parity":
            pol = [
                -sc.vpw_M(*kpar, kzs[0], 0, 0, 0) @ pol,
                sc.vpw_N(*kpar, kzs[1], 0, 0, 0) @ pol,
            ]
        elif poltype == "helicity":
            pol = sc.vpw_A(*kpar, kzs[::-1], 0, 0, 0, [1, 0]) @ pol
        else:
            raise ValueError(f"invalid 'poltype': {poltype}")
    res = [pol[x[2]] * (np.abs(np.array(kpar) - x[:2]) < 1e-14).all() for x in basis]
    return PhysicsArray(
        res,
        basis=basis,
        k0=k0,
        material=material,
        modetype=modetype,
        poltype=poltype,
    )


def _plane_wave(
    kvec,
    pol,
    *,
    k0=None,
    basis=None,
    material=None,
    modetype=None,
    poltype=None,
):
    if basis is None:
        basis = PWBUV.default([kvec])
    norm = np.sqrt(np.sum(np.power(kvec, 2)))
    qvec = kvec / norm
    if pol == 0 or pol == -1:
        pol = [1, 0]
    elif pol == 1:
        pol = [0, 1]
    elif len(pol) == 3:
        if None not in (k0, material):
            kvec = Material(material).ks(k0) * qvec[:, None]
        else:
            kvec = qvec
        poltype = config.POLTYPE if poltype is None else poltype
        if poltype == "parity":
            pol = [
                -sc.vpw_M(*kvec[:, 0], 0, 0, 0) @ pol,
                sc.vpw_N(*kvec[:, 1], 0, 0, 0) @ pol,
            ]
        elif poltype == "helicity":
            pol = sc.vpw_A(*kvec, 0, 0, 0, [1, 0]) @ pol
        else:
            raise ValueError(f"invalid 'poltype': {poltype}")
    res = [pol[x[3]] * (np.abs(qvec - x[:3]) < 1e-14).all() for x in basis]
    return PhysicsArray(
        res,
        basis=basis,
        k0=k0,
        material=material,
        modetype=modetype,
        poltype=poltype,
    )


def plane_wave(
    kvec,
    pol,
    *,
    k0=None,
    basis=None,
    material=None,
    modetype=None,
    poltype=None,
):
    """Array describing a plane wave.

    Args:
        kvec (Sequence): Wave vector.
        pol (int or Sequence): Polarization index (see
            :ref:`params:Polarizations`) to have a unit amplitude wave of the
            corresponding wave, two values to specify the amplitude for each
            polarization, or three values in a sequence to specify the Cartesian
            electric field components. In the latter case, if the electric field has
            longitudinal components they are neglected.
        basis (PlaneWaveBasis, optional): Basis definition.
        k0 (float, optional): Wave number in vacuum.
        material (Material, optional): Material definition.
        modetype (str, optional): Mode type (see :ref:`params:Mode types`).
        poltype (str, optional): Polarization type (see
            :ref:`polarization:Polarizations`).
    """
    if len(kvec) == 2:
        return _plane_wave_partial(
            kvec,
            pol,
            k0=k0,
            basis=basis,
            material=material,
            modetype=modetype,
            poltype=poltype,
        )
    elif len(kvec) == 3:
        return _plane_wave(
            kvec,
            pol,
            k0=k0,
            basis=basis,
            material=material,
            modetype=modetype,
            poltype=poltype,
        )
    raise ValueError(f"invalid length of 'kvec': {len(kvec)}")


def plane_wave_angle(theta, phi, pol, **kwargs):
    qvec = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    return plane_wave(qvec, pol, **kwargs)


def spherical_wave(
    l,  # noqa: E741
    m,
    pol,
    *,
    k0=None,
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
    res[basis.index((0, l, m, pol))] = 1
    return PhysicsArray(
        res, basis=basis, k0=k0, material=material, modetype=modetype, poltype=poltype
    )


def cylindrical_wave(
    kz,
    m,
    pol,
    *,
    k0=None,
    basis=None,
    material=None,
    modetype=None,
    poltype=None,
):
    if basis is None:
        basis = CWB.default([kz], abs(m))
    if not basis.isglobal:
        raise ValueError("basis must be global")
    res = [0] * len(basis)
    res[basis.index((0, kz, m, pol))] = 1
    return PhysicsArray(
        res, basis=basis, k0=k0, material=material, modetype=modetype, poltype=poltype
    )
