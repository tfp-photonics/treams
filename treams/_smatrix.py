import copy

import numpy as np

from treams import config, util
from treams._core import PhysicsArray
from treams._core import PlaneWaveBasisByComp as PWBC
from treams._material import Material
from treams._operators import translate
from treams._tmatrix import TMatrixC
from treams.coeffs import fresnel


class SMatrix(PhysicsArray):
    """S-matrix for a plane wave."""

    def _check(self):
        super()._check()
        shape = np.shape(self)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise util.AnnotationError(f"invalid shape: '{shape}'")
        if not isinstance(self.k0, (int, float, np.floating, np.integer)):
            raise util.AnnotationError("invalid k0")
        if self.poltype is None:
            self.poltype = config.POLTYPE
        if self.poltype not in ("parity", "helicity"):
            raise util.AnnotationError("invalid poltype")
        material = (None, None) if self.material is None else self.material
        if isinstance(material, tuple):
            self.material = tuple(Material() if m is None else m for m in material)
        if self.basis is None:
            raise util.AnnotationError("basis not set")
        if not isinstance(self.basis, PWBC):
            self.basis = self.basis.bycomp(self.k0, self.material)


class OperatorAttributeSMatrices:
    def __init__(self, name):
        self._name = name
        self._obj = self._objtype = None

    def __get__(self, obj, objtype=None):
        self._obj = obj
        self._objtype = type(obj) if objtype is None else objtype
        return self

    def __call__(self, *args, **kwargs):
        res = [
            [getattr(ii, self._name)(*args, **kwargs) for ii in i] for i in self._obj
        ]
        return self._objtype(res)

    def apply_left(self, *args, **kwargs):
        res = [
            [getattr(ii, self._name).apply_left(*args, **kwargs) for ii in i]
            for i in self._obj
        ]
        return self._objtype(res)

    def apply_right(self, *args, **kwargs):
        res = [
            [getattr(ii, self._name).apply_right(*args, **kwargs) for ii in i]
            for i in self._obj
        ]
        return self._objtype(res)


class SMatrices:
    r"""Collection of four S-matrices with a plane wave basis.

    The S-matrix describes the scattering of incoming into outgoing modes using a plane
    wave basis, with functions :func:`treams.special.vsw_A`,
    :func:`treams.special.vsw_M`, and :func:`treams.special.vsw_N`. The primary
    direction of propagation is parallel or anti-parallel to the z-axis. The scattering
    object itself is infinitely extended in the x- and y-directions. The S-matrix is
    divided into four submatrices :math:`S_{\uparrow \uparrow}`,
    :math:`S_{\uparrow \downarrow}`, :math:`S_{\downarrow \uparrow}`, and
    :math:`S_{\downarrow \downarrow}`:

    .. math::

        S = \begin{pmatrix}
            S_{\uparrow \uparrow} & S_{\uparrow \downarrow} \\
            S_{\downarrow \uparrow} & S_{\downarrow \downarrow}
        \end{pmatrix}\,.

    These matrices describe the transmission of plane waves propagating into positive
    z-direction, reflection of plane waves into the positive z-direction, reflection of
    plane waves into negative z-direction, and the transmission of plane waves
    propagating in negative z-direction, respectively. Each of those for matrices
    contain different diffraction orders and different polarizations. The polarizations
    can be in either helicity of parity basis.

    The wave number :attr:`k0` and, if not vacuum, the material :attr:`material` are
    also required.

    Args:
        smats (SMatrix): S-matrices.
        k0 (float): Wave number in vacuum.
        basis (PlaneWaveBasisByComp): The basis for the S-matrices.
        material (tuple, Material, optional): Material definition, if a tuple of length
            two is specified, this refers to the materials above and below the S-matrix.
        poltype (str, optional): Polarization type (:ref:`params:Polarizations`).
    """
    permute = OperatorAttributeSMatrices("permute")
    translate = OperatorAttributeSMatrices("translate")
    rotate = OperatorAttributeSMatrices("rotate")
    changepoltype = OperatorAttributeSMatrices("changepoltype")

    def __init__(self, smats, **kwargs):
        """Initialization."""
        if len(smats) != 2 or len(smats[0]) != 2 or len(smats[1]) != 2:
            raise ValueError("invalid shape of S-matrices")
        if "material" in kwargs:
            material = kwargs["material"]
            del kwargs["material"]
        elif hasattr(smats[0][0], "material"):
            material = smats[0][0].material
        elif hasattr(smats[1][1], "material"):
            material = smats[1][1].material
            if isinstance(material, tuple):
                material = material[::-1]
        elif hasattr(smats[0][1], "material"):
            material = smats[0][1].material
        elif hasattr(smats[1][0], "material"):
            material = smats[1][0].material
        else:
            material = Material()
        if isinstance(material, tuple):
            ma, mb = material
        else:
            ma = mb = Material(material)
        material = [[(ma, mb), (ma, ma)], [(mb, mb), (mb, ma)]]
        modetype = [[("up", "up"), ("up", "down")], [("down", "up"), ("down", "down")]]
        self._sms = [
            [
                SMatrix(s, material=m, modetype=t, **kwargs)
                for s, m, t in zip(ar, mr, tr)
            ]
            for ar, mr, tr in zip(smats, material, modetype)
        ]
        self.material = self[0, 0].material
        self.basis = self[0, 0].basis
        self.k0 = self[0, 0].k0
        self.poltype = self[0, 0].poltype
        for i in (self[1, 0], self[0, 1], self[1, 1]):
            i.k0 = self.k0
            i.basis = self.basis
            i.poltype = self.poltype

    def __eq__(self, other):
        return all(
            (np.abs(self[i, j] - other[i][j]) < 1e-14).all()
            for i in range(2)
            for j in range(2)
        )

    def __getitem__(self, key):
        keys = {0: 0, "up": 0, 1: 1, "down": 1}
        if key in keys:
            return self._sms[keys[key]]
        if not isinstance(key, tuple) or len(key) not in (1, 2):
            raise KeyError(f"invalid key: '{key}'")
        if len(key) == 1:
            return self[key[0]]
        key = tuple(keys[k] for k in key)
        return self._sms[key[0]][key[1]]

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(self._sms)

    @classmethod
    def interface(cls, basis, k0, materials):
        """Planar interface between two media.

        Args:
            basis (PlaneWaveBasisByComp): Basis definitions.
            k0 (float): Wave number in vacuum
            materials (Sequence[Material]): Material definitions.

        Returns:
            SMatrix
        """
        materials = tuple(map(Material, materials))
        ks = np.array([m.ks(k0) for m in materials])
        choice = basis.pol == 0
        kxs = basis.kx[choice], basis.kx[~choice]
        kys = basis.ky[choice], basis.ky[~choice]
        qs = np.zeros((2, 2, len(basis), len(basis)), complex)
        if all(kxs[0] == kxs[1]) and all(kys[0] == kys[1]):
            kzs = np.stack(
                [
                    m.kzs(k0, kxs[0][:, None], kys[0][:, None], np.array([[0, 1]]))
                    for m in materials
                ],
                -2,
            )
            vals = fresnel(ks, kzs, [m.impedance for m in materials])
            qs[:, :, choice, choice] = np.moveaxis(vals[:, :, :, 0, 0], 0, -1)
            qs[:, :, choice, ~choice] = np.moveaxis(vals[:, :, :, 0, 1], 0, -1)
            qs[:, :, ~choice, choice] = np.moveaxis(vals[:, :, :, 1, 0], 0, -1)
            qs[:, :, ~choice, ~choice] = np.moveaxis(vals[:, :, :, 1, 1], 0, -1)
        else:
            for i, (kx, ky, pol) in enumerate(basis):
                for ii, (kx2, ky2, pol2) in enumerate(basis):
                    if kx != kx2 or ky != ky2:
                        continue
                    kzs = np.array([m.kzs(k0, kx, ky) for m in materials])
                    vals = fresnel(ks, kzs, [m.impedance for m in materials])
                    qs[:, :, ii, i] = vals[:, :, pol2, pol]
        return cls(qs, k0=k0, basis=basis, material=materials[::-1], poltype="helicity")

    @classmethod
    def slab(cls, thickness, basis, k0, materials):
        """Slab of material.

        A slab of material, defined by a thickness and three materials. Consecutive
        slabs of material can be defined by `n` thicknesses and and `n + 2` material
        parameters.

        Args:
            k0 (float): Wave number in vacuum.
            basis (PlaneWaveBasisByComp): Basis definition.
            tickness (Sequence[Float]): Thickness of the slab or the thicknesses of all
                slabs in order from negative to positive z.
            materials (Sequenze[Material]): Material definitions from negative to
                positive z.

        Returns:
            SMatrix
        """
        try:
            iter(thickness)
        except TypeError:
            thickness = [thickness]
        res = cls.interface(basis, k0, materials[:2])
        for d, ma, mb in zip(thickness, materials[1:-1], materials[2:]):
            if np.ndim(d) == 0:
                d = [0, 0, d]
            x = cls.propagation(d, basis, k0, ma)
            res = res.add(x)
            res = res.add(cls.interface(basis, k0, (ma, mb)))
        return res

    @classmethod
    def stack(cls, items):
        """Stack of S-matrices.

        Electromagnetically couple multiple S-matrices in the order given. Before
        coupling it can be checked for matching materials and modes.

        Args:
            items (Sequence[SMatrix]): An array of S-matrices in their intended order
                from negative to positive z.

        Returns:
            SMatrix
        """
        acc = items[0]
        for item in items[1:]:
            acc = acc.add(item)
        return acc

    @classmethod
    def propagation(cls, r, basis, k0, material=Material()):
        """S-matrix for the propagation along a distance.

        This S-matrix translates the reference origin along `r`.

        Args:
            r (float, (3,)-array): Translation vector.
            k0 (float): Wave number in vacuum.
            basis (PlaneWaveBasis): Basis definition.
            material (Material, optional): Material definition.

        Returns:
            SMatrix
        """
        sup = translate(r, basis=basis, k0=k0, material=material, modetype="up")
        sdown = translate(
            np.negative(r), basis=basis, k0=k0, material=material, modetype="down"
        )
        zero = np.zeros_like(sup)
        material = Material(material)
        return cls([[sup, zero], [zero, sdown]], basis=basis, k0=k0, material=material)

    @classmethod
    def from_array(cls, tm, basis, *, eta=0):
        """S-matrix from an array of (cylindrical) T-matrices.

        Create a S-matrix for a two-dimensional array of objects described by the
        T-Matrix or an one-dimensional array of objects described by a cylindrical
        T-matrix.

        Args:
            tm (TMatrix or TMatrixC): (Cylindrical) T-matrix to place in the array.
            basis (PlaneWaveBasisByComp): Basis definition.
            eta (float or complex, optional): Splitting parameter in the lattice sum.

        Returns:
            SMatrix
        """
        if isinstance(tm, TMatrixC):
            basis = basis.permute(-1)
        pu, pd = (
            tm.expandlattice(basis=basis, modetype=i, eta=eta) for i in ("up", "down")
        )
        au, ad = (tm.expand.eval_inv(basis, i) for i in ("up", "down"))
        eye = np.eye(len(basis))
        res = cls([[eye + pu @ au, pu @ ad], [pd @ au, eye + pd @ ad]])
        if isinstance(tm, TMatrixC):
            res = res.permute()
        return res

    def add(self, sm):
        """Couple another S-matrix on top of the current one.

        See also :func:`treams.SMatrix.stack` for a function that does not change the
        current S-matrix but creates a new one.

        Args:
            sm (SMatrix): S-matrix to add.

        Returns:
            SMatrix
        """
        dim = len(self.basis)
        snew = [[None, None], [None, None]]
        s_tmp = np.linalg.solve(np.eye(dim) - self[0, 1] @ sm[1][0], self[0, 0])
        snew[0][0] = sm[0][0] @ s_tmp
        snew[1][0] = self[1, 0] + self[1, 1] @ sm[1][0] @ s_tmp
        s_tmp = np.linalg.solve(np.eye(dim) - sm[1][0] @ self[0, 1], sm[1][1])
        snew[1][1] = self[1, 1] @ s_tmp
        snew[0][1] = sm[0][1] + sm[0][0] @ self[0, 1] @ s_tmp
        return SMatrices(snew)

    def double(self, n=1):
        """Double the S-matrix.

        By default this function doubles the S-matrix but it can also create a
        :math:`2^n`-fold repetition of itself:

        Args:
            n (int, optional): Number of times to double itself. Defaults to 1.

        Returns:
            SMatrix
        """
        res = self
        for _ in range(n):
            res = res.add(self)
        return res

    def illuminate(self, illu, illu2=None, /, *, smat=None):
        """Field coefficients above and below the S-matrix.

        Given an illumination defined by the coefficients of each incoming mode
        calculate the coefficients for the outgoing field above and below the S-matrix.
        If a second SMatrix is given, the field expansions in between are also
        calculated.

        Args:
            illu (array-like): Illumination, if `modetype` is specified, the direction
                will be chosen accordingly.
            illu2 (array-like, optional): Second illumination. If used, the first
                argument is taken to be coming from below and this one to be coming from
                above.
            smat (SMatrix, optional): Second S-matrix for the calculation of the
                field expansion between two S-matrices.

        Returns:
            tuple
        """
        modetype = getattr(illu, "modetype", "up")
        if isinstance(modetype, tuple):
            modetype = modetype[max(-2, -len(tuple))]
        illu2 = np.zeros(np.shape(illu)[-2:]) if illu2 is None else illu2
        if modetype == "down":
            illu, illu2 = illu2, illu
        if smat is None:
            field_up = self[0, 0] @ illu + self[0, 1] @ illu2
            field_down = self[1, 0] @ illu + self[1, 1] @ illu2
            return field_up, field_down
        stmp = np.eye(len(self.basis)) - self[0, 1] @ smat[1, 0]
        field_in_up = np.linalg.solve(
            stmp, self[0, 0] @ illu + self[0, 1] @ smat[1, 1] @ illu2
        )
        field_in_down = smat[1, 0] @ field_in_up + smat[1, 1] @ illu2
        field_up = smat[0, 1] @ illu2 + smat[0, 0] @ field_in_up
        field_down = self[1, 0] @ illu + self[1, 1] @ field_in_down
        return field_up, field_down, field_in_up, field_in_down

    def tr(self, illu):
        """Transmittance and reflectance.

        Calculate the transmittance and reflectance for one S-matrix with the given
        illumination and direction.

        Args:
            illu (complex, array_like): Expansion coefficients for the incoming light

        Returns:
            tuple
        """
        modetype = getattr(illu, "modetype", "up")
        if isinstance(modetype, tuple):
            modetype = modetype[max(-2, -len(tuple))]
        trans, refl = self.illuminate(illu)
        material = self.material
        if not isinstance(material, tuple):
            material = material, material
        paz = [poynting_avg_z(self.basis, self.k0, m, self.poltype) for m in material]
        if modetype == "down":
            trans, refl = refl, trans
            paz.reverse()
        illu = np.asarray(illu)
        s_t = np.real(trans.conjugate().T @ paz[0][0] @ trans)
        s_r = np.real(refl.conjugate().T @ paz[1][0] @ refl)
        s_i = np.real(np.conjugate(illu).T @ paz[1][0] @ illu)
        s_ir = np.real(
            refl.conjugate().T @ paz[1][1] @ illu
            - np.conjugate(illu).T @ paz[1][1] @ refl
        )
        return s_t / (s_i + s_ir), s_r / (s_i + s_ir)

    def cd(self, illu):
        """Transmission and absorption circular dichroism.

        Calculate the transmission and absorption CD for one S-matrix with the given
        illumination and direction.

        Args:
            illu (complex, PhysicsArray): Expansion coefficients for the incoming light

        Returns:
            tuple
        """
        minus, plus = self.basis.pol == 0, self.basis.pol == 1
        if self.poltype == "helicity":
            illuopposite = np.zeros_like(illu)
            illuopposite[minus] = illu[plus]
            illuopposite[plus] = illu[minus]
        else:
            illuopposite = copy.deepcopy(illu)
            illuopposite[minus] *= -1
        tm, rm = self.tr(illu)
        tp, rp = self.tr(illuopposite)
        return (tp - tm) / (tp + tm), (tp + rp - tm - rm) / (tp + rp + tm + rm)

    def periodic(self):
        r"""Periodic repetition of the S-matrix.

        Transform the S-matrix to an infinite periodic arrangement of itself defined
        by

        .. math::

            \begin{pmatrix}
                S_{\uparrow \uparrow} & S_{\uparrow \downarrow} \\
                -S_{\downarrow \downarrow}^{-1}
                S_{\downarrow \uparrow} S_{\uparrow \uparrow} &
                S_{\downarrow \downarrow}^{-1} (\mathbb{1} - S_{\downarrow \uparrow}
                S_{\uparrow \downarrow})
            \end{pmatrix}\,.

        Returns:
            complex, array_like

        """
        dim = len(self.basis)
        res = np.empty((2 * dim, 2 * dim), dtype=complex)
        res[0:dim, 0:dim] = self[0, 0]
        res[0:dim, dim:] = self[0, 1]
        res[dim:, 0:dim] = -np.linalg.solve(self[1, 1], self[1, 0] @ self[0, 0])
        res[dim:, dim:] = np.linalg.solve(
            self[1, 1], np.eye(dim) - self[1, 0] @ self[0, 1]
        )
        return res

    def bands_kz(self, az):
        r"""Band structure calculation.

        Calculate the band structure for the given S-matrix, assuming it is periodically
        repeated along the z-axis. The function returns the z-components of the wave
        vector :math:`k_z` and the corresponding eigenvectors :math:`v` of

        .. math::

            \begin{pmatrix}
                S_{\uparrow \uparrow} & S_{\uparrow \downarrow} \\
                -S_{\downarrow \downarrow}^{-1} S_{\downarrow \uparrow}
                    S_{\uparrow \uparrow} &
                S_{\downarrow \downarrow}^{-1} (\mathbb{1} - S_{\downarrow \uparrow}
                    S_{\uparrow \downarrow})
            \end{pmatrix}
            \boldsymbol v
            =
            \mathrm{e}^{\mathrm{i}k_z a_z}
            \boldsymbol v\,.

        Args:
            az (float): Lattice pitch along the z direction

        Returns:
            tuple

        """
        w, v = np.linalg.eig(self.periodic())
        return -1j * np.log(w) / az, v

    def __repr__(self):
        return f"""{type(self).__name__}(...,
    basis={self.basis},
    k0={self.k0},
    material={self.material},
    poltype='{self.poltype}',
)"""


def chirality_density(basis, k0, material=Material(), poltype=None, z=(0, 0)):
    poltype = config.POLTYPE if poltype is None else poltype
    material = Material(material)
    kx, ky, kz = basis.kvecs(k0, material)
    k = material.ks(k0)[basis.pol]
    re, im = np.real(kz), np.imag(kz)
    prefuu = 1 + np.real((k * k - kz * 2j * im) / (k * k.conjugate()))
    prefdu = 1 + np.real((k * k - kz * 2 * re) / (k * k.conjugate()))
    if np.abs(z[1] - z[0]) > 1e-16:
        prefuu *= np.sinh(re * (z[1] - z[0])) / (re * (z[1] - z[0]))
        prefdd = np.exp(im * (z[1] + z[0])) * prefuu
        prefuu *= np.exp(-im * (z[1] + z[0]))
        prefdu *= (
            np.sin(re * (z[1] - z[0]))
            * np.cos(re * (z[1] + z[0]))
            / (re * (z[1] - z[0]))
        )
    else:
        prefdd = prefuu
    if poltype == "helicity":
        return (
            np.diag(prefuu * (2 * basis.pol - 1)),
            np.diag(prefdd * (2 * basis.pol - 1)),
            np.diag(2 * prefdu * (2 * basis.pol - 1)),
        )
    if poltype == "parity":
        where = (
            (kx[:, None] == kx)
            & (ky[:, None] == ky)
            & (basis.pol[:, None] != basis.pol)
        )
        return where * prefuu, where * prefdd, where * 2 * prefdu
    raise ValueError(f"invalid poltype: '{poltype}'")


def poynting_avg_z(basis, k0, material=Material(), poltype=None):
    r"""Time-averaged z-component of the Poynting vector.

    Calculate the time-averaged Poynting vector's z-component

    .. math::

        \langle S_z \rangle
        = \frac{1}{2}
        \Re (\boldsymbol E \times \boldsymbol H^\ast) \boldsymbol{\hat{z}}

    on one side of the S-matrix with the given coefficients.

    Returns:
        tuple
    """
    poltype = config.POLTYPE if poltype is None else poltype
    material = Material(material)
    kx, ky, kzs = basis.kvecs(k0, material)
    gamma = kzs / (material.ks(k0)[basis.pol] * material.impedance)
    selection = (kx[:, None] == kx) & (ky[:, None] == ky)
    if poltype == "parity":
        pol = basis.pol
        selection = selection & (pol[:, None] == pol)
        a = selection * ((1 - pol) * gamma.conjugate() + pol * gamma) * 0.25
        b = selection * ((1 - pol) * gamma.conjugate() - pol * gamma) * 0.25
        return a, b
    if poltype == "helicity":
        pol = 2 * basis.pol - 1
        a = selection * (pol[:, None] * pol * gamma[:, None].conjugate() + gamma) * 0.25
        b = selection * (pol[:, None] * pol * gamma[:, None].conjugate() - gamma) * 0.25
        return a, b
    raise ValueError(f"invalid poltype: '{poltype}'")
