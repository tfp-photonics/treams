import copy
import itertools

import numpy as np

from treams import config, util
from treams._core import PhysicsArray
from treams._core import PlaneWaveBasisPartial as PWBP
from treams._material import Material
from treams._operators import translate
from treams.coeffs import fresnel


class _SMatrix(PhysicsArray):
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
        if self.basis is None:
            raise util.AnnotationError("basis not set")
        elif not isinstance(self.basis, PWBP):
            self.basis = self.basis.partial()
        material = (None, None) if self.material is None else self.material
        if isinstance(material, tuple):
            self.material = tuple(Material() if m is None else m for m in material)


class SMatrix:
    r"""S-matrix (plane wave basis).

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
    z-direction, reflection of plane waves into the positive z-direction, reflection
    of plane waves into negative z-direction, and the transmission of plane waves
    propagating in negative z-direction, respectively. Each of those for matrices
    contain different diffraction orders and different polarizations. The polarizations
    can be in either helicity of parity basis.

    The embedding medium is described by permittivity, permeability, and the chirality
    parameter. The two sides of the S-matrix can have different materials.

    Args:
        smats (float or complex, array): S-matrices
        k0 (float): Wave number in vacuum
        modes (iterable, optional): The modes corresponding to the S-matrices rows and
            columns. It must contain three arrays of equal length for the wave vectors'
            x- and y-component, and the polarization.
        epsilon (float or complex, optional): Relative permittivity of the medium
        mu (float or complex, optional): Relative permeability of the medium
        kappa (float or complex, optional): Chirality parameter of the medium
        helicity (bool, optional): Helicity or parity modes
        kz (float or complex, (N, 2)-array, optional): The z-component of the wave
            vector on both sides of the S-matrix

    Attributes:
        q (complex, (2, 2, N, N)-array): S-matrices for the layer. The first dimension
            corresponds to the output side, the second to the input side, the third one
            to the outgoing modes and the last one to the incoming modes.
        k0 (float): Wave number in vacuum
        epsilon (float or complex, optional): Relative permittivity of the embedding
            medium
        mu (complex): Relative permeability of the embedding medium
        kappa (complex): Chirality parameter of the embedding medium
        helicity (bool): Helicity or parity modes
        kx (float, (N,)-array): X-component of the wave vector for each column/row of
            the S-matrix
        ky (float, (N,)-array): Y-component of the wave vector for each column/row of
            the S-matrix
        kz (float, (2, N)-array): Z-component of the wave vector for each column/row of
            the S-matrix and for both sides
        pol (int, (N,)-array): Polarization of the mode for each column/row of the
            S-matrix
        ks (float or complex (2, 2)-array): Wave numbers in the medium for both sides
            and both polarizations
    """

    def __init__(self, smats, **kwargs):
        if "material" in kwargs:
            material = kwargs["material"]
            if isinstance(material, Material):
                ma, mb = (material, material)
            else:
                ma, mb = material
            del kwargs["material"]
            materials = [[(ma, mb), (ma, ma)], [(mb, mb), (mb, ma)]]
        else:
            materials = [[None, None], [None, None]]
        if len(smats) != 2 or len(smats[0]) != 2 or len(smats[1]) != 2:
            raise ValueError("invalid S-matrices")
        self._sms = [
            [_SMatrix(s, material=m, **kwargs) for s, m in zip(row, mrow)]
            for row, mrow in zip(smats, materials)
        ]
        self.material = self[0, 0].material
        if isinstance(self.material, Material):
            self.material = (self.material, self.material)
        self[0, 0].modetype = "up"
        self[0, 1].modetype = ("up", "down")
        self[0, 1].material = self.material[0]
        self[1, 0].modetype = ("down", "up")
        self[1, 0].material = self.material[1]
        self[1, 1].modetype = "down"
        self[1, 1].material = (self.material[1], self.material[0])
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
            raise KeyError("invalid key")
        if len(key) == 1:
            return self[key[0]]
        key = tuple(keys[k] for k in key)
        return self._sms[key[0]][key[1]]

    @classmethod
    def interface(cls, k0, basis, materials):
        """Planar interface between two media.

        Args:
            k0 (float): Wave number in vacuum
            kpars (float, (N, 2)-array): Tangential components of the wave vector
            epsilon (float or complex): Relative permittivity on both sides of the
                interface
            mu (float or complex, optional): Relative permittivity on both sides of the
                interface
            kappa (float or complex, optional): Relative permittivity on both sides of
                the interface

        Returns:
            SMatrix
        """
        materials = tuple(map(Material, materials))
        ks = k0 * np.array((materials[0].nmp, materials[1].nmp))
        choice = basis.pol == 0
        kxs = basis.kx[choice], basis.kx[~choice]
        kys = basis.ky[choice], basis.ky[~choice]
        qs = np.zeros((2, 2, len(basis), len(basis)), complex)
        if all(kxs[0] == kxs[1]) and all(kys[0] == kys[1]):
            kzs = np.stack(
                [
                    m.kzs(k0, kxs[0][:, None], kys[0][:, None], [[0, 1]])
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
    def slab(cls, k0, basis, thickness, materials):
        """Slab of material.

        A slab of material, defined by a thickness and three materials. Consecutive
        slabs of material can be defined by `n` thicknesses and and `n + 2` material
        parameters.

        Args:
            k0 (float): Wave number in vacuum
            kpars (float, (N, 2)-array): Tangential components of the wave vector
            epsilon (float or complex): Relative permittivity on both sides of the
                interface
            mu (float or complex, optional): Relative permittivity on both sides of the
                interface
            kappa (float or complex, optional): Relative permittivity on both sides of
                the interface

        Returns:
            SMatrix
        """
        try:
            iter(thickness)
        except TypeError:
            thickness = [thickness]
        res = cls.interface(k0, basis, materials[:2])
        for d, ma, mb in zip(thickness, materials[1:-1], materials[2:]):
            if np.ndim(d) == 0:
                d = [0, 0, d]
            res = res.add(cls.propagation(d, k0, basis, ma))
            res = res.add(cls.interface(k0, basis, (ma, mb)))
        return res

    @classmethod
    def stack(cls, items):
        """Stack of S-matrices.

        Electromagnetically couple multiple S-matrices in the order given. Before
        coupling it can be checked for matching materials and modes.

        Args:
            items (SMatrix, array-like): An array of S-matrices in their intended order
            check_materials (bool, optional): Check for matching material parameters
                at each S-matrix
            check_materials (bool, optional): Check for matching modes at each S-matrix

        Returns:
            SMatrix
        """
        acc = items[0]
        for item in items[1:]:
            acc = acc.add(item)
        return acc

    @classmethod
    def propagation(cls, r, k0, basis, material):
        """S-matrix for the propagation along a distance.

        This S-matrix translates the reference origin along `r`.

        Args:
            r (float, (3,)-array): Translation vector
            k0 (float): Wave number in vacuum
            kpars (float, (N, 2)-array): Tangential components of the wave vector
            epsilon (float or complex, optional): Relative permittivity of the medium
            mu (float or complex, optional): Relative permeability of the medium
            kappa (float or complex, optional): Chirality parameter of the medium

        Returns:
            SMatrix
        """
        sup = translate(r, k0=k0, basis=basis, material=material, modetype="up")
        sdown = translate(r, k0=k0, basis=basis, material=material, modetype="down")
        zero = np.zeros_like(sup)
        material = Material(material)
        return cls([[sup, zero], [zero, sdown]], k0=k0, basis=basis, material=material)

    @classmethod
    def from_array(cls, tm, basis, *, eta=0):
        """S-matrix from an array of (cylindrical) T-matrices.

        Create a S-matrix for a two-dimensional array of objects described by the
        T-Matrix or an one-dimensional array of objects described by a cylindrical
        T-matrix.

        Args:
            tmat (TMatrix or TMatrixC): (Cylindrical) T-matrix to put in the arry
            kpars (float, (N, 2)-array): Tangential components of the wave vector
            a (array): Definition of the lattice
            eta (float or complex, optional): Splitting parameter in the lattice sum

        Returns:
            SMatrix
        """
        if tm.lattice is None:
            lattice = basis.hints["lattice"]
            kpar = basis.hints["kpar"]
            tm = tm.interacted_lattice(lattice, kpar, eta=eta)
        pu, pd = [tm.expandlattice(basis=basis, modetype=i) for i in ("up", "down")]
        au, ad = [tm.expand.inv(basis=basis, modetype=i) for i in ("up", "down")]
        eye = np.eye(len(basis))
        return cls(
            [[eye + pu @ tm @ au, pu @ tm @ ad], [pd @ tm @ au, eye + pd @ tm @ ad]]
        )

    def add(self, sm):
        """Couple another S-matrix on top of the current one.

        See also :func:`treams.SMatrix.stack` for a function that does not change the
        current S-matrix but creates a new one.

        Args:
            items (SMatrix): S-matrices in their intended order
            check_materials (bool, optional): Check for matching material parameters
                at each S-matrix
            check_materials (bool, optional): Check for matching modes at each S-matrix

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
        return SMatrix(snew)

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

        Args:
            illu (tuple): A 2-tuple of arrays, with the entries corresponding to
                upwards and downwards incoming modes.

        Returns:
            tuple
        """
        modetype = getattr(illu, "modetype", "up")
        if isinstance(modetype, tuple):
            modetype = modetype[max(-2, -len(tuple))]
        illu2 = np.zeros_like(self.kx) if illu2 is None else illu2
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

    def poynting_avg(self, coeffs, above=True):
        r"""Time-averaged z-component of the Poynting vector.

        Calculate the time-averaged Poynting vector's z-component

        .. math::

            \langle S_z \rangle
            = \frac{1}{2}
            \Re (\boldsymbol E \times \boldsymbol H^\ast) \boldsymbol{\hat{z}}

        on one side of the S-matrix with the given coefficients.

        Args:
            coeffs (2-tuple): The first entry are the upwards propagating modes the
                second one the downwards propagating modes
            above (bool, optional): Calculate the Poynting vector above or below the
                S-matrix

        Returns:
            float
        """
        choice = int(bool(above))
        ky, ky, pol = self.modes
        selections = pol == 0, pol == 1
        pref = (
            self.kz[choice, selections[0]] / self.ks[choice, 0],
            self.kz[choice, selections[1]] / self.ks[choice, 1],
        )
        coeffs = [np.zeros_like(self.kx) if c is None else np.array(c) for c in coeffs]
        allcoeffs = [
            (1, -1, coeffs[0][selections[0]]),
            (1, 1, coeffs[0][selections[1]]),
            (-1, -1, coeffs[1][selections[0]]),
            (-1, 1, coeffs[1][selections[1]]),
        ]
        res = 0
        if self.helicity:
            for (dira, pola, a), (dirb, polb, b) in itertools.product(
                allcoeffs, repeat=2
            ):
                res += a @ (
                    b.conjugate()
                    * (
                        pola * polb * pref[(polb + 1) // 2].conjugate() * dirb
                        + pref[(pola + 1) // 2] * dira
                    )
                )
            res *= 0.25
        else:
            for (dira, _, a), (dirb, _, b) in itertools.product(
                allcoeffs[::2], repeat=2
            ):
                res += a @ (b.conjugate() * pref[0].conjugate() * dirb)
            for (dira, _, a), (dirb, _, b) in itertools.product(
                allcoeffs[1::2], repeat=2
            ):
                res += a @ (b.conjugate() * pref[1] * dira)
            res *= 0.5
        return np.real(
            res / np.conjugate(np.sqrt(self.mu[choice] / self.epsilon[choice]))
        )

    def chirality_density(self, coeffs, z=None, above=True):
        r"""Volume-averaged chirality density.

        Calculate the volume-averaged chirality density

        .. math::

            \int_{A_\text{unit cell}} \mathrm d A \int_{z_0}^{z_1} \mathrm d z
            |G_+(\boldsymbol r)|^2 - |G_-(\boldsymbol r)|^2

        on one side of the S-matrix with the given coefficients. The calculation can
        also be done for an infinitely thin sheet. The Riemann-Silberstein vectors are
        :math:`\sqrt{2} \boldsymbol G_\pm(\boldsymbol r) = \boldsymbol E(\boldsymbol r)
        + \mathrm i Z_0 Z \boldsymbol H(\boldsymbol r)`.

        Args:
            coeffs (2-tuple): The first entry are the upwards propagating modes the
                second one the downwards propagating modes
            above (bool, optional): Calculate the chirality density above or below the
                S-matrix

        Returns:
            float
        """
        choice = int(bool(above))
        z = (0, 0) if z is None else z
        selections = self.pol == 0, self.pol == 1
        kzs = (self.kz[choice, selections[0]], self.kz[choice, selections[1]])
        coeffs = [np.zeros_like(self.kx) if c is None else np.array(c) for c in coeffs]
        allcoeffs = [
            (1, 0, coeffs[0][selections[0]]),
            (1, 1, coeffs[0][selections[1]]),
            (-1, 0, coeffs[1][selections[0]]),
            (-1, 1, coeffs[1][selections[1]]),
        ]
        res = 0
        if self.helicity:
            for (dira, pola, a), (dirb, polb, b) in itertools.product(
                allcoeffs, repeat=2
            ):
                if pola != polb:
                    continue
                res += (2 * pola - 1) * _coeff_chirality_density(
                    self.ks[choice, pola], kzs[pola], a, dira, b, dirb, z
                )
        else:
            for (dira, _, a), (dirb, _, b) in itertools.product(
                allcoeffs[::2], allcoeffs[1::2]
            ):
                res += _coeff_chirality_density(
                    self.ks[choice, 0], kzs[0], a, dira, b, dirb, z, False
                )
        return 0.5 * np.real(res)

    def tr(self, illu, direction=1):
        """Transmittance and reflectance.

        Calculate the transmittance and reflectance for one S-matrix with the given
        illumination and direction.

        Args:
            illu (complex, array_like): Expansion coefficients for the incoming light
            direction (int, optional): The direction of the field, options are `-1` and
                `1`

        Returns:
            tuple
        """
        if direction not in (-1, 1):
            raise ValueError(f"direction must be '-1' or '1', but is '{direction}''")
        illu_full = (illu, None) if direction == 1 else (None, illu)
        field_above, field_below = self.field_outside(illu_full)
        r = (None, field_below) if direction == 1 else (field_above, None)
        ir = (illu, field_below) if direction == 1 else (field_above, illu)
        t = (field_above, None) if direction == 1 else (None, field_below)

        s_r = self.poynting_avg(r, above=direction == -1)
        s_ir = self.poynting_avg(ir, above=direction == -1)
        s_t = self.poynting_avg(t, above=direction == 1)
        return s_t / (s_ir - s_r), s_r / (s_r - s_ir)

    def cd(self, illu, direction=1):
        """Transmission and absorption circular dichroism.

        Calculate the transmission and absoption CD for one S-matrix with the given
        illumination and direction.

        Args:
            illu (complex, array_like): Expansion coefficients for the incoming light
            direction (int, optional): The direction of the field, options are `-1` and
                `1`

        Returns:
            tuple
        """
        minus, plus = self.pol == 0, self.pol == 1
        illu = np.array(illu)
        if self.helicity:
            illuopposite = np.zeros_like(illu)
            illuopposite[minus] = illu[plus]
            illuopposite[plus] = illu[minus]
        else:
            illuopposite = copy.deepcopy(illu)
            illuopposite[minus] *= -1
        tm, rm = self.tr(illu, direction=direction)
        tp, rp = self.tr(illuopposite, direction=direction)
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
        dim = self.q.shape[2]
        res = np.empty((2 * dim, 2 * dim), dtype=complex)
        res[0:dim, 0:dim] = self.q[0, 0, :, :]
        res[0:dim, dim : 2 * dim] = self.q[0, 1, :, :]
        res[dim : 2 * dim, 0:dim] = -np.linalg.solve(
            self.q[1, 1, :, :], self.q[1, 0, :, :] @ self.q[0, 0, :, :]
        )
        res[dim : 2 * dim, dim : 2 * dim] = np.linalg.solve(
            self.q[1, 1, :, :], np.eye(dim) - self.q[1, 0, :, :] @ self.q[0, 1, :, :]
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


def _coeff_chirality_density(k, kz, a, dira, b, dirb, z=None, helicity=True):
    z = (0, 0) if z is None else z
    tmp = (dira * kz - dirb * kz.conjugate()) * 0.5
    pref = np.exp(1j * tmp * (z[1] + z[0]))
    if np.abs(z[1] - z[0]) > 1e-16:
        pref *= np.sin(tmp * (z[1] - z[0])) / (tmp * (z[1] - z[0]))
    pref *= 1 + (k * k - kz * (kz - dira * dirb * kz.conjugate())) / (k * k.conjugate())
    if helicity:
        return np.sum(a * b.conjugate() * pref)
    return 2 * np.sum(np.real(a * b.conjugate()) * pref)
