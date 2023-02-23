"""Operators for common transformations including different types of waves."""

import numpy as np

import treams._core as core
import treams.config as config
import treams.special as sc
import treams.util as util
from treams import cw, pw, sw
from treams._lattice import Lattice
from treams._material import Material


class Operator:
    """Operator base class.

    An operator is mainly intended to be used as a descriptor on a class. It will then
    automatically obtain attributes from the class that correspond to the variable names
    of the function that is linked to the operator in the class attribute `_func`. When
    called, the remainding arguments have to be specified.
    """

    def __get__(self, obj, objtype=None):
        """Descriptor method.

        This function obtains the keyword arguments from the object to which it is
        attached.
        """
        kwargs = getattr(obj, "ann", [{}, {}])[-2:]
        self._kwargs = tuple(
            map(
                lambda x: {
                    key: x[key] for key in self._func.__code__.co_varnames if key in x
                },
                kwargs,
            )
        )
        return self

    def __call__(self, *args, **kwargs):
        """Call the operator function."""
        kwargsa, kwargsb = self._parse_kwargs(kwargs, self._kwargs[0])
        return self._func(*args, **kwargsa, **kwargsb)

    def inv(self, *args, **kwargs):
        """Inverse operator function."""
        kwargsa, kwargsb = self._parse_kwargs(self._kwargs[-1], kwargs)
        return self._func(*args, **kwargsa, **kwargsb)

    @staticmethod
    def _parse_kwargs(kwargsa, kwargsb):
        """Merge special keyword arguments.

        The keywords `basis`, `modetype`, and `poltype` are merged if they appear in
        both sets of keyword arguments into a tuple.

        Args:
            kwargsa (mapping): First keyword set.
            kwargsb (mapping): Second keyword set.
        """
        kwargsa = {**kwargsa}
        kwargsb = {**kwargsb}
        for key in ("basis", "modetype", "poltype"):
            if key in kwargsa and key in kwargsb:
                kwargsa[key] = kwargsa[key], kwargsb.pop(key)
        return kwargsa, kwargsb


def _sw_rotate(phi, theta, psi, basis, to_basis, where):
    """Rotate spherical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    res = sw.rotate(
        *(m[:, None] for m in to_basis["lmp"]),
        *basis["lmp"],
        phi,
        theta,
        psi,
        where=where,
    )
    res[..., ~where] = 0
    return core.PhysicsArray(res, basis=(to_basis, basis))


def _cw_rotate(phi, basis, to_basis, where):
    """Rotate cylindrical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    res = cw.rotate(
        *(m[:, None] for m in to_basis["kzmp"]),
        *basis["kzmp"],
        phi,
        where=where,
    )
    res[..., ~where] = 0
    return core.PhysicsArray(res, basis=(to_basis, basis))


def _pw_rotate(phi, basis, where):
    """Rotate plane waves (actually rotates the basis)."""
    # TODO: rotate hints: lattice, kpar
    c1, s1 = np.cos(phi), np.sin(phi)
    r = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    kvecs = r @ np.array([basis.kx, basis.ky, basis.kz])
    modes = zip(*kvecs, basis.pol)
    res = np.eye(len(basis))
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(res, basis=(core.PlaneWaveBasis(modes), basis))


def _pwp_rotate(phi, basis, where):
    """Rotate partial plane waves (actually rotates the basis)."""
    # TODO: rotate hints: lattice, kpar
    if basis.alignment != "xy":
        ValueError(f"rotation on alignment: '{basis.alignment}'")
    c1, s1 = np.cos(phi), np.sin(phi)
    r = np.array([[c1, -s1], [s1, c1]])
    kx, ky, pol = basis[()]
    modes = zip(*(r @ np.array([kx, ky])), pol)
    res = np.eye(len(basis))
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(res, basis=(core.PlaneWaveBasisPartial(modes), basis))


def rotate(phi, theta=0, psi=0, *, basis, where=True):
    """Rotation matrix.

    For the given Euler angles apply a rotation for the given basis. In some basis sets
    only rotations around the z-axis are permitted.

    Args:
        phi (float): First Euler angle (rotation about z)
        theta (float, optional): Second Euler angle (rotation about y)
        psi (float, optional): Third Euler angle (rotation about z)
        basis (:class:`~treams.BasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        where (array-like, bool, optional): Only evaluate parts of the rotation matrix,
            the given array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis

    if isinstance(basis, core.SphericalWaveBasis):
        return _sw_rotate(phi, theta, psi, basis, to_basis, where)
    if theta != 0:
        raise ValueError("non-zero theta for rotation")
    if isinstance(basis, core.CylindricalWaveBasis):
        return _cw_rotate(phi, basis, to_basis, where)
    if to_basis != basis:
        raise ValueError("invalid basis")
    if isinstance(basis, core.PlaneWaveBasisPartial):
        return _pwp_rotate(phi, basis, where)
    if isinstance(basis, core.PlaneWaveBasis):
        return _pw_rotate(phi, basis, where)
    raise TypeError("invalid basis")


class Rotate(Operator):
    """Rotation matrix.

    When called as attribute of an object it returns a suitable rotation matrix to
    transform it. See also :func:`rotate`.
    """

    _func = staticmethod(rotate)

    def inv(self, *args, **kwargs):
        """Inverse rotation.

        This function applies the inverse rotation. In all basis sets this is just the
        hermitian conjugated matrix, because the rotations for them are unitary.
        """
        return self(*args, **kwargs).conj().T


def _sw_translate(r, basis, to_basis, k0, material, poltype, where):
    """Translate spherical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    ks = k0 * material.nmp
    r = sc.car2sph(r)
    res = sw.translate(
        *(m[:, None] for m in to_basis["lmp"]),
        *basis["lmp"],
        ks[basis.pol] * r[..., None, None, 0],
        r[..., None, None, 1],
        r[..., None, None, 2],
        singular=False,
        poltype=poltype,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        k0=(k0, k0),
        basis=(to_basis, basis),
        poltype=(poltype, poltype),
        material=(material, material),
    )


def _cw_translate(r, basis, k0, to_basis, material, where):
    """Translate cylindrical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    ks = material.ks(k0)[basis.pol]
    krhos = np.sqrt(ks * ks - basis.kz * basis.kz + 0j)
    krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
    r = sc.car2cyl(r)
    res = cw.translate(
        *(m[:, None] for m in to_basis["kzmp"]),
        *basis["kzmp"],
        krhos * r[..., None, None, 0],
        r[..., None, None, 1],
        r[..., None, None, 2],
        singular=False,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        k0=(k0, k0),
        basis=(to_basis, basis),
        material=(material, material),
    )


def _pw_translate(r, basis, to_basis, where):
    """Translate plane waves."""
    where = (
        where
        & (to_basis.kx[:, None] == basis.kx)
        & (to_basis.ky[:, None] == basis.ky)
        & (to_basis.kz[:, None] == basis.kz)
        & (to_basis.pol[:, None] == basis.pol)
    )
    res = pw.translate(
        basis.kx,
        basis.ky,
        basis.kz,
        r[..., None, None, 0],
        r[..., None, None, 1],
        r[..., None, None, 2],
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(res, basis=(to_basis, basis))


def _pwp_translate(r, basis, to_basis, k0, material, modetype, where):
    """Translate partial plane waves."""
    basis_c = basis.complete(k0, material, modetype)
    to_basis_c = to_basis.complete(k0, material, modetype)
    res = _pw_translate(r, basis_c, to_basis_c, where=where)
    del res.basis
    res.basis = (to_basis, basis)
    res.k0 = (k0, k0)
    res.material = (material, material)
    res.modetype = (modetype, modetype)
    return res


def translate(
    r, *, basis, k0=None, material=Material(), modetype="up", poltype=None, where=True
):
    """Translation matrix.

    Translate the given basis modes along the translation vector.

    Args:
        r (array-like): Translation vector
        basis (:class:`~treams.BasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        k0 (float, optional): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode, only used for
            :class:`~treams.PlaneWaveBasisPartial`.
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
        where (array-like, bool, optional): Only evaluate parts of the translation
            matrix, the given array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis
    poltype = config.POLTYPE if poltype is None else poltype
    material = Material(material)

    r = np.asanyarray(r)
    if r.shape[-1] != 3:
        raise ValueError("invalid 'r'")

    if isinstance(basis, core.PlaneWaveBasisPartial):
        return _pwp_translate(r, basis, to_basis, k0, material, modetype, where)
    if isinstance(basis, core.PlaneWaveBasis):
        return _pw_translate(r, basis, to_basis, where)
    if isinstance(basis, core.SphericalWaveBasis):
        return _sw_translate(r, basis, to_basis, k0, material, poltype, where)
    if isinstance(basis, core.CylindricalWaveBasis):
        return _cw_translate(r, basis, k0, to_basis, material, where)
    raise TypeError("invalid basis")


class Translate(Operator):
    """Translation matrix.

    When called as attribute of an object it returns a suitable translation matrix to
    transform it. See also :func:`translate`.
    """

    _func = staticmethod(translate)

    def inv(self, *args, **kwargs):
        """Inverse translation.

        The tranlation is essentially along :math:`-r`.
        """
        args = list(args)
        if len(args) > 0:
            args[0] = np.negative(args[0])
        if "r" in kwargs:
            kwargs["r"] = np.negative(kwargs["r"])
        return super().inv(*args, **kwargs)


def _sw_changepoltype(basis, to_basis, poltype, where):
    """Change the polarization type of spherical waves."""
    where = (
        (to_basis.l[:, None] == basis.l)
        & (to_basis.m[:, None] == basis.m)
        & (to_basis.pidx[:, None] == basis.pidx)
        & where
    )
    res = np.zeros_like(where, float)
    res[where] = np.sqrt(0.5)
    res[where & (to_basis.pol[:, None] == basis.pol) & (basis.pol == 0)] = -np.sqrt(0.5)
    return core.PhysicsArray(res, basis=(to_basis, basis), poltype=poltype)


def _cw_changepoltype(basis, to_basis, poltype, where):
    """Change the polarization type of cylindrical waves."""
    where = (
        (to_basis.kz[:, None] == basis.kz)
        & (to_basis.m[:, None] == basis.m)
        & (to_basis.pidx[:, None] == basis.pidx)
        & where
    )
    res = np.zeros_like(where, float)
    res[where] = np.sqrt(0.5)
    res[where & (to_basis.pol[:, None] == basis.pol) & (basis.pol == 0)] = -np.sqrt(0.5)
    return core.PhysicsArray(res, basis=(to_basis, basis), poltype=poltype)


def _pw_changepoltype(basis, to_basis, poltype, where):
    """Change the polarization type of plane waves."""
    where = (
        (to_basis.kx[:, None] == basis.kx)
        & (to_basis.ky[:, None] == basis.ky)
        & (to_basis.kz[:, None] == basis.kz)
        & where
    )
    res = np.zeros_like(where, float)
    res[where] = np.sqrt(0.5)
    res[where & (to_basis.pol[:, None] == basis.pol) & (basis.pol == 0)] = -np.sqrt(0.5)
    return core.PhysicsArray(res, basis=(to_basis, basis), poltype=poltype)


def _pwp_changepoltype(basis, to_basis, poltype, where):
    """Change the polarization type of partial plane waves."""
    if to_basis.alignment != basis.alignment:
        raise ValueError("incompatible basis alignments")
    bkx, bky, bpol = to_basis[()]
    where = (bkx[:, None] == basis._kx) & (bky[:, None] == basis._ky) & where
    res = np.zeros_like(where, float)
    res[where] = np.sqrt(0.5)
    res[where & (to_basis.pol[:, None] == basis.pol) & (basis.pol == 0)] = -np.sqrt(0.5)
    return core.PhysicsArray(res, basis=(to_basis, basis), poltype=poltype)


def changepoltype(poltype=None, *, basis, where=True):
    """Matrix to change polarization types.

    The polarization is switched between `helicity` and `parity`.

    Args:
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
        basis (:class:`~treams.BasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        where (array-like, bool, optional): Only evaluate parts of the matrix, the given
            array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis
    poltype = config.POLTYPE if poltype is None else poltype
    if poltype == "helicity":
        poltype = ("helicity", "parity")
    elif poltype == "parity":
        poltype = ("parity", "helicity")
    if poltype != ("helicity", "parity") and poltype != ("parity", "helicity"):
        raise ValueError(f"invalid poltype '{poltype}'")

    if isinstance(basis, core.SphericalWaveBasis):
        return _sw_changepoltype(basis, to_basis, poltype, where)
    if isinstance(basis, core.CylindricalWaveBasis):
        return _cw_changepoltype(basis, to_basis, poltype, where)
    if isinstance(basis, core.PlaneWaveBasisPartial):
        return _pwp_changepoltype(basis, to_basis, poltype, where)
    if isinstance(basis, core.PlaneWaveBasis):
        return _pw_changepoltype(basis, to_basis, poltype, where)
    raise TypeError("invalid basis")


class ChangePoltype(Operator):
    """Matrix to change polarization types.

    When called as attribute of an object it returns a suitable matrix to change the
    polarization types between `helicity` and `parity`. See also :func:`changepoltype`.
    """

    _func = staticmethod(changepoltype)

    @staticmethod
    def _parse_kwargs(kwargsa, kwargsb):
        """Merge special keyword arguments.

        The keywords `basis`, `modetype`, and `poltype` are merged if they appear in
        both sets of keyword arguments into a tuple.

        Args:
            kwargsa (mapping): First keyword set.
            kwargsb (mapping): Second keyword set.
        """
        kwargsa = {**kwargsa}
        kwargsb = {**kwargsb}
        opp = {"parity": "helicity", "helicity": "parity"}
        for key in ("basis", "modetype", "poltype"):
            if key in kwargsa and key in kwargsb:
                kwargsa[key] = kwargsa[key], kwargsb.pop(key)
            elif key == "poltype" and key in kwargsb:
                kwargsa[key] = opp[kwargsb[key]], kwargsb.pop(key)
        return kwargsa, kwargsb


def _sw_sw_expand(basis, to_basis, to_modetype, k0, material, modetype, poltype, where):
    """Expand spherical waves in spherical waves."""
    if not (
        modetype == "regular" == to_modetype
        or modetype == "singular" == to_modetype
        or (modetype == "singular" and to_modetype == "regular")
    ):
        raise ValueError(f"invalid expansion from {modetype} to {modetype}")
    rs = sc.car2sph(to_basis.positions[:, None, :] - basis.positions)
    ks = k0 * material.nmp
    res = sw.translate(
        *(m[:, None] for m in to_basis["lmp"]),
        *basis["lmp"],
        ks[basis.pol] * rs[to_basis.pidx[:, None], basis.pidx, 0],
        rs[to_basis.pidx[:, None], basis.pidx, 1],
        rs[to_basis.pidx[:, None], basis.pidx, 2],
        poltype=poltype,
        singular=modetype != to_modetype,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    res = core.PhysicsArray(
        res, k0=k0, basis=(to_basis, basis), poltype=poltype, material=material
    )
    if modetype == "singular" and to_modetype == "regular":
        res.modetype = (to_modetype, modetype)
    return res


def _sw_cw_expand(basis, to_basis, k0, material, poltype, where):
    """Expand cylindrical waves in spherical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    ks = material.ks(k0)[basis.pol]
    res = cw.to_sw(
        *(m[:, None] for m in to_basis["lmp"]),
        *basis["kzmp"],
        ks,
        poltype=poltype,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        poltype=poltype,
        material=material,
        modetype=("regular", "regular"),
    )


def _sw_pw_expand(basis, to_basis, k0, material, modetype, poltype, where):
    """Expand plane waves in spherical waves."""
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        basis_c = basis.complete(k0, material, modetype)
    elif k0 is not None:
        basis_c = basis.complete(k0, material, modetype)
    else:
        basis_c = basis
    res = pw.to_sw(
        *(m[:, None] for m in to_basis["lmp"]),
        *basis_c[()],
        poltype=poltype,
        where=where,
    ) * pw.translate(
        basis_c.kx,
        basis_c.ky,
        basis_c.kz,
        to_basis.positions[to_basis.pidx, None, 0],
        to_basis.positions[to_basis.pidx, None, 1],
        to_basis.positions[to_basis.pidx, None, 2],
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        basis=(to_basis, basis),
        k0=k0,
        material=material,
        poltype=poltype,
        modetype=("regular", modetype),
    )


def _cw_cw_expand(basis, to_basis, to_modetype, k0, material, modetype, poltype, where):
    """Expand cylindrical waves in cylindrical waves."""
    if modetype == "regular" == to_modetype or modetype == "singular" == to_modetype:
        modetype = to_modetype = None
    elif modetype != "singular" or to_modetype != "regular":
        raise ValueError(f"invalid expansion from {modetype} to {modetype}")
    rs = sc.car2cyl(to_basis.positions[:, None, :] - basis.positions)
    krhos = material.krhos(k0, basis.kz, basis.pol)
    res = cw.translate(
        *(m[:, None] for m in to_basis["kzmp"]),
        *basis["kzmp"],
        krhos * rs[to_basis.pidx[:, None], basis.pidx, 0],
        rs[to_basis.pidx[:, None], basis.pidx, 1],
        rs[to_basis.pidx[:, None], basis.pidx, 2],
        singular=modetype != to_modetype,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    res = core.PhysicsArray(res, k0=k0, basis=(to_basis, basis), material=material)
    if modetype == "singular" and to_modetype == "regular":
        res.modetype = (to_modetype, modetype)
    return res


def _cw_pw_expand(basis, to_basis, k0, material, modetype, where):
    """Expand plane waves in cylindrical waves."""
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        basis_c = basis.complete(k0, material, modetype)
    elif k0 is not None:
        basis_c = basis.complete(k0, material, modetype)
    else:
        basis_c = basis
    res = pw.to_cw(
        *(m[:, None] for m in to_basis["kzmp"]),
        *basis_c[()],
        where=where,
    ) * pw.translate(
        basis_c.kx,
        basis_c.ky,
        basis_c.kz,
        to_basis.positions[to_basis.pidx, None, 0],
        to_basis.positions[to_basis.pidx, None, 1],
        to_basis.positions[to_basis.pidx, None, 2],
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        basis=(to_basis, basis),
        k0=k0,
        material=material,
        modetype=("regular", modetype),
    )


def _pw_pw_expand(basis, to_basis, k0, material, modetype, where):
    """Expand plane waves in plane waves."""
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        basis_c = basis.complete(k0, material, modetype)
    elif k0 is not None:
        basis_c = basis.complete(k0, material, modetype)
    else:
        basis_c = basis
    if isinstance(to_basis, core.PlaneWaveBasisPartial):
        to_basis_c = basis.complete(k0, material, modetype)
    elif k0 is not None:
        to_basis_c = basis.complete(k0, material, modetype)
    else:
        to_basis_c = basis
    res = np.array(
        where
        & (to_basis_c.kx[:, None] == basis_c.kx)
        & (to_basis_c.ky[:, None] == basis_c.ky)
        & (to_basis_c.kz[:, None] == basis_c.kz)
        & (to_basis_c.pol[:, None] == basis_c.pol),
        int,
    )
    return core.PhysicsArray(
        res, basis=(to_basis, basis), k0=k0, material=material, modetype=modetype
    )


def expand(
    basis,
    modetype=None,
    *,
    k0=None,
    material=Material(),
    poltype=None,
    where=True,
):
    """Expansion matrix.

    Expand the modes from one basis set to another basis set. If applicable the modetype
    can also be changed, like for spherical and cylindrical waves from `singular` to
    `regular`. Not all expansions are available, only those that result in a discrete
    set of modes. For example, plane waves can be expanded in spherical waves, but the
    opposite transformation generally requires a continous spectrum (an integral) over
    plane waves.

    Args:
        basis (:class:`~treams.BasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        modetype (str, optional): Wave mode, used for
            :class:`~treams.SphericalWaveBasis` and :class:`CylindricalWaveBasis`.
        k0 (float, optional): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
        where (array-like, bool, optional): Only evaluate parts of the expansion matrix,
            the given array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis
    if isinstance(modetype, (tuple, list)):
        to_modetype, modetype = modetype
    else:
        to_modetype = None
    poltype = config.POLTYPE if poltype is None else poltype
    material = Material(material)
    if isinstance(basis, core.SphericalWaveBasis) and isinstance(
        to_basis, core.SphericalWaveBasis
    ):
        modetype = "regular" if modetype is None else modetype
        to_modetype = modetype if to_modetype is None else to_modetype
        return _sw_sw_expand(
            basis, to_basis, to_modetype, k0, material, modetype, poltype, where
        )
    if isinstance(basis, core.CylindricalWaveBasis):
        if isinstance(to_basis, core.CylindricalWaveBasis):
            modetype = "regular" if modetype is None else modetype
            to_modetype = modetype if to_modetype is None else to_modetype
            return _cw_cw_expand(
                basis, to_basis, to_modetype, k0, material, modetype, poltype, where
            )
        if isinstance(to_basis, core.SphericalWaveBasis):
            if modetype != "regular" and to_modetype not in (None, "regular"):
                raise ValueError("invalid modetype")
            return _sw_cw_expand(basis, to_basis, k0, material, poltype, where)
    if isinstance(basis, core.PlaneWaveBasis):
        if isinstance(to_basis, core.PlaneWaveBasis):
            to_modetype = modetype if to_modetype is None else to_modetype
            return _pw_pw_expand(basis, to_basis, k0, material, modetype, where)
        if isinstance(to_basis, core.CylindricalWaveBasis):
            if to_modetype not in (None, "regular"):
                raise ValueError("invalid modetype")
            return _cw_pw_expand(basis, to_basis, k0, material, modetype, where)
        if isinstance(to_basis, core.SphericalWaveBasis):
            if to_modetype not in (None, "regular"):
                raise ValueError("invalid modetype")
            return _sw_pw_expand(
                basis, to_basis, k0, material, modetype, poltype, where
            )
    raise TypeError("invalid basis")


class Expand(Operator):
    """Expansion matrix.

    When called as attribute of an object it returns a suitable transformation matrix to
    expand one set of modes into another basis set (and mode type, if applicable).
    See also :func:`expand`.
    """

    _func = staticmethod(expand)

    def __call__(self, basis=None, modetype=None, **kwargs):
        """Create the expansion matrix."""
        if basis is not None:
            kwargs["basis"] = basis
        if modetype is not None:
            kwargs["modetype"] = modetype
        return super().__call__(**kwargs)

    def inv(self, basis, modetype=None, **kwargs):
        """Inverse expansion.

        The inverse transformation is not available for all transformations.
        """
        kwargs["basis"] = basis
        if modetype is not None:
            kwargs["modetype"] = modetype
        return super().inv(**kwargs)


def _swl_expand(basis, to_basis, eta, k0, kpar, lattice, material, poltype, where):
    """Expand spherical waves in a lattice."""
    lattice = Lattice(lattice)
    ks = k0 * material.nmp
    if len(kpar) == 1:
        x = kpar[0]
        kpar = [np.nan, np.nan, x]
    elif len(kpar) == 2:
        x = kpar
        kpar = kpar + [np.nan]
    elif len(kpar) == 3:
        if lattice.dim == 3:
            x = kpar = [0 if np.isnan(x) else x for x in kpar]
        elif lattice.dim == 2:
            kpar = [0 if np.isnan(x) else x for x in kpar[:2]] + [np.nan]
            x = kpar[:2]
        elif lattice.dim == 1:
            x = 0 if np.isnan(kpar[2]) else kpar[2]
            kpar = [np.nan, np.nan, x]
    res = sw.translate_periodic(
        ks,
        x,
        lattice[...],
        to_basis.positions,
        to_basis[()],
        basis[()],
        basis.positions,
        poltype=poltype,
        eta=eta,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        poltype=poltype,
        material=material,
        lattice=lattice,
        modetype=("regular", "singular"),
        kpar=kpar,
    )


def _cw_sw_expand(basis, to_basis, k0, kpar, lattice, material, poltype, where):
    """Expand spherical waves in a lattice in cylindrical waves."""
    ks = k0 * material.nmp
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    if len(kpar) == 1:
        kpar = [np.nan, np.nan, kpar[0]]
    kpar = [np.nan, np.nan, kpar[2]]
    res = sw.periodic_to_cw(
        *(m[:, None] for m in to_basis["kzmp"]),
        *basis["lmp"],
        ks[basis.pol],
        Lattice(lattice, "z").volume,
        poltype=poltype,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        poltype=poltype,
        material=material,
        modetype="singular",
        lattice=lattice,
        kpar=kpar,
    )


def _pw_sw_expand(
    basis, to_basis, k0, kpar, lattice, material, modetype, poltype, where
):
    """Expand spherical waves in a lattice in plane waves."""
    if isinstance(to_basis, core.PlaneWaveBasis):
        if modetype is None and isinstance(to_basis, core.PlaneWaveBasisPartial):
            modetype = "up"
        to_basis_c = to_basis.complete(k0, material, modetype)
    if len(kpar) == 2:
        kpar = [kpar[0], kpar[1], np.nan]
    kpar[2] = np.nan
    res = sw.periodic_to_pw(
        *(m[:, None] for m in to_basis_c["xyzp"]),
        *basis["lmp"],
        Lattice(lattice, "xy").volume,
        poltype=poltype,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        basis=(to_basis, basis),
        k0=k0,
        kpar=kpar,
        lattice=lattice,
        material=material,
        modetype=(modetype, "singular"),
        poltype=poltype,
    )


def _cwl_expand(basis, to_basis, eta, k0, kpar, lattice, material, poltype, where):
    """Expand cylindrical waves in a lattice."""
    ks = material.ks(k0)
    alignment = (
        "x" if not isinstance(lattice, Lattice) and np.size(lattice) == 1 else None
    )
    lattice = Lattice(lattice, alignment)
    if len(kpar) == 1:
        x = kpar[0]
        kpar = [x, np.nan, basis.hints["kpar"][2]]
    elif len(kpar) == 2:
        x = kpar
        kpar = kpar + [basis.hints["kpar"][2]]
    elif len(kpar) == 3:
        if lattice.dim == 2:
            kpar = [0 if np.isnan(x) else x for x in kpar[:2]] + kpar[2:3]
            x = [kpar[0], kpar[1]]
        elif lattice.dim == 1:
            x = kpar[0] = 0 if np.isnan(kpar[0]) else kpar[0]

    res = cw.translate_periodic(
        ks,
        x,
        lattice[...],
        to_basis.positions,
        to_basis[()],
        basis[()],
        basis.positions,
        eta=eta,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        material=material,
        lattice=lattice,
        modetype=("regular", "singular"),
        kpar=kpar,
    )


def _pw_cw_expand(basis, to_basis, k0, lattice, kpar, material, modetype, where):
    """Expand cylindrical waves in a lattice in plane waves."""
    if isinstance(to_basis, core.PlaneWaveBasis):
        if modetype is None and isinstance(to_basis, core.PlaneWaveBasisPartial):
            modetype = "up"
        to_basis_c = to_basis.complete(k0, material, modetype)
    if len(kpar) == 1:
        kpar = [kpar[0], np.nan, np.nan]
    kpar[0] = 0 if np.isnan(kpar[0]) else kpar[0]
    res = cw.periodic_to_pw(
        *(m[:, None] for m in to_basis_c["xyzp"]),
        *basis["kzmp"],
        lattice.volume,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        k0=k0,
        basis=(to_basis, basis),
        material=material,
        modetype=(modetype, "singular"),
        lattice=lattice,
        kpar=kpar,
    )


def expandlattice(
    lattice=None,
    kpar=None,
    *,
    basis,
    eta=0,
    k0=None,
    material=Material(),
    modetype=None,
    poltype=None,
    where=True,
):
    """Expansion matrix in lattices.

    Expand the modes from one basis set which are assumed to be periodically repeated on
    a lattice into another basis set.

    Args:
        lattice (:class:`~treams.Lattice` or array-like, optional): Lattice definition.
            In some cases this argument can be omitted, when the lattice can be inferred
            from the basis.
        kpar (sequence, optional): The components of the wave vector tangential to the
            lattice. In some cases this argument can be omitted, when the lattice can be
            inferred from the basis.
        basis (:class:`~treams.BasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        k0 (float, optional): Wave number.
        eta (float or complex, optional): Split parameter used when the Ewald summation
            is applied for the lattice sums. By setting it to 0 the split is set
            automatically.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode, used for
            :class:`~treams.PlaneWaveBasisPartial`.
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
        where (array-like, bool, optional): Only evaluate parts of the expansion matrix,
            the give array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis
    if lattice is None:
        try:
            lattice = to_basis.hints["lattice"]
        except KeyError:
            lattice = basis.hints["lattice"]
    if not isinstance(lattice, Lattice) and np.size(lattice) == 1:
        alignment = "x" if isinstance(basis, core.CylindricalWaveBasis) else "z"
    else:
        alignment = None
    lattice = Lattice(lattice, alignment)
    if kpar is None:
        try:
            kpar = to_basis.hints["kpar"]
        except KeyError:
            kpar = basis.hints.get("kpar")
    try:
        kpar = list(kpar)
    except TypeError:
        if kpar is None:
            kpar = [np.nan] * 3
        else:
            kpar = [kpar]
    poltype = config.POLTYPE if poltype is None else poltype
    material = Material(material)
    if isinstance(basis, core.SphericalWaveBasis):
        if isinstance(to_basis, core.SphericalWaveBasis):
            return _swl_expand(
                basis, to_basis, eta, k0, kpar, lattice, material, poltype, where
            )
        if isinstance(to_basis, core.CylindricalWaveBasis):
            return _cw_sw_expand(
                basis, to_basis, k0, kpar, lattice, material, poltype, where
            )
        if isinstance(to_basis, core.PlaneWaveBasis):
            if isinstance(modetype, tuple):
                modetype = modetype[0]
            return _pw_sw_expand(
                basis, to_basis, k0, kpar, lattice, material, modetype, poltype, where
            )
    if isinstance(basis, core.CylindricalWaveBasis):
        if isinstance(to_basis, core.CylindricalWaveBasis):
            return _cwl_expand(
                basis, to_basis, eta, k0, kpar, lattice, material, poltype, where
            )
        if isinstance(to_basis, core.PlaneWaveBasis):
            if isinstance(modetype, tuple):
                modetype = modetype[0]
            return _pw_cw_expand(
                basis, to_basis, k0, lattice, kpar, material, modetype, where
            )
    raise TypeError("invalid basis")


class ExpandLattice(Operator):
    """Expansion matrix in a lattice.

    When called as attribute of an object it returns a suitable transformation matrix to
    expand one set of modes that is periodically repeated into another basis set.
    See also :func:`expandlattice`.
    """

    _func = staticmethod(expandlattice)

    def __call__(self, lattice=None, kpar=None, **kwargs):
        """Create the expansion matrix for periodic arrangements."""
        if lattice is not None:
            kwargs["lattice"] = lattice
        if kpar is not None:
            kwargs["kpar"] = kpar
        return super().__call__(**kwargs)

    def inv(self, *args, **kwargs):
        """Inverse expansion for periodic arrangements.

        The inverse transformation is not available.
        """
        raise NotImplementedError


def _pwp_permute(basis, n):
    """Permute axes in a partial plane wave basis."""
    alignment = basis.alignment
    dct = {"xy": "yz", "yz": "zx", "zx": "xy"}
    kpar = basis.hints.get("kpar")
    while n > 0:
        alignment = dct[alignment]
        kpar = kpar[[2, 0, 1]] if kpar is not None else kpar
        n -= 1
    obj = type(basis)(zip(basis._kx, basis._ky, basis.pol), alignment)
    if "lattice" in basis.hints:
        obj.hints["lattice"] = basis.hints["lattice"].permute(n)
    if kpar is not None:
        obj.hints["kpar"] = kpar
    return core.PhysicsArray(np.eye(len(basis)), basis=(obj, basis))


def _pw_permute(basis, n):
    """Permute axes in a plane wave basis."""
    kx, ky, kz = basis.kx, basis.ky, basis.kz
    kpar = basis.hints.get("kpar")
    while n > 0:
        kx, ky, kz = kz, kx, ky
        kpar = kpar[[2, 0, 1]] if kpar is not None else None
        n -= 1
    obj = type(basis)(zip(kx, ky, kz, basis.pol))
    if "lattice" in basis.hints:
        obj.hints["lattice"] = basis.hints["lattice"].permute(n)
    if kpar is not None:
        obj.hints["kpar"] = kpar
    return core.PhysicsArray(np.eye(len(basis)), basis=(obj, basis))


def permute(n=1, *, basis):
    """Permutation matrix.

    Permute the axes of a plane wave basis expansion.

    Args:
        n (int, optional): Number of permutations, defaults to 1.
        basis (:class:`~treams.BasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
    """
    if n != int(n):
        raise ValueError("'n' must be integer")
    n = n % 3
    if isinstance(basis, core.PlaneWaveBasisPartial):
        return _pwp_permute(basis, n)
    if isinstance(basis, core.PlaneWaveBasis):
        return _pw_permute(basis, n)
    raise TypeError("invalid basis")


class Permute(Operator):
    """Axes permutation matrix.

    When called as attribute of an object it returns a suitable transformation matrix to
    permute the axis definitions of plane waves. See also :func:`permute`.
    """

    _func = staticmethod(permute)

    def inv(self, *args, **kwargs):
        """Inverse permutation matrix.

        The inverse transformation is simply the transposed result.
        """
        x = self(*args, **kwargs)
        return x.T


def _sw_efield(r, basis, k0, material, modetype, poltype):
    """Electric field of spherical waves."""
    ks = k0 * material.nmp
    rsph = sc.car2sph(r - basis.positions)
    res = None
    if poltype == "helicity":
        if modetype == "regular":
            res = sc.vsw_rA(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
                basis.pol,
            )
        elif modetype == "singular":
            res = sc.vsw_A(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
                basis.pol,
            )
    elif poltype == "parity":
        if modetype == "regular":
            res = (1 - basis.pol[:, None]) * sc.vsw_rM(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            ) + basis.pol[:, None] * sc.vsw_rN(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            )
        elif modetype == "singular":
            res = (1 - basis.pol[:, None]) * sc.vsw_M(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            ) + basis.pol[:, None] * sc.vsw_N(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            )
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(sc.vsph2car(res, rsph[..., basis.pidx, :]))
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    res.ann[-2]["modetype"] = modetype
    return res


def _cw_efield(r, basis, k0, material, modetype, poltype):
    """Electric field of cylindrical waves."""
    material = Material(material)
    ks = material.ks(k0)[basis.pol]
    krhos = material.krhos(k0, basis.kz, basis.pol)
    krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
    poltype = config.POLTYPE if poltype is None else poltype
    rcyl = sc.car2cyl(r - basis.positions)
    res = None
    if poltype == "helicity":
        if modetype == "regular":
            res = sc.vcw_rA(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
                basis.pol,
            )
        elif modetype == "singular":
            res = sc.vcw_A(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
                basis.pol,
            )
    elif poltype == "parity":
        if modetype == "regular":
            res = (1 - basis.pol[:, None]) * sc.vcw_rM(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
            ) + basis.pol[:, None] * sc.vcw_rN(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
            )
        elif modetype == "singular":
            res = (1 - basis.pol[:, None]) * sc.vcw_M(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
            ) + basis.pol[:, None] * sc.vcw_N(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
            )
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(sc.vcyl2car(res, rcyl[..., basis.pidx, :]))
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    res.ann[-2]["modetype"] = modetype
    return res


def _pw_efield(r, basis, poltype):
    """Electric field of plane waves."""
    res = None
    if poltype == "helicity":
        res = sc.vpw_A(
            basis.kx,
            basis.ky,
            basis.kz,
            r[..., 0],
            r[..., 1],
            r[..., 2],
            basis.pol,
        )
    elif poltype == "parity":
        res = (1 - basis.pol[:, None]) * sc.vpw_M(
            basis.kx,
            basis.ky,
            basis.kz,
            r[..., 0],
            r[..., 1],
            r[..., 2],
        ) + basis.pol[:, None] * sc.vpw_N(
            basis.kx,
            basis.ky,
            basis.kz,
            r[..., 0],
            r[..., 1],
            r[..., 2],
        )
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(res)
    res.ann[-2]["basis"] = basis
    res.ann[-2]["poltype"] = poltype
    return res


def _pwp_efield(r, basis, k0, material, modetype, poltype):
    """Electric field of partial plane waves."""
    basis_c = basis.complete(k0, material, modetype)
    res = _pw_efield(r, basis_c, poltype)
    del res.ann[-2]["basis"]
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["modetype"] = modetype
    return res


def efield(r, *, basis, k0=None, material=Material(), modetype=None, poltype=None):
    """Electric field.

    The resulting matrix maps the electric field coefficients of the given basis to the
    electric field in Cartesian coordinates.

    Args:
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float, optional): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
    """
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_efield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_efield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        return _pwp_efield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasis):
        if k0 is not None:
            basis.complete(k0, material, modetype)
        return _pw_efield(r, basis, poltype)
    raise TypeError("invalid basis")


class EField(Operator):
    """Electric field evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`efield`.
    """

    _func = staticmethod(efield)

    def inv(self, *args, **kwargs):
        """Inverse transformation of an electric field to modes.

        The inverse transformation is not available.
        """
        raise NotImplementedError


def _sw_hfield(r, basis, k0, material, modetype, poltype):
    """Magnetic field of spherical waves."""
    ks = k0 * material.nmp
    rsph = sc.car2sph(r - basis.positions)
    res = None
    if poltype == "helicity":
        if modetype == "regular":
            res = (2 * basis.pol[:, None] - 1) * sc.vsw_rA(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
                basis.pol,
            )
        elif modetype == "singular":
            res = (2 * basis.pol[:, None] - 1) * sc.vsw_A(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
                basis.pol,
            )
    elif poltype == "parity":
        if modetype == "regular":
            res = basis.pol[:, None] * sc.vsw_rM(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            ) + (1 - basis.pol[:, None]) * sc.vsw_rN(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            )
        elif modetype == "singular":
            res = basis.pol[:, None] * sc.vsw_M(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            ) + (1 - basis.pol[:, None]) * sc.vsw_N(
                basis.l,
                basis.m,
                ks[basis.pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            )
    res *= -1j / material.impedance
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(sc.vsph2car(res, rsph[..., basis.pidx, :]))
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    res.ann[-2]["modetype"] = modetype
    return res


def _cw_hfield(r, basis, k0, material, modetype, poltype):
    """Magnetic field of cylindrical waves."""
    material = Material(material)
    ks = material.ks(k0)[basis.pol]
    krhos = material.krhos(k0, basis.kz, basis.pol)
    krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
    poltype = config.POLTYPE if poltype is None else poltype
    rcyl = sc.car2cyl(r - basis.positions)
    res = None
    if poltype == "helicity":
        if modetype == "regular":
            res = (2 * basis.pol[:, None] - 1) * sc.vcw_rA(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
                basis.pol,
            )
        elif modetype == "singular":
            res = (2 * basis.pol[:, None] - 1) * sc.vcw_A(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
                basis.pol,
            )
    elif poltype == "parity":
        if modetype == "regular":
            res = basis.pol[:, None] * sc.vcw_rM(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
            ) + (1 - basis.pol[:, None]) * sc.vcw_rN(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
            )
        elif modetype == "singular":
            res = basis.pol[:, None] * sc.vcw_M(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
            ) + (1 - basis.pol[:, None]) * sc.vcw_N(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
            )
    res *= -1j / material.impedance
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(sc.vcyl2car(res, rcyl[..., basis.pidx, :]))
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    res.ann[-2]["modetype"] = modetype
    return res


def _pw_hfield(r, basis, material, poltype):
    """Magnetic field of plane waves."""
    res = None
    if poltype == "helicity":
        res = (2 * basis.pol[:, None] - 1) * sc.vpw_A(
            basis.kx,
            basis.ky,
            basis.kz,
            r[..., 0],
            r[..., 1],
            r[..., 2],
            basis.pol,
        )
    elif poltype == "parity":
        res = basis.pol[:, None] * sc.vpw_M(
            basis.kx,
            basis.ky,
            basis.kz,
            r[..., 0],
            r[..., 1],
            r[..., 2],
        ) + (1 - basis.pol[:, None]) * sc.vpw_N(
            basis.kx,
            basis.ky,
            basis.kz,
            r[..., 0],
            r[..., 1],
            r[..., 2],
        )
    res *= -1j / material.impedance
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(res)
    res.ann[-2]["basis"] = basis
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    return res


def _pwp_hfield(r, basis, k0, material, modetype, poltype):
    """Magnetic field of partial plane waves."""
    basis_c = basis.complete(k0, material, modetype)
    res = _pw_efield(r, basis_c, poltype, material)
    del res.ann[-2]["basis"]
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["modetype"] = modetype
    return res


def hfield(r, *, basis, k0=None, material=Material(), modetype=None, poltype=None):
    r"""Magnetic field.

    The resulting matrix maps the electric field coefficients of the given basis to the
    magnetic field in Cartesian coordinates.

    The magnetic field is given in units of :math:`\frac{1}{Z_0} [\boldsymbol E]`.

    Args:
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float, optional): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
    """
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_hfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_hfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        return _pwp_hfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasis):
        if k0 is not None:
            basis.complete(k0, material, modetype)
        return _pw_hfield(r, basis, material, poltype)
    raise TypeError("invalid basis")


class HField(Operator):
    """Magnetic field evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`hfield`.
    """

    _func = staticmethod(efield)

    def inv(self, *args, **kwargs):
        """Inverse transformation of a magnetic field to modes.

        The inverse transformation is not available.
        """
        raise NotImplementedError


def _sw_dfield(r, basis, k0, material, modetype, poltype):
    """Displacement field of spherical waves."""
    res = _sw_efield(r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol][:, None] / material.impedance
    else:
        res *= material.epsilon
    return res


def _cw_dfield(r, basis, k0, material, modetype, poltype):
    """Displacement field of cylindrical waves."""
    res = _cw_efield(r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol][:, None] / material.impedance
    else:
        res *= material.epsilon
    return res


def _pw_dfield(r, basis, material, poltype):
    """Displacement field of plane waves."""
    res = _pw_efield(r, basis, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol][:, None] / material.impedance
    else:
        res *= material.epsilon
    res.ann[-2]["material"] = material
    return res


def _pwp_dfield(r, basis, k0, material, modetype, poltype):
    """Displacement field of partial plane waves."""
    res = _pwp_efield(r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol][:, None] / material.impedance
    else:
        res *= material.epsilon
    return res


def dfield(r, *, basis, k0=None, material=Material(), modetype=None, poltype=None):
    r"""Displacement field.

    The resulting matrix maps the electric field coefficients of the given basis to the
    displacement field in Cartesian coordinates.

    The displacement field is given in units of :math:`\epsilon_0 [\boldsymbol E]`.

    Args:
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float, optional): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
    """
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_dfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_dfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        return _pwp_dfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasis):
        if k0 is not None:
            basis.complete(k0, material, modetype)
        return _pw_dfield(r, basis, material, poltype)
    raise TypeError("invalid basis")


class DField(Operator):
    """Displacement field evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`dfield`.
    """

    _func = staticmethod(efield)

    def inv(self, *args, **kwargs):
        """Inverse transformation of a displacement field to modes.

        The inverse transformation is not available.
        """
        raise NotImplementedError


def _sw_bfield(r, basis, k0, material, modetype, poltype):
    """Magnetic flux density of spherical waves."""
    res = _sw_hfield(r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol][:, None] * material.impedance
    else:
        res *= material.mu
    return res


def _cw_bfield(r, basis, k0, material, modetype, poltype):
    """Magnetic flux density of cylindrical waves."""
    res = _cw_hfield(r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol][:, None] * material.impedance
    else:
        res *= material.mu
    return res


def _pw_bfield(r, basis, poltype, material):
    """Magnetic flux density of plane waves."""
    res = _pw_hfield(r, basis, material, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol][:, None] * material.impedance
    else:
        res *= material.mu
    return res


def _pwp_bfield(r, basis, k0, material, modetype, poltype):
    """Magnetic flux density of partial plane waves."""
    res = _pwp_hfield(r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol][:, None] * material.impedance
    else:
        res *= material.mu
    return res


def bfield(r, *, basis, k0=None, material=Material(), modetype=None, poltype=None):
    r"""Magnetic flux density.

    The resulting matrix maps the electric field coefficients of the given basis to the
    magnetic flux density in Cartesian coordinates.

    The magnetic flux density is given in units of
    :math:`\frac{1}{c_0} [\boldsymbol E]`.

    Args:
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float, optional): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
    """
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_bfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_bfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        return _pwp_bfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasis):
        if k0 is not None:
            basis.complete(k0, material, modetype)
        return _pw_bfield(r, basis, poltype, material)
    raise TypeError("invalid basis")


class BField(Operator):
    """Magnetic flux density evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`bfield`.
    """

    _func = staticmethod(efield)

    def inv(self, *args, **kwargs):
        """Inverse transformation of a magnetic flux density to modes.

        The inverse transformation is not available.
        """
        raise NotImplementedError


def _sw_gfield(pol, r, basis, k0, material, modetype, poltype):
    """Riemann-Silberstein field G of spherical waves."""
    ks = k0 * material.nmp
    rsph = sc.car2sph(r - basis.positions)
    res = None
    if poltype == "helicity":
        if modetype == "regular":
            res = (basis.pol[:, None] == pol) * sc.vsw_rA(
                basis.l,
                basis.m,
                ks[pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
                pol,
            )
        elif modetype == "singular":
            res = (basis.pol[:, None] == pol) * sc.vsw_A(
                basis.l,
                basis.m,
                ks[pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
                pol,
            )
    elif poltype == "parity":
        if modetype == "regular":
            res = (1 + 2 * (pol - 1) * basis.pol[:, None]) * sc.vsw_rM(
                basis.l,
                basis.m,
                ks[pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            ) + (2 * pol - 1 - 2 * (pol - 1) * basis.pol[:, None]) * sc.vsw_rN(
                basis.l,
                basis.m,
                ks[pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            )
        elif modetype == "singular":
            res = (1 + 2 * (pol - 1) * basis.pol[:, None]) * sc.vsw_M(
                basis.l,
                basis.m,
                ks[pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            ) + (2 * pol - 1 - 2 * (pol - 1) * basis.pol[:, None]) * sc.vsw_N(
                basis.l,
                basis.m,
                ks[pol] * rsph[..., basis.pidx, 0],
                rsph[..., basis.pidx, 1],
                rsph[..., basis.pidx, 2],
            )
    if res is None:
        raise ValueError("invalid parameters")
    res *= np.sqrt(2)
    res = util.AnnotatedArray(sc.vsph2car(res, rsph[..., basis.pidx, :]))
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    res.ann[-2]["modetype"] = modetype
    return res


def _cw_gfield(pol, r, basis, k0, material, modetype, poltype):
    """Riemann-Silberstein field G of cylindrical waves."""
    material = Material(material)
    ks = material.ks(k0)[basis.pol]
    krhos = material.krhos(k0, basis.kz, basis.pol)
    krhos[krhos.imag < 0] = -krhos[krhos.imag < 0]
    poltype = config.POLTYPE if poltype is None else poltype
    rcyl = sc.car2cyl(r - basis.positions)
    res = None
    if poltype == "helicity":
        if modetype == "regular":
            res = (basis.pol[:, None] == pol) * sc.vcw_rA(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
                basis.pol,
            )
        elif modetype == "singular":
            res = (basis.pol[:, None] == pol) * sc.vcw_A(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
                basis.pol,
            )
    elif poltype == "parity":
        if modetype == "regular":
            res = (1 + 2 * (pol - 1) * basis.pol[:, None]) * sc.vcw_rM(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
            ) + (2 * pol - 1 - 2 * (pol - 1) * basis.pol[:, None]) * sc.vcw_rN(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
            )
        elif modetype == "singular":
            res = (1 + 2 * (pol - 1) * basis.pol[:, None]) * sc.vcw_M(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
            ) + (2 * pol - 1 - 2 * (pol - 1) * basis.pol[:, None]) * sc.vcw_N(
                basis.kz,
                basis.m,
                krhos * rcyl[..., basis.pidx, 0],
                rcyl[..., basis.pidx, 1],
                rcyl[..., basis.pidx, 2],
                ks,
            )
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(sc.vcyl2car(res, rcyl[..., basis.pidx, :]))
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    res.ann[-2]["modetype"] = modetype
    return res


def _pw_gfield(pol, r, basis, poltype):
    """Riemann-Silberstein field G of plane waves."""
    res = None
    if poltype == "helicity":
        res = (basis.pol[:, None] == pol) * sc.vpw_A(
            basis.kx,
            basis.ky,
            basis.kz,
            r[..., 0],
            r[..., 1],
            r[..., 2],
            basis.pol,
        )
    elif poltype == "parity":
        res = (1 + 2 * (pol - 1) * basis.pol[:, None]) * sc.vpw_M(
            basis.kx,
            basis.ky,
            basis.kz,
            r[..., 0],
            r[..., 1],
            r[..., 2],
        ) + (2 * pol - 1 - 2 * (pol - 1) * basis.pol[:, None]) * sc.vpw_N(
            basis.kx,
            basis.ky,
            basis.kz,
            r[..., 0],
            r[..., 1],
            r[..., 2],
        )
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(res)
    res.ann[-2]["basis"] = basis
    res.ann[-2]["poltype"] = poltype
    return res


def _pwp_gfield(pol, r, basis, k0, material, modetype, poltype):
    """Riemann-Silberstein field G of partial plane waves."""
    basis_c = basis.complete(k0, material, modetype)
    res = _pw_gfield(pol, r, basis_c, poltype)
    del res.ann[-2]["basis"]
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["modetype"] = modetype
    return res


def gfield(pol, r, *, basis, k0=None, material=Material(), modetype=None, poltype=None):
    """Riemann-Silberstein field G.

    The resulting matrix maps the electric field coefficients of the given basis to the
    Riemann-Silberstein field G in Cartesian coordinates.

    Args:
        pol (int): Type of the Riemann-Silberstein field, must be 1 or -1. In analogy to
            to polarizations for helicity mode, 0 is also treated as -1.
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float, optional): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
    """
    if pol == -1:
        pol = 0
    elif pol not in (0, 1):
        raise ValueError(f"invalid argument pol: '{pol}'")
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_gfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_gfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        return _pwp_gfield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasis):
        if k0 is not None:
            basis.complete(k0, material, modetype)
        return _pw_gfield(pol, r, basis, poltype)
    raise TypeError("invalid basis")


class GField(Operator):
    """Riemann-Silberstein field G evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`gfield`.
    """

    _func = staticmethod(efield)

    def inv(self, *args, **kwargs):
        """Inverse transformation of a Riemann-Silberstein field to modes.

        The inverse transformation is not available.
        """
        raise NotImplementedError


def _sw_ffield(pol, r, basis, k0, material, modetype, poltype):
    """Riemann-Silberstein field F of spherical waves."""
    res = _sw_gfield(pol, r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol, None] / material.n
    return res


def _cw_ffield(pol, r, basis, k0, material, modetype, poltype):
    """Riemann-Silberstein field F of cylindrical waves."""
    res = _cw_gfield(pol, r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol, None] / material.n
    return res


def _pw_ffield(pol, r, basis, material, poltype):
    """Riemann-Silberstein field F of plane waves."""
    res = _pw_gfield(pol, r, basis, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol, None] / material.n
        res.ann[-2]["material"] = material
    return res


def _pwp_ffield(pol, r, basis, k0, material, modetype, poltype):
    """Riemann-Silberstein field F of partial plane waves."""
    res = _pwp_gfield(pol, r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol, None] / material.n
    return res


def ffield(pol, r, *, basis, k0=None, material=Material(), modetype=None, poltype=None):
    """Riemann-Silberstein field F.

    The resulting matrix maps the electric field coefficients of the given basis to the
    Riemann-Silberstein field F in Cartesian coordinates.

    Args:
        pol (int): Type of the Riemann-Silberstein field, must be 1 or -1. In analogy to
            to polarizations for helicity mode, 0 is also treated as -1.
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float, optional): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`polarizations:Polarizations`.
    """
    if pol == -1:
        pol = 0
    elif pol not in (0, 1):
        raise ValueError(f"invalid argument pol: '{pol}'")
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_ffield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_ffield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        return _pwp_ffield(r, basis, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasis):
        if k0 is not None:
            basis.complete(k0, material, modetype)
        return _pw_ffield(pol, r, basis, material, poltype)
    raise TypeError("invalid basis")


class FField(Operator):
    """Riemann-Silberstein field F evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`ffield`.
    """

    _func = staticmethod(efield)

    def inv(self, *args, **kwargs):
        """Inverse transformation of a Riemann-Silberstein field to modes.

        The inverse transformation is not available.
        """
        raise NotImplementedError
