"""Operators for common transformations including different types of waves."""

import inspect

import numpy as np

import treams._core as core
import treams.special as sc
from treams import config, cw, pw, sw, util
from treams._lattice import Lattice, WaveVector
from treams._material import Material


class Operator:
    """Operator base class.

    An operator is mainly intended to be used as a descriptor on a class. It will then
    automatically obtain attributes from the class that correspond to the variable names
    of the function that is linked to the operator in the class attribute `_FUNC`. When
    called, the remaining arguments have to be specified.
    """

    __array_priority__ = 1.0
    _FUNC = None

    def __init__(self, *args, isinv=False):
        self._args = args
        self._isinv = bool(isinv)

    @property
    def isinv(self):
        return self._isinv

    @property
    def inv(self):
        return type(self)(*self._args, isinv=not self.isinv)

    @property
    def FUNC(self):
        return self._FUNC

    def __call__(self, **kwargs):
        if self.isinv:
            return self._call_inv(**kwargs)
        return self.FUNC(*self._args, **kwargs)

    def _call_inv(self, **kwargs):
        return self.FUNC(*self._args, **kwargs)

    def __matmul__(self, other):
        if isinstance(other, Operator):
            raise NotImplementedError
        return self(**self.get_kwargs(other, max(-2, -np.ndim(other)))) @ other

    def __rmatmul__(self, other):
        return other @ self(**self.get_kwargs(other))

    def get_kwargs(self, obj, dim=-1):
        kwargs = {}
        for name, param in inspect.signature(self.FUNC).parameters.items():
            if param.kind == param.KEYWORD_ONLY:
                val = getattr(obj, name, None)
                if isinstance(val, tuple):
                    val = val[dim]
                if val is not None:
                    kwargs[name] = val
        return kwargs

    def __repr__(self):
        if len(self._args) == 1:
            args = f"({self._args[0]})"
        else:
            args = str(self._args)
        if self.isinv:
            args = args[:-1] + ", isinv=True)"
        return f"{type(self).__name__}" + args


class OperatorAttribute:
    def __init__(self, op):
        self._op = op
        self._obj = self._objtype = None

    @property
    def OP(self):
        return self._op

    def eval(self, *args, **kwargs):
        op = self.OP(*args)
        dim = max(-2, -np.ndim(self._obj))
        return op(**self._merge_kwargs(kwargs, op.get_kwargs(self._obj, dim)))

    @staticmethod
    def _merge_kwargs(kwargsa, kwargsb):
        for key, val in kwargsb.items():
            if key == "basis" and key in kwargsa:
                kwargsa[key] = kwargsa["basis"], val
            elif key in kwargsa:
                raise TypeError(f"got multiple values for keyword argument '{key}'")
            else:
                kwargsa[key] = val
        return kwargsa

    def eval_inv(self, *args, **kwargs):
        op = self.OP(*args).inv
        return op(**self._merge_kwargs(kwargs, op.get_kwargs(self._obj)))

    def __get__(self, obj, objtype=None):
        self._obj = obj
        self._objtype = type(obj) if objtype is None else objtype
        return self

    def __call__(self, *args, **kwargs):
        if np.ndim(self._obj) == 1:
            return self.apply_left(*args, **kwargs)
        try:
            inv = self.eval_inv(*args, **kwargs)
        except NotImplementedError:
            return self.apply_left(*args, **kwargs)
        if issubclass(self._objtype, util.AnnotatedArray):
            return self._objtype.relax(self.eval(*args, **kwargs) @ self._obj @ inv)
        return self.eval(*args, **kwargs) @ self._obj @ inv

    def apply_left(self, *args, **kwargs):
        return self.eval(*args, **kwargs) @ self._obj

    def apply_right(self, *args, **kwargs):
        return self._obj @ self.eval_inv(*args, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}({self.OP.__name__})"


def _sw_rotate(phi, theta, psi, basis, to_basis, where):
    """Rotate spherical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    res = sw.rotate(
        *(m[:, None] for m in to_basis.lms), *basis.lms, phi, theta, psi, where=where
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(res, basis=(to_basis, basis))


def _cw_rotate(phi, basis, to_basis, where):
    """Rotate cylindrical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    res = cw.rotate(*(m[:, None] for m in to_basis.zms), *basis.zms, phi, where=where)
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(res, basis=(to_basis, basis))


def _pwa_rotate(phi, basis, where):
    """Rotate plane waves (actually rotates the basis)."""
    c1, s1 = np.cos(phi), np.sin(phi)
    r = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    kvecs = r @ np.array([basis.qx, basis.qy, basis.qz])
    res = np.eye(len(basis))
    res[..., np.logical_not(where)] = 0
    newbasis = core.PlaneWaveBasisByUnitVector(zip(*kvecs, basis.pol))
    if basis.lattice is not None:
        newbasis.lattice = basis.lattice.rotate(phi)
    if basis.kpar is not None:
        newbasis.kpar = basis.kpar.rotate(phi)
    return core.PhysicsArray(res, basis=(newbasis, basis))


def _pwp_rotate(phi, basis, where):
    """Rotate partial plane waves (actually rotates the basis)."""
    if basis.alignment != "xy":
        raise ValueError(f"rotation on alignment: '{basis.alignment}'")
    c1, s1 = np.cos(phi), np.sin(phi)
    r = np.array([[c1, -s1], [s1, c1]])
    kx, ky, pol = basis[()]
    res = np.eye(len(basis))
    res[..., np.logical_not(where)] = 0
    newbasis = core.PlaneWaveBasisByComp(zip(*(r @ np.array([kx, ky])), pol))
    if basis.lattice is not None:
        newbasis.lattice = basis.lattice.rotate(phi)
    if basis.kpar is not None:
        newbasis.kpar = basis.kpar.rotate(phi)
    return core.PhysicsArray(res, basis=(newbasis, basis))


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
    phi = phi + psi
    if isinstance(basis, core.CylindricalWaveBasis):
        return _cw_rotate(phi, basis, to_basis, where)
    if to_basis != basis:
        raise ValueError("invalid basis")
    if isinstance(basis, core.PlaneWaveBasisByComp):
        return _pwp_rotate(phi, basis, where)
    if isinstance(basis, core.PlaneWaveBasisByUnitVector):
        return _pwa_rotate(phi, basis, where)
    raise TypeError("invalid basis")


class Rotate(Operator):
    _FUNC = staticmethod(rotate)

    def __init__(self, phi, theta=0, psi=0, *, isinv=False):
        super().__init__(phi, theta, psi, isinv=isinv)

    def _call_inv(self, **kwargs):
        if "basis" in kwargs and isinstance(kwargs["basis"], tuple):
            kwargs["basis"] = kwargs["basis"][::-1]
        return self.FUNC(*(-a for a in self._args[::-1]), **kwargs)


def _sw_translate(r, basis, to_basis, k0, material, poltype, where):
    """Translate spherical waves."""
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    ks = k0 * material.nmp
    r = sc.car2sph(r)
    res = sw.translate(
        *(m[:, None] for m in to_basis.lms),
        *basis.lms,
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
        *(m[:, None] for m in to_basis.zms),
        *basis.zms,
        krhos * r[..., None, None, 0],
        r[..., None, None, 1],
        r[..., None, None, 2],
        singular=False,
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res, k0=(k0, k0), basis=(to_basis, basis), material=(material, material)
    )


def _pw_translate(r, basis, k0, to_basis, material, modetype, where):
    """Translate plane waves."""
    kvecs = basis.kvecs(k0, material, modetype)
    kx, ky, kz = to_basis.kvecs(k0, material, modetype)
    where = (
        where
        & (np.abs(kx[:, None] - kvecs[0]) < 1e-14)
        & (np.abs(ky[:, None] - kvecs[1]) < 1e-14)
        & (np.abs(kz[:, None] - kvecs[2]) < 1e-14)
        & (to_basis.pol[:, None] == basis.pol)
    )
    res = pw.translate(
        *kvecs,
        r[..., None, None, 0],
        r[..., None, None, 1],
        r[..., None, None, 2],
        where=where,
    )
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(
        res,
        k0=(k0,) * 2,
        basis=(to_basis, basis),
        material=(material,) * 2,
        modetype=(modetype,) * 2,
    )


def translate(
    r, *, basis, k0=None, material=Material(), modetype=None, poltype=None, where=True
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
            :class:`~treams.PlaneWaveBasisByComp`.
        poltype (str, optional): Polarization, see also
            :ref:`params:Polarizations`.
        where (array-like, bool, optional): Only evaluate parts of the translation
            matrix, the given array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis
    poltype = config.POLTYPE if poltype is None else poltype
    material = Material(material)
    if poltype == "parity" and material.ischiral:
        raise ValueError("poltype 'parity' with chiral material")

    r = np.asanyarray(r)
    if r.shape[-1] != 3:
        raise ValueError("invalid 'r'")

    if isinstance(basis, core.PlaneWaveBasis):
        if isinstance(basis, core.PlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _pw_translate(r, basis, k0, to_basis, material, modetype, where)
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

    _FUNC = staticmethod(translate)

    def __init__(self, r, *, isinv=False):
        super().__init__(r, isinv=isinv)

    def _call_inv(self, **kwargs):
        if "basis" in kwargs and isinstance(kwargs["basis"], tuple):
            kwargs["basis"] = kwargs["basis"][::-1]
        return self.FUNC(np.negative(self._args[0]), **kwargs)

def _coeffs_poltype_change(poltype, basis, to_basis, where):
    res = np.zeros_like(where, float)
    if poltype[0] == poltype[1]:
        res[where & (to_basis.pol[:, None] == basis.pol)] = 1.0
    else:
        res[where] = np.sqrt(0.5)
        res[where & (to_basis.pol[:, None] == basis.pol) & (basis.pol == 0)] = -np.sqrt(0.5)
    return core.PhysicsArray(res, basis=(to_basis, basis), poltype=poltype)

def _sw_changepoltype(basis, to_basis, poltype, where):
    """Change or keep the polarization type of spherical waves."""
    where = (
        (to_basis.l[:, None] == basis.l)
        & (to_basis.m[:, None] == basis.m)
        & (to_basis.pidx[:, None] == basis.pidx)
        & where
    )
    return _coeffs_poltype_change(poltype, basis, to_basis, where)


def _cw_changepoltype(basis, to_basis, poltype, where):
    """Change the polarization type of cylindrical waves."""
    where = (
        (to_basis.kz[:, None] == basis.kz)
        & (to_basis.m[:, None] == basis.m)
        & (to_basis.pidx[:, None] == basis.pidx)
        & where
    )
    return _coeffs_poltype_change(poltype, basis, to_basis, where)


def _pwa_changepoltype(basis, to_basis, poltype, where):
    """Change the polarization type of plane waves."""
    where = (
        (to_basis.qx[:, None] == basis.qx)
        & (to_basis.qy[:, None] == basis.qy)
        & (to_basis.qz[:, None] == basis.qz)
        & where
    )
    return _coeffs_poltype_change(poltype, basis, to_basis, where)


def _pwp_changepoltype(basis, to_basis, poltype, where):
    """Change the polarization type of partial plane waves."""
    if to_basis.alignment != basis.alignment:
        raise ValueError("incompatible basis alignments")
    akx, aky, _ = basis[()]
    bkx, bky, _ = to_basis[()]
    where = (bkx[:, None] == akx) & (bky[:, None] == aky) & where
    res = np.zeros_like(where, float)
    res[where] = np.sqrt(0.5)
    res[where & (to_basis.pol[:, None] == basis.pol) & (basis.pol == 0)] = -np.sqrt(0.5)
    return core.PhysicsArray(res, basis=(to_basis, basis), poltype=poltype)

opposite_poltype = {"parity": "helicity", "helicity": "parity"}

def changepoltype(poltype=None, *, basis, where=True):
    """Matrix to change polarization types.

    The polarization is switched between `helicity` and `parity`.

    Args:
        poltype (str, optional): Polarization, see also
            :ref:`params:Polarizations`.
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

    if not isinstance(poltype, tuple):
        poltype = (poltype, opposite_poltype[poltype])

    # if poltype == "helicity":
    #     poltype = ("helicity", "parity")
    # elif poltype == "parity":
    #     poltype = ("parity", "helicity")

    if np.any([pt not in ("helicity", "parity") for pt in poltype]):
        raise ValueError(f"invalid poltype '{poltype}'")
    if isinstance(basis, core.SphericalWaveBasis):
        return _sw_changepoltype(basis, to_basis, poltype, where)
    if isinstance(basis, core.CylindricalWaveBasis):
        return _cw_changepoltype(basis, to_basis, poltype, where)
    if isinstance(basis, core.PlaneWaveBasisByComp):
        return _pwp_changepoltype(basis, to_basis, poltype, where)
    if isinstance(basis, core.PlaneWaveBasisByUnitVector):
        return _pwa_changepoltype(basis, to_basis, poltype, where)
    raise TypeError("invalid basis")


class ChangePoltype(Operator):
    """Matrix to change polarization types.

    When called as attribute of an object it returns a suitable matrix to change the
    polarization types between `helicity` and `parity`. See also :func:`changepoltype`.
    """

    _FUNC = staticmethod(changepoltype)

    def __init__(self, poltype=None, *, isinv=False):
        args = () if poltype is None else (poltype,)
        super().__init__(*args, isinv=isinv)

    def get_kwargs(self, obj, dim=-1):
        kwargs = super().get_kwargs(obj, dim)
        from_poltype = getattr(obj, "poltype", config.POLTYPE)
        if isinstance(from_poltype, tuple):
            from_poltype = from_poltype[dim]

        to_poltype = opposite_poltype[from_poltype]
        if self._args:
            to_poltype = self._args[0]
            self._args=[]
        
        kwargs["poltype"] = (to_poltype, from_poltype)
        return kwargs

    def _call_inv(self, **kwargs):
        if "basis" in kwargs and isinstance(kwargs["basis"], tuple):
            kwargs["basis"] = kwargs["basis"][::-1]
        opp = {"parity": "helicity", "helicity": "parity"}
        poltype = self._args[0] if self._args else kwargs.pop("poltype")
        if isinstance(poltype, tuple):
            poltype = poltype[::-1]
        else:
            poltype = opp[poltype]
        return self.FUNC(poltype, **kwargs)


def _sw_sw_expand(basis, to_basis, to_modetype, k0, material, modetype, poltype, where):
    """Expand spherical waves in spherical waves."""
    if not (
        modetype == "regular" == to_modetype
        or modetype == "singular" == to_modetype
        or (modetype == "singular" and to_modetype == "regular")
    ):
        raise ValueError(f"invalid expansion from {modetype} to {to_modetype}")
    rs = sc.car2sph(to_basis.positions[:, None, :] - basis.positions)
    ks = k0 * material.nmp
    res = sw.translate(
        *(m[:, None] for m in to_basis.lms),
        *basis.lms,
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
        *(m[:, None] for m in to_basis.lms),
        *basis.zms,
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
    if isinstance(basis, core.PlaneWaveBasisByComp):
        modetype = "up" if modetype is None else modetype
    kvecs = basis.kvecs(k0, material, modetype)
    res = pw.to_sw(
        *(m[:, None] for m in to_basis.lms),
        *kvecs,
        basis.pol,
        poltype=poltype,
        where=where,
    ) * pw.translate(
        *kvecs,
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


def _cw_cw_expand(basis, to_basis, to_modetype, k0, material, modetype, where):
    """Expand cylindrical waves in cylindrical waves."""
    if modetype == "regular" == to_modetype or modetype == "singular" == to_modetype:
        modetype = to_modetype = None
    elif modetype != "singular" or to_modetype != "regular":
        raise ValueError(f"invalid expansion from {modetype} to {to_modetype}")
    rs = sc.car2cyl(to_basis.positions[:, None, :] - basis.positions)
    krhos = material.krhos(k0, basis.kz, basis.pol)
    res = cw.translate(
        *(m[:, None] for m in to_basis.zms),
        *basis.zms,
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
    if isinstance(basis, core.PlaneWaveBasisByComp):
        modetype = "up" if modetype is None else modetype
    kvecs = basis.kvecs(k0, material, modetype)
    res = pw.to_cw(
        *(m[:, None] for m in to_basis.zms), *kvecs, basis.pol, where=where
    ) * pw.translate(
        *kvecs,
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
    if isinstance(basis, core.PlaneWaveBasisByComp):
        modetype = "up" if modetype is None else modetype
    kvecs = basis.kvecs(k0, material, modetype)
    if isinstance(to_basis, core.PlaneWaveBasisByComp):
        modetype = "up" if modetype is None else modetype
    kx, ky, kz = to_basis.kvecs(k0, material, modetype)
    res = np.array(
        where
        & (kx[:, None] == kvecs[0])
        & (ky[:, None] == kvecs[1])
        & (kz[:, None] == kvecs[2]),
        int,
    )
    return core.PhysicsArray(
        res, basis=(to_basis, basis), k0=k0, material=material, modetype=modetype
    )


def expand(
    basis, modetype=None, *, k0=None, material=Material(), poltype=None, where=True
):
    """Expansion matrix.

    Expand the modes from one basis set to another basis set. If applicable the modetype
    can also be changed, like for spherical and cylindrical waves from `singular` to
    `regular`. Not all expansions are available, only those that result in a discrete
    set of modes. For example, plane waves can be expanded in spherical waves, but the
    opposite transformation generally requires a continuous spectrum (an integral) over
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
            :ref:`params:Polarizations`.
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
                basis, to_basis, to_modetype, k0, material, modetype, where
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

    _FUNC = staticmethod(expand)

    def __init__(self, basis, modetype=None, *, isinv=False):
        args = (basis,) if modetype is None else (basis, modetype)
        super().__init__(*args, isinv=isinv)

    def __call__(self, **kwargs):
        if self.isinv:
            return self._call_inv(**kwargs)
        args = list(self._args)
        if "basis" in kwargs and not isinstance(args[0], tuple):
            args[0] = (args[0], kwargs.pop("basis"))
        if "modetype" in kwargs and len(args) > 1 and not isinstance(args[1], tuple):
            args[1] = (args[1], kwargs.pop("modetype"))
        return self.FUNC(*args, **kwargs)

    def _call_inv(self, **kwargs):
        args = list(self._args)
        if "basis" in kwargs and not isinstance(args[0], tuple):
            args[0] = (kwargs.pop("basis"), args[0])
        if "modetype" in kwargs and len(args) > 1 and not isinstance(args[1], tuple):
            args[1] = (kwargs.pop("modetype"), args[1])
        return self.FUNC(*args, **kwargs)

    def get_kwargs(self, obj, dim=-1):
        kwargs = super().get_kwargs(obj, dim)
        for name in ("basis", "modetype"):
            val = getattr(obj, name, None)
            if isinstance(val, tuple):
                val = val[dim]
            if val is not None:
                kwargs[name] = val
        return kwargs

    @property
    def inv(self):
        """Inverse expansion.

        The inverse transformation is not available for all transformations.
        """
        if len(self._args) == 1:
            basis, modetype = self._args[0], None
        else:
            basis, modetype = self._args
        if isinstance(basis, tuple):
            basis = tuple(basis[::-1])
        if isinstance(modetype, tuple):
            modetype = tuple(modetype[::-1])
        return type(self)(basis, modetype, isinv=not self.isinv)


def _swl_expand(basis, to_basis, eta, k0, kpar, lattice, material, poltype, where):
    """Expand spherical waves in a lattice."""
    ks = k0 * material.nmp
    if lattice.dim == 3:
        try:
            length = len(kpar)
        except TypeError:
            length = 1
        if length == 1:
            lattice = Lattice(lattice, "z")
        elif length == 2:
            lattice = Lattice(lattice, "xy")
        elif length == 3:
            # Last attempt to determine the dimension of the sum
            if np.isnan(kpar[2]):
                lattice = Lattice(lattice, "xy")
            elif np.isnan(kpar[0]) and np.isnan(kpar[1]):
                lattice = Lattice(lattice, "z")
    kpar = WaveVector(kpar)
    if lattice.dim == 1:
        x = kpar[2]
    elif lattice.dim == 2:
        x = kpar[:2]
    else:
        x = kpar[:]
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
    kpar = WaveVector(kpar)
    res = sw.periodic_to_cw(
        *(m[:, None] for m in to_basis.zms),
        *basis.lms,
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
    if modetype is None and isinstance(to_basis, core.PlaneWaveBasisByComp):
        modetype = "up"
    kpar = WaveVector(kpar)
    kvecs = to_basis.kvecs(k0, material, modetype)
    transl = pw.translate(
        *(b[:, None] for b in kvecs),
        -basis.positions[:, 0],
        -basis.positions[:, 1],
        -basis.positions[:, 2],
    )
    res = transl[:, basis.pidx] * sw.periodic_to_pw(
        *(b[:, None] for b in kvecs),
        to_basis.pol[:, None],
        *basis.lms,
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


def _cwl_expand(basis, to_basis, eta, k0, kpar, lattice, material, where):
    """Expand cylindrical waves in a lattice."""
    ks = material.ks(k0)
    alignment = (
        "x" if not isinstance(lattice, Lattice) and np.size(lattice) == 1 else None
    )
    lattice = Lattice(lattice, alignment)
    if lattice.dim == 3:
        try:
            length = len(kpar)
        except TypeError:
            length = 1
        if length == 1:
            lattice = Lattice(lattice, "x")
        elif length == 2:
            lattice = Lattice(lattice, "xy")
        elif length == 3:
            # Last attempt to determine the dimension of the sum
            if np.isnan(kpar[1]):
                lattice = Lattice(lattice, "x")
                kpar = WaveVector(kpar, "x")
            else:
                lattice = Lattice(lattice, "xy")
    if lattice.dim == 1:
        kpar = WaveVector(kpar, "x")
        x = kpar[0]
    elif lattice.dim == 2:
        kpar = WaveVector(kpar)
        x = kpar[:2]
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
    for b in (to_basis, basis):
        if b.kpar is not None:
            kpar = kpar & b.kpar
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
    if modetype is None and isinstance(to_basis, core.PlaneWaveBasisByComp):
        modetype = "up"
    if len(kpar) == 1:
        kpar = WaveVector(kpar, alignment="x")
    kvecs = to_basis.kvecs(k0, material, modetype)
    transl = pw.translate(
        *(b[:, None] for b in kvecs),
        -basis.positions[:, 0],
        -basis.positions[:, 1],
        -basis.positions[:, 2],
    )
    res = transl[:, basis.pidx] * cw.periodic_to_pw(
        *(b[:, None] for b in kvecs),
        to_basis.pol[:, None],
        *basis.zms,
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
    basis=None,
    modetype=None,
    *,
    eta=0,
    k0=None,
    material=Material(),
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
            :class:`~treams.PlaneWaveBasisByComp`.
        poltype (str, optional): Polarization, see also
            :ref:`params:Polarizations`.
        where (array-like, bool, optional): Only evaluate parts of the expansion matrix,
            the give array must have a shape that matches the output shape.
    """
    if isinstance(basis, (tuple, list)):
        to_basis, basis = basis
    else:
        to_basis = basis
    if lattice is None and to_basis.lattice is not None:
        lattice = to_basis.lattice
    if not isinstance(lattice, Lattice) and np.size(lattice) == 1:
        alignment = "x" if isinstance(basis, core.CylindricalWaveBasis) else "z"
    elif isinstance(to_basis, core.PlaneWaveBasis) and isinstance(
        basis, core.CylindricalWaveBasis
    ):
        alignment = "x"
    else:
        alignment = None
    lattice = Lattice(lattice, alignment)
    if kpar is None:
        if basis.kpar is None:
            kpar = to_basis.kpar
        else:
            kpar = basis.kpar
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
            return _cwl_expand(basis, to_basis, eta, k0, kpar, lattice, material, where)
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

    _FUNC = staticmethod(expandlattice)

    def __init__(self, lattice=None, kpar=None, basis=None, modetype=None, eta=0):
        # if lattice is None:
        #     if basis is None:
        #         raise ValueError("'basis' or 'lattice' needs to be defined")
        #     elif basis.lattice is None:
        #         raise ValueError("'basis.lattice' or 'lattice' needs to be defined")
        super().__init__(lattice, kpar, basis, modetype, eta)

    def __call__(self, **kwargs):
        if self.isinv:
            return self._call_inv(**kwargs)
        args = list(self._args)
        if "basis" in kwargs:
            if args[2] is None:
                args[2] = kwargs.pop("basis")
            else:
                args[2] = (args[2], kwargs.pop("basis"))
        if "modetype" in kwargs and args[3] is None:
            args[3] = kwargs.pop("modetype")
        for i, name in enumerate(("lattice", "kpar")):
            if name in kwargs and args[i] is None:
                args[i] = kwargs.pop(name)
        if args[4] != 0:
            kwargs["eta"] = args[4]
        args = args[:4]
        return self.FUNC(*args, **kwargs)

    def get_kwargs(self, obj, dim=-1):
        kwargs = super().get_kwargs(obj, dim)
        kwargs.pop("modetype", None)
        for name in ("basis", "lattice", "kpar"):
            val = getattr(obj, name, None)
            if isinstance(val, tuple):
                val = val[dim]
            if val is not None:
                kwargs[name] = val
        return kwargs

    @property
    def inv(self):
        """Inverse expansion for periodic arrangements.

        The inverse transformation is not available.
        """
        raise NotImplementedError


def _pwp_permute(basis, n, k0, material, modetype, poltype):
    """Permute axes in a partial plane wave basis."""
    if material is None:
        raise TypeError("missing definition of 'material'")
    modetype = "up" if modetype is None else modetype
    kvecs = np.array(basis.kvecs(k0, material, modetype))
    where = (kvecs[..., None] == kvecs[:, None, :]).all(0)
    if n == 1:
        res = pw.permute_xyz(
            kvecs[0],
            kvecs[1],
            kvecs[2],
            basis.pol[:, None],
            basis.pol,
            poltype=poltype,
            where=where,
        )
    elif n == 2:
        res = pw.permute_xyz(
            kvecs[0],
            kvecs[1],
            kvecs[2],
            basis.pol[:, None],
            basis.pol,
            poltype=poltype,
            inverse=True,
            where=where,
        )
    else:
        res = np.eye(len(basis))
    res[np.logical_not(where)] = 0
    return core.PhysicsArray(res, basis=(basis.permute(n), basis), poltype=poltype)


def _pwa_permute(basis, n, poltype):
    """Permute axes in a plane wave basis."""
    qx, qy, qz = basis.qx, basis.qy, basis.qz
    where = (qx[:, None] == qx) & (qy[:, None] == qy) & (qz[:, None] == qz)
    if n == 1:
        res = pw.permute_xyz(
            qx, qy, qz, basis.pol[:, None], basis.pol, poltype=poltype, where=where
        )
    elif n == 2:
        res = pw.permute_xyz(
            qx,
            qy,
            qz,
            basis.pol[:, None],
            basis.pol,
            poltype=poltype,
            inverse=True,
            where=where,
        )
    elif n == 0:
        res = np.eye(len(basis))
    res[np.logical_not(where)] = 0
    return core.PhysicsArray(res, basis=(basis.permute(n), basis), poltype=poltype)


def permute(n=1, *, basis, k0=None, material=None, modetype=None, poltype=None):
    """Permutation matrix.

    Permute the axes of a plane wave basis expansion.

    Args:
        n (int, optional): Number of permutations, defaults to 1.
        basis (:class:`~treams.BasisSet` or tuple): Basis set, if it is a tuple of two
            basis sets the output and input modes are taken accordingly, else both sets
            of modes are the same.
        k0 (float, optional): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`params:Polarizations`.
    """
    poltype = config.POLTYPE if poltype is None else poltype
    if n != int(n):
        raise ValueError("'n' must be integer")
    n = n % 3
    if isinstance(basis, core.PlaneWaveBasisByComp):
        return _pwp_permute(basis, n, k0, material, modetype, poltype)
    if isinstance(basis, core.PlaneWaveBasisByUnitVector):
        return _pwa_permute(basis, n, poltype)
    raise TypeError("invalid basis")


class Permute(Operator):
    """Axes permutation matrix.

    When called as attribute of an object it returns a suitable transformation matrix to
    permute the axis definitions of plane waves. See also :func:`permute`.
    """

    _FUNC = staticmethod(permute)

    def __init__(self, n=1, *, isinv=False):
        super().__init__(n, isinv=isinv)

    @property
    def inv(self):
        return type(self)(*self._args, isinv=not self.isinv)

    def _call_inv(self, **kwargs):
        if "basis" in kwargs:
            kwargs["basis"] = kwargs["basis"].permute(self._args[0])
        return self.FUNC(-self._args[0], **kwargs)


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


def _pw_efield(r, basis, k0, material, modetype, poltype):
    """Electric field of plane waves."""
    res = None
    kvecs = basis.kvecs(k0, material, modetype)
    if poltype == "helicity":
        res = sc.vpw_A(*kvecs, r[..., 0], r[..., 1], r[..., 2], basis.pol)
    elif poltype == "parity":
        res = (1 - basis.pol[:, None]) * sc.vpw_M(
            *kvecs, r[..., 0], r[..., 1], r[..., 2]
        ) + basis.pol[:, None] * sc.vpw_N(*kvecs, r[..., 0], r[..., 1], r[..., 2])
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(res)
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    return res


def efield(r, *, basis, k0, material=Material(), modetype=None, poltype=None):
    """Electric field.

    The resulting matrix maps the electric field coefficients of the given basis to the
    electric field in Cartesian coordinates.

    Args:
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`params:Polarizations`.
    """
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_efield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_efield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    if isinstance(basis, core.PlaneWaveBasis):
        if isinstance(basis, core.PlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _pw_efield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    raise TypeError("invalid basis")


class FieldOperator(Operator):
    def __init__(self, r):
        super().__init__(r)

    @property
    def inv(self):
        """Inverse transformation of an electric field to modes.

        The inverse transformation is not available.
        """
        raise NotImplementedError


class EField(FieldOperator):
    """Electric field evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`efield`.
    """

    _FUNC = staticmethod(efield)


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


def _pw_hfield(r, basis, k0, material, modetype, poltype):
    """Magnetic field of plane waves."""
    res = None
    kvecs = basis.kvecs(k0, material, modetype)
    if poltype == "helicity":
        res = (2 * basis.pol[:, None] - 1) * sc.vpw_A(
            *kvecs, r[..., 0], r[..., 1], r[..., 2], basis.pol
        )
    elif poltype == "parity":
        res = basis.pol[:, None] * sc.vpw_M(*kvecs, r[..., 0], r[..., 1], r[..., 2]) + (
            1 - basis.pol[:, None]
        ) * sc.vpw_N(*kvecs, r[..., 0], r[..., 1], r[..., 2])
    res *= -1j / material.impedance
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(res)
    res.ann[-2]["basis"] = basis
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    return res


def hfield(r, *, basis, k0, material=Material(), modetype=None, poltype=None):
    r"""Magnetic field.

    The resulting matrix maps the electric field coefficients of the given basis to the
    magnetic field in Cartesian coordinates.

    The magnetic field is given in units of :math:`\frac{1}{Z_0} [\boldsymbol E]`.

    Args:
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`params:Polarizations`.
    """
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_hfield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_hfield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    if isinstance(basis, core.PlaneWaveBasis):
        if isinstance(basis, core.PlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _pw_hfield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    raise TypeError("invalid basis")


class HField(FieldOperator):
    """Magnetic field evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`hfield`.
    """

    _FUNC = staticmethod(hfield)


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


def _pw_dfield(r, basis, k0, material, modetype, poltype):
    """Displacement field of partial plane waves."""
    res = _pw_efield(r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol][:, None] / material.impedance
    else:
        res *= material.epsilon
    return res


def dfield(r, *, basis, k0, material=Material(), modetype=None, poltype=None):
    r"""Displacement field.

    The resulting matrix maps the electric field coefficients of the given basis to the
    displacement field in Cartesian coordinates.

    The displacement field is given in units of :math:`\epsilon_0 [\boldsymbol E]`.

    Args:
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`params:Polarizations`.
    """
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_dfield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_dfield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    if isinstance(basis, core.PlaneWaveBasis):
        if isinstance(basis, core.PlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _pw_dfield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    raise TypeError("invalid basis")


class DField(FieldOperator):
    """Displacement field evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`dfield`.
    """

    _FUNC = staticmethod(dfield)


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


def _pw_bfield(r, basis, k0, material, modetype, poltype):
    """Magnetic flux density of partial plane waves."""
    res = _pw_hfield(r, basis, k0, material, modetype, poltype)
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
            :ref:`params:Polarizations`.
    """
    material = Material(material)
    poltype = config.POLTYPE if poltype is None else poltype
    r = np.asanyarray(r)
    r = r[..., None, :]
    if isinstance(basis, core.SphericalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _sw_bfield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_bfield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    if isinstance(basis, core.PlaneWaveBasis):
        if isinstance(basis, core.PlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _pw_bfield(r, basis, k0, material, modetype, poltype).swapaxes(-1, -2)
    raise TypeError("invalid basis")


class BField(FieldOperator):
    """Magnetic flux density evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`bfield`.
    """

    _FUNC = staticmethod(bfield)


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


def _pw_gfield(pol, r, basis, k0, material, modetype, poltype):
    """Riemann-Silberstein field G of plane waves."""
    res = None
    kvecs = basis.kvecs(k0, material, modetype)
    if poltype == "helicity":
        res = (basis.pol[:, None] == pol) * sc.vpw_A(
            *kvecs, r[..., 0], r[..., 1], r[..., 2], basis.pol
        )
    elif poltype == "parity":
        res = (1 + 2 * (pol - 1) * basis.pol[:, None]) * sc.vpw_M(
            *kvecs, r[..., 0], r[..., 1], r[..., 2]
        ) + (2 * pol - 1 - 2 * (pol - 1) * basis.pol[:, None]) * sc.vpw_N(
            *kvecs, r[..., 0], r[..., 1], r[..., 2]
        )
    if res is None:
        raise ValueError("invalid parameters")
    res = util.AnnotatedArray(res)
    res.ann[-2]["basis"] = basis
    res.ann[-2]["poltype"] = poltype
    return res


def gfield(pol, r, *, basis, k0, material=Material(), modetype=None, poltype=None):
    """Riemann-Silberstein field G.

    The resulting matrix maps the electric field coefficients of the given basis to the
    Riemann-Silberstein field G in Cartesian coordinates.

    Args:
        pol (int): Type of the Riemann-Silberstein field, must be 1 or -1. In analogy to
            to polarizations for helicity mode, 0 is also treated as -1.
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`params:Polarizations`.
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
        return _sw_gfield(pol, r, basis, k0, material, modetype, poltype).swapaxes(
            -1, -2
        )
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_gfield(pol, r, basis, k0, material, modetype, poltype).swapaxes(
            -1, -2
        )
    if isinstance(basis, core.PlaneWaveBasis):
        if isinstance(basis, core.PlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _pw_gfield(pol, r, basis, k0, material, modetype, poltype).swapaxes(
            -1, -2
        )
    raise TypeError("invalid basis")


class GField(FieldOperator):
    """Riemann-Silberstein field G evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`gfield`.
    """

    _FUNC = staticmethod(gfield)


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


def _pw_ffield(pol, r, basis, k0, material, modetype, poltype):
    """Riemann-Silberstein field F of partial plane waves."""
    res = _pw_gfield(pol, r, basis, k0, material, modetype, poltype)
    if poltype == "helicity":
        res *= material.nmp[basis.pol, None] / material.n
    return res


def ffield(pol, r, *, basis, k0, material=Material(), modetype=None, poltype=None):
    """Riemann-Silberstein field F.

    The resulting matrix maps the electric field coefficients of the given basis to the
    Riemann-Silberstein field F in Cartesian coordinates.

    Args:
        pol (int): Type of the Riemann-Silberstein field, must be 1 or -1. In analogy to
            to polarizations for helicity mode, 0 is also treated as -1.
        r (array-like): Evaluation points
        basis (:class:`~treams.BasisSet`): Basis set.
        k0 (float): Wave number.
        material (:class:`~treams.Material` or tuple, optional): Material parameters.
        modetype (str, optional): Wave mode.
        poltype (str, optional): Polarization, see also
            :ref:`params:Polarizations`.
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
        return _sw_ffield(pol, r, basis, k0, material, modetype, poltype).swapaxes(
            -1, -2
        )
    if isinstance(basis, core.CylindricalWaveBasis):
        modetype = "regular" if modetype is None else modetype
        return _cw_ffield(pol, r, basis, k0, material, modetype, poltype).swapaxes(
            -1, -2
        )
    if isinstance(basis, core.PlaneWaveBasis):
        if isinstance(basis, core.PlaneWaveBasisByComp):
            modetype = "up" if modetype is None else modetype
        return _pw_ffield(pol, r, basis, k0, material, modetype, poltype).swapaxes(
            -1, -2
        )
    raise TypeError("invalid basis")


class FField(FieldOperator):
    """Riemann-Silberstein field F evaluation matrix.

    When called as attribute of an object it returns a suitable matrix to evaluate field
    coefficients at specified points. See also :func:`ffield`.
    """

    _FUNC = staticmethod(ffield)
