import numpy as np

import ptsa._core as core
import ptsa.config as config
import ptsa.special as sc
from ptsa import cw, pw, sw
from ptsa._lattice import Lattice
from ptsa._material import Material


class Operator:
    def __get__(self, obj, objtype=None):
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
        kwargsa, kwargsb = self._parse_kwargs(kwargs, self._kwargs[-1])
        return self._func(*args, **kwargsa, **kwargsb)

    def inv(self, *args, **kwargs):
        kwargsa, kwargsb = self._parse_kwargs(self._kwargs[0], kwargs)
        return self._func(*args, **kwargsa, **kwargsb)

    def _parse_kwargs(self, kwargsa, kwargsb):
        kwargsa = {**kwargsa}
        kwargsb = {**kwargsb}
        for key in ("basis", "modetype", "poltype"):
            if key in kwargsa and key in kwargsb:
                kwargsa[key] = kwargsa[key], kwargsb.pop(key)
        return kwargsa, kwargsb


def _sw_rotate(phi, theta, psi, basis, to_basis, where):
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
    where = np.logical_and(where, to_basis.pidx[:, None] == basis.pidx)
    res = cw.rotate(
        *(m[:, None] for m in to_basis["kzmp"]), *basis["kzmp"], phi, where=where,
    )
    res[..., ~where] = 0
    return core.PhysicsArray(res, basis=(to_basis, basis))


def _pw_rotate(phi, basis, where):
    # TODO: rotate hints: lattice, kpar
    c1, s1 = np.cos(phi), np.sin(phi)
    r = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    kvecs = r @ np.array([basis.kx, basis.ky, basis.kz])
    modes = zip(*kvecs, basis.pol)
    res = np.eye(len(basis))
    res[..., np.logical_not(where)] = 0
    return core.PhysicsArray(res, basis=(core.PlaneWaveBasis(modes), basis))


def _pwp_rotate(phi, basis, where):
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
    """
    rotate(phi, theta=0, psi=0, *, basis, where=True)
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
    _func = staticmethod(rotate)

    def inv(self, *args, **kwargs):
        args = [-x for x in args]
        if len(args) > 1:
            args[1] = np.pi + args[1]
        if "phi" in kwargs:
            kwargs["phi"] = -kwargs["phi"]
        if "psi" in kwargs:
            kwargs["psi"] = -kwargs["psi"]
        if "theta" in kwargs:
            kwargs["theta"] = np.pi - kwargs["theta"]
        return super().inv(*args, **kwargs)


def _sw_translate(r, basis, to_basis, k0, material, poltype, where):
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
        res, k0=(k0, k0), basis=(to_basis, basis), material=(material, material),
    )


def _pw_translate(r, basis, to_basis, where):
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
    _func = staticmethod(translate)

    def inv(self, *args, **kwargs):
        if len(args) > 0:
            args[0] = np.negative(args[0])
        if "r" in kwargs:
            kwargs["r"] = np.negative(kwargs["r"])
        return super().inv(*args, **kwargs)


def _sw_changepoltype(basis, to_basis, poltype, where):
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
    if to_basis.alignment != basis.alignment:
        raise ValueError("incompatible basis alignments")
    bkx, bky, bpol = to_basis[()]
    where = (bkx[:, None] == basis._kx) & (bky[:, None] == basis._ky) & where
    res = np.zeros_like(where, float)
    res[where] = np.sqrt(0.5)
    res[where & (to_basis.pol[:, None] == basis.pol) & (basis.pol == 0)] = -np.sqrt(0.5)
    return core.PhysicsArray(res, basis=(to_basis, basis), poltype=poltype)


def changepoltype(poltype=None, *, basis, where=True):
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
    _func = staticmethod(changepoltype)

    def _parse_kwargs(self, kwargsa, kwargsb):
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
    if isinstance(basis, core.PlaneWaveBasisPartial):
        modetype = "up" if modetype is None else modetype
        basis_c = basis.complete(k0, material, modetype)
    elif k0 is not None:
        basis_c = basis.complete(k0, material, modetype)
    else:
        basis_c = basis
    res = pw.to_cw(
        *(m[:, None] for m in to_basis["kzmp"]), *basis_c[()], where=where,
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
    basis, modetype=None, *, k0=None, material=Material(), poltype=None, where=True,
):
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
    _func = staticmethod(expand)

    def __call__(self, *args, **kwargs):
        for key in ("basis", "modetype"):
            if len(args) == 0:
                break
            if key in kwargs:
                raise TypeError(f"__call__ got multiple values for argument '{key}'")
            kwargs[key] = args[0]
            args = args[1:]
        return super().__call__(*args, **kwargs)

    def inv(self, *args, **kwargs):
        for key in ("basis", "modetype"):
            if len(args) == 0:
                break
            if key in kwargs:
                raise TypeError(f"__call__ got multiple values for argument '{key}'")
            kwargs[key] = args[0]
            args = args[1:]
        return super().inv(*args, **kwargs)


def _swl_expand(basis, to_basis, eta, k0, kpar, lattice, material, poltype, where):
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
        modetype=(None, "singular"),
        lattice=lattice,
        kpar=kpar,
    )


def expandlattice(
    basis,
    lattice=None,
    kpar=None,
    *,
    eta=0,
    k0=None,
    material=Material(),
    modetype=None,
    poltype=None,
    where=True,
):
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
            return _pw_sw_expand(
                basis, to_basis, k0, kpar, lattice, material, modetype, poltype, where
            )
    if isinstance(basis, core.CylindricalWaveBasis):
        if isinstance(to_basis, core.CylindricalWaveBasis):
            return _cwl_expand(
                basis, to_basis, eta, k0, kpar, lattice, material, poltype, where
            )
        if isinstance(to_basis, core.PlaneWaveBasis):
            return _pw_cw_expand(
                basis, to_basis, k0, lattice, kpar, material, modetype, where
            )
    raise TypeError("invalid basis")


class ExpandLattice(Operator):
    _func = staticmethod(expandlattice)

    def inv(self, *args, **kwargs):
        raise NotImplementedError


def _pwp_permute(basis, n):
    if n != int(n):
        raise ValueError("'n' must be integer")
    n = n % 3
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
    if n != int(n):
        raise ValueError("'n' must be integer")
    n = n % 3
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
    if isinstance(basis, core.PlaneWaveBasisPartial):
        return _pwp_permute(basis, n)
    if isinstance(basis, core.PlaneWaveBasis):
        return _pw_permute(basis, n)
    raise TypeError("invalid basis")


class Permute(Operator):
    _func = staticmethod(permute)


def _sw_efield(r, basis, k0, material, modetype, poltype):
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
    res = core.AnnotatedArray(sc.vsph2car(res, rsph[..., basis.pidx, :]))
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    res.ann[-2]["modetype"] = modetype
    return res


def _cw_efield(r, basis, k0, material, modetype, poltype):
    material = Material(material)
    ks = material.ks(k0)[basis.pol]
    krhos = np.sqrt(ks * ks - basis.kz * basis.kz)
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
    res = core.AnnotatedArray(sc.vcyl2car(res, rcyl[..., basis.pidx, :]))
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["poltype"] = poltype
    res.ann[-2]["modetype"] = modetype
    return res


def _pw_efield(r, basis, poltype):
    res = None
    if poltype == "helicity":
        res = sc.vpw_A(
            basis.kx, basis.ky, basis.kz, r[..., 0], r[..., 1], r[..., 2], basis.pol,
        )
    elif poltype == "parity":
        res = (1 - basis.pol[:, None]) * sc.vpw_M(
            basis.kx, basis.ky, basis.kz, r[..., 0], r[..., 1], r[..., 2],
        ) + basis.pol[:, None] * sc.vpw_N(
            basis.kx, basis.ky, basis.kz, r[..., 0], r[..., 1], r[..., 2],
        )
    if res is None:
        raise ValueError("invalid parameters")
    res = core.AnnotatedArray(res)
    res.ann[-2]["basis"] = basis
    res.ann[-2]["poltype"] = poltype
    return res


def _pwp_efield(r, basis, k0, material, modetype, poltype):
    basis_c = basis.complete(k0, material, modetype)
    res = _pw_efield(r, basis_c, poltype)
    del res.ann[-2]["basis"]
    res.ann[-2]["basis"] = basis
    res.ann[-2]["k0"] = k0
    res.ann[-2]["material"] = material
    res.ann[-2]["modetype"] = modetype
    return res


def efield(r, *, basis, k0=None, material=Material(), modetype=None, poltype=None):
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
    _func = staticmethod(efield)

    def inv(self, *args, **kwargs):
        raise NotImplementedError
