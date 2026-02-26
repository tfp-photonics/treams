"""Loading and storing data.

Most functions rely on at least one of the external packages `h5py` or `gmsh`.
"""
import os
import sys
import uuid as _uuid
from importlib.metadata import version
import warnings
import numpy as np

import treams

try:
    import h5py
except ImportError:
    h5py = None

LENGTHS = {
    "ym": 1e-24,
    "zm": 1e-21,
    "am": 1e-18,
    "fm": 1e-15,
    "pm": 1e-12,
    "nm": 1e-9,
    "um": 1e-6,
    "µm": 1e-6,
    "mm": 1e-3,
    "cm": 1e-2,
    "dm": 1e-1,
    "m": 1,
    "dam": 1e1,
    "hm": 1e2,
    "km": 1e3,
    "Mm": 1e6,
    "Gm": 1e9,
    "Tm": 1e12,
    "Pm": 1e15,
    "Em": 1e18,
    "Zm": 1e21,
    "Ym": 1e24,
}

INVLENGTHS = {
    r"ym^{-1}": 1e24,
    r"zm^{-1}": 1e21,
    r"am^{-1}": 1e18,
    r"fm^{-1}": 1e15,
    r"pm^{-1}": 1e12,
    r"nm^{-1}": 1e9,
    r"um^{-1}": 1e6,
    r"µm^{-1}": 1e6,
    r"mm^{-1}": 1e3,
    r"cm^{-1}": 1e2,
    r"dm^{-1}": 1e1,
    r"m^{-1}": 1,
    r"dam^{-1}": 1e-1,
    r"hm^{-1}": 1e-2,
    r"km^{-1}": 1e-3,
    r"Mm^{-1}": 1e-6,
    r"Gm^{-1}": 1e-9,
    r"Tm^{-1}": 1e-12,
    r"Pm^{-1}": 1e-15,
    r"Em^{-1}": 1e-18,
    r"Zm^{-1}": 1e-21,
    r"Ym^{-1}": 1e-24,
}

FREQUENCIES = {
    "yHz": 1e-24,
    "zHz": 1e-21,
    "aHz": 1e-18,
    "fHz": 1e-15,
    "pHz": 1e-12,
    "nHz": 1e-9,
    "uHz": 1e-6,
    "µHz": 1e-6,
    "mHz": 1e-3,
    "cHz": 1e-2,
    "dHz": 1e-1,
    "s": 1,
    "daHz": 1e1,
    "hHz": 1e2,
    "kHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
    "THz": 1e12,
    "PHz": 1e15,
    "EHz": 1e18,
    "ZHz": 1e21,
    "YHz": 1e24,
    r"ys^{-1}": 1e24,
    r"zs^{-1}": 1e21,
    r"as^{-1}": 1e18,
    r"fs^{-1}": 1e15,
    r"ps^{-1}": 1e12,
    r"ns^{-1}": 1e9,
    r"us^{-1}": 1e6,
    r"µs^{-1}": 1e6,
    r"ms^{-1}": 1e3,
    r"cs^{-1}": 1e2,
    r"ds^{-1}": 1e1,
    r"s^{-1}": 1,
    r"das^{-1}": 1e-1,
    r"hs^{-1}": 1e-2,
    r"ks^{-1}": 1e-3,
    r"Ms^{-1}": 1e-6,
    r"Gs^{-1}": 1e-9,
    r"Ts^{-1}": 1e-12,
    r"Ps^{-1}": 1e-15,
    r"Es^{-1}": 1e-18,
    r"Zs^{-1}": 1e-21,
    r"Ys^{-1}": 1e-24,
}


def mesh_spheres(radii, positions, model, meshsize=None, meshsize_boundary=None):
    """Generate a mesh of multiple spheres.

    This function facilitates generating a mesh for a cluster spheres using gmsh. It
    requires the package `gmsh` to be installed.

    Examples:
        >>> import gmsh
        >>> import treams.io
        >>> gmsh.initialize()
        >>> gmsh.model.add("spheres")
        >>> treams.io.mesh_spheres([1, 2], [[0, 0, 2], [0, 0, -2]], gmsh.model)
        <class 'gmsh.model'>
        >>> gmsh.write("spheres.msh")  # doctest: +SKIP
        >>> gmsh.finalize()  # doctest: +SKIP

    Args:
        radii (float, array_like): Radii of the spheres.
        positions (float, (N, 3)-array): Positions of the spheres.
        model (gmsh.model): Gmsh model to modify.
        meshsize (float, optional): Mesh size, if None a fifth of the largest radius is
            used.
        meshsize (float, optional): Mesh size of the surfaces, if left empty it is set
            equal to the general mesh size.

    Returns:
        gmsh.model
    """
    if meshsize is None:
        meshsize = np.max(radii) * 0.2
    if meshsize_boundary is None:
        meshsize_boundary = meshsize

    spheres = []
    for i, (radius, position) in enumerate(zip(radii, positions)):
        tag = i + 1
        model.occ.addSphere(*position, radius, tag)
        spheres.append((3, tag))

    model.occ.synchronize()
    for _, tag in spheres:
        model.addPhysicalGroup(3, [tag], tag)
        # Add surfaces for other mesh formats like stl, ...
        model.addPhysicalGroup(2, [tag], tag)

    model.mesh.setSize(model.getEntities(0), meshsize)
    model.mesh.setSize(
        model.getBoundary(spheres, False, False, True), meshsize_boundary
    )
    return model


def _translate_polarizations(pols, poltype=None):
    """Translate the polarization index into words.

    The indices 0 and 1 are translated to "negative" and "positive", respectively, when
    helicity modes are chosen. For parity modes they are translated to "magnetic" and
    "electric".

    Args:
        pols (int, array_like): Array of indices 0 and 1.
        poltype (str, optional): Polarization type (:ref:`polarizations.Polarizations`).

    Returns:
        list[str]
    """
    poltype = treams.config.POLTYPE if poltype is None else poltype
    if poltype == "helicity":
        names = ["negative", "positive"]
    elif poltype == "parity":
        names = ["magnetic", "electric"]
    else:
        raise ValueError("unrecognized poltype")
    return [names[i] for i in pols]


def _translate_polarizations_inv(pols):
    """Translate the polarization into indices.

    This function is the inverse of :func:`treams.io._translate_polarizations`. The
    words "negative" and "minus" are translated to 0 and the words "positive" and "plus"
    are translated to 1, if helicity modes are chosen. For parity modes, modes
    "magnetic" or "te" are translated to 0 and the modes "electric" or "tm" to 1.

    Args:
        pols (string, array_like): Array of strings

    Returns:
        tuple[list[int], str]
    """
    helicity = {"plus": 1, "positive": 1, "minus": 0, "negative": 0}
    parity = {"te": 0, "magnetic": 0, "tm": 1, "electric": 1, "M": 0, "N": 1}
    if pols[0].decode() in helicity:
        dct = helicity
        poltype = "helicity"
    elif pols[0].decode() in parity:
        dct = parity
        poltype = "parity"
    else:
        raise ValueError(f"unrecognized polarization '{pols[0].decode()}'")
    return [dct[i.decode()] for i in pols], poltype


def _remove_leading_ones(shape):
    while len(shape) > 0 and shape[0] == 1:
        shape = shape[1:]
    return shape


def _collapse_dims(arr):
    for i in range(arr.ndim):
        test = arr[(slice(None),) * i + (slice(1),)]
        if np.all(arr == test):
            arr = test
    return arr.reshape(_remove_leading_ones(arr.shape))

def _save_mesh(comp_group, mesh, lunit="nm", encoding="utf-8"):
    """
    Store one shared mesh under /computation.

    mesh can be:
      - dict: {"mesh.xyz": "<text>", ...}  
      - str path to a text file: "---.xyz" -> stored as "mesh.xyz"
    """
    root = comp_group.file
    # dict -> /computation/mesh/<key>
    if isinstance(mesh, dict):
        mg = comp_group.require_group("mesh")
        mg.attrs["unit"] = lunit
        for name, text in mesh.items():
            if name in mg:
                del mg[name]
            mg[name] = text   
        target_path = mg.name       
     
    # path -> /computation/mesh.xyz
    elif isinstance(mesh, str) and os.path.exists(mesh):
        ext = os.path.splitext(mesh)[1] 
        dset_name = "mesh" + ext
        with open(mesh, "r", encoding=encoding) as f:
            text = f.read()
        if dset_name in comp_group:
            del comp_group[dset_name]
        comp_group[dset_name] = text
        comp_group[dset_name].attrs["unit"] = lunit
        target_path = comp_group[dset_name].name

    else:
        raise ValueError(f"invalid mesh type: {type(mesh)}")        
    existing = root.get("mesh", getlink=True)
    if existing is None:
        root["mesh"] = h5py.SoftLink(target_path)
    elif isinstance(existing, h5py.SoftLink):
        del root["mesh"]
        root["mesh"] = h5py.SoftLink(target_path)


def _save_scatterers_hdf5(h5file, scatterers, lunit):
    if isinstance(scatterers, dict):
        scatterers = [scatterers]

    for i, sc in enumerate(scatterers):
        gname = "scatterer" if len(scatterers) == 1 else f"scatterer_{i}"
        sg = h5file.require_group(gname)
        _name_descr_kw(
            sg,
            sc.get("name", ""),
            sc.get("description", ""),
            sc.get("keywords", ""),
        )

        # material
        mat = sg.require_group("material")
        m = sc.get("material", {})
        _name_descr_kw(
            mat,
            m.get("name", ""),
            m.get("description", ""),
            m.get("keywords", ""),
        )
        for key in ("relative_permittivity", "relative_permeability", "chirality"):
            if key in m:
                mat[key] = m[key]

        # geometry
        geo = sg.require_group("geometry")
        g = sc.get("geometry", {})
        _name_descr_kw(
            geo,
            g.get("name", ""),
            g.get("description", ""),
            g.get("keywords", ""),
        )
        if "shape" in g:
            geo.attrs["shape"] = g["shape"]
        geo.attrs["unit"] = g.get("unit", lunit)

        for k, v in g.items():
            if k in ("shape", "unit", "name", "description", "keywords"):
                continue
            geo[k] = v
            if hasattr(geo[k], "attrs"):
                geo[k].attrs["unit"] = geo.attrs["unit"]
                
        if "position" in sc:
            geo["position"] = sc["position"]
            geo["position"].attrs["unit"] = geo.attrs["unit"]

def _save_computation_hdf5(h5file, computation, lunit):
    comp = h5file.require_group("computation")
    _name_descr_kw(
        comp,
        computation.get("name", ""),
        computation.get("description", ""),
        computation.get("keywords", ""),
    )
    if "method" in computation:
        comp.attrs["method"] = computation["method"]

    software = computation.get("software", "")
    if software == "":
        software = (
            f"python={sys.version.split()[0]}, "
            f"h5py={version('h5py')}, "
            f"treams={version('treams')}, "
            f"numpy={version('numpy')}"
        )
    comp.attrs["software"] = software

    mesh = computation.get("mesh", None)
    if mesh is not None:
        _save_mesh(comp, mesh, lunit)
    else:
        kw = str(comp.attrs.get("keywords", ""))
        if "semi-analytical" not in kw:
            issues.append(
                "Computation keywords do not include 'semi-analytical' but no mesh was stored. "
                "This is against tmat.h5 v1 and will not be accepted by the T-matrix database."
            )

    # reproducibility files
    files = computation.get("files", None)
    if files:
        fg = comp.require_group("files")
        for item in files:
            if isinstance(item, str):
                path = item
                name = os.path.basename(path)
            else:
                path = item["path"]
                name = item.get("name", os.path.basename(path))
            with open(path, "r", encoding="utf-8") as f:
                fg[name] = f.read()

def save_hdf5(
    h5file,
    tms,
    name="",
    description="",
    keywords="",
    embedding_group=None,
    embedding_name="",
    embedding_description="",
    embedding_keywords="",
    uuid=None,
    uuid_version=4,
    scatterers=None, 
    computation=None,
    lunit="nm"
):
    """Save a set of T-matrices in a HDF5 file.

    With an open and writeable datafile, this function writes the main datasets 
    needed to store T-matrix data in the tmat.h5 v1 layout.
    Additional metadata can be added by the user after calling this function. 

    Args:
    h5file (h5py.Group): A HDF5 file opened with h5py.
    tms (TMatrix, array_like): Array of T-matrix instances.
    name (str): Name to add to the file as attribute.
    description (str): Description to add to file as attribute.
    keywords (str): Keywords to add to file as attribute, e.g.,
        "mirrorxyz, czinfinity, passive, reciprocal".

    embedding_group (h5py.Group, optional): Target group to store embedding data.
        For legacy/compatibility only. The tmat.h5 v1 standard requires the
        embedding to be stored under ``/embedding``.

    embedding_name (str, optional): Name of the embedding material. To match
        existing entries in the T-matrix database, use the same naming style as
        already used there (comma-separated identifier and common name), e.g.
        "Air, Vacuum", "H2O, Water", "C2H5OH, Ethanol", or "Custom". Same for the
        material of the scatterers.
    embedding_description (str, optional): Description of the embedding material.
    embedding_keywords (str, optional): Keywords for the embedding material.

    uuid (bytes, optional): UUID of the file (legacy compatibility, not required
        by the standard).
    uuid_version (int, optional): UUID version.

    scatterers (dict or list[dict], optional): Scatterer metadata to be written
        to ``/scatterer`` or ``/scatterer_<i>`` groups. Provide multiple scatterers
        only if a single T-matrix describes an arrangement of objects. This information is required by
        the v1 standard, but kept optional here for legacy compatibility.
        Do not combine unrelated single-object T-matrices (e.g. from a geometry parameter sweep)
        in one file.

    computation (dict, optional): Computation metadata to be written to the
        ``/computation`` group (recommended to include '/computation/files'). This information is required by
    the v1 standard, but kept optional here for legacy compatibility.
    lunit (str, optional): Length unit used for the positions and (as inverse)
        for the wave number.
    """
    issues = []
    tms_arr = np.array(tms)
    if tms_arr.dtype == object:
        raise ValueError("can only save T-matrices of the same size")
    tms_obj = np.empty(tms_arr.shape[:-2], object)
    tms_obj[:] = tms
    tm = tms_obj.flat[0]
    basis = tm.basis
    poltype = tm.poltype

    k0s = np.zeros((tms_arr.shape)[:-2])
    epsilon = np.zeros((tms_arr.shape)[:-2], complex)
    mu = np.zeros((tms_arr.shape)[:-2], complex)
    if poltype == "helicity":
        kappa = np.zeros((tms_arr.shape)[:-2], complex)

    for i, tm in enumerate(tms_obj.flat):
        if poltype != tm.poltype or basis != tm.basis:
            raise ValueError(
                "incompatible T-matrices: mixed poltypes or different bases"
            )
        k0s.flat[i] = tm.k0
        epsilon.flat[i] = tm.material.epsilon
        mu.flat[i] = tm.material.mu
        if poltype == "helicity":
            kappa.flat[i] = tm.material.kappa

    h5file["tmatrix"] = tms_arr
    if uuid is not None:
        h5file["uuid"] = uuid
        h5file["uuid"].attrs["version"] = uuid_version
    _name_descr_kw(h5file, name, description, keywords)
    h5file["angular_vacuum_wavenumber"] = _collapse_dims(k0s)
    h5file["angular_vacuum_wavenumber"].attrs["unit"] = lunit + r"^{-1}"
    h5file["modes/l"] = basis.l
    h5file["modes/m"] = basis.m
    h5file["modes/polarization"] = _translate_polarizations(basis.pol, poltype)
    if any(basis.pidx != 0):
        h5file["modes/index"] = basis.pidx
    if not np.array_equiv(basis.positions, [[0, 0, 0]]):
        h5file["modes/positions"] = basis.positions
        h5file["modes/positions"].attrs["unit"] = lunit


    embedding_group = h5file.require_group("embedding") if embedding_group is None else h5file.require_group(embedding_group.lstrip("/"))

    if embedding_group.name != "/embedding":
        issues.append(
            f"Embedding group path is '{embedding_group.name}', expected /embedding. "
            "File is not compliant with tmat.h5 v1 standard and will not be accepted by the T-matrix database."
        )

    embedding_group["relative_permittivity"] = _collapse_dims(epsilon)
    embedding_group["relative_permeability"] = _collapse_dims(mu)
    if poltype == "helicity":
        embedding_group["chirality"] = _collapse_dims(kappa)
    _name_descr_kw(
        embedding_group, embedding_name, embedding_description, embedding_keywords
    )

    if computation is not None:
        _save_computation_hdf5(h5file, computation, lunit)
    else:
        issues.append(
            "Missing /computation group metadata (method/software). File is not compliant with tmat.h5 v1 standard."
        )

    if scatterers is not None:
        _save_scatterers_hdf5(h5file, scatterers, lunit)
    else:
        issues.append(
            "Missing /scatterer or /scatterer_* groups. File is not compliant with tmat.h5 v1 standard."
        )
        
    if not issues:
        h5file.attrs["storage_format_version"] = "v1"
        
def _name_descr_kw(fobj, name, description="", keywords=""):
    for key, val in [
        ("name", name),
        ("description", description),
        ("keywords", keywords),
    ]:
        val = str(val)
        if val != "":
            fobj.attrs[key] = val


def _convert_to_k0(x, xtype, xunit, k0unit=r"nm^{-1}"):
    c = 299792458.0
    k0unit = INVLENGTHS[k0unit]
    if xtype == "frequency":
        xunit = FREQUENCIES[xunit]
        return 2 * np.pi * x / c * (xunit / k0unit)
    if xtype == "angular_frequency":
        xunit = FREQUENCIES[xunit]
        return x / c * (xunit / k0unit)
    if xtype == "vacuum_wavelength":
        xunit = LENGTHS[xunit]
        return 2 * np.pi / (x * xunit * k0unit)
    if xtype == "vacuum_wavenumber":
        xunit = INVLENGTHS[xunit]
        return 2 * np.pi * x * (xunit / k0unit)
    if xtype == "angular_vacuum_wavenumber":
        xunit = INVLENGTHS[xunit]
        return x * (xunit / k0unit)
    raise ValueError(f"unrecognized frequency/wavenumber/wavelength type: {xtype}")


def load_hdf5(filename, lunit="nm"):
    """Load a T-matrix stored in a HDF4 file.

    Args:
        filename (str or h5py.Group): Name of the h5py file or a handle to a h5py group.
        lunit (str, optional): Unit of length to be used in the T-matrices.

    Returns:
        np.ndarray[TMatrix]
    """
    if isinstance(filename, h5py.Group):
        return _load_hdf5(filename, lunit)
    with h5py.File(filename, "r") as f:
        return _load_hdf5(f, lunit)


def _load_hdf5(h5file, lunit=None):
    for freq_type in (
        "frequency",
        "angular_frequency",
        "vacuum_wavelength",
        "vacuum_wavenumber",
        "angular_vacuum_wavenumber",
    ):
        if freq_type in h5file:
            ld_freq = h5file[freq_type][()]
            break
    else:
        raise ValueError("no definition of frequency found")
    if "modes/positions" in h5file:
        k0unit = h5file["modes/positions"].attrs.get("unit", lunit) + r"^{-1}"
    else:
        k0unit = lunit + r"^{-1}"
    tms = h5file["tmatrix"][...]
    k0s = _convert_to_k0(ld_freq, freq_type, h5file[freq_type].attrs["unit"], k0unit)

    epsilon = h5file.get("embedding/relative_permittivity", np.array(None))[()]
    mu = h5file.get("embedding/relative_permeability", np.array(None))[()]
    if epsilon is None is mu:
        n = h5file.get("embedding/refractive_index", np.array(1))[...]
        z = h5file.get("embedding/relative_impedance", 1 / n)[...]
        epsilon = n / z
        mu = n * z
    epsilon = 1 if epsilon is None else epsilon
    mu = 1 if mu is None else mu

    kappa = z = h5file.get("embedding/chirality_parameter", np.array(0))[...]

    positions = h5file.get("modes/positions", np.zeros((1, 3)))[...]

    l_inc = h5file.get("modes/l", np.array(None))[...]
    l_inc = h5file.get("modes/l_incident", l_inc)[()]
    m_inc = h5file.get("modes/m", np.array(None))[...]
    m_inc = h5file.get("modes/m_incident", m_inc)[()]
    pol_inc = h5file.get("modes/polarization", np.array(None))[...]
    pol_inc = h5file.get("modes/polarization_incident", pol_inc)[()]
    l_sca = h5file.get("modes/l", np.array(None))[...]
    l_sca = h5file.get("modes/l_scattered", l_sca)[()]
    m_sca = h5file.get("modes/m", np.array(None))[...]
    m_sca = h5file.get("modes/m_scattered", m_sca)[()]
    pol_sca = h5file.get("modes/polarization", np.array(None))[...]
    pol_sca = h5file.get("modes/polarization_scattered", pol_sca)[()]

    if any(x is None for x in (l_inc, l_sca, m_inc, m_sca, pol_inc, pol_sca)):
        raise ValueError("mode definition missing")

    pol_inc, poltype = _translate_polarizations_inv(pol_inc)
    pol_sca, poltype_sca = _translate_polarizations_inv(pol_sca)

    if poltype_sca != poltype:
        raise ValueError("different modetypes")

    pidx_inc = h5file.get("modes/position_index", np.zeros_like(l_inc))[()]
    pidx_inc = h5file.get("modes/positions_index_scattered", pidx_inc)[()]
    pidx_sca = h5file.get("modes/position_index", np.zeros_like(l_sca))[()]
    pidx_sca = h5file.get("modes/positions_index_scattered", pidx_sca)[()]

    shape = tms.shape[:-2]
    k0s = np.broadcast_to(k0s, shape)
    epsilon = np.broadcast_to(epsilon, shape)
    mu = np.broadcast_to(mu, shape)
    kappa = np.broadcast_to(kappa, shape)

    basis_inc = treams.SphericalWaveBasis(
        zip(pidx_inc, l_inc, m_inc, pol_inc), positions
    )
    basis_sca = treams.SphericalWaveBasis(
        zip(pidx_sca, l_sca, m_sca, pol_sca), positions
    )
    basis = basis_inc | basis_sca

    ix_inc = [basis.index(b) for b in basis_inc]
    ix_sca = [[basis.index(b)] for b in basis_sca]

    res = np.empty(shape, object)
    for i in np.ndindex(*shape):
        res[i] = treams.TMatrix(
            np.zeros((len(basis),) * 2, complex),
            k0=k0s[i],
            basis=basis,
            poltype=poltype,
            material=treams.Material(epsilon[i], mu[i], kappa[i]),
        )
        res[i][ix_sca, ix_inc] = tms[i]
    if not res.shape:
        res = res.item()
    return res
