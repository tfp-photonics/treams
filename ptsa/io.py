"""
========================
Loading and storing data
========================

Most functions rely on at least one of the external packages `h5py` or `gmsh`.

.. rubric:: Functions

.. autosummary::
   :toctree: generated/

   mesh_spheres
   save_hdf5
   load_hdf5
"""

import importlib.metadata

import numpy as np

import ptsa

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
    """
    Generate a mesh of multiple spheres

    This function facilitates generating a mesh for a cluster speres using gmsh. It
    requires the package `gmsh` to be installed.

    Examples:
        >>> import gmsh
        >>> gmsh.initialize()
        >>> gmsh.model.add("spheres")
        >>> mesh_spheres([1, 2], [[0, 0, 2], [0, 0, -2]], gmsh.model)
        >>> gmsh.model.write("spheres.msh")
        >>> gmsh.finalize()

    Args:
        radii (float, array_like): Radii of the spheres
        positions (float, (N, 3)-array): Positions of the spheres
        model (gmsh.model): Gmsh model to modify
        meshsize (float, optional): Mesh size, if None a fifth of the largest radius is
            used
        meshsize (float, optional): Mesh size of the surfaces, if left empty it is set
            equal to the general mesh size

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
        model.addPhysicalGroup(3, [i + 1], tag)
        # Add surfaces for other mesh formats like stl, ...
        model.addPhysicalGroup(2, [i + 1], tag)

    model.mesh.setSize(model.getEntities(3), meshsize)
    model.mesh.setSize(
        model.getBoundary(spheres, False, False), meshsize_boundary
    )
    model.mesh.generate()
    return model


def _translate_polarizations(pols, helicity=True):
    """
    Translate the polarization index into words

    The indices 0 and 1 are translated to "negative" and "positive", respectively, when
    helicity modes are chosen. For parity modes they are translated to "magnetic" and
    "electric".

    Args:
        pols (int, array_like): Array of indices 0 and 1
        helicity (bool, optional): Usage of helicity or parity modes

    Returns
        string, array_like
    """
    if helicity:
        names = ["negative", "positive"]
    else:
        names = ["magnetic", "electric"]
    return [names[i] for i in pols]


def _translate_polarizations_inv(pols):
    """
    Translate the polarization into indices

    This function is the inverse of :func:`ptsa.io._translate_polarizations`. The words
    "negative" and "minus" are translated to 0 and the words "positive" and "plus" are
    translated to 1, if helicity modes are chosen. For parity modes, modes "magnetic" or
    "te" are translated to 0 and the modes "electric" or "tm" to 1.

    Args:
        pols (string, array_like): Array of strings
        helicity (bool, optional): Usage of helicity or parity modes

    Returns
        int, array_like
    """
    helicity = {"plus": 1, "positive": 1, "minus": 0, "negative": 0}
    parity = {"te": 0, "magnetic": 0, "tm": 1, "electric": 1}
    if pols[0].decode() in helicity:
        dct = helicity
    elif pols[0].decode() in parity:
        dct = parity
    else:
        raise ValueError(f"unrecognized polarization '{pols[0].decode()}'")
    return [dct[i.decode()] for i in pols], pols[0] in helicity


def save_hdf5(
    datafile,
    tmats,
    name,
    description,
    id=-1,
    unit_length="nm",
    embedding_name="Embedding",
    frequency_axis=None,
):
    """
    Save a set of T-matrices in a HDF5 file

    With an open and writeable datafile, this function stores the main parts of as
    T-matrix in the file. It is left open for the user to add additional metadata.

    Args:
        datafile (h5py.File): a HDF5 file opened with h5py
        tmats (TMatrix, array_like): Array of T-matrix instances
        name (string): Name to add to the T-matrix as attribute
        description (string): Description to add to the T-matrix as attribute
        id (int, optional): Id of the T-matrix, defaults to -1 (no id given)
        unit_length (string, optional): Length unit used for the positions and (as
            inverse) for the wave number
        embedding_name (string, optional): Name of the embedding material, defaults to
            "Embedding"
        frequency_axis (int, optional): Assign one axis of the T-matrices array to
            parametrize a frequency sweep

    Returns:
        h5py.File
    """
    tmats = np.array(tmats)
    shape = tmats.shape
    tmat_first = tmats.ravel()[0]
    helicity = tmat_first.helicity
    with np.nditer(
        [tmats] + [None] * 4,
        ["refs_ok"],
        [["readonly"]] + [["writeonly", "allocate"]] * 4,
        [None, float, complex, complex, complex],
    ) as it:
        for (tmat, k0, epsilon, mu, kappa) in it:
            tmat = tmat.item()
            if (
                np.any(tmat.l != tmat_first.l)
                or np.any(tmat.m != tmat_first.m)
                or np.any(tmat.pol != tmat_first.pol)
                or np.any(tmat.pidx != tmat_first.pidx)
            ):
                raise ValueError("non-matching T-matrix modes")
            if tmat.helicity != helicity:
                raise ValueError("non-matching basis sets")
            if np.any(tmat.positions.shape != tmat.positions.shape):
                raise ValueError("non-matching positions")
            k0[...] = tmat.k0
            epsilon[...] = tmat.epsilon
            mu[...] = tmat.mu
            kappa[...] = tmat.kappa
        k0s, epsilons, mus, kappas = it.operands[1:]

    tms = np.stack([tmat.t for tmat in tmats.flatten()]).reshape(
        shape + tmat_first.t.shape
    )
    positions = np.stack([tmat.positions for tmat in tmats.flatten()]).reshape(
        shape + tmat_first.positions.shape
    )

    if np.all(k0s == k0s.ravel()[0]):
        k0s = k0s.ravel()[0]
    if np.all(epsilons == epsilons.ravel()[0]):
        epsilons = epsilons.ravel()[0]
    if np.all(mus == mus.ravel()[0]):
        mus = mus.ravel()[0]
    if np.all(kappas == kappas.ravel()[0]):
        kappas = kappas.ravel()[0]
    if np.all(positions - tmat_first.positions == 0):
        positions = tmat_first.positions

    if frequency_axis is not None:
        k0slice = (
            (0,) * frequency_axis
            + (slice(k0s.shape[frequency_axis]),)
            + (0,) * (k0s.ndim - frequency_axis - 1)
        )
        k0s = k0s[k0slice]

    return _write_hdf5(
        datafile,
        id,
        name,
        description,
        tms,
        k0s,
        epsilons,
        mus,
        kappas,
        tmat_first.l,
        tmat_first.m,
        tmat_first.pol,
        tmat_first.pidx,
        positions,
        unit_length,
        embedding_name,
        tmat_first.helicity,
        frequency_axis,
    )


def _write_hdf5(
    datafile,
    id,
    name,
    description,
    tms,
    k0s,
    epsilons,
    mus,
    kappas,
    ls,
    ms,
    pols,
    pidxs,
    positions,
    unit_length="nm",
    embedding_name="Embedding",
    helicity=True,
    frequency_axis=None,
):
    datafile.create_dataset("tmatrix", data=tms)
    datafile["tmatrix"].attrs["id"] = id
    datafile["tmatrix"].attrs["name"] = name
    datafile["tmatrix"].attrs["description"] = description

    datafile.create_dataset("k0", data=k0s)
    datafile["k0"].attrs["unit"] = unit_length + r"^{-1}"

    datafile.create_dataset("modes/l", data=ls)
    datafile.create_dataset("modes/m", data=ms)
    datafile.create_dataset(
        "modes/polarization",
        data=_translate_polarizations(pols, helicity=helicity),
    )
    datafile.create_dataset("modes/position_index", data=pidxs)
    datafile["modes/position_index"].attrs[
        "description"
    ] = """
        For local T-matrices each mode is associated with an origin. This index maps
        the modes to an entry in positions.
    """
    datafile.create_dataset("modes/positions", data=positions)
    datafile["modes/positions"].attrs[
        "description"
    ] = """
        The postions of the origins for a local T-matrix.
    """
    datafile["modes/positions"].attrs["unit"] = unit_length

    datafile["modes/l"].make_scale("l")
    datafile["modes/m"].make_scale("m")
    datafile["modes/polarization"].make_scale("polarization")
    datafile["modes/position_index"].make_scale("position_index")

    ndims = len(datafile["tmatrix"].dims)
    datafile["tmatrix"].dims[ndims - 2].label = "Scattered modes"
    datafile["tmatrix"].dims[ndims - 2].attach_scale(datafile["modes/l"])
    datafile["tmatrix"].dims[ndims - 2].attach_scale(datafile["modes/m"])
    datafile["tmatrix"].dims[ndims - 2].attach_scale(datafile["modes/polarization"])
    datafile["tmatrix"].dims[ndims - 2].attach_scale(datafile["modes/position_index"])
    datafile["tmatrix"].dims[ndims - 1].label = "Incident modes"
    datafile["tmatrix"].dims[ndims - 1].attach_scale(datafile["modes/l"])
    datafile["tmatrix"].dims[ndims - 1].attach_scale(datafile["modes/m"])
    datafile["tmatrix"].dims[ndims - 1].attach_scale(datafile["modes/polarization"])
    datafile["tmatrix"].dims[ndims - 1].attach_scale(datafile["modes/position_index"])

    embedding_path = "materials/" + embedding_name.lower()
    datafile.create_group(embedding_path)
    datafile[embedding_path].attrs["name"] = embedding_name
    datafile.create_dataset(embedding_path + "/relative_permittivity", data=epsilons)
    datafile.create_dataset(embedding_path + "/relative_permeability", data=mus)
    if np.any(kappas != 0):
        datafile.create_dataset(embedding_path + "/chirality", data=kappas)

    datafile["embedding"] = h5py.SoftLink("/" + embedding_path)

    if frequency_axis is not None:
        datafile["k0"].make_scale("k0")
        datafile["tmatrix"].dims[frequency_axis].label = "Wave number"
        datafile["tmatrix"].dims[frequency_axis].attach_scale(datafile["k0"])
        if np.ndim(epsilons) != 0:
            datafile[embedding_path + "/relative_permittivity"].dims[
                frequency_axis
            ].label = "Wave number"
            datafile[embedding_path + "/relative_permittivity"].dims[
                frequency_axis
            ].attach_scale(datafile["k0"])
        if np.ndim(mus) != 0:
            datafile[embedding_path + "/relative_permeability"].dims[
                frequency_axis
            ].label = "Wave number"
            datafile[embedding_path + "/relative_permeability"].dims[
                frequency_axis
            ].attach_scale(datafile["k0"])
        if np.ndim(kappas) != 0 and np.any(kappas != 0):
            datafile[embedding_path + "/chirality"].dims[
                frequency_axis
            ].label = "Wave number"
            datafile[embedding_path + "/chirality"].dims[frequency_axis].attach_scale(
                datafile["k0"]
            )
        if np.ndim(positions) > 2:
            datafile["modes/positions"].dims[frequency_axis].label = "Wave number"
            datafile["modes/positions"].dims[frequency_axis].attach_scale(f["k0"])
        return datafile


def _convert_to_k0(x, xtype, xunit, k0unit=r"nm^{-1}"):
    c = 299792458.
    k0unit = INVLENGTHS[k0unit]
    if xtype in ("freq", "nu"):
        xunit = FREQUENCIES[xunit]
        return 2 * np.pi * x / c * (xunit / k0unit)
    elif xtype == "omega":
        xunit = FREQUENCIES[xunit]
        return x / c * (xunit / k0unit)
    elif xtype == "k0":
        xunit = INVLENGTHS[xunit]
        return x * (xunit / k0unit)
    elif xtype == "lambda0":
        xunit = LENGTHS[xunit]
        return 2 * np.pi / (x * xunit * k0unit)
    raise ValueError(f"unrecognized frequency/wavenumber/wavelength type: {xtype}")


def _scale_position(scale, data, offset=0):
    scale_axis = [i for i, x in enumerate(data.dims) if scale in x.values()]
    ndim = data.ndim
    if scale_axis:
        if len(scale_axis) == 1:
            return ndim - scale_axis[0] - 1 - offset
        raise Exception("scale added to multiple axes")
    return ndim - 1 - offset


def _load_parameter(param, group, tmatrix, frequency, append_dim=0, default=None):
    if param in group:
        dim = _scale_position(group[param], tmatrix, offset=2)
        if dim == 0:
            dim = _scale_position(frequency, group[param]) + append_dim
        res = group[param][...]
        return res.reshape(res.shape + (1,) * dim)
    return default


def load_hdf5(filename, unit_length="nm"):
    """
    Load a T-matrix stored in a HDF4 file

    Args:
        filename (string): Name of the h5py file
        unit_length (string, optional): Unit of length to be used in the T-matrices

    Returns:
        TMatrix, array_like
    """
    with h5py.File(filename, "r") as f:
        for freq_type in ("freq", "nu", "omega", "k0", "lambda0"):
            if freq_type in f:
                ld_freq = f[freq_type][...]
                break
        if "modes/positions" in f:
            k0unit = f["modes/positions"].attrs.get("unit", unit_length) + r"^{-1}"
        else:
            k0unit = unit_length + r"^{-1}"
        k0s = _convert_to_k0(ld_freq, freq_type, f[freq_type].attrs["unit"], k0unit)
        k0_dim = _scale_position(f[freq_type], f["tmatrix"], offset=2)
        k0s = k0s.reshape(k0s.shape + (1,) * k0_dim)

        found_epsilon_mu = False
        epsilon = _load_parameter(
            "relative_permittivity",
            f["embedding"],
            f["tmatrix"],
            f[freq_type],
            k0_dim,
            default=None,
        )
        mu = _load_parameter(
            "relative_permeability",
            f["embedding"],
            f["tmatrix"],
            f[freq_type],
            k0_dim,
            default=None,
        )
        if epsilon is None and mu is None:
            n = _load_parameter(
                "refractive_index",
                f["embedding"],
                f["tmatrix"],
                f[freq_type],
                k0_dim,
                default=1,
            )
            z = _load_parameter(
                "relative_impedance",
                f["embedding"],
                f["tmatrix"],
                f[freq_type],
                k0_dim,
                default=1,
            )
            epsilon = n / z
            mu = n * z
        epsilon = 1 if epsilon is None else epsilon
        mu = 1 if mu is None else mu

        kappa = _load_parameter(
            "chirality", f["embedding"], f["tmatrix"], f[freq_type], k0_dim, default=0
        )

        if "positions" in f["modes"]:
            dim = _scale_position(f["modes/positions"], f["tmatrix"], offset=2)
            if dim == 0:
                dim = _scale_position(f[freq_type], f["modes/positions"]) + k0_dim
            positions = f["modes/positions"][...]
            positions = positions.reshape(
                positions.shape[:-2] + (1,) * dim + positions.shape[-2:]
            )
        else:
            positions = np.array([[0, 0, 0]])

        # l_incident
        polarizations, helicity = _translate_polarizations_inv(
            f["modes/polarization"][...]
        )
        modes = (f["modes/l"][...], f["modes/m"][...], polarizations)
        if "position_index" in f["modes"]:
            modes = (f["modes/position_index"][...],) + modes

        shape = f["tmatrix"].shape[:-2]
        positions_shape = positions.shape[-2:]
        k0s = np.broadcast_to(k0s, shape)
        epsilon = np.broadcast_to(epsilon, shape)
        mu = np.broadcast_to(mu, shape)
        kappa = np.broadcast_to(kappa, shape)
        positions = np.broadcast_to(positions, shape + positions_shape)

        res = np.empty(shape, object)
        for i in np.ndindex(*shape):
            i_tmat = i + (slice(f["tmatrix"].shape[-2]), slice(f["tmatrix"].shape[-1]))
            i_positions = i + (slice(positions_shape[0]), slice(positions_shape[1]))
            res[i] = ptsa.TMatrix(
                f["tmatrix"][i_tmat],
                k0s[i],
                epsilon[i],
                mu[i],
                kappa[i],
                positions[i_positions],
                helicity,
                modes,
            )
        if res.shape == ():
            res = res.item()
        return res
