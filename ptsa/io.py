import importlib.metadata

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import gmsh
except ImportError:
    gmsh = None


def generate_mesh_spheres(
    radii, positions, savename, modelname="model1", meshsize=-1, meshsize_boundary=-1
):
    if gmsh is None:
        Exception("optional dependency 'gmsh' not found, cannot create mesh")

    if meshsize == -1:
        meshsize = np.max(radii) * 0.2
    if meshsize_boundary == -1:
        meshsize_boundary = np.max(radii) * 0.2

    gmsh.initialize()
    gmsh.model.add(modelname)

    spheres = []
    for i, (radius, position) in enumerate(zip(radii, positions)):
        tag = i + 1
        gmsh.model.occ.addSphere(*position, radius, tag)
        spheres.append((3, tag))
        gmsh.model.addPhysicalGroup(3, [i + 1], tag)
        # Add surfaces for other mesh formats like stl, ...
        gmsh.model.addPhysicalGroup(2, [i + 1], tag)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshsize)
    gmsh.model.mesh.setSize(
        gmsh.model.getBoundary(spheres, False, False, True), meshsize_boundary
    )

    gmsh.model.mesh.generate(3)
    gmsh.write(savename)
    gmsh.finalize()


def _translate_polarizations(pols, helicity=True):
    if helicity:
        names = ["negative", "positive"]
    else:
        names = ["magnetic", "electric"]
    return [names[i] for i in pols]


def save_hdf5(
    savename,
    tmats,
    name,
    description,
    id,
    unit_k0=r"nm^{-1}",
    unit_length="nm",
    embedding_name="Embedding",
    embedding_description="",
):
    if h5py is None:
        Exception("optional dependency 'h5py' not found, cannot create hdf5file")
    tmat_first = np.nditer(tmats)[0]
    with np.nditer(
        [tmats, None, None, None, None, None, None],
        [],
        [["readonly"]] + ["writeonly", "allocate"] * 6,
        [None, object, float, complex, complex, complex],
    ) as it:
        for (tmat, t, k0, epsilon, mu, kappa, positions) in it:
            if (
                np.any(tmat.l != tmat_first.l)
                or np.any(tmat.m != tmat_first.m)
                or np.any(tmat.pol != tmat_first.pol)
                or np.any(tmat.pidx != tmat_first.pidx)
            ):
                raise ValueError("non-matching T-matrix modes")
            if tmat.helicity != tmat_first.helicity:
                raise ValueError("non-matching basis sets")
            t[...] = tmat.t
            k0[...] = tmat.k0
            epsilon[...] = tmat.epsilon
            mu[...] = tmat.mu
            kappa[...] = tmat.kappa
            positions[...] = tmat.positions
        tms, k0s, epsilons, mus, kappas, positions = it.operands[1:]

    tms = np.array(tms)
    positions = np.array(positions)

    if np.all(k0s == k0s.ravel()[0]):
        k0s = k0s.ravel()[0]
    if np.all(epsilons == epsilons.ravel()[0]):
        epsilons = epsilons.ravel()[0]
    if np.all(mus == mus.ravel()[0]):
        mus = mus.ravel()[0]
    if np.all(kappas == kappas.ravel()[0]):
        kappas = kappas.ravel()[0]
    if np.all(positions - tmat_first.postions == 0):
        positions = tmat_first.positions

    with h5py.File(savename, "a") as f:
        f.create_dataset("tmatrix", data=tms)
        f["tmatrix"].attrs["id"] = id
        f["tmatrix"].attrs["name"] = name
        f["tmatrix"].attrs["description"] = description

        f.create_dataset("k0", data=k0s)
        f["k0"].attrs["unit"] = unit_k0

        f.create_dataset("modes/l", data=tmat_first.l)
        f.create_dataset("modes/m", data=tmat_first.m)
        f.create_dataset(
            "modes/polarization",
            data=_translate_polarizations(tmat_first.pol, helicity=tmat_first.helicity),
        )
        f.create_dataset("modes/position_index", data=tmat_first.pidx)
        f["modes/position_index"].attr[
            "description"
        ] = """
            For local T-matrices each mode is associated with an origin. This index maps
            the modes to an entry in positions.
        """
        f.create_dataset("modes/positions", data=positions)
        f["modes/positions"].attrs[
            "description"
        ] = """
            The postions of the origins for a local T-matrix.
        """
        f["modes/positions"].attrs["unit"] = unit_length

        f["modes/l"].make_scale("l")
        f["modes/m"].make_scale("m")
        f["modes/polarization"].make_scale("polarization")
        f["modes/position_index"].make_scale("position_index")

        f["tmatrix"].dims[0].label = "Scattered modes"
        f["tmatrix"].dims[0].attach_scale(f["modes/l"])
        f["tmatrix"].dims[0].attach_scale(f["modes/m"])
        f["tmatrix"].dims[0].attach_scale(f["modes/polarization"])
        f["tmatrix"].dims[0].attach_scale(f["modes/position_index"])
        f["tmatrix"].dims[1].label = "Incident modes"
        f["tmatrix"].dims[1].attach_scale(f["modes/l"])
        f["tmatrix"].dims[1].attach_scale(f["modes/m"])
        f["tmatrix"].dims[1].attach_scale(f["modes/polarization"])
        f["tmatrix"].dims[1].attach_scale(f["modes/position_index"])

        embedding_path = "materials/" + embedding_name.lower()
        f.create_group(embedding_path)
        f[embedding_path].attrs["name"] = embedding_name
        f[embedding_path].attrs["description"] = embedding_description
        f.create_dataset(embedding_path + "relative_permittivity", data=epsilons)
        f.create_dataset(embedding_path + "relative_permeability", data=mus)
        if np.any(kappas != 0):
            f.create_dataset(embedding_path + "kappa", data=kappas)

        f["embedding"] = h5py.SoftLink("/" + embedding_path)


def load_hdf5(filename):
    with h5py.File(filename, "r") as f:
        shape = f["tmatrix"].shape[:-2]
        res = np.empty(shape, object)
        for i in np.ndindex(*shape):
            res[i] = f["tmatrix"][i, :, :]
