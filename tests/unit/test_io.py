import os
import tempfile

import h5py
import numpy as np
import pytest

import treams
from treams import io


@pytest.mark.gmsh
def test_meshspheres():
    import gmsh

    gmsh.initialize()
    gmsh.model.add("spheres")
    io.mesh_spheres([1, 2], [[0, 0, 2], [0, 0, -2]], gmsh.model)
    with tempfile.TemporaryDirectory() as directory:
        filename = os.path.join(directory, "spheres.msh")
        gmsh.write(filename)
        gmsh.finalize()
        with open(filename, "r") as f:
            value = "".join([line for line, _ in zip(f, range(5))][3:])
    expect = "$Entities\n4 6 2 2\n"
    assert value == expect


class TestSaveHDF5:
    def test_helicity(self):
        with h5py.File("test.h5", "x", driver="core", backing_store=False) as fp:
            m = np.arange(4 * 3 * 16 * 16).reshape((4, 3, 16, 16))
            tms = [
                [
                    treams.TMatrix(
                        m[i, ii], k0=7, material=treams.Material(i + 1, ii + 1, 0.5)
                    )
                    for ii in range(3)
                ]
                for i in range(4)
            ]
            io.save_hdf5(
                fp,
                tms,
                "testname",
                "testdescr",
                lunit="µm",
                uuid=b"12345678123456781234567812345678",
            )
            assert np.all(fp["tmatrix"] == m)
            assert fp["angular_vacuum_wavenumber"][...] == 7
            assert np.all(
                fp["materials/embedding/relative_permittivity"]
                == np.arange(1, 5)[:, None] * [1, 1, 1]
            )
            assert np.all(
                fp["materials/embedding/relative_permeability"] == np.arange(1, 4)
            )
            assert fp["materials/embedding/chirality"][...] == 0.5
            assert fp["embedding"] == fp["materials/embedding"]
            assert np.all(fp["modes/l"] == tms[0][0].basis.l)
            assert np.all(fp["modes/m"] == tms[0][0].basis.m)
            assert np.all(
                [i.decode() for i in fp["modes/polarization"][...]]
                == 8 * ["positive", "negative"]
            )
            assert fp["uuid"][()] == b"12345678123456781234567812345678"
            assert fp.attrs["name"] == "testname"
            assert fp.attrs["description"] == "testdescr"
            assert fp["angular_vacuum_wavenumber"].attrs["unit"] == r"µm^{-1}"

    def test_parity(self):
        with h5py.File("test.h5", "x", driver="core", backing_store=False) as fp:
            m = np.arange(4 * 3 * 16 * 16).reshape((4, 3, 16, 16))
            tms = [
                [
                    treams.TMatrix(
                        m[i, ii], k0=i + 1, material=treams.Material(), poltype="parity"
                    )
                    for ii in range(3)
                ]
                for i in range(4)
            ]
            io.save_hdf5(fp, tms, "testname", "testdescr")
            assert (
                np.all(fp["tmatrix"] == m)
                and np.all(fp["angular_vacuum_wavenumber"] == np.arange(1, 5)[:, None])
                and fp["materials/embedding/relative_permittivity"][...] == 1
                and fp["materials/embedding/relative_permeability"][...] == 1
                and fp["embedding"] == fp["materials/embedding"]
                and np.all(fp["modes/l"] == tms[0][0].basis.l)
                and np.all(fp["modes/m"] == tms[0][0].basis.m)
                and np.all(
                    [i.decode() for i in fp["modes/polarization"][...]]
                    == 8 * ["electric", "magnetic"]
                )
            )


class TestLoadHdf5:
    def test(self):
        with h5py.File("test.h5", "x", driver="core", backing_store=False) as fp:
            fp.create_dataset(
                "tmatrix", data=np.arange(4 * 3 * 6 * 4).reshape((4, 3, 6, 4))
            )
            fp.create_dataset("frequency", data=np.arange(1, 5)[:, None])
            fp["frequency"].attrs["unit"] = "THz"
            fp.create_dataset("materials/foo/refractive_index", data=4)
            fp["embedding"] = h5py.SoftLink("/materials/foo")
            fp.create_dataset("modes/positions", data=[[0, 0, 0]])
            fp.create_dataset("modes/l_incident", data=[1, 1, 2, 2])
            fp.create_dataset("modes/l_scattered", data=6 * [1])
            fp.create_dataset("modes/m_incident", data=4 * [0])
            fp.create_dataset("modes/m_scattered", data=[-1, -1, 0, 0, 1, 1])
            fp.create_dataset(
                "modes/polarization_incident", data=2 * ["electric", "magnetic"]
            )
            fp.create_dataset(
                "modes/polarization_scattered", data=3 * ["electric", "magnetic"]
            )

            tms = io.load_hdf5(fp)

        basis = treams.SphericalWaveBasis(
            [
                (1, 0, 0),
                (1, 0, 1),
                (2, 0, 0),
                (2, 0, 1),
                (1, -1, 0),
                (1, -1, 1),
                (1, 1, 0),
                (1, 1, 1),
            ]
        )
        assert tms.shape == (4, 3)
        assert abs(tms[0, 0].k0 - 2 * np.pi / 299792.458) < 1e-16
        assert abs(tms[0, 1].k0 - 2 * np.pi / 299792.458) < 1e-16
        assert abs(tms[1, 0].k0 - 4 * np.pi / 299792.458) < 1e-16
        assert tms[0, 0].poltype == "parity"
        assert tms[0, 0].basis <= basis and basis <= tms[0, 0].basis
        assert np.all(
            tms[3, 2]
            == [
                [272, 273, 274, 275, 0, 0, 0, 0],
                [276, 277, 278, 279, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [264, 265, 266, 267, 0, 0, 0, 0],
                [268, 269, 270, 271, 0, 0, 0, 0],
                [280, 281, 282, 283, 0, 0, 0, 0],
                [284, 285, 286, 287, 0, 0, 0, 0],
            ]
        )
