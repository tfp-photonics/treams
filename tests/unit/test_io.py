import os
import tempfile

import gmsh
import h5py
import numpy as np

import ptsa
from ptsa import io

class TestMeshSpheres:
    def test(self):
        gmsh.initialize()
        gmsh.model.add("spheres")
        io.mesh_spheres([1, 2], [[0, 0, 2], [0, 0, -2]], gmsh.model)
        with tempfile.TemporaryDirectory() as dir:
            filename = os.path.join(dir, "spheres.msh")
            gmsh.write(filename)
            gmsh.finalize()
            with open(filename, 'r') as f:
                value = ''.join([line for line, _ in zip(f, range(5))][3:])
        expect = "$Entities\n4 6 2 2\n"
        assert value == expect

class TestSaveHdf5:
    def test(self):
        with h5py.File('test.h5', 'x', driver='core', backing_store=False) as fp:
            m = np.arange(4 * 3 * 16 * 16).reshape((4, 3, 16, 16))
            tms = [
                [
                    ptsa.TMatrix(m[i, j], 7, epsilon=i + 1, mu=j + 1, kappa=.5)
                    for j in range(3)
                ]
                for i in range(4)
            ]
            io.save_hdf5(fp, tms, "testname", "testdescr", -2, "µm")
            assert (
                np.all(fp['tmatrix'] == m)
                and fp['k0'][...] == 7
                and np.all(fp['materials/embedding/relative_permittivity'] == np.arange(1, 5)[:, None] * [1, 1, 1])
                and np.all(fp['materials/embedding/relative_permeability'] == np.arange(1, 4))
                and fp['materials/embedding/chirality'][...] == .5
                and fp['embedding'] == fp['materials/embedding']
                and np.all(fp['modes/l'] == tms[0][0].l)
                and np.all(fp['modes/m'] == tms[0][0].m)
                and np.all([i.decode() for i in fp['modes/polarization'][...]] == 8 * ['positive', 'negative'])
                and np.all(fp['modes/position_index'] == tms[0][0].pidx)
                and np.all(fp['modes/positions'] == tms[0][0].positions)
                and fp['tmatrix'].attrs['id'] == -2
                and fp['tmatrix'].attrs['name'] == "testname"
                and fp['tmatrix'].attrs['description'] == "testdescr"
                and fp['k0'].attrs['unit'] == r"µm^{-1}"
                and fp['modes/positions'].attrs['unit'] == "µm"
                and fp['materials/embedding'].attrs['name'] == "Embedding"
            )
    def test2(self):
        with h5py.File('test.h5', 'x', driver='core', backing_store=False) as fp:
            m = np.arange(4 * 3 * 16 * 16).reshape((4, 3, 16, 16))
            tms = [
                [
                    ptsa.TMatrix(m[i, j], i + 1, epsilon=1, mu=1, helicity=False)
                    for j in range(3)
                ]
                for i in range(4)
            ]
            io.save_hdf5(fp, tms, "testname", "testdescr", frequency_axis=0)
            assert (
                np.all(fp['tmatrix'] == m)
                and np.all(fp['k0'] == np.arange(1, 5))
                and fp['materials/embedding/relative_permittivity'][...] == 1
                and fp['materials/embedding/relative_permeability'][...] == 1
                and fp['embedding'] == fp['materials/embedding']
                and np.all(fp['modes/l'] == tms[0][0].l)
                and np.all(fp['modes/m'] == tms[0][0].m)
                and np.all([i.decode() for i in fp['modes/polarization'][...]] == 8 * ['electric', 'magnetic'])
                and np.all(fp['modes/position_index'] == tms[0][0].pidx)
                and np.all(fp['modes/positions'] == tms[0][0].positions)
                and fp['tmatrix'].attrs['id'] == -1
                and fp['tmatrix'].attrs['name'] == "testname"
                and fp['tmatrix'].attrs['description'] == "testdescr"
                and fp['k0'].attrs['unit'] == r"nm^{-1}"
                and fp['modes/positions'].attrs['unit'] == "nm"
                and fp['materials/embedding'].attrs['name'] == "Embedding"
            )

class TestLoadHdf5:
    def test(self):
        with h5py.File('test.h5', 'x', driver='core', backing_store=False) as fp:
            fp.create_dataset('tmatrix', data=np.arange(4 * 3 * 6 * 4).reshape((4, 3, 6, 4)))
            fp.create_dataset('nu', data=np.arange(1, 5))
            fp['nu'].attrs['unit'] = 'THz'
            fp.create_dataset('materials/foo/refractive_index', data=4)
            fp["embedding"] = h5py.SoftLink("/materials/foo")
            fp.create_dataset('modes/positions', data=[[0, 0, 0]])
            fp.create_dataset('modes/l_incident', data=[1, 1, 2, 2])
            fp.create_dataset('modes/l_scattered', data=6 * [1])
            fp.create_dataset('modes/m_incident', data=4 * [0])
            fp.create_dataset('modes/m_scattered', data=[-1, -1, 0, 0, 1, 1])
            fp.create_dataset('modes/polarization_incident', data=2 * ['electric', 'magnetic'])
            fp.create_dataset('modes/polarization_scattered', data=3 * ['electric', 'magnetic'])

            tms = io.load_hdf5(fp)

        assert (
            tms.shape == (4, 3)
            and abs(tms[0, 0].k0 - 2 * np.pi / 299792.458) < 1e-16
            and abs(tms[0, 1].k0 - 2 * np.pi / 299792.458) < 1e-16
            and abs(tms[1, 0].k0 - 4 * np.pi / 299792.458) < 1e-16
            and not tms[0, 0].helicity
            and np.all(tms[0, 0].l == 6 * [1] + 2 * [2])
            and np.all(tms[0, 0].m == [-1, -1, 0, 0, 1, 1, 0, 0])
            and np.all(tms[0, 0].pol == 4 * [0, 1])
            and np.all(tms[0, 0].positions == [[0, 0, 0]])
            and np.all(tms[3, 2].t == [
                [0, 0, 269, 268, 0, 0, 271, 270],
                [0, 0, 265, 264, 0, 0, 267, 266],
                [0, 0, 277, 276, 0, 0, 279, 278],
                [0, 0, 273, 272, 0, 0, 275, 274],
                [0, 0, 285, 284, 0, 0, 287, 286],
                [0, 0, 281, 280, 0, 0, 283, 282],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ])
        )
