import matplotlib.pyplot as plt
import numpy as np

import treams

k0 = 2 * np.pi / 700
materials = [treams.Material(-16.5 + 1j), treams.Material()]
lmax = mmax = 3
radii = [75, 65]
positions = [[-30, 0, -75], [30, 0, 75]]
lattice = treams.Lattice(300)
kz = 0.005

spheres = [treams.TMatrix.sphere(lmax, k0, r, materials) for r in radii]
chain = treams.TMatrix.cluster(spheres, positions).latticeinteraction.solve(lattice, kz)

bmax = 3.1 * lattice.reciprocal
cwb = treams.CylindricalWaveBasis.diffr_orders(kz, mmax, lattice, bmax, 2, positions)
chain_tmc = treams.TMatrixC.from_array(chain, cwb)

inc = treams.plane_wave(
    [np.sqrt(k0 ** 2 - kz ** 2), 0, kz],
    [np.sqrt(0.5), np.sqrt(0.5)],
    k0=chain.k0,
    material=chain.material,
)
sca = chain_tmc @ inc.expand(chain_tmc.basis)

grid = (
    np.mgrid[-300:300:61j, 0:1, -150:150:31j].squeeze().transpose((1, 2, 0))[:, :-1, :]
)
ez = np.zeros_like(grid[..., 0], complex)
valid = chain_tmc.valid_points(grid, radii)
ez[valid] = (inc.efield(grid[valid]) + sca.efield(grid[valid]))[..., 2]
ez[~valid] = np.nan
sca = chain @ inc.expand(chain.basis)
valid = ~valid & chain.valid_points(grid, radii)
vals = []
for i, r in enumerate(grid[valid]):
    swb = treams.SphericalWaveBasis.default(1, positions=[r])
    field = sca.expandlattice(basis=swb).efield(r)
    vals.append(inc.efield(r)[2] + field[2])
ez[valid] = vals

ez = np.concatenate(
    (
        np.real(ez.T * np.exp(-1j * kz * lattice[...])),
        ez.T.real,
        np.real(ez.T * np.exp(1j * kz * lattice[...])),
    )
)
zaxis = np.concatenate((grid[0, :, 2] - 300, grid[0, :, 2], grid[0, :, 2] + 300,))

fig, ax = plt.subplots(figsize=(10, 20))
pcm = ax.pcolormesh(grid[:, 0, 0], zaxis, ez, shading="nearest", vmin=-1, vmax=1,)
cb = plt.colorbar(pcm)
cb.set_label("$E_z$")
ax.set_xlabel("x (nm)")
ax.set_ylabel("z (nm)")
ax.set_aspect("equal")
fig.show()
