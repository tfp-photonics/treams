import matplotlib.pyplot as plt
import numpy as np

import treams

k0 = 2 * np.pi / 700
materials = [treams.Material(-16.5 + 1j), treams.Material()]
lmax = mmax = 3
radii = [75, 75]
positions = [[0, -30, -75], [0, 30, 75]]
lattice = treams.Lattice(300)
kz = 0.005

spheres = [treams.TMatrix.sphere(lmax, k0, r, materials) for r in radii]
chain = treams.TMatrix.cluster(spheres, positions).latticeinteraction.solve(lattice, kz)

inc = treams.plane_wave(
    [np.sqrt(k0**2 - kz**2), 0, kz], [0, 0, 1], k0=chain.k0, material=chain.material
)
bmax = 2.1 * lattice.reciprocal
cwb = treams.CylindricalWaveBasis.diffr_orders(kz, mmax, lattice, bmax, 2, positions)
chain_tmc = treams.TMatrixC.from_array(chain, cwb)
sca = chain_tmc @ inc.expand(chain_tmc.basis)

grid = np.mgrid[-300:300:121j, 0:1, -150:150:61j].squeeze().transpose((1, 2, 0))
ez = np.zeros_like(grid[..., 0])
valid = chain_tmc.valid_points(grid, radii)
ez[valid] = (
    np.sum(np.abs(inc.efield(grid[valid]) + sca.efield(grid[valid])) ** 2, -1) / 2
)
ez[~valid] = np.nan

fig, ax = plt.subplots(figsize=(10, 20))
pcm = ax.pcolormesh(
    grid[:, 0, 0],
    grid[0, :, 2],
    ez.T,
    shading="nearest",
    vmin=-0,
    vmax=1,
)
cb = plt.colorbar(pcm)
cb.set_label("$E_z$")
ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
ax.set_aspect("equal")
fig.show()
