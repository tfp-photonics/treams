import matplotlib.pyplot as plt
import numpy as np

import treams

k0 = 2 * np.pi / 700  # Wave number in vacuum
materials = [treams.Material(-16.5 + 1j), treams.Material()]
lmax = 2
radii = [75, 75]
spheres = [treams.TMatrix.sphere(lmax, k0, r, materials) for r in radii]
positions = [[-25, 0, -73], [25, 0, 73]]
chain_sph = treams.TMatrix.cluster(spheres, positions)
pitch = 300
kz = 0
chain_sph = chain_sph.interacted_lattice(pitch, kz)

pw = treams.plane_wave(k0, 0, 0, 0)
grid = np.mgrid[-150:150:51j, 0:1, -150:150:51j].squeeze().transpose((1, 2, 0))
field = np.zeros_like(grid, complex)

valid = chain_sph.valid_points(grid, radii) & (np.abs(grid[:, :, 0]) < 100)
sca = chain_sph @ pw.expand(chain_sph.basis) @ pw
for i, ii in zip(*np.where(valid)):
    probe = treams.SphericalWaveBasis.default(1, positions=grid[i, ii, :])
    field[i, ii, :] = (
        treams.efield(grid[i, ii, :], basis=probe, k0=k0).T
        @ sca.expandlattice(basis=probe)
        @ sca
    )

cwb = treams.CylindricalWaveBasis.with_periodicity(
    0, 2, pitch, 13 / pitch, nmax=2, positions=positions
)
chain = treams.TMatrixC.from_array(chain_sph, cwb)
valid = chain.valid_points(grid, radii)
field[valid, :] = np.sum(
    chain.efield(grid[valid]) * (chain @ pw.expand(chain.basis) @ pw)[:, None], axis=-2
)

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[:, 0, 0],
    grid[0, :, 2],
    np.sum(np.power(np.abs(field), 2), axis=-1).T,
    shading="nearest",
    vmax=3,
)
ax.plot([-100, -100], [-150, 150], c="r", linestyle=(0, (1, 10)))
ax.plot([100, 100], [-150, 150], c="r", linestyle=(0, (1, 10)))
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_aspect("equal", "box")
fig.show()
