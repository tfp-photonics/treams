import matplotlib.pyplot as plt
import numpy as np

import treams

k0 = 2 * np.pi / 1000  # Wave number in vacuum
kz = 0
mmax = 4
radii = [150, 100]
emb = treams.Material(4, 1, 0.5)
cylinders = [
    treams.TMatrixC.cylinder(kz, mmax, k0, r, [(16 + 1j, 1, 0), emb]) for r in radii
]
positions = [[75, 75, 0], [-110, -110, 0]]
cluster = treams.TMatrixC.cluster(cylinders, positions).interacted()

pw = treams.plane_wave([emb.ks(k0)[0], 0, 0], 0, material=emb, k0=k0)
xw = cluster.xw(pw)
print(f"scattering cross width: {xw[0]}, extinction cross width: {xw[1]}")

grid = np.mgrid[-300:300:101j, -300:300:101j, 0:1].squeeze().transpose((1, 2, 0))
field = np.zeros_like(grid, complex)

valid = cluster.valid_points(grid, radii)
sca = cluster @ pw.expand(cluster.basis) @ pw
field[valid, :] = (sca.efield(grid[valid, :]) * sca[:, None]).sum(axis=-2)

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[:, 0, 0],
    grid[0, :, 1],
    np.sum(np.power(np.abs(field), 2), axis=-1).T,
    shading="nearest",
    vmin=0,
    vmax=2.5,
)
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal", "box")
fig.show()
