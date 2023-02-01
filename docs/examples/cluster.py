import matplotlib.pyplot as plt
import numpy as np

import ptsa

k0 = 2 * np.pi / 1000
air = ptsa.Material()
snow = ptsa.Material(16 + 0.1j)
lmax = 4
radii = [150, 100, 40, 30]  # head, body, right and left hand

snowballs = [ptsa.TMatrix.sphere(lmax, k0, r, [snow, air]) for r in radii]

positions = [[0, 0, -100], [0, 0, 150], [-85, 145, -5], [-80, -135, -7.5]]
snowman = ptsa.TMatrix.cluster(snowballs, positions).interacted()

illu = ptsa.plane_wave(k0, 0, 0)
xs = snowman.xs(illu)
print(f"scattering cross section: {xs[0]}, extinction cross section: {xs[1]}")

grid = np.mgrid[-300:300:101j, 0:1, -300:300:101j].squeeze().transpose((1, 2, 0))
field = np.zeros_like(grid, complex)
valid = np.logical_and.reduce(
    [np.linalg.norm(grid - p, axis=-1) > r for p, r in zip(positions, radii)]
)
sca = snowman @ illu
field[valid, :] = (sca.efield(grid[valid, :]) * sca[:, None]).sum(axis=-2)

snowman = snowman.globalmat(ptsa.SphericalWaveBasis.default(6))
scattered_field_coeff = snowman.field(grid[outside, :])
field[outside, :] = np.sum(scattered_field_coeff * (snowman @ illu), axis=-2)
field[valid] += illu.efield(grid[valid, :])

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[:, 0, 0],
    grid[0, :, 2],
    np.sum(np.power(np.abs(field), 2), axis=-1).T,
    shading="nearest",
    vmin=0,
    vmax=5,
)
ax.plot(
    np.linspace(-250, 250, 200),
    np.sqrt(250 * 250 - np.linspace(-250, 250, 200) ** 2),
    c="r",
    linestyle=":",
)
ax.plot(
    np.linspace(-250, 250, 200),
    -np.sqrt(250 * 250 - np.linspace(-250, 250, 200) ** 2),
    c="r",
    linestyle=":",
)
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_aspect("equal", "box")
fig.show()
