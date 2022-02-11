import numpy as np
import matplotlib.pyplot as plt

import ptsa


k0 = 2 * np.pi / 1000  # Wave number in vacuum
epsilon_air = 1
epsilon_snow = 16 + 1j  # Permittivity of our snow
lmax = 4  # Multipole order
radii = [
    150,  # body
    100,  # head
    40,  # right hand
    30,  # left hand
]

snowballs = [
    ptsa.TMatrix.sphere(lmax, k0, r, [epsilon_snow, epsilon_air]) for r in radii
]

positions = [
    [0, 0, -100],  # body
    [0, 0, 150],  # head
    [-85, 145, -5],  # right hand
    [-80, -135, -7.5],  # left hand
]
snowman = ptsa.TMatrix.cluster(snowballs, positions)
snowman.interact()

illu = snowman.illuminate_pw(k0, 0, 0, 0)
xs = snowman.xs(illu)
print(f"scattering cross section: {xs[0][0]}")
print(f"extinction cross section: {xs[1][0]}")

grid = np.mgrid[-300:300:101j, 0:1, -300:300:101j].squeeze().transpose((1, 2, 0))
field = np.zeros_like(grid, complex)
outside = np.sum(np.power(grid, 2), axis=-1) > 250 * 250
in_between = np.logical_and(
    np.logical_not(outside),
    np.logical_and(
        np.sum(np.power(grid - positions[0], 2), axis=-1) > radii[0] * radii[0],
        np.sum(np.power(grid - positions[1], 2), axis=-1) > radii[1] * radii[1],
    ),
)
valid = np.logical_or(outside, in_between)
scattered_field_coeff = snowman.field(grid[in_between, :])
field[in_between, :] = np.sum(scattered_field_coeff * (snowman.t @ illu), axis=-2)

snowman.globalmat(modes=ptsa.TMatrix.defaultmodes(6))
illu = snowman.illuminate_pw(k0, 0, 0, 0)
scattered_field_coeff = snowman.field(grid[outside, :])
field[outside, :] = np.sum(scattered_field_coeff * (snowman.t @ illu), axis=-2)
field[valid] += ptsa.special.vpw_A(
    k0, 0, 0, grid[valid, 0], grid[valid, 1], grid[valid, 2], 0
)

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
