import matplotlib.pyplot as plt
import numpy as np

import ptsa

k0 = 2 * np.pi / 700  # Wave number in vacuum
permittivities = [1, -16.5 + 1j]
lmax = 2  # Multipole order
radii = [75, 75]
spheres = [ptsa.TMatrix.sphere(lmax, k0, r, permittivities) for r in radii]
positions = [[-25, 0, -73], [25, 0, 73]]
chain_sph = ptsa.TMatrix.cluster(spheres, positions)
pitch = 300
kz = 0
chain_sph.latticeinteract(kz, pitch)

illu = chain_sph.illuminate_pw(k0, 0, 0, 0)
grid = np.mgrid[-150:150:51j, 0:1, -150:150:51j].squeeze().transpose((1, 2, 0))
field = np.zeros_like(grid, complex)
outside = np.sum(np.power(grid[:, :, :2], 2), axis=-1) > 100 ** 2
in_between = np.logical_and(
    np.logical_not(outside),
    np.logical_and(
        np.sum(np.power(grid - positions[0], 2), axis=-1) > radii[0] * radii[0],
        np.sum(np.power(grid - positions[1], 2), axis=-1) > radii[1] * radii[1],
    ),
)
valid = np.logical_or(outside, in_between)

modes = ptsa.TMatrix.defaultmodes(1, np.sum(in_between))
field_expansion = chain_sph.lattice_field(grid[in_between, :], modes, kz, pitch)
field_per_mode = ptsa.special.vsph2car(
    ptsa.special.vsw_rA(*modes[1:3], 0, 0, 0, modes[3]), [0, 0, 0]
)
sca_field_coeff = np.sum(
    np.reshape(
        field_per_mode[:, None, :] * field_expansion[:, :, None],
        (-1, 6, chain_sph.t.shape[0], 3),
    ),
    axis=1,
)
field[in_between, :] = np.sum(sca_field_coeff * (chain_sph.t @ illu), axis=-2)

kzs = kz + np.arange(-3, 4) * 2 * np.pi / pitch
chain_cyl = ptsa.TMatrixC.array(ptsa.TMatrix.cluster(spheres, positions), kzs, pitch)
illu = chain_cyl.illuminate_pw(k0, 0, 0, 0)
scattered_field_coeff = chain_cyl.field(grid[outside, :])
field[outside, :] = np.sum(scattered_field_coeff * (chain_cyl.t @ illu), axis=-2)

field[valid] += ptsa.special.vpw_A(
    k0, 0, 0, grid[valid, 0], grid[valid, 1], grid[valid, 2], 0
)

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[:, 0, 0],
    grid[0, :, 2],
    np.sum(np.power(np.abs(field), 2), axis=-1).T,
    shading="nearest",
    vmax=30,
)
ax.plot([-100, -100], [-150, 150], c="r", linestyle=(0, (1, 10)))
ax.plot([100, 100], [-150, 150], c="r", linestyle=(0, (1, 10)))
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_aspect("equal", "box")
fig.show()
