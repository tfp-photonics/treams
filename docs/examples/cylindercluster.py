import numpy as np
import matplotlib.pyplot as plt

import ptsa


k0 = 2 * np.pi / 1000  # Wave number in vacuum
kz = 0
mmax = 4
radii = [150, 100]
cylinders = [
    ptsa.TMatrixC.cylinder(kz, mmax, k0, r, [4, 16 + 1j], kappa=[0.5, 0]) for r in radii
]
positions = [[75, 75, 0], [-110, -110, 0]]
cluster = ptsa.TMatrixC.cluster(cylinders, positions)
cluster.interact()

illu = cluster.illuminate_pw(np.real(cluster.ks[0]), 0, 0, 0)

grid = np.mgrid[-300:300:101j, -300:300:101j, 0:1].squeeze().transpose((1, 2, 0))
scattered_field = np.zeros_like(grid, complex)
valid = np.logical_and(
    np.sum(np.power(grid - positions[0], 2), axis=-1) > radii[0] * radii[0],
    np.sum(np.power(grid - positions[1], 2), axis=-1) > radii[1] * radii[1],
)
scattered_field_coeff = cluster.field(grid[valid, :])
scattered_field[valid, :] = np.sum(scattered_field_coeff * (cluster.t @ illu), axis=-2)

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    grid[:, 0, 0],
    grid[0, :, 1],
    0.5
    * np.sum(
        np.power(
            np.abs(
                scattered_field
                + ptsa.special.vpw_A(
                    k0, 0, 0, grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], 0
                )
            ),
            2,
        ),
        axis=-1,
    ).T,
    shading="nearest",
    vmax=2.5,
)
cb = plt.colorbar(pcm)
cb.set_label("Intensity")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal", "box")
fig.show()
