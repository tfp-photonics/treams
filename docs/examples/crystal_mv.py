import matplotlib.pyplot as plt
import numpy as np

import treams

k0s = 2 * np.pi * np.linspace(0.01, 1, 200)
materials = [treams.Material(12.4), treams.Material()]
mmax = 3
radius = 0.2
lattice = treams.Lattice.spher(0.5)
kpar = [0, 0, 0]

res = []
for k0 in k0s:
    sphere = treams.TMatrixC.cylinder(kpar[0], mmax, k0, radius, materials)
    svd = np.linalg.svd(sphere.latticeinteraction(lattice, kpar), compute_uv=False)
    res.append(svd[-1])

fig, ax = plt.subplots()
ax.set_xlabel("Frequency (THz)")
ax.set_ylabel("Smallest singular value")
ax.semilogy(299.792458 * k0s / (2 * np.pi), res)
fig.show()

input()
