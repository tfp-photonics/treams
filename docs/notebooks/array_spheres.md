---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
  formats: md:myst
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (ipykernel)
---

# Array of Spheres

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
```

```{code-cell}
import treams
```

```{code-cell}
k0s = 2 * np.pi * np.linspace(1 / 600, 1 / 350, 100)
material_slab = 3
thickness = 10
material_sphere = (4, 1, 0.05)
lattice = treams.Lattice.square(500)
radius = 100
lmax = 3
```

```{code-cell}
tr = np.zeros((len(k0s), 2))
for i, k0 in enumerate(k0s):
    kpar = [0, 0.3 * k0]
    spheres = treams.TMatrix.sphere(
        lmax, k0, radius, [material_sphere, 1]
    ).latticeinteraction.solve(lattice, kpar)

    pwb = treams.PlaneWaveBasisByComp.diffr_orders(kpar, lattice, 0.02)
    plw = treams.plane_wave(kpar, [1, 0, 0], k0=k0, basis=pwb, material=1)
    slab = treams.SMatrices.slab(thickness, pwb, k0, [1, material_slab, 1])
    dist = treams.SMatrices.propagation([0, 0, radius], pwb, k0, 1)
    array = treams.SMatrices.from_array(spheres, pwb)
    total = treams.SMatrices.stack([slab, dist, array])
    tr[i, :] = total.tr(plw)
```

```{code-cell}
fig, ax = plt.subplots()
ax.set_xlabel("Frequency (THz)")
ax.plot(299792.458 * k0s / (2 * np.pi), tr[:, 0])
ax.plot(299792.458 * k0s / (2 * np.pi), tr[:, 1])
ax.legend(["$T$", "$R$"])
fig.show()
```
