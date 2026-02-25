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

# Field Evaluations in Chain

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
```

```{code-cell}
import treams
```

```{code-cell}
k0 = 2 * np.pi / 700
materials = [treams.Material(-16.5 + 1j), treams.Material()]
lmax = 3
radii = [75, 75]
positions = [[-30, 0, -75], [30, 0, 75]]
lattice = 300
kz = 0
```

```{code-cell}
spheres = [treams.TMatrix.sphere(lmax, k0, r, materials) for r in radii]
chain = treams.TMatrix.cluster(spheres, positions).latticeinteraction.solve(lattice, kz)
```

```{code-cell}
inc = treams.plane_wave([k0, 0, 0], [0, 0, 1], k0=chain.k0, material=chain.material)
sca = chain @ inc.expand(chain.basis)
```

```{code-cell}
x = np.linspace(-150, 150, 31)
y = 0
z = np.linspace(-150, 150, 31)
xx, zz = np.meshgrid(x, z, indexing="ij")
yy = np.full_like(xx, y)
grid = np.stack((xx, yy, zz), axis=-1)
```

```{code-cell}
ez = np.zeros_like(xx)
valid = chain.valid_points(grid, radii)
vals = []
for i, r in enumerate(grid[valid]):
    swb = treams.SphericalWaveBasis.default(1, positions=[r])
    field = sca.expandlattice(basis=swb).efield(r)
    vals.append(np.real(inc.efield(r)[2] + field[2]))
ez[valid] = vals
ez[~valid] = np.nan
```

```{code-cell}
fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    xx, zz, ez, shading="nearest", vmin=-0.5, vmax=0.5,
)
cb = plt.colorbar(pcm)
cb.set_label("$E_z$")
ax.set_xlabel("x (nm)")
ax.set_ylabel("z (nm)")
fig.show()
```
