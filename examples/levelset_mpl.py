import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# +
# Create a 3D grid
grid_size = 100
x, y, z = np.mgrid[-1.5:1.5:grid_size*1j, -1.5:1.5:grid_size*1j, -1.5:1.5:grid_size*1j]

# Define the implicit function for a sphere: x^2 + y^2 + z^2 - 1 = 0
volume = x**2 + y**2 + z**2

# Use marching cubes to extract the surface mesh at the isovalue corresponding to the sphere
verts, faces, normals, values = measure.marching_cubes(volume, level=1.0)
# -

x.max()

np.linalg.norm(verts, axis=0)

verts

# +
# Plot the resulting mesh
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces], alpha=0.7)
ax.add_collection3d(mesh)

# Set plot limits
ax.set_xlim(0, volume.shape[0])
ax.set_ylim(0, volume.shape[1])
ax.set_zlim(0, volume.shape[2])
ax.set_title("Mesh of an Implicitly Defined Sphere")
plt.tight_layout()
plt.show()
# -


