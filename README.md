# Global Plate Tectonics Mesh

A collection of open-source meshing tools for editing 2D and 3D meshes to conform with structural geological features of the subsurface in both cartesian and spherical coordinates. These meshes are designed for geoscientific numerical simulations to investigate subsurface properties such as pressure, temperature, density, and stress, etc. Insights from forward and inverse modeling support applications in groundwater flow analysis, seismicity along crustal faults, and resource exploration (e.g., minerals and hydrogen).


## Demo
![Description of the image](images/slab_200f.gif)
Southeast Asian subduction zone slab.

---

## Features
- **Reconstruct slabs** from the **Slab2.0** dataset.
- **Reconstruct crust and lithosphere thickness** from:
  - **Crust1.0**
  - **Litho1.0**
  - **Crustal age grid datasets**
- **Adjust spherical top surfaces** to match elevation and bathymetry data.
- **Reconstruct plate boundaries** from available datasets.
- **Reconstruct crustal faults** from available datasets.

---

## TODO List 

- [x] Implement coordinate transformation functions.
- [ ] Determine the best approach to reconstruct slab geometries.
- [ ] Integrate surface plate geometry with slab structures.
- [ ] Develop a method to add plate boundaries to the mesh.

---

## Ideas & Future Enhancements

### Slab Geometry Reconstruction
- Convert slab geometry **point clouds** into **volumes**.
- Create **top and bottom surfaces** from slab points, then merge them into a **volume**.
- Develop methods to **control element sizes** inside this volume.
- Implement **edge and surface labeling** for improved identification.

### Surface Plate & Plate Boundary Integration
- Integrate **surface plate geometry** with slab structures.
- Improve plate boundary reconstruction techniques.

---

## 🚀 Contributing
Contributions and ideas are welcome! If you have suggestions or improvements, feel free to submit an issue or a pull request.