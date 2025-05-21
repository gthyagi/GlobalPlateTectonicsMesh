# ### Conclusion: 
# 1. you can not escape alpha shapes
# 2. point cloud alpha shapes are better than contours line ones

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import numpy as np
import pyvista as pv
import os
import glob
import coordinate_transform as ct


def read_gmt_contours2(fname):
    """
    Read a GMT-style .in contour file with lines starting '>' and columns lon, lat, depth, flag.
    Returns a list of segments, each an (M,3) array [lon, lat, depth].
    """
    segments = []
    current = []
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current:
                    segments.append(np.array(current, float))
                    current = []
                continue
            parts = line.split()
            lon, lat, depth = map(float, parts[:3])
            current.append((lon, lat, depth))
    if current:
        segments.append(np.array(current, float))
    return segments


def build_2d_surface_from_contours(contour_file, alpha=0.0035):
    """
    From GMT .in contour segments, create a 2D triangulated surface mesh.
    
    Parameters
    ----------
    contour_file : str
        Path to the `.in` file.
    alpha : float
        Alpha parameter for 2D Delaunay (controls maximum edge length).
    
    Returns
    -------
    mesh : pv.PolyData
        The 2D triangulated surface with (X,Y,Z) points.
    """
    # 1) Read segments
    segments = read_gmt_contours2(contour_file)
    
    # 2) Convert each segment's lon/lat/depth to Cartesian XYZ
    transformer = ct.CoordinateTransformSphere()
    xyz_list = [transformer.lld_to_xyz(seg) for seg in segments]
    
    # 3) Flatten all points
    pts3d = np.vstack(xyz_list)

    print(pts3d)
    
    # 4) Build a point cloud PolyData
    cloud = pv.PolyData(pts3d)
    cloud['depth'] = pts3d[:, 2]  # attach depth scalar
    
    # 5) Perform 2D Delaunay triangulation
    mesh = cloud.delaunay_2d(alpha=alpha)
    
    # 6) Attach depth scalar to mesh
    # depth is the Z coordinate in cloud; after triangulation, transfer
    mesh['depth'] = mesh.points[:, 2]
    
    return mesh


# +
# Usage example:
file_path = '/Users/tgol0006/PlateTectonicsTools/Slab2_AComprehe/Slab2Distribute_Mar2018/Slab2_CONTOURS/ARC_CONTOURS/'
contour_file = f'{file_path}sum_slab2_dep_02.23.18_contours2.in'
alpha_value = 0.015  # tune based on data spacing

mesh2d = build_2d_surface_from_contours(contour_file, alpha=alpha_value)
# -

# Visualize
pl = pv.Plotter()
pl.add_mesh(mesh2d, color='white', opacity=0.5, show_edges=True)
pl.show()

# +
# # # Save to VTK
# # mesh2d.save('sum_slab2_dep_02.23.18_2dmesh.vtk')
