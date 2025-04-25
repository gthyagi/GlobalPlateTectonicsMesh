# ### Generate Surface Mesh of Alaska(Aleutians) Subduction Region from Slab2
# 1. Generate the initial surface mesh using the Slab2 dataset.
# 2. Refine the surface mesh using MMG tools.
# 3. Construct a 3D slab volume from the refined surface mesh.
# 4. Embed the slab volume into the segment of the spherical mesh.

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import extract_boundary_points_in_2d as ebp
import xarray as xr
import pygmt
import point_shift as ps
import coordinate_transform as ct
import pyvista as pv
import subprocess
import meshio
from collections import defaultdict
from scipy.spatial import cKDTree
import tetgen
import gmsh

import plotly.graph_objects as go
import webbrowser
from pathlib import Path

# +
# output dir
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

# when this is set to True then existing files will remove and new files will be created
remove_old_file = True

# surface resolution: 'high' or 'low'
resolution = 'high'
# mmg parameters
if resolution=='low':
    hmax, hmin = 0.02, 0.019
else:
    hmax, hmin = 0.0018, 0.0017


# -

def remove_nan_rows(arr):
    '''
    Remove rows containing NaN values from the given NumPy array.
    '''
    return arr[~np.isnan(arr).any(axis=1)]


# +
# loading slab2 alu data
slab2_dir = '/Users/tgol0006/PlateTectonicsTools/Slab2_AComprehe/Slab2Distribute_Mar2018/Slab2_TXT/'
alu_dep_xyz = np.loadtxt(f'{slab2_dir}alu_slab2_dep_02.23.18.xyz', delimiter=',')
print(alu_dep_xyz)

# Remove rows containing NaN values and convert depth to positive values
alu_dep_arr = remove_nan_rows(alu_dep_xyz)
alu_dep_arr[:, 2] *= -1
print(alu_dep_arr)
print(alu_dep_arr[:,0].min(), alu_dep_arr[:,0].max())
print(alu_dep_arr[:,1].min(), alu_dep_arr[:,1].max())
print(alu_dep_arr[:,2].min(), alu_dep_arr[:,2].max())


# -

def create_scatter_plot(data, colorbar, vmin, vmax, colorbar_title='Depth (km)', _central_lon=0):
    """
    Create a scatter plot of data points on a map.

    Parameters:
    data (ndarray): Array of data points.
    colorbar (colormap): Colormap for the scatter plot.
    vmin (float): Minimum value for the colorbar.
    vmax (float): Maximum value for the colorbar.

    Returns:
    fig, ax: Figure and axes objects for the scatter plot.
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=_central_lon))

    # Add map features
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle='-', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Scatter plot colored by depth
    sc = ax.scatter(data[:,0], data[:,1], c=data[:,2], cmap=colorbar, 
                    s=1, edgecolor=None, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(sc, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label(colorbar_title)

    return fig, ax


# plotting slab depth
fig_slab, ax_slab = create_scatter_plot(alu_dep_arr, plt.cm.viridis_r.resampled(12), 0, 300, _central_lon=180)

# ### Transforming the geo lld to cubed sphere xyz

# +
# Get the minimum and maximum values in lon and lat
lon_min = np.min(alu_dep_xyz[:, 0])
lon_max = np.max(alu_dep_xyz[:, 0])
lat_min = np.min(alu_dep_xyz[:, 1])
lat_max = np.max(alu_dep_xyz[:, 1])

print(lon_min, lon_max, lat_min, lat_max)

coord_trans = ct.CoordinateTransformCubedsphere(g_lon_min=lon_min, 
                                                g_lon_max=lon_max,
                                                g_lat_min=lat_min, 
                                                g_lat_max=lat_max,)
# -

# slab surface top
alu_top_surf_c_xyz = coord_trans.geo_lld_to_cubedsphere_xyz(alu_dep_arr)
print(alu_top_surf_c_xyz)

# +
# Convert the points to a PyVista point cloud
alu_top_surf_cloud = pv.PolyData(alu_top_surf_c_xyz)

# Create a PyVista plotter
pl = pv.Plotter()

# Add the point clouds to the plotter
pl.add_points(alu_top_surf_cloud, color='red', point_size=2)

# Set the background color
pl.background_color = 'white'

# Show the plotter
pl.show(cpos='xy')


# -

def check_irregular_boundary_points(surface_mesh, visualize=True):
    """
    Check for boundary edge points connected to more than two edges and optionally visualize them.

    Parameters:
    - surface_mesh (pyvista.PolyData): The input surface mesh.
    - visualize (bool): Whether to visualize irregular boundary points.

    Returns:
    - irregular_points (list): List of IDs of points connected to more than two edges.
    """

    # Extract boundary edges
    boundary_edges = surface_mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    # Get boundary edges as an array
    edges = boundary_edges.lines.reshape(-1, 3)[:, 1:]

    # Count how many edges each point connects to
    point_edge_count = defaultdict(int)

    for edge in edges:
        point_edge_count[edge[0]] += 1
        point_edge_count[edge[1]] += 1

    # Find points connected to more than two edges
    irregular_points = [point_id for point_id, count in point_edge_count.items() if count > 2]

    print(f"Number of boundary points with more than two edge connections: {len(irregular_points)}")
    print(f"Irregular point IDs: {irregular_points}")

    # Optionally visualize irregular boundary points
    if visualize and irregular_points:
        irregular_coords = boundary_edges.points[irregular_points]

        plotter = pv.Plotter()
        plotter.add_mesh(surface_mesh, color='white', opacity=0.5, show_edges=True)
        plotter.add_mesh(boundary_edges, color='red', line_width=3)
        plotter.add_points(irregular_coords, color='green', point_size=12)
        plotter.show()
        print('Vary alpha until you get a mesh with no irregular points.')
    elif visualize:
        print("No boundary points with more than two edge connections found.")

    return # irregular_points


def convert_mesh_to_vtk(input_mesh_file, output_vtk_file):
    """
    Convert a Medit (.mesh) file to a VTK file.
    """
    mesh = meshio.read(input_mesh_file)
    meshio.write(output_vtk_file, mesh, file_format="vtk")
    
    print(f"Converted '{input_mesh_file}' to '{output_vtk_file}'.")


def run_mmgs_remesh(inputmesh, outputmesh, hmax=None, hmin=None, hausd=None, nosurf=True):
    """
    Run the MMGS (surface remesher) command on a 3D surface mesh.

    Parameters:
    - inputmesh (str): Path to input surface mesh (.mesh, .vtk, or .stl)
    - outputmesh (str): Path to output remeshed mesh
    - hmax (float, optional): Maximum element size
    - hmin (float, optional): Minimum element size
    - hausd (float, optional): Hausdorff distance for surface approximation
    - nosurf (bool): Whether to preserve the existing surface geometry (default: True)
    """
    cmd = ['mmgs_O3', inputmesh, '-o', outputmesh]

    if nosurf:
        cmd.append('-nosurf')
    if hmax is not None:
        cmd.extend(['-hmax', str(hmax)])
    if hmin is not None:
        cmd.extend(['-hmin', str(hmin)])
    if hausd is not None:
        cmd.extend(['-hausd', str(hausd)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("MMGS Command Output:")
    print(result.stdout)
    if result.stderr:
        print("MMGS Command Errors:")
        print(result.stderr)

    return result


def save_pyvista_to_mesh(pv_mesh, filename):
    """
    Convert a PyVista PolyData surface mesh to .mesh (MEDIT) format using meshio.

    Parameters:
    - pv_mesh (pyvista.PolyData): The surface mesh
    - filename (str): Output file path ending in .mesh
    """
    if not pv_mesh.is_all_triangles:
        raise ValueError("The mesh must be all triangles for MMG (.mesh format)")

    # Extract points and triangle faces
    points = pv_mesh.points
    faces = pv_mesh.faces.reshape((-1, 4))[:, 1:]  # drop leading '3's in each face

    # Create meshio.Mesh object
    mesh = meshio.Mesh(points=points, cells=[("triangle", faces)])

    # Save to .mesh format
    meshio.write(filename, mesh)
    print(f"Saved PyVista mesh to '{filename}' (.mesh format for MMG)")


# +
# creating slab top surface mesh

# create alphashape from point cloud
alu_top_surf_mesh = alu_top_surf_cloud.delaunay_2d(alpha=0.0035) # manual picking

if not os.path.isfile(f"{output_dir}alu_top_surf_mesh.vtk"):
    alu_top_surf_mesh.save(f"{output_dir}alu_top_surf_mesh.vtk")
else:
    print('Slab top surface mesh file exists!')

os.system('open ./output/alu_top_surf_mesh.vtk')

# Ensure the surface mesh has no boundary holes before passing it to MMG.
check_irregular_boundary_points(alu_top_surf_mesh)

if remove_old_file:
    os.remove('./output/alu_top_surf_mesh_mmg.vtk')

# run mmg
if not os.path.isfile(f'{output_dir}alu_top_surf_mesh_mmg.vtk'):
    save_pyvista_to_mesh(alu_top_surf_mesh, f'{output_dir}alu_top_surf_mesh.mesh')
    run_mmgs_remesh(f'{output_dir}alu_top_surf_mesh.mesh', f'{output_dir}alu_top_surf_mesh_mmg.mesh', 
                    hmax=hmax, hmin=hmin, hausd=None)
    convert_mesh_to_vtk(f'{output_dir}alu_top_surf_mesh_mmg.mesh', f'{output_dir}alu_top_surf_mesh_mmg.vtk')
else:
    print('Slab top surface mmg mesh file exists!')

# view vtk in paraview
os.system('open ./output/alu_top_surf_mesh_mmg.vtk')
# -

# Load mmg mesh
alu_top_surf_mesh_mmg = pv.read(f'{output_dir}alu_top_surf_mesh_mmg.vtk')

# Compute normals for the mesh points
alu_top_surf_normals = alu_top_surf_mesh_mmg.extract_surface().compute_normals(point_normals=True, cell_normals=False)
pl = pv.Plotter()
pl.add_mesh(alu_top_surf_normals, color='white', opacity=0.5)
pl.add_mesh(alu_top_surf_normals.glyph(orient='Normals', scale=False, factor=0.01), color='red')
pl.show(cpos='xy')

# +
# create volume from slab top surface
ext_surf = alu_top_surf_mesh_mmg.extract_surface()
ext_surf_with_normals = ext_surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

# Offset points along normals
offset_distance = 0.018  # adjust as needed
offset_points = ext_surf_with_normals.points - ext_surf_with_normals.point_normals * offset_distance

# Create offset surface mesh (copy topology, new points)
offset_surf = ext_surf_with_normals.copy()
offset_surf.points = offset_points
offset_surf['Normals'] = -offset_surf['Normals']

# +
# creating delaunay surfaces
ext_surf_delaunay = ext_surf.delaunay_2d(alpha=0.002)
print(ext_surf_delaunay.is_all_triangles)
# ext_surf_delaunay.plot()

offset_surf_delaunay = offset_surf.delaunay_2d(alpha=0.002)
print(offset_surf_delaunay.is_all_triangles)
# offset_surf_delaunay.plot()

# +
# method 1 to create volume

# 1. Extract and orient your top surface
top = alu_top_surf_mesh_mmg.extract_surface()
top = top.compute_normals(point_normals=True,
                          cell_normals=False,
                          auto_orient_normals=True)

# 2. Compute a single extrusion direction
#    (mean of point normals, then normalize)
mean_norm = top.point_normals.mean(axis=0)
mean_norm /= np.linalg.norm(mean_norm)

# 3. Extrude (this will build side walls and caps)
offset_distance = 0.018  # whatever thickness you need
extrusion_vector = -mean_norm * offset_distance

volume_surf_mesh = top.extrude(extrusion_vector, capping=True)

# # 4. (Optional) inspect
# volume_surf_mesh.plot(show_edges=True, color='lightgrey')

# Save a closed mesh for volume meshing →
volume_surf_mesh.save("./output/alu_vol_surf_mean_norm.vtk")

# 1) Extract just the outer shell, triangulate, clean, orient normals
surface = (
    volume_surf_mesh
    .extract_surface()      # drop any volumetric cells
    .triangulate()          # force all faces to be triangles
    .clean()                # merge dupes, remove degenerate tris
    .compute_normals(       # orient all triangles consistently outward
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False
    )
)

# sanity‐check
print(f"#cells: {surface.n_cells}, #pts: {surface.n_points}")
assert all(len(f)==3 for f in surface.faces.reshape(-1,4)[:,1:]), \
    "Non-triangular face detected!"

# 2) Feed it into TetGen as a PLC (piecewise linear complex)
tet = tetgen.TetGen(surface)

# -p tells TetGen “this is a closed surface PLC”
# optionally add -q for quality or -A to preserve region markers
volume = tet.tetrahedralize(switches='-pa0.001q1.4')

# 3) Grab your volumetric grid
slab_grid = tet.grid
print("Generated volume:", slab_grid)

# 4) Save / visualize
slab_grid.save("./output/alu_vol_surf_mean_norm_tet.vtk")
# +
# create volume from slab top surface

# Step 1: Load the open surface mesh
top_surf = alu_top_surf_mesh_mmg.extract_surface()

# Step 2: Compute normals
top_surf_with_normals = top_surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

# Step 3: Offset points along normals
offset_distance = 0.018  # adjust as needed
offset_points = top_surf_with_normals.points - top_surf_with_normals.point_normals * offset_distance

# Step 4: Create offset surface mesh (copy topology, new points)
offset_surf = top_surf_with_normals.copy()
offset_surf.points = offset_points
offset_surf['Normals'] = -offset_surf['Normals']


# 1) Extract the two boundary‐only meshes
top_bd = top_surf_with_normals.extract_feature_edges(
    boundary_edges=True, feature_edges=False,
    non_manifold_edges=False, manifold_edges=False
)
off_bd = offset_surf.extract_feature_edges(
    boundary_edges=True, feature_edges=False,
    non_manifold_edges=False, manifold_edges=False
)

# 2) Walk the lines array into an ordered loop of indices
def get_ordered_loop(bd):
    # bd.lines is a flat [2, i0, i1, 2, i1, i2, ...]
    edges = bd.lines.reshape(-1, 3)[:, 1:]
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    # start anywhere
    start, nxt = edges[0]
    loop = [start, nxt]
    while True:
        prev, curr = loop[-2], loop[-1]
        nbrs = [p for p in adj[curr] if p != prev]
        if not nbrs or nbrs[0] == loop[0]:
            break
        loop.append(nbrs[0])
    return loop

top_loop = get_ordered_loop(top_bd)
off_loop = get_ordered_loop(off_bd)
n = len(top_loop)
assert n == len(off_loop), "boundary loops must match"

# 3) Build the side‐wall faces, using only boundary points
#    (shift offset indices by n to live alongside the top loop)
faces = []
for i in range(n):
    i2 = (i + 1) % n
    t0, t1 = top_loop[i],   top_loop[i2]
    o0 = off_loop[i]  + n
    o1 = off_loop[i2] + n
    # two triangles per side quad
    faces.append([3, t0, t1, o0])
    faces.append([3, t1, o1, o0])
faces = np.hstack(faces)

# 4) Stack the two sets of boundary points into one array
all_pts = np.vstack([ top_bd.points,  # indices 0 .. n-1
                      off_bd.points ]) # indices n .. 2n-1

# 5) Make your side‐wall mesh
side_mesh = pv.PolyData(all_pts, faces)

# Combine all to make a closed surface
all_surfaces = top_surf_with_normals + side_mesh + offset_surf
closed_shell = all_surfaces.triangulate()

if closed_shell.is_manifold:
    print('Created volume is good for further operations')
else:
    print('Created volume is not closed')
    

if remove_old_file:
    os.remove(f"{output_dir}alu_vol_from_top_surf.vtk")

if not os.path.isfile(f"{output_dir}alu_vol_from_top_surf.vtk"):
    closed_shell.save(f"{output_dir}alu_vol_from_top_surf.vtk")
    save_pyvista_to_mesh(closed_shell, f"{output_dir}alu_vol_from_top_surf.mesh")
else:
    print('volume already exists')

os.system(f'open ./output/alu_vol_from_top_surf.vtk')

# +
# create volume mesh with TetGen
tgen = tetgen.TetGen(closed_shell)
tgen.make_manifold()
slab_volume = tgen.tetrahedralize()

# save as vtk
slab_volume_grid = tgen.grid
slab_volume_grid.save(f"{output_dir}alu_vol_from_top_surf_tet.vtk")
os.system(f'open {output_dir}alu_vol_from_top_surf_tet.vtk')
# +
# # plot half the slab
# mask = np.logical_or(slab_grid.points[:, 0] < 0, slab_grid.points[:, 0] > 0.1)
# half_slab = slab_grid.extract_points(mask)

# plotter = pv.Plotter()
# plotter.add_mesh(half_slab, color="w", show_edges=True)
# plotter.add_mesh(slab_grid, color="r", style="wireframe", opacity=0.05)
# plotter.show()

# +
# plotter = pv.Plotter(off_screen=True, window_size=[2000, 800])
# plotter.open_gif("slab.gif")
# plotter.add_mesh(slab_grid, color="r", style="wireframe", opacity=0.0)
# plotter.write_frame()

# # Zoom in slightly (1.5x zoom)
# plotter.camera.zoom(2.6)

# nframe = 200
# xb = np.array(slab_grid.bounds[0:2])
# step = xb.ptp() / nframe
# for val in np.arange(xb[0] + step, xb[1] + step, step):
#     mask = np.argwhere(slab_grid.cell_centers().points[:, 0] < val)
#     half_slab = slab_grid.extract_cells(mask)
#     plotter.add_mesh(half_slab, color="w", show_edges=True, name="building")
#     plotter.update()
#     plotter.write_frame()

# plotter.close()

# +
# plotter = pv.Plotter(off_screen=True, window_size=[2000, 1000])
# plotter.open_gif("slab_zoomed.gif")

# # Initial wireframe outline
# plotter.add_mesh(slab_grid, color="r", style="wireframe", opacity=0.0)
# plotter.write_frame()

# # Zoom in slightly (1.5x zoom)
# plotter.camera.zoom(2.15)

# nframe = 2
# xb = np.array(slab_grid.bounds[0:2])
# step = xb.ptp() / nframe

# # Use a tighter x-range to simulate zoom-in cropping
# xmin_zoom = xb[0] + 0.5 * xb.ptp()
# xmax_zoom = xb[1] - 0.5 * xb.ptp()

# for val in np.arange(xmin_zoom, xmax_zoom + step, step):
#     mask = np.argwhere(slab_grid.cell_centers().points[:, 0] < val)
#     half_slab = slab_grid.extract_cells(mask)
#     plotter.add_mesh(half_slab, color="w", show_edges=True, name="building")
#     plotter.update()
#     plotter.write_frame()

# plotter.close()

# +
# VTK_TETRA = 10 → 'tetra' in meshio
cells = [("tetra", slab_grid.cells_dict[10])]

# Create meshio Mesh object
mesh = meshio.Mesh(points=slab_grid.points, cells=cells)

# Save in Gmsh 4.1 format
meshio.write(f"{output_dir}alu_vol_from_top_surf_tet.msh", mesh, file_format="gmsh")

# +
# slab = pv.read(f"{output_dir}alu_vol_from_top_surf_tet.msh")
# -

# loading spherical mesh from gmsh
sph_msh = pv.read(f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016.msh")
sph_msh.save(f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016.vtk")

sph_msh

print(sph_msh['gmsh:dim_tags'])
print(np.unique(sph_msh['gmsh:dim_tags'][:,0]))
print(np.unique(sph_msh['gmsh:dim_tags'][:,1]))
print(np.unique(sph_msh['gmsh:physical']))
print(np.unique(sph_msh['gmsh:geometrical']))

# +
geom_ids = [11]  # the IDs you want

mask = np.isin(sph_msh.cell_data['gmsh:physical'], geom_ids)
subset = sph_msh.extract_cells(mask)

subset.plot(show_edges=True, cmap="tab20", scalars='gmsh:physical', cpos='xy')
# -

print(type(sph_msh.cells))

# +
# Extract only 'tetra' elements
volume_blocks = [i for i, block in enumerate(sph_msh.cells) if block.type == "tetra"]
tetra_cells = [sph_msh.cells[i] for i in volume_blocks]
tetra_tags = [sph_msh.cell_data['gmsh:physical'][i] for i in volume_blocks]

# Now filter for tag 11 in tetrahedra
filtered_cells = []
filtered_tags = []

for block, tags in zip(tetra_cells, tetra_tags):
    mask = np.isin(tags, [11])
    if np.any(mask):
        filtered_cells.append(block.data[mask])
        filtered_tags.append(tags[mask])

# Rebuild the subset mesh if any matches found
if filtered_cells:
    import pyvista as pv
    subset = pv.UnstructuredGrid({("tetra", fc) for fc in filtered_cells}, sph_msh.points)
    subset["gmsh:physical"] = np.concatenate(filtered_tags)
    subset.plot(show_edges=True, scalars="gmsh:physical", cmap="tab20", cpos="xy")
else:
    print("No tetrahedra found with physical tag 11.")


# +
# Extract surfaces
cap_surf = cap.extract_surface()
slab_surf = slab.extract_surface()

combined = cap + slab  # union (non-conforming)
combined.plot()

# Save surfaces as STL (for Gmsh input)
cap_surf.save(f'{output_dir}cap_surface.stl')
slab_surf.save(f'{output_dir}slab_surface.stl')
# -



# +
# def convert_stl_to_step_gmsh(stl_path, step_path):
#     # Ensure paths are absolute and normalized
#     stl_path = os.path.abspath(stl_path)
#     step_path = os.path.abspath(step_path)
    
#     geo_script = f"""SetFactory("OpenCASCADE");
#     Merge "{stl_path}";
#     Save "{step_path}";
#     """
    
#     geo_file = stl_path.replace(".stl", "_export.geo")
#     with open(geo_file, "w") as f:
#         f.write(geo_script)

#     print(f"Running Gmsh to convert {stl_path} → {step_path}")
#     subprocess.run(["gmsh", "-0", geo_file], check=True)
#     print(f"STEP file saved: {step_path}")

# # Example usage
# convert_stl_to_step_gmsh(f"{output_dir}cap_surface.stl", f"{output_dir}cap_surface.step")
# convert_stl_to_step_gmsh(f"{output_dir}slab_surface.stl", f"{output_dir}slab_surface.step")
# -

# Read the Medit mesh file
cap_mesh = meshio.read(f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016.mesh")

from scipy.spatial import cKDTree

slab_kdtree = cKDTree(slab.points)
dists, _ = slab_kdtree.query(cap_mesh.points)

# Extract top surface from the slab volume
slab_surface = slab.extract_surface().clean()  # makes it a watertight surface mesh

# Wrap the Medit points as a PyVista cloud (not needed for compute_implicit_distance)
cloud = pv.PolyData(cap_mesh.points)

# Compute signed distance at cap nodes to slab surface
cap_with_dist = cloud.compute_implicit_distance(slab_surface)

# +
# Rename the field for clarity
cap_with_dist["level_set"] = cap_with_dist["implicit_distance"]

# Save result
cap_with_dist.save(f"{output_dir}spherical_cap_with_level_set.vtk")


# -

def write_gmsh_sol(filename, mesh, field_name="level_set"):
    values = mesh.point_data[field_name]
    num_vertices = mesh.n_points

    with open(filename, 'w') as f:
        f.write("MeshVersionFormatted 2\n\n")
        f.write("Dimension 3\n\n")
        f.write("SolAtVertices\n")
        f.write(f"{num_vertices}\n")
        f.write("1 1\n\n")
        for val in values:
            f.write(f"{val:.15f} \n")
        f.write("\nEnd\n")


write_gmsh_sol(
    filename=f"{output_dir}spherical_cap_with_level_set.sol",
    mesh=cap_with_dist,
    field_name="level_set"
)

# +
input_mesh = f'{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016.mesh'
sol_file = f'{output_dir}spherical_cap_with_level_set.sol'
output_mesh = f'{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_mmg.mesh'

os.system(f'mmg3d_O3 {input_mesh} -sol {sol_file} -ls -nr -hausd 0.001 -hgrad 1.7 -hmax 0.05 -out {output_mesh}')

# +
# os.system(f'mmg3d_O3 {output_mesh} -noinsert -noswap -nomove -nsd 3 ') #-out {output_mesh}')

# +
# convert mmg output mesh to msh
output_msh = f'{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_mmg.msh'
os.system(f'gmsh {output_mesh} -o {output_msh} -nopopup -save')

output_vtk = output_msh.replace(".msh", ".vtk")
os.system(f'gmsh {output_msh} -o {output_vtk} -format vtk -nopopup -save')
# -





merged_msh = pv.read(f'{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_mmg.msh')

merged_msh

# Check available cell arrays
print(merged_msh.cell_data)

np.unique(merged_msh['gmsh:geometrical'])

# +
# # Plot by geometrical entity ID
# merged_msh.plot(scalars="gmsh:geometrical", show_edges=True, cmap="tab20", show_scalar_bar=True)

# +
# geom_id = 16
# subset = merged_msh.extract_cells(merged_msh.cell_data["gmsh:geometrical"] == geom_id)
# subset.plot(show_edges=True)

# +
geom_ids = [3, 10]  # the IDs you want

mask = np.isin(merged_msh.cell_data["gmsh:geometrical"], geom_ids)
subset = merged_msh.extract_cells(mask)

subset.plot(show_edges=True, cmap="tab20", scalars="gmsh:geometrical", cpos='xy')
# -
import pyvista as pv
sphere = pv.Sphere()
sphere.save('my_mesh.stl', binary=False)



