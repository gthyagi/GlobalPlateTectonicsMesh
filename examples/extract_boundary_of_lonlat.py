# ### Extract boundary points from lonlat data 

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

import plotly.graph_objects as go
import webbrowser
from pathlib import Path

# output dir
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)


def remove_nan_rows(arr):
    '''
    Remove rows containing NaN values from the given NumPy array.
    '''
    return arr[~np.isnan(arr).any(axis=1)]


# +
# loading slab2 sum data
slab2_dir = '/Users/tgol0006/PlateTectonicsTools/Slab2_AComprehe/Slab2Distribute_Mar2018/Slab2_TXT/'
slab2_data = np.loadtxt(f'{slab2_dir}sum_slab2_dep_02.23.18.xyz', delimiter=',')
print(slab2_data)

# Remove rows containing NaN values and convert depth to positive values
sum_dep_arr = remove_nan_rows(slab2_data)
sum_dep_arr[:, 2] *= -1
print(sum_dep_arr)


# -

def create_scatter_plot(data, colorbar, vmin, vmax, colorbar_title='Depth (km)'):
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
    ax = plt.axes(projection=ccrs.PlateCarree())

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
fig_slab, ax_slab = create_scatter_plot(sum_dep_arr, plt.cm.viridis_r.resampled(14), 0, 700)

# Computing the alpha shape and get the edges
edges = ebp.alpha_shape(sum_dep_arr[:,0:2], alpha=0.04, only_outer=True)


# nbi-prompt: write a it in a better form and make def
def create_interactive_plot(lon, lat, depth, edges, output_dir):
    """
    Create an interactive plot with Plotly.

    Parameters:
    lon (ndarray): Array of longitudes.
    lat (ndarray): Array of latitudes.
    depth (ndarray): Array of depths.
    edges (ndarray): Array of boundary edges.
    output_dir (str): Output directory for the HTML file.

    Returns:
    None
    """
    # Create scattergeo plot
    fig = go.Figure()

    # Add data points
    fig.add_trace(go.Scattergeo(
        lon=lon,
        lat=lat,
        mode='markers',
        marker=dict(
            size=5,
            color=depth,
            colorscale='Viridis_r',
            colorbar=dict(title='Depth (km)'),
            cmin=0,
            cmax=700,
        ),
        hoverinfo='text',
        text=[f'Lon: {lo}, Lat: {la}, Depth: {dp} km' for lo, la, dp in zip(lon, lat, depth)],
    ))

    # Add boundary edges
    for i, j in edges:
        fig.add_trace(go.Scattergeo(
            lon=[lon[i], lon[j]],
            lat=[lat[i], lat[j]],
            mode='lines',
            line=dict(width=1, color='black'),
            hoverinfo='skip'
        ))

    # Update layout to focus on area of interest
    fig.update_layout(
        title='Lon-Lat-Depth with boundary',
        geo=dict(
            resolution=50,
            coastlinecolor='gray',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showcountries=True,
            projection=dict(type='equirectangular'),
            lonaxis=dict(range=[lon.min()-1, lon.max()+1]),
            lataxis=dict(range=[lat.min()-2, lat.max()+2]),
        ),
    )

    # Save interactive plot as HTML file
    fig.write_html(f'{output_dir}lonlatdep_bd_interactive_plot.html')
# +
# create interactive plot
lon, lat, depth = sum_dep_arr[:,0], sum_dep_arr[:,1], sum_dep_arr[:,2]
create_interactive_plot(lon, lat, depth, edges, output_dir)

# Ensure the path is absolute
file_path = (Path(output_dir) / 'lonlatdep_bd_interactive_plot.html').resolve()

# # Open the file in browser
# webbrowser.open(file_path.as_uri())


# -


def get_boundary_points(boundary_edges):
    """
    Get the unique boundary points from the given boundary edges.

    Parameters:
    boundary_edges (ndarray): Array of boundary edges.

    Returns:
    ndarray: Array of unique boundary points.
    """
    stacked_array = boundary_edges.reshape(-1, 3)
    _, unique_indices = np.unique(stacked_array, axis=0, return_index=True)
    unique_points = stacked_array[np.sort(unique_indices)]
    return unique_points


# get boundary points
boundary_edges = sum_dep_arr[ebp.stitch_boundary(edges)]
boundary_points = get_boundary_points(boundary_edges)
print(boundary_points)

# add boundary points to the plot
fig_slab, ax_slab = create_scatter_plot(sum_dep_arr, plt.cm.viridis_r.resampled(14), 0, 700)
sc_1 = ax_slab.scatter(boundary_points[:,0], boundary_points[:,1], c='r', s=1, edgecolor=None, transform=ccrs.PlateCarree(),)


def print_info(grid_name, grid_path):
    '''
    print the information of the grid
    '''
    info = pygmt.grdinfo(f'{grid_path}{grid_name}')
    clean_info = info.replace(grid_path, '')
    # print(clean_info)

    # Define the keywords that indicate grid properties.
    keywords = ["x_min", "y_min", "v_min", "scale_factor"]
    
    # Split the string into individual lines.
    lines = clean_info.splitlines()
    
    # Filter and print only lines that contain one of the keywords.
    for line in lines:
        if any(keyword in line for keyword in keywords):
            print(line)
    
    return 


# +
# slab2 data dir

# 2018 files
slab2_dir = '/Users/tgol0006/PlateTectonicsTools/Slab2_AComprehe/Slab2Distribute_Mar2018/'
grd_file_list = ['sum_slab2_dep_02.23.18.grd', 'sum_slab2_thk_02.23.18.grd', 'sum_slab2_dip_02.23.18.grd', 'sum_slab2_str_02.23.18.grd']

grids = []
for grd in grd_file_list:
    print_info(grd, slab2_dir)
    grids.append(xr.open_dataarray(f'{slab2_dir}{grd}'))

# grids
# -

def xarray_to_numpy(xda):
    '''
    Stack the spatial dimensions of the given xarray DataArray into a single 'points' dimension
    and convert it to a 2D NumPy array with columns: lon, lat, value.
    '''
    stacked = xda.stack(points=("y", "x"))
    arr = np.column_stack((stacked.coords["x"].values,
                           stacked.coords["y"].values,
                           stacked.values))
    return arr


# converting xarray to numpy array
dep_arr, thk_arr, dip_arr, str_arr = [xarray_to_numpy(grid) for grid in grids]

# depth list
depth_values_list = np.linspace(5, 200, 40)
print(depth_values_list)


def find_matching_indices(dep_arr, boundary_points):
    """
    Get the indices in dep_arr for boundary points.

    Parameters:
    dep_arr (ndarray): Array of data points.
    boundary_points (ndarray): Array of boundary points.

    Returns:
    ndarray: Indices of matching rows in dep_arr.
    """
    
    dtype = [('lon', float), ('lat', float)] # Define structured dtype explicitly (assuming columns are lon, lat)
    # Convert to structured arrays
    dep_struct = np.array([tuple(np.round(row, 2)) for row in dep_arr[:,0:2]], dtype=dtype)
    boundary_struct = np.array([tuple(np.round(p,2)) for p in boundary_points[:,0:2]], dtype=dtype)
    mask = np.in1d(dep_struct, boundary_struct) # Use np.in1d to find matching rows
    indices = np.where(mask)[0] # Get the indices of matching rows

    return indices


# matches only lon, lat columns
boundary_points_indx = find_matching_indices(dep_arr, boundary_points)
print(boundary_points_indx.shape)

# add boundary points to the plot
fig_slab, ax_slab = create_scatter_plot(sum_dep_arr, plt.cm.viridis_r.resampled(14), 0, 700)
sc_1 = ax_slab.scatter(dep_arr[boundary_points_indx][:,0], dep_arr[boundary_points_indx][:,1], 
                       c='r', s=1, edgecolor=None, transform=ccrs.PlateCarree(),)

# +
# project the boundary points by constant depth

all_boundary_points_list = []
for depth in depth_values_list:
    # print('Projecting to depth: ', depth,'km')
    for i, index in enumerate(boundary_points_indx):
        shift_pt = ps.PointShift(dep_arr[index][0], dep_arr[index][1], -dep_arr[index][2], dip_arr[index][2], str_arr[index][2], -depth)
        if depth <= thk_arr[index][2]:
            all_boundary_points_list.append(shift_pt)

# Convert to numpy array
all_boundary_points_arr = np.array(all_boundary_points_list)

# +
# slab bottom surface array
bot_surf_dep_arr_nan = np.zeros_like(dep_arr)

# layer thickness is negative if it is shifted down and positive if it is shifted upward direction
# top surface depth must be positive
for i, values in enumerate(dep_arr):
    shift_pt = ps.PointShift(values[0], values[1], -values[2], dip_arr[i][2], str_arr[i][2], -thk_arr[i][2])
    bot_surf_dep_arr_nan[i][0] = shift_pt[0]
    bot_surf_dep_arr_nan[i][1] = shift_pt[1]
    bot_surf_dep_arr_nan[i][2] = shift_pt[2]

# Convert the NumPy arrays back to xarray DataArrays
bot_surf_dep_arr = remove_nan_rows(bot_surf_dep_arr_nan)
print(bot_surf_dep_arr)
# -

# ### Transforming the geo lld to cubed sphere xyz

# +
# Get the minimum and maximum values in lon and lat
lon_min = np.min(slab2_data[:, 0])
lon_max = np.max(slab2_data[:, 0])
lat_min = np.min(slab2_data[:, 1])
lat_max = np.max(slab2_data[:, 1])

print(lon_min, lon_max, lat_min, lat_max)

coord_trans = ct.CoordinateTransformCubedsphere(g_lon_min=lon_min, 
                                                g_lon_max=lon_max,
                                                g_lat_min=lat_min, 
                                                g_lat_max=lat_max,)
# -

# slab surface top
sum_top_surf_c_xyz = coord_trans.geo_lld_to_cubedsphere_xyz(sum_dep_arr)
print(sum_top_surf_c_xyz)

# slab surface bottom
sum_bot_surf_c_xyz = coord_trans.geo_lld_to_cubedsphere_xyz(bot_surf_dep_arr)
print(sum_bot_surf_c_xyz)

# slab boundary
all_boundary_points_c_xyz = coord_trans.geo_lld_to_cubedsphere_xyz(all_boundary_points_arr)
print(all_boundary_points_c_xyz)

# +
# Convert the points to a PyVista point cloud
sum_top_surf_cloud = pv.PolyData(sum_top_surf_c_xyz)
sum_bot_surf_cloud = pv.PolyData(sum_bot_surf_c_xyz)

# Create an empty PolyData object to store all boundary points
boundary_points_cloud = pv.PolyData(all_boundary_points_c_xyz)

# Create a PyVista plotter
pl = pv.Plotter()

# Add the point clouds to the plotter
pl.add_points(sum_top_surf_cloud, color='red', point_size=2)
pl.add_points(boundary_points_cloud, color='blue', point_size=2)
pl.add_points(sum_bot_surf_cloud, color='green', point_size=2)

# Set the background color
pl.background_color = 'white'

# Show the plotter
pl.show()

# +
# Create a surface from point cloud
sum_top_surf_mesh = sum_top_surf_cloud.delaunay_2d(alpha=0.0035) # manual picking
sum_top_surf_mesh.save(f"{output_dir}sum_top_surf_mesh.vtk")

# !open ./output/sum_top_surf_mesh.vtk
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


check_irregular_boundary_points(sum_top_surf_mesh)


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
# save .mesh from pyvista mesh
save_pyvista_to_mesh(sum_top_surf_mesh, f'{output_dir}sum_top_surf_mesh.mesh')

# running mmg surface
run_mmgs_remesh(f'{output_dir}sum_top_surf_mesh.mesh', f'{output_dir}sum_top_surf_mesh_mmg.mesh', 
                hmax=0.0015, hmin=0.0014, hausd=None)

# convert .mesh out to vtk
convert_mesh_to_vtk(f'{output_dir}sum_top_surf_mesh_mmg.mesh', f'{output_dir}sum_top_surf_mesh_mmg.vtk')

# view vtk in paraview
# !open ./output/sum_top_surf_mesh_mmg.vtk

# +
# Create a PyVista point cloud
sum_bot_surf_mesh = sum_bot_surf_cloud.delaunay_2d(alpha=0.0029)
sum_bot_surf_mesh.save(f"{output_dir}sum_bot_surf_mesh.vtk")

# !open ./output/sum_bot_surf_mesh.vtk
# -

check_irregular_boundary_points(sum_bot_surf_mesh)

# +
# save .mesh from pyvista mesh
save_pyvista_to_mesh(sum_bot_surf_mesh, f'{output_dir}sum_bot_surf_mesh.mesh')

# running mmg surface
run_mmgs_remesh(f'{output_dir}sum_bot_surf_mesh.mesh', f'{output_dir}sum_bot_surf_mesh_mmg.mesh', 
                hmax=0.0015, hmin=0.0014, hausd=None)

# convert .mesh out to vtk
convert_mesh_to_vtk(f'{output_dir}sum_bot_surf_mesh_mmg.mesh', f'{output_dir}sum_bot_surf_mesh_mmg.vtk')

# view vtk in paraview
# !open ./output/sum_bot_surf_mesh_mmg.vtk

# +
# # Create a PyVista point cloud
# point_cloud = pv.PolyData(all_boundary_points_c_xyz)
# all_boundary_points_surf_mesh = point_cloud.delaunay_2d(tol=1e-5, alpha=0.01)
# all_boundary_points_surf_mesh.save(f"{output_dir}all_boundary_points_surf_mesh.vtk")

# # !open ./output/all_boundary_points_surf_mesh.vtk

# +
# check_irregular_boundary_points(all_boundary_points_surf_mesh)

# +
# # save .mesh from pyvista mesh
# save_pyvista_to_mesh(all_boundary_points_surf_mesh, f'{output_dir}all_boundary_points_surf_mesh.mesh')

# # running mmg surface
# run_mmgs_remesh(f'{output_dir}all_boundary_points_surf_mesh.mesh', 
#                 f'{output_dir}all_boundary_points_surf_mesh_mmg.mesh', 
#                 hmax=0.0015, hmin=0.0014, hausd=None)

# # convert .mesh out to vtk
# convert_mesh_to_vtk(f'{output_dir}all_boundary_points_surf_mesh_mmg.mesh', 
#                     f'{output_dir}all_boundary_points_surf_mesh_mmg.vtk')

# # view vtk in paraview
# # !open ./output/all_boundary_points_surf_mesh_mmg.vtk
# -



# Load meshes (assuming they are initially PolyData)
sum_top_surf_mesh_mmg = pv.read(f'{output_dir}sum_top_surf_mesh_mmg.vtk')
sum_bot_surf_mesh_mmg = pv.read(f'{output_dir}sum_bot_surf_mesh_mmg.vtk')

# +
# Step 1: Load the open surface mesh
surf = sum_top_surf_mesh_mmg.extract_surface()

# Step 2: Compute normals
surf_with_normals = surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

# Step 3: Offset points along normals
offset_distance = 0.01  # adjust as needed
offset_points = surf_with_normals.points - surf_with_normals.point_normals * offset_distance

# Step 4: Create offset surface mesh (copy topology, new points)
offset_surf = surf_with_normals.copy()
offset_surf.points = offset_points
offset_surf['Normals'] = -offset_surf['Normals']


# Step 6: Stitch edges to form side walls
# Combine the two surfaces
combined = surf_with_normals + offset_surf

# Extract boundary edges and form faces between old and new surfaces
side_faces = []

boundary = surf_with_normals.extract_feature_edges(
    boundary_edges=True, feature_edges=False,
    non_manifold_edges=False, manifold_edges=False
)

# Build side faces by connecting corresponding boundary edges
b_pts = boundary.lines.reshape((-1, 3))[:, 1:]  # ignore line header "2"
for pt1, pt2 in b_pts:
    pt1_new = pt1 + surf_with_normals.n_points
    pt2_new = pt2 + surf_with_normals.n_points
    # Create two triangles for the quad side
    side_faces.append([3, pt1, pt2, pt2_new])
    side_faces.append([3, pt1, pt2_new, pt1_new])

side_faces = np.hstack(side_faces)

# Create the side wall mesh
side_mesh = pv.PolyData()
side_mesh.points = np.vstack([surf_with_normals.points, offset_points])
side_mesh.faces = side_faces

# Combine all to make a closed surface
closed_shell = surf_with_normals + offset_surf + side_mesh

# Optional: Convert to volume with TetGen
# tet = pv.TetGen(closed_shell)
# volume = tet.tetrahedralize()
# volume.plot()

# Visualize
closed_shell.plot(show_edges=True, color='white', opacity=0.6)

# -

side_mesh.plot()

# +
print(closed_shell.is_manifold)
closed_shell.save(f"{output_dir}closed_shell.vtk")

# !open ./output/closed_shell.vtk
# -



# +
# Extract boundary edges
sum_top_bd_edges = sum_top_surf_mesh_mmg.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                               manifold_edges=False, non_manifold_edges=False)

sum_bot_bd_edges = sum_bot_surf_mesh_mmg.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                               manifold_edges=False, non_manifold_edges=False)
# -

# Visualize boundary edges
plotter = pv.Plotter()
plotter.add_mesh(sum_top_surf_mesh_mmg, color='white', opacity=0.5)
plotter.add_mesh(sum_top_bd_edges, color='red', line_width=3)
plotter.add_mesh(sum_bot_surf_mesh_mmg, color='white', opacity=0.5)
plotter.add_mesh(sum_bot_bd_edges, color='green', line_width=3)
plotter.show()

# Compute normals for the mesh points
sum_top_surf_normals = sum_top_surf_mesh_mmg.extract_surface().compute_normals(point_normals=True, cell_normals=False)
sum_bot_surf_normals = sum_bot_surf_mesh_mmg.extract_surface().compute_normals(point_normals=True, cell_normals=False)


def project_boundary_points(boundary_edges, mesh_with_normals, projection_distance_list, direction='up'):
    """
    Project boundary points along their normals by a list of distances.

    Parameters:
        boundary_edges (pyvista.PolyData): Extracted boundary edges.
        mesh_with_normals (pyvista.PolyData): Original mesh with computed normals.
        projection_distance_list: List of distance arrays to project points.
        direction: up or down

    Returns:
        list of numpy.ndarray: List of projected boundary point sets.
    """
    boundary_points = boundary_edges.points
    mesh_kd_tree = cKDTree(mesh_with_normals.points)
    distances, indices = mesh_kd_tree.query(boundary_points)
    if direction=='up':
        boundary_normals = mesh_with_normals['Normals'][indices]
    elif direction=='down':
        boundary_normals = -mesh_with_normals['Normals'][indices]

    projected_points_list = []
    for projection_distance in projection_distance_list:
        projected_points = boundary_points + boundary_normals * projection_distance
        projected_points_list.append(projected_points)

    return np.vstack(projected_points_list)


# +
# projection sum top boundary points
sum_top_proj_bd_pts = project_boundary_points(sum_top_bd_edges, sum_top_surf_normals, [0.001, 0.002, 0.003, 0.004], direction='down')
sum_bot_proj_bd_pts = project_boundary_points(sum_bot_bd_edges, sum_bot_surf_normals, [0.001, 0.002, 0.003, 0.004], direction='up')

# Visualize original and projected boundary points
plotter = pv.Plotter()
plotter.add_mesh(sum_top_surf_mesh_mmg, color='white', opacity=0.5)
plotter.add_mesh(sum_bot_surf_mesh_mmg, color='white', opacity=0.5)

plotter.add_points(sum_top_bd_edges, color='blue', point_size=8, label='Original Boundary Points')
plotter.add_points(sum_top_proj_bd_pts, color='red', point_size=8, label='Projected Boundary Points')

plotter.add_points(sum_bot_bd_edges, color='green', point_size=8, label='Original Boundary Points')
plotter.add_points(sum_bot_proj_bd_pts, color='pink', point_size=8, label='Projected Boundary Points')

plotter.show()

# +
# Create a PyVista point cloud
point_cloud = pv.PolyData(np.vstack((sum_top_proj_bd_pts, sum_bot_proj_bd_pts)))
all_boundary_points_surf_mesh = point_cloud.delaunay_2d(tol=1e-5, alpha=0.01)
all_boundary_points_surf_mesh.save(f"{output_dir}all_boundary_points_surf_mesh.vtk")

# !open ./output/all_boundary_points_surf_mesh.vtk
# -



# +
# Assume 'sum_top_surf_mesh_mmg' is your original UnstructuredGrid
surface_mesh = sum_top_surf_mesh_mmg.extract_surface()

# Now compute normals for the surface
surface_with_normals = surface_mesh.compute_normals(point_normals=True, cell_normals=False)

# Access the computed normals
normals = surface_with_normals['Normals']
print("Computed normals:", normals)

# Visualize mesh with normals
surface_with_normals.plot_normals(mag=0.01, color='red')
# -

sum_top_surf_mesh.compute_normals

boundary_edges.points.shape





import open3d as o3d

# +
# Load or define your point cloud data
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(sum_slab_c_xyz)

# Estimate normals (required for Poisson reconstruction)
pcd.estimate_normals()

# +
# # Perform Poisson reconstruction
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=16)
# o3d.visualization.draw_geometries([mesh])

# Compute the alpha shape (alpha controls the level of detail)
alpha = 0.03  # smaller alpha -> tighter shape, larger alpha -> looser shape
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

# Assume 'mesh' already created
o3d.io.write_triangle_mesh(f"{output_dir}mesh_output.ply", mesh)

# convert ply to vtk
mesh = pv.read(f"{output_dir}mesh_output.ply")
mesh.save(f"{output_dir}mesh_output.vtk")

# +
# Create a PyVista point cloud
point_cloud = pv.PolyData(sum_slab_c_xyz)

# Create a volume mesh using 3D Delaunay triangulation with an alpha parameter
# The 'alpha' parameter controls the level of detail of the alpha shape
volume = point_cloud.delaunay_3d(alpha=9e-3, tol=1e-3, offset=2.5,)

# Extract the surface of the volume mesh to get the alpha shape
alpha_shape = volume.extract_geometry()

# Option 1: Save the full volumetric mesh
volume.save(f"{output_dir}sum_slab_vol_mesh.vtk")
# -



surf = point_cloud.reconstruct_surface(nbr_sz=10, sample_spacing=0.06)
# save vtk
surf.save(f"{output_dir}sum_slab_surf_mesh.vtk")










