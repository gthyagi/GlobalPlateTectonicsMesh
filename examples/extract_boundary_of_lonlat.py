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
import tetgen
import gmsh

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

# depth list
depth_values_list = np.linspace(5, 200, 40)
print(depth_values_list)

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
sum_top_surf_mesh = sum_top_surf_cloud.delaunay_2d(alpha=0.0035) # manual picking

if not os.path.isfile(f"{output_dir}sum_top_surf_mesh.vtk"):
    sum_top_surf_mesh.save(f"{output_dir}sum_top_surf_mesh.vtk")
else:
    print('Slab top surface mesh file exists!')

os.system('open ./output/sum_top_surf_mesh.vtk')

# Ensure the surface mesh has no boundary holes before passing it to MMG.
check_irregular_boundary_points(sum_top_surf_mesh)

# mmg parameters
hmax = 0.02 # 0.0015
hmin = 0.019 # 0.0014

os.remove('./output/sum_top_surf_mesh_mmg.vtk')

# run mmg
if not os.path.isfile(f'{output_dir}sum_top_surf_mesh_mmg.vtk'):
    save_pyvista_to_mesh(sum_top_surf_mesh, f'{output_dir}sum_top_surf_mesh.mesh')
    run_mmgs_remesh(f'{output_dir}sum_top_surf_mesh.mesh', f'{output_dir}sum_top_surf_mesh_mmg.mesh', 
                    hmax=hmax, hmin=hmin, hausd=None)
    convert_mesh_to_vtk(f'{output_dir}sum_top_surf_mesh_mmg.mesh', f'{output_dir}sum_top_surf_mesh_mmg.vtk')
else:
    print('Slab top surface mmg mesh file exists!')

# view vtk in paraview
os.system('open ./output/sum_top_surf_mesh_mmg.vtk')

# +
# creating slab bottom surface mesh

# create alphashape from point cloud
sum_bot_surf_mesh = sum_bot_surf_cloud.delaunay_2d(alpha=0.0029)

if not os.path.isfile(f"{output_dir}sum_bot_surf_mesh.vtk"):
    sum_bot_surf_mesh.save(f"{output_dir}sum_bot_surf_mesh.vtk")
else:
    print('Slab bottom surface mesh file exists!')

os.system('open ./output/sum_bot_surf_mesh.vtk')

# checking for holes in the surface
check_irregular_boundary_points(sum_bot_surf_mesh)

# mmg parameters
hmax = 0.02 # 0.0015
hmin = 0.019 # 0.0014

os.remove('./output/sum_bot_surf_mesh_mmg.vtk')

# run mmg
if not os.path.isfile(f'{output_dir}sum_bot_surf_mesh_mmg.vtk'):
    save_pyvista_to_mesh(sum_bot_surf_mesh, f'{output_dir}sum_bot_surf_mesh.mesh')
    run_mmgs_remesh(f'{output_dir}sum_bot_surf_mesh.mesh', f'{output_dir}sum_bot_surf_mesh_mmg.mesh', 
                    hmax=hmax, hmin=hmin, hausd=None)
    convert_mesh_to_vtk(f'{output_dir}sum_bot_surf_mesh_mmg.mesh', f'{output_dir}sum_bot_surf_mesh_mmg.vtk')
else:
    print('Slab bottom surface mmg mesh file exists!')

# view vtk in paraview
os.system('open ./output/sum_bot_surf_mesh_mmg.vtk')
# +
# # Create a PyVista point cloud
# point_cloud = pv.PolyData(all_boundary_points_c_xyz)
# all_boundary_points_surf_mesh = point_cloud.delaunay_2d(tol=1e-5, alpha=0.01)
# all_boundary_points_surf_mesh.save(f"{output_dir}all_boundary_points_surf_mesh.vtk")

# # !open ./output/all_boundary_points_surf_mesh.vtk

# check_irregular_boundary_points(all_boundary_points_surf_mesh)

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

# Load mmg mesh
sum_top_surf_mesh_mmg = pv.read(f'{output_dir}sum_top_surf_mesh_mmg.vtk')
sum_bot_surf_mesh_mmg = pv.read(f'{output_dir}sum_bot_surf_mesh_mmg.vtk')

# +
# Compute normals for the mesh points
sum_top_surf_normals = sum_top_surf_mesh_mmg.extract_surface().compute_normals(point_normals=True, cell_normals=False)
sum_bot_surf_normals = sum_bot_surf_mesh_mmg.extract_surface().compute_normals(point_normals=True, cell_normals=False)

# Visualize mesh with normals
sum_top_surf_normals.plot_normals(mag=0.01, color='red')

# +
# create volume from slab top surface

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
# side_mesh.plot()

if closed_shell.is_manifold:
    print('Created volume is good for further operations')
else:
    print('Created volume is not closed')
    
closed_shell_2 = closed_shell.triangulate()

os.remove(f"{output_dir}sum_vol_from_top_surf.vtk")

if not os.path.isfile(f"{output_dir}sum_vol_from_top_surf.vtk"):
    print(closed_shell_2.is_all_triangles)
    
    closed_shell_2.save(f"{output_dir}sum_vol_from_top_surf.vtk")
    save_pyvista_to_mesh(closed_shell_2, f"{output_dir}sum_vol_from_top_surf.mesh")
else:
    print('volume already exists')

os.system(f'open ./output/sum_vol_from_top_surf.vtk')
# -
# create volume mesh with TetGen
tet = tetgen.TetGen(closed_shell_2)
volume = tet.tetrahedralize()

# save as vtk
slab_grid = tet.grid
slab_grid.save(f"{output_dir}sum_vol_from_top_surf_tet.vtk")
os.system(f'open {output_dir}sum_vol_from_top_surf_tet.vtk')

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
meshio.write(f"{output_dir}sum_vol_from_top_surf_tet.msh", mesh, file_format="gmsh")

# +
# slab = pv.read(f"{output_dir}sum_vol_from_top_surf_tet.msh")
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




