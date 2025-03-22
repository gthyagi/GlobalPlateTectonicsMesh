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

depth_values_list


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
# # project the boundary by constant depth

# # layer thickness is negative if it is shifted down and positive if it is shifted upward direction
# # top surface depth must be positive

# all_boundary_points_list = []
# for depth in depth_values_list:
#     print('Projecting to depth: ', depth,'km')
#     boundary_points_at_depth = np.zeros_like(dep_arr[boundary_points_indx])
#     for i, index in enumerate(boundary_points_indx):
#         shift_pt = ps.PointShift(dep_arr[index][0], dep_arr[index][1], -dep_arr[index][2], dip_arr[index][2], str_arr[index][2], -depth)
#         boundary_points_at_depth[i][0] = shift_pt[0]
#         boundary_points_at_depth[i][1] = shift_pt[1]
#         boundary_points_at_depth[i][2] = shift_pt[2] # obtained positive depth. no need for conversion.
#     all_boundary_points_list.append(boundary_points_at_depth)

# # Convert to numpy array
# all_boundary_points_arr = np.vstack(all_boundary_points_list)
# +
# project the boundary by constant depth

# layer thickness is negative if it is shifted down and positive if it is shifted upward direction
# top surface depth must be positive

all_boundary_points_list = []
for depth in depth_values_list:
    print('Projecting to depth: ', depth,'km')
    # boundary_points_at_depth = np.zeros_like(dep_arr[boundary_points_indx])
    for i, index in enumerate(boundary_points_indx):
        shift_pt = ps.PointShift(dep_arr[index][0], dep_arr[index][1], -dep_arr[index][2], dip_arr[index][2], str_arr[index][2], -depth)
        if depth <= thk_arr[index][2]:
            # boundary_points_at_depth[i][0] = shift_pt[0]
            # boundary_points_at_depth[i][1] = shift_pt[1]
            # boundary_points_at_depth[i][2] = shift_pt[2] # obtained positive depth. no need for conversion.
            all_boundary_points_list.append(shift_pt)
    # all_boundary_points_list.append(boundary_points_at_depth)

# Convert to numpy array
# all_boundary_points_arr = np.vstack(all_boundary_points_list)
all_boundary_points_arr = np.array(all_boundary_points_list)
# -

all_boundary_points_arr[:,2].min()

# +
# # layer thickness is negative if it is shifted down and positive if it is shifted upward direction
# # top surface depth must be positive

# all_boundary_points_list = []

# for i, index in enumerate(boundary_points_indx):
#     shift_pt = ps.PointShift(dep_arr[index][0], dep_arr[index][1], -dep_arr[index][2], dip_arr[index][2], str_arr[index][2], -thk_arr[index][2])
#     print(-dep_arr[index][2], shift_pt[2])

#     res = 5
#     if -dep_arr[index][2] < shift_pt[2]:
#         depth_arr = np.round(np.arange(-dep_arr[index][2]+res, shift_pt[2], res), 0)
#     else:
#         print('reverse case')
#         depth_arr = np.round(np.arange(shift_pt[2]+res, -dep_arr[index][2], res), 0)
        
#     for depth in depth_arr:
#         bd_shift_pt = ps.PointShift(dep_arr[index][0], dep_arr[index][1], -dep_arr[index][2], dip_arr[index][2], str_arr[index][2], -depth)
    
#         all_boundary_points_list.append(bd_shift_pt)

# # Convert to numpy array
# all_boundary_points_arr = np.array(all_boundary_points_list)

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

# +
# # slab bottom surface array
# bot_surf_dep_arr_nan = np.zeros_like(dep_arr)
# all_boundary_points_list = []

# # layer thickness is negative if it is shifted down and positive if it is shifted upward direction
# # top surface depth must be positive
# for i, values in enumerate(dep_arr):
#     shift_pt = ps.PointShift(values[0], values[1], -values[2], dip_arr[i][2], str_arr[i][2], -thk_arr[i][2])
#     bot_surf_dep_arr_nan[i][0] = shift_pt[0]
#     bot_surf_dep_arr_nan[i][1] = shift_pt[1]
#     bot_surf_dep_arr_nan[i][2] = shift_pt[2]
    
#     for index in boundary_points_indx:
#         if i==index:
#             res = 3
#             depth_arr = np.round(np.arange(-values[2] + res, shift_pt[2], res), 0)
#             for depth in depth_arr:
#                 bd_shift_pt = ps.PointShift(dep_arr[index][0], dep_arr[index][1], -dep_arr[index][2], dip_arr[index][2], str_arr[index][2], -depth)
#                 # ps.PointShift(values[0], values[1], -values[2], dip_arr[i][2], str_arr[i][2], -depth)
#                 all_boundary_points_list.append(bd_shift_pt)

# # Convert to numpy array
# bot_surf_dep_arr = remove_nan_rows(bot_surf_dep_arr_nan)
# all_boundary_points_arr = np.array(all_boundary_points_list)
# -

bot_surf_dep_arr

# ### Transforming the geo lld to cubed sphere xyz

# +
# Get the minimum and maximum values in lon and lat
lon_min = np.min(slab2_data[:, 0])
lon_max = np.max(slab2_data[:, 0])
lat_min = np.min(slab2_data[:, 1])
lat_max = np.max(slab2_data[:, 1])

print(lon_min, lon_max, lat_min, lat_max)
# -

coord_trans = ct.CoordinateTransformCubedsphere(g_lon_min=lon_min, 
                                                g_lon_max=lon_max,
                                                g_lat_min=lat_min, 
                                                g_lat_max=lat_max,)

sum_top_surf_c_xyz = coord_trans.geo_lld_to_cubedsphere_xyz(sum_dep_arr)
print(sum_top_surf_c_xyz)

sum_bot_surf_c_xyz = coord_trans.geo_lld_to_cubedsphere_xyz(bot_surf_dep_arr)
print(sum_bot_surf_c_xyz)

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

sum_slab_c_xyz = np.vstack((sum_top_surf_c_xyz, all_boundary_points_c_xyz, sum_bot_surf_c_xyz))

sum_slab_c_xyz

sum_slab_c_xyz.shape

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







# Create a PyVista point cloud
point_cloud = pv.PolyData(sum_top_surf_c_xyz)
top_surf = point_cloud.delaunay_2d(alpha=0.003)
top_surf.save(f"{output_dir}sum_top_surf_mesh.vtk")

# !open ./output/sum_top_surf_mesh.vtk

# Create a PyVista point cloud
point_cloud = pv.PolyData(sum_bot_surf_c_xyz)
bot_surf = point_cloud.delaunay_2d(alpha=0.003)
bot_surf.save(f"{output_dir}sum_bot_surf_mesh.vtk")

# !open ./output/sum_bot_surf_mesh.vtk

# Create a PyVista point cloud
point_cloud = pv.PolyData(all_boundary_points_c_xyz)
surf = point_cloud.delaunay_2d(tol=1e-5, alpha=0.01)
# save vtk
surf.save(f"{output_dir}all_boundary_points_surf_mesh.vtk")

# !open ./output/all_boundary_points_surf_mesh.vtk


