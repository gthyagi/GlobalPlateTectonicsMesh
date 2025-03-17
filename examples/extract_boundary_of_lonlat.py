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
data = np.loadtxt(f'{slab2_dir}sum_slab2_dep_02.23.18.xyz', delimiter=',')
print(data)

# Remove rows containing NaN values and convert depth to positive values
sum_dep_arr = remove_nan_rows(data)
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

# Open the file in browser
webbrowser.open(file_path.as_uri())


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

depth_values_list = np.linspace(3, 90, 30)


def find_matching_indices(dep_arr, boundary_points):
    # Define structured dtype explicitly (assuming columns are lon, lat, depth)
    dtype = [('lon', float), ('lat', float)]

    # Convert to structured arrays
    dep_struct = np.array([tuple(np.round(row, 2)) for row in dep_arr[:,0:2]], dtype=dtype)
    boundary_struct = np.array([tuple(np.round(p,2)) for p in boundary_points[:,0:2]], dtype=dtype)

    # Use np.in1d to find matching rows
    mask = np.in1d(dep_struct, boundary_struct)
    
    # Get the indices of matching rows
    indices = np.where(mask)[0]

    return indices



# matches only lon, lat columns
boundary_points_indx = find_matching_indices(dep_arr, boundary_points)
print(boundary_points_indx.shape)

dep_arr[boundary_points_indx]

# add boundary points to the plot
fig_slab, ax_slab = create_scatter_plot(sum_dep_arr, plt.cm.viridis_r.resampled(14), 0, 700)
sc_1 = ax_slab.scatter(dep_arr[boundary_points_indx][:,0], dep_arr[boundary_points_indx][:,1], 
                       c='r', s=1, edgecolor=None, transform=ccrs.PlateCarree(),)



# +
# layer thickness is negative if it is shifted down and positive if it is shifted upward direction
# top surface depth must be positive

all_boundary_points_list = []
for depth in depth_values_list:
    print('Projecting to depth: ',depth)
    boundary_points_at_depth = np.zeros_like(dep_arr[boundary_points_indx])
    for i, indice in enumerate(boundary_points_indx):
        shift_pt = ps.PointShift(dep_arr[indice][0], dep_arr[indice][1], -dep_arr[indice][2], dip_arr[indice][2], str_arr[indice][2], -thk_arr[indice][2])
        boundary_points_at_depth[i][0] = shift_pt[0]
        boundary_points_at_depth[i][1] = shift_pt[1]
        boundary_points_at_depth[i][2] = -shift_pt[2]
    all_boundary_points_list.append(boundary_points_at_depth)
# -

all_boundary_points_list


