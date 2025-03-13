# ### Here we will plot and understand different datasets provided by slab2
#
# *Author: Thyagarajulu Gollapalli*
#
# Notes: 
# 1. PyGMT plotting can be challenging—managing all the parameters to fine-tune plot details can be nontrivial and frustrating. If you don't like peaceful mind then go for it.
# 2. I recommend using Cartopy for a smoother and less frustrating plotting experience.

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cmcrameri import cm
import os
import pyvista as pv
import pygmt

import coordinate_transform as ct
import point_shift as ps


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


def plot_grd(grid, cmap='hawaii', series=[0, 1], cmap_title='Depth (km)'):
    '''
    plot the grid file using pygmt
    '''
    fig = pygmt.Figure()
    pygmt.makecpt(series=series, cmap=cmap, reverse=True) # Define the colormap for the figure
    fig.grdimage(grid=grid, projection="M6i", frame="ag", cmap=True)
    fig.coast(shorelines="1/0.75p", resolution='l')
    fig.colorbar(frame=f'xaf+l{cmap_title}')
    return fig


def plot_grd_subplot(grids, cmap_list, series_list, cmap_title_list, ncols=2):
    """
    Create a subplot figure from multiple grid files using PyGMT.
    Each grid uses its corresponding colormap settings provided in lists.
    """
    nplots = len(grids)
    nrows = (nplots + ncols - 1) // ncols

    fig = pygmt.Figure()
    with fig.subplot(nrows=nrows, ncols=ncols, figsize=("15i", f"{6*nrows}i"),
                     sharex=True, sharey=True, margins=["-1.5c", "2.5c"],):
        for i, grid in enumerate(grids):
            with fig.set_panel(panel=i):
                pygmt.makecpt(series=series_list[i], cmap=cmap_list[i], reverse=True)
                fig.grdimage(grid=grid, projection="M6i", frame="ag", cmap=True)
                fig.coast(shorelines="1/0.75p", resolution="l", projection="M6i")
                fig.colorbar(frame=f"xaf+l{cmap_title_list[i]}",
                             position="jBC+w4i/0.2i+o-1i/-1i")
    return fig


def plot_grd_subplot_cartopy(grids, cmap_list, series_list, cmap_title_list, ncols=2, output_dir_filename='', figsize=(12, 12), 
                             left=0.04, right=0.98, bottom=0.05, top=0.999, wspace=0.2, hspace=0.1):
    """
    Create a subplot figure from multiple xarray DataArrays using Cartopy.
        
    Returns:
        fig (matplotlib.figure.Figure): The resulting figure with subplots.
    """
    # Calculate total number of plots and required number of rows
    nplots = len(grids)
    nrows = math.ceil(nplots / ncols)
    
    # Create a figure and a grid of axes with a Cartopy (PlateCarree) projection
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    # # Adjust spacing: change these values as needed
    plt.subplots_adjust(
                        left=left,   # Space from the left edge of the figure to the subplots (5% of the figure width)
                        right=right,  # Space from the right edge of the figure to the subplots (95% of the figure width)
                        bottom=bottom, # Space from the bottom edge of the figure to the subplots (5% of the figure height)
                        top=top,    # Space from the top edge of the figure to the subplots (95% of the figure height)
                        wspace=wspace, # Horizontal spacing between subplots (negative value causes overlapping)
                        hspace=hspace  # Vertical spacing between subplots (25% of the average subplot height)
                    )
    

    axes = np.atleast_1d(axes).flatten()  # Flatten axes to iterate easily

    # Plot each xarray DataArray on its corresponding axis
    for i, ax in enumerate(axes[:nplots]):
        data_array = grids[i]              # Use the provided xarray DataArray
        vmin, vmax, _ = series_list[i]       # Get normalization limits
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(data_array.x, data_array.y, data_array,
                           cmap=cmap_list[i], norm=norm,
                           transform=ccrs.PlateCarree(), rasterized=True)
        ax.coastlines(resolution='50m')      # Add coastlines for context

        # Add gridlines with labels on left and bottom
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='-')
        gl.top_labels = False
        gl.right_labels = False

        # Create an inset axis for a horizontal colorbar at the bottom of the subplot.
        cax = inset_axes(ax, width="80%", height="5%", loc='lower center',
                         bbox_to_anchor=(0, -0.12, 1, 1),
                         bbox_transform=ax.transAxes, borderpad=0)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label(cmap_title_list[i])
    
    # This loop hides any extra axes that aren't used for a plot
    for ax in axes[nplots:]:
        ax.set_visible(False)

    fig.savefig(output_dir_filename, dpi=250)
        
    return fig

# +
# slab2 data dir

# 2018 files
slab2_dir = '/Users/tgol0006/PlateTectonicsTools/Slab2_AComprehe/Slab2Distribute_Mar2018/'
grd_file_list = ['sum_slab2_dep_02.23.18.grd', 'sum_slab2_thk_02.23.18.grd', 'sum_slab2_dip_02.23.18.grd', 'sum_slab2_str_02.23.18.grd']

# # 2021 files
# slab2_dir = '/Users/tgol0006/PlateTectonicsTools/sum_surf_09.21_slab2_output/'
# grd_file_list = ['sum_slab2_surf_dep_09-21.grd', 'sum_slab2_surf_thk_09-21.grd', 'sum_slab2_surf_dip_09-21.grd', 'sum_slab2_surf_str_09-21.grd']

grids = []
for grd in grd_file_list:
    print_info(grd, slab2_dir)
    grids.append(xr.open_dataarray(f'{slab2_dir}{grd}'))

# grids

# +
# subplot figure

# cmap_list = ["hawaii", "davos", "turku", "tokyo"]
# fig = plot_grd_subplot(grids, cmap_list, series_list, cmap_title_list, ncols=2)
# fig.savefig("sum_dep_thk_dip_str_gmt.pdf")
# fig.show()

if not os.path.isfile('sum_dep_thk_dip_str.pdf'):
    series_list = [[-600, 0, 12], [0, 150, 15], [0, 80, 8], [0, 360, 18]]
    cmap_title_list = ["Depth (km)", "Thickness (km)", "Dip (°)", "Strike (°)"]
    cmap_list = [cm.hawaii.resampled(12), cm.davos_r.resampled(15), cm.turku.resampled(8), cm.tokyo_r.resampled(18)]
    fig = plot_grd_subplot_cartopy(grids, cmap_list, series_list, cmap_title_list, ncols=2, output_dir_filename=f'sum_dep_thk_dip_str.pdf')


# -

# ### Get bottom surface of the slab using above datasets

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

# slab bottom surface array
bot_surf_dep_arr = np.copy(dep_arr)

# layer thickness is negative if it is shifted down and positive if it is shifted upward direction
# top surface depth must be positive
for i, values in enumerate(dep_arr):
    shift_pt = ps.PointShift(values[0], values[1], -values[2], dip_arr[i][2], str_arr[i][2], -thk_arr[i][2])
    bot_surf_dep_arr[i][2] = -shift_pt[2]


def numpy_to_xarray(arr):
    """
    Convert a numpy array to an xarray DataArray.
    """
    x_coords = np.unique(arr[:, 0])
    y_coords = np.unique(arr[:, 1])
    data_2d = arr[:, 2].reshape(len(y_coords), len(x_coords))
    return xr.DataArray(data_2d, coords=[y_coords, x_coords], dims=["y", "x"])


# +
# Convert the NumPy arrays back to xarray DataArrays
bot_surf_dep_xda = numpy_to_xarray(bot_surf_dep_arr)

# Plot the bottom surface depth using Cartopy
if not os.path.isfile('sum_bot_surf_dep.pdf'):
    series_list = [[-700, 0, 14]]
    cmap_title_list = ["Depth (km)"]
    cmap_list = [cm.hawaii.resampled(14)]
    fig = plot_grd_subplot_cartopy([bot_surf_dep_xda], cmap_list, series_list, cmap_title_list, ncols=1, figsize=(6, 6),
                                   output_dir_filename=f'sum_bot_surf_dep.pdf', 
                                   left=0.125, right=0.9, bottom=0.11, top=0.98, wspace=0.2, hspace=0.2)
# -

# ### Transforming the geo llr to cubed sphere xyz

# +
# Get the minimum and maximum values in lon and lat
lon_min = np.min(bot_surf_dep_arr[:, 0])
lon_max = np.max(bot_surf_dep_arr[:, 0])
lat_min = np.min(bot_surf_dep_arr[:, 1])
lat_max = np.max(bot_surf_dep_arr[:, 1])

print(lon_min, lon_max, lat_min, lat_max)
# -

coord_trans = ct.CoordinateTransformCubedsphere(g_lon_min=lon_min, 
                                                g_lon_max=lon_max,
                                                g_lat_min=lat_min, 
                                                g_lat_max=lat_max,)


def remove_nan_rows(arr):
    '''
    Remove rows containing NaN values from the given NumPy array.
    '''
    return arr[~np.isnan(arr).any(axis=1)]


# +
# Remove rows containing NaN values from the bottom surface depth array and convert depth to positive values
bot_surf_pos_dep_arr = remove_nan_rows(bot_surf_dep_arr)
bot_surf_pos_dep_arr[:, 2] *= -1
print(bot_surf_pos_dep_arr)

top_surf_pos_dep_arr = remove_nan_rows(dep_arr)
top_surf_pos_dep_arr[:, 2] *= -1
print(top_surf_pos_dep_arr)
# -

sph_top_surf = np.copy(dep_arr)
sph_top_surf[:, 2] = 0.0
sph_top_surf

# +
# get sum slab surface top, bottom in cubedsphere xyz 
sum_top_surf_c_xyz = coord_trans.geo_llr_to_cubedsphere_xyz(top_surf_pos_dep_arr)
print(sum_top_surf_c_xyz)

sum_bot_surf_c_xyz = coord_trans.geo_llr_to_cubedsphere_xyz(bot_surf_pos_dep_arr)
print(sum_bot_surf_c_xyz)

sph_top_surf_c_xyz = coord_trans.geo_llr_to_cubedsphere_xyz(sph_top_surf)
print(sph_top_surf_c_xyz)
# -

np.round(coord_trans.cubedsphere_xyz_to_geo_llr(sph_top_surf_c_xyz), 2)

# +
# note: change radius to depth in the coordinate transform class
# -

# ### Cloud points and Alpha Shapes

# +
# Convert the points to a PyVista point cloud
sum_top_surf_cloud = pv.PolyData(sum_top_surf_c_xyz)
sum_bot_surf_cloud = pv.PolyData(sum_bot_surf_c_xyz)
sph_top_surf_cloud = pv.PolyData(sph_top_surf_c_xyz)

# Create a PyVista plotter
plotter = pv.Plotter()

# Add the point clouds to the plotter
plotter.add_points(sum_top_surf_cloud, color='red', point_size=2)
plotter.add_points(sum_bot_surf_cloud, color='blue', point_size=2)
plotter.add_points(sph_top_surf_cloud, color='green', point_size=2, opacity=0.1)


# Set the background color
plotter.background_color = 'white'

# Show the plotter
plotter.show()
# -

sum_top_bot_surf_c_xyz = np.vstack((sum_top_surf_c_xyz, sum_bot_surf_c_xyz))

# +
# Create a PyVista point cloud
point_cloud = pv.PolyData(sum_top_bot_surf_c_xyz)

# Create a volume mesh using 3D Delaunay triangulation with an alpha parameter
# The 'alpha' parameter controls the level of detail of the alpha shape
volume = point_cloud.delaunay_3d(alpha=7.5e-3, tol=1e-3, offset=2.5,)

# Extract the surface of the volume mesh to get the alpha shape
alpha_shape = volume.extract_geometry()

# Option 1: Save the full volumetric mesh
volume.save("sum_slab_mesh.vtk")
# -

# surface edges
edges = volume.extract_all_edges()
# edges.plot(line_width=5, color='k')

# Visualize the alpha shape along with the original points
plotter = pv.Plotter()
plotter.add_mesh(edges)
plotter.add_mesh(alpha_shape, color='lightblue', opacity=0.7)
# plotter.add_points(point_cloud, color='red', point_size=5)
plotter.show()

# +
# try open3d alpha shape
