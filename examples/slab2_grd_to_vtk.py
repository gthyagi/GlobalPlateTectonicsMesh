# ## Convert slab2 data to vtk file

# You can download the Slab2 dataset (Slab2Distribute_Mar2018.tar.gz) from here:
#
# https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467#:~:text=Slab2Distribute_Mar2018.tar.gz

#
# **Table 1.** Subduction zone models included in Slab2, showing their shallow (Ss) and deep (Sd) seismogenic zone limits and corresponding seismogenic zone width (Sw). δ, Φ, and λ represent the average subduction zone interface dip, strike, and rake, respectively, in the seismogenic zone. *Grey italicized numbers* are considered poorly constrained (number of contributing earthquakes N < 50). Mₘₐₓ is the maximum historically observed magnitude; *¹ indicates the model named “mex” in Slab 1.0.*
#
# This table is sourced from the Supplementary Information of the Slab2 paper (https://www.science.org/doi/10.1126/science.aat4723)

# | #  | Subduction Zone Arc          | Slab2 Code |   N  | Ss (km) | Sd (km) | Sw (km) | δ (º) | Φ (º) | λ (º) | Mmax |
# |----|------------------------------|-----------:|-----:|--------:|--------:|--------:|------:|------:|------:|-----:|
# | 1  | Aleutians                    |        alu |  470 |      12 |      45 |     124 |    14 |   265 |   124 |  9.2 |
# |    | Alaska                       |       alu1 |   42 |      11 |      45 |     193 |    11 |   233 |   195 |  9.2 |
# |    | Central Aleutians            |       alu2 |  345 |      13 |      46 |     110 |    17 |   260 |   104 |  8.6 |
# |    | West Aleutians               |       alu3 |   72 |      12 |      35 |      64 |    15 |   288 |   125 |  8.7 |
# | 2  | Calabria                     |        cal |    2 |     nan |     nan |         |       |       |       |      |
# | 3  | Central America              |    cam(*1) |  701 |      11 |      42 |      91 |    20 |   301 |    86 |  8.1 |
# |    | Mexico                       |       cam1 |  117 |      10 |      33 |      77 |    18 |   298 |    86 |  8.1 |
# |    | El Salvador                  |       cam2 |  585 |      12 |      44 |      86 |    22 |   301 |    86 |  7.8 |
# | 4  | Caribbean                    |        car |   41 |      15 |      51 |         |       |       |       |      |
# |    | Lesser Antilles              |            |      |         |         |         |       |       |       |      |
# |    | Puerto Rico                  |            |      |         |         |         |       |       |       |      |
# | 5  | Cascadia                     |        cas |   21 |       6 |      46 |         |       |       |       |      |
# | 6  | Cotabato                     |        cot |   23 |      13 |      40 |         |       |       |       |      |
# | 7  | Halmahera                    |        hal |  N/A |     N/A |         |         |       |       |       |      |
# | 8  | Hellenic                     |        hel |   22 |      15 |      57 |         |       |       |       |      |
# |    | Greece                       |            |      |         |         |         |       |       |       |      |
# |    | Cyrus                        |            |      |         |         |         |       |       |       |      |
# | 9  | Himalaya                     |        him |   21 |      10 |      31 |         |       |       |       |      |
# | 10 | Hindu Kush                   |        hin |  N/A |     N/A |         |         |       |       |       |      |
# | 11 | Izu-Bonin                    |        izu |  218 |      10 |      37 |      96 |    15 |   174 |    93 |  7.5 |
# |    | Izu                          |       izu1 |  111 |      10 |      39 |     104 |    15 |   168 |    91 |  7.5 |
# |    | North Mariana                |       izu2 |   42 |      10 |      40 |     103 |    16 |   145 |    90 |  7.2 |
# |    | South Mariana                |       izu3 |   65 |       9 |      32 |     105 |    13 |   261 |   106 |  7.0 |
# | 12 | Kermadec                     |        ker |  707 |      10 |      47 |     129 |    16 |   205 |    93 | 8.3* |
# |    | Tonga                        |       ker1 |  261 |       9 |      31 |     104 |    12 |   192 |    86 | 8.3* |
# |    | Central Kermadec             |       ker2 |  410 |      11 |      51 |     109 |    19 |   201 |    94 |  8.0 |
# |    | New Zealand                  |       ker3 |   34 |      11 |      38 |         |       |       |       |      |
# | 13 | Kuril                        |        kur | 1162 |      13 |      54 |     138 |    15 |   227 |    89 |  9.1 |
# |    | Kamchatka                    |       kur1 |  228 |      16 |      52 |     100 |    16 |   228 |    86 |  9.0 |
# |    | Central Kuril                |       kur2 |   81 |      10 |      49 |     131 |    13 |   230 |   103 |  8.3 |
# |    | South Kuril                  |       kur3 |  138 |      11 |      45 |     124 |    14 |   242 |   104 |  8.5 |
# |    | Japan                        |       kur4 |  721 |      13 |      56 |     174 |    11 |   212 |    85 |  9.1 |
# | 14 | Makran                       |        mak |   17 |      13 |      40 |         |       |       |       |      |
# | 15 | Manila                       |        man |   90 |      14 |      48 |      69 |    29 |   130 |    89 |  7.2 |
# | 16 | Muertos                      |        mue |    2 |     nan |     nan |         |       |       |       |      |
# | 17 | Pamir                        |        pam |  N/A |     N/A |         |         |       |       |       |      |
# | 18 | New Guinea                   |        png |   81 |      10 |      41 |     191 |     9 |   106 |    89 |  8.2 |
# | 19 | Philippines                  |        phi |  121 |      11 |      49 |      79 |    28 |   142 |    76 |  7.8 |
# |    | North Philippines            |            |   4  |      nan|      nan|         |       |       |       |      |
# |    | Central Philippines          |       phi2 |  103 |      10 |      50 |      90 |    26 |   145 |    77 |  7.8 |
# |    | South Philippines            |            |   22 |      12 |      53 |         |       |       |       |      |
# | 20 | Puysegur                     |        puy |   20 |      11 |      30 |         |       |       |       |      |
# | 21 | Ryukyu                       |        ryu |  265 |      13 |      47 |     111 |    17 |   236 |    99 |  8.8 |
# |    | Nankai                       |            |   28 |      13 |      51 |         |       |       |       |      |
# |    | Central Ryukyu               |       ryu2 |  217 |      12 |      45 |     119 |    15 |   222 |    97 |      |
# |    | South Ryukyu                 |            |   23 |      12 |      37 |         |       |       |       |      |
# | 22 | South America                |        sam | 1370 |      11 |      45 |     118 |    15 |   356 |    85 |  9.5 |
# |    | Colombia                     |       sam7 |   83 |      10 |      41 |     111 |    16 |   356 |    79 |  8.8 |
# |    | Ecuador-Peru                 |            |      |         |         |         |       |       |       |      |
# |    | Central Peru                 |            |      |         |         |         |       |       |       |      |
# |    | Southern Peru                |       sam4 |  138 |      13 |      40 |      93 |    17 |   314 |    93 |  8.8 |
# |    | Northern Chile               |       sam5 |  771 |      11 |      48 |     129 |    15 |     7 |    86 |  8.8 |
# |    | Southern Chile               |       sam6 |  352 |      11 |      38 |     101 |    12 |    15 |    84 |  9.5 |
# | 23 | Scotia                       |        sco |   70 |      11 |      46 |      87 |    17 |   175 |   109 |  7.5 |
# | 24 | Solomon Islands              |        sol |  627 |      12 |      53 |      77 |    33 |   287 |    96 |  8.0 |
# |    | Solomon                      |       sol1 |   72 |      11 |      41 |      57 |    32 |   297 |    96 |  8.0 |
# |    | Bougainville                 |       sol2 |  245 |      14 |      54 |      64 |    36 |   310 |    92 |  8.0 |
# |    | New Britain                  |       sol3 |   297|      13 |      52 |      88 |    26 |   250 |    99 |  7.8 |
# | 25 | Sulawesi                     |       sul  |   42 |      17 |      44 |         |       |       |       |      |
# | 26 | Sumatra/Java                 |        sum |  706 |      11 |      53 |     153 |    17 |   281 |   100 |  9.1 |
# |    | Andaman Islands              |       sum1 |  209 |      10 |      53 |     167 |    15 |   345 |    93 |  9.1 |
# |    | Sumatra                      |       sum2 |  365 |      12 |      51 |     153 |    14 |   320 |    99 |  9.1 |
# |    | Java                         |       sum3 |  100 |      13 |      57 |     165 |    16 |   284 |   115 |  7.8 |
# |    | Timor                        |            |    23|       12|      44 |         |       |       |       |      |
# |    | Maluku                       |            |    21|       12|      28 |         |       |       |       |      |
# | 27 | Vanuatu                      |        van |  467 |      11 |      49 |      75 |    29 |   322 |    90 |  8.0 |
#

# **Table 2.** Subduction zone arc and Slab2 code

# | #  | Subduction Zone Arc     | Slab2 Code |
# |----|-------------------------|-----------:|
# | 1  | Aleutians               |        alu |
# | 2  | Calabria                |        cal |
# | 3  | Central America         |        cam |
# | 4  | Caribbean               |        car |
# | 5  | Cascadia                |        cas |
# | 6  | Cotabato                |        cot |
# | 7  | Halmahera               |        hal |
# | 8  | Hellenic (Greece)       |        hel |
# | 9  | Himalaya                |        him |
# | 10 | Hindu Kush              |        hin |
# | 11 | Izu-Bonin               |        izu |
# | 12 | Kermadec                |        ker |
# | 13 | Kuril                   |        kur |
# | 14 | Makran                  |        mak |
# | 15 | Manila                  |        man |
# | 16 | Muertos                 |        mue |
# | 17 | Pamir                   |        pam |
# | 18 | New Guinea              |        png |
# | 19 | Philippines             |        phi |
# | 20 | Puysegur                |        puy |
# | 21 | Ryukyu                  |        ryu |
# | 22 | South America           |        sam |
# | 23 | Scotia                  |        sco |
# | 24 | Solomon Islands         |        sol |
# | 25 | Sulawesi                |        sul |
# | 26 | Sumatra/Java            |        sum |
# | 27 | Vanuatu                 |        van |

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import coordinate_transform as ct
import pyvista as pv
from collections import defaultdict
from matplotlib.lines import Line2D

# +
# Specify the directory containing your .grd files
target_dir = '/Users/tgol0006/PlateTectonicsTools/Slab2_AComprehe/Slab2Distribute_Mar2018'
# target_dir = '/Users/tgol0006/Downloads/hal_surf_09.21_slab2_output'

# Build the search pattern
pattern = os.path.join(target_dir, '*.grd')

# Find all .grd files in the target directory
grd_files = glob.glob(pattern)

# Extract the first three characters of each filename (basename)
prefixes = {os.path.basename(f)[:3] for f in grd_files}

# Convert to a sorted list
slab2_code_list = sorted(prefixes)

print(slab2_code_list)


# -

def remove_nan_rows(arr):
    '''
    Remove rows containing NaN values from the given NumPy array.
    '''
    return arr[~np.isnan(arr).any(axis=1)]


def load_grd_to_arr(
    file_path: str,
    engine: str = 'netcdf4',
    positive_z: bool = False,
    remove_nan: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Load a GMT .grd file and convert it to an (N, 3) NumPy array of (lon, lat, z),
    with NaN rows removed and optional conversion of the z column to positive values.
    
    Parameters:
    - file_path: Path to the .grd file.
    - engine: Backend engine for xarray.open_dataset (e.g., 'netcdf4', 'scipy', 'h5netcdf').
    - positive_z: If True, convert z values (depth) to positive via absolute value.
    - remove_nan: Remove nan rows
    - verbose: If True, print debug information.
    
    Returns:
    - A NumPy array of shape (n_points, 3).
    """
    # Open as xarray Dataset
    ds = xr.open_dataset(file_path, engine=engine)
    if verbose:
        print(ds)
    
    # Extract coordinate arrays
    lons = ds['x'].values
    lats = ds['y'].values
    
    # Identify the main grid variable
    varname = list(ds.data_vars)[0]
    zs = ds[varname].values
    
    # Create 2D meshgrid
    lon2d, lat2d = np.meshgrid(lons, lats)
    
    # Stack into (n_points, 3)
    lon_lat_z = np.column_stack((lon2d.ravel(), lat2d.ravel(), zs.ravel()))
    if verbose:
        print("Raw stacked array: \n", lon_lat_z)
    
    # Remove rows with NaNs
    if remove_nan:
        cleaned = remove_nan_rows(lon_lat_z)
        if verbose:
            print("After removing NaNs: \n", cleaned)
    
    # Optionally convert z values to positive
    if positive_z:
        cleaned[:, 2] = np.abs(cleaned[:, 2])
        if verbose:
            print("Applied absolute to z column: \n", cleaned)
    
    return cleaned


# +
# Build the search pattern
pattern = os.path.join(target_dir, '*_dep_*.grd')

# Find all depth .grd files in the target directory
dep_grd_files = glob.glob(pattern)

# create slab dep dict
slab_dep_dict = {}
for file_path in dep_grd_files:
    key = os.path.basename(file_path)[:3]
    slab_dep_dict[key] = load_grd_to_arr(file_path, positive_z=True, verbose=False)

# # Example: inspect keys and one array
# print("Keys:", list(slab_dep_dict.keys()))
# print("Array for 'alu':", slab_dep_dict.get('alu'))

# +
# Load clip boundaries
clip_dir = target_dir+'/Slab2Clips'
pattern_clp = os.path.join(clip_dir, '*_clp_*.csv')
clip_files = glob.glob(pattern_clp)

clip_dict = {}
for f in clip_files:
    key = os.path.basename(f)[:3]
    clip_dict[key] = np.loadtxt(f)


# -

def create_scatter_with_clips(
    data_dict,
    clip_dict,
    depth_cmap='viridis',
    vmin=0,
    vmax=700,
    legend_cmap='nipy_spectral',
    colorbar_title='Depth (km)',
    central_lon=0
):
    """
    Plot slab point‐clouds colored by depth on a Robinson map,
    and overlay each slab’s clip boundary in a unique color.

    Parameters
    ----------
    data_dict : dict[str, np.ndarray]
        { key: (N,3) array of [lon, lat, depth] }
    clip_dict : dict[str, np.ndarray]
        { key: (M,2) array of [lon, lat] boundary coords }
    depth_cmap : str
        Colormap for the scatter points (continuous).
    vmin, vmax : float
        Colorbar limits for depth.
    legend_cmap : str
        A qualitative colormap name for boundary colors.
    colorbar_title : str
        Label for the depth colorbar.
    central_lon : float
        Central meridian for the Robinson projection.
    """
    # Set up depth normalizer & color­map
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=depth_cmap, norm=norm)
    sm.set_array([])  # for the colorbar

    # Qualitative colors for boundaries
    keys = list(clip_dict.keys())
    n    = len(keys)
    qual = plt.get_cmap(legend_cmap, n)(np.arange(n))

    # Create figure & map
    fig = plt.figure(figsize=(12, 6))
    ax  = plt.axes(projection=ccrs.Robinson(central_longitude=central_lon))
    ax.set_global()
    ax.coastlines(resolution='110m', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.7)
    gl.top_labels = gl.right_labels = False

    # 1) scatter all slabs colored by depth
    for arr in data_dict.values():
        ax.scatter(
            arr[:,0], arr[:,1],
            c=arr[:,2],
            cmap=depth_cmap,
            norm=norm,
            s=1,
            transform=ccrs.PlateCarree()
        )

    # 2) overlay each clip boundary
    handles, labels = [], []
    for i, key in enumerate(keys):
        boundary = clip_dict[key]
        ax.plot(
            boundary[:,0], boundary[:,1],
            transform=ccrs.PlateCarree(),
            color=qual[i],
            linewidth=1.5
        )
        handles.append(Line2D([0],[0], color=qual[i], lw=2))
        labels.append(key)

    # 3) legend for boundaries in three rows at bottom
    ncol = int(np.ceil(n/3))
    ax.legend(
        handles, labels,
        loc='lower center',
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.25),
        frameon=False
    )

    # 4) colorbar for depth
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label(colorbar_title)

    plt.show()



# plot slab depth
create_scatter_with_clips(
    slab_dep_dict,
    clip_dict,
    depth_cmap='viridis',
    vmin=0, vmax=700,
    legend_cmap='Reds',
    central_lon=180
)

# +
# visualize slab xyz point cloud
pl = pv.Plotter()
pl.background_color = 'white'

# slab surface top
for key, slab_dep_arr in slab_dep_dict.items():
    slab_top_surf_xyz = ct.CoordinateTransformSphere().lld_to_xyz(slab_dep_arr)
    
    # Convert the points to a PyVista point cloud
    slab_top_surf_pc = pv.PolyData(slab_top_surf_xyz)
    slab_top_surf_pc['depth'] = slab_dep_arr[:, 2]
    
    # Add it, using the 'depth' array to color points
    pl.add_points(
        slab_top_surf_pc,
        scalars='depth',
        cmap=plt.cm.viridis.resampled(14),   
        point_size=1,
        render_points_as_spheres=True,
        clim=[0, 700]
    )

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


# create alphashape from point cloud
for key, slab_dep_arr in slab_dep_dict.items():
    print(f"Processing slab: {key}")
    
    # Convert lon/lat/depth to XYZ coordinates
    slab_top_surf_xyz = ct.CoordinateTransformSphere().lld_to_xyz(slab_dep_arr)
    
    # Create a PyVista point cloud
    slab_top_surf_pc = pv.PolyData(slab_top_surf_xyz)
    
    # Generate a 2D mesh via Delaunay triangulation with alpha parameter
    slab_top_surf_mesh = slab_top_surf_pc.delaunay_2d(alpha=0.0035)
    
    # # Optionally save the mesh for each slab
    # mesh_filename = os.path.join(output_dir, f"{key}_slab_top_surf_mesh.vtk")
    # if not os.path.isfile(mesh_filename):
    #     slab_top_surf_mesh.save(mesh_filename)
    #     print(f"Saved mesh for {key} to {mesh_filename}")
    # else:
    #     print(f"Mesh for {key} already exists: {mesh_filename}")
    
    # Check for boundary irregularities
    check_irregular_boundary_points(slab_top_surf_mesh)
    print('\n')

# +
# alu=0.003
# sul=0.0025
# puy=0.0025
# sam=0.0025
# hel=0.0025
# ker=
# hal=
# -

# create alphashape from point cloud
for key, slab_dep_arr in slab_dep_dict.items():
    if key=='ker':
        print(f"Processing slab: {key}")
        
        # Convert lon/lat/depth to XYZ coordinates
        slab_top_surf_xyz = ct.CoordinateTransformSphere().lld_to_xyz(slab_dep_arr)
        
        # Create a PyVista point cloud
        slab_top_surf_pc = pv.PolyData(slab_top_surf_xyz)
        
        # Generate a 2D mesh via Delaunay triangulation with alpha parameter
        slab_top_surf_mesh = slab_top_surf_pc.delaunay_2d(alpha=0.0047)
        
        # # Optionally save the mesh for each slab
        # mesh_filename = os.path.join(output_dir, f"{key}_slab_top_surf_mesh.vtk")
        # if not os.path.isfile(mesh_filename):
        #     slab_top_surf_mesh.save(mesh_filename)
        #     print(f"Saved mesh for {key} to {mesh_filename}")
        # else:
        #     print(f"Mesh for {key} already exists: {mesh_filename}")
        
        # Check for boundary irregularities
        check_irregular_boundary_points(slab_top_surf_mesh)
        print('\n')

        slab_top_surf_xyz = ct.CoordinateTransformSphere().lld_to_xyz(slab_dep_arr)
    
        # Convert the points to a PyVista point cloud
        slab_top_surf_pc = pv.PolyData(slab_top_surf_xyz)
        slab_top_surf_pc['depth'] = slab_dep_arr[:, 2]    

plotter = pv.Plotter()
plotter.add_mesh(slab_top_surf_mesh, color='white', opacity=0.5, show_edges=True)
plotter.add_points(slab_top_surf_pc, scalars='depth', cmap=plt.cm.viridis.resampled(14), point_size=5, render_points_as_spheres=True, clim=[0, 700])
plotter.show()


