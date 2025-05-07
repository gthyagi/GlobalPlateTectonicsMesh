# ### Generate Surface Mesh of Sumatra Subduction Region from Slab2
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
import xarray as xr
import pyvista as pv
import subprocess
import meshio
from collections import defaultdict
from scipy.spatial import cKDTree
import tetgen
import gmsh
from enum import Enum
import sympy
import cmcrameri.cm as cmc

import shutil
from petsc4py import PETSc

# +
# output dir
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

# surface resolution: 'high' or 'low'
resolution = 'high'
# mmg parameters
if resolution=='low':
    hmax, hmin = 0.02, 0.019
else:
    hmax, hmin = 0.0019, 0.0018 #0.0015, 0.0014

# when this is set to True then existing files will remove and new files will be created
remove_old_file = True


# -

def remove_nan_rows(arr):
    '''
    Remove rows containing NaN values from the given NumPy array.
    '''
    return arr[~np.isnan(arr).any(axis=1)]


# +
# loading slab2 sum data
slab2_dir = '/Users/tgol0006/PlateTectonicsTools/Slab2_AComprehe/Slab2Distribute_Mar2018/Slab2_TXT/'
sum_dep_xyz = np.loadtxt(f'{slab2_dir}sum_slab2_dep_02.23.18.xyz', delimiter=',')
print(sum_dep_xyz)

# Remove rows containing NaN values and convert depth to positive values
sum_dep_arr = remove_nan_rows(sum_dep_xyz)
sum_dep_arr[:, 2] *= -1
print(sum_dep_arr)
print(sum_dep_arr[:,2].min(), sum_dep_arr[:,2].max())


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
fig_slab, ax_slab = create_scatter_plot(sum_dep_arr, plt.cm.viridis_r.resampled(14), 0, 700)


# ### Transforming the geo lld to cubed sphere xyz

class CoordinateTransformCubedsphere(object):
	"""
	Transform coordinates in geographical longitude, latitude, depth to cubedsphere xyz and vice versa.

	Notes:
	g-geographical, t-transformed, c-cubedsphere
	llr - longitude, latitude, radius (km)
	lld - longitude, latitude, depth (km)
	xyz - x, y, z
	"""
	def __init__(self, g_lon_min: float = -45., g_lon_max: float = 45., g_lat_min: float = -45., g_lat_max: float = 45., radius: float = 6371.0):
		
		if abs(g_lon_max - g_lon_min) > 180 or abs(g_lat_max - g_lat_min) > 180:
			raise ValueError("Longitude and Latitude extent should be less than 180 degrees")
		
		self.radius = radius
		self.g_lon_min, self.g_lon_max = g_lon_min, g_lon_max
		self.g_lat_min, self.g_lat_max = g_lat_min, g_lat_max
		self.t_lon_min, self.t_lon_max = np.mod(g_lon_min, 360), np.mod(g_lon_max, 360)
		self.t_lat_min, self.t_lat_max = 90 - g_lat_max, 90 - g_lat_min
		self.mid_t_lon = (self.t_lon_max + self.t_lon_min) / 2
		self.mid_t_lat = (self.t_lat_max + self.t_lat_min) / 2

		print(f"Transformed longitude ranges from 0 to 360 degrees, and transformed latitude ranges from 0 to 180 degrees. \n"
			  f"The North Pole is at 0 degrees, and the South Pole is at 180 degrees. \n"
			  f"Model Extent. In longitude: {abs(g_lon_max - g_lon_min)}, latitude: {abs(g_lat_max - g_lat_min)}. \n"
			  f"Geographical longitudes: [{self.g_lon_min}, {self.g_lon_max}] -> Transformed longitudes: [{self.t_lon_min}, {self.t_lon_max}]. \n"
			  f"Geographical latitudes: [{self.g_lat_min}, {self.g_lat_max}] -> Transformed latitudes: [{self.t_lat_max}, {self.t_lat_min}]. \n"
			  f"Midpoint of transformed longitude and latitude: {self.mid_t_lon}, {self.mid_t_lat}")
	
	
	def geo_lld_to_cubedsphere_xyz(self, geo_lonlatdep: np.ndarray) -> np.ndarray:
		"""
		Converts geographical coordinates (longitude, latitude, depth) to cubedsphere coordinates (x, y, z).

		Steps:
		1. Converts geographical longitude range to [0, 360] and latitude range to [0, 180].
		2. Converts transformed (longitude, latitude, depth) into cubedsphere domain (longitude, latitude, radius).
		3. Converts cubedsphere (longitude, latitude, radius) to cubedsphere (x, y, z).
		"""

		if geo_lonlatdep.shape[1] != 3:
			raise ValueError("Input data must be in the format (longitude, latitude, depth) in geographical coordinates.")

		# Step 1
		g_lld = self._convert_geo_to_transformed_lld(geo_lonlatdep)

		# Step 2
		c_llr = self._convert_transformed_lld_to_cubedsphere_llr(g_lld)

		# Step 3
		c_xyz = self._convert_cubedsphere_llr_to_xyz(c_llr)
		return c_xyz

	def _convert_geo_to_transformed_lld(self, geo_lonlatdep: np.ndarray) -> np.ndarray:
		"""
		Converts geographical coordinates (longitude, latitude, depth) to transformed coordinates.
		"""
		t_lld = geo_lonlatdep.copy()
		t_lld[:, 0] = np.mod(t_lld[:, 0], 360) # Normalize longitude to [0, 360]
		t_lld[:, 1] = 90 - t_lld[:, 1] # Transform latitude from [-90, 90] to [0, 180]
		return t_lld

	def _convert_transformed_lld_to_cubedsphere_llr(self, t_lld: np.ndarray) -> np.ndarray:
		"""
		Converts transformed coordinates (longitude, latitude, depth) into cubedsphere domain (longitude, latitude, radius).
		"""
		c_lon = t_lld[:, 0] - self.mid_t_lon
		c_lat = self.mid_t_lat - t_lld[:, 1]
		c_radius = (self.radius - t_lld[:, 2]) / self.radius
		return np.column_stack((c_lon, c_lat, c_radius))

	def _convert_cubedsphere_llr_to_xyz(self, c_llr: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere coordinates from (longitude, latitude, radius) to (x, y, z).

		Calculations:
		  - Compute the tangent of the longitude and latitude (in radians).
		  - Compute d = radius / sqrt(tan(lon)^2 + tan(lat)^2 + 1).
		  - Compute:
			  x = d * tan(lon)
			  y = d * tan(lat)
			  z = d
		"""
		# Compute tangent values for longitude and latitude (converted from degrees to radians)
		tan_lon = np.tan(np.deg2rad(c_llr[:, 0]))
		tan_lat = np.tan(np.deg2rad(c_llr[:, 1]))
		denom = np.sqrt(tan_lon**2 + tan_lat**2 + 1)
		d = c_llr[:, 2] / denom
		return np.column_stack((d * tan_lon, d * tan_lat, d))
	

	def cubedsphere_xyz_to_geo_lld(self, c_xyz: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere (x, y, z) coordinates to geographical (longitude, latitude, depth).
		Steps:
		  1. Convert cubedsphere (x, y, z) to cubedsphere (lon, lat, radius).
		  2. Convert cubedsphere (lon, lat, radius) to transformed (lon, lat, depth).
		  3. Convert transformed (lon, lat, depth) to geographical (lon, lat, depth).
		"""
		# Step 1
		c_llr = self._convert_cubedsphere_xyz_to_llr(c_xyz)
		# Step 2
		t_lld = self._convert_cubedsphere_llr_to_transformed_lld(c_llr)
		# Step 3
		g_lld = self._convert_transformed_lld_to_geo_lld(t_lld)
		return g_lld

	def _convert_cubedsphere_xyz_to_llr(self, c_xyz: np.ndarray) -> np.ndarray:
		"""
		Converts cubedsphere (x, y, z) coordinates to cubedsphere (longitude, latitude, radius).
		"""
		if np.any(np.isclose(c_xyz[:, 2], 0)):
			raise ValueError("z coordinate is zero for one or more points; cannot perform conversion.")

		tan_lon = c_xyz[:, 0] / c_xyz[:, 2]
		tan_lat = c_xyz[:, 1] / c_xyz[:, 2]
		factor = np.sqrt(tan_lon**2 + tan_lat**2 + 1)

		lon = np.degrees(np.arctan(tan_lon))
		lat = np.degrees(np.arctan(tan_lat))
		radius = c_xyz[:, 2] * factor
		return np.column_stack((lon, lat, radius))

	def _convert_cubedsphere_llr_to_transformed_lld(self, c_llr: np.ndarray) -> np.ndarray:
		"""
		Convert cubedsphere coordinates (longitude, latitude, radius) to
		transformed coordinates (longitude, latitude, depth).
		"""
		t_lld = np.empty_like(c_llr)
		t_lld[:, 0] = c_llr[:, 0] + self.mid_t_lon
		t_lld[:, 1] = self.mid_t_lat - c_llr[:, 1]
		t_lld[:, 2] = (1.0 - c_llr[:, 2]) * self.radius
		return t_lld

	def _convert_transformed_lld_to_geo_lld(self, t_lld: np.ndarray) -> np.ndarray:
		"""
		Converts transformed (lon, lat, depth) to geographical (lon, lat, depth) coordinates.
		The transformation converts the latitude by subtracting from 90 degrees.
		"""
		g_lld = t_lld.copy()
		g_lld[:, 1] = 90 - t_lld[:, 1]
		return g_lld

# +
# Get the minimum and maximum values in lon and lat
lon_min = np.min(sum_dep_xyz[:, 0])
lon_max = np.max(sum_dep_xyz[:, 0])
lat_min = np.min(sum_dep_xyz[:, 1])
lat_max = np.max(sum_dep_xyz[:, 1])

print(lon_min, lon_max, lat_min, lat_max)

coord_trans = CoordinateTransformCubedsphere(g_lon_min=lon_min, 
                                                g_lon_max=lon_max,
                                                g_lat_min=lat_min, 
                                                g_lat_max=lat_max,)
# -

# slab surface top
sum_top_surf_c_xyz = coord_trans.geo_lld_to_cubedsphere_xyz(sum_dep_arr)
print(sum_top_surf_c_xyz)

# +
# Convert the points to a PyVista point cloud
sum_top_surf_cloud = pv.PolyData(sum_top_surf_c_xyz)

# Create a PyVista plotter
pl = pv.Plotter()
pl.add_points(sum_top_surf_cloud, color='blue', point_size=2)
pl.background_color = 'white'
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


def find_mmg(exe):
    '''
    Get mmg tools petsc path
    '''
    
    # try system PATH
    path = shutil.which(exe)
    if path:
        return path

    # fallback to PETSc build dir via env
    petsc_dir  = os.environ.get("PETSC_DIR", "")
    petsc_arch = os.environ.get("PETSC_ARCH", "")
    candidate = os.path.join(petsc_dir, petsc_arch, "bin", exe)
    return candidate if os.path.isfile(candidate) else None


# +
# get mmg tools from petsc
mmg2d_petsc = find_mmg('mmg2d_debug')
mmgs_petsc = find_mmg('mmgs_debug')
mmg3d_petsc = find_mmg('mmg3d_debug')

print(mmg3d_petsc, '\n', mmg2d_petsc, '\n', mmgs_petsc)


# -

def run_mmgs_remesh(mmgs_petsc, inputmesh, outputmesh, hmax=None, hmin=None, hausd=None, nosurf=True):
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
    cmd = [mmgs_petsc, inputmesh, '-o', outputmesh]

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

if remove_old_file:
    os.system('open ./output/sum_top_surf_mesh.vtk')

# Ensure the surface mesh has no boundary holes before passing it to MMG.
check_irregular_boundary_points(sum_top_surf_mesh)

if remove_old_file:
    os.remove('./output/sum_top_surf_mesh_mmg.vtk')

# run mmg
if not os.path.isfile(f'{output_dir}sum_top_surf_mesh_mmg.vtk'):
    save_pyvista_to_mesh(sum_top_surf_mesh, f'{output_dir}sum_top_surf_mesh.mesh')
    run_mmgs_remesh(mmgs_petsc, f'{output_dir}sum_top_surf_mesh.mesh', f'{output_dir}sum_top_surf_mesh_mmg.mesh', 
                    hmax=hmax, hmin=hmin, hausd=None)
    convert_mesh_to_vtk(f'{output_dir}sum_top_surf_mesh_mmg.mesh', f'{output_dir}sum_top_surf_mesh_mmg.vtk')
else:
    print('Slab top surface mmg mesh file exists!')

# view vtk in paraview
os.system('open ./output/sum_top_surf_mesh_mmg.vtk')

# +
# Load mmg mesh
sum_top_surf_mesh_mmg = pv.read(f'{output_dir}sum_top_surf_mesh_mmg.vtk')

# Compute normals for the mesh points
sum_top_surf_normals = sum_top_surf_mesh_mmg.extract_surface().compute_normals(point_normals=True, 
                                                                               cell_normals=False)

# Visualize mesh with normals
sum_top_surf_normals.plot_normals(mag=0.01, color='red')

# +
# create volume from slab top surface

# Step 1: Load the open surface mesh
surf = sum_top_surf_mesh_mmg.extract_surface()

# Step 2: Compute normals
surf_with_normals = surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

# Step 3: Offset points along normals
offset_distance = 0.018  # adjust as needed
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
all_surfaces = surf_with_normals + offset_surf + side_mesh
closed_shell = all_surfaces.triangulate()

# Visualize
closed_shell.plot(show_edges=True, color='white', opacity=0.6)

if closed_shell.is_manifold:
    print('Created volume is good for further operations')
else:
    print('Created volume is not closed')
    
if remove_old_file:
    os.remove(f"{output_dir}sum_vol_from_top_surf.vtk")

if not os.path.isfile(f"{output_dir}sum_vol_from_top_surf.vtk"):
    closed_shell.save(f"{output_dir}sum_vol_from_top_surf.vtk")
    save_pyvista_to_mesh(closed_shell, f"{output_dir}sum_vol_from_top_surf.mesh")
else:
    print('volume already exists')

os.system(f'open ./output/sum_vol_from_top_surf.vtk')
# +
# create volume mesh with TetGen
tgen = tetgen.TetGen(closed_shell)
tgen.make_manifold()
slab_volume = tgen.tetrahedralize(maxvolume=0.0001, quality=1.4)

# save as vtk
slab_volume_grid = tgen.grid
slab_volume_grid.save(f'{output_dir}sum_vol_from_top_surf_tet.vtk')
os.system(f'open {output_dir}sum_vol_from_top_surf_tet.vtk')
# +
# 1) Read your VTK file
mesh = meshio.read(f'{output_dir}sum_vol_from_top_surf_tet.vtk')

# 2) Write out to Medit “.mesh” format
# ensure points are little‑endian float64
if mesh.points.dtype.byteorder == '>' or (mesh.points.dtype.byteorder == '=' and np.little_endian is False):
    mesh.points = mesh.points.astype('<f8')
meshio.write(f'{output_dir}sum_vol_from_top_surf_tet.mesh', mesh, file_format="medit")

# # — or — write to Gmsh “.msh” format
# meshio.write(f'{output_dir}sum_vol_from_top_surf_tet.msh', mesh, file_format="gmsh22")
# -

# cleaning tet mesh with mmg
input_mesh = f'{output_dir}sum_vol_from_top_surf_tet.mesh'
output_mesh = f'{output_dir}sum_vol_from_top_surf_tet_mmg.mesh'
os.system(f'{mmg3d_petsc} -optim {input_mesh} -hmin 0.001 -hmax 0.005 -out {output_mesh}')


# convert gmsh to medit format
gmsh.initialize()
gmsh.open(f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_refined.msh")
gmsh.write(f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_refined.mesh")
gmsh.finalize()

# +
# loading mesh datasets

# reading back refined slab volume from mmg
slab = pv.read(f'{output_dir}sum_vol_from_top_surf_tet_mmg.mesh')

# loading refined spherical mesh from gmsh
sph_gmsh = pv.read(f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_refined.msh")
sph_mesh = pv.read(f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_refined.mesh")
# -

print(sph_gmsh['gmsh:dim_tags'])
print(np.unique(sph_gmsh['gmsh:dim_tags'][:,0]))
print(np.unique(sph_gmsh['gmsh:dim_tags'][:,1]))
print(np.unique(sph_gmsh['gmsh:physical']))
print(np.unique(sph_gmsh['gmsh:geometrical']))

# +
geom_ids = [12]  # the IDs you want

mask = np.isin(sph_gmsh.cell_data['gmsh:physical'], geom_ids)
subset = sph_gmsh.extract_cells(mask)

subset.plot(show_edges=True, cmap="tab20", scalars='gmsh:physical', cpos='xy')
# -

# create kd tree for slab points
slab_kdtree = cKDTree(slab.points)
dists, _ = slab_kdtree.query(sph_mesh.points)

# Extract top surface from the slab volume
slab_surface = slab.extract_surface().clean()  # makes it a watertight surface mesh

# Wrap the Medit points as a PyVista cloud (not needed for compute_implicit_distance)
cloud = pv.PolyData(sph_mesh.points)

# Compute signed distance at cap nodes to slab surface
sph_with_dist = cloud.compute_implicit_distance(slab_surface)

# +
# Rename the field for clarity
sph_with_dist["level_set"] = sph_with_dist["implicit_distance"]

# Save result
sph_with_dist.save(f"{output_dir}sph_cap_level_set.vtk")


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


# create level set file
write_gmsh_sol(filename=f"{output_dir}sph_cap_level_set.sol", mesh=sph_with_dist, 
               field_name="level_set")

# +
input_mesh = f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_refined.mesh"
sol_file = f'{output_dir}sph_cap_level_set.sol'
output_mesh = f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_refined_mmg.mesh"

os.system(f'{mmg3d_petsc} {input_mesh} -sol {sol_file} -ls -nr -hausd 0.001 -hgrad 1.7 -hmin 0.01 -hmax 0.05 -out {output_mesh}')

# +
# os.system(f'mmg3d_O3 {output_mesh} -noinsert -noswap -nomove -nsd 3 ') #-out {output_mesh}')

# convert mmg output mesh to msh
output_msh = f"{output_dir}uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_refined_mmg.msh"
os.system(f'gmsh {output_mesh} -o {output_msh} -nopopup -save')

output_vtk = output_msh.replace(".msh", ".vtk")
os.system(f'gmsh {output_msh} -o {output_vtk} -format vtk -nopopup -save')
# -

# visualize the mesh with slab volume
merged_msh = pv.read(output_msh)
print(np.unique(merged_msh['gmsh:geometrical']))
merged_msh

# +
geom_ids = [3]  # the IDs you want

mask = np.isin(merged_msh.cell_data["gmsh:geometrical"], geom_ids)
subset = merged_msh.extract_cells(mask)

subset.plot(show_edges=True, cmap="tab20", scalars="gmsh:geometrical", cpos='xy')
# +
class surface_boundaries(Enum):
    slab_surface = 10
    Lower        = 11
    Upper        = 12
    East         = 13
    West         = 14
    South        = 15
    North        = 16

class element_tag(Enum):
    inside_slab  = 3
    outside_slab = 2


# +
# 1) Initialize and read the mesh
gmsh.initialize()
gmsh.open(output_msh)   # replace with your filename

# 2) Print all Physical Groups (dim, tag, name)
print("=== Physical Groups ===")
for dim, tag in gmsh.model.getPhysicalGroups():
    name = gmsh.model.getPhysicalName(dim, tag)
    print(f"  dim={dim:1d}, tag={tag:3d}, name='{name}'")

# 3) Print all CAD entities and their tags
print("\n=== CAD Entities ===")
for dim in range(4):  # 0=points, 1=curves, 2=surfaces, 3=volumes
    ents = gmsh.model.getEntities(dim)
    if ents:
        print(f"  Dimension {dim}:")
        for _, tag in ents:
            print(f"    entity tag = {tag}")

# 1) Make sure each physical group exists (add if missing) and name it
for surf in surface_boundaries:
    # addPhysicalGroup will return an existing group if one with same tag already exists
    pg = gmsh.model.addPhysicalGroup(2, [surf.value])
    gmsh.model.setPhysicalName(2, pg, surf.name)

for vol in element_tag:
    pg = gmsh.model.addPhysicalGroup(3, [vol.value])
    gmsh.model.setPhysicalName(3, pg, vol.name)

gmsh.write(f'./output/uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_refined_mmg_relabel.msh')

# 5) Finalize
gmsh.finalize()

# -

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
import underworld3.visualisation as vis

mesh = uw.discretisation.Mesh(f'./output/uw_sos_ro1.0_ri0.87_lon52.0_lat47.0_csize0.016_refined_mmg_relabel.msh', 
                              boundaries=surface_boundaries, 
                              # boundary_normals=boundary_normals_2D,
                              markVertices=True, useMultipleTags=True, useRegions=True,
                              coordinate_system_type=uw.coordinates.CoordinateSystemType.SPHERICAL, )


mesh.dm.view()

cell_tags = uw.discretisation.MeshVariable(r"cell_tags", mesh, 1, degree=0, continuous=False)
v_soln = uw.discretisation.MeshVariable(r"u", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
p_soln = uw.discretisation.MeshVariable(r"p", mesh, 1, degree=1, continuous=True)
rho = uw.discretisation.MeshVariable('RHO', mesh, 1, degree=0, continuous=False)

label_info = [
    ("inside_slab", 8),
    ("outside_slab", 9),
]

# +
# 0) Grab the plex and find the cell‑point range
dm = mesh.dm
cellStart, cellEnd = dm.getHeightStratum(0)

# 6) Loop over each label, fetch its cells, and assign
for label_name, label_val in label_info:
    label = dm.getLabel(label_name)
    if label is None:
        raise KeyError(f"Label '{label_name}' not found in mesh")
    iset: PETSc.IS = label.getStratumIS(label_val)
    if iset is None:
        print(f"  [Warning] No cells found with {label_name} = {label_val}")
        continue

    # global point indices of the cells
    pts = iset.getIndices()

    # convert to local cell‑indices (0 … nLocalCells‑1)
    local_cells = [p - cellStart for p in pts if cellStart <= p < cellEnd]

    # stamp those entries
    with mesh.access(cell_tags, rho):
        cell_tags.data[local_cells] = label_val
        if label_val==8:
            rho.data[local_cells] = 1.0
        if label_val==9:
            rho.data[local_cells] = 0.0

# +
# plotting 
pvmesh = vis.mesh_to_pv_mesh(mesh)

with mesh.access(cell_tags, rho):
    pvmesh.cell_data["cell_tags"] = cell_tags.data
    pvmesh.cell_data["rho"] = rho.data
    
subset = pvmesh.threshold(
    value=(0.5, 1.0),
    scalars="rho"
)

pl = pv.Plotter(window_size=(750, 750))
pl.add_mesh(
    subset,
    edge_color="k",
    show_edges=True,
    scalars="rho",
    cmap=plt.cm.tab10.resampled(3),
    clim=[1, 3],
    show_scalar_bar=True
)
pl.show(cpos="xy")
pl.camera.zoom(1.4)

# -

# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR[0], mesh.CoordinateSystem.xR[1]
phi_uw =sympy.Piecewise((2*sympy.pi + mesh.CoordinateSystem.xR[2], mesh.CoordinateSystem.xR[2]<0), 
                        (mesh.CoordinateSystem.xR[2], True)
                       )

# Create Stokes object
stokes = uw.systems.Stokes(mesh, velocityField=v_soln, pressureField=p_soln,) # solver_name="stokes")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Upper.name)
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Lower.name)
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.East.name)
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.West.name)
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.North.name)
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.South.name)

gravity_fn = -1.0 * unit_rvec
stokes.bodyforce = rho.sym*gravity_fn

unit_rvec

stokes.bodyforce.sym

# +
# Stokes settings
stokes.tolerance = 1e-6
stokes.petsc_options["ksp_monitor"] = None

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# mg, multiplicative - very robust ... similar to gamg, additive
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")
# -

stokes.solve(verbose=True, debug=False)

with mesh.access(v_soln, p_soln):
    print(v_soln.data[:,0].min(), v_soln.data[:,0].max())
    print(v_soln.data[:,1].min(), v_soln.data[:,1].max())
    print(v_soln.data[:,2].min(), v_soln.data[:,2].max())
    print(p_soln.data.min(), p_soln.data.max())

# +
clim, vmag, vfreq = [0., 0.001], 5e2, 75
    
if uw.mpi.size == 1:
    vis.plot_vector(mesh, v_soln, vector_name='v_sol', cmap=cmc.lapaz.resampled(21), clim=clim, vmag=vmag, vfreq=vfreq,
                    save_png=True, dir_fname=output_dir+'vel_uw.png', clip_angle=135, cpos='yz', show_arrows=False)

    vis.save_colorbar(colormap=cmc.lapaz.resampled(21), cb_bounds=None, vmin=clim[0], vmax=clim[1], figsize_cb=(5, 5), primary_fs=18, 
                      cb_orient='horizontal', cb_axis_label='Velocity', cb_label_xpos=0.5, cb_label_ypos=-2.05, fformat='pdf', 
                      output_path=output_dir, fname='v_sol')
# -


