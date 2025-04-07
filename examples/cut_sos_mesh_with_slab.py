# ### Cut segment of sphere with slab2 surface implicit function
# *Author: Thyagarajulu Gollapalli*
#
# Notes:
# 1. If the surface doesn't span the entire domain, it may not function correctly.

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import gmsh
import pyvista as pv
import os
import numpy as np
import coordinate_transform as ct
from scipy.interpolate import RBFInterpolator
import subprocess
import meshio

# output dir
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

# ### Create generic mesh
#
# 1. The mesh is generated in Gmsh using simple geometric shapes (e.g., boxes, spheres, cylinders).
# 2. In the uw.meshing script, insert a `break` statement immediately after `gmsh.finalize()` to prevent plex creation.

# +
# mesh details
r_o = 1.0
r_i = (6371. - 800.)/6371.
lon_ext = 52.0
lat_ext = 47.0
cellsize = 1/64

meshname = f'uw_sos_ro{r_o}_ri{np.round(r_i, 2)}_lon{lon_ext}_lat{lat_ext}_csize{np.round(cellsize, 3)}'
# -

if not os.path.isfile(f'{meshname}.msh'):
    import underworld3 as uw
    
    mesh = uw.meshing.SegmentofSphere(radiusOuter=r_o, radiusInner=r_i,
                                      longitudeExtent=lon_ext, latitudeExtent=lat_ext,
                                      cellSize=cellsize, filename=f'{output_dir}{meshname}.msh')

# +
# read existing .msh file and convert to vtk
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # Set mesh file version to 2.2 for MMG compatibility
gmsh.open(f"{output_dir}{meshname}.msh")
nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes() # Get node information
nodes = np.array(nodeCoords).reshape((-1, 3)) # Convert flat Nx3 array (x, y, z)
print(f"Loaded {nodes.shape[0]} nodes from the mesh.")

gmsh.write(f"{output_dir}{meshname}_2p2.msh")
# gmsh.fltk.run()
if not os.path.isfile(f'{output_dir}{meshname}.vtk'):
    gmsh.write(f"{output_dir}{meshname}.vtk")
else:
    print('Mesh vtk file exits')
    
gmsh.finalize()
# -

# view mesh
mesh = pv.read(f"{output_dir}{meshname}.vtk")
pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, )
pl.show()


# ### Load sum slab2 surface
#
# 1. Slab2 data is in (lon, lat, -dep)
# 2. Convert this into cubedsphere xyz

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

# +
# create cubedsphere coordinate transformer
lon_min = np.min(data[:, 0])
lon_max = np.max(data[:, 0])
lat_min = np.min(data[:, 1])
lat_max = np.max(data[:, 1])
print(lon_min, lon_max, lat_min, lat_max)

coord_trans = ct.CoordinateTransformCubedsphere(g_lon_min=lon_min, 
                                                g_lon_max=lon_max,
                                                g_lat_min=lat_min, 
                                                g_lat_max=lat_max,)
# -

# get cubedsphere xyz 
sum_top_surf_c_xyz = coord_trans.geo_lld_to_cubedsphere_xyz(sum_dep_arr[::4])
print(sum_top_surf_c_xyz)

# +
# view mesh and sum surface cloud point
point_cloud = pv.PolyData(sum_top_surf_c_xyz)

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, opacity=0.1)
pl.add_points(point_cloud, color='red', point_size=3)
pl.show()
# -

# ### Create implicit function for slab surface

# For an implicit function, we assign a function value of 0 at the slab surface.
values_slab = np.zeros(sum_top_surf_c_xyz.shape[0])
values_slab[:] = 1e6

# Create the RBF interpolator. The epsilon parameter controls the scaling and may need tuning.
sum_rbf = RBFInterpolator(sum_top_surf_c_xyz, values_slab, kernel='multiquadric', epsilon=1.0)

# Evaluate the implicit function at each node of the mesh
sum_sol_values = sum_rbf(nodes)

# ### Create implicit function at 660km

# +
# create surface at 660 km
dep = 660.
num_points = 150
lon_arr = np.linspace(lon_min, lon_max, num_points)
lat_arr = np.linspace(lat_min, lat_max, num_points)
lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

# Combine longitude, latitude, and constant depth into one array
lonlatdep_arr = np.column_stack((lon_grid.ravel(), lat_grid.ravel(), np.full(lon_grid.size, dep)))
print(lonlatdep_arr)

# Convert geographic coordinates to cubed-sphere XYZ coordinates
surf_350_c_xyz = coord_trans.geo_lld_to_cubedsphere_xyz(lonlatdep_arr)
print(surf_350_c_xyz)
# -

# Compute function values.
x, y, z = surf_350_c_xyz[:,0], surf_350_c_xyz[:,1], surf_350_c_xyz[:,2]
values_surf_660 = np.sqrt(x**2 + y**2 + z**2) - ((6371.-660)/6371.)

# Create the RBF interpolator.
surf_660_rbf = RBFInterpolator(surf_350_c_xyz, values_surf_660*1e16, kernel='multiquadric', epsilon=1e0)

# Evaluate the implicit function at each node of the mesh
surf_660_sol_values = surf_660_rbf(nodes)


# ### Create .sol files

def write_medit_solution(sol_values, sol_filename="solution.sol"):
    """
    Write the scalar field solution in MEDIT .sol format to a file.

    Parameters:
      sol_values (iterable): Scalar values for each vertex.
      sol_filename (str): Output filename. Defaults to "solution.sol".
    """
    with open(sol_filename, 'w') as f:
        f.write("MeshVersionFormatted 2\n\n")
        f.write("Dimension 3\n\n")
        f.write("SolAtVertices\n")
        f.write(f"{len(sol_values)}\n")
        f.write("1 1 \n\n")
        for val in sol_values:
            f.write(f"{val}\n")
        f.write("End\n")
    print(f"Solution file written to '{sol_filename}'.")


# creating .sol files
write_medit_solution(sum_sol_values, sol_filename=f'{output_dir}sum_surf.sol')
write_medit_solution(surf_660_sol_values, sol_filename=f'{output_dir}surf_660.sol')


# ### Convert .msh to .mesh format

def gmsh2p2_to_mesh(input_msh_file, output_mesh_file):
    """
    Convert a Gmsh file to a Medit (.mesh) file.
    """
    mesh = meshio.read(input_filename)
    meshio.write(output_mesh_file, mesh, file_format="medit")
    print(f"Converted '{input_filename}' to Medit format: '{output_filename}'")


def get_vtk_from_tetgen(meshfile):
    """
    create vtk from .mesh
    """
    cmd = ["./tetgen_1.6", "-kA", meshfile]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("Standard Output:\n", result.stdout)
    print("Standard Error:\n", result.stderr)
    
    return


def run_mmg3d_remesh(inputmesh, solfile, outputmesh, isovalue=1e6, hmax=None):
    """
    Run the MMG3D remeshing command with the level-set option.
    """
    if hmax == None:
        cmd = ['mmg3d_O3', inputmesh, '-ls', str(isovalue), '-sol', solfile, '-out', outputmesh]
    else:
        cmd = ['mmg3d_O3', inputmesh, '-ls', str(isovalue), '-sol', solfile, '-out', outputmesh, '-hmax', str(hmax)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("MMG3D Command Output:")
    print(result.stdout)
    if result.stderr:
        print("MMG3D Command Errors:")
        print(result.stderr)
    
    return


def convert_mesh_to_vtk(input_mesh_file, output_vtk_file):
    """
    Convert a Medit (.mesh) file to a VTK file.
    """
    mesh = meshio.read(input_mesh_file)
    meshio.write(output_vtk_file, mesh, file_format="vtk")
    
    print(f"Converted '{input_mesh_file}' to '{output_vtk_file}'.")


# convert gmsh to .mesh format
if not os.path.isfile(f"{output_dir}{meshname}.mesh"):
    gmsh2p2_to_mesh(f"{output_dir}{meshname}_2p2.msh" , f"{output_dir}{meshname}.mesh" )
else:
    print('.mesh exists')

# +
# # slab surface remeshing
# run_mmg3d_remesh(f"{output_dir}{meshname}.mesh", f'{output_dir}sum_surf.sol', 
#                  f'{output_dir}cut_sum_slab.mesh', isovalue = 1e6)

# # create vtk of remesh file
# get_vtk_from_tetgen(f'{output_dir}cut_sum_slab.mesh')
# # !open ./output/cut_sum_slab.1.vtk

# # # this does not get the indexing of tet's properly
# # convert_mesh_to_vtk(f'{output_dir}cut_sum_slab.mesh', f'{output_dir}cut_sum_slab.vtk')
# # # !open ./output/cut_sum_slab.vtk

# +
# surface 660km remeshing
run_mmg3d_remesh(f"{output_dir}{meshname}.mesh", f'{output_dir}surf_660.sol', 
                 f'{output_dir}cut_surf_660.mesh', isovalue=0., hmax=0.01)

# create vtk of remesh file
get_vtk_from_tetgen(f'{output_dir}cut_surf_660.mesh')
# !open ./output/cut_surf_660.1.vtk

# # this does not get the indexing of tet's properly
# convert_mesh_to_vtk(f'{output_dir}cut_surf_660.mesh', f'{output_dir}cut_surf_660.vtk')
# # !open ./output/cut_surf_660.vtk
# -



