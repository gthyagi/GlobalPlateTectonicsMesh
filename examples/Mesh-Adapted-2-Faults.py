# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python (Pixi)
#     language: python
#     name: pixi-kernel-python3
# ---

# %% [markdown]
# ## Create / Save mesh with adaptation to fault locations
#

# %%
# to fix trame issue
import nest_asyncio

nest_asyncio.apply()

# %%
import numpy as np
import pyvista as pv
import sympy
import underworld3 as uw


# %%
# Read the point cloud / numpy array

# point_data = np.load("Structures/faults_as_swarm_points_xyz.npz")

# %%
# fault_swarm_points_llz = point_data["arr_0"]
# fault_swarm_points_xyz = point_data["arr_1"]

# %%
import glob

# %%
# Read surface meshes (vtk)

segment_points_array = np.zeros((0, 9))
segment_cells_array = np.zeros((0, 9))

segment_surface_list = []
segment_id_list = []
for i, vtkfile in enumerate(glob.glob("Meshes/fault_seg_10dip_*.vtk")):
    segment_surface_list.append(pv.read(vtkfile))
    segment_id_list.append(int(vtkfile.split(sep="_")[-1].split(sep=".")[0]))

    segment_points = segment_surface_list[-1].points
    segment_normals = segment_surface_list[-1].point_normals
    segment_ids = np.ones(shape=(segment_points.shape[0], 1)) * i
    segment_fault = np.ones(shape=(segment_points.shape[0], 1)) * segment_id_list[-1]
    segment_index = np.indices((segment_points.shape[0], 1))[0]

    segment_cell_points = segment_surface_list[-1].cell_centers().points
    segment_cell_normals = segment_surface_list[-1].cell_normals
    segment_cell_ids = np.ones(shape=(segment_cell_points.shape[0], 1)) * i
    segment_cell_fault = (
        np.ones(shape=(segment_cell_points.shape[0], 1)) * segment_id_list[-1]
    )
    segment_cell_index = np.indices((segment_cell_points.shape[0], 1))[0]

    concat_array = np.hstack(
        (segment_points, segment_normals, segment_ids, segment_fault, segment_index)
    )
    segment_points_array = np.vstack((segment_cells_array, concat_array))

    concat_array = np.hstack(
        (
            segment_cell_points,
            segment_cell_normals,
            segment_cell_ids,
            segment_cell_fault,
            segment_cell_index,
        )
    )
    segment_cells_array = np.vstack((segment_cells_array, concat_array))

segment_points_kdtree = uw.kdtree.KDTree(
    np.ascontiguousarray(segment_points_array[:, 0:3])
)
segment_cells_kdtree = uw.kdtree.KDTree(
    np.ascontiguousarray(segment_cells_array[:, 0:3])
)

# %%
## Meshing - higher resolution than the computational mesh to allow for the swarm to be evaluated

grid_resolution = (60, 60, 30)
expt_extent = (135.8, 137.3, -34.3, -33.1)
mesh_depth_extent = (0, 40)

radius_outer = 1.0
radius_inner = 1.0 - (40 / 6370)

SWcorner = [expt_extent[0], expt_extent[2]]
NEcorner = [expt_extent[1], expt_extent[3]]

cs_mesh = uw.meshing.RegionalSphericalBox(
    SWcorner=SWcorner,
    NEcorner=NEcorner,
    radiusOuter=radius_outer,
    radiusInner=radius_inner,
    numElementsLon=grid_resolution[0],
    numElementsLat=grid_resolution[1],
    numElementsDepth=grid_resolution[2],
    simplex=True,
)

x, y, z = cs_mesh.X
r, th, ph = cs_mesh.CoordinateSystem.R

# Note the sign change between co-latitude and the SN array

unit_vertical = cs_mesh.CoordinateSystem.unit_e_0
unit_SN = -cs_mesh.CoordinateSystem.unit_e_1
unit_EW = cs_mesh.CoordinateSystem.unit_e_2

# Now we'll move the nodes to meet the surface topography

from osgeo import gdal

ep_topo = gdal.Open("./Topography/EyrePeninsula_SRTMGL3.tif")
ep_topo_img = ep_topo.ReadAsArray()
trans = ep_topo.GetGeoTransform()
ep_topo_extent = [
    trans[0],
    trans[0] + ep_topo.RasterXSize * trans[1],
    trans[3] + ep_topo.RasterYSize * trans[5],
    trans[3],
]


def ep_topo_value(lon, lat):

    pixels_lon = ep_topo_img.shape[0]
    pixels_lat = ep_topo_img.shape[1]

    pixel_lon = int(
        pixels_lon * (lon - ep_topo_extent[0]) / (ep_topo_extent[1] - ep_topo_extent[0])
    )
    pixel_lat = int(
        pixels_lat * (lat - ep_topo_extent[2]) / (ep_topo_extent[3] - ep_topo_extent[2])
    )

    return ep_topo_img.T[pixel_lon, pixels_lat - pixel_lat]


# Data on the mesh

topo = uw.discretisation.MeshVariable(
    "h", cs_mesh, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"\mathcal{H}"
)
fault_distance = uw.discretisation.MeshVariable(
    "df", cs_mesh, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"d_{F}"
)
H = uw.discretisation.MeshVariable("H", cs_mesh, 1)
Metric = uw.discretisation.MeshVariable("M", cs_mesh, 1, degree=1)

# %%
pv_cs_mesh = uw.visualisation.mesh_to_pv_mesh(cs_mesh)

# %%
# These distances are to the interpolated surface, so they are more accurate that nearest neighbour from
# a k-D tree.

with cs_mesh.access(fault_distance):
    fault_distance.data[:, 0] = 1e10

    for i, segment in enumerate(segment_surface_list):
        fault_segment_surface = segment_surface_list[i]
        dist = pv_cs_mesh.compute_implicit_distance(fault_segment_surface)

        fault_distance.data[:, 0] = np.minimum(
            fault_distance.data[:, 0], np.abs(dist.point_data["implicit_distance"])
        )

# %%
## Mesh Adaptation (uniform depth) 0.27e7 -> 50k elements, 0.55e7 -> 100k elements, 1->1e7 -> 200k elements, 2e7 -> 400k elements, 5e7 -> 1000k elements
## Mesh Adaptation (variable depth) 0.33e7 -> 50k elements, 0.65e7 -> 100k elements, 1.33->1e7 -> 200k elements, 2.66e7 -> 400k elements, 6.5e7 -> 1000k elements

with cs_mesh.access(H):
    H.data[:, 0] = uw.function.evalf(
        sympy.Piecewise(
            (6.5e7, fault_distance.sym[0] < cs_mesh.get_min_radius() * 33),
            (100, True),
        ),
        H.coords,
    )

    print(H.data.min())
    print(H.data.max())

# %%
# cs_mesh.view()

# %%
icoord, meshA = uw.adaptivity.mesh_adapt_meshVar(cs_mesh, H, Metric)

# %%
meshA.view()

# %%
# Mesh variables on new mesh

topoA = uw.discretisation.MeshVariable(
    "hA", meshA, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"\mathcal{H}"
)
fault_distanceA = uw.discretisation.MeshVariable(
    "dfA", meshA, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"d_{F_{a}}"
)
fault_normalsA = uw.discretisation.MeshVariable(
    "NA", meshA, vtype=uw.VarType.VECTOR, degree=1, varsymbol=r"{\hat{n}_{a}}"
)

rho = uw.discretisation.MeshVariable(
    "rho", meshA, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"\rho"
)
rhoN = uw.discretisation.MeshVariable(
    "rhoN", meshA, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"\rho_{sn}"
)

# %%
with meshA.access(topoA):
    R = uw.function.evalf(meshA.CoordinateSystem.R, meshA.data)

    for node in range(meshA.data.shape[0]):
        ph1 = R[node, 2]
        th1 = R[node, 1]
        topoA.data[node, 0] = ep_topo_value(
            360 * ph1 / (2 * np.pi), 90 - 360 * th1 / (2 * np.pi)
        )

delta_r = uw.function.evalf(
    topo.sym[0] * (r - radius_inner) / (radius_outer - radius_inner), meshA.data
)

new_coords = meshA.data.copy()
new_coords *= (radius_outer + delta_r.reshape(-1, 1) / 6370000) / radius_outer
meshA.deform_mesh(new_coords)

# %%
## Recompute fault distance and fault normals


## What we really need to do here is store closest point for closest fault and
## use the normal from that point everywhere in the mesh. Otherwise we have
## the free parameter for distance to the nearest fault and order becomes important.

import underworld3.visualisation as vis

pvmeshA = vis.mesh_to_pv_mesh(meshA)

with meshA.access(fault_distanceA, fault_normalsA):
    fault_distanceA.data[:, 0] = 1e10

    for i, segment in enumerate(segment_surface_list):
        fault_segment_surface = segment_surface_list[i]
        dist = pvmeshA.compute_implicit_distance(fault_segment_surface)

        fault_distanceA.data[:, 0] = np.minimum(
            fault_distanceA.data[:, 0], np.abs(dist.point_data["implicit_distance"])
        )

    # Now find the closest fault-surface node to transfer the normal

    closest_points, dist_sq, _ = segment_cells_kdtree.find_closest_point(
        fault_normalsA.coords
    )

    fault_normalsA.data[...] = segment_cells_array[closest_points, 3:6]

# %% [markdown]
# ## Map MTT to the mesh locations
#

# %%
from pyproj import CRS
from pyproj import Transformer

mt_arr = np.loadtxt("Models/UoA_Regional_EP.csv", skiprows=1, delimiter=None)

from_crs = CRS.from_proj4(
    "+proj=utm +zone=53 +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
)
to_crs = CRS.from_epsg(4326)

proj = Transformer.from_crs(from_crs, to_crs, always_xy=True)
lons, lats = proj.transform(mt_arr[:, 0], mt_arr[:, 1])
mt_extent = (float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max()))
mt_depth_extent = ((-1.0e-3 * mt_arr[:, 2]).min(), (-1.0e-3 * mt_arr[:, 2]).max())

## Map array to xyz

mt_arr_rtp = mt_arr.copy()
mt_arr_xyz = mt_arr.copy()

mt_arr_rtp[:,0] = 1+mt_arr[:,2]/6370000
mt_arr_rtp[:,1] = np.radians(lats)
mt_arr_rtp[:,2] = np.radians(lons)

mt_arr_xyz[:,0] = mt_arr_rtp[:,0] * np.cos(mt_arr_rtp[:,1]) * np.cos(mt_arr_rtp[:,2])
mt_arr_xyz[:,1] = mt_arr_rtp[:,0] * np.cos(mt_arr_rtp[:,1]) * np.sin(mt_arr_rtp[:,2])
mt_arr_xyz[:,2] = mt_arr_rtp[:,0] * np.sin(mt_arr_rtp[:,1]) 

## kdtree for RBF interpolator

var_index = uw.kdtree.KDTree(rho.coords)
tomo_index = uw.kdtree.KDTree(np.ascontiguousarray(mt_arr_xyz[:,0:3]))

with meshA.access(rho, rhoN):
    rho.data[...] = var_index.rbf_interpolator_local_to_kdtree(mt_arr_xyz[:,0:3], np.log(0.001 + mt_arr_xyz[:,3]), nnn=10)
    rhoN.data[...] = tomo_index.rbf_interpolator_local_from_kdtree(rho.coords, np.log(0.001+mt_arr_xyz[:,3]), nnn=1)



# %%
if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh0 = vis.mesh_to_pv_mesh(cs_mesh)
    pvmesh0.point_data["Fd"] = vis.scalar_fn_to_pv_points(pvmesh0, fault_distance.sym)
    pvmesh0.point_data["H"] = vis.scalar_fn_to_pv_points(pvmesh0, H.sym[0])
    pvmesh0.points = pvmesh0.points * 0.9995

    pvmeshA = vis.mesh_to_pv_mesh(meshA)
    pvmeshA.points = pvmeshA.points * 0.9995

    pvmeshA.point_data["h"] = vis.scalar_fn_to_pv_points(pvmeshA, topoA.sym[0])
    pvmeshA.point_data["Fd"] = vis.scalar_fn_to_pv_points(
        pvmeshA, fault_distanceA.sym[0]
    )
    pvmeshA.point_data["N"] = vis.vector_fn_to_pv_points(pvmeshA, fault_normalsA.sym)
    pvmeshA.point_data["RhoN"] = vis.scalar_fn_to_pv_points(pvmeshA, rhoN.sym)

    pl = pv.Plotter(window_size=(1000, 750))

    # pl.add_points(fault_points)

    pl.add_mesh(
        pvmeshA,
        style="wireframe",
        scalars="Fd",
        cmap="Greys",
        use_transparency=False,
        show_edges=False,
        opacity=0.2,
        show_scalar_bar=False,
        line_width=0.1,
        
    )

    # pl.add_mesh(
    #             pvmesh0,
    #             style="wireframe",
    #             color="Black",
    #             use_transparency=False,
    #             opacity=1,
    #             show_scalar_bar=False,
    #            )

    colors = ["Red", "Orange", "Blue", "Green"]

    for i, segment in enumerate(segment_surface_list):
        j = segment_id_list[i]
        pl.add_mesh(segment, style="wireframe", color=colors[j % 4], opacity=0.5, show_edges=True)

    def slice_mesh_EW(value):
        normal = uw.function.evalf(unit_EW, pvmeshA.center_of_mass().reshape(1,3))
        origin = pvmeshA.center_of_mass() + 0.01 * value * normal
        sliced = pvmeshA.slice(normal=normal, origin=origin)
        pl.add_mesh(sliced, name="sloiceEW", scalars="RhoN", cmap="RdBu_r", opacity=1)
    
    def slice_mesh_SN(value):
        normal = uw.function.evalf(unit_SN, pvmeshA.center_of_mass().reshape(1,3))
        origin = pvmeshA.center_of_mass() + 0.01 * value * normal
        sliced = pvmeshA.slice(normal=normal, origin=origin)
        pl.add_mesh(sliced, name="sloiceSN", scalars="RhoN", cmap="RdBu_r", opacity=1)
    
    def slice_mesh_UD(value):
        normal = uw.function.evalf(unit_vertical, pvmeshA.center_of_mass().reshape(1,3))
        origin = pvmeshA.center_of_mass() + 0.0025 * value * normal
        sliced = pvmeshA.slice(normal=normal, origin=origin)
        pl.add_mesh(sliced, name="sloiceUD", scalars="RhoN", cmap="RdBu_r", opacity=1)
    
    pl.add_slider_widget(slice_mesh_EW, [-1,1], title='EW', pointa = (0.6, 0.95), pointb=(1.0, 0.95), slider_width=0.02, tube_width=0.002, )
    pl.add_slider_widget(slice_mesh_SN, [-1,1], title='SN', pointa = (0.6, 0.88), pointb=(1.0, 0.88), slider_width=0.02, tube_width=0.002,)
    pl.add_slider_widget(slice_mesh_UD, [-1,1], title='UD', pointa = (0.6, 0.81), pointb=(1.0, 0.81), slider_width=0.02, tube_width=0.002,)

    # pl.add_arrows(pvmeshA.points, pvmeshA.point_data["N"], mag=0.0005)

    pl.camera.SetPosition((-0.640, 0.560, -0.587))

    pl.export_html("AdaptedFaultMeshWireframes.html")

    pl.show()

# %%
## Compute

0 / 0

# %%
# Save mesh, fault distance, topography information

meshA.write_timestep(
    "AdaptedMesh_10deg_1M_Elts",
    index=0,
    outputPath="Meshes",
    meshVars=[topoA, fault_distanceA, fault_normalsA, rho, rhoN],
    meshUpdates=True,
)

# %%
meshA.view()

# %%
uw.function.evalf(unit_vertical, meshA.data.mean(axis=0).reshape(1, 3))

# %%
meshA.data.mean(axis=0).reshape(1, 3)

# %%
