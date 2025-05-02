# ### Create / Save mesh adapted to Slab2 geometry
#
# 1. Begin by creating a coarse mesh in Underworld.  
# 2. Define a field on this mesh to guide refinement.  
# 3. Use Gmsh to adapt the mesh to the slab surface based on this field.  
# 4. Advantage: all labels and tags are preserved, allowing the mesh to be used directly in Underworld for simulations or passed to MMG for further refinement.

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import numpy as np
import underworld3 as uw
import sympy
import os
import gmsh
from typing import Optional, Tuple
from enum import Enum
import math

if uw.mpi.size == 1:
    import pyvista as pv
    from matplotlib import pyplot as plt
    import underworld3.visualisation as vis
# -

# output dir
if uw.mpi.rank == 0:
    output_dir = './output/'
    os.makedirs(output_dir, exist_ok=True)

# +
# mesh details
r_o = 1.0
r_i = (6371. - 800.)/6371.
lon_ext = 52.0
lat_ext = 47.0
cellsize = 1/64

meshname = f'uw_sos_ro{r_o}_ri{np.round(r_i, 2)}_lon{lon_ext}_lat{lat_ext}_csize{np.round(cellsize, 3)}'

# +
# mesh
mesh = uw.meshing.SegmentofSphere(radiusOuter=r_o, radiusInner=r_i,
                                  longitudeExtent=lon_ext, latitudeExtent=lat_ext,
                                  cellSize=cellsize, filename=f'{output_dir}{meshname}.msh')

# mesh variable
fault_distance = uw.discretisation.MeshVariable("df", mesh, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"d_{F}")
H = uw.discretisation.MeshVariable("H", mesh, 1)
# -

mesh.view()

# slab vtk
# slab_vtk = pv.read(f'{output_dir}sum_top_surf_mesh_mmg.vtk') # this slab surface
slab_vtk = pv.read(f'{output_dir}sum_vol_from_top_surf_tet.vtk') # this slab volume
surface_list = [slab_vtk.extract_surface()]

# Initialise pyvista engine from existing mesh
pvmesh = vis.mesh_to_pv_mesh(mesh)

if uw.mpi.size == 1:
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh,
        style="wireframe",
        cmap="Greys",
        use_transparency=False,
        show_edges=False,
        opacity=0.2,
        show_scalar_bar=False,
        line_width=0.1,
        )

    pl.add_mesh(slab_vtk)
    # pl.export_html(f"{output_dir}NonAdaptedFaultMeshWireframes.html")
    pl.show()

# Compute the implicit distance from the mesh nodes to a surface
with mesh.access(fault_distance):
    fault_distance.data[:, 0] = 1e10

    for i, segment in enumerate(surface_list):
        fault_segment_surface = surface_list[i]
        dist = pvmesh.compute_implicit_distance(fault_segment_surface)
        fault_distance.data[:, 0] = dist.point_data["implicit_distance"]
        
    print(fault_distance.data.min(), fault_distance.data.max())

if uw.mpi.size == 1:
    pvmesh.point_data["Fd"] = vis.scalar_fn_to_pv_points(pvmesh, fault_distance.sym)

    print(pvmesh.point_data["Fd"].min(), pvmesh.point_data["Fd"].max())
    
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh,
        scalars="Fd",
        cmap=plt.cm.Spectral.resampled(4),
        use_transparency=False,
        show_edges=False,
        opacity=1,
        show_scalar_bar=True,
        line_width=0.1,
        clim=(-0.05, 0.05),
        )
    
    pl.show()

# +
# Mesh Adaptation metric
metric_on_fault =  cellsize/6 # smaller the value means smaller the cell size
metric_on_nonfault = cellsize
zone_width = 2.5 # this value controls the width of the refinement around the slab

with mesh.access(H):
    H.data[:, 0] = uw.function.evalf(
        sympy.Piecewise(
            (metric_on_fault, fault_distance.sym[0] < mesh.get_min_radius() * zone_width),
            (metric_on_nonfault, True),
        ),
        H.coords,
    )

    print(H.data.min(), H.data.max())
# -

if uw.mpi.size == 1:
    pvmesh.point_data["H"] = vis.scalar_fn_to_pv_points(pvmesh, H.sym[0])

    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh,
        scalars="H",
        cmap=plt.cm.tab10.resampled(2),
        use_transparency=False,
        show_edges=True,
        opacity=1,
        show_scalar_bar=True,
        line_width=0.1,
        clim= (metric_on_fault, metric_on_nonfault),
        )
    
    pl.show()

if uw.mpi.size == 1:
    pvmesh.point_data["H"] = vis.scalar_fn_to_pv_points(pvmesh, H.sym[0])

    # Apply a clip (e.g., along x = 0 plane, invert=False means keep x > 0)
    clipped_mesh = pvmesh.clip(normal="x", origin=(0, 0, 0), invert=False, crinkle=True)

    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        clipped_mesh,
        scalars="H",
        cmap=plt.cm.tab10.resampled(2),
        use_transparency=False,
        show_edges=True,
        opacity=1,
        show_scalar_bar=True,
        line_width=0.1,
        clim=(metric_on_fault, metric_on_nonfault),
    )

    pl.show()


# +
# # both methods for checking is not working
# mesh.petsc_save_checkpoint(index=0, meshVars=[H, fault_distance], outputPath=output_dir)
# mesh.write_timestep('test', meshUpdates=True, meshVars=[H, fault_distance], outputPath=output_dir, index=0, )

# +
# creating .pos file, this as field data for the background mesh
with mesh.access(H):
    coords = H.coords
    values = H.data  

# Create .pos content
lines = ['View "H" {']
for i in range(coords.shape[0]):
    x, y, z = coords[i]
    v = values[i, 0]
    lines.append(f"  SP({x:.8e}, {y:.8e}, {z:.8e}){{{v:.8e}}};")
lines.append("};")

# Write to file
with open(f'{output_dir}H_field.pos', "w") as f:
    f.write("\n".join(lines))

print("Written: H_field.pos")

# +
# checking labels and tags in the .msh file

gmsh.initialize()
gmsh.open(f'{output_dir}{meshname}.msh')

# get all node coordinates
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
print(f"Number of nodes: {len(node_tags)}")

# get elements
element_types, element_tags, node_tags_by_element = gmsh.model.mesh.getElements()
print(f"Element types: {element_types}")

for etype, tags, conn in zip(element_types, element_tags, node_tags_by_element):
    if etype == 2:
        print(f"Triangles: {len(tags)}")
        tri_connectivity = conn.reshape((-1, 3))
    elif etype == 4:
        print(f"Tetrahedra: {len(tags)}")
        tet_connectivity = conn.reshape((-1, 4))

# Get all physical groups
phys_groups = gmsh.model.getPhysicalGroups()

for dim, tag in gmsh.model.getPhysicalGroups():
    name = gmsh.model.getPhysicalName(dim, tag)
    print(f"Physical group: dim={dim}, tag={tag}, name='{name}'")

    # Check if there are actual mesh entities associated
    entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
    if not entity_tags:
        print(f"No entities found for physical group tag={tag}")
        continue

gmsh.finalize()


# -

def SegmentofSphere(
    radiusOuter: float = 1.0,
    radiusInner: float = 0.547,
    longitudeExtent: float = 90.0,
    latitudeExtent: float = 90.0,
    cellSize: float = 0.1,
    degree: int = 1,
    qdegree: int = 2,
    filename=None,
    refinement=None,
    gmsh_verbosity=0,
    verbose=False,
    centroid: Tuple = (0.0, 0.0, 0.0),
):
    """
    Generate a 3D mesh of a spherical segment using Gmsh, defined by inner/outer radii and angular extents in longitude and latitude. 
    Parameters control mesh size, element degree, file output, refinement, and verbosity. The mesh is centered at a specified centroid 
    and saved to a file if a name is provided.
    """

    class boundaries(Enum):
        Lower = 11
        Upper = 12
        East = 13
        West = 14
        South = 15
        North = 16

    if filename is None:
        if uw.mpi.rank == 0:
            os.makedirs(".meshes", exist_ok=True)

        uw_filename = f".meshes/uw_segmentofsphere_ro{radiusOuter}_ri{radiusInner}_longext{longitudeExtent}_latext{latitudeExtent}_csize{cellSize}.msh"
    else:
        uw_filename = filename

    if (
        radiusInner <= 0
        or not (0 < longitudeExtent < 180)
        or not (0 < latitudeExtent < 180)
    ):
        raise ValueError(
            "Invalid input parameters: "
            "radiusInner must be greater than 0, "
            "and longitudeExtent and latitudeExtent must be within the range (0, 180)."
        )


    def getSphericalXYZ(point):
        """
        Perform Cubed-sphere projection on coordinates.
        Converts (radius, lon, lat) in spherical region to (x, y, z) in spherical region.

        Parameters
        ----------
        Input:
            Coordinates in rthetaphi format (radius, lon, lat)
        Output
            Coordinates in XYZ format (x, y, z)
        """

        (x, y) = (
            math.tan(point[1] * math.pi / 180.0),
            math.tan(point[2] * math.pi / 180.0),
        )
        d = point[0] / math.sqrt(x**2 + y**2 + 1)
        coordX, coordY, coordZ = (
            centroid[0] + d * x,
            centroid[1] + d * y,
            centroid[2] + d,
        )

        return (coordX, coordY, coordZ)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", gmsh_verbosity)

    gmsh.merge(f'{output_dir}H_field.pos')
    
    gmsh.model.add("SegmentOfSphere")

    p0 = gmsh.model.geo.addPoint(
        centroid[0], centroid[1], centroid[2], meshSize=cellSize
    )

    # Create segment of sphere
    dim = 3

    long_half = longitudeExtent / 2
    lat_half = latitudeExtent / 2

    pt1 = getSphericalXYZ((radiusInner, -long_half, -lat_half))
    pt2 = getSphericalXYZ((radiusInner, long_half, -lat_half))
    pt3 = getSphericalXYZ((radiusInner, long_half, lat_half))
    pt4 = getSphericalXYZ((radiusInner, -long_half, lat_half))
    pt5 = getSphericalXYZ((radiusOuter, -long_half, -lat_half))
    pt6 = getSphericalXYZ((radiusOuter, long_half, -lat_half))
    pt7 = getSphericalXYZ((radiusOuter, long_half, lat_half))
    pt8 = getSphericalXYZ((radiusOuter, -long_half, lat_half))

    p1 = gmsh.model.geo.addPoint(pt1[0], pt1[1], pt1[2], meshSize=cellSize)
    p2 = gmsh.model.geo.addPoint(pt2[0], pt2[1], pt2[2], meshSize=cellSize)
    p3 = gmsh.model.geo.addPoint(pt3[0], pt3[1], pt3[2], meshSize=cellSize)
    p4 = gmsh.model.geo.addPoint(pt4[0], pt4[1], pt4[2], meshSize=cellSize)
    p5 = gmsh.model.geo.addPoint(pt5[0], pt5[1], pt5[2], meshSize=cellSize)
    p6 = gmsh.model.geo.addPoint(pt6[0], pt6[1], pt6[2], meshSize=cellSize)
    p7 = gmsh.model.geo.addPoint(pt7[0], pt7[1], pt7[2], meshSize=cellSize)
    p8 = gmsh.model.geo.addPoint(pt8[0], pt8[1], pt8[2], meshSize=cellSize)

    l1 = gmsh.model.geo.addCircleArc(p1, p0, p2)
    l2 = gmsh.model.geo.addCircleArc(p2, p0, p3)
    l3 = gmsh.model.geo.addCircleArc(p3, p0, p4)
    l4 = gmsh.model.geo.addCircleArc(p4, p0, p1)
    l5 = gmsh.model.geo.addCircleArc(p5, p0, p6)
    l6 = gmsh.model.geo.addCircleArc(p6, p0, p7)
    l7 = gmsh.model.geo.addCircleArc(p7, p0, p8)
    l8 = gmsh.model.geo.addCircleArc(p8, p0, p5)
    l9 = gmsh.model.geo.addLine(p5, p1)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p7, p3)
    l12 = gmsh.model.geo.addLine(p4, p8)

    cl = gmsh.model.geo.addCurveLoop((l1, l2, l3, l4))
    lower = gmsh.model.geo.addSurfaceFilling([cl], tag=boundaries.Lower.value)

    cl = gmsh.model.geo.addCurveLoop((l5, l6, l7, l8))
    upper = gmsh.model.geo.addSurfaceFilling([cl], tag=boundaries.Upper.value)

    cl = gmsh.model.geo.addCurveLoop((l10, l6, l11, -l2))
    east = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.East.value)

    cl = gmsh.model.geo.addCurveLoop((l9, -l4, l12, l8))
    west = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.West.value)

    cl = gmsh.model.geo.addCurveLoop((l1, l10, -l5, l9))
    south = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.South.value)

    cl = gmsh.model.geo.addCurveLoop((-l3, -l11, l7, -l12))
    north = gmsh.model.geo.addPlaneSurface([cl], tag=boundaries.North.value)

    sloop = gmsh.model.geo.addSurfaceLoop([south, east, north, upper, west, lower])
    volume = gmsh.model.geo.addVolume([sloop])

    gmsh.model.geo.synchronize()

    # Add Physical groups
    for b in boundaries:
        tag = b.value
        name = b.name
        gmsh.model.addPhysicalGroup(2, [tag], tag)
        gmsh.model.setPhysicalName(2, tag, name)

    # Add the volume entity to a physical group with a high tag number (99999) and name it "Elements"
    gmsh.model.addPhysicalGroup(3, [volume], 99999)
    gmsh.model.setPhysicalName(3, 99999, "Elements")

    gmsh.model.occ.synchronize()

    # Add the post-processing view as a new size field:
    bg_field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(bg_field, "ViewIndex", 0)
    
    # Apply the view as the current background mesh size field:
    gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)
    
    # In order to compute the mesh sizes from the background mesh only, and
    # disregard any other size constraints, one can set:
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(3)
    gmsh.write(uw_filename)
    gmsh.finalize()

    return

# create refined mesh
SegmentofSphere(radiusOuter=r_o, radiusInner=r_i,
                longitudeExtent=lon_ext, latitudeExtent=lat_ext,
                cellSize=cellsize, filename=f'{output_dir}{meshname}_refined.msh')

# convert msh to vtk
# !gmsh $f'{output_dir}{meshname}_refined.msh' -0 -format vtk -o $f'{output_dir}{meshname}_refined.vtk'

# pyvista visualization
test_mesh = pv.read(f'{output_dir}{meshname}_refined.msh')
test_mesh.plot(show_edges=True,)



