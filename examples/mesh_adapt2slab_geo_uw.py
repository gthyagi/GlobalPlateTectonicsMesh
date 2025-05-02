# ### Create / Save mesh adapted to Slab2 geometry
#
# 1. We use built-in Underworld and PETSc tools for mesh adaptation.
# 2. Limitation: all metadata (labels, tags, etc.) is embedded in the DMPlex object and can only be preserved in `.h5` format, not in standard mesh formats required for further operations with MMG.
#

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

# mesh variables
fault_distance = uw.discretisation.MeshVariable("df", mesh, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"d_{F}")
H = uw.discretisation.MeshVariable("H", mesh, 1)
Metric = uw.discretisation.MeshVariable("M", mesh, 1, degree=1)
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
        # style="wireframe",
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
metric_on_fault =  1e5 # larger the value means smaller the cell size
metric_on_nonfault = 1e2
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
        clim= (metric_on_nonfault, metric_on_fault),
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
        clim=(metric_on_nonfault, metric_on_fault),
    )

    pl.show()

# +
# # both methods for checking is not working
# mesh.petsc_save_checkpoint(index=0, meshVars=[H, fault_distance], outputPath=output_dir)
# mesh.write_timestep('test', meshUpdates=True, meshVars=[H, fault_distance], outputPath=output_dir, index=0, )
# -

# #### Call mesh adaption from uw

# +
# mesh adaption using uw/petsc tools
icoord, meshA = uw.adaptivity.mesh_adapt_meshVar(mesh, H, Metric)

# mesh variables on adapted mesh
fault_distance_A = uw.discretisation.MeshVariable("df_A", meshA, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"d_{F}")
H_A = uw.discretisation.MeshVariable("H_A", meshA, 1)
Metric_A = uw.discretisation.MeshVariable("M_A", meshA, 1, degree=1)
# -

meshA.view()

# Initialise pyvista engine from existing mesh
pvmesh_A = uw.visualisation.mesh_to_pv_mesh(meshA)

if uw.mpi.size == 1:
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A,
        style="wireframe",
        cmap="Greys",
        use_transparency=False,
        show_edges=False,
        opacity=0.5,
        show_scalar_bar=False,
        line_width=0.1,
        )

    pl.add_mesh(slab_vtk)
    # pl.export_html(f"{output_dir}AdaptedFaultMeshWireframes.html")
    pl.show()

# Compute the implicit distance from the mesh nodes to a surface
with meshA.access(fault_distance_A):
    fault_distance_A.data[:, 0] = 1e10

    for i, segment in enumerate(surface_list):
        fault_segment_surface = surface_list[i]
        dist = pvmesh_A.compute_implicit_distance(fault_segment_surface)
        fault_distance_A.data[:, 0] = dist.point_data["implicit_distance"]
        
    print(fault_distance_A.data.min(), fault_distance_A.data.max())

if uw.mpi.size == 1:
    pvmesh_A.point_data["Fd"] = vis.scalar_fn_to_pv_points(pvmesh_A, fault_distance_A.sym)

    print(pvmesh_A.point_data["Fd"].min(), pvmesh_A.point_data["Fd"].max())
    
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A,
        scalars="Fd",
        cmap=plt.cm.Spectral.resampled(4),
        use_transparency=False,
        show_edges=False,
        opacity=1,
        show_scalar_bar=True,
        line_width=0.1,
        clim=(-0.005, 0.005),
        )
    
    pl.show()

# +
# Mesh Adaptation metric
metric_on_fault = 1e5 # larger the value means smaller the cell size
metric_on_nonfault = 1e2
zone_width = 1.0 # this value controls the width of the refinement around the slab

with meshA.access(H_A):
    H_A.data[:, 0] = uw.function.evalf(
        sympy.Piecewise(
            (metric_on_fault, fault_distance_A.sym[0] < meshA.get_min_radius() * zone_width),
            (metric_on_nonfault, True),
        ),
        H_A.coords,
    )

    print(H_A.data.min(), H_A.data.max())
# -

if uw.mpi.size == 1:
    pvmesh_A.point_data["H"] = vis.scalar_fn_to_pv_points(pvmesh_A, H_A.sym[0])

    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A,
        scalars="H",
        cmap=plt.cm.tab10.resampled(2),
        use_transparency=False,
        show_edges=True,
        opacity=1,
        show_scalar_bar=True,
        line_width=0.1,
        clim=(metric_on_nonfault, metric_on_fault),
        )
    
    pl.show()

# save adapted mesh as vtk
pvmesh_A.save(f'{output_dir}AdaptedFaultMesh.vtk')

# +
# second adaption of adapted mesh
icoord2, meshA2 = uw.adaptivity.mesh_adapt_meshVar(meshA, H_A, Metric_A)

# mesh variable
fault_distance_A2 = uw.discretisation.MeshVariable("df_A2", meshA2, vtype=uw.VarType.SCALAR, degree=1, varsymbol=r"d_{F}")
# -

meshA2.view()

# Initialise pyvista engine from existing mesh
pvmesh_A2 = vis.mesh_to_pv_mesh(meshA2)

if uw.mpi.size == 1:
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A2,
        style="wireframe",
        cmap="Greys",
        use_transparency=False,
        show_edges=False,
        opacity=0.5,
        show_scalar_bar=False,
        line_width=0.1,
        )

    pl.add_mesh(slab_vtk)
    # pl.export_html(f"{output_dir}AdaptedFaultMeshWireframes2.html")
    pl.show()

# Compute the implicit distance from the mesh nodes to a surface
with meshA2.access(fault_distance_A2):
    fault_distance_A2.data[:, 0] = 1e10

    for i, segment in enumerate(surface_list):
        fault_segment_surface = surface_list[i]
        dist = pvmesh_A2.compute_implicit_distance(fault_segment_surface)
        fault_distance_A2.data[:, 0] = dist.point_data["implicit_distance"]
        
    print(fault_distance_A2.data.min(), fault_distance_A2.data.max())

if uw.mpi.size == 1:
    pvmesh_A2.point_data["Fd"] = vis.scalar_fn_to_pv_points(pvmesh_A2, fault_distance_A2.sym)

    print(pvmesh_A2.point_data["Fd"].min(), pvmesh_A2.point_data["Fd"].max())
    
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A2,
        scalars="Fd",
        cmap=plt.cm.Spectral.resampled(4),
        use_transparency=False,
        show_edges=False,
        opacity=1,
        show_scalar_bar=True,
        line_width=0.1,
        clim=(-0.005, 0.005),
        )
    
    pl.show()

if uw.mpi.size == 1:
    # Extract the subset where field is negative
    negative_subset = pvmesh_A2.threshold(value=0.0, scalars="Fd", invert=True)

    print(f"Negative Fd values: min={negative_subset.point_data['Fd'].min()}, max={negative_subset.point_data['Fd'].max()}")

    # Plot the negative subset mesh
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        negative_subset,
        scalars="Fd",
        cmap=plt.cm.coolwarm_r.resampled(1), #plt.cm.Spectral.resampled(4),
        use_transparency=False,
        show_edges=True,
        opacity=1,
        show_scalar_bar=True,
        line_width=0.1,
        clim=(-0.005, 0),  # Adjust clim to negative values range
    )

    pl.show()

# save adapted mesh as vtk
pvmesh_A2.save(f'{output_dir}AdaptedFaultMesh2.vtk')



