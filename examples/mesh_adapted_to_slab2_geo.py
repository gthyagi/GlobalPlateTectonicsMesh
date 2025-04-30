# ### Create / Save mesh with adaptation to slab2 geometry

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import numpy as np
import underworld3 as uw
import sympy
import os

if uw.mpi.size == 1:
    import pyvista as pv
    from matplotlib import pyplot as plt
    import underworld3.visualisation as vis
# -

import h5py

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
cellsize = 1/32

meshname = f'uw_sos_ro{r_o}_ri{np.round(r_i, 2)}_lon{lon_ext}_lat{lat_ext}_csize{np.round(cellsize, 3)}'
# -

# uw mesh
mesh = uw.meshing.SegmentofSphere(radiusOuter=r_o, radiusInner=r_i,
                                  longitudeExtent=lon_ext, latitudeExtent=lat_ext,
                                  cellSize=cellsize, filename=f'{output_dir}{meshname}.msh')

mesh.view()

# data on the mesh
fault_distance = uw.discretisation.MeshVariable("df", mesh, vtype=uw.VarType.SCALAR, degree=1, 
                                               varsymbol=r"d_{F}")
H = uw.discretisation.MeshVariable("H", mesh, 1)
Metric = uw.discretisation.MeshVariable("M", mesh, 1, degree=1)

# slab surface vtk
# sum_top_surf = pv.read(f'{output_dir}sum_top_surf_mesh_mmg.vtk')
sum_top_surf = pv.read(f'{output_dir}sum_vol_from_top_surf_tet.vtk')
segment_surface_list = [sum_top_surf.extract_surface()]

# Initialise pyvista engine from existing mesh
pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)

if uw.mpi.size == 1:
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh,
        style="wireframe",
        # scalars="Fd",
        cmap="Greys",
        use_transparency=False,
        show_edges=False,
        opacity=0.2,
        show_scalar_bar=False,
        line_width=0.1,
        )

    pl.add_mesh(sum_top_surf)
    pl.export_html(f"{output_dir}NonAdaptedFaultMeshWireframes.html")
    pl.show()

# +
# These distances are to the interpolated surface, so they are more accurate that nearest neighbour from
# a k-D tree.

with mesh.access(fault_distance):
    
    fault_distance.data[:, 0] = 1e10

    for i, segment in enumerate(segment_surface_list):
        fault_segment_surface = segment_surface_list[i]
        dist = pvmesh.compute_implicit_distance(fault_segment_surface)

        # fault_distance.data[:, 0] = np.minimum(
        #     fault_distance.data[:, 0], np.abs(dist.point_data["implicit_distance"])
        # )
        fault_distance.data[:, 0] = dist.point_data["implicit_distance"]
        
    print(fault_distance.data.min(), fault_distance.data.max())
# -

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
        clim=(-0.005, 0.005),
        )
    
    pl.show()

# Mesh Adaptation metric
metric_on_fault = 1e5 # larger the value means smaller the cell size
metric_on_nonfault = 1e2
with mesh.access(H):
    H.data[:, 0] = uw.function.evalf(
        sympy.Piecewise(
            (metric_on_fault, fault_distance.sym[0] < mesh.get_min_radius() * 1.0),
            (metric_on_nonfault, True),
        ),
        H.coords,
    )

    print(H.data.min(), H.data.max())

if uw.mpi.size == 1:
    pvmesh.point_data["H"] = vis.scalar_fn_to_pv_points(pvmesh, H.sym[0])

    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh,
        # style="wireframe",
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

with mesh.access(Metric):
    print(Metric.data.min(), Metric.data.max())

icoord, meshA = uw.adaptivity.mesh_adapt_meshVar(mesh, H, Metric)

with mesh.access(Metric):
    print(Metric.data.min(), Metric.data.max())

mesh.dm.metricCreateIsotropic()

# data on the mesh
fault_distance_A = uw.discretisation.MeshVariable("df_A", meshA, vtype=uw.VarType.SCALAR, degree=1, 
                                               varsymbol=r"d_{F}")
H_A = uw.discretisation.MeshVariable("H_A", meshA, 1)
Metric_A = uw.discretisation.MeshVariable("M_A", meshA, 1, degree=1)

meshA.view()

if uw.mpi.size == 1:
    pvmesh_A = vis.mesh_to_pv_mesh(meshA)
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A,
        style="wireframe",
        # scalars="Fd",
        cmap="Greys",
        use_transparency=False,
        show_edges=False,
        opacity=0.5,
        show_scalar_bar=False,
        line_width=0.1,
        )

    pl.add_mesh(sum_top_surf)
    pl.export_html(f"{output_dir}AdaptedFaultMeshWireframes.html")
    pl.show()

# +
# These distances are to the interpolated surface, so they are more accurate that nearest neighbour from
# a k-D tree.

with meshA.access(fault_distance_A):

    # Initialise pyvista engine from existing mesh
    pvmesh_A = uw.visualisation.mesh_to_pv_mesh(meshA)
    
    fault_distance_A.data[:, 0] = 1e10

    for i, segment in enumerate(segment_surface_list):
        fault_segment_surface = segment_surface_list[i]
        dist = pvmesh_A.compute_implicit_distance(fault_segment_surface)

        # fault_distance.data[:, 0] = np.minimum(
        #     fault_distance.data[:, 0], np.abs(dist.point_data["implicit_distance"])
        # )
        fault_distance_A.data[:, 0] = dist.point_data["implicit_distance"]
        
    print(fault_distance_A.data.min(), fault_distance_A.data.max())
# -

if uw.mpi.size == 1:
    pvmesh_A.point_data["Fd"] = vis.scalar_fn_to_pv_points(pvmesh_A, fault_distance_A.sym)

    print(pvmesh_A.point_data["Fd"].min(), pvmesh_A.point_data["Fd"].max())
    
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A,
        # style="wireframe",
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

# Mesh Adaptation metric
metric_on_fault = 1e5 # larger the value means smaller the cell size
metric_on_nonfault = 1e2
with meshA.access(H_A):
    H_A.data[:, 0] = uw.function.evalf(
        sympy.Piecewise(
            (metric_on_fault, fault_distance_A.sym[0] < meshA.get_min_radius() * 1.0),
            (metric_on_nonfault, True),
        ),
        H_A.coords,
    )

    print(H_A.data.min(), H_A.data.max())

if uw.mpi.size == 1:
    pvmesh_A.point_data["H"] = vis.scalar_fn_to_pv_points(pvmesh_A, H_A.sym[0])

    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A,
        # style="wireframe",
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

icoord2, meshA2 = uw.adaptivity.mesh_adapt_meshVar(meshA, H_A, Metric_A)

if uw.mpi.size == 1:
    pvmesh_A2 = vis.mesh_to_pv_mesh(meshA2)
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A2,
        style="wireframe",
        # scalars="Fd",
        cmap="Greys",
        use_transparency=False,
        show_edges=False,
        opacity=0.5,
        show_scalar_bar=False,
        line_width=0.1,
        )

    pl.add_mesh(sum_top_surf)
    pl.export_html(f"{output_dir}AdaptedFaultMeshWireframes2.html")
    pl.show()

# data on the mesh
fault_distance_A2 = uw.discretisation.MeshVariable("df_A2", meshA2, vtype=uw.VarType.SCALAR, degree=1, 
                                                   varsymbol=r"d_{F}")

# +
# These distances are to the interpolated surface, so they are more accurate that nearest neighbour from
# a k-D tree.

with meshA2.access(fault_distance_A2):

    # Initialise pyvista engine from existing mesh
    pvmesh_A2 = uw.visualisation.mesh_to_pv_mesh(meshA2)
    
    fault_distance_A2.data[:, 0] = 1e10

    for i, segment in enumerate(segment_surface_list):
        fault_segment_surface = segment_surface_list[i]
        dist = pvmesh_A2.compute_implicit_distance(fault_segment_surface)

        # fault_distance.data[:, 0] = np.minimum(
        #     fault_distance.data[:, 0], np.abs(dist.point_data["implicit_distance"])
        # )
        fault_distance_A2.data[:, 0] = dist.point_data["implicit_distance"]
        
    print(fault_distance_A2.data.min(), fault_distance_A2.data.max())
# -

if uw.mpi.size == 1:
    pvmesh_A2.point_data["Fd"] = vis.scalar_fn_to_pv_points(pvmesh_A2, fault_distance_A2.sym)

    print(pvmesh_A2.point_data["Fd"].min(), pvmesh_A2.point_data["Fd"].max())
    
    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_mesh(
        pvmesh_A2,
        # style="wireframe",
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

# save adapted mesh as vtk
pvmesh_A2.save(f'{output_dir}AdaptedFaultMesh2.vtk')

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



