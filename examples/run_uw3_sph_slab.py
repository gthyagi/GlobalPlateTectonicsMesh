# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
import underworld3.visualisation as vis
from enum import Enum
import sympy
import cmcrameri.cm as cmc
import os
from petsc4py import PETSc
import pyvista as pv
import matplotlib.pyplot as plt

# output dir
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)


# +
# class surface_boundaries(Enum):
#     slab_surface = 10
#     Lower        = 11
#     Upper        = 12
#     East         = 13
#     West         = 14
#     South        = 15
#     North        = 16

# class element_tag(Enum):
#     inside_slab  = 3
#     outside_slab = 2
    
class surface_boundaries(Enum):
    slab_surface = 1
    Lower        = 2
    Upper        = 3
    East         = 4
    West         = 5
    South        = 6
    North        = 7

class element_tag(Enum):
    inside_slab  = 8
    outside_slab = 9


# -

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
    clim=[8, 10],
    show_scalar_bar=True
)
pl.show(cpos="xy")
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

# +
# boundary conditions

# Noslip
# stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Upper.name)
# stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.Lower.name)
# stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.East.name)
# stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.West.name)
# stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.North.name)
# stokes.add_essential_bc(sympy.Matrix([0.0, 0.0, 0.0]), mesh.boundaries.South.name)

# Freeslip
Gamma = mesh.Gamma
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Upper")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Lower")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "North")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "East")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "West")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "South")
# -

gravity_fn = -1.0 * unit_rvec
stokes.bodyforce = rho.sym*gravity_fn

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
# saving h5 and xdmf file
mesh.petsc_save_checkpoint(index=0, meshVars=[v_soln, p_soln, rho, cell_tags], outputPath=os.path.relpath(output_dir)+'/output')



