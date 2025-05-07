# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import gmsh
import math

def name_and_tag_surfaces(tol=1e-3):
    """
    Identify and tag each surface with a Physical Group and name based on its geometry:
      - Box faces: Left (x=0), Right (x=2), Front (y=0), Back (y=2), Bottom (z=0), Top (z=2)
      - Internal spherical interface: Everything else
    """
    surfaces = gmsh.model.getEntities(dim=2)
    for _, tag in surfaces:
        x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.getBoundingBox(2, tag)
        if abs(x_min) < tol and abs(x_max) < tol:
            face = "Left"
        elif abs(x_min - 2) < tol and abs(x_max - 2) < tol:
            face = "Right"
        elif abs(y_min) < tol and abs(y_max) < tol:
            face = "Front"
        elif abs(y_min - 2) < tol and abs(y_max - 2) < tol:
            face = "Back"
        elif abs(z_min) < tol and abs(z_max) < tol:
            face = "Bottom"
        elif abs(z_min - 2) < tol and abs(z_max - 2) < tol:
            face = "Top"
        else:
            face = "SphereInterface"
        phys = gmsh.model.addPhysicalGroup(2, [tag])
        gmsh.model.setPhysicalName(2, phys, face)


def name_and_tag_volumes(sphere_center=(1,1,1), sphere_radius=1.0):
    """
    Identify and tag volumes as inside or outside the sphere:
      - Inside: volume centroid within sphere_radius of sphere_center
      - Outside: otherwise
    """
    volumes = gmsh.model.getEntities(dim=3)
    inside_tags = []
    outside_tags = []
    cx, cy, cz = sphere_center

    for _, tag in volumes:
        print('tag: ', tag)
        x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.getBoundingBox(3, tag)
        # centroid of bounding box as proxy for element location
        mx = 0.5*(x_min + x_max)
        my = 0.5*(y_min + y_max)
        mz = 0.5*(z_min + z_max)
        # distance from sphere center
        d = math.sqrt((mx-cx)**2 + (my-cy)**2 + (mz-cz)**2)
        if d < sphere_radius:
            inside_tags.append(tag)
        else:
            outside_tags.append(tag)

    print(inside_tags, outside_tags)
    pid_in = gmsh.model.addPhysicalGroup(3, inside_tags)
    gmsh.model.setPhysicalName(3, pid_in, "InsideSphere")
    pid_out = gmsh.model.addPhysicalGroup(3, outside_tags)
    gmsh.model.setPhysicalName(3, pid_out, "OutsideSphere")

# 1) Initialize and name the model
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.model.add("box_sphere_named_surfaces_volumes")

# 2) Create a box from (0,0,0) to (2,2,2)
box = gmsh.model.occ.addBox(0, 0, 0, 2, 2, 2)

# 3) Create a unit sphere centered at (1,1,1)
sphere = gmsh.model.occ.addSphere(1, 1, 1, 1)

# 4) Fuse & split volumes (Boolean fragment)
gmsh.model.occ.fragment([(3, box)], [(3, sphere)])

# 5) Sync CAD with Gmsh model before tagging or meshing
gmsh.model.occ.synchronize()

# 6) Tag surfaces by name
name_and_tag_surfaces()

# 7) Tag volumes inside/outside sphere
name_and_tag_volumes(sphere_center=(1,1,1), sphere_radius=1.0)

gmsh.model.occ.synchronize()

# 8) Generate a 3D meshing
gmsh.model.mesh.generate(3)

# 9) Write mesh to file for later use
gmsh.write("box_sphere_named_surfaces_volumes.msh")
# gmsh.write("box_sphere_named_surfaces_volumes.vtk")

# # 10) Launch FLTK viewer to inspect geometry & mesh
# gmsh.fltk()

# 11) Clean up and finalize
gmsh.finalize()

# -

import underworld3 as uw

mesh = uw.discretisation.Mesh("box_sphere_named_surfaces_volumes.msh", # boundaries=boundaries, 
                              # boundary_normals=boundary_normals_2D,
                              markVertices=True, useMultipleTags=True, useRegions=True,
                              coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN, )

mesh.dm.view()

import pyvista as pv

import underworld3.visualisation as vis

pv_mesh = vis.mesh_to_pv_mesh(mesh)

pv_mesh


