import FreeCAD, Part, Mesh


def stl_to_step(stl_path, step_path):
    mesh = Mesh.Mesh(stl_path)
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology, 0.05)
    solid = Part.Solid(shape)
    Part.export([solid], step_path)


# Replace with your actual paths
stl_to_step("output/cap_surface.stl", "output/cap_surface.step")
stl_to_step("output/slab_surface.stl", "output/slab_surface.step")


