# +
import gmsh
import math
import sys

gmsh.initialize()
gmsh.model.add("signed_distance_circle_in_square")

# --- 1. Define geometry ---

# Outer square (box from -1 to 1)
square_points = [
    gmsh.model.occ.addPoint(-1, -1, 0),
    gmsh.model.occ.addPoint( 1, -1, 0),
    gmsh.model.occ.addPoint( 1,  1, 0),
    gmsh.model.occ.addPoint(-1,  1, 0)
]
square_lines = [
    gmsh.model.occ.addLine(square_points[0], square_points[1]),
    gmsh.model.occ.addLine(square_points[1], square_points[2]),
    gmsh.model.occ.addLine(square_points[2], square_points[3]),
    gmsh.model.occ.addLine(square_points[3], square_points[0])
]
square_loop = gmsh.model.occ.addCurveLoop(square_lines)
square_surface = gmsh.model.occ.addPlaneSurface([square_loop])

# Inner circle
r = 0.3
circle = gmsh.model.occ.addCircle(0, 0, 0, r)
circle_loop = gmsh.model.occ.addCurveLoop([circle])
circle_surface = gmsh.model.occ.addPlaneSurface([circle_loop])

# Subtract circle from square to get annular region
gmsh.model.occ.cut([(2, square_surface)], [(2, circle_surface)], removeObject=True, removeTool=True)
gmsh.model.occ.synchronize()

# --- 2. Define signed distance-like field ---
# Gmsh Distance fields are unsigned, so we mimic signed by combining two fields:
# - Distance from circle (inside)
# - Distance from square edges (outside)
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "EdgesList", [circle])

# Optional: mimic sign by flipping inside and clamping outer region
# Use Threshold to apply mesh size based on distance from circle
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.01)
gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.2)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.3)

# Apply field as background
gmsh.model.mesh.field.setAsBackgroundMesh(2)

# --- 3. Generate mesh ---
gmsh.model.mesh.generate(2)

# --- 4. Export ---
gmsh.write("circle_in_square_signed_distance_refined.vtk")

# if "-nopopup" not in sys.argv:
#     gmsh.fltk.run()

gmsh.finalize()

# -

gmsh.model.mesh.


