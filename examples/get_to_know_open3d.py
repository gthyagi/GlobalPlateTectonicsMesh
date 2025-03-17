import open3d as o3d
import numpy as np
import copy

# ### point cloud

print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)

print("Print a normal vector of the 0th point")
print(downpcd.normals[0])

print("Print the normal vectors of the first 10 points")
print(np.asarray(downpcd.normals)[:10, :])

print("Load a polygon volume and use it to crop the original point cloud")
demo_crop_data = o3d.data.DemoCropPointCloud()
pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
chair = vol.crop_point_cloud(pcd)
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

print("Paint chair")
chair.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

# +
# Load data
demo_crop_data = o3d.data.DemoCropPointCloud()
pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
chair = vol.crop_point_cloud(pcd)

dists = pcd.compute_point_cloud_distance(chair)
dists = np.asarray(dists)
ind = np.where(dists > 0.01)[0]
pcd_without_chair = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd_without_chair],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
# -

aabb = chair.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = chair.get_oriented_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([chair, aabb, obb],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

# +
bunny = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(bunny.path)
mesh.compute_vertex_normals()

pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
hull, _ = pcl.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([pcl, hull_ls])
# -

# ### mesh

# +
print("Testing mesh in Open3D...")
armadillo_mesh = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)

knot_mesh = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))
# -

print("Try to render a mesh with normals (exist: " +
      str(mesh.has_vertex_normals()) + ") and colors (exist: " +
      str(mesh.has_vertex_colors()) + ")")
o3d.visualization.draw_geometries([mesh])
print("A mesh with no normals and no colors does not look good.")

print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])

print("We make a partial mesh of only the first half triangles.")
mesh1 = copy.deepcopy(mesh)
mesh1.triangles = o3d.utility.Vector3iVector(
    np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
mesh1.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])
print(mesh1.triangles)
o3d.visualization.draw_geometries([mesh1])

print("Painting the mesh")
mesh1.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([mesh1])

# +
import os
import sys

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)
# -

import open3d_example as o3dex


def check_properties(name, mesh):
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(o3dex.edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(o3dex.edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        geoms.append(o3dex.edges_to_lineset(mesh, edges, (1, 0, 1)))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)


knot_mesh_data = o3d.data.KnotMesh()
knot_mesh = o3d.io.read_triangle_mesh(knot_mesh_data.path)
check_properties('KnotMesh', knot_mesh)
check_properties('Mobius', o3d.geometry.TriangleMesh.create_mobius(twists=1))
check_properties("non-manifold edge", o3dex.get_non_manifold_edge_mesh())
check_properties("non-manifold vertex", o3dex.get_non_manifold_vertex_mesh())
check_properties("open box", o3dex.get_open_box_mesh())
check_properties("intersecting_boxes", o3dex.get_intersecting_boxes_mesh())

import o3dex

# +
print("Testing kdtree in Open3D...")
print("Load a point cloud and paint it gray.")

sample_pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
pcd.paint_uniform_color([0.5, 0.5, 0.5])
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
# -

print("Paint the 1501st point red.")
pcd.colors[1500] = [1, 0, 0]

print("Find its 200 nearest neighbors, and paint them blue.")
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

print("Find its neighbors with distance less than 0.2, and paint them green.")
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

print("Visualize the point cloud.")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.5599,
                                  front=[-0.4958, 0.8229, 0.2773],
                                  lookat=[2.1126, 1.0163, -1.8543],
                                  up=[0.1007, -0.2626, 0.9596])

print("Testing IO for point cloud ...")
sample_pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
print(pcd)
o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)

# +
bunny = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(bunny.path)
mesh.compute_vertex_normals()

pcd = mesh.sample_points_poisson_disk(750)
o3d.visualization.draw_geometries([pcd])
alpha = 0.025
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# -

# Generate some neat n times 3 matrix using a variant of sync function
x = np.linspace(-3, 3, 401)
mesh_x, mesh_y = np.meshgrid(x, x)
z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
z_norm = (z - z.min()) / (z.max() - z.min())
xyz = np.zeros((np.size(mesh_x), 3))
xyz[:, 0] = np.reshape(mesh_x, -1)
xyz[:, 1] = np.reshape(mesh_y, -1)
xyz[:, 2] = np.reshape(z_norm, -1)
print('xyz')
print(xyz)

# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("sync.ply", pcd)

# Convert Open3D.o3d.geometry.PointCloud to numpy array
o3d.visualization.draw_geometries([pcd])

# +
# Load saved point cloud and visualize it
pcd_load = o3d.io.read_point_cloud("sync.ply")

# Convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_load = np.asarray(pcd_load.points)
print('xyz_load')
print(xyz_load)
o3d.visualization.draw_geometries([pcd_load])

# +
# Load mesh and convert to open3d.t.geometry.TriangleMesh
armadillo_data = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh(armadillo_data.path)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh

# +
query_point = o3d.core.Tensor([[10, 10, 10]], dtype=o3d.core.Dtype.Float32)

# Compute distance of the query point from the surface
unsigned_distance = scene.compute_distance(query_point)
signed_distance = scene.compute_signed_distance(query_point)
occupancy = scene.compute_occupancy(query_point)
# -

print("unsigned distance", unsigned_distance.numpy())
print("signed_distance", signed_distance.numpy())
print("occupancy", occupancy.numpy())

# +
min_bound = mesh.vertex.positions.min(0).numpy()
max_bound = mesh.vertex.positions.max(0).numpy()

N = 256
query_points = np.random.uniform(low=min_bound, high=max_bound,
                                 size=[N, 3]).astype(np.float32)

# Compute the signed distance for N random points
signed_distance = scene.compute_signed_distance(query_points)
# -

import matplotlib.pyplot as plt

# +
xyz_range = np.linspace(min_bound, max_bound, num=32)

# query_points is a [32,32,32,3] array ..
query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

# signed distance is a [32,32,32] array
signed_distance = scene.compute_signed_distance(query_points)

# We can visualize a slice of the distance field directly with matplotlib
plt.imshow(signed_distance.numpy()[:, :, 15])

# +
cube = o3d.t.geometry.TriangleMesh.from_legacy(
    o3d.geometry.TriangleMesh.create_box().translate([-1.2, -1.2, 0]))
sphere = o3d.t.geometry.TriangleMesh.from_legacy(
    o3d.geometry.TriangleMesh.create_sphere(0.5).translate([0.7, 0.8, 0]))

scene = o3d.t.geometry.RaycastingScene()
# Add triangle meshes and remember ids
mesh_ids = {}
mesh_ids[scene.add_triangles(cube)] = 'cube'
mesh_ids[scene.add_triangles(sphere)] = 'sphere'

# +
query_point = o3d.core.Tensor([[0, 0, 0]], dtype=o3d.core.Dtype.Float32)

# We compute the closest point on the surface for the point at position [0,0,0].
ans = scene.compute_closest_points(query_point)

# Compute_closest_points provides the point on the surface, the geometry id,
# and the primitive id.
# The dictionary keys are
#.    points
#.    geometry_ids
#.    primitive_ids
print('The closest point on the surface is', ans['points'].numpy())
print('The closest point is on the surface of the',
      mesh_ids[ans['geometry_ids'][0].item()])
print('The closest point belongs to triangle', ans['primitive_ids'][0].item())
# -

rays = np.concatenate(
    [query_point.numpy(),
     np.ones(query_point.shape, dtype=np.float32)],
    axis=-1)
intersection_counts = scene.count_intersections(rays).numpy()
# A point is inside if the number of intersections with the scene is even
# This assumes that inside and outside are well-defined for the scene.
is_inside = intersection_counts % 2 == 1


def compute_signed_distance_and_closest_goemetry(query_points: np.ndarray):
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    return distance, closest_points['geometry_ids'].numpy()


# +
# compute range
xyz_range = np.linspace([-2, -2, -2], [2, 2, 2], num=32)
# query_points is a [32,32,32,3] array ..
query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

sdf, closest_geom = compute_signed_distance_and_closest_goemetry(query_points)

# We can visualize a slice of the grids directly with matplotlib
fig, axes = plt.subplots(1, 2)
axes[0].imshow(sdf[:, :, 16])
axes[1].imshow(closest_geom[:, :, 16])

# +
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

# Example: load or define your slab surface data.
# Assume `points` is an (N, 3) array with columns [longitude, latitude, depth]
# For points on the slab surface, the implicit function value is 0.
# If needed, add extra off-surface points with small offsets (e.g., ±delta)
# to ensure a robust interpolation.

# Example slab surface points (replace with your actual data)
points = np.array([
    [100.0, 0.0, 50.0],
    [101.0, 0.5, 55.0],
    [102.0, 1.0, 60.0],
    [103.0, 1.5, 65.0]
])
# For on-surface points, the function value is 0.
values = np.zeros(points.shape[0])

# Optionally, add a few off-surface points for stability:
delta = 0.1
off_surface_points = np.array([
    [100.5, 0.2, 50.0 + delta],
    [101.5, 0.7, 55.0 - delta]
])
off_surface_values = np.array([delta, -delta])

# Combine the data
all_points = np.vstack([points, off_surface_points])
all_values = np.concatenate([values, off_surface_values])

# Create the RBF interpolator.
# You can experiment with different kernels such as 'multiquadric', 'gaussian',
# or 'thin_plate_spline'. The epsilon parameter controls the scaling.
rbf_interp = RBFInterpolator(all_points, all_values, kernel='thin_plate_spline', epsilon=1.0)

# Now define a grid or set of query points where you want to evaluate the implicit function.
# For illustration, here we evaluate on a grid in the first two dimensions.
lon_vals = np.linspace(90.0, 110.0, 50)
lat_vals = np.linspace(-5, 5, 50)
lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

# Here, we choose a fixed depth, or you can evaluate over a range of depths.
fixed_depth = 55.0
grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel(), np.full(lon_grid.size, fixed_depth)])

# Evaluate the interpolated function at the grid points.
f_values = rbf_interp(grid_points)
f_values_grid = f_values.reshape(lon_grid.shape)

# Visualize the level curve (zero-level set) at the fixed depth slice.
plt.contour(lon_grid, lat_grid, f_values_grid, levels=[0])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Zero-Level Set (Slab Interface) at Depth = {} km".format(fixed_depth))
plt.show()


# +
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

# Example slab surface points (replace with your actual data)
points = np.array([
    [100.0, 0.0, 50.0],
    [101.0, 0.5, 55.0],
    [102.0, 1.0, 60.0],
    [103.0, 1.5, 65.0]
])
# On-surface points have a function value of 0
values = np.zeros(points.shape[0])

# Adding a couple of off-surface points for stability
delta = 0.1
off_surface_points = np.array([
    [100.5, 0.2, 50.0 + delta],
    [101.5, 0.7, 55.0 - delta]
])
off_surface_values = np.array([delta, -delta])

# Combine the points and corresponding values
all_points = np.vstack([points, off_surface_points])
all_values = np.concatenate([values, off_surface_values])

# Create the RBF interpolator using the thin_plate_spline kernel
rbf_interp = RBFInterpolator(all_points, all_values, kernel='thin_plate_spline', epsilon=1.0)

# Define a mesh grid in the (lon, lat) plane at a fixed depth
lon_vals = np.linspace(90, 110, 100)
lat_vals = np.linspace(-5, 5, 100)
lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
fixed_depth = 55.0  # Change this to the depth slice you want to evaluate

# Create query points for the entire grid at the fixed depth
grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel(), np.full(lon_grid.size, fixed_depth)])
f_values = rbf_interp(grid_points)
f_values_grid = f_values.reshape(lon_grid.shape)

# Plot the levelset values using filled contours
plt.figure(figsize=(8, 6))
contourf = plt.contourf(lon_grid, lat_grid, f_values_grid, levels=50, cmap='viridis')
plt.colorbar(contourf, label='Implicit Function Value')

# Overlay the zero-level set (the interface) in red
plt.contour(lon_grid, lat_grid, f_values_grid, levels=[0], colors='red', linewidths=2)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Implicit Function Levelset at Depth = {fixed_depth} km")
plt.show()

# -


