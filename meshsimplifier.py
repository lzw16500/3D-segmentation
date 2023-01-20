import numpy as np
import open3d as o3d

# http://www.open3d.org/docs/0.12.0/tutorial/geometry/mesh.html

subj = 1

# pcd = o3d.io.read_point_cloud(f"pp{subj}_original_labeled.ply")

mesh = o3d.io.read_triangle_mesh(f"pp{subj}_original_labeled.ply")



o3d.visualization.draw_geometries([mesh])

# pcd = o3d.io.read_point_cloud(f"pp{subj}_original_labeled.ply")
# o3d.visualization.draw_geometries([pcd])



mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=40000)

o3d.visualization.draw_geometries([mesh_smp])

o3d.io.write_triangle_mesh(f"./simplified/pp{subj}_original_labeled.ply", mesh_smp)

# o3d.io.write_point_cloud(f"./simplified/pp{subj}_original_labeled.ply", mesh_smp)
