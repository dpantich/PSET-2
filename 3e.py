import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def lifting_map(x, y):
    """Lift 2D points to 3D using z = x^2 + y^2"""
    return x**2 + y**2

def compute_surface_normals(points_3D, delaunay):
    """Compute surface normals for each triangle in the lifted mesh."""
    normals = np.zeros((len(points_3D), 3))
    count = np.zeros(len(points_3D))
    
    for simplex in delaunay.simplices:
        A, B, C = points_3D[simplex]
        normal = np.cross(B - A, C - A)
        normal /= np.linalg.norm(normal)  # Normalize the normal vector
        
        for i in simplex:
            normals[i] += normal
            count[i] += 1
    
    # Normalize the vertex normals
    vertex_normals = normals / count[:, np.newaxis]
    return vertex_normals

# Load the 2D point cloud
points_2D = np.loadtxt("mesh.dat", skiprows=1)

# Compute Delaunay triangulation in 2D
delaunay = Delaunay(points_2D)

# Lift the points to 3D
points_3D = np.column_stack((points_2D, lifting_map(points_2D[:, 0], points_2D[:, 1])))

# Compute vertex normals
vertex_normals = compute_surface_normals(points_3D, delaunay)

# Plot the lifted mesh with vertex normals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], color="b", alpha=0.5, label="Lifted Points")

# Plot Delaunay triangulation in 3D
for simplex in delaunay.simplices:
    tri_vertices = points_3D[simplex]
    ax.add_collection3d(Poly3DCollection([tri_vertices], alpha=0.3, edgecolor='k'))

# Plot vertex normals
ax.quiver(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], 
          vertex_normals[:, 0], vertex_normals[:, 1], vertex_normals[:, 2], length=0.3, color='r', normalize=True)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lifted Mesh with Vertex Normals")
plt.savefig("3e_lifted_mesh_vertex_normals.png")
plt.show()
