import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from numpy.linalg import eig, inv

def lifting_map(x, y):
    """Lift 2D points to 3D using z = x^2 + y^2"""
    return x**2 + y**2

def compute_vertex_normals(points_3D, delaunay):
    """Compute vertex normals for the lifted mesh."""
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

def compute_second_fundamental_form(points_2D, vertex_normals):
    """Compute the second fundamental form components analytically."""
    L = 2  # d²z/dx² = 2
    M = 0  # d²z/dxdy = 0
    N = 2  # d²z/dy² = 2
    
    II = np.zeros((len(points_2D), 2, 2))  # Second fundamental form per vertex
    for i, normal in enumerate(vertex_normals):
        II[i, 0, 0] = L * np.dot(normal, [1, 0, 0])  # L * e_x . N
        II[i, 0, 1] = M * np.dot(normal, [1, 0, 0])  # M * e_x . N
        II[i, 1, 0] = M * np.dot(normal, [0, 1, 0])  # M * e_y . N
        II[i, 1, 1] = N * np.dot(normal, [0, 1, 0])  # N * e_y . N
    
    return II

def compute_shape_operator(II):
    """Compute the shape operator from the second fundamental form."""
    shape_operators = np.array([-inv(II[i]) for i in range(len(II))])
    return shape_operators

def compute_curvatures(shape_operators):
    """Compute principal, Gaussian, and mean curvatures from the shape operator."""
    principal_curvatures = np.array([eig(shape_operators[i])[0] for i in range(len(shape_operators))])
    k1 = np.max(principal_curvatures, axis=1)  # Maximum principal curvature
    k2 = np.min(principal_curvatures, axis=1)  # Minimum principal curvature
    Gaussian_curvature = k1 * k2
    Mean_curvature = (k1 + k2) / 2
    return k1, k2, Gaussian_curvature, Mean_curvature

def convert_vertex_to_triangle_values(delaunay, vertex_values):
    """Convert per-vertex values to per-triangle values by averaging over each triangle's vertices."""
    return np.array([np.mean(vertex_values[simplex]) for simplex in delaunay.simplices])

# Load the 2D point cloud
points_2D = np.loadtxt("mesh.dat", skiprows=1)

# Compute Delaunay triangulation in 2D
delaunay = Delaunay(points_2D)

# Lift the points to 3D
points_3D = np.column_stack((points_2D, lifting_map(points_2D[:, 0], points_2D[:, 1])))

# Compute vertex normals
vertex_normals = compute_vertex_normals(points_3D, delaunay)

# Compute second fundamental form
second_fundamental_form = compute_second_fundamental_form(points_2D, vertex_normals)

# Compute shape operator
shape_operators = compute_shape_operator(second_fundamental_form)

# Compute curvatures
k1, k2, Gaussian_curvature, Mean_curvature = compute_curvatures(shape_operators)

# Convert per-vertex curvature values to per-triangle values
k1_tri = convert_vertex_to_triangle_values(delaunay, k1)
k2_tri = convert_vertex_to_triangle_values(delaunay, k2)
Gaussian_curvature_tri = convert_vertex_to_triangle_values(delaunay, Gaussian_curvature)
Mean_curvature_tri = convert_vertex_to_triangle_values(delaunay, Mean_curvature)

# Plot curvature heatmaps
fig, axs = plt.subplots(1, 4, figsize=(20, 6))
curvatures = [k1_tri, k2_tri, Gaussian_curvature_tri, Mean_curvature_tri]
titles = ["Max Principal Curvature (k1)", "Min Principal Curvature (k2)", "Gaussian Curvature", "Mean Curvature"]

for ax, data, title in zip(axs, curvatures, titles):
    tpc = ax.tripcolor(points_2D[:, 0], points_2D[:, 1], delaunay.simplices, facecolors=data, cmap="viridis", edgecolors='k')
    fig.colorbar(tpc, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

plt.savefig("3g_curvature_heatmaps.png")
plt.show()
