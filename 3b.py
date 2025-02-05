import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def lifting_map(x, y):
    """Lift 2D points to 3D using z = x^2 + y^2"""
    return x**2 + y**2

def triangle_area(A, B, C):
    """Compute the area of a triangle given its three vertices in 2D or 3D"""
    return 0.5 * np.linalg.norm(np.cross(B - A, C - A))

# Load the 2D point cloud
points_2D = np.loadtxt("mesh.dat", skiprows=1)

# Compute Delaunay triangulation in 2D
delaunay = Delaunay(points_2D)

# Lift the points to 3D
points_3D = np.column_stack((points_2D, lifting_map(points_2D[:, 0], points_2D[:, 1])))

# Compute area change for each triangle
area_ratios = []
for simplex in delaunay.simplices:
    A_2D, B_2D, C_2D = points_2D[simplex]
    A_3D, B_3D, C_3D = points_3D[simplex]
    
    area_2D = triangle_area(A_2D, B_2D, C_2D)
    area_3D = triangle_area(A_3D, B_3D, C_3D)
    
    area_ratio = area_3D / area_2D if area_2D > 0 else 0  # Avoid division by zero
    area_ratios.append(area_ratio)

# Create heatmap of area ratios using `tripcolor`
plt.figure(figsize=(8,6))
plt.tripcolor(points_2D[:, 0], points_2D[:, 1], delaunay.simplices, facecolors=area_ratios, cmap="viridis", edgecolors='k')
plt.colorbar(label="Area Ratio (Lifted / Original)")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Area Change After Lifting Map")
plt.savefig("3b_lifting_map_heatmap.png")
plt.show()
