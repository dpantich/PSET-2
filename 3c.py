import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def lifting_map(x, y):
    """Lift 2D points to 3D using z = x^2 + y^2"""
    return x**2 + y**2

def compute_induced_metric(points_2D):
    """Compute the induced metric after lifting to 3D."""
    g_xx = 1 + (2 * points_2D[:, 0])**2  # 1 + (dz/dx)^2
    g_yy = 1 + (2 * points_2D[:, 1])**2  # 1 + (dz/dy)^2
    g_xy = (2 * points_2D[:, 0]) * (2 * points_2D[:, 1])  # (dz/dx)(dz/dy)
    
    return g_xx, g_yy, g_xy

def compute_triangle_metric(delaunay, metric_values):
    """Convert point-based metric values to triangle-based by averaging over each triangle."""
    return np.array([np.mean(metric_values[simplex]) for simplex in delaunay.simplices])

# Load the 2D point cloud
points_2D = np.loadtxt("mesh.dat", skiprows=1)

# Compute Delaunay triangulation in 2D
delaunay = Delaunay(points_2D)

# Compute induced metric components per point
g_xx, g_yy, g_xy = compute_induced_metric(points_2D)

# Convert point-based metric values to per-triangle values
g_xx_tri = compute_triangle_metric(delaunay, g_xx)
g_yy_tri = compute_triangle_metric(delaunay, g_yy)
g_xy_tri = compute_triangle_metric(delaunay, g_xy)

# Create heatmaps for g_xx, g_yy, and g_xy
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for ax, data, title in zip(axs, [g_xx_tri, g_yy_tri, g_xy_tri], ["g_xx", "g_yy", "g_xy"]):
    tpc = ax.tripcolor(points_2D[:, 0], points_2D[:, 1], delaunay.simplices, facecolors=data, cmap="viridis", edgecolors='k')
    fig.colorbar(tpc, ax=ax, label=title)
    ax.set_title(f"Induced Metric Component: {title}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

plt.savefig("3c_induced_metric_heatmap.png")
plt.show()
