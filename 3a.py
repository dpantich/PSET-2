import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

# Load the 2D point cloud
points = np.loadtxt("mesh.dat", skiprows=1)  # Ensure mesh.dat is in the working directory

# Compute the convex hull
hull = ConvexHull(points)

# Compute the Delaunay triangulation
delaunay = Delaunay(points)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# --- First subplot: Convex Hull ---
axs[0].scatter(points[:, 0], points[:, 1], color="r", label="Points")
for simplex in hull.simplices:
    axs[0].plot(points[simplex, 0], points[simplex, 1], "k-")  # Plot hull edges

axs[0].set_title("Convex Hull")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].legend()

# --- Second subplot: Delaunay Triangulation ---
axs[1].triplot(points[:, 0], points[:, 1], delaunay.simplices, "b-")  # Triangulation edges
axs[1].scatter(points[:, 0], points[:, 1], color="r", label="Points")

axs[1].set_title("Delaunay Triangulation")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].legend()

# Save and display the plot
plt.savefig("3a_convex_hull_delaunay_2D.png")
plt.show()
