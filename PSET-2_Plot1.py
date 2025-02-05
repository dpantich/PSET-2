#First Plot
##
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create the hemisphere
u = np.linspace(0, np.pi, 20)  # Theta range
v = np.linspace(0, np.pi, 20)  # Phi range
U, V = np.meshgrid(u, v)

# Convert spherical to Cartesian coordinates
X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

# Compute normal vectors (pointing outward)
N_x = X
N_y = Y
N_z = Z

# Compute tangent vectors (t)
T_x = -np.sin(V)
T_y = np.cos(V)
T_z = np.zeros_like(V)

# Compute cotangent vectors (ct)
CT_x = np.cos(U) * np.cos(V)
CT_y = np.cos(U) * np.sin(V)
CT_z = -np.sin(U)

# Create the plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the hemisphere
ax.plot_surface(X, Y, Z, color='gray', alpha=0.5)

# Plot normal vectors (red)
ax.quiver(X, Y, Z, N_x, N_y, N_z, color='r', length=0.1, normalize=True, label="n")

# Plot tangent vectors (green)
ax.quiver(X, Y, Z, T_x, T_y, T_z, color='g', length=0.1, normalize=True, label="t")

# Plot cotangent vectors (blue)
ax.quiver(X, Y, Z, CT_x, CT_y, CT_z, color='b', length=0.1, normalize=True, label="ct")

# Labels and legend
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Surface with Normal Vectors")
ax.legend()

# Save the plot
plt.savefig("surface_with_vectors.png")

plt.show()
