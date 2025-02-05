import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to Cartesian"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def spherical_basis_vectors(theta, phi):
    """Compute spherical basis vectors in Cartesian coordinates"""
    e_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_r, e_theta, e_phi

# Generate the sphere
u = np.linspace(0, np.pi, 20)  # Theta
v = np.linspace(0, 2 * np.pi, 40)  # Phi
U, V = np.meshgrid(u, v)

# Convert to Cartesian coordinates
X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

# Create figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the unit sphere
ax.plot_surface(X, Y, Z, color='gray', alpha=0.3)

# Choose sample points to plot basis vectors
theta_vals = np.linspace(0.1, np.pi - 0.1, 6)  # Avoid poles
phi_vals = np.linspace(0, 2 * np.pi, 12)

for theta in theta_vals:
    for phi in phi_vals:
        point = spherical_to_cartesian(1, theta, phi)
        e_r, e_theta, e_phi = spherical_basis_vectors(theta, phi)
        
        # Scale for visibility
        scale = 0.2  
        
        # Plot basis vectors
        ax.quiver(*point, *(scale * e_r), color='r', label=r'$\hat{e}_r$' if theta == theta_vals[0] and phi == phi_vals[0] else "")
        ax.quiver(*point, *(scale * e_theta), color='g', label=r'$\hat{e}_\theta$' if theta == theta_vals[0] and phi == phi_vals[0] else "")
        ax.quiver(*point, *(scale * e_phi), color='b', label=r'$\hat{e}_\phi$' if theta == theta_vals[0] and phi == phi_vals[0] else "")

# Labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Unit Sphere with Spherical Basis Vectors")

# Avoid duplicate labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

# Save the plot
plt.savefig("spherical_basis_vectors.png")

plt.show()
