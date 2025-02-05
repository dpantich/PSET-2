import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stereographic_projection(x, y, z):
    """Compute stereographic projection of (x, y, z) onto the xy-plane"""
    x_proj = x / (1 - z)
    y_proj = y / (1 - z)
    return x_proj, y_proj

def compute_tangent_vectors(curve1, curve2):
    """Compute tangent vectors at intersection points"""
    tangent1 = np.gradient(curve1, axis=0)
    tangent2 = np.gradient(curve2, axis=0)
    
    # Normalize vectors
    tangent1 /= np.linalg.norm(tangent1, axis=1)[:, np.newaxis]
    tangent2 /= np.linalg.norm(tangent2, axis=1)[:, np.newaxis]
    
    return tangent1, tangent2

def angle_between_vectors(v1, v2):
    """Compute the angle (in degrees) between two tangent vectors"""
    dot_product = np.sum(v1 * v2, axis=1)
    angles = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angles)

# Generate unit sphere mesh
theta = np.linspace(0, np.pi, 30)  # Latitude
phi = np.linspace(0, 2 * np.pi, 60)  # Longitude
Theta, Phi = np.meshgrid(theta, phi)

# Convert to Cartesian coordinates
X = np.sin(Theta) * np.cos(Phi)
Y = np.sin(Theta) * np.sin(Phi)
Z = np.cos(Theta)

# Define two curves: meridians and parallels
theta_curve = np.linspace(0, np.pi - 0.1, 30)  # Avoid z=1 (North Pole)
phi_curve = np.linspace(0, 2 * np.pi, 30)

# First curve: Meridional (longitudinal curve)
x1 = np.sin(theta_curve) * np.cos(0)
y1 = np.sin(theta_curve) * np.sin(0)
z1 = np.cos(theta_curve)

# Second curve: Parallel (latitudinal curve)
x2 = np.sin(np.pi / 3) * np.cos(phi_curve)
y2 = np.sin(np.pi / 3) * np.sin(phi_curve)
z2 = np.full_like(phi_curve, np.cos(np.pi / 3))

# Compute tangent vectors on the sphere
tangent1, tangent2 = compute_tangent_vectors(np.column_stack((x1, y1, z1)),
                                             np.column_stack((x2, y2, z2)))

# Compute angles on the sphere
angles_sphere = angle_between_vectors(tangent1, tangent2)

# Apply stereographic projection
x1_proj, y1_proj = stereographic_projection(x1, y1, z1)
x2_proj, y2_proj = stereographic_projection(x2, y2, z2)

# Compute tangent vectors after projection
tangent1_proj, tangent2_proj = compute_tangent_vectors(np.column_stack((x1_proj, y1_proj, np.zeros_like(x1_proj))),
                                                       np.column_stack((x2_proj, y2_proj, np.zeros_like(x2_proj))))

# Compute angles in the projection plane
angles_proj = angle_between_vectors(tangent1_proj, tangent2_proj)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# --- First subplot: Sphere ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, color='c', alpha=0.3, edgecolor='gray')
ax1.plot(x1, y1, z1, color='r', label="Meridian")
ax1.plot(x2, y2, z2, color='b', label="Parallel")
ax1.set_title("Curves on the Unit Sphere")
ax1.legend()

# --- Second subplot: Projection ---
ax2 = axs[1]
ax2.plot(x1_proj, y1_proj, color='r', label="Projected Meridian")
ax2.plot(x2_proj, y2_proj, color='b', label="Projected Parallel")
ax2.set_title("Stereographic Projection")
ax2.legend()
ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)

# Display the plots
plt.savefig("stereographic_projection.png")
plt.show()
