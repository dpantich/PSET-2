import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stereographic_projection(x, y, z):
    """Compute stereographic projection of (x, y, z) onto the xy-plane"""
    x_proj = x / (1 - z)
    y_proj = y / (1 - z)
    return x_proj, y_proj

def compute_inner_product(v1, v2):
    """Compute the inner product of two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def generate_vectors_at_point(theta, phi):
    """Generate two tangent vectors at a given point on the sphere."""
    # Point on the sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Tangent vectors (perturbing theta and phi)
    d_theta = 0.01
    d_phi = 0.01
    
    x_theta = np.sin(theta + d_theta) * np.cos(phi) - x
    y_theta = np.sin(theta + d_theta) * np.sin(phi) - y
    z_theta = np.cos(theta + d_theta) - z
    
    x_phi = np.sin(theta) * np.cos(phi + d_phi) - x
    y_phi = np.sin(theta) * np.sin(phi + d_phi) - y
    z_phi = z  # No change in z for small phi variation
    
    v1 = np.array([x_theta, y_theta, z_theta])
    v2 = np.array([x_phi, y_phi, z_phi])
    
    return np.array([x, y, z]), v1, v2

def plot_inner_product_after_projection():
    """Plot inner product before and after stereographic projection."""
    theta_values = np.linspace(0.1, np.pi - 0.1, 50)
    inner_products_before = []
    inner_products_after = []
    
    for theta in theta_values:
        point, v1, v2 = generate_vectors_at_point(theta, np.pi / 4)
        inner_products_before.append(compute_inner_product(v1, v2))
        
        # Apply stereographic projection
        x_proj, y_proj = stereographic_projection(point[0], point[1], point[2])
        v1_proj_x, v1_proj_y = stereographic_projection(point[0] + v1[0], point[1] + v1[1], point[2] + v1[2])
        v2_proj_x, v2_proj_y = stereographic_projection(point[0] + v2[0], point[1] + v2[1], point[2] + v2[2])
        
        v1_proj = np.array([v1_proj_x - x_proj, v1_proj_y - y_proj])
        v2_proj = np.array([v2_proj_x - x_proj, v2_proj_y - y_proj])
        
        inner_products_after.append(compute_inner_product(v1_proj, v2_proj))
    
    # Plot the inner products
    plt.figure(figsize=(8,6))
    plt.plot(theta_values, inner_products_before, label="Before Projection", linestyle='--')
    plt.plot(theta_values, inner_products_after, label="After Projection", linestyle='-')
    plt.xlabel("Theta (radians)")
    plt.ylabel("Inner Product")
    plt.title("Inner Product Before and After Stereographic Projection")
    plt.legend()
    plt.grid()
    plt.savefig("2d_inner_product_projection.png")
    plt.show()

# Plot the inner product comparison
plot_inner_product_after_projection()
