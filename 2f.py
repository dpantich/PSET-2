import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stereographic_projection(x, y, z):
    """Compute stereographic projection of (x, y, z) onto the xy-plane"""
    x_proj = x / (1 - z)
    y_proj = y / (1 - z)
    return x_proj, y_proj

def generate_parallel_transport_loop(theta_start, phi_start, steps=100):
    """Generate a parallel transport trajectory along a closed loop on the unit sphere."""
    phi_vals = np.linspace(phi_start, phi_start + 2 * np.pi, steps)
    theta_vals = np.full_like(phi_vals, theta_start)
    
    x = np.sin(theta_vals) * np.cos(phi_vals)
    y = np.sin(theta_vals) * np.sin(phi_vals)
    z = np.cos(theta_vals)
    
    return x, y, z

def compute_holonomy_on_projection(theta_start, steps=100):
    """Compute the deviation in vector orientation after projection."""
    x, y, z = generate_parallel_transport_loop(theta_start, 0, steps)
    x_proj, y_proj = stereographic_projection(x, y, z)
    
    # Compute the change in orientation over the loop
    initial_vector = np.array([x_proj[0], y_proj[0]])
    final_vector = np.array([x_proj[-1], y_proj[-1]])
    
    # Compute angle deviation
    dot_product = np.dot(initial_vector, final_vector) / (np.linalg.norm(initial_vector) * np.linalg.norm(final_vector))
    angle_deviation = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    return np.degrees(angle_deviation)

def plot_holonomy_after_projection():
    """Plot the holonomy deviation after stereographic projection."""
    theta_values = np.linspace(0.1, np.pi / 2, 20)  # Avoid exactly 0 and π/2
    holonomy_deviations = [compute_holonomy_on_projection(theta) for theta in theta_values]
    
    plt.figure(figsize=(8,6))
    plt.plot(theta_values, holonomy_deviations, marker='o', linestyle='-')
    plt.xlabel("Theta (radians)")
    plt.ylabel("Holonomy Deviation (degrees)")
    plt.title("Effect of Stereographic Projection on Holonomy")
    plt.grid()
    plt.savefig("2f_holonomy_projection.png")
    plt.show()

# Plot holonomy deviation due to stereographic projection
plot_holonomy_after_projection()
