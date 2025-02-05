import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stereographic_projection(x, y, z):
    """Compute stereographic projection of (x, y, z) onto the xy-plane"""
    x_proj = x / (1 - z)
    y_proj = y / (1 - z)
    return x_proj, y_proj

def generate_parallel_transport_trajectory(theta_start, phi_start, steps=100):
    """Generate a parallel transport trajectory along a closed loop on the unit sphere."""
    phi_vals = np.linspace(phi_start, phi_start + 2 * np.pi, steps)
    theta_vals = np.full_like(phi_vals, theta_start)
    
    x = np.sin(theta_vals) * np.cos(phi_vals)
    y = np.sin(theta_vals) * np.sin(phi_vals)
    z = np.cos(theta_vals)
    
    return x, y, z

def plot_parallel_transport_stereographic():
    """Plot parallel transport trajectories under stereographic projection."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Generate sphere mesh
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Generate parallel transport trajectories
    theta_values = [np.pi / 6, np.pi / 4, np.pi / 3]  # Different initial angles
    colors = ['r', 'g', 'b']
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, color='gray', alpha=0.3, edgecolor='gray')
    ax1.set_title("Parallel Transport on Sphere")
    
    ax2 = axs[1]
    ax2.set_title("Stereographic Projection of Parallel Transport")
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    
    for theta, color in zip(theta_values, colors):
        x, y, z = generate_parallel_transport_trajectory(theta, 0)
        x_proj, y_proj = stereographic_projection(x, y, z)
        
        ax1.plot(x, y, z, color=color, label=f"Theta={theta:.2f}")
        ax2.plot(x_proj, y_proj, color=color, label=f"Theta={theta:.2f}")
    
    ax1.legend()
    ax2.legend()
    
    # Save the figure
    plt.savefig("2c_parallel_transport_stereographic.png")
    plt.show()

# Plot the parallel transport trajectories
plot_parallel_transport_stereographic()
