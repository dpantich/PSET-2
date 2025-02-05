import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stereographic_projection(x, y, z):
    """Compute stereographic projection of (x, y, z) onto the xy-plane"""
    x_proj = x / (1 - z)
    y_proj = y / (1 - z)
    return x_proj, y_proj

def generate_great_circles():
    """Generate great circles on the unit sphere."""
    phi = np.linspace(0, 2 * np.pi, 100)
    
    # Great circle passing through (1,0,0)
    x1 = np.cos(phi)
    y1 = np.sin(phi)
    z1 = np.zeros_like(phi)
    
    # Another great circle passing through equator tilted at 45 degrees
    theta = np.linspace(0, np.pi, 100)
    x2 = np.sin(theta) * np.cos(np.pi / 4)
    y2 = np.sin(theta) * np.sin(np.pi / 4)
    z2 = np.cos(theta)
    
    return (x1, y1, z1), (x2, y2, z2)

def plot_great_circles():
    """Plot great circles on the sphere and their stereographic projection."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Generate sphere mesh
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Generate great circles
    (x1, y1, z1), (x2, y2, z2) = generate_great_circles()
    
    # --- First subplot: Great circles on the sphere ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, color='gray', alpha=0.3, edgecolor='gray')
    ax1.plot(x1, y1, z1, color='r', label="Great Circle 1")
    ax1.plot(x2, y2, z2, color='b', label="Great Circle 2")
    ax1.set_title("Great Circles on the Unit Sphere")
    ax1.legend()
    
    # Apply stereographic projection
    x1_proj, y1_proj = stereographic_projection(x1, y1, z1)
    x2_proj, y2_proj = stereographic_projection(x2, y2, z2)
    
    # --- Second subplot: Projection of great circles ---
    ax2 = axs[1]
    ax2.plot(x1_proj, y1_proj, color='r', label="Projected Great Circle 1")
    ax2.plot(x2_proj, y2_proj, color='b', label="Projected Great Circle 2")
    ax2.set_title("Stereographic Projection of Great Circles")
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.legend()
    
    # Save the figure
    plt.savefig("2b_stereographic_great_circles.png")
    plt.show()

# Plot great circles and their projection
plot_great_circles()
