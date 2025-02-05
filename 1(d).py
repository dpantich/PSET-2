import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_surface(x_range, y_range, resolution, func):
    """Generate a mesh grid and compute z = f(x,y)."""
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    return X, Y, Z

def compute_local_frame(X, Y, Z):
    """Compute local coordinate system: tangent and normal vectors."""
    dx, dy = X[0,1] - X[0,0], Y[1,0] - Y[0,0]  # Grid spacing

    # Compute gradient (tangent vectors)
    dZ_dx, dZ_dy = np.gradient(Z, dx, dy, axis=(1, 0))
    
    # Create tangent vectors
    T1 = np.stack([np.ones_like(dZ_dx), np.zeros_like(dZ_dx), dZ_dx], axis=-1)
    T2 = np.stack([np.zeros_like(dZ_dy), np.ones_like(dZ_dy), dZ_dy], axis=-1)

    # Compute normal using cross product
    Normal = np.cross(T1, T2)
    
    # Normalize vectors
    T1 /= np.linalg.norm(T1, axis=-1, keepdims=True)
    T2 /= np.linalg.norm(T2, axis=-1, keepdims=True)
    Normal /= np.linalg.norm(Normal, axis=-1, keepdims=True)
    
    return T1, T2, Normal

def plot_surface_with_frame(X, Y, Z, T1, T2, Normal, step=5):
    """Plot the surface and the local coordinate frames."""
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.5)
    
    # Select points for quiver plot (skip some for clarity)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Zs = Z[::step, ::step]
    T1s = T1[::step, ::step]
    T2s = T2[::step, ::step]
    Normals = Normal[::step, ::step]
    
    # Plot tangent vectors
    ax.quiver(Xs, Ys, Zs, T1s[:,:,0], T1s[:,:,1], T1s[:,:,2], color='r', length=0.2, normalize=True, label="T1")
    ax.quiver(Xs, Ys, Zs, T2s[:,:,0], T2s[:,:,1], T2s[:,:,2], color='g', length=0.2, normalize=True, label="T2")

    # Plot normal vectors
    ax.quiver(Xs, Ys, Zs, Normals[:,:,0], Normals[:,:,1], Normals[:,:,2], color='b', length=0.2, normalize=True, label="Normal")

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Surface with Local Coordinate Frames")
    ax.legend()
    
    # Save plot
    plt.savefig("surface_local_frames.png")
    plt.show()

# Define the surface function
def surface_function(x, y):
    return np.sin(x) * np.cos(y)

# Generate the surface mesh
X, Y, Z = generate_surface((-2, 2), (-2, 2), 30, surface_function)

# Compute the local coordinate frame
T1, T2, Normal = compute_local_frame(X, Y, Z)

# Plot the surface with vectors
plot_surface_with_frame(X, Y, Z, T1, T2, Normal)
