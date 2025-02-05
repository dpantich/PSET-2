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

def parallel_transport_phi(alpha, beta, theta, phi_start, phi_end, steps=50):
    """Perform parallel transport of a vector along φ from φ_start to φ_end at fixed θ"""
    phi_vals = np.linspace(phi_start, phi_end, steps)
    transported_vectors = []
    positions = []
    
    # Initial basis at φ0
    _, e_theta0, e_phi0 = spherical_basis_vectors(theta, phi_start)
    
    # Initial transported vector n(θ, φ0)
    n_theta = alpha
    n_phi = beta
    
    for phi in phi_vals:
        _, e_theta, e_phi = spherical_basis_vectors(theta, phi)
        
        # The parallel transport equation keeps n_θ constant and n_φ constant along φ
        transported_vector = n_theta * e_theta + n_phi * e_phi
        transported_vectors.append(transported_vector)
        positions.append(spherical_to_cartesian(1, theta, phi))
    
    return np.array(positions), np.array(transported_vectors)

def plot_parallel_transport_phi(positions, transported_vectors):
    """Plot the unit sphere and the transported vector along φ path"""
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the unit sphere
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.3)
    
    # Plot transported vectors
    for i in range(len(positions)):
        ax.quiver(*positions[i], *transported_vectors[i], color='r', length=0.2, normalize=True)
    
    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Parallel Transport Along φ on the Unit Sphere")
    
    # Save the figure
    plt.savefig("parallel_transport_phi_1(f).png")
    plt.show()

# Set initial conditions
theta = np.pi / 4  # Fixed θ
theta_start = theta
theta_end = theta
phi_start = 0
phi_end = 2 * np.pi
alpha = 1.0  # Coefficient for e_theta
beta = 1.0   # Coefficient for e_phi

# Compute the parallel transport
positions, transported_vectors = parallel_transport_phi(alpha, beta, theta, phi_start, phi_end)

# Plot the result
plot_parallel_transport_phi(positions, transported_vectors)
