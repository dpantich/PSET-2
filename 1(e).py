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

def parallel_transport(alpha, beta, theta_start, theta_end, steps=20):
    """Perform parallel transport of vector along θ from θ_start to θ_end"""
    theta_vals = np.linspace(theta_start, theta_end, steps)
    phi = 0  # The vector remains at φ=0

    transported_vectors = []
    positions = []

    # Initial basis at θ0
    _, e_theta0, e_phi0 = spherical_basis_vectors(theta_start, phi)

    # Initial transported vector n(θ0, φ=0)
    n_theta = alpha
    n_phi = beta * np.sin(theta_start)
    
    for theta in theta_vals:
        _, e_theta, e_phi = spherical_basis_vectors(theta, phi)

        # Parallel transport equation: n_φ(θ) ∝ 1/sin(θ)
        if np.sin(theta) != 0:
            n_phi_new = n_phi * (np.sin(theta_start) / np.sin(theta))
        else:
            n_phi_new = 0  # Avoid division by zero at the poles

        # Reconstruct the transported vector in Cartesian coordinates
        transported_vector = n_theta * e_theta + n_phi_new * e_phi
        transported_vectors.append(transported_vector)
        positions.append(spherical_to_cartesian(1, theta, phi))

    return np.array(positions), np.array(transported_vectors)

def plot_parallel_transport(positions, transported_vectors):
    """Plot the unit sphere and the transported vector along the path"""
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
    ax.set_title("Parallel Transport on the Unit Sphere")

    # Save the figure
    plt.savefig("parallel_transport_sphere_1(e).png")
    plt.show()

# Set initial conditions
theta_start = np.pi / 5
theta_end = np.pi / 2
alpha = 1.0  # Coefficient for e_theta
beta = 1.0   # Coefficient for e_phi

# Compute the parallel transport
positions, transported_vectors = parallel_transport(alpha, beta, theta_start, theta_end)

# Plot the result
plot_parallel_transport(positions, transported_vectors)
