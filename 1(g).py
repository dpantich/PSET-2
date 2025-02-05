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

def compute_holonomy(theta_values, alpha=1.0, beta=1.0, steps=50):
    """Compute the inner product of the initial and final transported vector for different θ0"""
    inner_products = []
    for theta in theta_values:
        _, transported_vectors = parallel_transport_phi(alpha, beta, theta, 0, 2 * np.pi, steps)
        initial_vector = transported_vectors[0]
        final_vector = transported_vectors[-1]
        inner_product = np.dot(initial_vector, final_vector) / (np.linalg.norm(initial_vector) * np.linalg.norm(final_vector))
        inner_products.append(inner_product)
    
    return np.array(inner_products)

def plot_holonomy(theta_values, inner_products):
    """Plot holonomy strength as a function of θ0"""
    plt.figure(figsize=(8,6))
    plt.plot(theta_values, inner_products, marker='o', linestyle='-')
    plt.xlabel("θ0 (radians)")
    plt.ylabel("Inner Product (Holonomy Strength)")
    plt.title("Holonomy Strength vs. Initial θ0")
    plt.grid(True)
    plt.savefig("holonomy_plot_1(g).png")
    plt.show()

# Compute holonomy for different θ0 values
theta_values = np.linspace(0.1, np.pi/2, 20)  # Avoid exactly 0 and π/2 to prevent singularities
inner_products = compute_holonomy(theta_values)

# Plot the holonomy strength
plot_holonomy(theta_values, inner_products)
