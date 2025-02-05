import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def spherical_basis_to_cartesian(theta, phi):
    e_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_r, e_theta, e_phi

def plot_sphere_with_basis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.3)
    
    # Choose a point on the sphere
    theta, phi = np.pi / 4, np.pi / 3
    point = spherical_to_cartesian(1, theta, phi)
    ax.scatter(*point, color='r', s=100, label='Point on Sphere')
    
    # Plot the local basis vectors
    e_r, e_theta, e_phi = spherical_basis_to_cartesian(theta, phi)
    ax.quiver(*point, *e_r, color='r', length=0.2, normalize=True, label='e_r')
    ax.quiver(*point, *e_theta, color='g', length=0.2, normalize=True, label='e_theta')
    ax.quiver(*point, *e_phi, color='b', length=0.2, normalize=True, label='e_phi')
    
    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Save the plot instead of displaying it
    plt.savefig('PSET-2_Plot1.png')

if __name__ == "__main__":
    plot_sphere_with_basis()
