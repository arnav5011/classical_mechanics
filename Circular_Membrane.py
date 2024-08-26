import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy import special as sc
import math

def bessel(r, m, tol=1e-10):
    term = (r / 2) ** m / math.gamma(m + 1)
    sum_ = term
    k = 0
    
    while abs(term) > tol:  # Series expansion of Bessel Function
        k += 1
        term *= - (r**2 / 4) / (k * (m + k))
        sum_ += term
    
    return sum_

A = 1
B = 0
C = 1 
D = 0
m = 1
n = 1
a = 1
c = 1

# Compute the first root of the Bessel function J1 for the fundamental mode
alpha_mn = sc.jn_zeros(m, n)[0] / a  # First zero for m=1 and n=1

# Time and space arrays
t_max = 10
time = np.linspace(0, t_max, int(t_max/0.1))
radial = np.linspace(0, a, int(a/0.01))
angular = np.linspace(0, 2*np.pi, int(2*np.pi/0.01))

# Initialize the displacement matrix
u = np.zeros((len(radial), len(angular), len(time)))

# Compute displacement for each time, radial, and angular point
for i, t in enumerate(time):
    for j, r in enumerate(radial):
        for k, theta in enumerate(angular):
            term1 = A*np.cos(c*alpha_mn*t) + B*np.sin(c*alpha_mn*t)
            term2 = bessel(r, m)
            term3 = C*np.cos(m*theta) + D*np.sin(m*theta)
            u[j, k, i] = term1 * term2 * term3

# Create meshgrid for plotting
r_mesh, theta_mesh = np.meshgrid(radial, angular)
x_mesh = r_mesh * np.cos(theta_mesh)
y_mesh = r_mesh * np.sin(theta_mesh)

# Initialize the figure and 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
z_mesh = u[:, :, 0].T
surface = ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Displacement')
ax.set_zlim(-2, 2)  # Adjust z-limits for better visualization

# Function to update the surface for each frame
def update(frame):
    ax.clear()
    z_mesh = u[:, :, frame].T
    surface = ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Displacement')
    ax.set_zlim(-2, 2)  # Maintain consistent z-limits
    return surface,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(time), interval=50, blit=False)

# Show the animation
plt.show()
