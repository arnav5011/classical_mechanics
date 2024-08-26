import numpy as np

def distance(position):
    return np.linalg.norm(position, axis=1)  # Calculate distance from the origin

def gravitational_force(M1, M2, r1, r2):
    """Calculate the gravitational force exerted on M1 by M2."""
    G = 6.67 * 10**(-11)  # Gravitational constant
    r = r2 - r1  # Vector from M1 to M2
    r_magnitude = np.linalg.norm(r)
    force = G * M1 * M2 / r_magnitude**3 * r  # Force vector
    return force

def three_body_simulation(masses, positions0, velocities0, time_max, dt):
    """
    Simulate the three-body problem using a jerk-based integrator.

    Parameters:
    masses (list of float): Masses of the three bodies [M1, M2, M3].
    positions0 (numpy.ndarray): Initial positions as a 2D array [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]].
    velocities0 (numpy.ndarray): Initial velocities as a 2D array [[vx1, vy1, vz1], [vx2, vy2, vz2], [vx3, vy3, vz3]].
    time_max (float): The end time for the simulation.
    dt (float): The time step size.

    Returns:
    numpy.ndarray: 3D array of position vectors for each body over time.
    numpy.ndarray: 3D array of velocity vectors for each body over time.
    """

    time_array = np.arange(0, time_max, dt)  # Creating time axis
    num_bodies = len(masses)

    # Initialize position and velocity arrays
    positions = np.array(positions0)
    velocities = np.array(velocities0)

    # Arrays to store position and velocity history
    position_history = [positions]
    velocity_history = [velocities]

    # Compute initial accelerations
    accelerations = np.zeros_like(positions)
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                accelerations[i] += gravitational_force(masses[i], masses[j], positions[i], positions[j]) / masses[i]

    # Initial position and velocity using Verlet integrator
    new_positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    new_velocities = velocities + accelerations * dt

    position_history.append(new_positions)
    velocity_history.append(new_velocities)

    # Verlet Integration with Jerk
    for t in range(2, len(time_array)):
        old_positions = positions
        old_velocities = velocities
        positions = new_positions
        velocities = new_velocities

        # Compute new accelerations and jerks
        new_accelerations = np.zeros_like(positions)
        jerks = np.zeros_like(positions)
        for i in range(num_bodies):
            for j in range(num_bodies):
                if i != j:
                    new_accelerations[i] += gravitational_force(masses[i], masses[j], positions[i], positions[j]) / masses[i]
                    a_diff = new_accelerations[i] - accelerations[i]
                    v_diff = velocities[i] - old_velocities[i]
                    if np.linalg.norm(v_diff) != 0:
                        jerks[i] = a_diff / dt

        # Update positions and velocities
        new_positions = positions + velocities * dt + 0.5 * new_accelerations * dt**2 + (1/6) * jerks * dt**3
        new_velocities = velocities + new_accelerations * dt + 0.5 * jerks * dt**2

        # Update history
        position_history.append(new_positions)
        velocity_history.append(new_velocities)

        # Update accelerations for the next step
        accelerations = new_accelerations

    return np.array(position_history), np.array(velocity_history)

# Example usage:
masses = [1.0e30, 1.0e30, 1.0e30]  # Masses of the three bodies (e.g., stars)
positions0 = np.array([[1.0e11, 0.0, 0.0], [-1.0e11, 0.0, 0.0], [0.0, 1.0e11, 0.0]])  # Initial positions
velocities0 = np.array([[0.0, 1.0e4, 0.0], [0.0, -1.0e4, 0.0], [1.0e4, 0.0, 0.0]])  # Initial velocities

time_max = 1.0e8  # End time for the simulation
dt = 1.0e4  # Time step size

positions, velocities = three_body_simulation(masses, positions0, velocities0, time_max, dt)
