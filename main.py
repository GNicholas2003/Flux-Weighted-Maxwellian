# Task 1

import numpy as np

# Boltzmann constant
kB = 1.0


def thermal_reflection(n, T):
    """
    Generate a reflected velocity according to the
    flux-weighted Maxwellian distribution.

    Parameters
    ----------
    n : array-like
        Inward-pointing unit surface normal vector.
    T : float
        Temperature of the wall.

    Returns
    -------
    v_reflected : numpy array
        Random reflected velocity vector.
    """

    # Convert the normal vector into a NumPy array
    n = np.asarray(n, dtype=float)

    # Check that the normal vector is a non-zero vector
    norm_n = np.linalg.norm(n)
    if norm_n == 0:
        raise ValueError("The normal vector cannot be zero.")

    # Normalize the surface normal
    n = n / norm_n

    # Check that temperature is positive
    if T <= 0:
        raise ValueError("Temperature must be positive.")

    # Generate the normal velocity component
    # It follows a Rayleigh distribution
    vn = np.random.rayleigh(scale=np.sqrt(kB * T))

    # Generate the tangential velocity component
    # It follows a Gaussian distribution
    vt = np.random.normal(loc=0, scale=np.sqrt(kB * T))

    # Construct a unit tangential vector perpendicular to n
    t = np.array([-n[1], n[0]])

    # Combine normal and tangential components
    v_reflected = vn * n + vt * t

    return v_reflected


def specular_reflection(v, n):
    """
    Calculate the velocity after a specular reflection.

    Parameters
    ----------
    v : array-like
        Incoming velocity vector.
    n : array-like
        Unit surface normal vector.

    Returns
    -------
    v_reflected : numpy array
        Reflected velocity vector.
    """

    # Convert inputs into NumPy arrays
    v = np.asarray(v, dtype=float)
    n = np.asarray(n, dtype=float)

    # Normalize the surface normal
    norm_n = np.linalg.norm(n)
    if norm_n == 0:
        raise ValueError("The normal vector cannot be zero.")

    n = n / norm_n

    # Apply the specular reflection formula
    v_reflected = v - 2 * np.dot(v, n) * n

    return v_reflected

"""
Test 1:

# Test parameters
T = 1.0
N = 1_000_000

# Choose an arbitrary surface normal
n = np.array([1.0, 0.0])

# Generate N reflected velocities
velocities = np.array([
    thermal_reflection(n, T)
    for _ in range(N)
])

# Calculate the speed of every particle
speeds = np.linalg.norm(velocities, axis=1)

# Calculate the mean speed
mean_speed = np.mean(speeds)

# Calculate the mean squared speed
mean_squared_speed = np.mean(speeds**2)

# Theoretical predictions
theoretical_mean_speed = np.sqrt(8 * kB * T / np.pi)
theoretical_mean_squared_speed = 3 * kB * T

# Print the results
print("THERMAL REFLECTION TEST")
print("-----------------------")

print("Mean speed:")
print("Simulation:", mean_speed)
print("Theory:    ", theoretical_mean_speed)

print("\nMean squared speed:")
print("Simulation:", mean_squared_speed)
print("Theory:    ", theoretical_mean_squared_speed)
"""

"""
Test 2:

# Incoming velocity
v = np.array([3.0, -2.0])

# Surface normal (unit vector)
n = np.array([0.0, 1.0])

# Perform specular reflection
v_reflected = specular_reflection(v, n)

# Calculate speeds before and after collision
speed_before = np.linalg.norm(v)
speed_after = np.linalg.norm(v_reflected)

print("SPECULAR REFLECTION TEST")
print("------------------------")

print("Incoming velocity: ", v)
print("Reflected velocity:", v_reflected)

print("\nSpeed before:", speed_before)
print("Speed after: ", speed_after)
"""

"""
Test 3:

# Define an arbitrary surface normal
n = np.array([1.0, 1.0])
n = n / np.linalg.norm(n)

# Generate one thermal velocity
v = thermal_reflection(n, T=1.0)

# Calculate normal and tangential components
vn = np.dot(v, n)

t = np.array([-n[1], n[0]])
vt = np.dot(v, t)

print("GENERAL ORIENTATION TEST")
print("------------------------")

print("Surface normal:", n)
print("Generated velocity:", v)

print("Normal component:", vn)
print("Tangential component:", vt)

# Check that the particle moves away from the wall
assert vn > 0

print("\nTest passed: normal velocity is positive!")
"""



# Task 2


# Task 3


# Task 4