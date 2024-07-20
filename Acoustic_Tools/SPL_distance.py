import numpy as np

def calculate_spl_point_source(p0, k, distance, p_ref=20e-6):
    """
    Calculate the SPL at a given distance for a point source.

    Parameters:
    p0 (float): Initial pressure amplitude.
    k (float): Wave number (2 * pi / wavelength).
    distance (float): Distance from the source.
    p_ref (float): Reference pressure, default is 20e-6 Pa (human hearing threshold).

    Returns:
    float: SPL value at the given distance.
    """
    # Calculate pressure at distance r
    p_r = p0 * np.exp(1j * k * distance) / distance
    # Calculate SPL
    spl_r = 20 * np.log10(np.abs(p_r) / p_ref)
    return spl_r


def calculate_spl_plane_source(p0, k, distance, p_ref=20e-6):
    """
    Calculate the SPL at a given distance for a plane source with frequency considered.

    Parameters:
    p0 (float): Initial pressu re amplitude.
    k (float): Wave number (2 * pi / wavelength).
    distance (float): Distance from the source.
    p_ref (float): Reference pressure, default is 20e-6 Pa (human hearing threshold).

    Returns:
    float: SPL value at the given distance.
    """
    # Calculate pressure at distance r
    p_r = p0 * np.exp(1j * k * distance) / np.sqrt(distance)
    # Calculate SPL
    spl_r = 20 * np.log10(np.abs(p_r) / p_ref)
    return spl_r


# Parameters
p0 = 1  # Initial pressure amplitude in Pascals
frequency = 1000  # Frequency in Hz
wavelength = 343 / frequency  # Speed of sound is 343 m/s
k = 2 * np.pi / wavelength  # Wave number

# Distances
r1 = 1
r2 = 2

# Calculate SPL at both distances
spl_r1_point = calculate_spl_point_source(p0, k, r1)
spl_r2_point = calculate_spl_point_source(p0, k, r2)

spl_r1_plane = calculate_spl_plane_source(p0, k, r1)
spl_r2_plane = calculate_spl_plane_source(p0, k, r2)

# Calculate SPL difference
spl_difference_point = spl_r1_point - spl_r2_point
spl_difference_plane = spl_r1_plane - spl_r2_plane

print(f"SPL at {r1*100:.1f} cm: {spl_r1_point:.2f} dB")
print(f"SPL at {r2*100:.1f} cm: {spl_r2_point:.2f} dB")
print(f"SPL difference between two point sources {r1*100:.1f} cm and {r2*100:.1f} cm: {spl_difference_point:.2f} dB")

print("-------------------------------------")
print(f"SPL at {r1*100:.1f} cm: {spl_r1_plane:.2f} dB")
print(f"SPL at {r2*100:.1f} cm: {spl_r2_plane:.2f} dB")
print(f"SPL difference between two point sources {r1*100:.1f} cm and {r2*100:.1f} cm: {spl_difference_plane:.2f} dB")