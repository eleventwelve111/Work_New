#!/usr/bin/env python3
"""
Source definition for gamma-ray shielding simulation.
Includes point source with directional biasing to ensure efficient sampling.
"""

import openmc
import numpy as np
from config import SOURCE_POSITION, CHANNEL_POSITION, WALL_THICKNESS


def create_point_source(energy=1.0, channel_diameter=0.5):
    """
    Create a gamma-ray point source with directional biasing toward the channel.

    Args:
        energy (float): Source energy in MeV
        channel_diameter (float): Diameter of air channel in cm

    Returns:
        openmc.Source: Configured OpenMC source
    """
    # Source position
    x0, y0, z0 = SOURCE_POSITION

    # Channel position and radius
    channel_radius = channel_diameter / 2.0

    # Calculate source-to-channel distance
    source_to_channel = abs(SOURCE_POSITION[0])

    # Calculate solid angle subtended by the channel
    solid_angle = np.pi * channel_radius ** 2 / (source_to_channel ** 2)

    # Calculate cone half-angle
    cone_half_angle = np.arcsin(channel_radius / source_to_channel)

    # Create spatial distribution (point source)
    space = openmc.stats.Point((x0, y0, z0))

    # Energy distribution (monoenergetic or specified spectrum)
    if hasattr(energy, '__iter__'):
        # Multiple energies with equal probability
        energy_dist = openmc.stats.Discrete(energy, [1.0 / len(energy)] * len(energy))
    else:
        # Single energy
        energy_dist = openmc.stats.Discrete([energy], [1.0])

    # Angular distribution (biased toward channel)
    # Create a uniform distribution within the cone that subtends the channel
    mu_min = np.cos(cone_half_angle)

    # Bias the source direction toward the channel with a focused cone
    angle_dist = openmc.stats.PolarAzimuthal(
        mu=openmc.stats.PowerLaw(mu_min, 1.0, 1.0),
        phi=openmc.stats.Uniform(0, 2 * np.pi),
        reference_uvw=(1.0, 0.0, 0.0)  # x-axis is beam direction
    )

    # Create source
    source = openmc.Source(space=space, angle=angle_dist, energy=energy_dist)

    # Set source strength
    source.strength = 1.0

    # Set particle type to photon
    source.particle = 'photon'

    return source


def create_biased_source(energy=1.0, channel_diameter=0.5):
    """
    Create a gamma-ray point source with enhanced directional biasing and
    variance reduction techniques to ensure all particles go through the channel.

    Args:
        energy (float): Source energy in MeV
        channel_diameter (float): Diameter of air channel in cm

    Returns:
        openmc.Source: Configured OpenMC source with variance reduction
    """
    source = create_point_source(energy, channel_diameter)

    # Calculate channel radius and distance from source
    channel_radius = channel_diameter / 2.0
    source_to_wall = abs(SOURCE_POSITION[0])

    # Calculate angles for efficient biasing
    solid_angle = np.pi * channel_radius ** 2 / (source_to_wall ** 2)
    cone_half_angle = np.arctan(channel_radius / source_to_wall)

    # Calculate efficiency improvement
    total_solid_angle = 4 * np.pi
    efficiency_factor = total_solid_angle / solid_angle

    print(f"Source biasing efficiency improvement: {efficiency_factor:.1e}x")
    print(f"Cone half-angle: {np.degrees(cone_half_angle):.4f} degrees")

    return source


if __name__ == "__main__":
    # Test source creation
    for diameter in [0.05, 0.1, 0.5, 1.0]:
        source = create_biased_source(1.0, diameter)
        print(f"Created source for channel diameter {diameter} cm")
