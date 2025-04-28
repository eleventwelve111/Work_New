#!/usr/bin/env python3
"""
Weight windows implementation for variance reduction in the shielding simulation.
"""

import openmc
import numpy as np
from config import (
    WEIGHT_WINDOW_BOUNDS, USE_WEIGHT_WINDOWS,
    SOURCE_POSITION, WALL_THICKNESS
)


def create_weight_windows(channel_diameter=0.5):
    """
    Create weight windows for variance reduction.

    Args:
        channel_diameter (float): Channel diameter in cm

    Returns:
        openmc.WeightWindows: Weight windows object for variance reduction
    """
    if not USE_WEIGHT_WINDOWS:
        return None

    # Define weight window lower bounds mesh
    mesh = openmc.RegularMesh()
    mesh.dimension = [20, 20, 1]  # x, y, z dimensions
    mesh.lower_left = [SOURCE_POSITION[0], -50, -0.5]  # cm
    mesh.upper_right = [WALL_THICKNESS + 100, 50, 0.5]  # cm

    # Calculate channel radius
    channel_radius = channel_diameter / 2.0

    # Source to wall distance
    source_to_wall = abs(SOURCE_POSITION[0])

    # Create array for weight window bounds
    # Initialize with high values
    lower_bounds = np.ones(mesh.dimension) * 0.1

    # Calculate spatial importance
    x_centers = np.linspace(mesh.lower_left[0] + (mesh.upper_right[0] - mesh.lower_left[0]) / (2 * mesh.dimension[0]),
                            mesh.upper_right[0] - (mesh.upper_right[0] - mesh.lower_left[0]) / (2 * mesh.dimension[0]),
                            mesh.dimension[0])

    y_centers = np.linspace(mesh.lower_left[1] + (mesh.upper_right[1] - mesh.lower_left[1]) / (2 * mesh.dimension[1]),
                            mesh.upper_right[1] - (mesh.upper_right[1] - mesh.lower_left[1]) / (2 * mesh.dimension[1]),
                            mesh.dimension[1])

    # Define importances based on position relative to channel
    for i, x in enumerate(x_centers):
        for j, y in enumerate(y_centers):
            # Calculate distance from channel center line
            if x < 0:  # Before wall
                # Higher importance toward channel direction
                angle = np.arctan2(y, channel_radius)
                distance = np.sqrt((y) ** 2)
                if distance < 5:  # Near beam center line
                    lower_bounds[i, j, 0] = 0.01
                elif distance < 10:
                    lower_bounds[i, j, 0] = 0.05
                else:
                    lower_bounds[i, j, 0] = 0.1
            elif x < WALL_THICKNESS:  # Inside wall
                # Higher importance in channel
                distance = np.sqrt((y) ** 2)
                if distance < channel_radius:  # Inside channel
                    # Importance increases as we get closer to exit
                    progress = x / WALL_THICKNESS
                    lower_bounds[i, j, 0] = 0.01 * (1 - progress) + 0.001 * progress
                else:
                    # Low importance outside channel in wall
                    lower_bounds[i, j, 0] = 0.5
            else:  # After wall
                # Higher importance near channel exit and along beam line
                distance = np.sqrt((y) ** 2)
                depth = x - WALL_THICKNESS
                if distance < channel_radius + depth / 5:  # Expanding beam
                    lower_bounds[i, j, 0] = 0.001
                elif distance < channel_radius + depth / 2:
                    lower_bounds[i, j, 0] = 0.005
                elif distance < channel_radius + depth:
                    lower_bounds[i, j, 0] = 0.01
                else:
                    lower_bounds[i, j, 0] = 0.05

    # Create weight windows
    weight_windows = openmc.WeightWindows(mesh,
                                          lower_bounds,
                                          upper_bounds=lower_bounds * 5,
                                          energy_bounds=WEIGHT_WINDOW_BOUNDS)

    # Set particle type
    weight_windows.particle_type = 'photon'

    # Set survival weight ratios
    weight_windows.survival_ratio = 0.5

    return weight_windows


def configure_variance_reduction(settings, channel_diameter=0.5):
    """
    Configure variance reduction techniques in the simulation settings.

    Args:
        settings (openmc.Settings): Settings object to modify
        channel_diameter (float): Channel diameter in cm

    Returns:
        openmc.Settings: Modified settings object
    """
    if USE_WEIGHT_WINDOWS:
        # Create weight windows
        weight_windows = create_weight_windows(channel_diameter)

        # Add weight windows to settings
        settings.weight_windows = weight_windows

        # Configure Russian roulette and splitting
        settings.survival_biasing = True

        print("Configured variance reduction with weight windows")
    else:
        # Use simpler survival biasing
        settings.survival_biasing = True
        print("Configured basic variance reduction with survival biasing")

    return settings


if __name__ == "__main__":
    # Test weight window creation
    ww = create_weight_windows(0.5)

    if ww:
        print(f"Created weight windows with dimensions: {ww.mesh.dimension}")

        # Create settings to test
        settings = openmc.Settings()
        settings = configure_variance_reduction(settings)

        # Export settings to XML
        settings.export_to_xml()
        print("Exported settings with weight windows")

