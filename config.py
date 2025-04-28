#!/usr/bin/env python3
"""
Configuration settings for gamma radiation shielding simulation through a concrete wall with air channel.
Includes geometry, source, material, and analysis parameters.
"""

import numpy as np
import os
import json

# Paths and filenames
OUTPUT_DIR = "results"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.h5")
RESULTS_FILE = os.path.join(DATA_DIR, "simulation_results.json")

# Ensure directories exist
for directory in [OUTPUT_DIR, PLOT_DIR, DATA_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Wall parameters (all dimensions in cm)
WALL_THICKNESS = 2 * 30.48  # 2 ft in cm
SOURCE_TO_WALL_DISTANCE = 6 * 30.48  # 6 ft in cm
WALL_WIDTH = 200  # cm
WALL_HEIGHT = 200  # cm

# Channel parameters
CHANNEL_DIAMETERS = [0.05, 0.1, 0.5, 1.0]  # diameters in cm
CHANNEL_POSITION = [0, 0, 0]  # centered at origin

# Source parameters
SOURCE_POSITION = [-SOURCE_TO_WALL_DISTANCE, 0, 0]  # Along negative x-axis
SOURCE_ENERGIES = [0.1, 0.5, 1.0, 2.0, 5.0]  # MeV

# ICRU sphere detector parameters
DETECTOR_DIAMETER = 30.0  # cm
DETECTOR_BASE_DISTANCES = [30, 40, 60, 80, 100, 150]  # cm from wall
DETECTOR_ANGLES = [0, 5, 10, 15, 30, 45]  # degrees

# Simulation parameters
NUM_PARTICLES = 1_000_000
BATCHES = 100
INACTIVE_BATCHES = 10
CHECKPOINT_INTERVAL = 10  # batches

# Tally parameters
ENERGY_BINS = np.logspace(-2, 1, 100)  # 0.01 MeV to 10 MeV, 100 bins
ANGLE_BINS = np.linspace(0, 45, 46)  # 1-degree increments

# Mesh parameters
MESH_DIMENSION = [200, 200, 1]  # x, y, z points
MESH_LOWER_LEFT = [-50, -100, -0.5]  # cm
MESH_UPPER_RIGHT = [250, 100, 0.5]  # cm

# Fine mesh for detailed analysis
FINE_MESH_DIMENSION = [300, 300, 1]
FINE_MESH_LOWER_LEFT = [WALL_THICKNESS - 10, -50, -0.5]
FINE_MESH_UPPER_RIGHT = [WALL_THICKNESS + 150, 50, 0.5]

# Weight window parameters
WEIGHT_WINDOW_BOUNDS = [0.1, 1.0, 10.0]  # Lower, survival, upper weights
USE_WEIGHT_WINDOWS = True


# Detector configuration generator
def generate_detector_positions():
    """Generate all detector positions based on distances and angles."""
    positions = []

    for distance in DETECTOR_BASE_DISTANCES:
        for angle in DETECTOR_ANGLES:
            # Convert angle to radians
            angle_rad = np.radians(angle)

            # Calculate position (x is along beam axis, +x is after wall)
            x = WALL_THICKNESS + distance
            y = distance * np.tan(angle_rad)
            z = 0

            positions.append({
                'distance': distance,
                'angle': angle,
                'position': [x, y, z]
            })

    return positions


# Generate detector positions
DETECTOR_POSITIONS = generate_detector_positions()


# Save configuration
def save_config():
    """Save current configuration to JSON file."""
    config_data = {
        "wall_thickness": WALL_THICKNESS,
        "source_distance": SOURCE_TO_WALL_DISTANCE,
        "channel_diameters": CHANNEL_DIAMETERS,
        "source_energies": SOURCE_ENERGIES,
        "detector_positions": DETECTOR_POSITIONS,
        "simulation_particles": NUM_PARTICLES
    }

    with open(os.path.join(DATA_DIR, "config.json"), 'w') as f:
        json.dump(config_data, f, indent=2)


# Running this file directly saves the config
if __name__ == "__main__":
    save_config()
    print("Configuration saved successfully.")
