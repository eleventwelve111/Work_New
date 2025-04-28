#!/usr/bin/env python3
"""
Main simulation controller for gamma radiation shielding study.
Handles setup, execution, and checkpointing of OpenMC simulations.
"""

import os
import time
import json
import openmc
import numpy as np
from config import (
    CHECKPOINT_FILE, NUM_PARTICLES, BATCHES, INACTIVE_BATCHES,
    CHECKPOINT_INTERVAL, OUTPUT_DIR, RESULTS_FILE, SOURCE_ENERGIES,
    CHANNEL_DIAMETERS, DETECTOR_POSITIONS
)
from materials import create_material_library
from geometry import create_geometry, add_detector_to_geometry
from source import create_biased_source
from tally import create_all_tallies


def setup_simulation(energy, channel_diameter, detector_position):
    """
    Set up a simulation with specific parameters.

    Args:
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        detector_position (dict): Detector position information

    Returns:
        tuple: (model, tallies) - OpenMC model and tallies objects
    """
    print(f"\nSetting up simulation with:")
    print(f"  - Energy: {energy} MeV")
    print(f"  - Channel diameter: {channel_diameter} cm")
    print(f"  - Detector at distance: {detector_position['distance']} cm, angle: {detector_position['angle']}°")

    # Create materials
    materials_lib, materials_dict = create_material_library()

    # Create geometry with specified channel diameter
    geometry, universe = create_geometry(materials_dict, channel_diameter)

    # Add detector to geometry
    universe = add_detector_to_geometry(universe, materials_dict, detector_position['position'])

    # Export updated geometry
    geometry = openmc.Geometry(universe)
    geometry.export_to_xml()

    # Create source
    source = create_biased_source(energy, channel_diameter)

    # Create tallies
    tallies, detector_tallies = create_all_tallies(DETECTOR_POSITIONS)

    # Set detector cell filters on detector tallies
    # (This would require tracking the detector cell ID from add_detector_to_geometry)
    # For now, we'll assume the detector cell is the last one added
    detector_cell = universe.cells[-1]
    for tally in detector_tallies:
        tally.filters.append(openmc.CellFilter(detector_cell))

    # Export tallies
    tallies.export_to_xml()

    # Create settings
    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.particles = NUM_PARTICLES
    settings.batches = BATCHES
    settings.inactive = INACTIVE_BATCHES
    settings.photon_transport = True
    settings.electron_treatment = 'ttb'  # Thick target bremsstrahlung
    settings.source = source

    # Set up checkpointing
    settings.checkpoint_interval = CHECKPOINT_INTERVAL
    settings.restart_file = CHECKPOINT_FILE if os.path.exists(CHECKPOINT_FILE) else None

    # Export settings
    settings.export_to_xml()

    # Create model
    model = openmc.Model(geometry, materials_lib, settings, tallies)

    return model, tallies


def run_simulation(model, energy, channel_diameter, detector_position):
    """
    Run a single simulation with specific parameters.

    Args:
        model (openmc.Model): OpenMC model to run
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        detector_position (dict): Detector position information

    Returns:
        openmc.StatePoint: Results of the simulation
    """
    # Create simulation result ID
    result_id = f"E{energy}_D{channel_diameter}_Dist{detector_position['distance']}_Ang{detector_position['angle']}"
    print(f"\nRunning simulation {result_id}")

    # Start timer
    start_time = time.time()

    try:
        # Run the simulation
        result = model.run()

        # Record elapsed time
        elapsed_time = time.time() - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds")

        return result

    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        return None


def extract_results(sp, detector_position, energy, channel_diameter):
    """
    Extract results from a statepoint file.

    Args:
        sp (openmc.StatePoint): Statepoint with simulation results
        detector_position (dict): Detector position information
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm

    Returns:
        dict: Dictionary of results
    """
    results = {
        'energy': energy,
        'channel_diameter': channel_diameter,
        'detector_distance': detector_position['distance'],
        'detector_angle': detector_position['angle'],
        'position': detector_position['position'],
        'tallies': {}
    }

    # Extract all tallies
    for tally_id, tally in sp.tallies.items():
        tally_name = tally.name

        # Check if this is a detector tally
        if 'detector' in tally_name:
            # Get mean values and relative errors
            mean = tally.mean.flatten()
            rel_err = tally.std_dev.flatten() / mean if np.any(mean) else np.zeros_like(mean)

            results['tallies'][tally_name] = {
                'scores': tally.scores,
                'mean': mean.tolist(),
                'rel_err': rel_err.tolist()
            }

    return results


def run_parameter_sweep():
    """
    Run simulations for all combinations of parameters.

    Returns:
        dict: Compiled results from all simulations
    """
    all_results = []

    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parameter sweep
    for energy in SOURCE_ENERGIES:
        for channel_diameter in CHANNEL_DIAMETERS:
            for detector_position in DETECTOR_POSITIONS:
                # Setup simulation
                model, tallies = setup_simulation(energy, channel_diameter, detector_position)

                # Run simulation
                sp = run_simulation(model, energy, channel_diameter, detector_position)

                if sp:
                    # Extract results
                    results = extract_results(sp, detector_position, energy, channel_diameter)
                    all_results.append(results)

                    # Save intermediate results
                    with open(RESULTS_FILE, 'w') as f:
                        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    print("Starting parameter sweep simulation")
    results = run_parameter_sweep()
    print(f"Completed simulations and saved results to {RESULTS_FILE}")
