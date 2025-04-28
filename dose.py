#!/usr/bin/env python3
"""
Dose calculation and analysis module for radiation shielding simulation.
Includes conversion from tallies to dose rates and dose analysis functions.
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from config import (
    DATA_DIR, PLOT_DIR, DETECTOR_POSITIONS, SOURCE_ENERGIES, CHANNEL_DIAMETERS
)


def calculate_dose_rates(tally_data, conversion_type='flux_to_dose'):
    """
    Calculate dose rates from tally data using various conversion methods.

    Args:
        tally_data (dict): Tally data from simulation results
        conversion_type (str): Type of conversion to use

    Returns:
        float: Dose rate in rem/hr
    """
    # Extract mean values for different scoring methods
    if conversion_type == 'flux_to_dose':
        # This is already converted using flux-to-dose factors
        mean = tally_data['mean'][0]  # First score is 'flux' with dose multiplier
        return mean

    elif conversion_type == 'heating':
        # Convert heating (energy deposition) to dose
        # Heating is in eV/g, need to convert to rad or Gy and then to rem
        # Assume Q factor of 1 for photons
        for i, score in enumerate(tally_data['scores']):
            if score == 'heating' or score == 'heating-photon':
                heating_mean = tally_data['mean'][i]
                # Convert eV/g to rad (1 eV/g = 1.602e-8 rad)
                rad = heating_mean * 1.602e-8
                # Convert rad to rem (rem = rad * Q, Q=1 for photons)
                rem = rad * 1.0
                # Convert to rem/hr assuming tally is normalized per source particle
                # and source strength is 1 particle/sec
                rem_hr = rem * 3600  # rem/hr
                return rem_hr

    elif conversion_type == 'kerma':
        # Convert kerma to dose
        for i, score in enumerate(tally_data['scores']):
            if score == 'kerma-photon':
                kerma_mean = tally_data['mean'][i]
                # Kerma is in eV/g, convert to rad and then rem
                rad = kerma_mean * 1.602e-8
                rem = rad * 1.0  # Q=1 for photons
                rem_hr = rem * 3600  # rem/hr
                return rem_hr

    # Default return if no conversion method matched
    return 0.0


def load_results(filename=None):
    """
    Load simulation results from JSON file.

    Args:
        filename (str, optional): Path to results file

    Returns:
        list: Simulation results
    """
    if filename is None:
        filename = os.path.join(DATA_DIR, 'simulation_results.json')

    with open(filename, 'r') as f:
        results = json.load(f)

    return results


def analyze_dose_vs_angle(results, energy, channel_diameter, distance,
                          conversion_type='flux_to_dose'):
    """
    Analyze dose variation with detector angle at a fixed distance.

    Args:
        results (list): Simulation results
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        distance (float): Detector distance from wall in cm
        conversion_type (str): Type of dose conversion to use

    Returns:
        tuple: (angles, doses, rel_errors) - angles and corresponding doses
    """
    angles = []
    doses = []
    rel_errors = []

    # Filter results for the specified parameters
    for result in results:
        if (result['energy'] == energy and
                result['channel_diameter'] == channel_diameter and
                result['detector_distance'] == distance):

            # Find detector tally
            for tally_name, tally_data in result['tallies'].items():
                if 'detector' in tally_name:
                    # Calculate dose using specified conversion
                    dose = calculate_dose_rates(tally_data, conversion_type)

                    # If dose calculation was successful, add to lists
                    if dose > 0:
                        angles.append(result['detector_angle'])
                        doses.append(dose)

                        # Extract relative error for this tally
                        if conversion_type == 'flux_to_dose':
                            rel_error = tally_data['rel_err'][0]
                        elif conversion_type == 'heating':
                            idx = tally_data['scores'].index('heating')
                            rel_error = tally_data['rel_err'][idx]
                        elif conversion_type == 'kerma':
                            idx = tally_data['scores'].index('kerma-photon')
                            rel_error = tally_data['rel_err'][idx]
                        else:
                            rel_error = 0.0

                        rel_errors.append(rel_error)

    # Sort by angle
    sorted_idx = np.argsort(angles)
    angles = [angles[i] for i in sorted_idx]
    doses = [doses[i] for i in sorted_idx]
    rel_errors = [rel_errors[i] for i in sorted_idx]

    return angles, doses, rel_errors


def plot_dose_vs_angle(results, energy, distance, conversion_type='flux_to_dose'):
    """
    Plot dose vs angle for different channel diameters at a fixed energy and distance.

    Args:
        results (list): Simulation results
        energy (float): Source energy in MeV
        distance (float): Detector distance from wall in cm
        conversion_type (str): Type of dose conversion to use

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for channel_diameter in CHANNEL_DIAMETERS:
        angles, doses, rel_errors = analyze_dose_vs_angle(
            results, energy, channel_diameter, distance, conversion_type
        )

        # Skip if no data
        if not angles:
            continue

        # Calculate error bars
        yerr = np.array(doses) * np.array(rel_errors)

        # Plot with error bars
        label = f"Diameter = {channel_diameter} cm"
        ax.errorbar(angles, doses, yerr=yerr, marker='o', linestyle='-', label=label)

    # Set log scale for y-axis
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('Detector Angle (degrees)')
    ax.set_ylabel(f'Dose Rate (rem/hr) - {conversion_type}')
    ax.set_title(f'Dose vs Angle - Energy: {energy} MeV, Distance: {distance} cm')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend
    ax.legend()

    # Save figure
    filename = f"dose_vs_angle_E{energy}_Dist{distance}_{conversion_type}.png"
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


def analyze_dose_vs_distance(results, energy, channel_diameter, angle,
                             conversion_type='flux_to_dose'):
    """
    Analyze dose variation with detector distance at a fixed angle.

    Args:
        results (list): Simulation results
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        angle (float): Detector angle in degrees
        conversion_type (str): Type of dose conversion to use

    Returns:
        tuple: (distances, doses, rel_errors) - distances and corresponding doses
    """
    distances = []
    doses = []
    rel_errors = []

    # Filter results for the specified parameters
    for result in results:
        if (result['energy'] == energy and
                result['channel_diameter'] == channel_diameter and
                result['detector_angle'] == angle):

            # Find detector tally
            for tally_name, tally_data in result['tallies'].items():
                if 'detector' in tally_name:
                    # Calculate dose using specified conversion
                    dose = calculate_dose_rates(tally_data, conversion_type)

                    # If dose calculation was successful, add to lists
                    if dose > 0:
                        distances.append(result['detector_distance'])
                        doses.append(dose)

                        # Extract relative error for this tally
                        if conversion_type == 'flux_to_dose':
                            rel_error = tally_data['rel_err'][0]
                        elif conversion_type == 'heating':
                            idx = tally_data['scores'].index('heating')
                            rel_error = tally_data['rel_err'][idx]
                        elif conversion_type == 'kerma':
                            idx = tally_data['scores'].index('kerma-photon')
                            rel_error = tally_data['rel_err'][idx]
                        else:
                            rel_error = 0.0

                        rel_errors.append(rel_error)

    # Sort by distance
    sorted_idx = np.argsort(distances)
    distances = [distances[i] for i in sorted_idx]
    doses = [doses[i] for i in sorted_idx]
    rel_errors = [rel_errors[i] for i in sorted_idx]

    return distances, doses, rel_errors


def plot_dose_vs_distance(results, energy, angle, conversion_type='flux_to_dose'):
    """
    Plot dose vs distance for different channel diameters at a fixed energy and angle.

    Args:
        results (list): Simulation results
        energy (float): Source energy in MeV
        angle (float): Detector angle in degrees
        conversion_type (str): Type of dose conversion to use

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for channel_diameter in CHANNEL_DIAMETERS:
        distances, doses, rel_errors = analyze_dose_vs_distance(
            results, energy, channel_diameter, angle, conversion_type
        )

        # Skip if no data
        if not distances:
            continue

        # Calculate error bars
        yerr = np.array(doses) * np.array(rel_errors)

        # Plot with error bars
        label = f"Diameter = {channel_diameter} cm"
        ax.errorbar(distances, doses, yerr=yerr, marker='o', linestyle='-', label=label)

    # Set log scale for y-axis
    ax.set_yscale('log')

    # Add inverse square law reference line if data exists
    if len(distances) > 1 and len(doses) > 1:
        # Get first point as reference
        d0, dose0 = distances[0], doses[0]

        # Generate inverse square law curve
        ref_distances = np.linspace(min(distances), max(distances), 100)
        ref_doses = dose0 * (d0 / ref_distances) ** 2

        # Plot reference line
        ax.plot(ref_distances, ref_doses, 'k--', alpha=0.5, label='Inverse Square Law')

    # Labels and title
    ax.set_xlabel('Detector Distance (cm)')
    ax.set_ylabel(f'Dose Rate (rem/hr) - {conversion_type}')
    ax.set_title(f'Dose vs Distance - Energy: {energy} MeV, Angle: {angle}°')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend
    ax.legend()

    # Save figure
    filename = f"dose_vs_distance_E{energy}_Ang{angle}_{conversion_type}.png"
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


def compare_dose_methods(results):
    """
    Compare different dose calculation methods for validation.

    Args:
        results (list): Simulation results

    Returns:
        matplotlib.figure.Figure: Comparison figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Select a representative configuration
    energy = SOURCE_ENERGIES[2]  # Middle energy
    channel_diameter = CHANNEL_DIAMETERS[2]  # Middle diameter
    angle = 0  # On-axis

    conversion_types = ['flux_to_dose', 'heating', 'kerma']
    markers = ['o', 's', '^']
    colors = ['blue', 'red', 'green']

    for i, conversion_type in enumerate(conversion_types):
        distances, doses, rel_errors = analyze_dose_vs_distance(
            results, energy, channel_diameter, angle, conversion_type
        )

        # Skip if no data
        if not distances:
            continue

        # Calculate error bars
        yerr = np.array(doses) * np.array(rel_errors)

        # Plot with error bars
        label = f"{conversion_type}"
        ax.errorbar(distances, doses, yerr=yerr, marker=markers[i],
                    linestyle='-', color=colors[i], label=label)

    # Set log scale for y-axis
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('Detector Distance (cm)')
    ax.set_ylabel('Dose Rate (rem/hr)')
    ax.set_title(
        f'Comparison of Dose Calculation Methods\nEnergy: {energy} MeV, Channel Diameter: {channel_diameter} cm, Angle: {angle}°')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend
    ax.legend()

    # Save figure
    filename = f"dose_methods_comparison_E{energy}_D{channel_diameter}.png"
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Load results
    results = load_results()
    print(f"Loaded {len(results)} simulation results")

    # Generate dose vs angle plots for different energies and distances
    for energy in SOURCE_ENERGIES:
        for distance in [30, 60, 100]:
            plot_dose_vs_angle(results, energy, distance)

    # Generate dose vs distance plots for different energies and angles
    for energy in SOURCE_ENERGIES:
        for angle in [0, 15, 30]:
            plot_dose_vs_distance(results, energy, angle)

    # Compare dose calculation methods
    compare_dose_methods(results)

    print("Dose analysis complete and plots saved.")

