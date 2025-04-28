#!/usr/bin/env python3
"""
Analysis of radiation energy spectra at different positions in the geometry.
"""

import openmc
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import pandas as pd
from config import (
    PLOT_DIR, DATA_DIR, SOURCE_ENERGIES, CHANNEL_DIAMETERS, WALL_THICKNESS
)


def extract_energy_spectra(sp, tally_id, filters=None):
    """
    Extract energy spectra from a tally.

    Args:
        sp (openmc.StatePoint): StatePoint file containing results
        tally_id (int): ID of the tally containing spectral data
        filters (dict, optional): Filter values to extract specific spectra

    Returns:
        dict: Dictionary with energy bins and spectra
    """
    # Get the tally
    tally = sp.tallies[tally_id]

    # Get energy filter and bins
    energy_filter = next(f for f in tally.filters if isinstance(f, openmc.EnergyFilter))
    energy_bins = energy_filter.values

    # Create midpoints for plotting
    energy_midpoints = 0.5 * (energy_bins[1:] + energy_bins[:-1])

    # Get mean values for the spectrum
    if filters:
        # Filter the data based on the specified filter values
        filter_indices = {}

        for filter_name, filter_value in filters.items():
            for f in tally.filters:
                if f.short_name == filter_name:
                    if isinstance(filter_value, (list, tuple)):
                        # Find indices for multiple values
                        indices = [f.get_bin_index(v) for v in filter_value]
                    else:
                        # Find index for a single value
                        indices = [f.get_bin_index(filter_value)]

                    filter_indices[f] = indices

        # Extract spectrum with filters
        spectrum = tally.get_values(filters=filter_indices)
    else:
        # Just get the overall spectrum
        spectrum = tally.mean

    # Return the energy bins and spectrum
    return {
        'energy_bins': energy_bins,
        'energy_midpoints': energy_midpoints,
        'spectrum': spectrum
    }


def analyze_spectrum_hardening(spectra, positions=None):
    """
    Analyze spectrum hardening as radiation passes through the shield.

    Args:
        spectra (dict): Dictionary with spectra at different positions
        positions (list, optional): List of positions to analyze

    Returns:
        dict: Hardening metrics
    """
    if positions is None or not spectra or not spectra.get(positions[0]):
        return {}

    # Compare spectra at multiple positions
    results = {
        'average_energy': {},
        'hardening_ratio': {},
        'spectral_shift': {}
    }

    # Reference spectrum (at the first position)
    ref_pos = positions[0]
    ref_spectrum = spectra[ref_pos]['spectrum']
    ref_energy_midpoints = spectra[ref_pos]['energy_midpoints']

    # Calculate reference average energy
    ref_avg_energy = np.sum(ref_energy_midpoints * ref_spectrum) / np.sum(ref_spectrum)
    results['average_energy'][ref_pos] = ref_avg_energy

    for pos in positions[1:]:
        if pos not in spectra:
            continue

        # Get spectrum at this position
        curr_spectrum = spectra[pos]['spectrum']
        curr_energy_midpoints = spectra[pos]['energy_midpoints']

        # Calculate average energy
        avg_energy = np.sum(curr_energy_midpoints * curr_spectrum) / np.sum(curr_spectrum)
        results['average_energy'][pos] = avg_energy

        # Calculate hardening ratio (ratio of high to low energy components)
        # Define high and low energy thresholds
        low_e_threshold = 0.1  # MeV
        high_e_threshold = 0.5  # MeV

        # Find indices for thresholds
        low_indices = np.where(curr_energy_midpoints < low_e_threshold)[0]
        high_indices = np.where(curr_energy_midpoints > high_e_threshold)[0]

        # Sum flux in each energy range
        low_e_flux = np.sum(curr_spectrum[low_indices]) if low_indices.size > 0 else 0
        high_e_flux = np.sum(curr_spectrum[high_indices]) if high_indices.size > 0 else 0

        # Calculate hardening ratio
        if low_e_flux > 0:
            hardening_ratio = high_e_flux / low_e_flux
        else:
            hardening_ratio = float('inf')

        results['hardening_ratio'][pos] = hardening_ratio

        # Calculate spectral shift (change in average energy)
        spectral_shift = avg_energy / ref_avg_energy
        results['spectral_shift'][pos] = spectral_shift

    return results


def plot_spectrum_comparison(spectra, positions, title=None, filename=None):
    """
    Plot multiple energy spectra for comparison.

    Args:
        spectra (dict): Dictionary with spectra at different positions
        positions (list): List of positions to compare
        title (str, optional): Plot title
        filename (str, optional): Filename for the output plot

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each spectrum
    for pos in positions:
        if pos in spectra:
            # Get spectrum data
            energy_midpoints = spectra[pos]['energy_midpoints']
            spectrum = spectra[pos]['spectrum']

            # Plot spectrum
            if isinstance(pos, (list, tuple)):
                # Position is a tuple, format it nicely
                if len(pos) == 3:
                    label = f"Position ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
                else:
                    label = f"Position {pos}"
            else:
                label = f"Position {pos}"

            ax.plot(energy_midpoints, spectrum, label=label)

    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis labels
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Flux per Unit Energy (particles/cm²/src/MeV)')

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Energy Spectrum Comparison')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend
    ax.legend()

    # Save figure if filename provided
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


def calculate_attenuation_coefficients(spectra, positions, source_energy, channel_diameter):
    """
    Calculate effective attenuation coefficients at different positions.

    Args:
        spectra (dict): Dictionary with spectra at different positions
        positions (list): List of positions to analyze
        source_energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm

    Returns:
        dict: Attenuation coefficients and related metrics
    """
    results = {
        'total_flux': {},
        'attenuation': {},
        'effective_mu': {}
    }

    # Reference position (first position, typically before wall)
    ref_pos = positions[0]
    if ref_pos not in spectra:
        return results

    # Calculate total flux at reference position
    ref_spectrum = spectra[ref_pos]['spectrum']
    ref_energy_midpoints = spectra[ref_pos]['energy_midpoints']
    ref_energy_bins = spectra[ref_pos]['energy_bins']

    # Total flux is integral of spectrum over energy
    bin_widths = np.diff(ref_energy_bins)
    ref_total_flux = np.sum(ref_spectrum * bin_widths)
    results['total_flux'][ref_pos] = ref_total_flux

    # Analyze positions after the reference
    for pos in positions[1:]:
        if pos not in spectra:
            continue

        # Calculate total flux at current position
        curr_spectrum = spectra[pos]['spectrum']
        curr_energy_bins = spectra[pos]['energy_bins']

        # Total flux is integral of spectrum over energy
        bin_widths = np.diff(curr_energy_bins)
        curr_total_flux = np.sum(curr_spectrum * bin_widths)
        results['total_flux'][pos] = curr_total_flux

        # Calculate attenuation
        if ref_total_flux > 0:
            attenuation = curr_total_flux / ref_total_flux
        else:
            attenuation = 0

        results['attenuation'][pos] = attenuation

        # Calculate effective attenuation coefficient (μ)
        # For positions after the wall, account for distance
        if isinstance(pos, (list, tuple)) and len(pos) >= 1:
            # Assume x-coordinate is depth
            depth = pos[0] - ref_pos[0] if isinstance(ref_pos, (list, tuple)) else pos[0]

            if depth > 0 and attenuation > 0:
                # μ = -ln(I/I₀)/x where I/I₀ is attenuation and x is depth
                mu = -np.log(attenuation) / depth
            else:
                mu = 0

            results['effective_mu'][pos] = mu

    return results


def plot_attenuation_curve(attenuation_data, source_energy, channel_diameter, filename=None):
    """
    Plot attenuation curve as a function of depth.

    Args:
        attenuation_data (dict): Attenuation data from calculate_attenuation_coefficients
        source_energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        filename (str, optional): Filename for the output plot

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract data
    positions = []
    attenuation_values = []

    for pos, attenuation in attenuation_data['attenuation'].items():
        if isinstance(pos, (list, tuple)) and len(pos) >= 1:
            # Use x-coordinate as depth
            depth = pos[0]
            positions.append(depth)
            attenuation_values.append(attenuation)

    # Sort by position
    sorted_data = sorted(zip(positions, attenuation_values))
    positions = [d[0] for d in sorted_data]
    attenuation_values = [d[1] for d in sorted_data]

    # Plot attenuation curve
    ax.plot(positions, attenuation_values, 'o-', linewidth=2)

    # Mark wall position
    ax.axvline(x=WALL_THICKNESS, color='red', linestyle='--',
               label=f'Wall Exit ({WALL_THICKNESS} cm)')

    # Set log scale for y-axis
    ax.set_yscale('log')

    # Set axis labels
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Relative Flux (I/I₀)')

    # Set title
    ax.set_title(f'Radiation Attenuation - Energy: {source_energy} MeV, Channel Ø{channel_diameter} cm')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend
    ax.legend()

    # Save figure if filename provided
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


def plot_hardening_metrics(hardening_data, source_energy, channel_diameter, filename=None):
    """
    Plot spectrum hardening metrics as a function of depth.

    Args:
        hardening_data (dict): Hardening data from analyze_spectrum_hardening
        source_energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        filename (str, optional): Filename for the output plot

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Create second axis for another metric
    ax2 = ax1.twinx()

    # Extract data
    positions = []
    avg_energies = []
    hardening_ratios = []

    for pos in hardening_data['average_energy'].keys():
        if isinstance(pos, (list, tuple)) and len(pos) >= 1:
            # Use x-coordinate as depth
            depth = pos[0]
            positions.append(depth)
            avg_energies.append(hardening_data['average_energy'][pos])

            if pos in hardening_data['hardening_ratio']:
                hardening_ratios.append(hardening_data['hardening_ratio'][pos])
            else:
                hardening_ratios.append(0)

    # Sort by position
    sorted_data = sorted(zip(positions, avg_energies, hardening_ratios))
    positions = [d[0] for d in sorted_data]
    avg_energies = [d[1] for d in sorted_data]
    hardening_ratios = [d[2] for d in sorted_data]

    # Plot average energy
    line1 = ax1.plot(positions, avg_energies, 'b-o', linewidth=2, label='Average Energy')
    ax1.set_ylabel('Average Energy (MeV)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot hardening ratio
    line2 = ax2.plot(positions, hardening_ratios, 'r-^', linewidth=2, label='Hardening Ratio')
    ax2.set_ylabel('Hardening Ratio (High/Low E)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Mark wall position
    ax1.axvline(x=WALL_THICKNESS, color='green', linestyle='--',
                label=f'Wall Exit ({WALL_THICKNESS} cm)')

    # Set axis labels
    # Set axis labels
    ax1.set_xlabel('Position (cm)')

    # Set title
    fig.suptitle(f'Spectrum Hardening Metrics - Energy: {source_energy} MeV, Channel Ø{channel_diameter} cm')

    # Add grid
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Save figure if filename provided
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


def analyze_energy_dependent_attenuation(spectra, positions, source_energy, channel_diameter):
    """
    Analyze energy-dependent attenuation in the shield.

    Args:
        spectra (dict): Dictionary with spectra at different positions
        positions (list): List of positions to analyze
        source_energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm

    Returns:
        dict: Energy-dependent attenuation data
    """
    results = {
        'energy_bins': [],
        'attenuation_by_energy': {},
        'effective_mu_by_energy': {}
    }

    # Reference position (first position, typically before wall)
    ref_pos = positions[0]
    if ref_pos not in spectra:
        return results

    # Get reference spectrum
    ref_spectrum = spectra[ref_pos]['spectrum']
    ref_energy_midpoints = spectra[ref_pos]['energy_midpoints']
    results['energy_bins'] = ref_energy_midpoints

    # Analyze positions after the reference
    for pos in positions[1:]:
        if pos not in spectra:
            continue

        # Get current spectrum
        curr_spectrum = spectra[pos]['spectrum']

        # Calculate energy-dependent attenuation (I/I₀)
        with np.errstate(divide='ignore', invalid='ignore'):
            attenuation = np.divide(curr_spectrum, ref_spectrum)

        # Replace NaN and inf values with 0
        attenuation = np.nan_to_num(attenuation, nan=0, posinf=0, neginf=0)

        # Store attenuation
        results['attenuation_by_energy'][pos] = attenuation

        # Calculate effective attenuation coefficient (μ)
        if isinstance(pos, (list, tuple)) and len(pos) >= 1:
            # Assume x-coordinate is depth
            depth = pos[0] - ref_pos[0] if isinstance(ref_pos, (list, tuple)) else pos[0]

            if depth > 0:
                # μ = -ln(I/I₀)/x where I/I₀ is attenuation and x is depth
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu = -np.log(attenuation) / depth

                # Replace NaN and inf values with 0
                mu = np.nan_to_num(mu, nan=0, posinf=0, neginf=0)

                results['effective_mu_by_energy'][pos] = mu

    return results


def plot_energy_dependent_attenuation(energy_attenuation_data, positions, source_energy, channel_diameter,
                                      filename=None):
    """
    Plot energy-dependent attenuation coefficients.

    Args:
        energy_attenuation_data (dict): Energy attenuation data
        positions (list): List of positions to plot
        source_energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        filename (str, optional): Filename for the output plot

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get energy bins
    energy_bins = energy_attenuation_data['energy_bins']

    # Plot attenuation coefficient for each position
    for pos in positions[1:]:  # Skip reference position
        if pos in energy_attenuation_data['effective_mu_by_energy']:
            mu = energy_attenuation_data['effective_mu_by_energy'][pos]

            # Format position for label
            if isinstance(pos, (list, tuple)):
                if len(pos) == 3:
                    label = f"Pos ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
                else:
                    label = f"Pos {pos}"
            else:
                label = f"Pos {pos}"

            # Plot mu vs energy
            ax.plot(energy_bins, mu, label=label)

    # Set log scale for x-axis
    ax.set_xscale('log')

    # Set axis labels
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Effective Attenuation Coefficient (cm⁻¹)')

    # Set title
    ax.set_title(f'Energy-Dependent Attenuation - E:{source_energy} MeV, Channel Ø{channel_diameter} cm')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend
    ax.legend()

    # Save figure if filename provided
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


def export_spectrum_data(spectra, positions, source_energy, channel_diameter):
    """
    Export spectrum data to CSV file.

    Args:
        spectra (dict): Dictionary with spectra at different positions
        positions (list): List of positions to export
        source_energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm

    Returns:
        str: Path to the saved file
    """
    # Create data dictionary
    data = {
        'energy': None
    }

    # Get energy grid from first spectrum
    if positions and positions[0] in spectra:
        data['energy'] = spectra[positions[0]]['energy_midpoints']
    else:
        return None

    # Add spectrum data for each position
    for pos in positions:
        if pos in spectra:
            # Format position for column name
            if isinstance(pos, (list, tuple)):
                if len(pos) == 3:
                    col_name = f"flux_x{pos[0]:.1f}_y{pos[1]:.1f}_z{pos[2]:.1f}"
                else:
                    col_name = f"flux_pos{pos}"
            else:
                col_name = f"flux_pos{pos}"

            data[col_name] = spectra[pos]['spectrum']

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create filename
    filename = f"spectrum_data_E{source_energy}_D{channel_diameter}.csv"
    file_path = os.path.join(DATA_DIR, filename)

    # Save to CSV
    df.to_csv(file_path, index=False)

    return file_path


def analyze_spectral_indices(spectra, positions, source_energy, channel_diameter):
    """
    Calculate various spectral indices at different positions.

    Args:
        spectra (dict): Dictionary with spectra at different positions
        positions (list): List of positions to analyze
        source_energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm

    Returns:
        dict: Spectral indices at each position
    """
    results = {
        'thermal_to_fast_ratio': {},
        'epithermal_index': {},
        'average_energy': {},
        'peak_energy': {},
        'FWHM': {}  # Full Width at Half Maximum
    }

    # Energy range definitions
    thermal_max = 0.025  # 0.025 MeV (25 keV)
    epithermal_min = thermal_max
    epithermal_max = 0.5  # 0.5 MeV
    fast_min = epithermal_max

    for pos in positions:
        if pos not in spectra:
            continue

        # Get spectrum data
        energy_midpoints = spectra[pos]['energy_midpoints']
        spectrum = spectra[pos]['spectrum']

        # Find indices for energy ranges
        thermal_indices = np.where(energy_midpoints <= thermal_max)[0]
        epithermal_indices = np.where((energy_midpoints > epithermal_min) &
                                      (energy_midpoints <= epithermal_max))[0]
        fast_indices = np.where(energy_midpoints > fast_min)[0]

        # Calculate thermal to fast ratio
        thermal_flux = np.sum(spectrum[thermal_indices]) if thermal_indices.size > 0 else 0
        fast_flux = np.sum(spectrum[fast_indices]) if fast_indices.size > 0 else 0

        if fast_flux > 0:
            thermal_to_fast = thermal_flux / fast_flux
        else:
            thermal_to_fast = 0

        results['thermal_to_fast_ratio'][pos] = thermal_to_fast

        # Calculate epithermal index (modified 1/E slope)
        if epithermal_indices.size > 1:
            # Use ln(E) vs ln(ϕ·E) fit
            log_E = np.log(energy_midpoints[epithermal_indices])
            log_phi_E = np.log(spectrum[epithermal_indices] * energy_midpoints[epithermal_indices])

            # Linear fit to get -alpha + 1
            if len(log_E) > 2:  # Need at least 3 points for a meaningful fit
                try:
                    coefs = np.polyfit(log_E, log_phi_E, 1)
                    epithermal_index = 1 - coefs[0]  # alpha = 1 - slope
                except:
                    epithermal_index = 0
            else:
                epithermal_index = 0
        else:
            epithermal_index = 0

        results['epithermal_index'][pos] = epithermal_index

        # Calculate average energy
        if np.sum(spectrum) > 0:
            avg_energy = np.sum(energy_midpoints * spectrum) / np.sum(spectrum)
        else:
            avg_energy = 0

        results['average_energy'][pos] = avg_energy

        # Find peak energy
        if spectrum.size > 0:
            peak_idx = np.argmax(spectrum)
            peak_energy = energy_midpoints[peak_idx]
        else:
            peak_energy = 0

        results['peak_energy'][pos] = peak_energy

        # Calculate FWHM
        if spectrum.size > 0 and np.max(spectrum) > 0:
            half_max = np.max(spectrum) / 2

            # Interpolate to find exact points where spectrum equals half max
            # Create interpolation function
            interp_func = interp1d(energy_midpoints, spectrum - half_max,
                                   kind='linear', fill_value='extrapolate')

            # Find zeros of the function
            from scipy.optimize import fsolve

            # Try to find left and right intersection points
            try:
                left_idx = np.argmax(spectrum)
                right_idx = np.argmax(spectrum)

                # Find indices left and right of peak
                for i in range(peak_idx, 0, -1):
                    if spectrum[i] < half_max:
                        left_idx = i
                        break

                for i in range(peak_idx, len(spectrum) - 1):
                    if spectrum[i] < half_max:
                        right_idx = i
                        break

                # Find zeros using indices as initial guesses
                left_zero = fsolve(interp_func, energy_midpoints[left_idx])[0]
                right_zero = fsolve(interp_func, energy_midpoints[right_idx])[0]

                fwhm = right_zero - left_zero
            except:
                fwhm = 0
        else:
            fwhm = 0

        results['FWHM'][pos] = fwhm

    return results


def plot_spectral_indices(indices_data, source_energy, channel_diameter, filename=None):
    """
    Plot calculated spectral indices.

    Args:
        indices_data (dict): Spectral indices data
        source_energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        filename (str, optional): Filename for the output plot

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()

    # Extract positions and metrics
    positions = []
    thermal_fast_ratios = []
    epithermal_indices = []
    avg_energies = []
    peak_energies = []

    for pos in indices_data['average_energy'].keys():
        if isinstance(pos, (list, tuple)) and len(pos) >= 1:
            # Use x-coordinate as depth
            depth = pos[0]
            positions.append(depth)

            # Add metrics
            thermal_fast_ratios.append(indices_data['thermal_to_fast_ratio'].get(pos, 0))
            epithermal_indices.append(indices_data['epithermal_index'].get(pos, 0))
            avg_energies.append(indices_data['average_energy'].get(pos, 0))
            peak_energies.append(indices_data['peak_energy'].get(pos, 0))

    # Sort by position
    sorted_data = sorted(zip(positions, thermal_fast_ratios, epithermal_indices,
                             avg_energies, peak_energies))

    positions = [d[0] for d in sorted_data]
    thermal_fast_ratios = [d[1] for d in sorted_data]
    epithermal_indices = [d[2] for d in sorted_data]
    avg_energies = [d[3] for d in sorted_data]
    peak_energies = [d[4] for d in sorted_data]

    # Plot thermal/fast ratio
    axs[0].plot(positions, thermal_fast_ratios, 'o-', linewidth=2)
    axs[0].set_xlabel('Position (cm)')
    axs[0].set_ylabel('Thermal/Fast Ratio')
    axs[0].set_title('Thermal to Fast Neutron Ratio')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot epithermal index
    # Plot epithermal index
    axs[1].plot(positions, epithermal_indices, 'o-', linewidth=2)
    axs[1].set_xlabel('Position (cm)')
    axs[1].set_ylabel('Epithermal Index')
    axs[1].set_title('Epithermal Index (α)')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Plot average energy
    axs[2].plot(positions, avg_energies, 'o-', linewidth=2)
    axs[2].set_xlabel('Position (cm)')
    axs[2].set_ylabel('Energy (MeV)')
    axs[2].set_title('Average Energy')
    axs[2].grid(True, linestyle='--', alpha=0.6)

    # Plot peak energy
    axs[3].plot(positions, peak_energies, 'o-', linewidth=2)
    axs[3].set_xlabel('Position (cm)')
    axs[3].set_ylabel('Energy (MeV)')
    axs[3].set_title('Peak Energy')
    axs[3].grid(True, linestyle='--', alpha=0.6)

    # Add vertical line for wall position in all subplots
    for ax in axs:
        ax.axvline(x=WALL_THICKNESS, color='red', linestyle='--',
                   label=f'Wall Exit ({WALL_THICKNESS} cm)')

    # Add legend to first plot only
    axs[0].legend()

    # Add overall title
    fig.suptitle(f'Spectral Indices - Energy: {source_energy} MeV, Channel Ø{channel_diameter} cm',
                 fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure if filename provided
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Load sample results for testing
    try:
        # Load tallies from state point file if available
        from pathlib import Path

        # Load sample results (using dummy data for testing)
        results_file = os.path.join(DATA_DIR, 'simulation_results.json')

        if os.path.exists(results_file):
            print("Loading spectrum data from simulation results...")
            with open(results_file, 'r') as f:
                results = json.load(f)

            # Extract spectra from simulation results
            spectra = {}

            for sim_result in results:
                energy = sim_result.get('source_energy', 1.0)
                channel_diameter = sim_result.get('channel_diameter', 0.5)

                # Get spectra at different positions
                tally_data = sim_result.get('tallies', {})
                for tally_name, tally in tally_data.items():
                    if 'spectrum' in tally_name:
                        # Extract position information from tally name
                        # Example: "spectrum_x10_y0_z0"
                        parts = tally_name.split('_')
                        if len(parts) >= 4:
                            try:
                                x = float(parts[1][1:])  # Remove 'x' prefix
                                y = float(parts[2][1:])  # Remove 'y' prefix
                                z = float(parts[3][1:])  # Remove 'z' prefix
                                pos = (x, y, z)

                                # Get spectrum
                                spectra[pos] = {
                                    'energy_midpoints': tally.get('energy_grid', []),
                                    'energy_bins': tally.get('energy_bins', []),
                                    'spectrum': tally.get('result', [])
                                }
                            except ValueError:
                                print(f"Could not parse position from tally name: {tally_name}")

            # Define positions of interest
            positions = sorted(list(spectra.keys()), key=lambda p: p[0])

            # Analyze spectrum hardening
            hardening_metrics = analyze_spectrum_hardening(spectra, positions)

            # Plot spectrum comparison
            plot_spectrum_comparison(
                spectra, positions,
                title=f'Energy Spectrum Comparison - E:{energy} MeV, Channel Ø{channel_diameter} cm',
                filename=f'spectrum_comparison_E{energy}_D{channel_diameter}.png'
            )

            # Calculate and plot attenuation
            attenuation_data = calculate_attenuation_coefficients(
                spectra, positions, energy, channel_diameter
            )

            plot_attenuation_curve(
                attenuation_data, energy, channel_diameter,
                filename=f'attenuation_curve_E{energy}_D{channel_diameter}.png'
            )

            # Plot hardening metrics
            plot_hardening_metrics(
                hardening_metrics, energy, channel_diameter,
                filename=f'spectrum_hardening_E{energy}_D{channel_diameter}.png'
            )

            # Analyze energy-dependent attenuation
            energy_attenuation = analyze_energy_dependent_attenuation(
                spectra, positions, energy, channel_diameter
            )

            plot_energy_dependent_attenuation(
                energy_attenuation, positions, energy, channel_diameter,
                filename=f'energy_attenuation_E{energy}_D{channel_diameter}.png'
            )

            # Calculate spectral indices
            spectral_indices = analyze_spectral_indices(
                spectra, positions, energy, channel_diameter
            )

            plot_spectral_indices(
                spectral_indices, energy, channel_diameter,
                filename=f'spectral_indices_E{energy}_D{channel_diameter}.png'
            )

            # Export spectrum data
            export_path = export_spectrum_data(spectra, positions, energy, channel_diameter)

            print(f"Spectrum analysis complete. Data exported to: {export_path}")

        else:
            print(f"Spectrum analysis: No simulation results found at {results_file}")

    except Exception as e:
        import traceback

        print(f"Error in spectrum analysis: {str(e)}")
        traceback.print_exc()

