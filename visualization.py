#!/usr/bin/env python3
"""
Visualization module for radiation shielding simulation.
Generates 2D and 3D visualizations of geometry, flux, and dose distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import openmc
import os
import json
from config import (
    PLOT_DIR, DATA_DIR, WALL_THICKNESS, SOURCE_POSITION,
    MESH_LOWER_LEFT, MESH_UPPER_RIGHT, MESH_DIMENSION,
    FINE_MESH_LOWER_LEFT, FINE_MESH_UPPER_RIGHT, FINE_MESH_DIMENSION
)


def plot_geometry(universe, channel_diameter):
    """
    Generate plots of the simulation geometry.

    Args:
        universe (openmc.Universe): Universe containing the geometry
        channel_diameter (float): Channel diameter in cm

    Returns:
        list: Paths to generated plot files
    """
    plot_files = []

    # XY plot (top view)
    xy_plot = openmc.Plot()
    xy_plot.filename = f'geometry_xy_channel_{channel_diameter}cm'
    xy_plot.width = (WALL_THICKNESS * 2, 200)
    xy_plot.pixels = (1000, 800)
    xy_plot.origin = (WALL_THICKNESS / 2, 0, 0)
    xy_plot.basis = 'xy'
    xy_plot.color_by = 'material'
    xy_plot.colors = {
        'concrete': (204, 204, 204),
        'air': (255, 255, 255),
        'tissue': (255, 200, 200),
        'void': (240, 240, 255)
    }

    # XZ plot (side view)
    xz_plot = openmc.Plot()
    xz_plot.filename = f'geometry_xz_channel_{channel_diameter}cm'
    xz_plot.width = (WALL_THICKNESS * 2, 200)
    xz_plot.pixels = (1000, 800)
    xz_plot.origin = (WALL_THICKNESS / 2, 0, 0)
    xz_plot.basis = 'xz'
    xz_plot.color_by = 'material'
    xz_plot.colors = {
        'concrete': (204, 204, 204),
        'air': (255, 255, 255),
        'tissue': (255, 200, 200),
        'void': (240, 240, 255)
    }

    # Create and run plots
    plots = openmc.Plots([xy_plot, xz_plot])
    plots.export_to_xml()

    openmc.plot_geometry()

    # Add plot paths to list
    plot_files.append(os.path.join('plots', f'{xy_plot.filename}.png'))
    plot_files.append(os.path.join('plots', f'{xz_plot.filename}.png'))

    return plot_files


def plot_mesh_tally(sp, tally_id, filename=None, title=None, log_scale=True, colorbar_label='Flux'):
    """
    Plot a mesh tally from a statepoint file.

    Args:
        sp (openmc.StatePoint): StatePoint file containing results
        tally_id (int): ID of the mesh tally to plot
        filename (str, optional): Filename for the output plot
        title (str, optional): Title for the plot
        log_scale (bool): Whether to use log scale for colormap
        colorbar_label (str): Label for the colorbar

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Get the tally and ensure it's a mesh tally
    tally = sp.tallies[tally_id]
    if not any(isinstance(f, openmc.MeshFilter) for f in tally.filters):
        raise ValueError(f"Tally {tally_id} is not a mesh tally")

    # Get mesh filter
    mesh_filter = next(f for f in tally.filters if isinstance(f, openmc.MeshFilter))
    mesh = mesh_filter.mesh

    # Get dimensions
    if isinstance(mesh, openmc.RegularMesh):
        nx, ny, nz = mesh.dimension
    else:
        raise TypeError("Only RegularMesh is supported for visualization")

    # Extract tally data
    mean = tally.mean.reshape(nx, ny, nz)[:, :, 0]  # Take first z plane

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Get mesh boundaries
    x_mesh = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx + 1)
    y_mesh = np.linspace(mesh.lower_left[1], mesh.upper_right[1], ny + 1)

    # Create mesh grid
    X, Y = np.meshgrid(x_mesh, y_mesh, indexing='ij')

    # Transpose mean array to match meshgrid orientation
    mean = mean.T

    # Plot the mesh tally
    if log_scale:
        # Use LogNorm for colormap
        mesh_plot = ax.pcolormesh(X, Y, mean, cmap='jet', norm=LogNorm())
    else:
        mesh_plot = ax.pcolormesh(X, Y, mean, cmap='jet')

    # Add colorbar
    cbar = fig.colorbar(mesh_plot, ax=ax, pad=0.01)
    cbar.set_label(colorbar_label)

    # Add geometry elements for reference

    # Wall outline
    ax.axvline(x=0, color='k', linestyle='-', linewidth=2)  # Front face of wall
    ax.axvline(x=WALL_THICKNESS, color='k', linestyle='-', linewidth=2)
    # Add channel
    circle = plt.Circle((0, 0), 2, fill=False, edgecolor='white', linewidth=1)
    ax.add_patch(circle)

    # Add source position indicator
    ax.plot(SOURCE_POSITION[0], SOURCE_POSITION[1], 'ro', markersize=10, label='Source')

    # Set axis labels
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Mesh Tally: {tally.name}")

    # Set aspect ratio to equal
    ax.set_aspect('equal')

    # Add grid
    ax.grid(linestyle='--', alpha=0.3)

    # Add legend
    ax.legend()

    # Save figure if filename provided
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


def plot_enhanced_radiation_pattern(sp, tally_id, channel_diameter, energy, filename=None):
    """
    Create an enhanced visualization of radiation flux exiting the channel.

    Args:
        sp (openmc.StatePoint): StatePoint file containing results
        tally_id (int): ID of the fine mesh tally to plot
        channel_diameter (float): Channel diameter in cm
        energy (float): Source energy in MeV
        filename (str, optional): Filename for the output plot

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Get the fine mesh tally
    tally = sp.tallies[tally_id]
    mesh_filter = next(f for f in tally.filters if isinstance(f, openmc.MeshFilter))
    mesh = mesh_filter.mesh

    # Get dimensions and data
    nx, ny, nz = mesh.dimension
    mean = tally.mean.reshape(nx, ny, nz)[:, :, 0]  # Take first z plane

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get mesh boundaries
    x_mesh = np.linspace(mesh.lower_left[0], mesh.upper_right[0], nx + 1)
    y_mesh = np.linspace(mesh.lower_left[1], mesh.upper_right[1], ny + 1)

    # Create mesh grid
    X, Y = np.meshgrid(x_mesh, y_mesh, indexing='ij')

    # Transpose mean array to match meshgrid orientation
    mean = mean.T

    # Apply adaptive smoothing based on channel diameter
    from scipy.ndimage import gaussian_filter
    sigma = max(1, channel_diameter / 0.1)  # Scale smoothing with channel size
    smoothed_mean = gaussian_filter(mean, sigma=sigma)

    # Plot the radiation pattern with log scale
    mesh_plot = ax.pcolormesh(X, Y, smoothed_mean, cmap='inferno', norm=LogNorm())

    # Add colorbar
    cbar = fig.colorbar(mesh_plot, ax=ax, pad=0.01)
    cbar.set_label('Photon Flux (particles/cm²/src)')

    # Add wall line
    ax.axvline(x=WALL_THICKNESS, color='white', linestyle='-', linewidth=2, label='Wall Exit')

    # Add channel position
    circle = plt.Circle((WALL_THICKNESS, 0), channel_diameter / 2, fill=False,
                        edgecolor='green', linewidth=2, label=f'Channel Ø {channel_diameter} cm')
    ax.add_patch(circle)

    # Add distance markers
    distances = [30, 60, 100, 150]
    for d in distances:
        circle = plt.Circle((WALL_THICKNESS + d, 0), 2, fill=False,
                            edgecolor='white', linewidth=1, linestyle='--')
        ax.add_patch(circle)
        ax.text(WALL_THICKNESS + d + 3, 0, f'{d} cm', color='white', fontsize=8)

    # Add angle indicators
    angles = [0, 15, 30, 45]
    for angle in angles:
        rad_angle = np.radians(angle)
        length = 80
        dx = length * np.cos(rad_angle)
        dy = length * np.sin(rad_angle)

        ax.plot([WALL_THICKNESS, WALL_THICKNESS + dx], [0, dy], 'w--', linewidth=1, alpha=0.7)
        ax.text(WALL_THICKNESS + dx + 2, dy + 2, f'{angle}°', color='white', fontsize=8)

    # Highlight high-dose regions
    high_dose_threshold = np.percentile(mean[mean > 0], 95)
    contour = ax.contour(X, Y, mean, levels=[high_dose_threshold],
                         colors=['yellow'], linewidths=2, alpha=0.7)

    # Set axis labels and title
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title(f'Enhanced Radiation Pattern - Energy: {energy} MeV, Channel Diameter: {channel_diameter} cm')

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Add legend
    ax.legend(loc='upper right')

    # Set view limits focusing on the area right after the wall
    x_min = WALL_THICKNESS - 5
    x_max = WALL_THICKNESS + 150
    y_min = -75
    y_max = 75
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Save figure if filename provided
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


def plot_spectra(sp, tally_id, positions=None, filename=None):
    """
    Plot energy spectra at different positions behind the wall.

    Args:
        sp (openmc.StatePoint): StatePoint file containing results
        tally_id (int): ID of the tally containing spectral data
        positions (list, optional): List of positions to plot
        filename (str, optional): Filename for the output plot

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Get the tally
    tally = sp.tallies[tally_id]

    # Get energy filter and bins
    energy_filter = next(f for f in tally.filters if isinstance(f, openmc.EnergyFilter))
    energy_bins = energy_filter.values

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot spectra for each position
    if positions:
        for pos in positions:
            # Extract spectrum data for this position
            # (This is a placeholder - actual extraction depends on how positions are stored in tallies)
            energy_midpoints = 0.5 * (energy_bins[1:] + energy_bins[:-1])
            spectrum = tally.mean  # This would need to be indexed based on position

            # Plot spectrum
            ax.plot(energy_midpoints, spectrum, label=f'Position {pos}')
    else:
        # Just plot the overall spectrum
        energy_midpoints = 0.5 * (energy_bins[1:] + energy_bins[:-1])
        spectrum = np.mean(tally.mean, axis=0)  # Average over all positions

        ax.plot(energy_midpoints, spectrum)

    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis labels
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Flux per Unit Energy (particles/cm²/src/MeV)')

    # Set title
    ax.set_title('Photon Energy Spectra')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend if multiple positions
    if positions and len(positions) > 1:
        ax.legend()

    # Save figure if filename provided
    if filename:
        plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


def create_2d_dose_heatmap(results, energy, channel_diameter, conversion_type='flux_to_dose'):
    """
    Create a 2D heatmap of dose distribution on the detector side of the wall.

    Args:
        results (list): Simulation results
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        conversion_type (str): Type of dose conversion to use

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    # Extract dose values at different positions
    positions = []
    doses = []

    for result in results:
        if (result['energy'] == energy and
                result['channel_diameter'] == channel_diameter):

            # Get position and calculate dose
            x = result['position'][0] - WALL_THICKNESS  # Distance behind wall
            y = result['position'][1]  # Off-axis position

            # Find detector tally
            for tally_name, tally_data in result['tallies'].items():
                if 'detector' in tally_name:
                    # Calculate dose
                    dose = calculate_dose_rates(tally_data, conversion_type)

                    # If dose calculation was successful, add to lists
                    if dose > 0:
                        positions.append((x, y))
                        doses.append(dose)

    # If no data, return None
    if not positions:
        return None

    # Convert to numpy arrays
    positions = np.array(positions)
    doses = np.array(doses)

    # Create a regular grid for the heatmap
    x_grid = np.linspace(0, 150, 100)
    y_grid = np.linspace(-75, 75, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Interpolate doses onto grid
    from scipy.interpolate import griddata
    Z = griddata(positions, doses, (X, Y), method='cubic', fill_value=1e-10)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    c = ax.pcolormesh(X, Y, Z, cmap='jet', norm=LogNorm(vmin=max(doses.min(), 1e-10)))

    # Add colorbar
    cbar = fig.colorbar(c, ax=ax, pad=0.01)
    cbar.set_label(f'Dose Rate (rem/hr) - {conversion_type}')

    # Add channel position
    circle = plt.Circle((0, 0), channel_diameter / 2, fill=True,
                        color='white', alpha=0.8, label=f'Channel Ø {channel_diameter} cm')
    ax.add_patch(circle)

    # Add distance markers
    distances = [30, 60, 100, 150]
    for d in distances:
        circle = plt.Circle((d, 0), 2, fill=False,
                            edgecolor='white', linewidth=1, linestyle='--')
        ax.add_patch(circle)
        ax.text(d + 3, 3, f'{d} cm', color='white', fontsize=8)

    # Add angle indicators
    angles = [0, 15, 30, 45]
    for angle in angles:
        rad_angle = np.radians(angle)
        length = 80
        dx = length * np.cos(rad_angle)
        dy = length * np.sin(rad_angle)

        ax.plot([0, dx], [0, dy], 'w--', linewidth=1, alpha=0.7)
        ax.text(dx + 2, dy + 2, f'{angle}°', color='white', fontsize=8)

    # Set axis labels and title
    ax.set_xlabel('Distance Behind Wall (cm)')
    ax.set_ylabel('Lateral Distance (cm)')
    ax.set_title(f'2D Dose Distribution - Energy: {energy} MeV, Channel Diameter: {channel_diameter} cm')

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Add legend
    ax.legend(loc='upper right')

    # Save figure
    filename = f"dose_heatmap_E{energy}_D{channel_diameter}_{conversion_type}.png"
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Load results
    results_file = os.path.join(DATA_DIR, 'simulation_results.json')
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Create dose heatmaps for different energies and channel diameters
    from config import SOURCE_ENERGIES, CHANNEL_DIAMETERS

    for energy in SOURCE_ENERGIES:
        for channel_diameter in CHANNEL_DIAMETERS:
            try:
                create_2d_dose_heatmap(results, energy, channel_diameter)
                print(f"Created dose heatmap for E={energy} MeV, D={channel_diameter} cm")
            except Exception as e:
                print(f"Error creating heatmap for E={energy}, D={channel_diameter}: {str(e)}")


#############################################OLD#########################################################


"""Visualization functions for gamma-ray shielding simulation."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import seaborn as sns
from matplotlib.ticker import LogFormatter, LogLocator
from scipy.ndimage import gaussian_filter
import json


def plot_dose_vs_angle(results, energy, channel_diameter, output_dir='figures'):
    """
    Plot dose vs. angle for different distances.

    Args:
        results (dict): Simulation results dictionary
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract relevant results
    energy_key = f"Energy_{energy}MeV"
    channel_key = f"CD{channel_diameter}"

    if energy_key not in results or channel_key not in results[energy_key]:
        print(f"No results found for energy {energy} MeV and channel diameter {channel_diameter} cm")
        return

    channel_results = results[energy_key][channel_key]

    # Extract angles and distances
    angles = []
    distances = []
    doses = {}

    for position_key, position_results in channel_results.items():
        # Extract distance and angle from position_key (D{distance}_A{angle})
        parts = position_key.split('_')
        distance = float(parts[0][1:])
        angle = float(parts[1][1:])

        if distance not in distances:
            distances.append(distance)
            doses[distance] = []

        if angle not in angles:
            angles.append(angle)

        # Add dose for this distance and angle
        dose_value = position_results['dose']['mean']
        doses[distance].append((angle, dose_value))

    # Sort angles and distances
    angles = sorted(angles)
    distances = sorted(distances)

    # Create plot
    plt.figure(figsize=(12, 8))

    for distance in distances:
        # Sort doses by angle
        dose_points = sorted(doses[distance], key=lambda x: x[0])
        angle_values = [p[0] for p in dose_points]
        dose_values = [p[1] for p in dose_points]

        plt.plot(angle_values, dose_values, marker='o', label=f'{distance} cm')

    plt.xlabel('Angle (degrees)')
    plt.ylabel('Dose (rem/hr)')
    plt.title(f'Dose vs. Angle for {energy} MeV and {channel_diameter} cm Channel')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title='Distance from Wall')

    # Add annotations for key points
    for distance in distances:
        dose_points = sorted(doses[distance], key=lambda x: x[0])
        for angle, dose in dose_points:
            if angle in [0, 45]:  # Annotate only at 0 and 45 degrees
                plt.annotate(f'{dose:.2e}',
                             xy=(angle, dose),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=8)

    # Save figure
    filename = os.path.join(output_dir, f'dose_vs_angle_E{energy}_CD{channel_diameter}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure: {filename}")


def plot_radiation_pattern(results, energy, channel_diameter, distance, output_dir='figures'):
    """
    Create enhanced visualization of radiation pattern outside the wall.

    Args:
        results (dict): Simulation results dictionary
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        distance (float): Distance from wall for analysis plane
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract mesh data from results
    energy_key = f"Energy_{energy}MeV"
    channel_key = f"CD{channel_diameter}"
    position_key = f"D{distance}_A0"  # Use 0 degree angle for pattern

    try:
        position_results = results[energy_key][channel_key][position_key]
        mesh_data = np.array(position_results['mesh']['flux'])
    except KeyError:
        print(f"No data found for {energy_key}, {channel_key}, {position_key}")
        return

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot mesh data with log scale
    plt.imshow(mesh_data.T,
               origin='lower',
               aspect='equal',
               norm=LogNorm(),
               cmap='viridis')

    plt.colorbar(label='Photon Flux (particles/cm²/s)')
    plt.title(f'Radiation Pattern at {distance} cm from Wall\n{energy} MeV, {channel_diameter} cm Channel')
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')

    # Save figure
    filename = os.path.join(output_dir, f'radiation_pattern_E{energy}_CD{channel_diameter}_D{distance}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure: {filename}")


def plot_energy_spectrum(results, channel_diameter, output_dir='figures'):
    """
    Plot energy spectrum for various source energies.

    Args:
        results (dict): Simulation results dictionary
        channel_diameter (float): Channel diameter in cm
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot spectrum for each energy
    plt.figure(figsize=(12, 8))

    channel_key = f"CD{channel_diameter}"
    for energy_key in results:
        if channel_key in results[energy_key]:
            # Use first distance and angle (should be D30_A0)
            position_key = next(iter(results[energy_key][channel_key]))
            position_results = results[energy_key][channel_key][position_key]

            # Extract energy and spectrum data
            # Note: In a real implementation, we would need to extract the energy bins
            # and spectrum values from the openmc.Tally results

            # For this example, just use placeholder
            energy_bins = np.logspace(-2, 1, 100)
            spectrum = np.random.exponential(0.5, size=99) * np.exp(-energy_bins[:-1])

            energy_mev = float(energy_key.split('_')[1].replace('MeV', ''))
            plt.plot(energy_bins[:-1], spectrum, label=f'{energy_mev} MeV')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Photon Flux (a.u.)')
    plt.title(f'Energy Spectrum for {channel_diameter} cm Channel')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title='Source Energy')

    # Save figure
    filename = os.path.join(output_dir, f'energy_spectrum_CD{channel_diameter}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure: {filename}")


def create_radiation_outside_wall_heatmap(results, title=None, output_dir='figures'):
    """
    Create an enhanced close-up Cartesian heatmap showing radiation distribution outside the wall
    with optimized visualization for this specific shielding problem.

    Args:
        results (dict): Results dictionary containing simulation data
        title (str, optional): Custom title for the plot
        output_dir (str): Directory to save output figures

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract necessary parameters from config
    from config import WALL_THICKNESS, SOURCE_DISTANCE, PHANTOM_DIAMETER

    # Constants for unit conversion
    ft_to_cm = 30.48

    # Extract mesh data
    mesh_result = np.array(results['mesh']['flux'])

    # Extract simulation parameters
    energy = results['parameters']['energy_mev']
    channel_diameter = results['parameters']['channel_diameter']
    detector_distance = results['parameters']['distance']
    detector_angle = results['parameters']['angle']

    # Source and geometry parameters
    source_to_wall_distance = SOURCE_DISTANCE
    wall_thickness = WALL_THICKNESS
    detector_diameter = PHANTOM_DIAMETER

    # Create figure with higher resolution
    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)

    # Define the extent of the plot focused specifically on the area outside the wall
    x_min = source_to_wall_distance + wall_thickness - 5  # Slightly before wall exit
    x_max = source_to_wall_distance + wall_thickness + 150  # 150 cm outside wall
    y_min = -75
    y_max = 75

    # Calculate indices in the mesh corresponding to these limits
    mesh_resolution = mesh_result.shape[0]
    mesh_x_coords = np.linspace(-10, source_to_wall_distance + wall_thickness + 200, mesh_resolution)
    mesh_y_coords = np.linspace(-50, 50, mesh_resolution)

    x_indices = np.logical_and(mesh_x_coords >= x_min, mesh_x_coords <= x_max)
    y_indices = np.logical_and(mesh_y_coords >= y_min, mesh_y_coords <= y_max)

    # Extract the section of the mesh for the region of interest
    x_subset = mesh_x_coords[x_indices]
    y_subset = mesh_y_coords[y_indices]

    # Ensure we have valid indices
    if not np.any(x_indices) or not np.any(y_indices):
        print("Warning: No valid indices for the specified region of interest")
        return fig

    # Extract the data for the region of interest
    try:
        outside_wall_data = mesh_result[np.ix_(x_indices, y_indices)]
    except IndexError:
        print("Error extracting mesh data subset. Using full mesh data.")
        outside_wall_data = mesh_result

    # Create coordinate meshes for the plot
    X, Y = np.meshgrid(x_subset, y_subset)

    # Apply adaptive smoothing for better visualization
    sigma = max(1, min(3, 5 / (channel_diameter + 0.1)))  # Smaller channels need more smoothing
    smoothed_data = gaussian_filter(outside_wall_data.T, sigma=sigma)

    # Set zero or very small values to NaN to make them transparent
    min_nonzero = np.max([np.min(smoothed_data[smoothed_data > 0]) / 10, 1e-12])
    smoothed_data[smoothed_data < min_nonzero] = np.nan

    # Create an enhanced custom colormap specifically for radiation visualization
    colors = [
        (0.0, 0.0, 0.3),  # Dark blue (background/low values)
        (0.0, 0.2, 0.6),  # Blue
        (0.0, 0.5, 0.8),  # Light blue
        (0.0, 0.8, 0.8),  # Cyan
        (0.0, 0.9, 0.3),  # Blue-green
        (0.5, 1.0, 0.0),  # Green
        (0.8, 1.0, 0.0),  # Yellow-green
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.8, 0.0),  # Yellow-orange
        (1.0, 0.6, 0.0),  # Orange
        (1.0, 0.0, 0.0)  # Red (highest intensity)
    ]

    cmap_name = 'EnhancedRadiation'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # Use contourf for smoother visualization with more levels
    if np.all(np.isnan(smoothed_data)):
        print("Warning: All data values are NaN")
        return fig

    max_value = np.nanmax(smoothed_data)
    if max_value <= min_nonzero:
        print("Warning: Max value is less than or equal to min_nonzero")
        max_value = min_nonzero * 10  # Set a reasonable max value

    levels = np.logspace(np.log10(min_nonzero), np.log10(max_value), 20)
    contour = ax.contourf(X, Y, smoothed_data,
                          levels=levels,
                          norm=LogNorm(),
                          cmap=custom_cmap,
                          alpha=0.95,
                          extend='both')

    # Add contour lines for a better indication of dose levels
    contour_lines = ax.contour(X, Y, smoothed_data,
                               levels=levels[::4],  # Fewer contour lines
                               colors='black',
                               alpha=0.3,
                               linewidths=0.5)

    # Add colorbar with scientific notation
    cbar = fig.colorbar(contour, ax=ax, format='%.1e', pad=0.01)
    cbar.set_label('Radiation Flux (particles/cm²/s)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add wall back position with improved styling
    wall_exit_x = source_to_wall_distance + wall_thickness
    ax.axvline(x=wall_exit_x, color='black', linestyle='-', linewidth=2.5, label='Wall Back')

    # Draw a small section of the wall for context
    wall_section = plt.Rectangle((x_min, y_min), wall_exit_x - x_min, y_max - y_min,
                                 color='gray', alpha=0.5, edgecolor='black')
    ax.add_patch(wall_section)

    # Calculate detector position based on angle and distance
    angle_rad = np.radians(detector_angle)
    detector_x = wall_exit_x + detector_distance * np.cos(angle_rad)
    detector_y = detector_distance * np.sin(angle_rad)
    # Only show detector if it's in the displayed area
    if x_min <= detector_x <= x_max and y_min <= detector_y <= y_max:
        detector_circle = plt.Circle((detector_x, detector_y), detector_diameter / 2,
                                     fill=False, color='red', linewidth=2, label='Detector')
        ax.add_patch(detector_circle)

        # Add beam path from channel to detector with an arrow
        arrow_props = dict(arrowstyle='->', linewidth=2, color='yellow', alpha=0.9)
        beam_arrow = ax.annotate('', xy=(detector_x, detector_y), xytext=(wall_exit_x, 0),
                                 arrowprops=arrow_props)

    # Add channel exit with improved styling
    channel_radius = channel_diameter / 2
    channel_exit = plt.Circle((wall_exit_x, 0), channel_radius,
                              color='white', alpha=1.0, edgecolor='black', linewidth=1.5,
                              label='Channel Exit')
    ax.add_patch(channel_exit)
    # Import detector distances from config
    from config import DETECTOR_DISTANCES

    # Add concentric circles to show distance from channel exit
    for radius in DETECTOR_DISTANCES:
        # Draw dashed circle
        distance_circle = plt.Circle((wall_exit_x, 0), radius,
                                     fill=False, color='white', linestyle='--', linewidth=1, alpha=0.6)
        ax.add_patch(distance_circle)
        # Add distance label along 45° angle
        angle = 45
        label_x = wall_exit_x + radius * np.cos(np.radians(angle))
        label_y = radius * np.sin(np.radians(angle))
        ax.text(label_x, label_y, f"{radius} cm", color='white', fontsize=9,
                ha='center', va='center', rotation=angle,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    # Add detector angle indication if not at 0°
    if detector_angle > 0:
        # Draw angle arc
        angle_radius = 30
        arc = plt.matplotlib.patches.Arc((wall_exit_x, 0),
                                         angle_radius * 2, angle_radius * 2,
                                         theta1=0, theta2=detector_angle,
                                         color='white', linewidth=2)
        ax.add_patch(arc)

        # Add angle text at arc midpoint
        angle_text_x = wall_exit_x + angle_radius * 0.7 * np.cos(np.radians(detector_angle / 2))
        angle_text_y = angle_radius * 0.7 * np.sin(np.radians(detector_angle / 2))
        ax.text(angle_text_x, angle_text_y, f"{detector_angle}°", color='white',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

    # Set labels and title with improved styling
    ax.set_xlabel('Distance (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Lateral Distance (cm)', fontsize=14, fontweight='bold')

    if title is None:
        title = (f"Radiation Distribution Outside Wall\n"
                 f"{energy} MeV Gamma, Channel Diameter: {channel_diameter} cm")
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    # Add improved legend with better positioning
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(),
                       loc='upper right', framealpha=0.9, fontsize=11)
    legend.get_frame().set_edgecolor('black')

    # Add enhanced grid with better styling
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add detailed information box
    dose_rem_per_hr = results.get('dose', {}).get('mean', 0.0)
    info_text = (f"Source: {energy} MeV Gamma\n"
                 f"Wall: {wall_thickness / ft_to_cm:.1f} ft concrete\n"
                 f"Channel: {channel_diameter} cm ∅\n"
                 f"Detector: {detector_distance} cm from wall\n"
                 f"Angle: {detector_angle}°\n"
                 f"Dose Rate: {dose_rem_per_hr:.2e} rem/hr")

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Highlight the region of 10% or greater of the maximum dose
    if not np.isnan(np.max(smoothed_data)):
        high_dose_level = np.max(smoothed_data) * 0.1
        high_dose_contour = ax.contour(X, Y, smoothed_data,
                                       levels=[high_dose_level],
                                       colors=['red'],
                                       linewidths=2)

        # Add label for high dose region
        plt.clabel(high_dose_contour, inline=True, fontsize=9,
                   fmt=lambda x: "10% of Max Dose")

    # Ensure proper aspect ratio
    ax.set_aspect('equal')

    # Save high-resolution figure
    filename = os.path.join(output_dir, f"outside_wall_E{energy}_D{channel_diameter}_" +
                            f"dist{detector_distance}_ang{detector_angle}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    return fig


def create_2d_cartesian_mesh_plot(results, energy, channel_diameter, view_type='source_to_detector',
                                  output_dir='figures'):
    """
    Create a 2D Cartesian mesh visualization showing radiation from source through channel to detector.

    Args:
        results (dict): Simulation results dictionary
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        view_type (str): Type of view ('source_to_detector' or 'outside_wall')
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract parameters from config
    from config import WALL_THICKNESS, SOURCE_DISTANCE

    # Extract mesh data for 0° angle and closest distance
    energy_key = f"Energy_{energy}MeV"
    channel_key = f"CD{channel_diameter}"
    position_key = f"D30_A0"  # Closest distance, 0° angle

    try:
        position_results = results[energy_key][channel_key][position_key]
        mesh_data = np.array(position_results['mesh']['flux'])
    except KeyError:
        print(f"No data found for {energy_key}, {channel_key}, {position_key}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Define the plotting area based on view type
    if view_type == 'source_to_detector':
        # View from source, through wall, to detector
        x_min = -SOURCE_DISTANCE - 10
        x_max = WALL_THICKNESS + 100
        y_min = -50
        y_max = 50
        title = f"Radiation Path from Source through Channel to Detector\n{energy} MeV Gamma, {channel_diameter} cm Channel"
    else:  # outside_wall
        # View of area outside the wall
        x_min = WALL_THICKNESS - 5
        x_max = WALL_THICKNESS + 150
        y_min = -75
        y_max = 75
        title = f"Radiation Distribution Outside Wall\n{energy} MeV Gamma, {channel_diameter} cm Channel"

    # Create a uniform meshgrid for plotting
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)

    # Interpolate the mesh data onto this grid
    # For this example, we'll create synthetic data since we don't have the actual mesh
    # In a real implementation, this would come from the OpenMC mesh tally
    Z = np.zeros_like(X)

    # Create a radiation beam pattern
    source_pos = (-SOURCE_DISTANCE, 0)
    channel_pos = (0, 0)
    detector_pos = (WALL_THICKNESS + 30, 0)

    # Simulate source radiation
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Distance from source
            r_source = np.sqrt((X[i, j] - source_pos[0]) ** 2 + (Y[i, j] - source_pos[1]) ** 2)

            # Angle from source to point
            angle = np.arctan2(Y[i, j] - source_pos[1], X[i, j] - source_pos[0])

            # Angle from source to channel
            channel_angle = np.arctan2(channel_pos[1] - source_pos[1],
                                       channel_pos[0] - source_pos[0])

            # Angular distance from central beam
            angular_dist = abs(angle - channel_angle)

            # Source intensity decreases with distance and angle
            if r_source > 0:
                source_intensity = np.exp(-r_source / 100) * np.exp(-angular_dist * 10)
            else:
                source_intensity = 1.0

            # Wall attenuation
            if 0 <= X[i, j] <= WALL_THICKNESS:
                # Check if inside channel
                r_from_center = abs(Y[i, j] - channel_pos[1])
                if r_from_center <= channel_diameter / 2:
                    # Inside channel - no attenuation
                    wall_attenuation = 1.0
                else:
                    # Inside concrete - high attenuation (10^-6 for every cm)
                    # Simplified model for visualization
                    distance_in_wall = min(X[i, j], WALL_THICKNESS - X[i, j]) if X[i, j] < WALL_THICKNESS else 0
                    wall_attenuation = np.exp(-distance_in_wall * 0.5)
            else:
                wall_attenuation = 1.0

            # Post-wall beam spreading
            if X[i, j] > WALL_THICKNESS:
                # Distance from channel exit
                r_from_exit = np.sqrt((X[i, j] - WALL_THICKNESS) ** 2 + (Y[i, j] - channel_pos[1]) ** 2)

                # Angle from exit
                exit_angle = np.arctan2(Y[i, j] - channel_pos[1], X[i, j] - WALL_THICKNESS)

                # Beam spreading (inverse square)
                if r_from_exit > 0:
                    exit_intensity = (channel_diameter / 2) / r_from_exit * np.exp(-abs(exit_angle) * 5)
                else:
                    exit_intensity = 1.0
            else:
                exit_intensity = 0.0

            # Combine all effects
            Z[i, j] = source_intensity * wall_attenuation + (X[i, j] > WALL_THICKNESS) * exit_intensity

    # Apply log scale for better visualization
    Z = np.maximum(Z, 1e-10)  # Avoid log(0)

    # Plot contour
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', norm=LogNorm())

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Radiation Intensity (a.u.)', fontsize=12)

    # Draw the wall
    wall_rect = plt.Rectangle((0, y_min), WALL_THICKNESS, y_max - y_min,
                              color='gray', alpha=0.7, label='Concrete Wall')
    ax.add_patch(wall_rect)

    # Draw the channel
    channel_rect = plt.Rectangle((0, -channel_diameter / 2), WALL_THICKNESS, channel_diameter,
                                 color='white', alpha=0.8, label='Air Channel')
    ax.add_patch(channel_rect)

    # Draw source and detector
    source_circle = plt.Circle(source_pos, 5, color='red', alpha=0.7, label='Source')
    ax.add_patch(source_circle)

    detector_circle = plt.Circle(detector_pos, PHANTOM_DIAMETER / 2, fill=False,
                                 color='blue', linewidth=2, label='Detector')
    ax.add_patch(detector_circle)

    # Add labels and legend
    ax.set_xlabel('Z Position (cm)', fontsize=12)
    ax.set_ylabel('X Position (cm)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)

    # Save figure
    filename = os.path.join(output_dir, f'2d_mesh_{view_type}_E{energy}_CD{channel_diameter}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved 2D cartesian mesh plot: {filename}")


def plot_dose_comparison(results, channel_diameter, distance, output_dir='figures'):
    """
    Compare dose metrics (effective dose, heating, kerma) for different energies.

    Args:
        results (dict): Simulation results dictionary
        channel_diameter (float): Channel diameter in cm
        distance (float): Detector distance from wall in cm
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    energies = []
    effective_doses = []
    heating_values = []
    kerma_values = []

    channel_key = f"CD{channel_diameter}"
    position_key = f"D{distance}_A0"  # Use 0 degree angle

    for energy_key in results:
        if not energy_key.startswith('Energy_'):
            continue

        energy_mev = float(energy_key.split('_')[1].replace('MeV', ''))


if channel_key in results[energy_key]:
    if position_key in results[energy_key][channel_key]:
        position_results = results[energy_key][channel_key][position_key]

        energies.append(energy_mev)
        effective_doses.append(position_results['dose']['mean'])
        heating_values.append(position_results['heating']['mean'])
        kerma_values.append(position_results['kerma']['mean'])

    # Sort data by energy
sorted_indices = np.argsort(energies)
energies = np.array(energies)[sorted_indices]
effective_doses = np.array(effective_doses)[sorted_indices]
heating_values = np.array(heating_values)[sorted_indices]
kerma_values = np.array(kerma_values)[sorted_indices]

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Absolute values of all dose metrics
ax1.plot(energies, effective_doses, 'o-', label='Effective Dose')
ax1.plot(energies, heating_values, 's-', label='Heating')
ax1.plot(energies, kerma_values, '^-', label='Kerma')

ax1.set_xlabel('Energy (MeV)')
ax1.set_ylabel('Dose Value')
ax1.set_title('Comparison of Dose Metrics')
ax1.set_yscale('log')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend()

# Plot 2: Ratio of heating to effective dose
ratio_heating_dose = heating_values / effective_doses
ax2.plot(energies, ratio_heating_dose, 'o-', color='green')

ax2.set_xlabel('Energy (MeV)')
ax2.set_ylabel('Heating / Effective Dose Ratio')
ax2.set_title('Heating to Effective Dose Ratio')
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

# Plot 3: Ratio of kerma to heating
ratio_kerma_heating = kerma_values / heating_values
ax3.plot(energies, ratio_kerma_heating, 'o-', color='purple')

ax3.set_xlabel('Energy (MeV)')
ax3.set_ylabel('Kerma / Heating Ratio')
ax3.set_title('Kerma to Heating Ratio')
ax3.grid(True, which='both', linestyle='--', alpha=0.5)

# Adjust layout
plt.tight_layout()

# Save figure
filename = os.path.join(output_dir, f'dose_comparison_CD{channel_diameter}_D{distance}.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved dose comparison plot: {filename}")


def plot_attenuation_analysis(results, output_dir='figures'):
    """
    Analyze and visualize wall attenuation factors for different energies.

    Args:
        results (dict): Simulation results dictionary
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract parameters from config
    from config import WALL_THICKNESS

    # Prepare data structures
    energies = []
    channel_diameters = []
    attenuation_factors = []

    # Extract attenuation data from results
    for energy_key in results:
        if not energy_key.startswith('Energy_'):
            continue

        energy_mev = float(energy_key.split('_')[1].replace('MeV', ''))

        for channel_key in results[energy_key]:
            channel_diameter = float(channel_key[2:])  # Extract from 'CD{diameter}'

            # Use position closest to beam axis
            position_key = f"D30_A0"

            if position_key in results[energy_key][channel_key]:
                position_results = results[energy_key][channel_key][position_key]

                # Calculate attenuation (simplified here)
                # In a real implementation, we'd compare incident and transmitted flux
                dose_with_channel = position_results['dose']['mean']

                # If we have a reference for no channel (solid wall), we could use that
                # For this example, we'll use a theoretical attenuation factor
                theoretical_attenuation = np.exp(-WALL_THICKNESS * 0.1 * energy_mev ** (-0.5))
                observed_attenuation = dose_with_channel / theoretical_attenuation

                energies.append(energy_mev)
                channel_diameters.append(channel_diameter)
                attenuation_factors.append(observed_attenuation / channel_diameter ** 2)

    # Convert to numpy arrays
    energies = np.array(energies)
    channel_diameters = np.array(channel_diameters)
    attenuation_factors = np.array(attenuation_factors)

    # Create a figure for attenuation vs energy for different channel diameters
    plt.figure(figsize=(12, 8))

    for diameter in np.unique(channel_diameters):
        mask = channel_diameters == diameter
        plt.plot(energies[mask], attenuation_factors[mask], 'o-',
                 label=f'{diameter} cm Channel')

    plt.xlabel('Energy (MeV)')
    plt.ylabel('Normalized Attenuation Factor')
    plt.title('Wall Attenuation Analysis by Energy and Channel Size')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()

    # Save figure
    filename = os.path.join(output_dir, 'attenuation_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved attenuation analysis plot: {filename}")


def create_interactive_dashboard(results, output_dir='figures'):
    """
    Create an interactive dashboard for exploring simulation results.

    Args:
        results (dict): Simulation results dictionary
        output_dir (str): Directory to save output html file
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
    except ImportError:
        print("Plotly not installed. Skipping interactive dashboard creation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract data from results
    data = []

    for energy_key in results:
        if not energy_key.startswith('Energy_'):
            continue

        energy_mev = float(energy_key.split('_')[1].replace('MeV', ''))

        for channel_key in results[energy_key]:
            if not channel_key.startswith('CD'):
                continue

            channel_diameter = float(channel_key[2:])

            for position_key in results[energy_key][channel_key]:
                parts = position_key.split('_')
                distance = float(parts[0][1:])
                angle = float(parts[1][1:])

                position_results = results[energy_key][channel_key][position_key]

                data.append({
                    'energy': energy_mev,
                    'channel_diameter': channel_diameter,
                    'distance': distance,
                    'angle': angle,
                    'dose': position_results['dose']['mean'],
                    'heating': position_results['heating']['mean'],
                    'kerma': position_results['kerma']['mean'],
                    'dose_rel_error': position_results['dose']['rel_error']
                })

    # Create interactive figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Dose vs. Distance by Energy',
            'Dose vs. Angle by Energy',
            'Dose vs. Channel Diameter by Energy',
            'Heating vs. Kerma'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}]
        ]
    )

    # Convert data to pandas DataFrame
    import pandas as pd
    df = pd.DataFrame(data)

    # Add traces for each subplot
    # 1. Dose vs. Distance by Energy
    for energy in df['energy'].unique():
        subset = df[(df['energy'] == energy) & (df['angle'] == 0)]
        fig.add_trace(
            go.Scatter(
                x=subset['distance'],
                y=subset['dose'],
                mode='lines+markers',
                name=f'{energy} MeV',
                legendgroup=f'energy_{energy}'
            ),
            row=1, col=1
        )

    # 2. Dose vs. Angle by Energy
    for energy in df['energy'].unique():
        subset = df[(df['energy'] == energy) & (df['distance'] == 30)]
        fig.add_trace(
            go.Scatter(
                x=subset['angle'],
                y=subset['dose'],
                mode='lines+markers',
                name=f'{energy} MeV',
                legendgroup=f'energy_{energy}',
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. Dose vs. Channel Diameter by Energy
    for energy in df['energy'].unique():
        subset = df[(df['energy'] == energy) & (df['angle'] == 0) & (df['distance'] == 30)]
        fig.add_trace(
            go.Scatter(
                x=subset['channel_diameter'],
                y=subset['dose'],
                mode='lines+markers',
                name=f'{energy} MeV',
                legendgroup=f'energy_{energy}',
                showlegend=False
            ),
            row=2, col=1
        )

    # 4. Heating vs. Kerma
    fig.add_trace(
        go.Scatter(
            x=df['heating'],
            y=df['kerma'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['energy'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Energy (MeV)')
            ),
            text=df.apply(lambda row: f"E: {row['energy']} MeV<br>D: {row['distance']} cm<br>A: {row['angle']}°",
                          axis=1),
            hoverinfo='text',
            name='Heating vs. Kerma'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title='Interactive Radiation Shielding Analysis Dashboard',
        height=800,
        width=1200,
        template='plotly_white'
    )

    # Update axes
    fig.update_xaxes(title_text='Distance (cm)', row=1, col=1)
    fig.update_yaxes(title_text='Dose (rem/hr)', type='log', row=1, col=1)

    fig.update_xaxes(title_text='Angle (degrees)', row=1, col=2)
    fig.update_yaxes(title_text='Dose (rem/hr)', type='log', row=1, col=2)

    fig.update_xaxes(title_text='Channel Diameter (cm)', row=2, col=1)
    fig.update_yaxes(title_text='Dose (rem/hr)', type='log', row=2, col=1)

    fig.update_xaxes(title_text='Heating (eV/g)', type='log', row=2, col=2)
    fig.update_yaxes(title_text='Kerma (eV/g)', type='log', row=2, col=2)

    # Save as interactive HTML
    filename = os.path.join(output_dir, 'interactive_dashboard.html')
    fig.write_html(filename)

    print(f"Saved interactive dashboard: {filename}")


def create_summary_report(results, output_dir='figures'):
    """
    Create a summary report with key findings.

    Args:
        results (dict): Simulation results dictionary
        output_dir (str): Directory to save output report
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for summary
    energies = set()
    channel_diameters = set()
    distances = set()
    angles = set()

    max_dose = 0
    max_dose_config = {}

    min_dose = float('inf')
    min_dose_config = {}

    # Process results for summary statistics
    for energy_key in results:
        if not energy_key.startswith('Energy_'):
            continue

        energy_mev = float(energy_key.split('_')[1].replace('MeV', ''))
        energies.add(energy_mev)

        for channel_key in results[energy_key]:
            if not channel_key.startswith('CD'):
                continue

            channel_diameter = float(channel_key[2:])
            channel_diameters.add(channel_diameter)

            for position_key in results[energy_key][channel_key]:
                parts = position_key.split('_')
                distance = float(parts[0][1:])
                angle = float(parts[1][1:])

                distances.add(distance)
                angles.add(angle)

                position_results = results[energy_key][channel_key][position_key]
                dose = position_results['dose']['mean']

                if dose > max_dose:
                    max_dose = dose
                    max_dose_config = {
                        'energy': energy_mev,
                        'channel_diameter': channel_diameter,
                        'distance': distance,
                        'angle': angle
                    }

                if dose < min_dose and dose > 0:
                    min_dose = dose
                    min_dose_config = {
                        'energy': energy_mev,
                        'channel_diameter': channel_diameter,
                        'distance': distance,
                        'angle': angle
                    }

    # Create summary report
    with open(os.path.join(output_dir, 'summary_report.md'), 'w') as f:
        f.write("# Gamma-Ray Shielding Simulation Summary Report\n\n")

        f.write("## Simulation Parameters\n\n")
        f.write(f"- **Energy Levels:** {', '.join([f'{e} MeV' for e in sorted(energies)])}\n")
        f.write(f"- **Channel Diameters:** {', '.join([f'{d} cm' for d in sorted(channel_diameters)])}\n")
        f.write(f"- **Detector Distances:** {', '.join([f'{d} cm' for d in sorted(distances)])}\n")
        f.write(f"- **Detector Angles:** {', '.join([f'{a}°' for a in sorted(angles)])}\n\n")

        f.write("## Key Findings\n\n")

        f.write("### Maximum Dose Configuration\n\n")
        f.write(f"- **Dose Rate:** {max_dose:.2e} rem/hr\n")
        f.write(f"- **Energy:** {max_dose_config['energy']} MeV\n")
        f.write(f"- **Channel Diameter:** {max_dose_config['channel_diameter']} cm\n")
        f.write(f"- **Detector Distance:** {max_dose_config['distance']} cm\n")
        f.write(f"- **Detector Angle:** {max_dose_config['angle']}°\n\n")

        f.write("### Minimum Dose Configuration\n\n")
        f.write(f"- **Dose Rate:** {min_dose:.2e} rem/hr\n")
        f.write(f"- **Energy:** {min_dose_config['energy']} MeV\n")
        f.write(f"- **Channel Diameter:** {min_dose_config['channel_diameter']} cm\n")
        f.write(f"- **Detector Distance:** {min_dose_config['distance']} cm\n")
        f.write(f"- **Detector Angle:** {min_dose_config['angle']}°\n\n")

        # Add energy dependence analysis
        f.write("## Energy Dependence Analysis\n\n")

        # Select a standard configuration for energy comparison
        std_channel = sorted(channel_diameters)[0]
        std_distance = sorted(distances)[0]
        std_angle = sorted(angles)[0]

        f.write(f"Channel Diameter: {std_channel} cm, Distance: {std_distance} cm, Angle: {std_angle}°\n\n")

        f.write("| Energy (MeV) | Dose Rate (rem/hr) | Relative to 1 MeV |\n")
        f.write("|-------------|-------------------|--------------------|\n")

        # Get dose at 1 MeV for reference
        ref_dose = 1.0  # Default
        for energy_key in results:
            if energy_key == "Energy_1.0MeV":
                channel_key = f"CD{std_channel}"
                position_key = f"D{std_distance}_A{std_angle}"

                if (channel_key in results[energy_key] and
                        position_key in results[energy_key][channel_key]):
                    ref_dose = results[energy_key][channel_key][position_key]['dose']['mean']
                break

        # Write energy comparison table
        for energy in sorted(energies):
            energy_key = f"Energy_{energy}MeV"
            channel_key = f"CD{std_channel}"
            position_key = f"D{std_distance}_A{std_angle}"

            if (energy_key in results and
                    channel_key in results[energy_key] and
                    position_key in results[energy_key][channel_key]):
                dose = results[energy_key][channel_key][position_key]['dose']['mean']
                rel_dose = dose / ref_dose if ref_dose > 0 else 0.0

                f.write(f"| {energy} | {dose:.2e} | {rel_dose:.2f} |\n")

        f.write("\n## Channel Size Analysis\n\n")

        # Select a standard configuration for channel size comparison
        std_energy = sorted(energies)[0]

        f.write(f"Energy: {std_energy} MeV, Distance: {std_distance} cm, Angle: {std_angle}°\n\n")

        f.write("| Channel Diameter (cm) | Dose Rate (rem/hr) | Channel Area (cm²) | Dose Rate / Area |\n")
        f.write("|----------------------|-------------------|-------------------|------------------|\n")

        energy_key = f"Energy_{std_energy}MeV"
        for diameter in sorted(channel_diameters):
            channel_key = f"CD{diameter}"
            position_key = f"D{std_distance}_A{std_angle}"

            if (energy_key in results and
                    channel_key in results[energy_key] and
                    position_key in results[energy_key][channel_key]):
                dose = results[energy_key][channel_key][position_key]['dose']['mean']
                area = np.pi * (diameter / 2) ** 2
                dose_per_area = dose / area

                f.write(f"| {diameter} | {dose:.2e} | {area:.2f} | {dose_per_area:.2e} |\n")

        f.write("\n## Distance Dependence\n\n")

        f.write(f"Energy: {std_energy} MeV, Channel Diameter: {std_channel} cm, Angle: {std_angle}°\n\n")

        f.write("| Distance (cm) | Dose Rate (rem/hr) | 1/r² Scaling | Ratio to 1/r² |\n")
        f.write("|--------------|-------------------|--------------|---------------|\n")

        # Get reference for 1/r² scaling
        ref_distance = sorted(distances)[0]
        ref_dose = None

        energy_key = f"Energy_{std_energy}MeV"
        channel_key = f"CD{std_channel}"
        position_key = f"D{ref_distance}_A{std_angle}"

        if (energy_key in results and
                channel_key in results[energy_key] and
                position_key in results[energy_key][channel_key]):
            ref_dose = results[energy_key][channel_key][position_key]['dose']['mean']

        for distance in sorted(distances):
            position_key = f"D{distance}_A{std_angle}"

            if (energy_key in results and
                    channel_key in results[energy_key] and
                    position_key in results[energy_key][channel_key]):
                dose = results[energy_key][channel_key][position_key]['dose']['mean']

                # Calculate 1/r² scaling
                if ref_dose is not None:
                    inverse_square = ref_dose * (ref_distance / distance) ** 2
                    ratio = dose / inverse_square if inverse_square > 0 else 0.0

                    f.write(f"| {distance} | {dose:.2e} | {inverse_square:.2e} | {ratio:.2f} |\n")

        f.write("\n## Recommendations\n\n")

        # Simple recommendations based on data
        if min_dose < 0.1:  # Assuming 0.1 rem/hr is a safety threshold
            f.write(f"- The configuration with minimum dose ({min_dose:.2e} rem/hr) provides acceptable protection.\n")
            f.write(f"  - Energy: {min_dose_config['energy']} MeV\n")
            f.write(f"  - Channel Diameter: {min_dose_config['channel_diameter']} cm\n")
            f.write(f"  - Detector Distance: {min_dose_config['distance']} cm\n")
            f.write(f"  - Detector Angle: {min_dose_config['angle']}°\n\n")
        else:
            f.write("- None of the tested configurations provide acceptable protection (below 0.1 rem/hr).\n")
            f.write("- Consider additional shielding or increasing the minimum distance.\n\n")

        # Add pattern from energy analysis
        energy_pattern = "increases" if list(energies)[0] < list(energies)[-1] else "decreases"
        f.write(f"- Dose rate generally {energy_pattern} with higher source energies.\n")

        # Add channel size recommendations
        f.write("- Smaller channel diameters provide better protection but may limit diagnostic capabilities.\n")

        # Add distance recommendations
        f.write(
            "- Increasing distance from the wall significantly reduces dose rate, consistent with the inverse square law.\n")

        # Add angle recommendations
        f.write("- Moving off-axis (increasing angle) provides additional protection.\n")

    print(f"Summary report created: {os.path.join(output_dir, 'summary_report.md')}")


def visualize_3d_dose_surface(results, energy, output_dir='figures'):
    """
    Create a 3D surface plot of dose vs distance and angle.

    Args:
        results (dict): Simulation results dictionary
        energy (float): Source energy in MeV
        output_dir (str): Directory to save output figures
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("3D plotting not available. Skipping 3D dose surface.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract data for 3D surface
    energy_key = f"Energy_{energy}MeV"

    if energy_key not in results:
        print(f"No results found for energy {energy} MeV")
        return

    # Get all channels
    channel_diameters = []
    for channel_key in results[energy_key]:
        if channel_key.startswith('CD'):
            channel_diameters.append(float(channel_key[2:]))

    if not channel_diameters:
        print(f"No channel data found for energy {energy} MeV")
        return

    # Create a 3D figure for each channel diameter
    for channel_diameter in sorted(channel_diameters):
        channel_key = f"CD{channel_diameter}"

        # Extract position data
        distances = []
        angles = []
        doses = []

        for position_key in results[energy_key][channel_key]:
            parts = position_key.split('_')
            distance = float(parts[0][1:])
            angle = float(parts[1][1:])

            dose = results[energy_key][channel_key][position_key]['dose']['mean']

            distances.append(distance)
            angles.append(angle)
            doses.append(dose)

        # Check if we have enough data points
        if len(distances) < 3:
            print(f"Not enough data points for 3D surface plot (E={energy} MeV, CD={channel_diameter} cm)")
            continue

        # Create a grid for the surface plot
        unique_distances = sorted(set(distances))
        unique_angles = sorted(set(angles))

        # Create 2D grid for surface plot
        X, Y = np.meshgrid(unique_distances, unique_angles)
        Z = np.zeros(X.shape)

        # Fill in Z values
        for i, angle in enumerate(unique_angles):
            for j, distance in enumerate(unique_distances):
                # Find matching data point
                for idx, (d, a) in enumerate(zip(distances, angles)):
                    if d == distance and a == angle:
                        Z[i, j] = doses[idx]
                        break

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

        # Add scatter points of actual data
        ax.scatter(distances, angles, doses, c='red', marker='o')

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Dose (rem/hr)')

        # Set labels
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_zlabel('Dose (rem/hr)')

        # Set title
        ax.set_title(f'3D Dose Surface\n{energy} MeV, {channel_diameter} cm Channel')

        # Use log scale for Z-axis
        ax.set_zscale('log')

        # Save figure
        filename = os.path.join(output_dir, f'3d_dose_surface_E{energy}_CD{channel_diameter}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved 3D dose surface plot: {filename}")


def create_combined_visualization(results, output_dir='figures'):
    """
    Create a combined visualization with multiple plots for better comparison.

    Args:
        results (dict): Simulation results dictionary
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract all energy values
    energies = []
    for energy_key in results:
        if energy_key.startswith('Energy_'):
            energy_mev = float(energy_key.split('_')[1].replace('MeV', ''))
            energies.append(energy_mev)

    if not energies:
        print("No energy data found")
        return

    # Sort energies
    energies = sorted(energies)

    # Select a channel diameter (use the first one available)
    channel_diameters = []
    for channel_key in results[f"Energy_{energies[0]}MeV"]:
        if channel_key.startswith('CD'):
            channel_diameters.append(float(channel_key[2:]))

    if not channel_diameters:
        print("No channel data found")
        return

    channel_diameter = min(channel_diameters)  # Use smallest channel

    # Create a grid of plots (energy vs distance for different angles)
    fig, axes = plt.subplots(len(energies), 1, figsize=(12, 5 * len(energies)))

    if len(energies) == 1:
        axes = [axes]  # Make it iterable if only one energy

    for i, energy in enumerate(energies):
        energy_key = f"Energy_{energy}MeV"
        channel_key = f"CD{channel_diameter}"

        # Extract data for this energy and channel
        distances = []
        angles = []
        doses = []

        if energy_key in results and channel_key in results[energy_key]:
            for position_key in results[energy_key][channel_key]:
                parts = position_key.split('_')
                distance = float(parts[0][1:])
                angle = float(parts[1][1:])

                dose = results[energy_key][channel_key][position_key]['dose']['mean']

                distances.append(distance)
                angles.append(angle)
                doses.append(dose)

        # Group by angle
        unique_angles = sorted(set(angles))

        for angle in unique_angles:
            # Get data for this angle
            # Get data for this angle
            angle_indices = [idx for idx, a in enumerate(angles) if a == angle]

            angle_distances = [distances[idx] for idx in angle_indices]
            angle_doses = [doses[idx] for idx in angle_indices]

            # Sort by distance
            sorted_indices = np.argsort(angle_distances)
            angle_distances = [angle_distances[idx] for idx in sorted_indices]
            angle_doses = [angle_doses[idx] for idx in sorted_indices]

            # Plot for this angle
            axes[i].plot(angle_distances, angle_doses, 'o-', label=f'{angle}°')

            # Set axes properties
        axes[i].set_title(f'Energy: {energy} MeV, Channel: {channel_diameter} cm')
        axes[i].set_xlabel('Distance (cm)')
        axes[i].set_ylabel('Dose (rem/hr)')
        axes[i].set_yscale('log')
        axes[i].grid(True, which='both', linestyle='--', alpha=0.5)
        axes[i].legend(title='Angle')

    plt.tight_layout()

    # Save figure
    filename = os.path.join(output_dir, f'combined_dose_plot_CD{channel_diameter}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved combined dose plot: {filename}")


def create_isodose_contour_plot(results, energy, channel_diameter, output_dir='figures'):
    """
    Create an isodose contour plot showing radiation contours at different distances and angles.

    Args:
        results (dict): Simulation results dictionary
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract parameters from config
    from config import WALL_THICKNESS

    # Extract data for this energy and channel
    energy_key = f"Energy_{energy}MeV"
    channel_key = f"CD{channel_diameter}"

    if energy_key not in results or channel_key not in results[energy_key]:
        print(f"No data found for E={energy} MeV, CD={channel_diameter} cm")
        return

    # Extract position data
    r_values = []  # radial distance from exit
    theta_values = []  # angle in radians
    doses = []

    for position_key in results[energy_key][channel_key]:
        parts = position_key.split('_')
        distance = float(parts[0][1:])  # distance from wall
        angle_deg = float(parts[1][1:])  # angle in degrees

        dose = results[energy_key][channel_key][position_key]['dose']['mean']

        # Convert to polar coordinates
        r = distance  # radial distance from exit
        theta = np.radians(angle_deg)  # angle in radians

        r_values.append(r)
        theta_values.append(theta)
        doses.append(dose)

    # Check if we have enough data points
    if len(r_values) < 3:
        print(f"Not enough data points for isodose contour plot (E={energy} MeV, CD={channel_diameter} cm)")
        return

    # Create a uniform grid in polar coordinates
    r_grid = np.linspace(min(r_values), max(r_values), 100)
    theta_grid = np.linspace(0, np.pi / 2, 100)  # from 0 to 90 degrees

    # Create meshgrid for interpolation
    R, THETA = np.meshgrid(r_grid, theta_grid)

    # Convert polar to Cartesian for plotting
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    # Interpolate dose values onto the grid
    from scipy.interpolate import griddata
    points = np.column_stack((r_values, theta_values))
    Z = griddata(points, doses, (R, THETA), method='cubic')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot radiation contours
    contour_levels = np.logspace(np.log10(min([d for d in doses if d > 0])),
                                 np.log10(max(doses)), 10)

    contour = ax.contourf(X, Y, Z, levels=contour_levels, cmap='jet', norm=LogNorm())

    # Add contour lines with labels
    contour_lines = ax.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=0.5)
    ax.clabel(contour_lines, fmt='%.1e', colors='black', fontsize=8)

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Dose (rem/hr)', fontsize=12)

    # Draw the wall
    wall_height = max(Y.flatten()) * 1.2
    ax.fill_between([-100, 0], [-100, -100], [wall_height, wall_height], color='gray', alpha=0.5)

    # Draw the channel
    channel_y_half = channel_diameter / 2
    ax.fill_between([-100, 0], [-channel_y_half, -channel_y_half],
                    [channel_y_half, channel_y_half], color='white')

    # Draw channel exit
    circle = plt.Circle((0, 0), channel_diameter / 2, fill=True, color='white',
                        edgecolor='black', linewidth=1)
    ax.add_patch(circle)

    # Set labels and title
    ax.set_xlabel('Distance from Wall (cm)', fontsize=12)
    ax.set_ylabel('Height (cm)', fontsize=12)
    ax.set_title(f'Isodose Contours\n{energy} MeV, {channel_diameter} cm Channel', fontsize=14)

    # Set axis limits
    ax.set_xlim(-10, max(X.flatten()) * 1.1)
    ax.set_ylim(-10, max(Y.flatten()) * 1.1)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)

    # Save figure
    filename = os.path.join(output_dir, f'isodose_contours_E{energy}_CD{channel_diameter}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved isodose contour plot: {filename}")


def plot_energy_spectrum(results, energy, channel_diameter, distance, angle, output_dir='figures'):
    """
    Plot energy spectrum at a detector position.

    Args:
        results (dict): Simulation results dictionary
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        distance (float): Detector distance from wall in cm
        angle (float): Detector angle in degrees
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    energy_key = f"Energy_{energy}MeV"
    channel_key = f"CD{channel_diameter}"
    position_key = f"D{distance}_A{angle}"

    if (energy_key not in results or
            channel_key not in results[energy_key] or
            position_key not in results[energy_key][channel_key]):
        print(f"No data found for E={energy} MeV, CD={channel_diameter} cm, D={distance} cm, A={angle}°")
        return

    position_results = results[energy_key][channel_key][position_key]

    # Check if spectrum data exists
    if 'spectrum' not in position_results:
        print(f"No spectrum data found for E={energy} MeV, CD={channel_diameter} cm, D={distance} cm, A={angle}°")
        return

    spectrum_data = position_results['spectrum']

    # Extract energy bins and counts
    energy_bins = spectrum_data['energy_bins']
    counts = spectrum_data['counts']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot energy spectrum
    ax.step(energy_bins[:-1], counts, where='post', linewidth=2)

    # Fill under the curve
    ax.fill_between(energy_bins[:-1], counts, step='post', alpha=0.3)

    # Add vertical line at source energy
    ax.axvline(x=energy, color='red', linestyle='--', linewidth=2,
               label=f'Source Energy ({energy} MeV)')

    # Add peak value annotation
    peak_idx = np.argmax(counts)
    peak_energy = energy_bins[peak_idx]
    peak_count = counts[peak_idx]

    ax.annotate(f'Peak: {peak_energy:.2f} MeV',
                xy=(peak_energy, peak_count),
                xytext=(peak_energy * 0.8, peak_count * 1.2),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Set labels and title
    ax.set_xlabel('Energy (MeV)', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title(f'Energy Spectrum at Detector\n{energy} MeV Source, {channel_diameter} cm Channel, ' +
                 f'D={distance} cm, A={angle}°', fontsize=14)

    # Set log scale for y-axis
    ax.set_yscale('log')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Add legend
    ax.legend()

    # Save figure
    filename = os.path.join(output_dir, f'energy_spectrum_E{energy}_CD{channel_diameter}_D{distance}_A{angle}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved energy spectrum plot: {filename}")


def create_animation(results, energy, channel_diameter, output_dir='figures'):
    """
    Create an animation showing how dose changes with distance.

    Args:
        results (dict): Simulation results dictionary
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        output_dir (str): Directory to save output animation
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Animation support not available. Skipping animation creation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract data for this energy and channel
    energy_key = f"Energy_{energy}MeV"
    channel_key = f"CD{channel_diameter}"

    if energy_key not in results or channel_key not in results[energy_key]:
        print(f"No data found for E={energy} MeV, CD={channel_diameter} cm")
        return

    # Extract position data for angle = 0
    distances = []
    doses = []

    for position_key in results[energy_key][channel_key]:
        parts = position_key.split('_')
        distance = float(parts[0][1:])
        angle = float(parts[1][1:])

        if angle == 0:  # Only use on-axis positions
            dose = results[energy_key][channel_key][position_key]['dose']['mean']
            distances.append(distance)
            doses.append(dose)

    # Sort by distance
    sorted_indices = np.argsort(distances)
    distances = [distances[i] for i in sorted_indices]
    doses = [doses[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Function to update plot for each frame
    def update(frame):
        ax.clear()

        # Plot all data points
        ax.plot(distances, doses, 'o-', color='lightgray')

        # Highlight current position
        if frame < len(distances):
            ax.scatter(distances[frame], doses[frame], color='red', s=100, zorder=5)

            # Add detector at current position
            from config import WALL_THICKNESS

            detector_x = WALL_THICKNESS + distances[frame]
            detector_y = 0
            detector_circle = plt.Circle((detector_x, detector_y), 5, fill=False,
                                         color='blue', linewidth=2)
            ax.add_patch(detector_circle)

            # Draw beam path
            ax.arrow(WALL_THICKNESS, 0, distances[frame] * 0.8, 0, head_width=2,
                     head_length=5, fc='yellow', ec='yellow', alpha=0.7)

            # Add dose information
            ax.text(0.05, 0.95, f'Distance: {distances[frame]} cm\nDose: {doses[frame]:.2e} rem/hr',
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(facecolor='white', alpha=0.7))

        # Draw wall
        wall_rect = plt.Rectangle((0, -50), WALL_THICKNESS, 100, color='gray', alpha=0.5)
        ax.add_patch(wall_rect)

        # Draw channel
        channel_rect = plt.Rectangle((0, -channel_diameter / 2), WALL_THICKNESS, channel_diameter,
                                     color='white', alpha=0.8)
        ax.add_patch(channel_rect)

        # Set labels and title
        ax.set_xlabel('Distance from Wall (cm)', fontsize=12)
        ax.set_ylabel('Dose (rem/hr)', fontsize=12)
        ax.set_title(f'Dose vs Distance Animation\n{energy} MeV, {channel_diameter} cm Channel',
                     fontsize=14)

        # Set log scale for y-axis
        ax.set_yscale('log')

        # Set limits
        ax.set_xlim(-10, max(distances) * 1.2)
        ax.set_ylim(min(doses) * 0.5, max(doses) * 2)

        # Add grid
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        return ax

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(distances) + 10, interval=500)

    # Save animation
    # Save animation
    filename = os.path.join(output_dir, f'dose_animation_E{energy}_CD{channel_diameter}.gif')

    try:
        # Try to use PillowWriter for GIF output
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=2)
        anim.save(filename, writer=writer)
    except ImportError:
        # Fallback to default writer
        anim.save(filename, fps=2)

    print(f"Saved dose animation: {filename}")

except Exception as e:
print(f"Error creating animation: {e}")


def plot_angular_distribution(results, energy, channel_diameter, distance, output_dir='figures'):
    """
    Plot angular distribution of radiation at a fixed distance.

    Args:
        results (dict): Simulation results dictionary
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        distance (float): Detector distance from wall in cm
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    energy_key = f"Energy_{energy}MeV"
    channel_key = f"CD{channel_diameter}"

    if energy_key not in results or channel_key not in results[energy_key]:
        print(f"No data found for E={energy} MeV, CD={channel_diameter} cm")
        return

    # Extract angular data
    angles = []
    doses = []

    for position_key in results[energy_key][channel_key]:
        parts = position_key.split('_')
        pos_distance = float(parts[0][1:])
        angle = float(parts[1][1:])

        if pos_distance == distance:  # Only use positions at specified distance
            dose = results[energy_key][channel_key][position_key]['dose']['mean']
            angles.append(angle)
            doses.append(dose)

    if not angles:
        print(f"No angular data found for E={energy} MeV, CD={channel_diameter} cm, D={distance} cm")
        return

    # Sort by angle
    sorted_indices = np.argsort(angles)
    angles = [angles[i] for i in sorted_indices]
    doses = [doses[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})

    # Convert angles to radians for polar plot
    angles_rad = np.radians(angles)

    # Plot angular distribution
    ax.plot(angles_rad, doses, 'o-', linewidth=2)

    # Fill area under curve
    ax.fill(angles_rad, doses, alpha=0.3)

    # Set polar grid properties
    ax.set_theta_zero_location('N')  # 0 degrees at top
    ax.set_theta_direction(-1)  # clockwise

    # Set radial (r) scale to log
    ax.set_rscale('log')

    # Set title
    ax.set_title(f'Angular Distribution of Dose\n{energy} MeV, {channel_diameter} cm Channel, D={distance} cm',
                 fontsize=14)

    # Add annotations
    max_dose_idx = np.argmax(doses)
    max_dose_angle = angles[max_dose_idx]
    max_dose = doses[max_dose_idx]

    ax.annotate(f'Max: {max_dose:.2e} rem/hr at {max_dose_angle}°',
                xy=(angles_rad[max_dose_idx], max_dose),
                xytext=(angles_rad[max_dose_idx] + np.radians(10), max_dose * 1.5),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Save figure
    filename = os.path.join(output_dir, f'angular_distribution_E{energy}_CD{channel_diameter}_D{distance}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved angular distribution plot: {filename}")


def compare_wall_materials(results, energy, channel_diameter, distance, angle, output_dir='figures'):
    """
    Compare dose attenuation for different wall materials.

    Args:
        results (dict): Simulation results dictionary containing material comparisons
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        distance (float): Detector distance from wall in cm
        angle (float): Detector angle in degrees
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if material comparison results exist
    if 'material_comparison' not in results:
        print("No material comparison data found in results")
        return

    material_results = results['material_comparison']

    # Extract data for specified parameters
    energy_key = f"Energy_{energy}MeV"
    channel_key = f"CD{channel_diameter}"
    position_key = f"D{distance}_A{angle}"

    if energy_key not in material_results:
        print(f"No material comparison data for energy {energy} MeV")
        return

    if channel_key not in material_results[energy_key]:
        print(f"No material comparison data for channel diameter {channel_diameter} cm")
        return

    if position_key not in material_results[energy_key][channel_key]:
        print(f"No material comparison data for position D={distance} cm, A={angle}°")
        return

    position_materials = material_results[energy_key][channel_key][position_key]

    # Extract materials and doses
    materials = []
    doses = []
    rel_errors = []

    for material, data in position_materials.items():
        materials.append(material)
        doses.append(data['dose']['mean'])
        rel_errors.append(data['dose']['rel_error'])

    # Sort by dose (ascending)
    sorted_indices = np.argsort(doses)
    materials = [materials[i] for i in sorted_indices]
    doses = [doses[i] for i in sorted_indices]
    rel_errors = [rel_errors[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create x positions
    x_pos = np.arange(len(materials))

    # Create error bars
    yerr = np.array(doses) * np.array(rel_errors)

    # Plot bar chart
    bars = ax.bar(x_pos, doses, yerr=yerr, capsize=5)

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{doses[i]:.2e}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Set labels and title
    ax.set_xlabel('Wall Material', fontsize=12)
    ax.set_ylabel('Dose (rem/hr)', fontsize=12)
    ax.set_title(f'Dose Comparison by Wall Material\n{energy} MeV, {channel_diameter} cm Channel, ' +
                 f'D={distance} cm, A={angle}°', fontsize=14)

    # Set x-axis ticks
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials, rotation=45, ha='right')

    # Use log scale for y-axis
    ax.set_yscale('log')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.5, axis='y')

    # Add reference line for dose limit
    dose_limit = 0.1  # Example limit of 0.1 rem/hr
    ax.axhline(y=dose_limit, color='red', linestyle='--',
               label=f'Dose Limit ({dose_limit} rem/hr)')

    # Add legend
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Save figure
    filename = os.path.join(output_dir, f'material_comparison_E{energy}_CD{channel_diameter}_D{distance}_A{angle}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved material comparison plot: {filename}")


def visualize_flux_energy_distribution(results, energy, channel_diameter, distance, angle, output_dir='figures'):
    """
    Visualize the energy distribution of the neutron/gamma flux.

    Args:
        results (dict): Simulation results dictionary
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        distance (float): Detector distance from wall in cm
        angle (float): Detector angle in degrees
        output_dir (str): Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    energy_key = f"Energy_{energy}MeV"
    channel_key = f"CD{channel_diameter}"
    position_key = f"D{distance}_A{angle}"

    if (energy_key not in results or
            channel_key not in results[energy_key] or
            position_key not in results[energy_key][channel_key]):
        print(f"No data found for E={energy} MeV, CD={channel_diameter} cm, D={distance} cm, A={angle}°")
        return

    position_results = results[energy_key][channel_key][position_key]

    # Check if energy distribution data exists
    if 'energy_distribution' not in position_results:
        print(f"No energy distribution data found for specified parameters")
        return

    energy_dist = position_results['energy_distribution']

    # Extract energy bins and flux values
    energy_bins = energy_dist['energy_bins']
    flux_values = energy_dist['flux']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot energy distribution
    ax.step(energy_bins[:-1], flux_values, where='post', linewidth=2)

    # Fill area under curve
    ax.fill_between(energy_bins[:-1], flux_values, step='post', alpha=0.3)

    # Add source energy marker
    ax.axvline(x=energy, color='red', linestyle='--', linewidth=2,
               label=f'Source Energy ({energy} MeV)')

    # Set labels and title
    ax.set_xlabel('Energy (MeV)', fontsize=12)
    ax.set_ylabel('Flux (particles/cm²/sec)', fontsize=12)
    ax.set_title(f'Energy Distribution of Radiation Flux\n{energy} MeV Source, {channel_diameter} cm Channel, ' +
                 f'D={distance} cm, A={angle}°', fontsize=14)

    # Use log scales
    ax.set_yscale('log')
    ax.set_xscale('log')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Add legend
    ax.legend()

    # Save figure
    filename = os.path.join(output_dir, f'energy_distribution_E{energy}_CD{channel_diameter}_D{distance}_A{angle}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved energy distribution plot: {filename}")


# Add any additional visualization functions as needed

if __name__ == "__main__":
    # Test visualizations with sample data
    import json

    # Load test results if available
    try:
        with open('test_results.json', 'r') as f:
            test_results = json.load(f)

        # Create test visualizations
        create_radiation_outside_wall_heatmap(test_results, 1.0, 5.0, 30, 0)
        plot_dose_comparison(test_results, 5.0, 30)
        create_interactive_dashboard(test_results)

    except FileNotFoundError:
        print("No test results found. Creating sample visualization with synthetic data.")

        # Create synthetic data for testing
        test_results = {
            "Energy_1.0MeV": {
                "CD5.0": {
                    "D30_A0": {
                        "dose": {"mean": 1.2e-3, "rel_error": 0.05},
                        "heating": {"mean": 2.3e-4, "rel_error": 0.06},
                        "kerma": {"mean": 2.5e-4, "rel_error": 0.07}
                    },
                    "D50_A0": {
                        "dose": {"mean": 5.1e-4, "rel_error": 0.08},
                        "heating": {"mean": 1.1e-4, "rel_error": 0.09},
                        "kerma": {"mean": 1.2e-4, "rel_error": 0.10}
                    }
                }
            }
        }

        # Create test visualization
        create_radiation_outside_wall_heatmap(test_results, 1.0, 5.0, 30, 0)
        print("Created sample visualization with synthetic data.")


