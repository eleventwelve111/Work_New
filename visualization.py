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
