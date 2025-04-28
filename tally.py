#!/usr/bin/env python3
"""
Tally definitions for gamma radiation shielding simulation.
Includes flux, dose rate, kerma, and heating tallies.
"""

import openmc
import numpy as np
from config import (
    ENERGY_BINS, MESH_DIMENSION, MESH_LOWER_LEFT, MESH_UPPER_RIGHT,
    FINE_MESH_DIMENSION, FINE_MESH_LOWER_LEFT, FINE_MESH_UPPER_RIGHT,
    WALL_THICKNESS, DETECTOR_POSITIONS
)


def create_mesh_tallies(mesh_id_prefix=1):
    """
    Create mesh tallies for tracking radiation fluxes throughout the geometry.

    Args:
        mesh_id_prefix (int): Starting ID for mesh tallies

    Returns:
        tuple: (Regular mesh tally, fine mesh tally) for different resolutions
    """
    # Regular mesh for overall radiation field
    mesh = openmc.RegularMesh(mesh_id=mesh_id_prefix)
    mesh.dimension = MESH_DIMENSION
    mesh.lower_left = MESH_LOWER_LEFT
    mesh.upper_right = MESH_UPPER_RIGHT

    # Fine mesh for detailed analysis near channel exit
    fine_mesh = openmc.RegularMesh(mesh_id=mesh_id_prefix + 1)
    fine_mesh.dimension = FINE_MESH_DIMENSION
    fine_mesh.lower_left = FINE_MESH_LOWER_LEFT
    fine_mesh.upper_right = FINE_MESH_UPPER_RIGHT

    # Create mesh filter
    mesh_filter = openmc.MeshFilter(mesh)
    fine_mesh_filter = openmc.MeshFilter(fine_mesh)

    # Energy filter for energy-dependent tallies
    energy_filter = openmc.EnergyFilter(ENERGY_BINS)

    # Create regular mesh tally (photon flux)
    mesh_tally = openmc.Tally(name='mesh_photon_flux')
    mesh_tally.filters = [mesh_filter, energy_filter]
    mesh_tally.scores = ['flux']
    mesh_tally.nuclides = ['total']

    # Create fine mesh tally (photon flux with more detail)
    fine_mesh_tally = openmc.Tally(name='fine_mesh_photon_flux')
    fine_mesh_tally.filters = [fine_mesh_filter, energy_filter]
    fine_mesh_tally.scores = ['flux']
    fine_mesh_tally.nuclides = ['total']

    return mesh_tally, fine_mesh_tally


def create_surface_tallies(wall_thickness=WALL_THICKNESS):
    """
    Create surface tallies at the exit surface of the wall.

    Args:
        wall_thickness (float): Thickness of the wall in cm

    Returns:
        list: Surface tallies
    """
    # Create surface at wall exit
    exit_surface = openmc.XPlane(wall_thickness)

    # Surface filter
    surface_filter = openmc.SurfaceFilter(exit_surface)

    # Energy filter
    energy_filter = openmc.EnergyFilter(ENERGY_BINS)

    # Create surface flux tally
    surface_flux_tally = openmc.Tally(name='wall_exit_flux')
    surface_flux_tally.filters = [surface_filter, energy_filter]
    surface_flux_tally.scores = ['flux']

    # Surface current tally
    surface_current_tally = openmc.Tally(name='wall_exit_current')
    surface_current_tally.filters = [surface_filter, energy_filter]
    surface_current_tally.scores = ['current']

    return [surface_flux_tally, surface_current_tally]


def create_detector_tallies(detector_positions):
    """
    Create tallies for each detector position to evaluate dose at different locations.

    Args:
        detector_positions (list): List of detector position dictionaries

    Returns:
        list: Detector cell tallies
    """
    tallies = []

    # Energy filter
    energy_filter = openmc.EnergyFilter(ENERGY_BINS)

    # Cell filter will be added later when detector cells are created

    for i, detector_info in enumerate(detector_positions):
        # These tallies will be attached to detector cells later
        tally = openmc.Tally(name=f"detector_{i}_dose")
        tally.scores = ['flux', 'heating', 'heating-local', 'heating-photon', 'kerma-photon']
        tallies.append(tally)

    return tallies


def create_flux_to_dose_tally(detector_id=None):
    """
    Create a tally that uses flux-to-dose conversion factors.

    Args:
        detector_id (int, optional): Detector cell ID for cell filter

    Returns:
        openmc.Tally: Dose tally with flux-to-dose conversion
    """
    # Energy filter with specific groups for ANSI/ANS-6.1.1-1977 flux-to-dose conversion
    energy_groups = [
        1e-2, 3e-2, 5e-2, 7e-2, 1e-1, 1.5e-1, 2e-1, 2.5e-1, 3e-1, 3.5e-1, 4e-1,
        4.5e-1, 5e-1, 5.5e-1, 6e-1, 6.5e-1, 7e-1, 8e-1, 1.0, 1.4, 1.8, 2.2,
        2.6, 2.8, 3.25, 3.75, 4.25, 4.75, 5.0, 5.25, 5.75, 6.25, 6.75, 7.5,
        9.0, 11.0, 13.0, 15.0
    ]
    energy_filter = openmc.EnergyFilter(energy_groups)

    # Flux-to-dose conversion factors (rem/hr)/(photon/cm²/s) from ANSI/ANS-6.1.1-1977
    dose_factors = [
        3.96e-6, 5.82e-6, 8.08e-6, 1.03e-5, 1.56e-5, 2.44e-5, 3.51e-5,
        4.63e-5, 5.88e-5, 7.24e-5, 8.68e-5, 1.02e-4, 1.17e-4, 1.33e-4,
        1.48e-4, 1.64e-4, 1.80e-4, 2.05e-4, 2.51e-4, 3.26e-4, 3.86e-4,
        4.41e-4, 4.83e-4, 5.00e-4, 5.32e-4, 5.65e-4, 5.92e-4, 6.15e-4,
        6.26e-4, 6.37e-4, 6.55e-4, 6.72e-4, 6.87e-4, 7.08e-4, 7.42e-4,
        7.98e-4, 8.45e-4, 8.90e-4
    ]
    # Create flux-to-dose function
    dose_function = openmc.data.Function1D(energy_groups, dose_factors)

    # Create tally
    tally = openmc.Tally(name='flux_to_dose')
    tally.filters = [energy_filter]

    # Add cell filter if detector ID is provided
    if detector_id is not None:
        cell_filter = openmc.CellFilter(detector_id)
        tally.filters.append(cell_filter)

    # Add flux score with dose function as multiplier
    tally.scores = ['flux']
    tally.multiplier = dose_function

    return tally


def create_all_tallies(detector_positions):
    """
    Create all tallies needed for the simulation.

    Args:
        detector_positions (list): List of detector position dictionaries

    Returns:
        openmc.Tallies: Collection of all tallies
    """
    tallies_list = []

    # Create mesh tallies
    mesh_tally, fine_mesh_tally = create_mesh_tallies()
    tallies_list.extend([mesh_tally, fine_mesh_tally])

    # Create surface tallies
    surface_tallies = create_surface_tallies()
    tallies_list.extend(surface_tallies)

    # Create detector tallies (will be attached to cells later)
    detector_tallies = create_detector_tallies(detector_positions)
    tallies_list.extend(detector_tallies)

    # Create flux-to-dose tally
    flux_to_dose_tally = create_flux_to_dose_tally()
    tallies_list.append(flux_to_dose_tally)

    # Create tallies object
    tallies = openmc.Tallies(tallies_list)

    return tallies, detector_tallies


if __name__ == "__main__":
    # Test tally creation
    from config import DETECTOR_POSITIONS

    tallies, _ = create_all_tallies(DETECTOR_POSITIONS)
    print(f"Created {len(tallies)} tallies")
    tallies.export_to_xml()
