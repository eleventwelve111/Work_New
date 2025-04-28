#!/usr/bin/env python3
"""
Geometry definition for gamma radiation shielding simulation.
Creates a concrete wall with an air channel and an ICRU sphere detector.
"""

import openmc
import numpy as np
from config import (
    WALL_THICKNESS, WALL_WIDTH, WALL_HEIGHT,
    CHANNEL_POSITION, DETECTOR_DIAMETER,
    SOURCE_POSITION
)


def create_geometry(materials_dict, channel_diameter=0.5):
    """
    Create the geometry for the shielding simulation.

    Args:
        materials_dict (dict): Dictionary of materials
        channel_diameter (float): Diameter of air channel in cm

    Returns:
        openmc.Geometry: Complete geometry for simulation
    """
    # Create bounding box (world)
    min_bound = -500
    max_bound = 500
    world_region = +openmc.ZCylinder(r=max_bound)
    world = openmc.Cell(name='world')
    world.region = world_region
    world.fill = materials_dict['void']

    # Create concrete wall
    wall_min_x = 0
    wall_max_x = WALL_THICKNESS
    wall_min_y = -WALL_WIDTH / 2
    wall_max_y = WALL_WIDTH / 2
    wall_min_z = -WALL_HEIGHT / 2
    wall_max_z = WALL_HEIGHT / 2

    wall_region = (
            +openmc.XPlane(wall_min_x)
            & -openmc.XPlane(wall_max_x)
            & +openmc.YPlane(wall_min_y)
            & -openmc.YPlane(wall_max_y)
            & +openmc.ZPlane(wall_min_z)
            & -openmc.ZPlane(wall_max_z)
    )

    # Create the air channel through the wall
    # Channel is aligned along x-axis, centered at origin
    channel_radius = channel_diameter / 2.0
    channel_region = openmc.ZCylinder(x0=CHANNEL_POSITION[0],
                                      y0=CHANNEL_POSITION[1],
                                      z0=CHANNEL_POSITION[2],
                                      r=channel_radius)

    # Create wall cell (concrete with air channel)
    wall_cell = openmc.Cell(name='concrete_wall')
    wall_cell.region = wall_region & ~channel_region
    wall_cell.fill = materials_dict['concrete']

    # Create air channel cell
    channel_cell = openmc.Cell(name='air_channel')
    channel_cell.region = wall_region & channel_region
    channel_cell.fill = materials_dict['air']

    # Create void cell for everything else
    universe = openmc.Universe(cells=[world, wall_cell, channel_cell])

    # Create geometry from universe
    geometry = openmc.Geometry(universe)
    geometry.export_to_xml()

    return geometry, universe


def add_detector_to_geometry(universe, materials_dict, detector_position):
    """
    Add a detector sphere to the existing geometry at the specified position.

    Args:
        universe (openmc.Universe): The universe to add detector to
        materials_dict (dict): Dictionary of materials
        detector_position (list): [x, y, z] position of detector center

    Returns:
        openmc.Universe: Updated universe with detector
    """
    # Create detector sphere
    x, y, z = detector_position
    detector_region = openmc.Sphere(x=x, y=y, z=z, r=DETECTOR_DIAMETER / 2)

    # Create detector cell
    detector_cell = openmc.Cell(name='detector')
    detector_cell.region = detector_region
    detector_cell.fill = materials_dict['tissue']

    # Add detector to universe
    universe.add_cell(detector_cell)

    return universe


def create_cone_biasing_region(radius=None):
    """
    Create a conical region from source to channel for variance reduction.

    Args:
        radius (float, optional): Channel radius. If None, calculated based on solid angle.

    Returns:
        openmc.Region: Cone region from source to channel
    """
    if radius is None:
        # Default to the smallest channel radius
        radius = 0.05 / 2  # in cm

    # Calculate cone angle based on channel radius and source distance
    source_to_wall = abs(SOURCE_POSITION[0])
    cone_angle = np.arctan(radius / source_to_wall)

    # Create cone with apex at source, extending through channel
    cone = openmc.model.Cone(x0=SOURCE_POSITION[0],
                             y0=SOURCE_POSITION[1],
                             z0=SOURCE_POSITION[2],
                             r2=np.tan(cone_angle) ** 2,
                             dx=1)

    # Create half-space to limit cone length
    x_plane = openmc.XPlane(SOURCE_POSITION[0])

    # Cone region extending from source
    cone_region = +cone & +x_plane

    return cone_region


def create_visualization_geometry(geometry, channel_diameter):
    """
    Create geometry plots for visualization.

    Args:
        geometry (openmc.Geometry): Geometry to visualize
        channel_diameter (float): Channel diameter in cm

    Returns:
        list: OpenMC plot objects
    """
    plots = []

    # XY plot (top view)
    xy_plot = openmc.Plot()
    xy_plot.filename = f'geometry_xy_channel_{channel_diameter}cm'
    xy_plot.width = (WALL_WIDTH, WALL_HEIGHT)
    xy_plot.pixels = (800, 800)
    xy_plot.origin = (WALL_THICKNESS / 2, 0, 0)
    xy_plot.basis = 'xy'
    xy_plot.color_by = 'material'
    xy_plot.colors = {
        'concrete': (204, 204, 204),
        'air': (255, 255, 255),
        'tissue': (255, 200, 200),
        'void': (240, 240, 255)
    }
    plots.append(xy_plot)

    # XZ plot (side view)
    xz_plot = openmc.Plot()
    xz_plot.filename = f'geometry_xz_channel_{channel_diameter}cm'
    xz_plot.width = (WALL_THICKNESS + 100, WALL_HEIGHT)
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
    plots.append(xz_plot)

    # Close-up plot of channel region
    channel_plot = openmc.Plot()
    channel_plot.filename = f'channel_closeup_{channel_diameter}cm'
    channel_plot.width = (WALL_THICKNESS + 2, channel_diameter * 10)
    channel_plot.pixels = (800, 800)
    channel_plot.origin = (WALL_THICKNESS / 2, 0, 0)
    channel_plot.basis = 'xy'
    channel_plot.color_by = 'material'
    channel_plot.colors = {
        'concrete': (204, 204, 204),
        'air': (255, 255, 255),
        'void': (240, 240, 255)
    }
    plots.append(channel_plot)

    # Create plot file
    plot_file = openmc.Plots(plots)
    plot_file.export_to_xml()

    return plots


if __name__ == "__main__":
    from materials import create_materials

    materials_dict = create_materials()
    for diameter in [0.05, 0.1, 0.5, 1.0]:
        print(f"Creating geometry with channel diameter: {diameter} cm")
        geometry, universe = create_geometry(materials_dict, diameter)
        plots = create_visualization_geometry(geometry, diameter)
        print(f"Created {len(plots)} visualization plots")
