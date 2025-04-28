#!/usr/bin/env python3
"""
Material definitions for gamma radiation shielding simulation.
Includes concrete (ANSI/ANS-6.4-2006), air, and ICRU tissue equivalent material.
"""

import openmc
import numpy as np


def create_materials():
    """
    Create and return all materials needed for the simulation.

    Returns:
        dict: Dictionary of materials keyed by name
    """
    materials = {}

    # Create concrete material (ANSI/ANS-6.4-2006)
    concrete = openmc.Material(name='concrete')
    # Composition based on ANSI/ANS-6.4-2006 standard
    concrete.add_element('H', 0.01, 'wo')
    concrete.add_element('C', 0.001, 'wo')
    concrete.add_element('O', 0.529107, 'wo')
    concrete.add_element('Na', 0.016, 'wo')
    concrete.add_element('Mg', 0.002, 'wo')
    concrete.add_element('Al', 0.033872, 'wo')
    concrete.add_element('Si', 0.337021, 'wo')
    concrete.add_element('P', 0.000046, 'wo')
    concrete.add_element('S', 0.001283, 'wo')
    concrete.add_element('K', 0.013, 'wo')
    concrete.add_element('Ca', 0.044, 'wo')
    concrete.add_element('Ti', 0.001, 'wo')
    concrete.add_element('Mn', 0.000613, 'wo')
    concrete.add_element('Fe', 0.014, 'wo')
    concrete.add_element('Sr', 0.000007, 'wo')
    concrete.add_element('Zr', 0.000003, 'wo')
    concrete.set_density('g/cm3', 2.3)
    materials['concrete'] = concrete

    # Create air material
    air = openmc.Material(name='air')
    air.add_element('N', 0.7553, 'wo')
    air.add_element('O', 0.2318, 'wo')
    air.add_element('Ar', 0.0128, 'wo')
    air.add_element('C', 0.0001, 'wo')
    air.set_density('g/cm3', 0.001205)
    materials['air'] = air

    # Create ICRU tissue equivalent material (ICRU 44)
    tissue = openmc.Material(name='tissue')
    tissue.add_element('H', 0.101, 'wo')
    tissue.add_element('C', 0.111, 'wo')
    tissue.add_element('N', 0.026, 'wo')
    tissue.add_element('O', 0.762, 'wo')
    tissue.set_density('g/cm3', 1.0)
    materials['tissue'] = tissue

    # Create void
    void = openmc.Material(name='void')
    void.add_element('O', 1.0)  # Dummy composition
    void.set_density('g/cm3', 1e-10)
    materials['void'] = void

    return materials


def create_material_library():
    """
    Create a material library from the materials dictionary.

    Returns:
        openmc.Materials: Material library for OpenMC
    """
    materials_dict = create_materials()
    materials_lib = openmc.Materials(list(materials_dict.values()))
    materials_lib.cross_sections = '/path/to/cross_sections.xml'  # Update with your path

    # Export to XML file
    materials_lib.export_to_xml()

    return materials_lib, materials_dict


if __name__ == "__main__":
    # When run directly, create and export materials
    materials_lib, _ = create_material_library()
    print(f"Created {len(materials_lib)} materials and exported to XML.")
