#!/usr/bin/env python
import os
import sys
import time
import json
import argparse
import numpy as np
import openmc
from pathlib import Path
import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
import traceback

# Import local modules
import config
from logging_utils import setup_logging
from geometry import create_geometry
from materials import create_materials
from source import create_source
from tally import create_tallies
from simulation import run_simulation, resume_simulation
from dose import calculate_dose
from spectrum_analysis import analyze_spectra
from visualization import (create_geometry_plot, create_dose_heatmap,
                           plot_dose_vs_angle, plot_flux_spectra)
from weight_windows import generate_weight_windows
from ww_analysis import analyze_weight_windows
from ml_analysis import train_dose_predictor
from error_analysis import ErrorAnalysis
from analysis_report import generate_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gamma-ray shielding simulation')

    parser.add_argument('--config', type=str, default='simulation_config.json',
                        help='Configuration file path')
    parser.add_argument('--particles', type=int, default=None,
                        help='Number of particles to simulate')
    parser.add_argument('--batches', type=int, default=None,
                        help='Number of batches')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous simulation')
    parser.add_argument('--geometry-only', action='store_true',
                        help='Only create geometry plots without running simulation')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Only run analysis on existing results')

    return parser.parse_args()


def load_configuration(config_file):
    """
    Load configuration from file.

    Parameters:
    -----------
    config_file : str
        Path to configuration file

    Returns:
    --------
    dict
        Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cfg
    except Exception as e:
        logging.error(f"Error loading configuration file: {str(e)}")
        sys.exit(1)


def setup_output_directory(output_dir=None):
    """
    Set up the output directory for simulation results.

    Parameters:
    -----------
    output_dir : str, optional
        Path to output directory

    Returns:
    --------
    Path
        Path object for the output directory
    """
    if output_dir:
        output_path = Path(output_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results_{timestamp}")

    output_path.mkdir(exist_ok=True, parents=True)
    return output_path


def save_configuration(config_dict, output_dir):
    """
    Save the configuration to the output directory.

    Parameters:
    -----------
    config_dict : dict
        Configuration dictionary
    output_dir : Path
        Output directory path
    """
    config_file = output_dir / "simulation_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logging.info(f"Configuration saved to {config_file}")


def create_simulation_model(cfg):
    """
    Create an OpenMC model based on configuration.

    Parameters:
    -----------
    cfg : dict
        Configuration dictionary

    Returns:
    --------
    tuple
        OpenMC model components (materials, geometry, source, tallies)
    """
    # Create materials
    logging.info("Creating materials...")
    materials = create_materials(
        concrete_composition=cfg['materials']['concrete'],
        air_composition=cfg['materials']['air']
    )

    # Create geometry
    logging.info("Creating geometry...")
    geometry = create_geometry(
        wall_thickness=cfg['geometry']['wall_thickness'],
        channel_diameter=cfg['geometry']['channel_diameter'],
        source_distance=cfg['geometry']['source_distance'],
        detector_distances=cfg['geometry']['detector_distances'],
        detector_angles=cfg['geometry']['detector_angles'],
        detector_diameter=cfg['geometry']['detector_diameter'],
        materials=materials
    )

    # Create source
    logging.info("Creating radiation source...")
    source = create_source(
        energy_range=cfg['source']['energy_range'],
        position=cfg['source']['position'],
        channel_position=cfg['geometry']['channel_position'],
        channel_radius=cfg['geometry']['channel_diameter'] / 2,
        use_bias=cfg['source'].get('use_bias', True)
    )

    # Create tallies
    logging.info("Setting up tallies...")
    tallies = create_tallies(
        detector_positions=cfg['tallies']['detector_positions'],
        mesh_bounds=cfg['tallies']['mesh_bounds'],
        mesh_dimension=cfg['tallies']['mesh_dimension'],
        energy_bins=cfg['tallies']['energy_bins']
    )

    return materials, geometry, source, tallies


def run_complete_simulation(cfg, output_dir, resume=False):
    """
    Run a complete simulation workflow.

    Parameters:
    -----------
    cfg : dict
        Configuration dictionary
    output_dir : Path
        Output directory path
    resume : bool
        Whether to resume from a previous simulation

    Returns:
    --------
    dict
        Simulation results
    """
    # Create model components
    materials, geometry, source, tallies = create_simulation_model(cfg)

    # Set up the simulation settings
    settings = openmc.Settings()
    settings.batches = cfg['simulation']['batches']
    settings.particles = cfg['simulation']['particles']
    settings.run_mode = 'fixed source'
    settings.source = source
    settings.output = {'tallies': True, 'summary': True}

    # Create geometry plots for verification
    logging.info("Creating geometry visualization...")
    create_geometry_plot(geometry, output_dir)

    # Initialize model
    model = openmc.model.Model(geometry, materials, settings, tallies)
    model_file = output_dir / "model.xml"
    model.export_to_xml(directory=output_dir)

    # Generate weight windows if enabled
    if cfg.get('weight_windows', {}).get('enabled', False):
        logging.info("Generating weight windows...")
        ww_params = cfg.get('weight_windows', {})
        weight_windows = generate_weight_windows(
            model,
            ww_params.get('mesh_bounds'),
            ww_params.get('mesh_dimension'),
            output_dir
        )
        logging.info("Weight windows generated successfully")

    # Run or resume simulation
    if resume:
        logging.info("Resuming simulation from previous run...")
        results = resume_simulation(output_dir, settings.batches)
    else:
        logging.info(f"Running simulation with {settings.particles} particles per batch, {settings.batches} batches...")
        results = run_simulation(model, output_dir)

    logging.info("Simulation completed successfully")
    return results

def run_analysis(cfg, output_dir, simulation_results=None):
    """
    Run comprehensive analysis on simulation results.

    Parameters:
    -----------
    cfg : dict
        Configuration dictionary
    output_dir : Path
        Output directory path
    simulation_results : dict, optional
        Simulation results if already loaded

    Returns:
    --------
    dict
        Analysis results
    """
    analysis_results = {}

    # Calculate dose for all detector positions
    logging.info("Calculating dose at all detector positions...")
    dose_results = calculate_dose(
        output_dir,
        cfg['geometry']['detector_distances'],
        cfg['geometry']['detector_angles']
    )
    analysis_results['dose'] = dose_results

    # Create dose heatmaps
    logging.info("Creating dose distribution heatmaps...")
    heatmap_paths = create_dose_heatmap(
        output_dir,
        cfg['visualization'].get('heatmap_resolution', [200, 200]),
        cfg['visualization'].get('enhanced_visualization', True)
    )
    analysis_results['heatmaps'] = heatmap_paths

    # Plot dose vs angle
    logging.info("Creating dose vs angle plots...")
    angle_plot_paths = plot_dose_vs_angle(
        dose_results,
        cfg['geometry']['detector_distances'],
        cfg['geometry']['detector_angles'],
        cfg['source']['energy_range'],
        cfg['geometry']['channel_diameter'],
        output_dir
    )
    analysis_results['angle_plots'] = angle_plot_paths

    # Analyze energy spectra
    logging.info("Analyzing energy spectra...")
    spectra_results = analyze_spectra(
        output_dir,
        cfg['tallies']['energy_bins']
    )
    analysis_results['spectra'] = spectra_results

    # Plot flux spectra
    logging.info("Creating flux spectra plots...")
    flux_plot_paths = plot_flux_spectra(
        spectra_results,
        cfg['geometry']['detector_distances'],
        output_dir
    )
    analysis_results['flux_plots'] = flux_plot_paths

    # Analyze weight windows if enabled
    if cfg.get('weight_windows', {}).get('enabled', False):
        logging.info("Analyzing weight windows performance...")
        ww_analysis = analyze_weight_windows(output_dir)
        analysis_results['weight_windows'] = ww_analysis

    # Error analysis
    logging.info("Performing error analysis...")
    error_analyzer = ErrorAnalysis(output_dir, cfg)
    error_results = error_analyzer.run_full_analysis(
        ['dose_tally', 'flux_tally', 'heating_tally'],
        output_dir / 'error_analysis'
    )
    analysis_results['error_analysis'] = error_results

    # Run ML analysis if enabled
    if cfg.get('ml_analysis', {}).get('enabled', False):
        logging.info("Running machine learning analysis...")
        ml_results = train_dose_predictor(
            dose_results,
            cfg['geometry']['detector_distances'],
            cfg['geometry']['detector_angles'],
            cfg['geometry']['channel_diameter'],
            cfg['source']['energy_range'],
            output_dir
        )
        analysis_results['ml_results'] = ml_results

    # Generate comprehensive report
    logging.info("Generating analysis report...")
    report_path = generate_report(
        cfg,
        analysis_results,
        output_dir
    )
    analysis_results['report'] = str(report_path)

    # Save analysis results
    results_file = output_dir / "analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    logging.info(f"Analysis completed and saved to {results_file}")
    return analysis_results

def main():
    """Main entry point for the simulation."""
    args = parse_args()

    # Set up output directory
    output_dir = setup_output_directory(args.output_dir)

    # Set up logging
    setup_logging(output_dir / "simulation.log")

    # Load configuration
    cfg = load_configuration(args.config)

    # Override config with command line arguments if provided
    if args.particles:
        cfg['simulation']['particles'] = args.particles
    if args.batches:
        cfg['simulation']['batches'] = args.batches

    # Save configuration to output directory
    save_configuration(cfg, output_dir)

    try:
        # Run or analyze simulation based on command line arguments
        if args.geometry_only:
            # Just create geometry and exit
            logging.info("Creating geometry plots only...")
            materials, geometry, _, _ = create_simulation_model(cfg)
            create_geometry_plot(geometry, output_dir)
            logging.info(f"Geometry plots created in {output_dir}")
        elif args.analysis_only:
            # Run analysis on existing results
            logging.info("Running analysis on existing results...")
            run_analysis(cfg, output_dir)
        else:
            # Run full simulation
            start_time = time.time()
            results = run_complete_simulation(cfg, output_dir, resume=args.resume)
            end_time = time.time()

            # Log simulation duration
            duration = end_time - start_time
            logging.info(f"Simulation completed in {duration:.2f} seconds")

            # Run analysis
            run_analysis(cfg, output_dir, results)

        logging.info("Process completed successfully")
        return 0
    except Exception as e:
        logging.error(f"Error in simulation: {str(e)}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

main.py

