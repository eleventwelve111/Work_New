import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import openmc
import json
from typing import Dict, List, Tuple, Optional, Union
import warnings


class ErrorAnalysis:
    """Class for analyzing statistical errors and uncertainties in OpenMC simulations."""

    def __init__(self, results_dir, config=None):
        """
        Initialize the error analysis with simulation results.

        Parameters:
        -----------
        results_dir : str or Path
            Directory containing simulation results
        config : dict, optional
            Configuration dictionary
        """
        self.results_dir = Path(results_dir)
        self.config = config or {}
        self.logger = logging.getLogger('error_analysis')

        # Create output directory for error analysis
        self.output_dir = self.results_dir / 'error_analysis'
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load statepoint file if available
        sp_files = list(self.results_dir.glob('statepoint.*.h5'))
        self.statepoint = None
        if sp_files:
            try:
                latest_sp = sorted(sp_files)[-1]
                self.statepoint = openmc.StatePoint(latest_sp)
                self.logger.info(f"Loaded statepoint file: {latest_sp}")
            except Exception as e:
                self.logger.error(f"Failed to load statepoint file: {e}")

    def analyze_tally_errors(self, tally_id=None, tally_name=None):
        """
        Analyze the statistical errors in a specific tally.

        Parameters:
        -----------
        tally_id : int, optional
            Tally ID to analyze
        tally_name : str, optional
            Tally name to analyze

        Returns:
        --------
        dict
            Dictionary of error analysis results
        """
        if self.statepoint is None:
            self.logger.error("No statepoint file available for error analysis")
            return {}

        # Find the tally
        tally = None
        if tally_id is not None:
            tally = self.statepoint.get_tally(id=tally_id)
        elif tally_name is not None:
            for t in self.statepoint.tallies.values():
                if hasattr(t, 'name') and t.name == tally_name:
                    tally = t
                    break

        if tally is None:
            self.logger.error(f"Tally not found: id={tally_id}, name={tally_name}")
            return {}

        self.logger.info(f"Analyzing errors for tally: {tally.id}")

        # Extract mean and relative error
        mean = tally.mean.flatten()
        rel_err = tally.std_dev.flatten() / np.where(mean > 0, mean, 1.0)

        # Calculate error metrics
        max_err = np.max(rel_err)
        min_err = np.min(rel_err)
        mean_err = np.mean(rel_err)
        median_err = np.median(rel_err)

        # Calculate the fraction of bins with relative error < 0.1, 0.05, etc.
        err_lt_10 = np.mean(rel_err < 0.1)
        err_lt_5 = np.mean(rel_err < 0.05)
        err_lt_1 = np.mean(rel_err < 0.01)

        # Determine overall simulation quality
        if mean_err < 0.01 and err_lt_5 > 0.95:
            quality = "Excellent"
        elif mean_err < 0.05 and err_lt_10 > 0.9:
            quality = "Good"
        elif mean_err < 0.1 and err_lt_10 > 0.7:
            quality = "Acceptable"
        else:
            quality = "Poor"

        # Create histogram of relative errors
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rel_err, bins=50, alpha=0.7)
        ax.set_xlabel('Relative Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of Relative Errors for Tally {tally.id}')
        ax.axvline(0.05, color='r', linestyle='--', label='5% Error Threshold')
        ax.axvline(0.1, color='orange', linestyle='--', label='10% Error Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save the figure
        plot_file = self.output_dir / f'error_histogram_tally_{tally.id}.png'
        fig.savefig(plot_file)
        plt.close(fig)

        # Compile results
        results = {
            'tally_id': tally.id,
            'tally_name': getattr(tally, 'name', None),
            'statistics': {
                'max_error': float(max_err),
                'min_error': float(min_err),
                'mean_error': float(mean_err),
                'median_error': float(median_err),
                'fraction_lt_10pct': float(err_lt_10),
                'fraction_lt_5pct': float(err_lt_5),
                'fraction_lt_1pct': float(err_lt_1),
                'quality': quality
            },
            'error_plot': str(plot_file)
        }

        # Save to JSON
        results_file = self.output_dir / f'error_analysis_tally_{tally.id}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def analyze_convergence(self, tally_id=None, tally_name=None):
        """
        Analyze the convergence of tally results over batches.

        Parameters:
        -----------
        tally_id : int, optional
            Tally ID to analyze
        tally_name : str, optional
            Tally name to analyze

        Returns:
        --------
        dict
            Dictionary of convergence analysis results
        """
        # We'll need multiple statepoint files to analyze convergence
        sp_files = sorted(list(self.results_dir.glob('statepoint.*.h5')))
        if len(sp_files) < 3:
            self.logger.warning("Not enough statepoint files for convergence analysis")
            return {}

        batch_numbers = []
        tally_means = []
        tally_rel_errors = []

        for sp_file in sp_files:
            try:
                # Extract batch number from filename
                batch_num = int(sp_file.stem.split('.')[1])

                # Load statepoint file
                sp = openmc.StatePoint(sp_file)

                # Find the tally
                tally = None
                if tally_id is not None:
                    tally = sp.get_tally(id=tally_id)
                elif tally_name is not None:
                    for t in sp.tallies.values():
                        if hasattr(t, 'name') and t.name == tally_name:
                            tally = t
                            break

                if tally is None:
                    continue

                # Get mean and relative error
                mean = np.mean(tally.mean)
                std_dev = np.mean(tally.std_dev)
                rel_error = std_dev / mean if mean > 0 else 0.0

                batch_numbers.append(batch_num)
                tally_means.append(mean)
                tally_rel_errors.append(rel_error)

            except Exception as e:
                self.logger.error(f"Error processing file {sp_file}: {e}")

        if not batch_numbers:
            self.logger.error("No valid batch data found for convergence analysis")
            return {}

        # Sort by batch number
        sorted_indices = np.argsort(batch_numbers)
        batch_numbers = np.array(batch_numbers)[sorted_indices]
        tally_means = np.array(tally_means)[sorted_indices]
        tally_rel_errors = np.array(tally_rel_errors)[sorted_indices]

        # Calculate relative changes
        rel_changes = np.zeros_like(tally_means)
        for i in range(1, len(tally_means)):
            rel_changes[i] = abs(tally_means[i] - tally_means[i - 1]) / max(abs(tally_means[i - 1]), 1e-10)

        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot mean value
        ax1.plot(batch_numbers, tally_means, 'o-')
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Mean Value')
        ax1.set_title(f'Convergence of Mean Value for Tally {tally_id or tally_name}')
        ax1.grid(True, alpha=0.3)

        # Plot relative error
        ax2.plot(batch_numbers, tally_rel_errors * 100, 's-')
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('Relative Error (%)')
        ax2.set_title('Convergence of Relative Error')
        ax2.grid(True, alpha=0.3)

        # Add reference line at 5% error
        ax2.axhline(5, color='r', linestyle='--', label='5% Error Threshold')
        ax2.legend()

        # Save the figure
        plot_file = self.output_dir / f'convergence_tally_{tally_id or tally_name}.png'
        fig.tight_layout()
        fig.savefig(plot_file)
        plt.close(fig)

        # Compile results
        results = {
            'tally_id': tally_id,
            'tally_name': tally_name,
            'batches': batch_numbers.tolist(),
            'means': tally_means.tolist(),
            'rel_errors': tally_rel_errors.tolist(),
            'rel_changes': rel_changes.tolist(),
            'final_rel_error': float(tally_rel_errors[-1]),
            'final_rel_change': float(rel_changes[-1]),
            'converged': tally_rel_errors[-1] < 0.05 and rel_changes[-1] < 0.02,
            'convergence_plot': str(plot_file)
        }

        # Save to JSON
        results_file = self.output_dir / f'convergence_analysis_tally_{tally_id or tally_name}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_full_analysis(self, tally_ids_or_names, output_dir=None):
        """
        Run a comprehensive error analysis on multiple tallies.

        Parameters:
        -----------
        tally_ids_or_names : list
            List of tally IDs or names to analyze
        output_dir : str or Path, optional
            Directory to save results

        Returns:
        --------
        dict
            Dictionary of all analysis results
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True, parents=True)

        results = {}

        for tally_id_or_name in tally_ids_or_names:
            self.logger.info(f"Running error analysis for tally {tally_id_or_name}")

            # Determine if it's an ID or name
            if isinstance(tally_id_or_name, int) or (isinstance(tally_id_or_name, str) and tally_id_or_name.isdigit()):
                tally_id = int(tally_id_or_name)
                tally_name = None
            else:
                tally_id = None
                tally_name = tally_id_or_name

            # Run error analysis
            error_results = self.analyze_tally_errors(tally_id, tally_name)

            # Run convergence analysis
            convergence_results = self.analyze_convergence(tally_id, tally_name)

            # Store results
            tally_key = str(tally_id) if tally_id is not None else tally_name
            results[tally_key] = {
                'error': error_results,
                'convergence': convergence_results
            }

        # Save overall results
        results_file = self.output_dir / 'error_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Error analysis completed. Results saved to {results_file}")

        return results


# Additional functions to be added:

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def propagate_uncertainties(results, energy_kev, channel_diameter):
    """
    Perform uncertainty propagation for dose calculations and quantify confidence intervals.
    Parameters:
        results: Simulation results dictionary
        energy_kev: Energy in keV
        channel_diameter: Channel diameter in cm
    Returns:
        dict: Uncertainty analysis results
    """
    # Extract relevant data
    dose_data = results['dose_data']
    # Initialize uncertainty analysis
    uncertainty_analysis = {
        'energy_kev': energy_kev,
        'channel_diameter': channel_diameter,
        'statistical_uncertainty': {},
        'systematic_uncertainty': {},
        'combined_uncertainty': {},
        'confidence_intervals': {}
    }
    # 1. Statistical uncertainty from Monte Carlo simulation
    for distance in dose_data:
        if distance == 'metadata':
            continue
        uncertainty_analysis['statistical_uncertainty'][distance] = {}
        for angle in dose_data[distance]:
            if angle == 'spectrum':
                continue
            dose_info = dose_data[distance][angle]
            # Extract statistical uncertainties if available
            if 'kerma_uncertainty' in dose_info:
                rel_uncertainty = dose_info['kerma_uncertainty'] / dose_info['kerma'] if dose_info['kerma'] > 0 else 0
                uncertainty_analysis['statistical_uncertainty'][distance][angle] = {
                    'relative': rel_uncertainty,
                    'absolute': dose_info['kerma_uncertainty']
                }
            elif 'dose_uncertainty' in dose_info:
                rel_uncertainty = dose_info['dose_uncertainty'] / dose_info['dose'] if dose_info['dose'] > 0 else 0
                uncertainty_analysis['statistical_uncertainty'][distance][angle] = {
                    'relative': rel_uncertainty,
                    'absolute': dose_info['dose_uncertainty']
                }
                else:
                # Estimate uncertainty based on number of particles if available
                if 'num_particles' in results and results['num_particles'] > 0:
                    # Uncertainty scales as 1/√N
                    est_rel_uncertainty = 1.0 / np.sqrt(results['num_particles'])
                    # Get dose value
                    dose_value = 0
                    if 'kerma' in dose_info:
                        dose_value = dose_info['kerma']
                    elif 'dose' in dose_info:
                        dose_value = dose_info['dose']
                    uncertainty_analysis['statistical_uncertainty'][distance][angle] = {
                        'relative': est_rel_uncertainty,
                        'absolute': est_rel_uncertainty * dose_value,
                        'estimated': True
                    }

            # 2. Systematic uncertainties
            # Sources: cross-section data, material compositions, geometry approximations
            # Cross-section uncertainty (typical range 1-10%)
        cross_section_uncertainty = 0.05  # 5% relative uncertainty
        # Material composition uncertainty (typical range 1-5%)
        material_uncertainty = 0.03  # 3% relative uncertainty
        # Geometry approximation uncertainty (typical range 1-3%)
        geometry_uncertainty = 0.02  # 2% relative uncertainty
        # Energy-dependent component (higher uncertainty at lower energies)
        energy_factor = max(0.01, 0.2 / np.sqrt(energy_kev / 100))  # Decreases with energy

        # Combine systematic uncertainties (in quadrature)
        systematic_rel_uncertainty = np.sqrt(
            cross_section_uncertainty ** 2 +
            material_uncertainty ** 2 +
            geometry_uncertainty ** 2 +
            energy_factor ** 2
        )

        # Apply to all positions
        for distance in dose_data:
            if distance == 'metadata':
                continue
            uncertainty_analysis['systematic_uncertainty'][distance] = {}
            for angle in dose_data[distance]:
                if angle == 'spectrum':
                    continue
                dose_info = dose_data[distance][angle]
                # Get dose value
                dose_value = 0
                if 'kerma' in dose_info:
                    dose_value = dose_info['kerma']
                elif 'dose' in dose_info:
                    dose_value = dose_info['dose']
                uncertainty_analysis['systematic_uncertainty'][distance][angle] = {
                    'relative': systematic_rel_uncertainty,
                    'absolute': systematic_rel_uncertainty * dose_value
                }

        # 3. Combined uncertainty and confidence intervals
        for distance in dose_data:
            if distance == 'metadata':
                continue
            uncertainty_analysis['combined_uncertainty'][distance] = {}
            uncertainty_analysis['confidence_intervals'][distance] = {}
            for angle in dose_data[distance]:
                if angle == 'spectrum':
                    continue
                if (distance in uncertainty_analysis['statistical_uncertainty'] and
                        angle in uncertainty_analysis['statistical_uncertainty'][distance] and
                        distance in uncertainty_analysis['systematic_uncertainty'] and
                        angle in uncertainty_analysis['systematic_uncertainty'][distance]):
                    stat_unc = uncertainty_analysis['statistical_uncertainty'][distance][angle]['relative']
                    sys_unc = uncertainty_analysis['systematic_uncertainty'][distance][angle]['relative']
                    # Combined relative uncertainty (in quadrature)
                    combined_rel_uncertainty = np.sqrt(stat_unc ** 2 + sys_unc ** 2)
                    # Get dose value
                    dose_info = dose_data[distance][angle]
                    dose_value = 0
                    if 'kerma' in dose_info:
                        dose_value = dose_info['kerma']
                    elif 'dose' in dose_info:
                        dose_value = dose_info['dose']
                    combined_abs_uncertainty = combined_rel_uncertainty * dose_value
                    # Store combined uncertainty
                    uncertainty_analysis['combined_uncertainty'][distance][angle] = {
                        'relative': combined_rel_uncertainty,
                        'absolute': combined_abs_uncertainty
                    }
                    # 95% confidence interval (assuming normal distribution)
                    ci_95_lower = dose_value - 1.96 * combined_abs_uncertainty
                    ci_95_upper = dose_value + 1.96 * combined_abs_uncertainty
                    # 99% confidence interval
                    ci_99_lower = dose_value - 2.576 * combined_abs_uncertainty
                    ci_99_upper = dose_value + 2.576 * combined_abs_uncertainty
                    uncertainty_analysis['confidence_intervals'][distance][angle] = {
                        'value': dose_value,
                        'ci_95': [max(0, ci_95_lower), ci_95_upper],
                        'ci_99': [max(0, ci_99_lower), ci_99_upper]
                    }

        return uncertainty_analysis

    def analyze_simulation_convergence(tally_results, num_batches):
        """
        Analyze convergence of simulation results with increasing number of particle batches.
        Parameters:
            tally_results: Dictionary of tally results at different batch numbers
            num_batches: Array of batch numbers
        Returns:
            dict: Convergence analysis results
        """
        # Extract tally values and relative errors
        tally_values = np.array([result['value'] for result in tally_results])
        rel_errors = np.array([result['rel_error'] for result in tally_results])

        # Calculate convergence rate
        # Expected: rel_error ~ 1/√N
        log_batches = np.log(num_batches)
        log_errors = np.log(rel_errors)

        # Linear fit to log-log data
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_batches, log_errors)

        # Theoretical convergence rate is -0.5
        theoretical_slope = -0.5
        deviation_from_theory = slope - theoretical_slope

        # Figure of Merit (FOM) = 1/(rel_error²·T)
        # FOM should be approximately constant if simulation is efficient
        # T ~ N for constant time per particle
        fom = 1.0 / (rel_errors ** 2 * num_batches)
        fom_variation = np.std(fom) / np.mean(fom)

        # Check for bias by comparing final result with average of last few results
        final_value = tally_values[-1]
        last_values = tally_values[int(0.8 * len(tally_values)):]  # Last 20% of results
        expected_value = np.mean(last_values)
        bias = (final_value - expected_value) / expected_value if expected_value != 0 else 0

        # Standard deviation of relative statistical error of the mean
        statistical_error_std = np.std([result.get('rel_error', 0) for result in tally_results])

        # Results
        convergence_results = {
            'num_batches': num_batches.tolist(),
            'tally_values': tally_values.tolist(),
            'rel_errors': rel_errors.tolist(),
            'convergence_rate': {
                'slope': float(slope),
                'theoretical_slope': theoretical_slope,
                'deviation': float(deviation_from_theory),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value)
            },
            'figure_of_merit': {
                'values': fom.tolist(),
                'mean': float(np.mean(fom)),
                'variation': float(fom_variation)
            },
            'bias_analysis': {
                'final_value': float(final_value),
                'expected_value': float(expected_value),
                'relative_bias': float(bias)
            },
            'statistical_quality': {
                'error_std': float(statistical_error_std),
                'converged': float(rel_errors[-1]) < 0.05  # Consider converged if relative error < 5%
            }
        }

        return convergence_results

    def plot_uncertainty_analysis(uncertainty_analysis, filename=None):
        """
        Generate comprehensive plots for uncertainty analysis results.
        Parameters:
            uncertainty_analysis: Dictionary of uncertainty analysis results
            filename: Optional filename to save the plot
        Returns:
            matplotlib.figure.Figure: Figure with uncertainty plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract data
        distances = []
        stat_uncs = []
        sys_uncs = []
        combined_uncs = []
        for distance in uncertainty_analysis['statistical_uncertainty']:
            for angle in uncertainty_analysis['statistical_uncertainty'][distance]:
                if angle == '0':  # Only consider angle = 0 for simplicity
                    distances.append(int(distance))
                    stat_uncs.append(uncertainty_analysis['statistical_uncertainty'][distance][angle]['relative'])
                    sys_uncs.append(uncertainty_analysis['systematic_uncertainty'][distance][angle]['relative'])
                    combined_uncs.append(uncertainty_analysis['combined_uncertainty'][distance][angle]['relative'])

        # Sort by distance
        sort_idx = np.argsort(distances)
        distances = np.array(distances)[sort_idx]
        stat_uncs = np.array(stat_uncs)[sort_idx]
        sys_uncs = np.array(sys_uncs)[sort_idx]
        combined_uncs = np.array(combined_uncs)[sort_idx]

        # Plot 1: Relative uncertainties by distance
        ax = axes[0, 0]
        ax.plot(distances, stat_uncs * 100, 'o-', label='Statistical')
        ax.plot(distances, sys_uncs * 100, 's-', label='Systematic')
        ax.plot(distances, combined_uncs * 100, '^-', label='Combined')
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Relative Uncertainty (%)')
        ax.set_title('Uncertainty Components vs. Distance')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Uncertainty components breakdown
        ax = axes[0, 1]
        components = ['Statistical', 'Cross-section', 'Materials', 'Geometry', 'Energy-dependent']
        # These values should match those used in the propagate_uncertainties function
        values = [
            np.mean(stat_uncs) * 100,  # Average statistical uncertainty
            5.0,  # Cross-section uncertainty (5%)
            3.0,  # Material uncertainty (3%)
            2.0,  # Geometry uncertainty (2%)
            max(1.0, 20.0 / np.sqrt(uncertainty_analysis['energy_kev'] / 100))  # Energy factor
        ]
        ax.bar(components, values)
        ax.set_ylabel('Contribution (%)')
        ax.set_title('Uncertainty Components Breakdown')
        ax.grid(True, axis='y', alpha=0.3)

        # Plot 3: Confidence intervals for a specific position
        ax = axes[1, 0]
        # Select a reference distance (e.g., 30 cm)
        ref_distance = '30'
        if ref_distance in uncertainty_analysis['confidence_intervals'] and '0' in \
                uncertainty_analysis['confidence_intervals'][ref_distance]:
            ci_data = uncertainty_analysis['confidence_intervals'][ref_distance]['0']
            value = ci_data['value']
            ci_95 = ci_data['ci_95']
            ci_99 = ci_data['ci_99']
            ax.errorbar([1], [value], yerr=[[value - ci_95[0]], [ci_95[1] - value]],
                        fmt='o', capsize=5, label='95% CI')
            ax.errorbar([2], [value], yerr=[[value - ci_99[0]], [ci_99[1] - value]],
                        fmt='s', capsize=5, label='99% CI')
            ax.set_xlim(0, 3)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['95% CI', '99% CI'])
            ax.set_ylabel('Dose (rem/hr)')
            ax.set_title(f'Confidence Intervals at {ref_distance} cm')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Plot 4: Relative uncertainty vs. distance (log-log scale)
        ax = axes[1, 1]
        ax.loglog(distances, stat_uncs, 'o-', label='Statistical')
        # Theoretical 1/√r line
        ref_dist = distances[0]
        ref_unc = stat_uncs[0]
        theoretical_uncs = ref_unc * np.sqrt(ref_dist / np.array(distances))
        ax.loglog(distances, theoretical_uncs, 'k--', label='1/√r theory')
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Relative Uncertainty')
        ax.set_title('Statistical Uncertainty Scaling')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        return fig

    # Add a method to integrate with the ErrorAnalysis class
    def integrate_uncertainty_propagation_method(ErrorAnalysis):
        """
        Integrates the uncertainty propagation method into the ErrorAnalysis class.
        This allows for a more comprehensive analysis of simulation uncertainties.
        """

        def analyze_uncertainty_propagation(self, results, energy_kev=None, channel_diameter=None):
            """
            Analyze uncertainty propagation for simulation results.

            Parameters:
            -----------
            results : dict
                Simulation results dictionary
            energy_kev : float, optional
                Energy in keV (will try to extract from results if not provided)
            channel_diameter : float, optional
                Channel diameter in cm (will try to extract from config if not provided)

            Returns:
            --------
            dict
                Dictionary of uncertainty analysis results
            """
            # Try to extract energy from results if not provided
            if energy_kev is None:
                if 'dose_data' in results and 'metadata' in results['dose_data'] and 'energy_kev' in \
                        results['dose_data']['metadata']:
                    energy_kev = results['dose_data']['metadata']['energy_kev']
                else:
                    self.logger.warning(
                        "Energy not provided and could not be extracted from results. Using default 500 keV.")
                    energy_kev = 500

            # Try to extract channel diameter from config if not provided
            if channel_diameter is None:
                if self.config and 'geometry' in self.config and 'channel_diameter' in self.config['geometry']:
                    channel_diameter = self.config['geometry']['channel_diameter']
                else:
                    self.logger.warning(
                        "Channel diameter not provided and could not be extracted from config. Using default 10 cm.")
                    channel_diameter = 10

            # Run uncertainty propagation
            uncertainty_analysis = propagate_uncertainties(results, energy_kev, channel_diameter)

            # Generate plots
            plot_file = self.output_dir / 'uncertainty_analysis.png'
            plot_uncertainty_analysis(uncertainty_analysis, filename=plot_file)

            # Save results
            results_file = self.output_dir / 'uncertainty_propagation_results.json'
            with open(results_file, 'w') as f:
                # Convert numpy values to native Python types for JSON serialization
                cleaned_analysis = self._clean_for_json(uncertainty_analysis)
                json.dump(cleaned_analysis, f, indent=2)

            self.logger.info(f"Uncertainty propagation analysis completed. Results saved to {results_file}")

            return uncertainty_analysis

        def _clean_for_json(self, obj):
            """Helper method to convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: self._clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [self._clean_for_json(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

            # Add the method to the class

        ErrorAnalysis.analyze_uncertainty_propagation = analyze_uncertainty_propagation
        ErrorAnalysis._clean_for_json = _clean_for_json

    def analyze_convergence_with_batch_analysis(ErrorAnalysis):
        """
        Integrates the advanced convergence analysis method into the ErrorAnalysis class.
        This allows for more detailed batch convergence analysis.
        """

        def analyze_advanced_convergence(self, tally_id=None, tally_name=None):
            """
            Perform advanced convergence analysis using batch statistics.

            Parameters:
            -----------
            tally_id : int, optional
                Tally ID to analyze
            tally_name : str, optional
                Tally name to analyze

            Returns:
            --------
            dict
                Dictionary of advanced convergence analysis results
            """
            # We'll need multiple statepoint files to analyze convergence
            sp_files = sorted(list(self.results_dir.glob('statepoint.*.h5')))
            if len(sp_files) < 3:
                self.logger.warning("Not enough statepoint files for advanced convergence analysis")
                return {}

            # Collect tally results from each batch
            tally_results = []
            batch_numbers = []

            for sp_file in sp_files:
                try:
                    # Extract batch number from filename
                    batch_num = int(sp_file.stem.split('.')[1])
                    batch_numbers.append(batch_num)

                    # Load statepoint file
                    sp = openmc.StatePoint(sp_file)

                    # Find the tally
                    tally = None
                    if tally_id is not None:
                        tally = sp.get_tally(id=tally_id)
                    elif tally_name is not None:
                        for t in sp.tallies.values():
                            if hasattr(t, 'name') and t.name == tally_name:
                                tally = t
                                break

                    if tally is None:
                        continue

                    # Get mean and relative error
                    mean = np.mean(tally.mean)
                    std_dev = np.mean(tally.std_dev)
                    rel_error = std_dev / mean if mean > 0 else 0.0

                    tally_results.append({
                        'value': float(mean),
                        'std_dev': float(std_dev),
                        'rel_error': float(rel_error)
                    })

                except Exception as e:
                    self.logger.error(f"Error processing file {sp_file}: {e}")

            if not tally_results:
                self.logger.error("No valid tally results found for advanced convergence analysis")
                return {}

            # Sort by batch number
            sorted_indices = np.argsort(batch_numbers)
            batch_numbers = np.array(batch_numbers)[sorted_indices]
            tally_results = [tally_results[i] for i in sorted_indices]

            # Perform advanced convergence analysis
            convergence_analysis = analyze_simulation_convergence(tally_results, np.array(batch_numbers))

            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: Tally value convergence
            ax = axes[0, 0]
            tally_values = np.array(convergence_analysis['tally_values'])
            batches = np.array(convergence_analysis['num_batches'])
            ax.plot(batches, tally_values, 'o-')
            ax.set_xlabel('Batch Number')
            ax.set_ylabel('Tally Value')
            ax.set_title('Tally Value Convergence')
            ax.grid(True, alpha=0.3)

            # Plot 2: Relative error convergence
            ax = axes[0, 1]
            rel_errors = np.array(convergence_analysis['rel_errors'])
            ax.plot(batches, rel_errors * 100, 's-')
            ax.set_xlabel('Batch Number')
            ax.set_ylabel('Relative Error (%)')
            ax.set_title('Relative Error Convergence')
            ax.grid(True, alpha=0.3)
            # Add 5% threshold line
            ax.axhline(5, color='r', linestyle='--', label='5% Threshold')
            ax.legend()

            # Plot 3: Log-log plot of relative error vs batch number
            ax = axes[1, 0]
            ax.loglog(batches, rel_errors, 'o-', label='Actual')
            # Add theoretical 1/√N line
            first_batch = batches[0]
            first_error = rel_errors[0]
            theoretical_errors = first_error * np.sqrt(first_batch / batches)
            ax.loglog(batches, theoretical_errors, 'k--', label='Theoretical (1/√N)')
            ax.set_xlabel('Batch Number (log scale)')
            ax.set_ylabel('Relative Error (log scale)')
            ax.set_title('Convergence Rate Analysis')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()

            # Plot 4: Figure of Merit
            ax = axes[1, 1]
            fom = np.array(convergence_analysis['figure_of_merit']['values'])
            ax.plot(batches, fom / fom[0], 'o-')  # Normalize to initial value
            ax.set_xlabel('Batch Number')
            ax.set_ylabel('Normalized Figure of Merit')
            ax.set_title('Figure of Merit (FOM) Stability')
            ax.axhline(1.0, color='k', linestyle='--', label='Ideal (constant FOM)')
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()

            # Save the plot
            plot_file = self.output_dir / f'advanced_convergence_tally_{tally_id or tally_name}.png'
            plt.savefig(plot_file)
            plt.close(fig)

            # Add plot path to results
            convergence_analysis['plot_file'] = str(plot_file)

            # Save results to JSON
            results_file = self.output_dir / f'advanced_convergence_tally_{tally_id or tally_name}.json'
            with open(results_file, 'w') as f:
                json.dump(convergence_analysis, f, indent=2)

            self.logger.info(f"Advanced convergence analysis completed. Results saved to {results_file}")

            return convergence_analysis

        # Add the method to the class
        ErrorAnalysis.analyze_advanced_convergence = analyze_advanced_convergence

    def enhance_error_analysis_class():
        """
        Enhance the ErrorAnalysis class with additional methods for uncertainty propagation
        and advanced convergence analysis.
        """
        integrate_uncertainty_propagation_method(ErrorAnalysis)
        analyze_convergence_with_batch_analysis(ErrorAnalysis)

    # If this script is run directly, enhance the ErrorAnalysis class
    enhance_error_analysis_class()

    if __name__ == "__main__":
        import argparse

        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Perform error analysis on OpenMC simulation results')
        parser.add_argument('--results-dir', type=str, required=True, help='Directory containing simulation results')
        parser.add_argument('--config', type=str, help='Configuration file path')
        parser.add_argument('--tally-id', type=int, help='Tally ID to analyze')
        parser.add_argument('--tally-name', type=str, help='Tally name to analyze')
        parser.add_argument('--output-dir', type=str, help='Output directory for analysis results')
        parser.add_argument('--analyze-all', action='store_true', help='Analyze all tallies')
        args = parser.parse_args()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Load configuration if provided
        config = None
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load configuration: {e}")
                config = {}

        # Initialize error analysis
        ea = ErrorAnalysis(args.results_dir, config)

        # Set output directory if provided
        if args.output_dir:
            ea.output_dir = Path(args.output_dir)
            ea.output_dir.mkdir(exist_ok=True, parents=True)

        # Perform analysis
        if args.analyze_all:
            # Try to load statepoint file to get all tallies
            sp_files = list(Path(args.results_dir).glob('statepoint.*.h5'))
            if not sp_files:
                logging.error("No statepoint files found for analysis")
                sys.exit(1)

            try:
                sp = openmc.StatePoint(sp_files[-1])  # Use latest statepoint file
                tally_ids = [t.id for t in sp.tallies.values()]
                results = ea.run_full_analysis(tally_ids)
                logging.info(f"Completed analysis of {len(tally_ids)} tallies")
            except Exception as e:
                logging.error(f"Error during full analysis: {e}")
        else:
            if args.tally_id:
                # Analyze specific tally by ID
                error_results = ea.analyze_tally_errors(tally_id=args.tally_id)
                convergence_results = ea.analyze_convergence(tally_id=args.tally_id)
                advanced_convergence = ea.analyze_advanced_convergence(tally_id=args.tally_id)
                logging.info(f"Completed analysis for tally ID {args.tally_id}")
            elif args.tally_name:
                # Analyze specific tally by name
                error_results = ea.analyze_tally_errors(tally_name=args.tally_name)
                convergence_results = ea.analyze_convergence(tally_name=args.tally_name)
                advanced_convergence = ea.analyze_advanced_convergence(tally_name=args.tally_name)
                logging.info(f"Completed analysis for tally '{args.tally_name}'")
            else:
                logging.error("Either --tally-id, --tally-name, or --analyze-all must be specified")
                sys.exit(1)


