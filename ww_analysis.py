#!/usr/bin/env python3
"""
Analysis of weight window performance and optimization for variance reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
import openmc
import os
import json
from config import DATA_DIR, PLOT_DIR
import time


def analyze_weight_window_performance(results_with_ww, results_without_ww):
    """
    Compare simulation performance with and without weight windows.

    Args:
        results_with_ww (list): Results from simulations with weight windows
        results_without_ww (list): Results from simulations without weight windows

    Returns:
        dict: Performance comparison metrics
    """
    # Extract relevant metrics
    metrics = {
        'with_ww': {
            'rel_errors': [],
            'runtimes': [],
            'fom': []  # Figure of merit
        },
        'without_ww': {
            'rel_errors': [],
            'runtimes': [],
            'fom': []
        }
    }

    # Process results with weight windows
    for result in results_with_ww:
        # Extract runtime if available
        runtime = result.get('runtime', 0)

        # Extract relative errors from detector tallies
        for tally_name, tally_data in result['tallies'].items():
            if 'detector' in tally_name:
                rel_error = np.mean(tally_data['rel_err'])

                # Calculate figure of merit: FOM = 1/(rel_error²·T)
                if runtime > 0 and rel_error > 0:
                    fom = 1.0 / (rel_error ** 2 * runtime)
                else:
                    fom = 0

                metrics['with_ww']['rel_errors'].append(rel_error)
                metrics['with_ww']['runtimes'].append(runtime)
                metrics['with_ww']['fom'].append(fom)

    # Process results without weight windows
    for result in results_without_ww:
        # Extract runtime if available
        runtime = result.get('runtime', 0)

        # Extract relative errors from detector tallies
        for tally_name, tally_data in result['tallies'].items():
            if 'detector' in tally_name:
                rel_error = np.mean(tally_data['rel_err'])

                # Calculate figure of merit: FOM = 1/(rel_error²·T)
                if runtime > 0 and rel_error > 0:
                    fom = 1.0 / (rel_error ** 2 * runtime)
                else:
                    fom = 0

                metrics['without_ww']['rel_errors'].append(rel_error)
                metrics['without_ww']['runtimes'].append(runtime)
                metrics['without_ww']['fom'].append(fom)

    # Calculate average metrics
    metrics['avg_improvement'] = {
        'rel_error': np.mean(metrics['without_ww']['rel_errors']) /
                     np.mean(metrics['with_ww']['rel_errors']) if metrics['with_ww']['rel_errors'] else 0,
        'runtime': np.mean(metrics['with_ww']['runtimes']) /
                   np.mean(metrics['without_ww']['runtimes']) if metrics['without_ww']['runtimes'] else 0,
        'fom': np.mean(metrics['with_ww']['fom']) /
               np.mean(metrics['without_ww']['fom']) if metrics['without_ww']['fom'] else 0
    }

    return metrics


def plot_weight_window_comparison(metrics):
    """
    Create plots comparing performance with and without weight windows.

    Args:
        metrics (dict): Performance comparison metrics

    Returns:
        list: Generated figure objects
    """
    figures = []

    # Create figure for relative error comparison
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Plot relative error histograms
    bins = np.logspace(-3, 0, 20)
    ax1.hist(metrics['with_ww']['rel_errors'], bins=bins, alpha=0.7, label='With Weight Windows')
    ax1.hist(metrics['without_ww']['rel_errors'], bins=bins, alpha=0.7, label='Without Weight Windows')

    # Set log scale
    ax1.set_xscale('log')

    # Add vertical lines for average relative errors
    avg_re_ww = np.mean(metrics['with_ww']['rel_errors'])
    avg_re_no_ww = np.mean(metrics['without_ww']['rel_errors'])
    ax1.axvline(x=avg_re_ww, color='blue', linestyle='--',
                label=f'Avg with WW: {avg_re_ww:.3f}')
    ax1.axvline(x=avg_re_no_ww, color='orange', linestyle='--',
                label=f'Avg without WW: {avg_re_no_ww:.3f}')

    # Add labels and title
    ax1.set_xlabel('Relative Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Comparison of Relative Errors With and Without Weight Windows')

    # Add grid
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend
    ax1.legend()

    # Save figure
    plt.savefig(os.path.join(PLOT_DIR, 'ww_rel_error_comparison.png'), dpi=300, bbox_inches='tight')
    figures.append(fig1)

    # Create figure for figure of merit comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Plot FOM scatter plot
    ax2.scatter(metrics['without_ww']['rel_errors'], metrics['without_ww']['fom'],
                label='Without Weight Windows', alpha=0.7)
    ax2.scatter(metrics['with_ww']['rel_errors'], metrics['with_ww']['fom'],
                label='With Weight Windows', alpha=0.7)

    # Set log scales
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Add labels and title
    ax2.set_xlabel('Relative Error')
    ax2.set_ylabel('Figure of Merit (1/rel_error²·T)')
    ax2.set_title('Figure of Merit vs. Relative Error')

    # Add grid
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)

    # Add legend
    ax2.legend()

    # Save figure
    plt.savefig(os.path.join(PLOT_DIR, 'ww_fom_comparison.png'), dpi=300, bbox_inches='tight')
    figures.append(fig2)

    # Create summary bar chart
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Prepare data
    metrics_names = ['Relative Error\nImprovement', 'Runtime\nRatio', 'FOM\nImprovement']
    metrics_values = [
        metrics['avg_improvement']['rel_error'],
        metrics['avg_improvement']['runtime'],
        metrics['avg_improvement']['fom']
    ]

    # Plot bars
    bars = ax3.bar(metrics_names, metrics_values)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')

    # Add reference line at y=1
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)

    # Add labels and title
    ax3.set_ylabel('Improvement Factor')
    ax3.set_title('Weight Window Performance Metrics')

    # Add grid
    ax3.grid(axis='y', linestyle='--', alpha=0.6)

    # Save figure
    plt.savefig(os.path.join(PLOT_DIR, 'ww_performance_summary.png'), dpi=300, bbox_inches='tight')
    figures.append(fig3)

    return figures


def optimize_weight_windows(settings, simulation_function, test_cases):
    """
    Optimize weight window parameters for a set of test cases.

    Args:
        settings (openmc.Settings): Settings template to modify
        simulation_function: Function to run a simulation with given parameters
        test_cases (list): List of test case configurations

    Returns:
        dict: Optimization results
    """
    results = {
        'test_cases': [],
        'optimal_parameters': {},
        'performance_metrics': {}
    }

    # Define weight window parameters to test
    survival_ratios = [0.25, 0.5, 0.75]
    upper_bound_factors = [2, 5, 10]

    print("Starting weight window optimization...")

    # Test each case with different parameter combinations
    for i, test_case in enumerate(test_cases):
        print(f"Optimizing for test case {i + 1}/{len(test_cases)}")

        case_results = {
            'configuration': test_case,
            'parameter_tests': []
        }

        # Test each parameter combination
        for sr in survival_ratios:
            for ubf in upper_bound_factors:
                print(f"  Testing survival_ratio={sr}, upper_bound_factor={ubf}")

                # Create weight windows with these parameters
                weight_windows = create_weight_windows(
                    test_case.get('channel_diameter', 0.5),
                    survival_ratio=sr,
                    upper_bound_factor=ubf
                )

                # Apply to settings
                test_settings = settings.clone()
                test_settings.weight_windows = weight_windows

                # Time the simulation
                start_time = time.time()

                # Run simulation
                sim_result = simulation_function(
                    test_settings,
                    test_case.get('energy', 1.0),
                    test_case.get('channel_diameter', 0.5)
                )

                # Record elapsed time
                elapsed_time = time.time() - start_time

                # Extract performance metrics
                rel_errors = []
                for tally_name, tally_data in sim_result['tallies'].items():
                    if 'detector' in tally_name:
                        rel_errors.extend(tally_data['rel_err'])

                avg_rel_error = np.mean(rel_errors) if rel_errors else 1.0

                # Calculate figure of merit
                if avg_rel_error > 0 and elapsed_time > 0:
                    fom = 1.0 / (avg_rel_error ** 2 * elapsed_time)
                else:
                    fom = 0

                # Add results
                case_results['parameter_tests'].append({
                    'survival_ratio': sr,
                    'upper_bound_factor': ubf,
                    'runtime': elapsed_time,
                    'avg_rel_error': avg_rel_error,
                    'fom': fom
                })

        # Find optimal parameters for this case
        if case_results['parameter_tests']:
            # Sort by FOM (higher is better)
            sorted_tests = sorted(case_results['parameter_tests'],
                                  key=lambda x: x['fom'], reverse=True)

            optimal = sorted_tests[0]
            case_results['optimal'] = optimal

            print(f"  Optimal parameters for case {i + 1}: "
                  f"survival_ratio={optimal['survival_ratio']}, "
                  f"upper_bound_factor={optimal['upper_bound_factor']}, "
                  f"FOM={optimal['fom']:.2f}")

        results['test_cases'].append(case_results)

    # Determine overall optimal parameters
    if results['test_cases']:
        # Collect all FOMs per parameter combination
        param_foms = {}

        for case in results['test_cases']:
            for test in case['parameter_tests']:
                key = (test['survival_ratio'], test['upper_bound_factor'])
                if key not in param_foms:
                    param_foms[key] = []

                param_foms[key].append(test['fom'])

        # Find combination with highest average FOM
        best_avg_fom = 0
        best_params = None

        for params, foms in param_foms.items():
            avg_fom = np.mean(foms)
            if avg_fom > best_avg_fom:
                best_avg_fom = avg_fom
                best_params = params

        if best_params:
            results['optimal_parameters'] = {
                'survival_ratio': best_params[0],
                'upper_bound_factor': best_params[1],
                'avg_fom': best_avg_fom
            }

            print(f"\nOverall optimal parameters: "
                  f"survival_ratio={best_params[0]}, "
                  f"upper_bound_factor={best_params[1]}, "
                  f"average FOM={best_avg_fom:.2f}")

    return results


def plot_optimization_results(optimization_results):
    """
    Create plots of weight window optimization results.

    Args:
        optimization_results (dict): Results from optimize_weight_windows

    Returns:
        list: Generated figure objects
    """
    figures = []

    # Create figure for parameter performance across test cases
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Extract data
    sr_values = []
    ubf_values = []
    fom_values = []
    rel_error_values = []

    for case in optimization_results['test_cases']:
        for test in case['parameter_tests']:
            sr_values.append(test['survival_ratio'])
            ubf_values.append(test['upper_bound_factor'])
            fom_values.append(test['fom'])
            rel_error_values.append(test['avg_rel_error'])

    # Create scatter plot
    scatter = ax1.scatter(sr_values, ubf_values, c=fom_values,
                          s=100 / np.array(rel_error_values), alpha=0.7,
                          cmap='viridis')

    # Add colorbar
    cbar = fig1.colorbar(scatter, ax=ax1)
    cbar.set_label('Figure of Merit')

    # Mark optimal parameters
    if 'optimal_parameters' in optimization_results:
        opt_sr = optimization_results['optimal_parameters']['survival_ratio']
        opt_ubf = optimization_results['optimal_parameters']['upper_bound_factor']
        ax1.plot(opt_sr, opt_ubf, 'ro', markersize=12, label='Optimal Parameters')

    # Add labels and title
    ax1.set_xlabel('Survival Ratio')
    ax1.set_ylabel('Upper Bound Factor')
    ax1.set_title('Weight Window Parameter Performance')

    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Add legend
    ax1.legend()

    # Save figure
    plt.savefig(os.path.join(PLOT_DIR, 'ww_optimization_parameters.png'), dpi=300, bbox_inches='tight')
    figures.append(fig1)

    # Create figure for FOM vs test case
    if optimization_results['test_cases']:
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        # Extract data for each test case
        case_labels = []
        optimal_foms = []

        for i, case in enumerate(optimization_results['test_cases']):
            if 'optimal' in case:
                case_labels.append(f"Case {i + 1}")
                optimal_foms.append(case['optimal']['fom'])

        # Create bar chart
        bars = ax2.bar(case_labels, optimal_foms)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}', ha='center', va='bottom')

        # Add labels and title
        ax2.set_ylabel('Figure of Merit')
        ax2.set_title('Optimal Figure of Merit by Test Case')

        # Add grid
        ax2.grid(axis='y', linestyle='--', alpha=0.6)

        # Save figure
        plt.savefig(os.path.join(PLOT_DIR, 'ww_optimization_by_case.png'), dpi=300, bbox_inches='tight')
        figures.append(fig2)

    return figures


def create_weight_windows(channel_diameter=0.5, survival_ratio=0.5, upper_bound_factor=5):
    """
    Create weight windows with specified parameters.
    This is a modified version for optimization.

    Args:
        channel_diameter (float): Channel diameter in cm
        survival_ratio (float): Survival ratio for weight windows
        upper_bound_factor (float): Factor to multiply lower bounds by for upper bounds

    Returns:
        openmc.WeightWindows: Weight windows object
    """
    # This would be implementation-specific based on how weight windows are created
    # For now, return a placeholder
    return None


if __name__ == "__main__":
    # Load results
    ww_results_file = os.path.join(DATA_DIR, 'simulation_results_with_ww.json')
    no_ww_results_file = os.path.join(DATA_DIR, 'simulation_results_without_ww.json')

    try:
        with open(ww_results_file, 'r') as f:
            results_with_ww = json.load(f)

        with open(no_ww_results_file, 'r') as f:
            results_without_ww = json.load(f)

        # Analyze performance
        metrics = analyze_weight_window_performance(results_with_ww, results_without_ww)

        # Plot comparisons
        plot_weight_window_comparison(metrics)

        print("Weight window analysis complete and plots saved.")
    except FileNotFoundError as e:
        print(f"Error loading results files: {str(e)}")
        print("Make sure to run simulations with and without weight windows first.")

