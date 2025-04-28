#!/usr/bin/env python3
"""
Machine learning module for analyzing radiation patterns and predicting dose rates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
from config import DATA_DIR, MODEL_DIR, PLOT_DIR


def prepare_dataset(results):
    """
    Prepare dataset from simulation results for machine learning.

    Args:
        results (list): Simulation results

    Returns:
        pandas.DataFrame: Prepared dataset
    """
    data = []

    for result in results:
        # Get features
        energy = result['energy']
        channel_diameter = result['channel_diameter']
        distance = result['detector_distance']
        angle = result['detector_angle']

        # Extract doses from tallies
        doses = {}
        for tally_name, tally_data in result['tallies'].items():
            if 'detector' in tally_name:
                # Get different scoring methods
                for i, score in enumerate(tally_data['scores']):
                    if score in ['flux', 'heating', 'heating-photon', 'kerma-photon']:
                        # Use mean value
                        doses[score] = tally_data['mean'][i]

        # Skip if no dose data
        if not doses:
            continue

        # Create data row
        row = {
            'energy': energy,
            'channel_diameter': channel_diameter,
            'distance': distance,
            'angle': angle
        }
        row.update(doses)

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate additional features
    df['channel_area'] = np.pi * (df['channel_diameter'] / 2) ** 2
    df['solid_angle'] = df['channel_area'] / (df['distance'] ** 2)
    df['off_axis_distance'] = df['distance'] * np.sin(np.radians(df['angle']))
    df['on_axis_distance'] = df['distance'] * np.cos(np.radians(df['angle']))

    # Add feature for wall attenuation (simplified Beer-Lambert law)
    # μ = attenuation coefficient (cm^-1) - approximate values for concrete
    mu_dict = {
        0.1: 0.5,  # 0.1 MeV - high attenuation
        0.5: 0.2,  # 0.5 MeV - medium attenuation
        1.0: 0.15,  # 1.0 MeV
        2.0: 0.1,  # 2.0 MeV
        3.0: 0.08,  # 3.0 MeV
        5.0: 0.06  # 5.0 MeV - low attenuation
    }

    # Apply attenuation factors for different materials and geometries
    df['attenuation_factor'] = df.apply(
        lambda row: np.exp(-mu_dict.get(row['energy'], 0.1) *
                           (row['channel_diameter'] / 20)),  # Simplified attenuation model
        axis=1
    )

    return df


def train_dose_prediction_model(df, dose_type='flux'):
    """
    Train a machine learning model to predict dose rates.

    Args:
        df (pandas.DataFrame): Dataset
        dose_type (str): Type of dose to predict

    Returns:
        tuple: (model, X_test, y_test, scaler) - trained model and test data
    """
    # Ensure output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Define features and target
    X = df[['energy', 'channel_diameter', 'distance', 'angle',
            'channel_area', 'solid_angle', 'off_axis_distance',
            'on_axis_distance', 'attenuation_factor']]
    y = df[dose_type]

    # Log-transform dose values
    y_log = np.log10(y + 1e-10)  # Add small value to prevent log(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print(f"Training model to predict {dose_type}...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)

    # Convert back from log scale
    y_test_actual = 10 ** y_test
    y_pred_actual = 10 ** y_pred

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model performance for {dose_type}:")
    print(f"Mean squared error (log scale): {mse:.4f}")
    print(f"R² score (log scale): {r2:.4f}")

    # Calculate relative errors
    rel_errors = np.abs(y_pred_actual - y_test_actual) / (y_test_actual + 1e-10)
    mean_rel_error = np.mean(rel_errors)
    median_rel_error = np.median(rel_errors)

    print(f"Mean relative error: {mean_rel_error:.2%}")
    print(f"Median relative error: {median_rel_error:.2%}")

    # Save model
    model_filename = os.path.join(MODEL_DIR, f"dose_prediction_{dose_type}.joblib")
    scaler_filename = os.path.join(MODEL_DIR, f"scaler_{dose_type}.joblib")

    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    print(f"Model saved to {model_filename}")

    return model, X_test, y_test, scaler


def plot_feature_importance(model, feature_names, dose_type='flux'):
    """
    Plot feature importance from the trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): Names of features
        dose_type (str): Type of dose predicted

    Returns:
        matplotlib.figure.Figure: Feature importance plot
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot feature importances
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')

    # Add labels and title
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title(f'Feature Importance for {dose_type.capitalize()} Prediction')

    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(PLOT_DIR, f'feature_importance_{dose_type}.png'), dpi=300)

    return fig


def plot_prediction_vs_actual(y_test, y_pred, dose_type='flux'):
    """
    Plot predicted vs actual dose values.

    Args:
        y_test: Actual (log-transformed) dose values
        y_pred: Predicted (log-transformed) dose values
        dose_type (str): Type of dose predicted

    Returns:
        matplotlib.figure.Figure: Prediction vs actual plot
    """
    # Convert from log scale
    y_test_actual = 10 ** y_test
    y_pred_actual = 10 ** y_pred

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points
    ax.scatter(y_test_actual, y_pred_actual, alpha=0.6)

    # Plot perfect prediction line
    min_val = min(y_test_actual.min(), y_pred_actual.min())
    max_val = max(y_test_actual.max(), y_pred_actual.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    # Add factor-of-2 and factor-of-10 lines
    ax.plot([min_val, max_val], [min_val * 2, max_val * 2], 'g--', alpha=0.5, label='Factor of 2')
    ax.plot([min_val, max_val], [min_val / 2, max_val / 2], 'g--', alpha=0.5)
    ax.plot([min_val, max_val], [min_val * 10, max_val * 10], 'b--', alpha=0.3, label='Factor of 10')
    ax.plot([min_val, max_val], [min_val / 10, max_val / 10], 'b--', alpha=0.3)

    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel(f'Actual {dose_type.capitalize()}')
    ax.set_ylabel(f'Predicted {dose_type.capitalize()}')
    ax.set_title(f'Predicted vs Actual {dose_type.capitalize()} Values')

    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    # Add legend
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(PLOT_DIR, f'prediction_vs_actual_{dose_type}.png'), dpi=300)

    return fig


def predict_dose_surface(model, scaler, energy, channel_diameter, distance_range, angle_range, dose_type='flux'):
    """
    Generate a prediction surface for visualization.

    Args:
        model: Trained model
        scaler: Fitted scaler
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        distance_range (tuple): (min_distance, max_distance) in cm
        angle_range (tuple): (min_angle, max_angle) in degrees
        dose_type (str): Type of dose predicted

    Returns:
        tuple: (X_grid, Y_grid, Z_grid) - meshgrid and predicted values
    """
    # Create meshgrid for distance and angle
    distances = np.linspace(distance_range[0], distance_range[1], 50)
    angles = np.linspace(angle_range[0], angle_range[1], 50)
    X_grid, Y_grid = np.meshgrid(distances, angles)

    # Create feature matrix for prediction
    feature_rows = []
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            distance = X_grid[i, j]
            angle = Y_grid[i, j]

            # Calculate derived features
            channel_area = np.pi * (channel_diameter / 2) ** 2
            solid_angle = channel_area / (distance ** 2)
            off_axis_distance = distance * np.sin(np.radians(angle))
            on_axis_distance = distance * np.cos(np.radians(angle))

            # Approximate attenuation factor based on energy
            # Simplified calculation of μ (attenuation coefficient)
            if energy <= 0.1:
                mu = 0.5
            elif energy <= 0.5:
                mu = 0.2
            elif energy <= 1.0:
                mu = 0.15
            elif energy <= 2.0:
                mu = 0.1
            elif energy <= 3.0:
                mu = 0.08
            else:
                mu = 0.06

            attenuation_factor = np.exp(-mu * (channel_diameter / 20))

            feature_rows.append([
                energy, channel_diameter, distance, angle,
                channel_area, solid_angle, off_axis_distance,
                on_axis_distance, attenuation_factor
            ])

    # Convert to array and scale
    features = np.array(feature_rows)
    features_scaled = scaler.transform(features)

    # Predict log dose
    log_dose_pred = model.predict(features_scaled)

    # Convert from log scale
    dose_pred = 10 ** log_dose_pred

    # Reshape to grid
    Z_grid = dose_pred.reshape(X_grid.shape)

    return X_grid, Y_grid, Z_grid


def plot_dose_surface(X_grid, Y_grid, Z_grid, energy, channel_diameter, dose_type='flux'):
    """
    Plot a 3D surface of predicted dose rates.

    Args:
        X_grid: Distance meshgrid
        Y_grid: Angle meshgrid
        Z_grid: Predicted dose meshgrid
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        dose_type (str): Type of dose predicted

    Returns:
        matplotlib.figure.Figure: 3D surface plot
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=20)
    cbar.set_label(f'{dose_type.capitalize()} (particles/cm²/src)')

    # Set log scale for Z-axis
    ax.zscale('log')

    # Set labels
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_zlabel(f'{dose_type.capitalize()}')

    # Set title
    ax.set_title(f'Predicted {dose_type.capitalize()} - Energy: {energy} MeV, Channel Ø{channel_diameter} cm')

    # Save figure
    plt.savefig(os.path.join(PLOT_DIR, f'dose_surface_{energy}MeV_{channel_diameter}cm_{dose_type}.png'),
                dpi=300, bbox_inches='tight')

    return fig


def plot_dose_contour(X_grid, Y_grid, Z_grid, energy, channel_diameter, dose_type='flux'):
    """
    Plot a 2D contour of predicted dose rates.

    Args:
        X_grid: Distance meshgrid
        Y_grid: Angle meshgrid
        Z_grid: Predicted dose meshgrid
        energy (float): Source energy in MeV
        channel_diameter (float): Channel diameter in cm
        dose_type (str): Type of dose predicted

    Returns:
        matplotlib.figure.Figure: Contour plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot contour with log scale
    contour = ax.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap='inferno',
                          norm=plt.matplotlib.colors.LogNorm())

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(f'{dose_type.capitalize()} (particles/cm²/src)')

    # Add contour lines
    contour_lines = ax.contour(X_grid, Y_grid, Z_grid, colors='white', linewidths=0.5,
                               levels=10, norm=plt.matplotlib.colors.LogNorm())
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1e')

    # Set labels
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Angle (degrees)')

    # Set title
    ax.set_title(f'Predicted {dose_type.capitalize()} Contour - Energy: {energy} MeV, Channel Ø{channel_diameter} cm')

    # Add grid
    ax.grid(linestyle='--', alpha=0.3)

    # Save figure
    plt.savefig(os.path.join(PLOT_DIR, f'dose_contour_{energy}MeV_{channel_diameter}cm_{dose_type}.png'),
                dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Load results
    results_file = os.path.join(DATA_DIR, 'simulation_results.json')
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Prepare dataset
    df = prepare_dataset(results)
    print(f"Prepared dataset with {len(df)} rows and {len(df.columns)} columns")

    # Save dataset
    df.to_csv(os.path.join(DATA_DIR, 'ml_dataset.csv'), index=False)

    # Train models for different dose types
    dose_types = ['flux', 'heating', 'kerma-photon']

    for dose_type in dose_types:
        if dose_type in df.columns:
            # Train model
            model, X_test, y_test, scaler = train_dose_prediction_model(df, dose_type)

            # Plot feature importance
            plot_feature_importance(model, X_test.columns, dose_type)

            # Plot predictions vs actual
            y_pred = model.predict(scaler.transform(X_test))
            plot_prediction_vs_actual(y_test, y_pred, dose_type)

            # Generate and plot prediction surfaces for various configurations
            energy = 1.0  # MeV
            channel_diameter = 0.5  # cm

            X_grid, Y_grid, Z_grid = predict_dose_surface(
                model, scaler, energy, channel_diameter,
                distance_range=(10, 150), angle_range=(0, 45),
                dose_type=dose_type
            )

            plot_dose_surface(X_grid, Y_grid, Z_grid, energy, channel_diameter, dose_type)
            plot_dose_contour(X_grid, Y_grid, Z_grid, energy, channel_diameter, dose_type)

            print(f"Completed analysis for {dose_type}")
        else:
            print(f"Dose type {dose_type} not found in dataset")

