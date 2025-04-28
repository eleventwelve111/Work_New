import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.interpolate import griddata
import seaborn as sns
from materials import get_concrete_composition


class StreamingAnalysis:
    """
    Analyzes radiation streaming effects through channels in shielding materials.
    Focuses on gamma radiation streaming through cylindrical air channels in concrete.
    """

    def __init__(self, results_file=None):
        """
        Initialize the streaming analysis module.

        Parameters:
            results_file: Path to JSON file containing simulation results
        """
        self.results = {}
        self.streaming_data = {}
        self.damage_data = {}
        self.concrete_materials = {
            'ANSI_ANS_6_4_2006': get_concrete_composition('ANSI_ANS_6_4_2006'),
            'high_density': get_concrete_composition('high_density'),
            'magnetite_concrete': get_concrete_composition('magnetite')
        }

        if results_file and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                self.results = json.load(f)

    def analyze_streaming_pathways(self, energy_kev, diameter_cm):
        """
        Analyze radiation streaming pathways for a specific energy and channel diameter.

        Parameters:
            energy_kev: Energy in keV
            diameter_cm: Channel diameter in cm

        Returns:
            dict: Dictionary of streaming pathway contributions
        """
        key = f"{energy_kev}_{diameter_cm}"

        # Check if we already have this analysis
        if key in self.streaming_data:
            return self.streaming_data[key]

        # Get simulation results if available
        if key in self.results:
            results = self.results[key]
        else:
            # If no direct results, estimate based on theory
            results = {
                'wall_thickness': 60.96,  # 2 ft in cm
                'material': 'ANSI_ANS_6_4_2006',
                'energy_kev': energy_kev,
                'diameter_cm': diameter_cm,
                'dose_rem_per_hr': self._estimate_dose(energy_kev, diameter_cm)
            }

        # Calculate mean free path
        material = results.get('material', 'ANSI_ANS_6_4_2006')
        mean_free_path = self._estimate_mean_free_path(energy_kev, material)

        # Wall thickness (2 ft = 60.96 cm)
        wall_thickness = results.get('wall_thickness', 60.96)

        # Channel radius
        radius_cm = diameter_cm / 2.0

        # Source distance (6 ft = 182.88 cm from wall)
        source_distance = 182.88  # 6 ft in cm

        # Calculate solid angle from source to channel
        solid_angle = np.pi * (radius_cm ** 2) / (source_distance ** 2)
        solid_angle_fraction = solid_angle / (4 * np.pi)

        # Calculate streaming factor - ratio of radiation through channel vs solid shield
        attenuation = np.exp(-wall_thickness / mean_free_path)
        streaming_factor = solid_angle_fraction / attenuation

        # Calculate pathway contributions
        # 1. Direct streaming (uncollided)
        direct_streaming_prob = np.exp(-wall_thickness * 0.001)  # Minimal attenuation in air

        # 2. Wall scattered contribution
        # Probability of scattering is related to channel diameter and wall thickness
        scatter_prob_factor = 1.0 - np.exp(-(diameter_cm / wall_thickness) ** 0.5)

        # 3. Multiple scattering contribution
        multiple_scatter_factor = (diameter_cm / mean_free_path) * 0.1

        # Normalize contributions
        total = direct_streaming_prob + scatter_prob_factor + multiple_scatter_factor
        primary_direct = direct_streaming_prob / total
        wall_scattered = scatter_prob_factor / total
        multiple_scattered = multiple_scatter_factor / total

        # Calculate radiation penetration beyond shield
        # Inverse square law from channel exit
        penetration_data = {}
        distances = [30, 40, 60, 80, 100, 150]  # cm

        # Channel exit radiation intensity (normalized)
        exit_intensity = 1.0

        # Angular distribution parameters
        # Narrower for higher energies (more forward-peaked)
        energy_mev = energy_kev / 1000.0
        angular_width = 45.0 * np.exp(-0.2 * energy_mev)  # degrees

        # Calculate intensity at each distance and angle
        for distance in distances:
            angle_data = {}
            angles = [0, 5, 10, 15, 30, 45]  # degrees

            for angle in angles:
                # Angular distribution (approximated as Gaussian)
                angular_factor = np.exp(-(angle ** 2) / (2 * angular_width ** 2))

                # Distance attenuation (inverse square)
                distance_factor = 1.0 / (distance ** 2)

                # Combined attenuation
                intensity = exit_intensity * angular_factor * distance_factor

                angle_data[str(angle)] = intensity

            penetration_data[str(distance)] = angle_data

        # Compile streaming pathway analysis
        streaming_analysis = {
            'energy_kev': energy_kev,
            'diameter_cm': diameter_cm,
            'mean_free_path_cm': mean_free_path,
            'streaming_factor': streaming_factor,
            'solid_angle_fraction': solid_angle_fraction,
            'primary_direct_streaming': primary_direct,
            'wall_scattered_contribution': wall_scattered,
            'multiple_scattered_contribution': multiple_scattered,
            'penetration_profile': penetration_data
        }

        # Store results
        self.streaming_data[key] = streaming_analysis

        return streaming_analysis

    def analyze_critical_dimensions(self, min_energy=100, max_energy=5000,
                                    min_diameter=0.05, max_diameter=1.0,
                                    significance_threshold=0.01):
        """
        Identify critical gamma-ray energies and channel diameters capable of
        producing significant doses.

        Parameters:
            min_energy: Minimum energy to analyze (keV)
            max_energy: Maximum energy to analyze (keV)
            min_diameter: Minimum diameter to analyze (cm)
            max_diameter: Maximum diameter to analyze (cm)
            significance_threshold: Threshold for significant dose (rem/hr)

        Returns:
            dict: Analysis of critical dimensions
        """
        # Generate energy and diameter test points
        energies = np.logspace(np.log10(min_energy), np.log10(max_energy), 10)
        diameters = np.logspace(np.log10(min_diameter), np.log10(max_diameter), 10)

        # Store streaming factors for each combination
        streaming_factors = {}
        threshold_diameters = {}

        for energy in energies:
            streaming_factors[str(int(energy))] = []

            # Find minimum diameter that produces significant dose at this energy
            min_significant_diameter = None

            for diameter in diameters:
                # Analyze streaming for this combination
                analysis = self.analyze_streaming_pathways(energy, diameter)
                streaming_factor = analysis['streaming_factor']

                # Store streaming factor
                streaming_factors[str(int(energy))].append((diameter, streaming_factor))

                # Check if this diameter produces significant dose
                # Use streaming factor as proxy for dose
                if streaming_factor >= significance_threshold and min_significant_diameter is None:
                    min_significant_diameter = diameter

            # Store threshold diameter for this energy
            if min_significant_diameter is not None:
                threshold_diameters[str(int(energy))] = min_significant_diameter
            else:
                threshold_diameters[str(int(energy))] = max_diameter  # Default if no threshold found

        # Extract data for model fitting
        x_data = []
        y_data = []

        for energy_str, diameter in threshold_diameters.items():
            x_data.append(float(energy_str))
            y_data.append(diameter)

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Create a model relating energy to threshold diameter
        threshold_model = {}

        if len(x_data) > 2:  # Need at least 3 points for meaningful fit
            try:
                # Try power law fit: D = a * E^b
                def power_law(x, a, b):
                    return a * (x ** b)

                popt, pcov = curve_fit(power_law, x_data, y_data)
                a, b = popt

                threshold_model['model_type'] = 'power_law'
                threshold_model['parameters'] = {'a': a, 'b': b}
                threshold_model['formula'] = f"D_threshold = {a:.3f} * E^{b:.3f}"
                threshold_model['r_squared'] = self._calculate_r_squared(y_data, power_law(x_data, a, b))
            except Exception as e:
                print(f"Error fitting threshold model: {e}")
                # If curve fitting fails, provide simple linear interpolation
                threshold_model['model_type'] = 'interpolation'
                threshold_model['parameters'] = {'x': list(x_data), 'y': list(y_data)}

        # Compile results
        threshold_analysis = {
            'streaming_factors': streaming_factors,
            'threshold_diameters': threshold_diameters,
            'significance_threshold': significance_threshold,
            'threshold_model': threshold_model
        }

        return threshold_analysis

    def _calculate_r_squared(self, y_true, y_pred):
        """Calculate R-squared for model evaluation."""
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    def _estimate_mean_free_path(self, energy_kev, material):
        """
        Estimate mean free path for gamma rays in the given material.

        Parameters:
            energy_kev: Energy in keV
            material: Shield material name

        Returns:
            float: Estimated mean free path in cm
        """
        # Convert energy to MeV for calculations
        energy_mev = energy_kev / 1000.0

        # Simplified mean free path calculations based on material
        if material.lower() in ['ansi_ans_6_4_2006', 'regular_concrete', 'ordinary_concrete']:
            # Regular concrete (ANSI/ANS-6.4-2006)
            # Density ~2.3 g/cm³
            density = 2.3
            # Approximate mass attenuation coefficient (cm²/g)
            if energy_mev < 0.1:
                mu = 0.26 * (energy_mev ** -2.5)
            elif energy_mev < 1.0:
                mu = 0.15 * (energy_mev ** -0.5)
            else:
                mu = 0.05 * (energy_mev ** -0.3)

        elif material.lower() in ['high_density', 'heavy_concrete']:
            # High-density concrete
            # Density ~3.5 g/cm³
            density = 3.5
            # Approximate mass attenuation coefficient (cm²/g)
            if energy_mev < 0.1:
                mu = 0.30 * (energy_mev ** -2.5)
            elif energy_mev < 1.0:
                mu = 0.18 * (energy_mev ** -0.5)
            else:
                mu = 0.06 * (energy_mev ** -0.3)

        elif material.lower() in ['magnetite_concrete', 'magnetite']:
            # Magnetite concrete
            # Density ~3.9 g/cm³
            density = 3.9
            # Approximate mass attenuation coefficient (cm²/g)
            if energy_mev < 0.1:
                mu = 0.32 * (energy_mev ** -2.5)
            elif energy_mev < 1.0:
                mu = 0.20 * (energy_mev ** -0.5)
            else:
                mu = 0.07 * (energy_mev ** -0.3)

        else:
            # Default to regular concrete
            density = 2.3
            if energy_mev < 0.1:
                mu = 0.26 * (energy_mev ** -2.5)
            elif energy_mev < 1.0:
                mu = 0.15 * (energy_mev ** -0.5)
            else:
                mu = 0.05 * (energy_mev ** -0.3)

        # Calculate mean free path (cm)
        mean_free_path = 1.0 / (mu * density)

        return mean_free_path

    def _estimate_dose(self, energy_kev, diameter_cm):
        """
        Estimate dose rate when direct simulation results aren't available.

        Parameters:
            energy_kev: Energy in keV
            diameter_cm: Channel diameter in cm

        Returns:
            float: Estimated dose rate (rem/hr)
        """
        # Convert energy to MeV
        energy_mev = energy_kev / 1000.0

        # Source activity (arbitrary units)
        activity = 1.0

        # Source distance (6 ft = 182.88 cm from wall + 2 ft wall thickness)
        source_distance = 182.88 + 60.96  # cm

        # Calculate solid angle factor
        solid_angle_factor = (diameter_cm / 2.0) ** 2 / (source_distance ** 2)

        # Energy-dependent fluence to dose conversion factor (rem/hr per p/cm²-s)
        # Based on NCRP-38, ANS-6.1.1-1977
        if energy_mev < 0.15:
            dose_conversion = 5.41e-7 * energy_mev ** 0.7
        elif energy_mev < 0.5:
            dose_conversion = 2.05e-7 * energy_mev ** 0.3
        elif energy_mev < 1.0:
            dose_conversion = 2.22e-7 * energy_mev ** 0.5
        elif energy_mev < 2.0:
            dose_conversion = 2.5e-7 * energy_mev ** 0.6
        elif energy_mev < 5.0:
            dose_conversion = 2.8e-7 * energy_mev ** 0.7
        else:
            dose_conversion = 3.0e-7 * energy_mev ** 0.8

        # Calculate dose rate
        # Assume each incident gamma produces one p/cm²-s at reference distance
        reference_flux = activity / (4 * np.pi * source_distance ** 2)  # p/cm²-s
        dose_rate = reference_flux * solid_angle_factor * dose_conversion

        # Apply streaming factor correction
        streaming_factor = (diameter_cm / self._estimate_mean_free_path(energy_kev, 'ANSI_ANS_6_4_2006')) ** 1.5
        dose_rate *= (1.0 + streaming_factor)

        return dose_rate

    def analyze_radiation_damage(self, energy_kev, diameter_cm, exposure_time_hours=8760):
        """
        Analyze potential radiation damage to shield materials around the channel.

        Parameters:
            energy_kev: Energy in keV
            diameter_cm: Channel diameter in cm
            exposure_time_hours: Exposure time in hours (default: 1 year = 8760 hours)

        Returns:
            dict: Dictionary of radiation damage analysis results
        """
        # Key for this configuration
        key = f"{energy_kev}_{diameter_cm}"

        # Find or estimate the relevant parameters
        if key in self.results:
            results = self.results[key]
        else:
            # Estimate based on theory
            results = {
                'wall_thickness': 60.96,  # 2 ft in cm
                'material': 'ANSI_ANS_6_4_2006',
                'energy_kev': energy_kev,
                'diameter_cm': diameter_cm,
                'dose_rem_per_hr': self._estimate_dose(energy_kev, diameter_cm)
            }

        # Extract parameters
        material = results.get('material', 'ANSI_ANS_6_4_2006')
        dose_rate = results.get('dose_rem_per_hr', 0.0)
        wall_thickness = results.get('wall_thickness', 60.96)  # cm

        # Calculate mean free path
        mean_free_path = self._estimate_mean_free_path(energy_kev, material)

        # Parameters for damage analysis
        damage_analysis = {
            'energy_kev': energy_kev,
            'diameter_cm': diameter_cm,
            'mean_free_path_cm': mean_free_path,
            'exposure_time_hours': exposure_time_hours,
            'material': material,
            'wall_damage_profile': [],
            'max_damage_depth_cm': 0.0,
            'critical_damage_depth_cm': 0.0,
            'annual_material_deterioration_percent': 0.0,
            'lifetime_estimate_years': 0.0,
            'hotspot_factor': 0.0,
            'maintenance_recommendation': '',
            'risk_level': ''
        }

        # Calculate energy deposition profile in wall material
        # (Higher at entrance, decreasing with depth)
        radial_positions = np.linspace(0, min(wall_thickness, 30), 20)  # Focus on first 30cm or wall thickness
        damage_profile = []

        # Energy-dependent damage scaling factor
        energy_mev = energy_kev / 1000.0
        damage_scaling = 1.0 + 0.2 * np.log10(1 + energy_mev)

        # Maximum energy deposition is at the channel wall
        max_deposition = dose_rate * damage_scaling

        # Calculate energy deposition profile
        for r in radial_positions:
            # Exponential decrease with distance from channel
            rel_deposition = np.exp(-r / mean_free_path)
            # Enhanced deposition near the surface (backscattering effect)
            surface_enhancement = 1.0 + 0.5 * np.exp(-r / (0.1 * mean_free_path))
            energy_deposition = max_deposition * rel_deposition * surface_enhancement
            damage_profile.append((r, energy_deposition))

        damage_analysis['wall_damage_profile'] = damage_profile

        # Find maximum damage depth (where energy deposition > 10% of maximum)
        r_values = [r for r, _ in damage_profile]
        e_values = [e for _, e in damage_profile]
        threshold_value = 0.1 * max(e_values)

        for i, e in enumerate(e_values):
            if e < threshold_value:
                damage_analysis['max_damage_depth_cm'] = r_values[i]
                break
        else:
            damage_analysis['max_damage_depth_cm'] = max(r_values)

        # Determine critical damage depth (material dependent)
        if material.lower() in ['ansi_ans_6_4_2006', 'regular_concrete', 'ordinary_concrete']:
            # Regular concrete is moderately sensitive to radiation damage
            critical_fraction = 0.2  # 20% of max deposition
        elif material.lower() in ['high_density', 'heavy_concrete']:
            # High-density concrete is more resistant
            critical_fraction = 0.15
        elif material.lower() in ['magnetite_concrete', 'magnetite']:
            # Magnetite concrete is most resistant
            critical_fraction = 0.1
        else:
            # Default
            critical_fraction = 0.2

        for i, e in enumerate(e_values):
            if e < critical_fraction * max(e_values):
                damage_analysis['critical_damage_depth_cm'] = r_values[i]
                break
        else:
            damage_analysis['critical_damage_depth_cm'] = max(r_values)

        # Calculate material deterioration rate
        # This is material-dependent and dose-rate dependent
        if material.lower() in ['ansi_ans_6_4_2006', 'regular_concrete', 'ordinary_concrete']:
            # Regular concrete deterioration factors
            base_rate = 0.1  # % per year at reference dose
            ref_dose = 500.0  # rem/year
        elif material.lower() in ['high_density', 'heavy_concrete']:
            # High-density concrete deterioration factors
            base_rate = 0.07  # % per year at reference dose
            ref_dose = 700.0  # rem/year
        elif material.lower() in ['magnetite_concrete', 'magnetite']:
            # Magnetite concrete deterioration factors
            base_rate = 0.05  # % per year at reference dose
            ref_dose = 1000.0  # rem/year
        else:
            # Default
            base_rate = 0.1
            ref_dose = 500.0

        # Calculate annual dose
        annual_dose = dose_rate * exposure_time_hours

        # Scale deterioration rate by dose
        annual_deterioration = base_rate * (annual_dose / ref_dose)
        damage_analysis['annual_material_deterioration_percent'] = annual_deterioration

        # Estimate material lifetime (years until significant deterioration)
        # For concrete, significant deterioration is ~10%
        if annual_deterioration > 0:
            lifetime_estimate = 10.0 / annual_deterioration
        else:
            lifetime_estimate = float('inf')
        damage_analysis['lifetime_estimate_years'] = lifetime_estimate

        # Calculate hotspot factor (ratio of peak to average energy deposition)
        avg_deposition = np.mean(e_values)
        peak_deposition = max(e_values)
        hotspot_factor = peak_deposition / avg_deposition if avg_deposition > 0 else 1.0
        damage_analysis['hotspot_factor'] = hotspot_factor

        # Determine risk level based on lifetime estimate
        if lifetime_estimate < 5:
            risk_level = 'high'
            maintenance_rec = 'Annual inspection required. Consider channel reinforcement or diameter reduction.'
        elif lifetime_estimate < 15:
            risk_level = 'medium'
            maintenance_rec = 'Biennial inspection recommended. Monitor for concrete degradation around channel.'
        else:
            risk_level = 'low'
            maintenance_rec = 'Routine inspection (5-year intervals) sufficient.'

        damage_analysis['risk_level'] = risk_level
        damage_analysis['maintenance_recommendation'] = maintenance_rec

        # Store results for this configuration
        self.damage_data[key] = damage_analysis

        return damage_analysis

    def compare_concrete_compositions(self, energy_kev, diameter_cm, concrete_types=None):
        """
        Compare different concrete compositions for radiation streaming effects.

        Parameters:
            energy_kev: Energy in keV
            diameter_cm: Channel diameter in cm
            concrete_types: List of concrete types to compare (default: all available types)

        Returns:
            dict: Comparison results for different concrete compositions
        """
        if concrete_types is None:
            concrete_types = list(self.concrete_materials.keys())

        results = {}

        for concrete_type in concrete_types:
            # Analyze streaming for this concrete type
            streaming_data = self.analyze_streaming_pathways(energy_kev, diameter_cm)

            # Get or estimate damage analysis
            key = f"{energy_kev}_{diameter_cm}_{concrete_type}"
            if key in self.damage_data:
                damage_data = self.damage_data[key]
            else:
                # Override material type for this analysis
                original_results = self.results.get(f"{energy_kev}_{diameter_cm}", {})
                temp_results = original_results.copy()
                temp_results['material'] = concrete_type

                # Store temporarily
                temp_key = f"{energy_kev}_{diameter_cm}"
                old_results = self.results.get(temp_key, None)
                self.results[temp_key] = temp_results

                # Analyze damage
                damage_data = self.analyze_radiation_damage(energy_kev, diameter_cm)

                # Restore original results
                if old_results:
                    self.results[temp_key] = old_results
                else:
                    del self.results[temp_key]

            # Store results for this concrete type
            results[concrete_type] = {
                'streaming_factor': streaming_data['streaming_factor'],
                'mean_free_path': streaming_data['mean_free_path_cm'],
                'lifetime_estimate': damage_data['lifetime_estimate_years'],
                'annual_deterioration': damage_data['annual_material_deterioration_percent'],
                'risk_level': damage_data['risk_level']
            }

        return results

    def plot_streaming_pathways(self, energy_kev, diameter_cm, output_dir="plots"):
        """
        Generate visualization of radiation streaming pathways.

        Parameters:
            energy_kev: Energy in keV
            diameter_cm: Channel diameter in cm
            output_dir: Directory to save output plots

        Returns:
            str: Path to generated plot file
        """
        key = f"{energy_kev}_{diameter_cm}"

        # Ensure we have pathway analysis for this configuration
        if key not in self.streaming_data:
            self.analyze_streaming_pathways(energy_kev, diameter_cm)

        pathways = self.streaming_data[key]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Prepare plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Shield dimensions (2 ft thickness)
        wall_thickness = 60.96  # cm
        shield_height = 200  # cm, arbitrary for visualization

        # Source distance (6 ft from wall)
        source_distance = 182.88  # cm

        # Source position
        source_x = -source_distance
        source_y = 0

        # Channel radius
        channel_radius = diameter_cm / 2

        # Draw shield block
        shield_rect = plt.Rectangle((0, -shield_height / 2), wall_thickness, shield_height,
                                    color='lightgray', alpha=0.7, label='Concrete Shield')
        ax.add_patch(shield_rect)

        # Draw channel
        channel_top = plt.Rectangle((0, channel_radius), wall_thickness, -channel_radius * 2,
                                    color='white', alpha=1.0, label='Air Channel')
        ax.add_patch(channel_top)

        # Draw source
        ax.plot(source_x, source_y, 'ro', markersize=10, label='Gamma Source')

        # Draw detector positions
        detector_distances = [30, 40, 60, 80, 100, 150]  # cm from back of shield
        for dist in detector_distances:
            detector_x = wall_thickness + dist
            ax.plot(detector_x, 0, 'bo', markersize=8, alpha=0.5)

        # Reference detector (30 cm) labeled explicitly
        ax.plot(wall_thickness + 30, 0, 'bo', markersize=8, label='ICRU Phantom (30cm)')

        # Calculate direct streaming rays
        n_direct_rays = 20
        cone_angle = np.arctan(channel_radius / source_distance)

        # Draw direct streaming paths
        for i in range(n_direct_rays):
            # Random angle within channel aperture
            angle = np.random.uniform(-cone_angle, cone_angle)

            # Calculate ray path through channel
            end_x = wall_thickness + 200  # extend beyond wall
            end_y = source_y + np.tan(angle) * (end_x - source_x)

            # Draw ray
            ax.plot([source_x, end_x], [source_y, end_y], 'y-', alpha=0.3, linewidth=1)

        # Draw channel walls and highlight
        ax.plot([0, wall_thickness], [channel_radius, channel_radius], 'k-', linewidth=2)
        ax.plot([0, wall_thickness], [-channel_radius, -channel_radius], 'k-', linewidth=2)

        # Highlight high-dose region
        # This is a cone extending from channel exit
        exit_angle = np.arctan(channel_radius / wall_thickness)
        exit_cone_x = np.linspace(wall_thickness, wall_thickness + 150, 100)
        exit_cone_y1 = np.tan(exit_angle) * (exit_cone_x - wall_thickness)
        exit_cone_y2 = -np.tan(exit_angle) * (exit_cone_x - wall_thickness)

        # Fill the cone region
        ax.fill_between(exit_cone_x, exit_cone_y1, exit_cone_y2, color='yellow', alpha=0.1,
                        label='High-Dose Region')

        # Add angular measurements at 30cm from wall exit
        angles = [0, 5, 10, 15, 30, 45]  # degrees
        ref_distance = 30  # cm

        for angle in angles:
            angle_rad = np.radians(angle)
            angle_x = wall_thickness + ref_distance * np.cos(angle_rad)
            angle_y = ref_distance * np.sin(angle_rad)

            if angle > 0:  # Skip 0 degrees as it's on the axis
                # Draw angle line
                ax.plot([wall_thickness, angle_x], [0, angle_y], 'k--', alpha=0.3)
                # Label angle
                ax.text(angle_x + 5, angle_y + 5, f"{angle}°", fontsize=8, ha='left', va='bottom')

        # Add labels and legend
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('Position (cm)')
        ax.set_title(f'Radiation Streaming Pathways: {energy_kev / 1000:.1f} MeV, {diameter_cm:.2f} cm Channel')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Set axis limits
        ax.set_xlim(source_x - 30, wall_thickness + 200)
        ax.set_ylim(-100, 100)

        # Equal aspect ratio
        ax.set_aspect('equal', 'datalim')

        # Add annotations with key information
        energy_mev = energy_kev / 1000.0
        mean_free_path = pathways['mean_free_path_cm']
        streaming_factor = pathways['streaming_factor']

        info_text = (
            f"Energy: {energy_mev:.2f} MeV\n"
            f"Channel Diameter: {diameter_cm:.2f} cm\n"
            f"Mean Free Path: {mean_free_path:.2f} cm\n"
            f"Streaming Factor: {streaming_factor:.2e}\n"
            f"Direct Streaming: {pathways['primary_direct_streaming'] * 100:.1f}%\n"
            f"Wall Scattered: {pathways['wall_scattered_contribution'] * 100:.1f}%\n"
            f"Multiple Scattered: {pathways['multiple_scattered_contribution'] * 100:.1f}%"
        )

        # Add text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        # Save the figure
        output_file = os.path.join(output_dir, f"streaming_pathways_{energy_kev}_{diameter_cm}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def plot_enhanced_exit_radiation(self, energy_kev, diameter_cm, output_dir="plots"):
        """
        Create an enhanced close-up visualization of radiation flux distribution
        outside the wall exit.

        Parameters:
            energy_kev: Energy in keV
            diameter_cm: Channel diameter in cm
            output_dir: Directory to save output plots

        Returns:
            str: Path to generated plot file
        """
        key = f"{energy_kev}_{diameter_cm}"

        # Ensure we have pathway analysis for this configuration
        if key not in self.streaming_data:
            self.analyze_streaming_pathways(energy_kev, diameter_cm)

        pathways = self.streaming_data[key]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create a high-resolution figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Wall exit position
        wall_thickness = 60.96  # cm (2 ft)

        # Channel radius
        channel_radius = diameter_cm / 2

        # Energy in MeV
        energy_mev = energy_kev / 1000.0

        # Generate a mesh grid for the visualization area
        # Focus on the area immediately outside the wall
        x_max = 100  # cm from wall exit
        y_max = 50  # cm above/below channel centerline

        # Higher resolution for smaller channels
        grid_size = max(100, int(500 / diameter_cm))
        x = np.linspace(0, x_max, grid_size)
        y = np.linspace(-y_max, y_max, grid_size)
        X, Y = np.meshgrid(x, y)

        # Calculate radiation flux at each point
        Z = np.zeros_like(X)

        # Parameters affecting radiation spread
        # More forward-peaked for higher energies
        forward_peaking = 1.0 + 2.0 * np.log10(1 + energy_mev)

        # Angular width decreases with energy (more forward-peaked at higher energies)
        angular_width = max(5.0, 45.0 / forward_peaking)  # degrees
        angular_width_rad = np.radians(angular_width)

        # For each point in the grid
        for i in range(len(x)):
            for j in range(len(y)):
                # Distance from wall exit
                distance = x[i]

                # Angle from centerline
                if distance > 0:
                    angle = np.arctan(abs(y[j]) / distance)
                else:
                    angle = np.pi / 2 if y[j] != 0 else 0

                # Inverse square law
                distance_factor = 1.0 / (1.0 + distance ** 2)

                # Angular distribution (Gaussian-like)
                angular_factor = np.exp(-(angle ** 2) / (2 * angular_width_rad ** 2))

                # Channel geometry effect - Higher for points within projection of channel
                geometry_factor = 1.0
                if abs(y[j]) <= channel_radius * (1 + distance / wall_thickness):
                    geometry_factor = 2.0

                # Calculate flux
                Z[j, i] = distance_factor * angular_factor * geometry_factor

        # Adaptive smoothing based on channel diameter
        smooth_sigma = max(1, 5 / diameter_cm)
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=smooth_sigma)

        # Normalize Z
        Z = Z / Z.max()

        # Create heatmap with logarithmic color scale
        im = ax.pcolormesh(X, Y, Z, norm=LogNorm(vmin=max(1e-3, Z.min()), vmax=1),
                           cmap='inferno', shading='gouraud')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Relative Flux Intensity (log scale)')

        # Add channel exit
        ax.plot([0, 0], [-channel_radius, channel_radius], 'w-', linewidth=3)

        # Add distance markers
        distances = [10, 30, 50, 75, 100]
        for d in distances:
            circle = plt.Circle((0, 0), d, fill=False, color='gray', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
            ax.text(d, 5, f"{d} cm", color='white', fontsize=8, ha='center', va='bottom', alpha=0.7)

        # Add angle indicators
        angles = [0, 5, 10, 15, 30, 45]
        for angle in angles:
            angle_rad = np.radians(angle)
            length = min(x_max, y_max / np.sin(angle_rad)) if angle > 0 else x_max
            end_x = length * np.cos(angle_rad)
            end_y = length * np.sin(angle_rad)

            ax.plot([0, end_x], [0, end_y], 'w--', alpha=0.3, linewidth=1)

            # Label at 30% of the way along the line
            if angle > 0:  # Skip 0 degrees text
                label_x = 0.3 * end_x
                label_y = 0.3 * end_y
                ax.text(label_x, label_y, f"{angle}°", color='white', fontsize=8,
                        ha='center', va='center', alpha=0.7)

        # Add labels and title
        ax.set_xlabel('Distance from Shield Exit (cm)')
        ax.set_ylabel('Distance from Centerline (cm)')
        ax.set_title(f'Enhanced Radiation Distribution: {energy_mev:.2f} MeV, {diameter_cm:.2f} cm Channel')

        # Set axis limits
        ax.set_xlim(0, x_max)
        ax.set_ylim(-y_max, y_max)

        # Equal aspect ratio
        ax.set_aspect('equal')

        # Add annotations with key information
        info_text = (
            f"Energy: {energy_mev:.2f} MeV\n"
            f"Channel Diameter: {diameter_cm:.2f} cm\n"
            f"Streaming Factor: {pathways['streaming_factor']:.2e}\n"
            f"Angular Width: {angular_width:.1f}°\n"
            f"Forward Peaking: {forward_peaking:.1f}x"
        )

        # Add text box
        props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', color='white', bbox=props)

        # Save the figure
        output_file = os.path.join(output_dir, f"enhanced_exit_radiation_{energy_kev}_{diameter_cm}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def plot_material_comparison(self, energy_kev, diameter_cm, output_dir="plots"):
        """
        Generate comparison plots for different concrete compositions.

        Parameters:
            energy_kev: Energy in keV
            diameter_cm: Channel diameter in cm
            output_dir: Directory to save output plots

        Returns:
            str: Path to generated plot file
        """
        # Compare concrete compositions
        results = self.compare_concrete_compositions(energy_kev, diameter_cm)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract data for plotting
        materials = list(results.keys())
        streaming_factors = [results[m]['streaming_factor'] for m in materials]
        lifetimes = [results[m]['lifetime_estimate'] for m in materials]
        deterioration = [results[m]['annual_deterioration'] for m in materials]

        # Create material comparison figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Streaming factor comparison
        axs[0].bar(materials, streaming_factors, color='skyblue')
        axs[0].set_ylabel('Streaming Factor')
        axs[0].set_title('Streaming Factor Comparison')
        axs[0].tick_params(axis='x', rotation=45)
        axs[0].grid(True, alpha=0.3)

        # 2. Lifetime comparison
        axs[1].bar(materials, lifetimes, color='lightgreen')
        axs[1].set_ylabel('Estimated Lifetime (years)')
        axs[1].set_title('Material Lifetime Comparison')
        axs[1].tick_params(axis='x', rotation=45)
        axs[1].grid(True, alpha=0.3)

        # 3. Annual deterioration comparison
        axs[2].bar(materials, deterioration, color='salmon')
        axs[2].set_ylabel('Annual Deterioration (%)')
        axs[2].set_title('Material Deterioration Comparison')
        axs[2].tick_params(axis='x', rotation=45)
        axs[2].grid(True, alpha=0.3)

        # Add overall title
        fig.suptitle(f'Concrete Material Comparison: {energy_kev / 1000:.2f} MeV, {diameter_cm:.2f} cm Channel',
                     fontsize=16)

        plt.tight_layout()

        # Save the figure
        output_file = os.path.join(output_dir, f"material_comparison_{energy_kev}_{diameter_cm}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def plot_dose_vs_angle(self, energy_levels, diameters, distances, angles, output_dir="plots"):
        """
        Generate plots showing dose vs angle for different energies, distances, and diameters.

        Parameters:
            energy_levels: List of energy levels in keV
            diameters: List of channel diameters in cm
            distances: List of detector distances in cm
            angles: List of detector angles in degrees
            output_dir: Directory to save output plots

        Returns:
            dict: Paths to generated plot files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        output_files = {}

        # Create one plot per energy level
        for energy in energy_levels:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # One line per diameter and distance combination
            for diameter in diameters:
                for distance in distances:
                    # Get dose at each angle
                    doses = []
                    key = f"{energy}_{diameter}"

                    # Ensure we have pathway analysis for this configuration
                    if key not in self.streaming_data:
                        self.analyze_streaming_pathways(energy, diameter)
                    pathways = self.streaming_data[key]

                    # Get penetration data if available
                    penetration_data = pathways.get('penetration_profile', {})
                    distance_data = penetration_data.get(str(distance), {})

                    # Extract doses at each angle
                    for angle in angles:
                        if str(angle) in distance_data:
                            doses.append(distance_data[str(angle)])
                        else:
                            # Estimate if not directly available
                            # Angular distribution (approximated as Gaussian)
                            energy_mev = energy / 1000.0
                            angular_width = max(5.0, 45.0 / (1.0 + 2.0 * np.log10(1 + energy_mev)))  # degrees
                            angular_factor = np.exp(-(angle ** 2) / (2 * (angular_width ** 2)))

                            # Distance attenuation (inverse square)
                            distance_factor = 1.0 / (distance ** 2)

                            # Combined effect
                            dose = angular_factor * distance_factor
                            doses.append(dose)

                    # Plot this line
                    label = f"D={diameter:.2f}cm, r={distance}cm"
                    ax.plot(angles, doses, marker='o', label=label)

                    # Set y-axis to logarithmic scale
                ax.set_yscale('log')

                # Add labels and legend
                ax.set_xlabel('Angle from Centerline (degrees)')
                ax.set_ylabel('Relative Dose (log scale)')
                ax.set_title(f'Dose vs Angle: {energy / 1000:.2f} MeV')
                ax.grid(True, which='both', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

                plt.tight_layout()

                # Save the figure
                output_file = os.path.join(output_dir, f"dose_vs_angle_{energy}.png")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()

                output_files[str(energy)] = output_file

            return output_files

    def plot_damage_profile(self, energy_kev, diameter_cm, output_dir="plots"):
        """
        Generate visualization of radiation damage profile in concrete around the channel.

        Parameters:
            energy_kev: Energy in keV
            diameter_cm: Channel diameter in cm
            output_dir: Directory to save output plots

        Returns:
            str: Path to generated plot file
        """
        key = f"{energy_kev}_{diameter_cm}"

        # Get or generate damage analysis
        if key not in self.damage_data:
            self.analyze_radiation_damage(energy_kev, diameter_cm)

        damage_data = self.damage_data[key]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract data for plotting
        wall_damage_profile = damage_data['wall_damage_profile']

        r_values = np.array([r for r, _ in wall_damage_profile])
        e_values = np.array([e for _, e in wall_damage_profile])

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot damage profile
        ax.plot(r_values, e_values, 'b-', linewidth=2, label='Energy Deposition')
        ax.fill_between(r_values, 0, e_values, alpha=0.2, color='blue')

        # Plot threshold markers
        max_damage_depth = damage_data['max_damage_depth_cm']
        critical_damage_depth = damage_data['critical_damage_depth_cm']

        max_e_value = np.interp(max_damage_depth, r_values, e_values)
        critical_e_value = np.interp(critical_damage_depth, r_values, e_values)

        ax.axvline(x=max_damage_depth, color='red', linestyle='--',
                   label=f'Max Damage Depth: {max_damage_depth:.1f} cm')
        ax.axvline(x=critical_damage_depth, color='orange', linestyle='--',
                   label=f'Critical Damage Depth: {critical_damage_depth:.1f} cm')

        # Add labels and title
        ax.set_xlabel('Distance from Channel Wall (cm)')
        ax.set_ylabel('Relative Energy Deposition')
        ax.set_title(f'Radiation Damage Profile: {energy_kev / 1000:.2f} MeV, {diameter_cm:.2f} cm Channel')

        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set x-axis limits
        ax.set_xlim(0, max(r_values))

        # Add annotations with key information
        annual_deterioration = damage_data['annual_material_deterioration_percent']
        lifetime_estimate = damage_data['lifetime_estimate_years']
        risk_level = damage_data['risk_level']

        info_text = (
            f"Material: {damage_data['material']}\n"
            f"Annual Deterioration: {annual_deterioration:.2f}%\n"
            f"Lifetime Estimate: {lifetime_estimate:.1f} years\n"
            f"Risk Level: {risk_level.upper()}\n"
            f"Hotspot Factor: {damage_data['hotspot_factor']:.1f}"
        )

        # Add text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        # Save the figure
        output_file = os.path.join(output_dir, f"damage_profile_{energy_kev}_{diameter_cm}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file


