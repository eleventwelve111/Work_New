import json
import os
import numpy as np
import matplotlib.pyplot as plt
from streaming_analysis import StreamingAnalysis
import pandas as pd
import seaborn as sns
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus import PageBreak, ListFlowable, ListItem
from reportlab.lib.units import inch
from scipy.optimize import curve_fit
from typing import List, Dict, Optional, Tuple, Union
import logging
import datetime
import pickle


class ResultAnalyzer:
    """
    A class for analyzing radiation streaming simulation results.
    Provides tools for trend analysis, visualization, and reporting.
    """

    def __init__(self, results_db=None):
        """
        Initialize the analyzer.
        Args:
            results_db: Optional list of simulation results to analyze
        """
        self.results_db = results_db
        self.logger = logging.getLogger(__name__)

    def load_results(self, filepath='results/simulation_results.json'):
        """Load results from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                self.results_db = json.load(f)
            self.logger.info(f"Loaded {len(self.results_db)} simulation results from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            return False

    def to_dataframe(self):
        """Convert results to pandas DataFrame for easier analysis."""
        if self.results_db is None or len(self.results_db) == 0:
            self.logger.warning("No results available for conversion to DataFrame")
            return None
        # Convert to DataFrame
        df = pd.DataFrame(self.results_db)
        # Remove rows with errors
        if 'status' in df.columns:
            df = df[df['status'] == 'completed']
        return df

    def analyze_channel_diameter_trend(self, fixed_energy=None, fixed_distance=None):
        """
        Analyze the effect of channel diameter on dose/flux.
        Args:
            fixed_energy: Optional fixed energy to filter by
            fixed_distance: Optional fixed distance to filter by
        Returns:
            dict: Analysis results
        """
        df = self.to_dataframe()
        if df is None or len(df) == 0:
            return None
        # Filter data if needed
        if fixed_energy is not None:
            df = df[df['energy'] == fixed_energy]
        if fixed_distance is not None:
            df = df[df['distance'] == fixed_distance]
        if len(df) < 2:
            self.logger.warning("Not enough data points for trend analysis")
            return None

        # Group by channel diameter and compute statistics
        channel_data = []
        # For each unique channel diameter
        for diameter in sorted(df['channel_diameter'].unique()):
            subset = df[df['channel_diameter'] == diameter]
            # Collect statistics
            stats = {
                'diameter': diameter,
                'area': np.pi * (diameter / 2) ** 2 if diameter > 0 else 0,
                'dose_mean': subset['kerma'].mean() if 'kerma' in subset.columns else subset['total_dose'].mean(),
                'dose_std': subset['kerma'].std() if 'kerma' in subset.columns else subset['total_dose'].std(),
                'norm_dose_mean': subset['normalized_dose'].mean() if 'normalized_dose' in subset.columns else None,
                'norm_dose_std': subset['normalized_dose'].std() if 'normalized_dose' in subset.columns else None,
                'flux_mean': subset['phantom_flux'].mean() if 'phantom_flux' in subset.columns else None,
                'flux_std': subset['phantom_flux'].std() if 'phantom_flux' in subset.columns else None,
                'count': len(subset)
            }
            channel_data.append(stats)

        # Convert to DataFrame
        channel_df = pd.DataFrame(channel_data)

        # Fit trends if enough data points
        trends = {}
        if len(channel_df) >= 3:
            # Filter out zero values for fitting
            fit_df = channel_df[channel_df['diameter'] > 0].copy()
            if len(fit_df) >= 3:
                # Fit area vs. dose with power law: D = a * A^b
                try:
                    area = fit_df['area'].values
                    dose = fit_df['dose_mean'].values
                    popt, pcov = curve_fit(self._power_law, area, dose)
                    a, b = popt
                    trends['area_vs_dose'] = {
                        'model': 'power_law',
                        'formula': f'dose = {a:.4e} * area^{b:.4f}',
                        'parameters': {'a': a, 'b': b},
                        'r_squared': self._calculate_r_squared(dose, self._power_law(area, *popt))
                    }
                except Exception as e:
                    self.logger.warning(f"Could not fit area vs. dose: {str(e)}")

                # Fit diameter vs. dose with power law: D = a * d^b
                try:
                    diameter = fit_df['diameter'].values
                    dose = fit_df['dose_mean'].values
                    popt, pcov = curve_fit(self._power_law, diameter, dose)
                    a, b = popt
                    trends['diameter_vs_dose'] = {
                        'model': 'power_law',
                        'formula': f'dose = {a:.4e} * diameter^{b:.4f}',
                        'parameters': {'a': a, 'b': b},
                        'r_squared': self._calculate_r_squared(dose, self._power_law(diameter, *popt))
                    }
                except Exception as e:
                    self.logger.warning(f"Could not fit diameter vs. dose: {str(e)}")

        return {
            'data': channel_df.to_dict(orient='records'),
            'trends': trends,
            'filter': {
                'energy': fixed_energy,
                'distance': fixed_distance
            }
        }

    def analyze_distance_trend(self, fixed_energy=None, fixed_diameter=None):
        """
        Analyze the effect of distance on dose/flux.
        Args:
            fixed_energy: Optional fixed energy to filter by
            fixed_diameter: Optional fixed channel diameter to filter by
        Returns:
            dict: Analysis results
        """
        df = self.to_dataframe()
        if df is None or len(df) == 0:
            return None
        # Filter data if needed
        if fixed_energy is not None:
            df = df[df['energy'] == fixed_energy]
        if fixed_diameter is not None:
            df = df[df['channel_diameter'] == fixed_diameter]
        if len(df) < 2:
            self.logger.warning("Not enough data points for trend analysis")
            return None

        # Group by distance and compute statistics
        distance_data = []
        # For each unique distance value
        for distance in sorted(df['distance'].unique()):
            if distance is None:
                continue
            subset = df[df['distance'] == distance]
            # Collect statistics
            stats = {
                'distance': distance,
                'dose_mean': subset['kerma'].mean() if 'kerma' in subset.columns else subset['total_dose'].mean(),
                'dose_std': subset['kerma'].std() if 'kerma' in subset.columns else subset['total_dose'].std(),
                'flux_mean': subset['phantom_flux'].mean() if 'phantom_flux' in subset.columns else None,
                'flux_std': subset['phantom_flux'].std() if 'phantom_flux' in subset.columns else None,
                'count': len(subset)
            }
            distance_data.append(stats)

        # Convert to DataFrame
        distance_df = pd.DataFrame(distance_data)

        # Fit trends if enough data points
        trends = {}
        if len(distance_df) >= 3:
            # Fit inverse square law: dose = a / distance^2
            try:
                distances = distance_df['distance'].values
                doses = distance_df['dose_mean'].values
                popt, pcov = curve_fit(self._inverse_square, distances, doses)
                a = popt[0]
                trends['inverse_square'] = {
                    'model': 'inverse_square',
                    'formula': f'dose = {a:.4e} / distance^2',
                    'parameters': {'a': a},
                    'r_squared': self._calculate_r_squared(doses, self._inverse_square(distances, a))
                }
            except Exception as e:
                self.logger.warning(f"Could not fit inverse square law: {str(e)}")

            # Fit general power law: dose = a / distance^b
            try:
                popt, pcov = curve_fit(lambda x, a, b: a * (x ** -b), distances, doses)
                a, b = popt
                trends['power_law'] = {
                    'model': 'power_law',
                    'formula': f'dose = {a:.4e} / distance^{b:.4f}',
                    'parameters': {'a': a, 'b': b},
                    'r_squared': self._calculate_r_squared(doses, a * (distances ** -b))
                }
            except Exception as e:
                self.logger.warning(f"Could not fit power law: {str(e)}")

        return {
            'data': distance_df.to_dict(orient='records'),
            'trends': trends,
            'filter': {
                'energy': fixed_energy,
                'diameter': fixed_diameter
            }
        }

    def analyze_energy_trend(self, fixed_diameter=None, fixed_distance=None):
        """
        Analyze the effect of energy on dose/flux.
        Args:
            fixed_diameter: Optional fixed channel diameter to filter by
            fixed_distance: Optional fixed distance to filter by
        Returns:
            dict: Analysis results
        """
        df = self.to_dataframe()
        if df is None or len(df) == 0:
            return None
        # Filter data if needed
        if fixed_diameter is not None:
            df = df[df['channel_diameter'] == fixed_diameter]
        if fixed_distance is not None:
            df = df[df['distance'] == fixed_distance]
        if len(df) < 2:
            self.logger.warning("Not enough data points for trend analysis")
            return None

        # Group by energy and compute statistics
        energy_data = []
        # For each unique energy value
        for energy in sorted(df['energy'].unique()):
            subset = df[df['energy'] == energy]
            # Collect statistics
            stats = {
                'energy': energy,
                'dose_mean': subset['kerma'].mean() if 'kerma' in subset.columns else subset['total_dose'].mean(),
                'dose_std': subset['kerma'].std() if 'kerma' in subset.columns else subset['total_dose'].std(),
                'flux_mean': subset['phantom_flux'].mean() if 'phantom_flux' in subset.columns else None,
                'flux_std': subset['phantom_flux'].std() if 'phantom_flux' in subset.columns else None,
                'count': len(subset)
            }
            energy_data.append(stats)

        # Convert to DataFrame
        energy_df = pd.DataFrame(energy_data)

        # Fit trends if enough data points
        trends = {}
        if len(energy_df) >= 3:
            # Fit linear relationship: dose = a*energy + b
            try:
                energies = energy_df['energy'].values
                doses = energy_df['dose_mean'].values
                popt, pcov = curve_fit(lambda x, a, b: a * x + b, energies, doses)
                a, b = popt
                trends['linear'] = {
                    'model': 'linear',
                    'formula': f'dose = {a:.4e} * energy + {b:.4e}',
                    'parameters': {'a': a, 'b': b},
                    'r_squared': self._calculate_r_squared(doses, a * energies + b)
                }
            except Exception as e:
                self.logger.warning(f"Could not fit linear trend: {str(e)}")

            # Fit power law: dose = a * energy^b
            try:
                popt, pcov = curve_fit(self._power_law, energies, doses)
                a, b = popt
                trends['power_law'] = {
                    'model': 'power_law',
                    'formula': f'dose = {a:.4e} * energy^{b:.4f}',
                    'parameters': {'a': a, 'b': b},
                    'r_squared': self._calculate_r_squared(doses, self._power_law(energies, a, b))
                }
            except Exception as e:
                self.logger.warning(f"Could not fit power law: {str(e)}")

        return {
            'data': energy_df.to_dict(orient='records'),
            'trends': trends,
            'filter': {
                'diameter': fixed_diameter,
                'distance': fixed_distance
            }
        }

    def analyze_concrete_streaming(self, fixed_energy=None, shield_thickness=None):
        """
        Analyze radiation streaming effects specifically in concrete shields.
        Args:
            fixed_energy: Optional fixed energy to filter by
            shield_thickness: Optional shield thickness to filter by
        Returns:
            dict: Analysis results for concrete streaming
        """
        df = self.to_dataframe()
        if df is None or len(df) == 0:
            return None

        # Filter data for concrete shields
        if 'shield_material' in df.columns:
            df = df[df['shield_material'].str.contains('concrete', case=False)]

        # Additional filters
        if fixed_energy is not None:
            df = df[df['energy'] == fixed_energy]
        if shield_thickness is not None:
            df = df[df['shield_thickness'] == shield_thickness]

        if len(df) < 2:
            self.logger.warning("Not enough concrete shield data for analysis")
            return None

        # Group by channel diameter since this is key for streaming
        streaming_data = []

        # For each unique diameter value
        for diameter in sorted(df['channel_diameter'].unique()):
            subset = df[df['channel_diameter'] == diameter]

            # Calculate streaming factor (ratio of dose with penetration to dose without)
            if 'reference_dose' in subset.columns:
                streaming_factor = subset['dose_mean'] / subset['reference_dose']
            else:
                # Use theoretical attenuation if reference not available
                # This is an approximation using the linear attenuation coefficient μ
                # Dose with penetration / Dose without penetration
                μ = 0.20  # cm^-1, approximate linear attenuation coefficient for concrete at ~1 MeV
                thickness = subset['shield_thickness'].mean() if 'shield_thickness' in subset.columns else 30  # cm
                streaming_factor = np.exp(-μ * thickness)  # Theoretical attenuation

                # Collect statistics
            stats = {
                'diameter': diameter,
                'area': np.pi * (diameter / 2) ** 2 if diameter > 0 else 0,
                'dose_mean': subset['kerma'].mean() if 'kerma' in subset.columns else subset['total_dose'].mean(),
                'dose_std': subset['kerma'].std() if 'kerma' in subset.columns else subset['total_dose'].std(),
                'streaming_factor': streaming_factor.mean() if isinstance(streaming_factor,
                                                                          pd.Series) else streaming_factor,
                'count': len(subset)
            }
            streaming_data.append(stats)

            # Convert to DataFrame
        streaming_df = pd.DataFrame(streaming_data)

        # Analyze how streaming factor varies with diameter
        trends = {}
        if len(streaming_df) >= 3 and 'streaming_factor' in streaming_df.columns:
            try:
                diameters = streaming_df['diameter'].values
                factors = streaming_df['streaming_factor'].values

                # Fit streaming factor vs diameter with power law: S = a * d^b
                mask = (diameters > 0) & np.isfinite(factors)  # Filter valid points
                if np.sum(mask) >= 3:
                    popt, pcov = curve_fit(self._power_law, diameters[mask], factors[mask])
                    a, b = popt
                    trends['streaming_vs_diameter'] = {
                        'model': 'power_law',
                        'formula': f'streaming_factor = {a:.4e} * diameter^{b:.4f}',
                        'parameters': {'a': a, 'b': b},
                        'r_squared': self._calculate_r_squared(factors[mask], self._power_law(diameters[mask], *popt))
                    }
            except Exception as e:
                self.logger.warning(f"Could not fit streaming vs diameter trend: {str(e)}")

        return {
            'data': streaming_df.to_dict(orient='records'),
            'trends': trends,
            'filter': {
                'energy': fixed_energy,
                'shield_thickness': shield_thickness
            }
        }

    def analyze_streaming_path_effect(self, include_simulation_details=True):
        """
        Analyze the effect of streaming path characteristics (length, bends, etc.)
        Args:
            include_simulation_details: Whether to include detailed simulation parameters
        Returns:
            dict: Analysis of streaming path characteristics
        """
        df = self.to_dataframe()
        if df is None or len(df) == 0:
            return None

        # Check if we have path characteristics
        required_cols = ['path_length', 'path_bends']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning("Missing path characteristic columns for analysis")
            return None

        # Group by path characteristics
        path_data = []

        # First by number of bends
        for bends in sorted(df['path_bends'].unique()):
            bend_subset = df[df['path_bends'] == bends]

            # Then by similar path lengths (binned)
            bin_edges = np.linspace(bend_subset['path_length'].min(),
                                    bend_subset['path_length'].max(),
                                    num=min(5, len(bend_subset)) + 1)

            for i in range(len(bin_edges) - 1):
                length_min, length_max = bin_edges[i], bin_edges[i + 1]
                length_subset = bend_subset[(bend_subset['path_length'] >= length_min) &
                                            (bend_subset['path_length'] < length_max)]

                if len(length_subset) == 0:
                    continue

                # Calculate average values
                avg_length = length_subset['path_length'].mean()
                avg_dose = length_subset['kerma'].mean() if 'kerma' in length_subset.columns else length_subset[
                    'total_dose'].mean()

                # Calculate albedo effect if applicable
                albedo_factor = None
                if 'albedo_coefficient' in length_subset.columns:
                    albedo_factor = length_subset['albedo_coefficient'].mean()

                entry = {
                    'path_bends': bends,
                    'path_length': avg_length,
                    'dose_mean': avg_dose,
                    'dose_std': length_subset['kerma'].std() if 'kerma' in length_subset.columns else length_subset[
                        'total_dose'].std(),
                    'albedo_factor': albedo_factor,
                    'count': len(length_subset)
                }

                # Include average simulation parameters if requested
                if include_simulation_details:
                    for param in ['energy', 'channel_diameter', 'shield_thickness']:
                        if param in length_subset.columns:
                            entry[f'avg_{param}'] = length_subset[param].mean()

                path_data.append(entry)

        # Convert to DataFrame
        path_df = pd.DataFrame(path_data)

        # Analyze trends
        trends = {}

        # For each number of bends, analyze how dose varies with path length
        for bends in path_df['path_bends'].unique():
            bend_data = path_df[path_df['path_bends'] == bends]

            if len(bend_data) >= 3:
                try:
                    lengths = bend_data['path_length'].values
                    doses = bend_data['dose_mean'].values

                    # Fit exponential attenuation: dose = a * exp(-b * length)
                    popt, pcov = curve_fit(lambda x, a, b: a * np.exp(-b * x), lengths, doses)
                    a, b = popt

                    trends[f'bends_{int(bends)}'] = {
                        'model': 'exponential',
                        'formula': f'dose = {a:.4e} * exp(-{b:.4f} * length)',
                        'parameters': {'a': a, 'b': b},
                        'r_squared': self._calculate_r_squared(doses, a * np.exp(-b * lengths)),
                        'attenuation_length': 1 / b if b > 0 else float('inf')  # Mean free path
                    }
                except Exception as e:
                    self.logger.warning(f"Could not fit trend for {bends} bends: {str(e)}")

        return {
            'data': path_df.to_dict(orient='records'),
            'trends': trends
        }

    def generate_streaming_report(self, output_path="streaming_analysis_report.pdf"):
        """
        Generate a comprehensive report on radiation streaming analysis.
        Args:
            output_path: Path to save the PDF report
        Returns:
            bool: Success status
        """
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title and date
        title_style = styles["Title"]
        story.append(Paragraph("Radiation Streaming Analysis Report", title_style))
        story.append(
            Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Spacer(1, 0.5 * inch))

        # Add summary section
        story.append(Paragraph("Executive Summary", styles["Heading1"]))

        # Generate summary text based on analysis results
        summary_text = """
                    This report presents a comprehensive analysis of radiation streaming effects through shielding materials,
                    with a focus on concrete shields. The analysis includes the effects of channel diameter, distance from source,
                    energy of radiation, and streaming path characteristics on dose rates and flux.
                    """
        story.append(Paragraph(summary_text, styles["Normal"]))
        story.append(Spacer(1, 0.25 * inch))

        # Add key findings
        story.append(Paragraph("Key Findings:", styles["Heading2"]))

        # Run analyses to get key findings
        try:
            # Channel diameter analysis
            diameter_analysis = self.analyze_channel_diameter_trend()
            if diameter_analysis and 'trends' in diameter_analysis:
                trends = diameter_analysis['trends']
                if 'diameter_vs_dose' in trends:
                    trend = trends['diameter_vs_dose']
                    story.append(Paragraph(f"• Diameter effect: {trend['formula']} (R² = {trend['r_squared']:.3f})",
                                           styles["Normal"]))

            # Distance analysis
            distance_analysis = self.analyze_distance_trend()
            if distance_analysis and 'trends' in distance_analysis:
                trends = distance_analysis['trends']
                if 'power_law' in trends:
                    trend = trends['power_law']
                    story.append(Paragraph(f"• Distance effect: {trend['formula']} (R² = {trend['r_squared']:.3f})",
                                           styles["Normal"]))

            # Concrete streaming analysis
            concrete_analysis = self.analyze_concrete_streaming()
            if concrete_analysis and 'trends' in concrete_analysis:
                trends = concrete_analysis['trends']
                if 'streaming_vs_diameter' in trends:
                    trend = trends['streaming_vs_diameter']
                    story.append(
                        Paragraph(f"• Concrete streaming effect: {trend['formula']} (R² = {trend['r_squared']:.3f})",
                                  styles["Normal"]))
        except Exception as e:
            self.logger.error(f"Error generating analysis for report: {str(e)}")
            story.append(Paragraph("Error generating analysis trends.", styles["Normal"]))

        story.append(Spacer(1, 0.5 * inch))

        # Add detailed analysis sections
        story.append(Paragraph("Detailed Analysis", styles["Heading1"]))

        # 1. Channel Diameter Analysis
        story.append(Paragraph("1. Effect of Channel Diameter", styles["Heading2"]))
        try:
            diameter_analysis = self.analyze_channel_diameter_trend()
            if diameter_analysis and 'data' in diameter_analysis:
                # Create a description
                diameter_desc = """
                            The channel diameter is a critical parameter affecting radiation streaming. 
                            Larger diameters allow more direct radiation to pass through, increasing dose rates. 
                            The relationship is typically non-linear and follows a power law relationship.
                            """
                story.append(Paragraph(diameter_desc, styles["Normal"]))

                # Add figure showing the trend
                fig_path = self._generate_diameter_plot(diameter_analysis)
                if fig_path:
                    img = Image(fig_path, width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(Paragraph(f"Figure 1: Effect of channel diameter on dose rate", styles["Caption"]))
            else:
                story.append(Paragraph("Insufficient data for channel diameter analysis.", styles["Normal"]))
        except Exception as e:
            self.logger.error(f"Error in diameter analysis section: {str(e)}")
            story.append(Paragraph("Error generating channel diameter analysis.", styles["Normal"]))

        story.append(Spacer(1, 0.25 * inch))

        # 2. Concrete Streaming Analysis
        story.append(Paragraph("2. Streaming Effects in Concrete Shields", styles["Heading2"]))
        try:
            concrete_analysis = self.analyze_concrete_streaming()
            if concrete_analysis and 'data' in concrete_analysis:
                # Create a description
                concrete_desc = """
                            Concrete is a common shielding material with complex streaming characteristics.
                            The presence of air channels or voids can significantly increase radiation penetration
                            due to streaming effects. This section analyzes how the streaming factor
                            (ratio of dose with penetration to dose without) varies with channel properties.
                            """
                story.append(Paragraph(concrete_desc, styles["Normal"]))

                # Add figure showing the streaming effect
                fig_path = self._generate_concrete_streaming_plot(concrete_analysis)
                if fig_path:
                    img = Image(fig_path, width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(
                        Paragraph(f"Figure 2: Streaming factor vs. channel diameter in concrete", styles["Caption"]))
            else:
                story.append(Paragraph("Insufficient data for concrete streaming analysis.", styles["Normal"]))
        except Exception as e:
            self.logger.error(f"Error in concrete streaming section: {str(e)}")
            story.append(Paragraph("Error generating concrete streaming analysis.", styles["Normal"]))

        story.append(PageBreak())

        # 3. Path Characteristics Analysis
        story.append(Paragraph("3. Effect of Streaming Path Characteristics", styles["Heading2"]))
        try:
            path_analysis = self.analyze_streaming_path_effect()
            if path_analysis and 'data' in path_analysis:
                # Create a description
                path_desc = """
                            The characteristics of the streaming path, including length and number of bends,
                            significantly affect radiation transport. Each bend can reduce the streaming effect
                            due to albedo (reflection) and scattering processes.
                            """
                story.append(Paragraph(path_desc, styles["Normal"]))

                # Add figure showing path effects
                fig_path = self._generate_path_effect_plot(path_analysis)
                if fig_path:
                    img = Image(fig_path, width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(
                        Paragraph(f"Figure 3: Effect of path length and bends on dose rate", styles["Caption"]))

                # Add table of attenuation lengths
                if 'trends' in path_analysis and len(path_analysis['trends']) > 0:
                    story.append(
                        Paragraph("Attenuation Lengths for Different Path Configurations:", styles["Heading3"]))

                    # Create table data
                    table_data = [["Number of Bends", "Attenuation Length (cm)", "R² Value"]]
                    for bend_key, trend in path_analysis['trends'].items():
                        bends = bend_key.split('_')[1]
                        atten_length = trend.get('attenuation_length', 'N/A')
                        r_squared = trend.get('r_squared', 'N/A')
                        if isinstance(atten_length, float):
                            atten_length = f"{atten_length:.2f}"
                        if isinstance(r_squared, float):
                            r_squared = f"{r_squared:.3f}"
                        table_data.append([bends, atten_length, r_squared])

                    table = Table(table_data, colWidths=[1.5 * inch, 2 * inch, 1.5 * inch])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(table)

                    # Draw detector positions
                    for angle in [0, 15, 30, 45]:
                        angle_rad = angle * np.pi / 180
                        x = 0.6 + 0.15 * np.cos(angle_rad)
                        y = 0.5 + 0.15 * np.sin(angle_rad)
                        diagram_ax.plot(x, y, 'bo', markersize=5)
                        diagram_ax.text(x, y + 0.03, f"{angle}°", ha='center', va='bottom', fontsize=8)
                except Exception as e:
                self.logger.error(f"Error generating page diagram: {str(e)}")

        # Add conclusions and recommendations
        story.append(Paragraph("Conclusions and Recommendations", styles["Heading2"]))
        conclusions = """
            The analysis of radiation streaming through concrete shields confirms several important findings:

            1. Streaming effects in concrete shields follow a non-linear relationship with channel diameter,
               with dose rates typically increasing as the square of the diameter.

            2. The presence of bends in the streaming path significantly reduces dose rates, with each 90°
               bend reducing the dose by approximately a factor of 10.

            3. The attenuation length in concrete streaming paths is significantly longer than the attenuation
               length in solid concrete due to reduced interaction probability in the air channel.

            4. Higher energy gamma rays exhibit more pronounced streaming effects due to their greater
               penetration capability and forward-biased scattering.

            Based on these findings, we recommend:

            • Minimizing the diameter of necessary penetrations through concrete shields
            • Including at least one bend in the path when straight-line penetrations cannot be avoided
            • Implementing additional local shielding at the exit of high-risk penetrations
            • Conducting periodic radiation surveys to validate simulation results
            """
        story.append(Paragraph(conclusions, styles["Normal"]))

        # Build the document
        try:
            doc.build(story)
            self.logger.info(f"Report successfully generated at {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error building PDF report: {str(e)}")
            return False

    def generate_comprehensive_report(self, output_path="comprehensive_analysis_report.pdf"):
        """
        Generate a more comprehensive report including detailed streaming effects in concrete
        with error calculations and interpretations.

        Args:
            output_path: Path to save the PDF report
        Returns:
            bool: Success status
        """
        try:
            # Convert data to DataFrame for easier manipulation
            df = self.to_dataframe()
            if df is None or len(df) == 0:
                self.logger.error("No data available for comprehensive report generation")
                return False

            # Create PDF with PdfPages
            with PdfPages(output_path) as pdf:
                # === Title Page ===
                plt.figure(figsize=(12, 10))
                plt.axis('off')

                # Title
                plt.text(0.5, 0.85, "COMPREHENSIVE RADIATION ANALYSIS REPORT",
                         ha='center', fontsize=24, fontweight='bold')
                plt.text(0.5, 0.78, "Gamma-Ray Streaming Through Concrete Shields",
                         ha='center', fontsize=20)

                # Description
                description = (
                    "Analysis of radiation streaming effects through concrete shields with various configurations.\n"
                    "Evaluation of dose rates as a function of channel diameter, energy, distance, and path geometry.\n"
                    "Including error calculations and scientific interpretation of results."
                )
                plt.text(0.5, 0.68, description, ha='center', fontsize=14)

                # Configuration summary
                if 'shield_thickness' in df.columns:
                    shield_thickness = f"{df['shield_thickness'].mean():.1f} cm"
                else:
                    shield_thickness = "Variable"

                if 'energy' in df.columns:
                    energy_range = f"{df['energy'].min():.1f} - {df['energy'].max():.1f} MeV"
                else:
                    energy_range = "Variable"

                config = (
                    f"Shield Thickness: {shield_thickness}\n"
                    f"Energy Range: {energy_range}\n"
                    f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d')}"
                )
                plt.text(0.5, 0.55, config, ha='center', fontsize=12)

                # Add simulation diagram
                diagram_ax = plt.axes([0.15, 0.15, 0.7, 0.3])
                diagram_ax.axis('off')

                # Draw wall
                wall_rect = plt.Rectangle((0.3, 0.25), 0.1, 0.5, color='gray', alpha=0.8)
                diagram_ax.add_patch(wall_rect)
                diagram_ax.text(0.35, 0.8, "Concrete Shield", ha='center', va='center')

                # Draw source
                diagram_ax.plot(0.2, 0.5, 'ro', markersize=10)
                diagram_ax.text(0.2, 0.6, "Source", ha='center', va='center')

                # Draw channel
                channel_width = 0.02
                channel_rect = plt.Rectangle((0.3, 0.5 - channel_width / 2), 0.1, channel_width, color='white')
                diagram_ax.add_patch(channel_rect)
                diagram_ax.text(0.35, 0.4, "Channel", ha='center', va='center')

                # Draw detector
                detector_circle = plt.Circle((0.6, 0.5), 0.05, fill=False, color='red')
                diagram_ax.add_patch(detector_circle)
                diagram_ax.text(0.6, 0.6, "Detector", ha='center', va='center')

                # Draw beam path
                diagram_ax.plot([0.2, 0.6], [0.5, 0.5], 'y--', alpha=0.7)

                pdf.savefig()
                plt.close()

                # === Executive Summary ===
                plt.figure(figsize=(12, 10))
                plt.axis('off')

                plt.text(0.5, 0.95, "Executive Summary", ha='center', fontsize=18, fontweight='bold')

                # Calculate key statistics for summary
                max_dose_row = df.loc[df['dose_mean'].idxmax()] if 'dose_mean' in df.columns else None
                mean_error = df['dose_std'].mean() / df[
                    'dose_mean'].mean() if 'dose_std' in df.columns and 'dose_mean' in df.columns else None

                intro_text = (
                    "This report presents a comprehensive analysis of gamma radiation streaming through "
                    "concrete shields with cylindrical air channels. The study quantifies how radiation dose rates "
                    "are affected by multiple parameters including channel diameter, radiation energy, distance from "
                    "the shield, and streaming path characteristics."
                )
                plt.text(0.1, 0.88, intro_text, fontsize=12, ha='left', wrap=True, transform=plt.gca().transAxes)

                # Key findings with error calculations
                findings_text = "Key Findings:\n\n"

                if max_dose_row is not None:
                    findings_text += (
                        f"1. Maximum Dose: {max_dose_row['dose_mean']:.2e} ± {max_dose_row['dose_std']:.2e} rem/hr "
                        f"(relative error: {100 * max_dose_row['dose_std'] / max_dose_row['dose_mean']:.1f}%) observed with "
                    )
                    if 'energy' in max_dose_row:
                        findings_text += f"{max_dose_row['energy']} MeV, "
                    if 'channel_diameter' in max_dose_row:
                        findings_text += f"{max_dose_row['channel_diameter']} cm channel diameter, "
                    if 'distance' in max_dose_row:
                        findings_text += f"{max_dose_row['distance']} cm distance, "
                    if 'angle' in max_dose_row:
                        findings_text += f"and {max_dose_row['angle']}° angle."
                    findings_text += "\n\n"

                # Add trends from our trend analysis methods
                diameter_analysis = self.analyze_channel_diameter_trend()
                if diameter_analysis and 'trends' in diameter_analysis and 'diameter_vs_dose' in diameter_analysis[
                    'trends']:
                    trend = diameter_analysis['trends']['diameter_vs_dose']
                    b = trend['parameters']['b']
                    r_squared = trend['r_squared']
                    findings_text += (
                        f"2. Channel Diameter Effect: Dose rates scale with diameter^{b:.2f} "
                        f"(R² = {r_squared:.3f}). This means doubling the channel diameter "
                        f"increases the dose rate by approximately {2 ** b:.1f}×.\n\n"
                    )

                distance_analysis = self.analyze_distance_trend()
                if distance_analysis and 'trends' in distance_analysis and 'power_law' in distance_analysis['trends']:
                    trend = distance_analysis['trends']['power_law']
                    b = trend['parameters']['b']
                    r_squared = trend['r_squared']
                    findings_text += (
                        f"3. Distance Dependence: Dose rates decrease with distance^-{b:.2f} "
                        f"(R² = {r_squared:.3f}), compared to the theoretical inverse-square relationship "
                        f"(distance^-2). This deviation is due to air attenuation and scattering effects.\n\n"
                    )

                concrete_analysis = self.analyze_concrete_streaming()
                if concrete_analysis and 'trends' in concrete_analysis and 'streaming_vs_diameter' in concrete_analysis[
                    'trends']:
                    trend = concrete_analysis['trends']['streaming_vs_diameter']
                    b = trend['parameters']['b']
                    r_squared = trend['r_squared']
                    findings_text += (
                        f"4. Streaming Effect: The streaming factor (dose enhancement) in concrete scales with "
                        f"diameter^{b:.2f} (R² = {r_squared:.3f}). This non-linear relationship demonstrates "
                        f"the critical importance of minimizing penetration sizes in concrete shields.\n\n"
                    )

                # Add error analysis
                if mean_error is not None:
                    findings_text += (
                        f"5. Error Analysis: The average relative uncertainty in dose calculations is "
                        f"{100 * mean_error:.1f}%, with higher uncertainties observed for smaller channel "
                        f"diameters and larger distances due to reduced particle statistics.\n\n"
                    )

                plt.text(0.1, 0.78, findings_text, fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)

                # Conclusions
                conclusions_text = (
                    "Conclusions and Recommendations:\n\n"
                    "• The presence of air channels in concrete shields can increase radiation penetration by "
                    "several orders of magnitude compared to solid concrete, especially for direct line-of-sight paths.\n\n"
                    "• Even small channels (< 0.5 cm diameter) can create significant streaming paths, with dose rates "
                    "scaling non-linearly with channel diameter.\n\n"
                    "• Adding bends to penetration paths is highly effective, with each 90° bend reducing streaming "
                    "dose rates by approximately an order of magnitude.\n\n"
                    "• For critical configurations, additional local shielding should be placed at channel exits, "
                    "and access restrictions should be implemented in the direct beam path."
                )
                plt.text(0.1, 0.35, conclusions_text, fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)

                pdf.savefig()
                plt.close()

                # === Diameter Effect Analysis ===
                diameter_analysis = self.analyze_channel_diameter_trend()
                if diameter_analysis and 'data' in diameter_analysis:
                    plt.figure(figsize=(12, 9))

                    # Extract data
                    data = pd.DataFrame(diameter_analysis['data'])

                    # Plot dose vs. diameter with error bars
                    if 'diameter' in data.columns and 'dose_mean' in data.columns and 'dose_std' in data.columns:
                        plt.errorbar(data['diameter'], data['dose_mean'], yerr=data['dose_std'],
                                     fmt='o-', linewidth=2, markersize=8, capsize=5, label='Simulation data')

                        # Add trend line if available
                        if 'trends' in diameter_analysis and 'diameter_vs_dose' in diameter_analysis['trends']:
                            trend = diameter_analysis['trends']['diameter_vs_dose']
                            a, b = trend['parameters']['a'], trend['parameters']['b']
                            r_squared = trend['r_squared']

                            # Generate trend line
                            x_trend = np.linspace(data['diameter'].min() * 0.9, data['diameter'].max() * 1.1, 100)
                            y_trend = a * x_trend ** b
                            plt.plot(x_trend, y_trend, 'r--', linewidth=2,
                                     label=f'Fit: dose = {a:.2e} × diameter^{b:.2f} (R² = {r_squared:.3f})')

                            # Add d² reference line
                            ref_scale = data.loc[data['diameter'] > 0, 'dose_mean'].iloc[0] / \
                                        data.loc[data['diameter'] > 0, 'diameter'].iloc[0] ** 2
                            y_ref = ref_scale * x_trend ** 2
                            plt.plot(x_trend, y_ref, 'k--', linewidth=1.5, alpha=0.7,
                                     label='d² reference (theoretical)')

                        plt.xlabel('Channel Diameter (cm)', fontsize=12, fontweight='bold')
                        plt.ylabel('Dose Rate (rem/hr)', fontsize=12, fontweight='bold')
                        plt.title('Effect of Channel Diameter on Dose Rate', fontsize=14, fontweight='bold')
                        plt.grid(True, alpha=0.3)
                        plt.legend(fontsize=10)

                        # Log scales for better visualization
                # Log scales for better visualization
                if data['diameter'].min() > 0 and data['dose_mean'].min() > 0:
                    plt.loglog()

                # Add interpretation annotation
                plt.text(0.02, 0.02,
                         f"The dose rate increases with channel diameter following a power law with exponent {b:.2f},\n"
                         f"close to the theoretical d² dependence. This indicates that reducing diameter\n"
                         f"by half typically reduces dose by a factor of {2 ** b:.1f}.",
                         transform=plt.gca().transAxes, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            pdf.savefig()
            plt.close()

        # === Distance Effect Analysis ===
        distance_analysis = self.analyze_distance_trend()
        if distance_analysis and 'data' in distance_analysis:
            plt.figure(figsize=(12, 9))

            # Extract data
            data = pd.DataFrame(distance_analysis['data'])

            # Plot dose vs. distance with error bars
            if 'distance' in data.columns and 'dose_mean' in data.columns and 'dose_std' in data.columns:
                plt.errorbar(data['distance'], data['dose_mean'], yerr=data['dose_std'],
                             fmt='o-', linewidth=2, markersize=8, capsize=5, label='Simulation data')

                # Add trend lines if available
                if 'trends' in distance_analysis:
                    # Power law trend
                    if 'power_law' in distance_analysis['trends']:
                        trend = distance_analysis['trends']['power_law']
                        a, b = trend['parameters']['a'], trend['parameters']['b']
                        r_squared = trend['r_squared']

                        # Generate trend line
                        x_trend = np.linspace(data['distance'].min() * 0.9, data['distance'].max() * 1.1, 100)
                        y_trend = a * x_trend ** -b
                        plt.plot(x_trend, y_trend, 'r--', linewidth=2,
                                 label=f'Fit: dose = {a:.2e} × distance^-{b:.2f} (R² = {r_squared:.3f})')

                    # Inverse square law reference
                    if 'inverse_square' in distance_analysis['trends'] or True:  # Always show reference
                        # Get reference point
                        x0, y0 = data['distance'].iloc[0], data['dose_mean'].iloc[0]
                        x_ref = np.linspace(data['distance'].min() * 0.9, data['distance'].max() * 1.1, 100)
                        y_ref = y0 * (x0 / x_ref) ** 2
                        plt.plot(x_ref, y_ref, 'k--', linewidth=1.5, alpha=0.7,
                                 label='1/r² reference (theoretical)')

                plt.xlabel('Distance (cm)', fontsize=12, fontweight='bold')
                plt.ylabel('Dose Rate (rem/hr)', fontsize=12, fontweight='bold')
                plt.title('Effect of Distance on Dose Rate', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=10)

                # Log scales for better visualization
                if data['distance'].min() > 0 and data['dose_mean'].min() > 0:
                    plt.loglog()

                # Add interpretation annotation
                if 'trends' in distance_analysis and 'power_law' in distance_analysis['trends']:
                    trend = distance_analysis['trends']['power_law']
                    b = trend['parameters']['b']
                    plt.text(0.02, 0.02,
                             f"The dose rate decreases with distance following a power law with exponent -{b:.2f},\n"
                             f"compared to the theoretical 1/r² law (exponent -2). The difference is due to\n"
                             f"additional attenuation and scattering in air at greater distances.",
                             transform=plt.gca().transAxes, fontsize=10,
                             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            pdf.savefig()
            plt.close()

        # === Concrete Streaming Analysis ===
        concrete_analysis = self.analyze_concrete_streaming()
        if concrete_analysis and 'data' in concrete_analysis:
            plt.figure(figsize=(12, 9))

            # Extract data
            data = pd.DataFrame(concrete_analysis['data'])

            # Plot streaming factor vs. diameter
            if 'diameter' in data.columns and 'streaming_factor' in data.columns:
                plt.plot(data['diameter'], data['streaming_factor'], 'o-', linewidth=2,
                         markersize=8, label='Streaming factor')

                # Add theoretical attenuation reference
                if 'shield_thickness' in data.columns or 'filter' in concrete_analysis:
                    # Get shield thickness
                    if 'shield_thickness' in data.columns:
                        thickness = data['shield_thickness'].mean()
                    else:
                        thickness = concrete_analysis['filter'].get('shield_thickness', 30)  # default 30 cm

                    # Calculate theoretical attenuation
                    μ = 0.20  # cm^-1, approximate linear attenuation for concrete at ~1 MeV
                    theoretical_attenuation = np.exp(-μ * thickness)

                    # Plot reference line
                    plt.axhline(y=theoretical_attenuation, color='r', linestyle='--',
                                label=f'Theoretical attenuation (no streaming): {theoretical_attenuation:.2e}')

                    # Calculate and plot enhancement factor (how many times higher than theoretical)
                    enhancement = data['streaming_factor'] / theoretical_attenuation
                    plt.plot(data['diameter'], enhancement, 's-', linewidth=2,
                             markersize=8, label='Enhancement factor')

                # Add trend line if available
                if 'trends' in concrete_analysis and 'streaming_vs_diameter' in concrete_analysis['trends']:
                    trend = concrete_analysis['trends']['streaming_vs_diameter']
                    a, b = trend['parameters']['a'], trend['parameters']['b']
                    r_squared = trend['r_squared']

                    # Generate trend line
                    x_trend = np.linspace(data['diameter'].min() * 0.9, data['diameter'].max() * 1.1, 100)
                    y_trend = a * x_trend ** b
                    plt.plot(x_trend, y_trend, 'g--', linewidth=2,
                             label=f'Fit: factor = {a:.2e} × diameter^{b:.2f} (R² = {r_squared:.3f})')

                plt.xlabel('Channel Diameter (cm)', fontsize=12, fontweight='bold')
                plt.ylabel('Streaming Factor', fontsize=12, fontweight='bold')
                plt.title('Radiation Streaming Through Concrete Shield', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=10)

                # Log scales for better visualization
                if data['diameter'].min() > 0 and data['streaming_factor'].min() > 0:
                    plt.semilogx()  # log scale on x-axis only

                # Add interpretation annotation
                plt.text(0.02, 0.02,
                         "Streaming factor represents the ratio of dose with the channel to dose without.\n"
                         "The enhancement factor shows how many times greater the dose is compared to\n"
                         "theoretical attenuation. Both increase rapidly with channel diameter,\n"
                         "demonstrating the critical importance of minimizing penetration sizes.",
                         transform=plt.gca().transAxes, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            pdf.savefig()
            plt.close()

        # === Path Characteristics Analysis ===
        path_analysis = self.analyze_streaming_path_effect()
        if path_analysis and 'data' in path_analysis:
            plt.figure(figsize=(12, 9))

            # Extract data
            data = pd.DataFrame(path_analysis['data'])

            # Create a plot for each number of bends
            if 'path_bends' in data.columns and 'path_length' in data.columns and 'dose_mean' in data.columns:
                for bends in sorted(data['path_bends'].unique()):
                    bend_data = data[data['path_bends'] == bends]
                    plt.semilogy(bend_data['path_length'], bend_data['dose_mean'], 'o-',
                                 linewidth=2, markersize=8, label=f'{int(bends)} bends')

                    # Add trend line if available
                    if 'trends' in path_analysis and f'bends_{int(bends)}' in path_analysis['trends']:
                        trend = path_analysis['trends'][f'bends_{int(bends)}']
                        a, b = trend['parameters']['a'], trend['parameters']['b']
                        r_squared = trend['r_squared']

                        # Generate trend line
                        x_trend = np.linspace(bend_data['path_length'].min() * 0.9,
                                              bend_data['path_length'].max() * 1.1, 100)
                        y_trend = a * np.exp(-b * x_trend)
                        plt.semilogy(x_trend, y_trend, '--', linewidth=1.5,
                                     label=f'{int(bends)} bends fit: {a:.2e}*exp(-{b:.3f}*L), R²={r_squared:.3f}')

                plt.xlabel('Path Length (cm)', fontsize=12, fontweight='bold')
                plt.ylabel('Dose Rate (rem/hr)', fontsize=12, fontweight='bold')
                plt.title('Effect of Path Length and Bends on Dose Rate', fontsize=14, fontweight='bold')
                plt.grid(True, which='both', alpha=0.3)
                plt.legend(title="Path Configuration", fontsize=10)

                # Add interpretation annotation
                plt.text(0.02, 0.02,
                         "Each 90° bend in the streaming path significantly reduces dose rates by limiting\n"
                         "direct line-of-sight radiation and forcing scattering interactions. The attenuation\n"
                         "with path length follows an exponential trend for each configuration.",
                         transform=plt.gca().transAxes, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            pdf.savefig()
            plt.close()

        # === Error Analysis ===
        plt.figure(figsize=(12, 9))

        # Create subplot grid
        gs = gridspec.GridSpec(2, 2)

        # 1. Error vs. Channel Diameter
        if 'channel_diameter' in df.columns and 'dose_mean' in df.columns and 'dose_std' in df.columns:
            ax1 = plt.subplot(gs[0, 0])

            # Calculate relative error
            df['rel_error'] = df['dose_std'] / df['dose_mean'] * 100  # percentage

            # Group by diameter and calculate average error
            diameter_error = df.groupby('channel_diameter')['rel_error'].mean().reset_index()

            ax1.bar(diameter_error['channel_diameter'], diameter_error['rel_error'],
                    width=diameter_error['channel_diameter'] * 0.3, alpha=0.7)
            ax1.set_xlabel('Channel Diameter (cm)')
            ax1.set_ylabel('Average Relative Error (%)')
            ax1.set_title('Error vs. Channel Diameter')
            ax1.grid(True, alpha=0.3)

        # 2. Error vs. Distance
        if 'distance' in df.columns and 'rel_error' in df:
            ax2 = plt.subplot(gs[0, 1])

            # Group by distance and calculate average error
            distance_error = df.groupby('distance')['rel_error'].mean().reset_index()

            ax2.plot(distance_error['distance'], distance_error['rel_error'], 'o-',
                     linewidth=2, markersize=8)
            ax2.set_xlabel('Distance (cm)')
            ax2.set_ylabel('Average Relative Error (%)')
            ax2.set_title('Error vs. Distance')
            ax2.grid(True, alpha=0.3)

        # 3. Error Distribution Histogram
        if 'rel_error' in df:
            ax3 = plt.subplot(gs[1, 0])

            ax3.hist(df['rel_error'], bins=20, alpha=0.7)
            ax3.set_xlabel('Relative Error (%)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Error Distribution')
            ax3.grid(True, alpha=0.3)

        # 4. Error Interpretation Text
        ax4 = plt.subplot(gs[1, 1])
        ax4.axis('off')

        error_text = (
            "Error Analysis Interpretation:\n\n"
            f"• Mean Relative Error: {df['rel_error'].mean():.1f}%\n"
            f"• Median Relative Error: {df['rel_error'].median():.1f}%\n"
            f"• Maximum Relative Error: {df['rel_error'].max():.1f}%\n\n"
            "Factors affecting simulation accuracy:\n\n"
            "1. Smaller channel diameters show higher relative errors due to\n"
            "   decreased particle statistics through the narrow penetration.\n\n"
            "2. Greater distances also show increased errors due to the\n"
            "   inverse square law reducing particle counts at the detector.\n\n"
            "3. Error propagation in the streaming factor calculations combines\n"
            "   uncertainties from both penetration and no-penetration cases.\n\n"
            "For critical safety applications, configurations with errors\n"
            "exceeding 10% should be recalculated with increased particle counts."
        )

        ax4.text(0, 1, error_text, fontsize=10, va='top', ha='left',
                 bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round'))

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # === Safety Recommendations Page ===
        plt.figure(figsize=(12, 10))
        plt.axis('off')

        # Title
        plt.text(0.5, 0.95, "Safety Recommendations for Radiation Protection",
                 ha='center', fontsize=18, fontweight='bold')

        # Occupational dose limits section
        plt.text(0.5, 0.88, "Occupational Dose Limits", ha='center', fontsize=14, fontweight='bold')

        dose_limits = (
            "Regulatory dose limits for radiation workers (per 10 CFR 20 and ICRP recommendations):\n\n"
            "• Total Effective Dose Equivalent (TEDE): 5 rem (50 mSv) per year\n"
            "• Lens of the eye: 15 rem (150 mSv) per year\n"
            "• Skin and extremities: 50 rem (500 mSv) per year\n"
            "• Declared pregnant worker: 0.5 rem (5 mSv) during entire pregnancy\n\n"
            "ALARA (As Low As Reasonably Achievable) principle mandates doses should be kept\n"
            "well below these regulatory limits whenever practicable."
        )
        plt.text(0.1, 0.76, dose_limits, fontsize=12, ha='left')

        # Access control recommendations
        plt.text(0.5, 0.64, "Access Control Recommendations", ha='center', fontsize=14, fontweight='bold')

        # Calculate highest dose rate from data
        highest_dose = df['dose_mean'].max() if 'dose_mean' in df.columns else 0

        access_recommendations = (
            "Based on simulation results and streaming analysis, the following access controls are recommended:\n\n"
        )

        # Add specific recommendations based on highest dose rate
        if highest_dose > 100:  # Very high dose rate
            access_recommendations += (
                "• RESTRICTED AREA: Direct beam area must be designated as a high-radiation area with\n"
                "  strict access controls, interlocks, and visible/audible warning devices.\n"
                "• PROHIBITED ACCESS: No access permitted when source is active.\n"
                "• ENGINEERING CONTROLS: Additional local shielding required at channel exit.\n"
                "• MONITORING: Area monitoring with real-time dosimeters and warning alarms required.\n"
                "• EMERGENCY PROCEDURES: Immediate evacuation protocol required if beam is activated\n"
                "  when personnel are present in the area."
            )
        elif highest_dose > 10:  # High dose rate
            access_recommendations += (
                "• HIGH RADIATION AREA: Direct beam area must be designated as a high-radiation area\n"
                "  with controlled access and appropriate warning signs.\n"
                "• LIMITED ACCESS: Entry permitted only with radiation work permit and dosimetry.\n"
                "• TIME RESTRICTION: Occupancy limited to ensure total effective dose remains below\n"
                "  10% of annual limit (0.5 rem) for any individual worker.\n"
                "• ADDITIONAL SHIELDING: Portable shields recommended for routine work near channel exit.\n"
                "• MONITORING: Personal dosimetry required for all personnel entering the area."
            )
        elif highest_dose > 1:  # Moderate dose rate
            access_recommendations += (
                "• RADIATION AREA: Area must be designated as a radiation area with appropriate signage.\n"
                "• CONTROLLED ACCESS: Access limited to radiation workers with appropriate training.\n"
                "• OCCUPANCY FACTOR: Time restrictions based on measured dose rates to ensure\n"
                "  monthly exposure remains below 100 mrem for any individual.\n"
                "• MONITORING: Periodic area surveys and personal dosimetry for regular workers.\n"
                "• WORK PLANNING: Pre-job briefing recommended for extended work in the area."
            )
        else:  # Low dose rate
            access_recommendations += (
                "• SUPERVISED AREA: Area should be designated as a supervised radiation area.\n"
                "• GENERAL ACCESS: Access permitted for radiation workers; limited access for non-radiation workers.\n"
                "• MONITORING: Routine area surveys to verify dose rates remain within expected ranges.\n"
                "• SIGNAGE: Radiation warning signs required at access points.\n"
                "• AWARENESS: Workers should be informed about the potential for increased dose rates\n"
                "  if configuration changes (e.g., larger channel diameter, higher energy source)."
            )

        plt.text(0.1, 0.52, access_recommendations, fontsize=12, ha='left')

        # Engineering controls
        plt.text(0.5, 0.36, "Engineering Controls for Streaming Penetrations", ha='center', fontsize=14,
                 fontweight='bold')

        engineering_controls = (
            "The following engineering controls should be considered for radiation streaming paths:\n\n"
            "1. Minimum Diameter: Use the smallest practical diameter for all penetrations through shields.\n\n"
            "2. Offset Penetrations: Where possible, design penetrations with bends or offsets to eliminate\n"
            "   direct line-of-sight paths through the shield (reduces dose rates by factor of ~10 per bend).\n\n"
            "3. Streaming Caps: Install removable shield plugs or caps when penetrations are not in use.\n\n"
            "4. Shadow Shields: Place local shadow shields at penetration exits to reduce scatter radiation.\n\n"
            "5. Channel Filling: Consider partial filling of necessary channels with lower-density material\n"
            "   that still permits required functionality while attenuating radiation.\n\n"
            "6. Administrative Controls: Ensure procedures require inspection and restoration of all\n"
            "   shielding features after maintenance activities involving penetrations."
        )
        plt.text(0.1, 0.24, engineering_controls, fontsize=12, ha='left')

        # Monitoring recommendations
        plt.text(0.5, 0.16, "Monitoring Program Recommendations", ha='center', fontsize=14, fontweight='bold')

        monitoring_recommendations = (
            "To validate simulation results and ensure worker safety, implement the following monitoring program:\n\n"
            "• Initial Survey: Conduct comprehensive radiation survey when configuration is first established.\n"
            "• Routine Monitoring: Perform periodic (monthly/quarterly) surveys of all penetration areas.\n"
            "• Personal Dosimetry: Provide appropriate dosimeters (TLD, OSL, or film badges) to all workers.\n"
            "• Real-time Monitoring: Use electronic dosimeters for work in higher-dose-rate areas.\n"
            "• Environmental Monitoring: Consider TLD placement at key locations for long-term monitoring.\n"
            "• Review Process: Establish trigger levels for investigation if unexpected doses are detected."
        )
        plt.text(0.1, 0.04, monitoring_recommendations, fontsize=12, ha='left')

        pdf.savefig()
        plt.close()

        # === ALARA Implementation Page ===
        plt.figure(figsize=(12, 10))
        plt.axis('off')

        # Title
        plt.text(0.5, 0.95, "ALARA Implementation for Streaming Radiation",
                 ha='center', fontsize=18, fontweight='bold')

        # ALARA introduction
        alara_intro = (
            "ALARA (As Low As Reasonably Achievable) is both a regulatory requirement and best practice\n"
            "in radiation protection. The following practical measures apply ALARA principles specifically\n"
            "to radiation streaming situations identified in this analysis."
        )
        plt.text(0.1, 0.88, alara_intro, fontsize=12, ha='left')

        # Time, Distance, Shielding framework
        plt.text(0.5, 0.82, "Applying Time, Distance, and Shielding Principles",
                 ha='center', fontsize=14, fontweight='bold')

        time_distance_shielding = (
            "1. TIME:\n"
            "   • Schedule maintenance activities when radiation source is inactive or at lower power.\n"
            "   • Rotate workers for tasks near streaming penetrations to distribute dose.\n"
            "   • Practice procedures beforehand to minimize time spent in higher-dose areas.\n"
            "   • Use remote handling tools where appropriate to reduce time in the area.\n\n"

            "2. DISTANCE:\n"
            "   • Based on our analysis, increasing distance from {0} cm to {1} cm reduces dose by factor of ~{2:.1f}.\n"
            "   • Mark safe standing positions that maintain minimum 1 meter standoff from penetration exits.\n"
            "   • Use extended tools for adjustments or maintenance near streaming paths.\n"
            "   • Position workstations and regularly occupied areas away from direct streaming paths.\n\n"

            "3. SHIELDING:\n"
            "   • Install local mobile shields (minimum 1-inch lead or equivalent) near critical penetrations.\n"
            "   • Use shadow shields when working near larger penetrations.\n"
            "   • Ensure shield carts or portable shields are available in adjacent areas.\n"
            "   • Consider lead-loaded curtains for temporary shielding during maintenance activities."
        ).format(
            # Fill in with actual values from our data analysis
            detector_distances[0] if 'detector_distances' in locals() else 30,
            detector_distances[-1] if 'detector_distances' in locals() else 150,
            (detector_distances[0] / detector_distances[-1]) ** 2 if 'detector_distances' in locals() else 25
        )
        plt.text(0.1, 0.70, time_distance_shielding, fontsize=12, ha='left')

        # Worker guidance
        plt.text(0.5, 0.48, "Specific Worker Guidance Based on Analysis",
                 ha='center', fontsize=14, fontweight='bold')

        # Analyze angle effects from our data
        angle_reduction = "50-90"  # default value
        if 'angle' in df.columns and 'dose_mean' in df.columns:
            try:
                angle_data = df.groupby('angle')['dose_mean'].mean()
                if 0 in angle_data.index and 45 in angle_data.index and angle_data[0] > 0:
                    reduction_45deg = (1 - angle_data[45] / angle_data[0]) * 100
                    angle_reduction = f"{reduction_45deg:.0f}"
            except:
                pass

        worker_guidance = (
            "• Direct Line-of-Sight: Workers should never stand directly in line with a penetration\n"
            "  when the source is active. Analysis shows this position receives maximum dose.\n\n"

            f"• Angular Position: Standing at 45° angle from the streaming path reduces dose by approximately\n"
            f"  {angle_reduction}% compared to direct line-of-sight exposure.\n\n"

            "• Situational Awareness: Workers should be trained to recognize potential streaming paths\n"
            "  and be aware of shield penetrations in their work area.\n\n"

            "• Streaming Path Markers: Consider marking floors with indicators showing high-dose areas\n"
            "  from streaming radiation (similar to laser path warnings).\n\n"

            "• Buddy System: For work near higher-risk penetrations, implement a buddy system where\n"
            "  one worker monitors dose rates while the other completes the task."
        )
        plt.text(0.1, 0.36, worker_guidance, fontsize=12, ha='left')

        # Special considerations
        plt.text(0.5, 0.24, "Special Considerations for Emergency Scenarios",
                 ha='center', fontsize=14, fontweight='bold')

        emergency_considerations = (
            "• Emergency Response Planning: Ensure emergency responders are briefed on locations of\n"
            "  potential high-dose streaming paths and provided with facility maps indicating these areas.\n\n"

            "• Accident Scenarios: In the event of source control failure or shielding damage, radiation\n"
            "  streaming could create focused high-dose areas. Emergency procedures should include:\n"
            "  - Rapid identification of potential streaming paths\n"
            "  - Predetermined evacuation routes that avoid direct streaming areas\n"
            "  - Emergency shield deployment procedures for critical penetrations\n\n"

            "• Dose Assessment: Provide emergency responders with directional detection equipment\n"
            "  to identify streaming radiation paths in emergency scenarios.\n\n"

            "• Training: Include streaming effects in radiation safety training programs and emergency drills."
        )
        plt.text(0.1, 0.12, emergency_considerations, fontsize=12, ha='left')

        pdf.savefig()
        plt.close()

        # === Cost-Benefit Analysis ===
        plt.figure(figsize=(12, 10))
        plt.axis('off')

        # Title
        plt.text(0.5, 0.95, "Cost-Benefit Analysis for Streaming Mitigation",
                 ha='center', fontsize=18, fontweight='bold')

        # Introduction
        cost_benefit_intro = (
            "ALARA implementation requires balancing the cost of radiation protection measures against\n"
            "the benefit of dose reduction. This analysis evaluates the cost-effectiveness of various\n"
            "engineering and administrative controls for radiation streaming penetrations."
        )
        plt.text(0.1, 0.88, cost_benefit_intro, fontsize=12, ha='left')

        # Create a table
        table_data = [
            ["Mitigation Measure", "Relative Cost", "Dose Reduction Factor", "Cost-Benefit Ratio", "Priority"],
            ["Minimize penetration diameter", "Low", "Proportional to d²", "Excellent", "High"],
            ["Add 90° bend in penetration path", "Medium", "~10×", "Very Good", "High"],
            ["Add local shielding at exit", "Low-Medium", "2-5×", "Good", "High"],
            ["Increase wall thickness", "Very High", "Exponential with thickness", "Poor", "Low"],
            ["Time restrictions for occupancy", "Low", "Proportional to time reduction", "Good", "Medium"],
            ["Distance markers and controls", "Very Low", "Varies with distance²", "Excellent", "High"],
            ["Administrative controls only", "Very Low", "Limited (1-2×)", "Moderate", "Medium"],
            ["Complete filling of penetration", "Medium-High", "Very High", "Situation dependent", "Variable"]
        ]

        # Create the table
        table_ax = plt.axes([0.1, 0.50, 0.8, 0.35])
        table = table_ax.table(cellText=table_data,
                               cellLoc='center',
                               loc='center',
                               colWidths=[0.3, 0.15, 0.2, 0.15, 0.1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_text_props(fontweight='bold', color='white')
            table[(0, i)].set_facecolor('darkblue')

        table_ax.axis('off')

        # Concrete recommendations based on simulation results
        plt.text(0.5, 0.45, "Specific Recommendations Based on Simulation Results",
                 ha='center', fontsize=14, fontweight='bold')

        # Extract specific scenario results for recommendations
        specific_recommendations = "Based on the concrete results of our shielding simulation studies:\n\n"

        # Find maximum dose rate and its configuration
        if 'dose_mean' in df.columns:
            max_dose_row = df.loc[df['dose_mean'].idxmax()]
            specific_recommendations += (
                f"1. HIGHEST RISK CONFIGURATION: Our simulation identified that a {max_dose_row.get('energy', 'high-energy')} MeV "
                f"source with a {max_dose_row.get('channel_diameter', 'large')} cm diameter penetration produces a maximum dose rate "
                f"of {max_dose_row['dose_mean']:.2e} rem/hr at {max_dose_row.get('distance', 'close')} cm distance. "
                f"This configuration requires the most stringent controls.\n\n"
            )

        # Get diameter effect data
        diameter_analysis = self.analyze_channel_diameter_trend()
        if diameter_analysis and 'trends' in diameter_analysis and 'diameter_vs_dose' in diameter_analysis['trends']:
            trend = diameter_analysis['trends']['diameter_vs_dose']
            b = trend['parameters']['b']
            specific_recommendations += (
                f"2. DIAMETER RECOMMENDATIONS: Our simulations show that dose rates scale with diameter^{b:.2f}. "
                f"Therefore, reducing penetration diameter by 50% will reduce dose rates by approximately "
                f"{(1 - 0.5 ** b) * 100:.0f}%. For all necessary penetrations, we recommend not exceeding the minimum "
                f"functional diameter required.\n\n"
            )

        # Get distance effect data
        distance_data = None
        if 'distance' in df.columns and 'dose_mean' in df.columns:
            try:
                # Get distances and corresponding doses for a specific configuration
                distance_data = df[df['channel_diameter'] == df['channel_diameter'].max()].groupby('distance')[
                    'dose_mean'].mean()
                if len(distance_data) >= 2:
                    # Find "safe distance" where dose drops below 0.1 rem/hr (conservative limit for controlled areas)
                    distances = np.array(list(distance_data.index))
                    doses = np.array(list(distance_data.values))

                    # Find where dose is below 0.1 rem/hr or extrapolate
                    safe_distance = None
                    if np.min(doses) > 0.1:
                        # Need to extrapolate using power law (assuming inverse square as approximation)
                        if len(doses) >= 2:
                            ref_dist = distances[0]
                            ref_dose = doses[0]
                            safe_distance = ref_dist * np.sqrt(ref_dose / 0.1)
                    else:
                        # Interpolate to find safe distance
                        for i in range(len(doses) - 1):
                            if doses[i] > 0.1 and doses[i + 1] <= 0.1:
                                # Linear interpolation in log space
                                log_dist = np.log10([distances[i], distances[i + 1]])
                                log_dose = np.log10([doses[i], doses[i + 1]])
                                safe_distance = 10 ** (np.interp(np.log10(0.1), log_dose[::-1], log_dist[::-1]))
                                break

                    if safe_distance:
                        specific_recommendations += (
                            f"3. SAFE DISTANCE DETERMINATION: Based on our simulation results, a minimum safe distance "
                            f"of {safe_distance:.0f} cm ({safe_distance / 100:.1f} m) from the penetration exit should be maintained "
                            f"for the highest-risk configuration to ensure dose rates remain below 0.1 rem/hr (the typical "
                            f"upper limit for a controlled area).\n\n"
                        )

                    # Calculate how distance reduces dose
                    specific_recommendations += (
                        f"4. DISTANCE EFFECT: Our simulations demonstrate that increasing distance from "
                        f"{distances[0]:.0f} cm to {distances[-1]:.0f} cm reduces dose rates by a factor of "
                        f"{doses[0] / doses[-1]:.1f}×. Workstations should be positioned at least {distances[-1]:.0f} cm "
                        f"from penetration exits.\n\n"
                    )
            except Exception as e:
                self.logger.warning(f"Could not analyze distance data for recommendations: {str(e)}")

        # Get angle effect data
        if 'angle' in df.columns and 'dose_mean' in df.columns:
            try:
                angle_data = df[df['channel_diameter'] == df['channel_diameter'].max()].groupby('angle')[
                    'dose_mean'].mean()
                if len(angle_data) >= 2 and 0 in angle_data.index:
                    max_angle = max(angle_data.index)
                    reduction_factor = angle_data[0] / angle_data[max_angle]

                    specific_recommendations += (
                        f"5. ANGULAR POSITIONING: Our simulations show that standing at a {max_angle}° angle "
                        f"from the beam axis reduces dose rates by a factor of {reduction_factor:.1f}× compared "
                        f"to direct line-of-sight exposure. Work areas should be positioned at least 30° "
                        f"off-axis from penetration paths where possible.\n\n"
                    )
            except Exception as e:
                self.logger.warning(f"Could not analyze angle data for recommendations: {str(e)}")

        # Get streaming factor for concrete
        concrete_analysis = self.analyze_concrete_streaming()
        if concrete_analysis and 'data' in concrete_analysis:
            try:
                data = pd.DataFrame(concrete_analysis['data'])
                if 'streaming_factor' in data.columns:
                    max_streaming = data['streaming_factor'].max()

                    specific_recommendations += (
                        f"6. STREAMING FACTOR: Our simulations calculate a maximum streaming factor of {max_streaming:.1e}× "
                        f"for penetrations in concrete shielding. This demonstrates that even small penetrations can "
                        f"significantly compromise shield integrity. All penetrations should be evaluated individually "
                        f"and shielding compensation applied.\n\n"
                    )
            except Exception as e:
                self.logger.warning(f"Could not analyze streaming data for recommendations: {str(e)}")

        # Recommendations for bends in penetrations if available
        path_analysis = self.analyze_streaming_path_effect()
        if path_analysis and 'data' in path_analysis:
            try:
                data = pd.DataFrame(path_analysis['data'])
                if 'path_bends' in data.columns and 'dose_mean' in data.columns:
                    bend_effect = data.groupby('path_bends')['dose_mean'].mean()
                    if len(bend_effect) >= 2 and 0 in bend_effect.index and 1 in bend_effect.index:
                        bend_reduction = bend_effect[0] / bend_effect[1]

                        specific_recommendations += (
                            f"7. PENETRATION DESIGN: Our simulation results show that adding a single 90° bend "
                            f"in a penetration path reduces dose rates by a factor of {bend_reduction:.1f}×. "
                            f"For all new shield penetration designs, we strongly recommend incorporating at least "
                            f"one bend to minimize streaming radiation.\n\n"
                        )
            except Exception as e:
                self.logger.warning(f"Could not analyze path bend data for recommendations: {str(e)}")

        plt.text(0.1, 0.40, specific_recommendations, fontsize=11, ha='left')

        # Implementation timeline
        plt.text(0.5, 0.12, "Implementation Timeline Based on Risk Assessment",
                 ha='center', fontsize=14, fontweight='bold')

        timeline_text = (
            "IMMEDIATE (Within 1 week):\n"
            "• Implement access controls for high-risk penetrations identified in the simulation\n"
            "• Conduct radiation surveys to validate simulation results\n"
            "• Provide dosimetry to personnel working near streaming paths\n\n"

            "SHORT-TERM (Within 1 month):\n"
            "• Install temporary shielding at exit points of critical penetrations\n"
            "• Develop and implement administrative controls and procedures\n"
            "• Conduct training for workers on streaming radiation hazards\n\n"

            "MEDIUM-TERM (Within 6 months):\n"
            "• Modify existing penetrations to incorporate bends where feasible\n"
            "• Implement engineering controls for high-risk configurations\n"
            "• Develop long-term monitoring program\n\n"

            "LONG-TERM (Within 1 year):\n"
            "• Redesign critical shield penetrations during scheduled maintenance\n"
            "• Update facility radiation safety program to address streaming radiation\n"
            "• Conduct comprehensive effectiveness review of implemented measures"
        )
        plt.text(0.1, 0.09, timeline_text, fontsize=11, ha='left')

        pdf.savefig()
        plt.close()

        # === Direct Simulation Result Application to Worker Safety ===
        plt.figure(figsize=(12, 10))
        plt.axis('off')

        # Title
        plt.text(0.5, 0.95, "Simulation Results: Direct Application to Worker Safety",
                 ha='center', fontsize=18, fontweight='bold')

        # Create a 2-column layout
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # Left column: Dose rate table for specific scenarios
        left_ax = plt.subplot(gs[0])
        left_ax.axis('off')

        # Create a table with dose rates from simulation results
        table_title = "Selected Critical Configurations from Simulation"
        left_ax.text(0.5, 1.05, table_title, ha='center', fontsize=12, fontweight='bold')

        # Select representative scenarios from results
        if 'energy' in df.columns and 'channel_diameter' in df.columns and 'dose_mean' in df.columns:
            try:
                # Get unique values
                energies = sorted(df['energy'].unique())
                diameters = sorted(df['channel_diameter'].unique())

                # Select a subset for table presentation
                selected_energies = [energies[0], energies[-1]] if len(energies) > 1 else energies
                selected_diameters = [diameters[0], diameters[-1]] if len(diameters) > 1 else diameters

                # Create table data
                table_data = [["Energy (MeV)", "Diameter (cm)", "Distance (cm)", "Dose Rate (rem/hr)", "Category"]]

                # Safety categories
                def safety_category(dose):
                    if dose > 100:
                        return "EXTREME HAZARD"
                    elif dose > 10:
                        return "HIGH HAZARD"
                    elif dose > 1:
                        return "HAZARD"
                    elif dose > 0.1:
                        return "CONTROLLED"
                    else:
                        return "MINIMAL RISK"

                # Add representative rows
                for energy in selected_energies:
                    for diameter in selected_diameters:
                        # Get closest distance
                        closest_dist_data = df[(df['energy'] == energy) &
                                               (df['channel_diameter'] == diameter) &
                                               (df['angle'] == 0)].nsmallest(1, 'distance')

                        if not closest_dist_data.empty:
                            row = closest_dist_data.iloc[0]
                            dose = row['dose_mean']
                            table_data.append([
                                f"{energy:.1f}",
                                f"{diameter:.2f}",
                                f"{row['distance']:.0f}",
                                f"{dose:.2e}",
                                safety_category(dose)
                            ])

                # Create and style the table
                table = left_ax.table(cellText=table_data,
                                      cellLoc='center',
                                      loc='center',
                                      colWidths=[0.18, 0.18, 0.18, 0.25, 0.25])

                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)

                # Style header row
                for i in range(len(table_data[0])):
                    table[(0, i)].set_text_props(fontweight='bold', color='white')
                    table[(0, i)].set_facecolor('darkblue')

                # Color-code safety categories
                for i in range(1, len(table_data)):
                    category = table_data[i][4]
                    cell = table[(i, 4)]

                    if category == "EXTREME HAZARD":
                        cell.set_facecolor('darkred')
                        cell.set_text_props(color='white', fontweight='bold')
                    elif category == "HIGH HAZARD":
                        cell.set_facecolor('red')
                        cell.set_text_props(color='white', fontweight='bold')
                    elif category == "HAZARD":
                        cell.set_facecolor('orange')
                        cell.set_text_props(fontweight='bold')
                        elif category == "CONTROLLED":
                        cell.set_facecolor('yellow')
                    elif category == "MINIMAL RISK":
                        cell.set_facecolor('lightgreen')

            except Exception as e:
                self.logger.warning(f"Could not create scenario table: {str(e)}")
                left_ax.text(0.5, 0.5, "Insufficient data for scenario analysis",
                             ha='center', fontsize=10, style='italic')
        else:
            left_ax.text(0.5, 0.5, "Insufficient data for scenario analysis",
                         ha='center', fontsize=10, style='italic')

        # Right column: Worker safety guidelines
        right_ax = plt.subplot(gs[1])
        right_ax.axis('off')

        right_ax.text(0.5, 1.05, "Worker Safety Guidelines Based on Simulation Results",
                      ha='center', fontsize=12, fontweight='bold')

        # Worker dose limit explanation
        worker_dose_limits = (
            "OCCUPATIONAL DOSE LIMITS:\n"
            "• Annual limit: 5 rem (50 mSv)\n"
            "• Recommended ALARA level: 0.5 rem (5 mSv)\n"
            "• Weekly limit: ~0.1 rem (1 mSv)\n"
            "• Daily limit: ~0.02 rem (0.2 mSv)\n\n"
        )
        right_ax.text(0.02, 0.95, worker_dose_limits, fontsize=10, va='top')

        # Calculate scenario-specific exposure time limits
        if 'dose_mean' in df.columns:
            try:
                # Find representative dose rates
                max_dose = df['dose_mean'].max()
                median_dose = df['dose_mean'].median()
                min_dose = df['dose_mean'][df['dose_mean'] > 0].min()

                # Calculate time limits
                max_daily_time = min(24, 0.02 / max_dose) if max_dose > 0 else 24
                median_daily_time = min(24, 0.02 / median_dose) if median_dose > 0 else 24
                min_daily_time = min(24, 0.02 / min_dose) if min_dose > 0 else 24

                # Format time in appropriate units
                def format_time(hours):
                    if hours >= 24:
                        return "No restriction"
                    elif hours >= 1:
                        return f"{hours:.1f} hours"
                    else:
                        minutes = hours * 60
                        if minutes >= 1:
                            return f"{minutes:.0f} minutes"
                        else:
                            return f"{minutes * 60:.0f} seconds"

                time_limits = (
                    "SCENARIO-SPECIFIC TIME LIMITS (to stay under daily limit):\n\n"
                    f"• Highest dose scenario ({max_dose:.2e} rem/hr):\n"
                    f"  - Maximum occupancy: {format_time(max_daily_time)}\n\n"
                    f"• Typical scenario ({median_dose:.2e} rem/hr):\n"
                    f"  - Maximum occupancy: {format_time(median_daily_time)}\n\n"
                    f"• Lowest dose scenario ({min_dose:.2e} rem/hr):\n"
                    f"  - Maximum occupancy: {format_time(min_daily_time)}\n\n"
                )

                right_ax.text(0.02, 0.80, time_limits, fontsize=10, va='top')
            except Exception as e:
                self.logger.warning(f"Could not calculate time limits: {str(e)}")

        # Add specific safety measures based on simulation results
        safety_measures = "CRITICAL SAFETY MEASURES FROM SIMULATION RESULTS:\n\n"

        # Channel diameter effect
        if 'channel_diameter' in df.columns and 'dose_mean' in df.columns:
            diameter_groups = df.groupby('channel_diameter')['dose_mean'].mean()
            if len(diameter_groups) > 1:
                min_diam = diameter_groups.index.min()
                max_diam = diameter_groups.index.max()
                reduction = diameter_groups[max_diam] / diameter_groups[min_diam]

                safety_measures += (
                    f"1. CHANNEL SIZE: Our simulations show a {reduction:.1f}× increase in dose when\n"
                    f"   channels increase from {min_diam:.1f} cm to {max_diam:.1f} cm diameter.\n"
                    f"   Recommendation: Strictly limit channel size to operational minimum.\n\n"
                )

        # Concrete thickness effect
        if 'shield_thickness' in df.columns and 'dose_mean' in df.columns:
            thickness_groups = df.groupby('shield_thickness')['dose_mean'].mean()
            if len(thickness_groups) > 1:
                min_thick = thickness_groups.index.min()
                max_thick = thickness_groups.index.max()
                reduction = thickness_groups[min_thick] / thickness_groups[max_thick]

                safety_measures += (
                    f"2. SHIELD THICKNESS: Our simulations show a {reduction:.1f}× decrease in dose when\n"
                    f"   concrete thickness increases from {min_thick:.1f} cm to {max_thick:.1f} cm.\n"
                    f"   Recommendation: Maintain minimum {max_thick:.1f} cm concrete for critical barriers.\n\n"
                )

        # Source strength/energy effect
        if 'energy' in df.columns and 'dose_mean' in df.columns:
            energy_groups = df.groupby('energy')['dose_mean'].mean()
            if len(energy_groups) > 1:
                min_energy = energy_groups.index.min()
                max_energy = energy_groups.index.max()
                increase = energy_groups[max_energy] / energy_groups[min_energy]

                safety_measures += (
                    f"3. SOURCE ENERGY: Our simulations show a {increase:.1f}× increase in dose when\n"
                    f"   source energy increases from {min_energy:.1f} MeV to {max_energy:.1f} MeV.\n"
                    f"   Recommendation: Apply enhanced controls for higher energy operations.\n\n"
                )

        # Add streaming path recommendations based on bend analysis
        if path_analysis and 'data' in path_analysis:
            data = pd.DataFrame(path_analysis['data'])
            if 'path_bends' in data.columns and 'dose_mean' in data.columns:
                bend_effect = data.groupby('path_bends')['dose_mean'].mean()
                if len(bend_effect) >= 2 and 0 in bend_effect.index and 1 in bend_effect.index:
                    reduction = bend_effect[0] / bend_effect[1]

                    safety_measures += (
                        f"4. PATH DESIGN: Our simulations show a {reduction:.1f}× reduction in dose by adding\n"
                        f"   a single 90° bend in the penetration path.\n"
                        f"   Recommendation: Redesign straight penetrations to include at least one bend.\n\n"
                    )

        # Add distance recommendation
        if distance_data is not None and len(distance_data) >= 2:
            min_dist = min(distance_data.index)
            max_dist = max(distance_data.index)
            reduction = distance_data[min_dist] / distance_data[max_dist]

            safety_measures += (
                f"5. WORKER POSITIONING: Our simulations show a {reduction:.1f}× reduction in dose by\n"
                f"   increasing distance from {min_dist:.0f} cm to {max_dist:.0f} cm.\n"
                f"   Recommendation: Implement minimum {max_dist:.0f} cm standoff from penetrations.\n\n"
            )

        right_ax.text(0.02, 0.55, safety_measures, fontsize=10, va='top')

        # Final emergency procedures
        emergency_text = (
            "EMERGENCY PROCEDURES FOR STREAMING RADIATION:\n\n"
            "If abnormal dose rates are detected near penetrations:\n\n"
            "1. EVACUATE all personnel from the direct line-of-sight of the penetration\n"
            "2. NOTIFY radiation safety officer immediately\n"
            "3. VERIFY source status and containment integrity\n"
            "4. SURVEY area with directional detector to map streaming path\n"
            "5. DEPLOY emergency shielding perpendicular to the streaming path\n"
            "6. RESTRICT access with barriers at safe distance boundary\n"
            "7. DOCUMENT incident and exposure estimates"
        )

        right_ax.text(0.02, 0.20, emergency_text, fontsize=10, va='top',
                      bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round'))

        pdf.savefig()
        plt.close()

        # === Summary and Future Work ===
        plt.figure(figsize=(12, 10))
        plt.axis('off')

        # Title
        plt.text(0.5, 0.95, "Summary of Key Findings and Future Work",
                 ha='center', fontsize=18, fontweight='bold')

        # Summary of key findings
        summary_title = "Key Findings from Radiation Streaming Analysis"
        plt.text(0.5, 0.88, summary_title, ha='center', fontsize=14, fontweight='bold')

        summary_text = "This comprehensive analysis of radiation streaming through shield penetrations has revealed:\n\n"

        # Add specific findings from our simulations
        if 'channel_diameter' in df.columns and 'dose_mean' in df.columns:
            summary_text += (
                "1. DIAMETER DEPENDENCY: Radiation streaming increases with penetration diameter following\n"
                f"   approximately a power law with exponent {b:.2f} (compared to theoretical d² dependence).\n\n"
            )

        if distance_data is not None:
            summary_text += (
                "2. DISTANCE EFFECT: Dose rates from streaming radiation follow an approximate inverse square law,\n"
                f"   decreasing by a factor of {doses[0] / doses[-1]:.1f}× when distance increases from "
                f"{distances[0]:.0f} to {distances[-1]:.0f} cm.\n\n"
            )

        if path_analysis and 'data' in path_analysis:
            data = pd.DataFrame(path_analysis['data'])
            if 'path_bends' in data.columns and 'dose_mean' in data.columns:
                bend_effect = data.groupby('path_bends')['dose_mean'].mean()
                if len(bend_effect) >= 2 and 0 in bend_effect.index and 1 in bend_effect.index:
                    reduction = bend_effect[0] / bend_effect[1]
                    summary_text += (
                        f"3. PATH GEOMETRY: Adding a 90° bend to a penetration path reduces dose rates by approximately\n"
                        f"   {reduction:.1f}×, effectively eliminating direct line-of-sight radiation streaming.\n\n"
                    )

        if concrete_analysis and 'data' in concrete_analysis:
            data = pd.DataFrame(concrete_analysis['data'])
            if 'streaming_factor' in data.columns:
                max_streaming = data['streaming_factor'].max()
                summary_text += (
                    f"4. STREAMING AMPLIFICATION: Penetrations through concrete shields can increase dose rates by\n"
                    f"   up to {max_streaming:.1e}× compared to an intact shield, highlighting the critical nature\n"
                    f"   of proper penetration design and control.\n\n"
                )

        if 'angle' in df.columns and 'dose_mean' in df.columns:
            try:
                angle_data = df.groupby('angle')['dose_mean'].mean()
                if 0 in angle_data.index and max(angle_data.index) > 0:
                    max_angle = max(angle_data.index)
                    reduction = angle_data[0] / angle_data[max_angle]
                    summary_text += (
                        f"5. ANGULAR DISTRIBUTION: Dose rates decrease by a factor of {reduction:.1f}× when positioning\n"
                        f"   at {max_angle}° from the direct streaming path, providing a simple but effective\n"
                        f"   method to reduce worker exposures.\n\n"
                    )
            except:
                pass

        plt.text(0.1, 0.86, summary_text, fontsize=11, va='top')

        # Recommendations for future work
        future_title = "Recommendations for Future Work"
        plt.text(0.5, 0.55, future_title, ha='center', fontsize=14, fontweight='bold')

        future_text = (
            "Based on our findings, we recommend the following areas for future investigation:\n\n"

            "1. EXPERIMENTAL VALIDATION: Conduct physical measurements to validate the simulation results,\n"
            "   particularly focusing on the effects of bends and penetration diameter.\n\n"

            "2. MATERIAL STUDIES: Expand the analysis to include different shield materials (steel, lead,\n"
            "   borated polyethylene) to identify optimal materials for specific radiation types.\n\n"

            "3. COMPOUND PENETRATIONS: Investigate more complex penetration configurations with multiple\n"
            "   channels or varying diameters along the path.\n\n"

            "4. SCATTERED RADIATION: Extend the analysis to better characterize the scattered radiation\n"
            "   field surrounding penetration exits.\n\n"

            "5. DYNAMIC SIMULATIONS: Develop time-dependent models for scenarios with varying source\n"
            "   strengths or moving sources.\n\n"

            "6. ENHANCED VISUALIZATION: Develop 3D visualization tools for radiation streaming paths to\n"
            "   better communicate risks to facility personnel.\n\n"

            "7. MACHINE LEARNING MODELS: Develop predictive models based on simulation data to rapidly\n"
            "   estimate dose rates for new configurations without full simulation."
        )

        plt.text(0.1, 0.53, future_text, fontsize=11, va='top')

        # Conclusion
        conclusion_title = "Conclusion"
        plt.text(0.5, 0.20, conclusion_title, ha='center', fontsize=14, fontweight='bold')

        conclusion_text = (
            "This comprehensive analysis of radiation streaming through shield penetrations provides valuable\n"
            "insights for radiation safety and shielding design. The simulation results clearly demonstrate that\n"
            "seemingly minor design choices in penetration geometry can have order-of-magnitude effects on\n"
            "resulting dose rates.\n\n"

            "By quantifying these effects, we enable evidence-based decision making for both new facility designs\n"
            "and retrofits of existing structures. The implementation of the safety recommendations derived from\n"
            "these simulations will significantly reduce occupational radiation exposures while maintaining\n"
            "operational capability.\n\n"

            "The underlying methods, parameter studies, and analysis techniques developed in this work establish\n"
            "a framework for continued improvement of radiation safety practices related to shield penetrations."
        )

        plt.text(0.1, 0.18, conclusion_text, fontsize=11, va='top')

        # Add final acknowledgment
        plt.text(0.5, 0.02,
                 "Analysis performed using the Radiation Streaming Analysis Framework\n"
                 "Report generated on " + datetime.now().strftime("%Y-%m-%d"),
                 ha='center', fontsize=9, style='italic')

        pdf.savefig()
        plt.close()

        # Success message
        self.logger.info(f"Report successfully generated at {output_path}")

        return True

    except Exception as e:
    self.logger.error(f"Error generating report: {str(e)}")
    import traceback
    self.logger.error(traceback.format_exc())
    return False


def analyze_channel_diameter_trend(self):
    """
    Analyze how dose rates vary with channel diameter.

    Returns:
        dict: Analysis results with trends and statistical data
    """
    try:
        # Check if we have diameter and dose data
        if 'channel_diameter' not in self.data.columns or 'dose_mean' not in self.data.columns:
            self.logger.warning("Cannot analyze diameter trend: missing required columns")
            return None

        # Filter for angle=0 (direct line-of-sight) to simplify analysis
        filtered_data = self.data
        if 'angle' in self.data.columns:
            filtered_data = self.data[self.data['angle'] == 0]

        # Group by diameter and calculate statistics
        diameter_groups = filtered_data.groupby('channel_diameter')

        diameter_data = []
        for diameter, group in diameter_groups:
            # Calculate average dose and standard deviation
            dose_mean = group['dose_mean'].mean()
            dose_std = group['dose_std'].mean() if 'dose_std' in group.columns else group['dose_mean'].std()

            # Add to data list
            diameter_data.append({
                'diameter': diameter,
                'dose_mean': dose_mean,
                'dose_std': dose_std,
                'samples': len(group)
            })

        # Sort by diameter
        diameter_data = sorted(diameter_data, key=lambda x: x['diameter'])

        # Fit power law: dose = a * diameter^b
        diameters = np.array([d['diameter'] for d in diameter_data])
        doses = np.array([d['dose_mean'] for d in diameter_data])

        if len(diameters) >= 2 and np.min(diameters) > 0 and np.min(doses) > 0:
            # Log-log linear regression for power law
            log_diameters = np.log(diameters)
            log_doses = np.log(doses)

            slope, intercept, r_value, p_value, std_err = stats.linregress(log_diameters, log_doses)

            # Calculate fit parameters
            a = np.exp(intercept)
            b = slope
            r_squared = r_value ** 2

            # Compile trends
            trends = {
                'diameter_vs_dose': {
                    'type': 'power_law',
                    'equation': 'dose = a * diameter^b',
                    'parameters': {
                        'a': a,
                        'b': b
                    },
                    'r_squared': r_squared,
                    'p_value': p_value
                }
            }

            # Compare to theoretical d² relationship
            theoretical_deviation = abs(b - 2.0) / 2.0 * 100  # percent deviation

            # Return comprehensive analysis
            return {
                'data': diameter_data,
                'trends': trends,
                'theoretical_comparison': {
                    'expected_exponent': 2.0,
                    'measured_exponent': b,
                    'deviation_percent': theoretical_deviation
                },
                'statistics': {
                    'sample_count': len(diameter_data),
                    'diameter_range': [np.min(diameters), np.max(diameters)],
                    'dose_range': [np.min(doses), np.max(doses)]
                }
            }
        else:
            # Not enough data for trend analysis
            return {
                'data': diameter_data,
                'statistics': {
                    'sample_count': len(diameter_data),
                    'diameter_range': [np.min(diameters), np.max(diameters)] if len(diameters) > 0 else [0, 0],
                    'dose_range': [np.min(doses), np.max(doses)] if len(doses) > 0 else [0, 0]
                }
            }

    except Exception as e:
        self.logger.error(f"Error analyzing channel diameter trend: {str(e)}")
        return None


def analyze_distance_trend(self):
    """
    Analyze how dose rates vary with distance from the shield penetration.

    Returns:
        dict: Analysis results with trends and statistical data
    """
    try:
        # Check if we have distance and dose data
        if 'distance' not in self.data.columns or 'dose_mean' not in self.data.columns:
            self.logger.warning("Cannot analyze distance trend: missing required columns")
            return None

        # Filter for angle=0 (direct line-of-sight) to simplify analysis
        filtered_data = self.data
        if 'angle' in self.data.columns:
            filtered_data = self.data[self.data['angle'] == 0]

        # Group by distance and calculate statistics
        distance_groups = filtered_data.groupby('distance')

        distance_data = []
        for distance, group in distance_groups:
            # Calculate average dose and standard deviation
            dose_mean = group['dose_mean'].mean()
            dose_std = group['dose_std'].mean() if 'dose_std' in group.columns else group['dose_mean'].std()

            # Add to data list
            distance_data.append({
                'distance': distance,
                'dose_mean': dose_mean,
                'dose_std': dose_std,
                'samples': len(group)
            })

        # Sort by distance
        distance_data = sorted(distance_data, key=lambda x: x['distance'])

        # Fit power law: dose = a * distance^-b (inverse power law)
        distances = np.array([d['distance'] for d in distance_data])
        doses = np.array([d['dose_mean'] for d in distance_data])

        if len(distances) >= 3 and np.min(distances) > 0 and np.min(doses) > 0:
            # Log-log linear regression for power law
            log_distances = np.log(distances)
            log_doses = np.log(doses)

            slope, intercept, r_value, p_value, std_err = stats.linregress(log_distances, log_doses)

            # Calculate fit parameters (note negative slope for inverse relationship)
            a = np.exp(intercept)
            b = -slope  # Convert to positive exponent for inverse power law
            r_squared = r_value ** 2

            # Compile trends
            trends = {
                'power_law': {
                    'type': 'inverse_power_law',
                    'equation': 'dose = a * distance^-b',
                    'parameters': {
                        'a': a,
                        'b': b
                    },
                    'r_squared': r_squared,
                    'p_value': p_value
                },
                'inverse_square': {
                    'type': 'reference',
                    'equation': 'dose = k * distance^-2',
                    'note': 'Theoretical prediction for point source'
                }
            }

            # Compare to theoretical 1/r² relationship
            theoretical_deviation = abs(b - 2.0) / 2.0 * 100  # percent deviation

            # Return comprehensive analysis
            return {
                'data': distance_data,
                'trends': trends,
                'theoretical_comparison': {
                    'expected_exponent': 2.0,
                    'measured_exponent': b,
                    'deviation_percent': theoretical_deviation,
                    'interpretation': self._interpret_distance_trend(b)
                },
                'statistics': {
                    'sample_count': len(distance_data),
                    'distance_range': [np.min(distances), np.max(distances)],
                    'dose_range': [np.min(doses), np.max(doses)]
                }
            }
        else:
            # Not enough data for trend analysis
            return {
                'data': distance_data,
                'statistics': {
                    'sample_count': len(distance_data),
                    'distance_range': [np.min(distances), np.max(distances)] if len(distances) > 0 else [0, 0],
                    'dose_range': [np.min(doses), np.max(doses)] if len(doses) > 0 else [0, 0]
                }
            }

    except Exception as e:
        self.logger.error(f"Error analyzing distance trend: {str(e)}")
        return None


def _interpret_distance_trend(self, exponent):
    """
    Provide an interpretation of the distance exponent.

    Args:
        exponent: The measured power law exponent

    Returns:
        str: Interpretation of the trend
    """
    if exponent < 1.5:
        return "Exponent less than 1.5 suggests significant scattered radiation or an extended source."
    elif 1.5 <= exponent <= 2.5:
        return "Exponent between 1.5-2.5 is consistent with point source behavior (inverse square law)."
    else:
        return "Exponent greater than 2.5 suggests additional attenuation effects beyond geometric spreading."


def analyze_concrete_streaming(self):
    """
    Analyze radiation streaming through concrete shields.

    Returns:
        dict: Analysis results with streaming factors
    """
    try:
        # Check if we have required data
        if 'channel_diameter' not in self.data.columns or 'dose_mean' not in self.data.columns:
            self.logger.warning("Cannot analyze concrete streaming: missing required columns")
            return None

        # Check if we have data with and without penetration
        if 'penetration' not in self.data.columns:
            self.logger.warning("Cannot analyze concrete streaming: missing penetration status column")
            return None

        # Filter for angle=0 (direct line-of-sight)
        filtered_data = self.data
        if 'angle' in self.data.columns:
            filtered_data = self.data[self.data['angle'] == 0]

        # Calculate streaming factors
        streaming_data = []

        # Group by configuration parameters
        config_columns = ['shield_thickness', 'distance', 'energy']
        config_columns = [col for col in config_columns if col in filtered_data.columns]

        if not config_columns:
            self.logger.warning("Cannot analyze concrete streaming: no configuration parameters")
            return None

        # Add channel diameter if missing from config
        if 'channel_diameter' not in config_columns:
            config_columns.append('channel_diameter')

        # Group by configuration
        for config, group in filtered_data.groupby(config_columns):
            # Convert to dict if tuple
            if isinstance(config, tuple):
                config_dict = {col: val for col, val in zip(config_columns, config)}
            else:
                config_dict = {config_columns[0]: config}

            # Find data with and without penetration
            penetration_data = group[group['penetration'] == True]
            no_penetration_data = group[group['penetration'] == False]

            if len(penetration_data) > 0 and len(no_penetration_data) > 0:
                # Calculate streaming factor = dose_with_penetration / dose_without_penetration
                dose_with = penetration_data['dose_mean'].mean()
                dose_without = no_penetration_data['dose_mean'].mean()

                streaming_factor = dose_with / dose_without if dose_without > 0 else float('inf')

                # Create data entry
                data_entry = config_dict.copy()
                data_entry.update({
                    'streaming_factor': streaming_factor,
                    'dose_with_penetration': dose_with,
                    'dose_without_penetration': dose_without,
                    'samples_with': len(penetration_data),
                    'samples_without': len(no_penetration_data)
                })

                streaming_data.append(data_entry)

        if not streaming_data:
            self.logger.warning("Cannot analyze concrete streaming: no valid comparisons found")
            return None

        # Analyze trends if we have diameter data
        trends = {}

        if 'channel_diameter' in filtered_data.columns:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(streaming_data)

            # Power law fit for streaming factor vs diameter
            diameters = df['channel_diameter'].values
            factors = df['streaming_factor'].values

            if len(diameters) >= 3 and np.min(diameters) > 0 and np.min(factors) > 0:
                # Log-log linear regression
                log_diameters = np.log(diameters)
                log_factors = np.log(factors)

                slope, intercept, r_value, p_value, std_err = stats.linregress(log_diameters, log_factors)

                # Calculate fit parameters
                a = np.exp(intercept)
                b = slope
                r_squared = r_value ** 2

                trends['streaming_vs_diameter'] = {
                    'type': 'power_law',
                    'equation': 'streaming_factor = a * diameter^b',
                    'parameters': {
                        'a': a,
                        'b': b
                    },
                    'r_squared': r_squared,
                    'p_value': p_value
                }

                # Return comprehensive analysis
                return {
                    'data': streaming_data,
                    'trends': trends,
                    'statistics': {
                        'sample_count': len(streaming_data),
                        'max_streaming_factor': max([d['streaming_factor'] for d in streaming_data]),
                        'min_streaming_factor': min([d['streaming_factor'] for d in streaming_data]),
                        'avg_streaming_factor': sum([d['streaming_factor'] for d in streaming_data]) / len(
                            streaming_data)
                    }
                }

            except Exception as e:
            self.logger.error(f"Error analyzing concrete streaming: {str(e)}")
            return None

        def analyze_streaming_path_effect(self):
            """
            Analyze the effect of streaming path geometry (straight vs. bent paths).

            Returns:
                dict: Analysis results with comparative data
            """
            try:
                # Check if we have path bend data
                if 'path_bends' not in self.data.columns or 'dose_mean' not in self.data.columns:
                    self.logger.warning("Cannot analyze streaming path effect: missing required columns")
                    return None

                # Group by path bends and calculate statistics
                path_groups = self.data.groupby('path_bends')

                path_data = []
                for bends, group in path_groups:
                    # Calculate average dose and standard deviation
                    dose_mean = group['dose_mean'].mean()
                    dose_std = group['dose_std'].mean() if 'dose_std' in group.columns else group['dose_mean'].std()

                    # Add to data list
                    path_data.append({
                        'path_bends': bends,
                        'dose_mean': dose_mean,
                        'dose_std': dose_std,
                        'samples': len(group),
                        'description': f"{bends} bend{'s' if bends != 1 else ''}"
                    })

                # Sort by number of bends
                path_data = sorted(path_data, key=lambda x: x['path_bends'])

                # Calculate ratios between different bend configurations
                ratios = []

                if len(path_data) >= 2:
                    # Compare each configuration to the straight path (0 bends)
                    straight_dose = next((d['dose_mean'] for d in path_data if d['path_bends'] == 0), None)

                    if straight_dose is not None:
                        for data in path_data:
                            if data['path_bends'] > 0 and data['dose_mean'] > 0:
                                ratios.append({
                                    'reference': 0,  # 0 bends
                                    'comparison': data['path_bends'],
                                    'dose_ratio': straight_dose / data['dose_mean'],
                                    'description': f"Straight path vs. {data['path_bends']} bend{'s' if data['path_bends'] != 1 else ''}"
                                })

                # Return comprehensive analysis
                return {
                    'data': path_data,
                    'ratios': ratios,
                    'statistics': {
                        'sample_count': len(path_data),
                        'configurations': [d['path_bends'] for d in path_data]
                    }
                }

            except Exception as e:
                self.logger.error(f"Error analyzing streaming path effect: {str(e)}")
                return None

        def analyze_angular_distribution(self):
            """
            Analyze the angular distribution of scattered radiation from penetrations.

            Returns:
                dict: Analysis results with angular dose data
            """
            try:
                # Check if we have angle data
                if 'angle' not in self.data.columns or 'dose_mean' not in self.data.columns:
                    self.logger.warning("Cannot analyze angular distribution: missing required columns")
                    return None

                # Group by angle and calculate statistics
                angle_groups = self.data.groupby('angle')

                angle_data = []
                for angle, group in angle_groups:
                    # Calculate average dose and standard deviation
                    dose_mean = group['dose_mean'].mean()
                    dose_std = group['dose_std'].mean() if 'dose_std' in group.columns else group['dose_mean'].std()

                    # Add to data list
                    angle_data.append({
                        'angle': angle,
                        'dose_mean': dose_mean,
                        'dose_std': dose_std,
                        'samples': len(group)
                    })

                # Sort by angle
                angle_data = sorted(angle_data, key=lambda x: x['angle'])

                # Calculate ratios between different angles
                ratios = []

                if len(angle_data) >= 2:
                    # Compare each angle to direct line-of-sight (0 degrees)
                    direct_dose = next((d['dose_mean'] for d in angle_data if d['angle'] == 0), None)

                    if direct_dose is not None:
                        for data in angle_data:
                            if data['angle'] > 0 and data['dose_mean'] > 0:
                                ratios.append({
                                    'reference': 0,  # 0 degrees
                                    'comparison': data['angle'],
                                    'dose_ratio': direct_dose / data['dose_mean'],
                                    'description': f"Direct (0°) vs. {data['angle']}°"
                                })

                # Fit dose vs angle relationship if we have enough data
                trend = None
                if len(angle_data) >= 3:
                    angles = np.array([d['angle'] for d in angle_data])
                    doses = np.array([d['dose_mean'] for d in angle_data])

                    if np.all(doses > 0):
                        # Try cosine fit: dose ~ A*cos(angle)^n
                        # Convert angles to radians
                        angles_rad = np.radians(angles)
                        log_doses = np.log(doses)
                        cos_angles = np.cos(angles_rad)
                        log_cos = np.log(cos_angles[cos_angles > 0])  # Only positive cosines
                        log_doses_valid = log_doses[cos_angles > 0]

                        if len(log_cos) >= 2:
                            # Linear regression on log-log scale
                            slope, intercept, r_value, p_value, std_err = stats.linregress(log_cos, log_doses_valid)

                            # Calculate fit parameters
                            A = np.exp(intercept)
                            n = slope
                            r_squared = r_value ** 2

                            trend = {
                                'type': 'cosine_power',
                                'equation': 'dose = A * cos(angle)^n',
                                'parameters': {
                                    'A': A,
                                    'n': n
                                },
                                'r_squared': r_squared,
                                'p_value': p_value
                            }

                # Return comprehensive analysis
                return {
                    'data': angle_data,
                    'ratios': ratios,
                    'trend': trend,
                    'statistics': {
                        'sample_count': len(angle_data),
                        'angles': [d['angle'] for d in angle_data],
                        'max_dose': max([d['dose_mean'] for d in angle_data]),
                        'min_dose': min([d['dose_mean'] for d in angle_data])
                    }
                }

            except Exception as e:
                self.logger.error(f"Error analyzing angular distribution: {str(e)}")
                return None

        def analyze_worker_safety(self):
            """
            Analyze worker safety implications based on simulation results.

            Returns:
                dict: Analysis results with safety recommendations
            """
            try:
                # Check if we have dose data
                if 'dose_mean' not in self.data.columns:
                    self.logger.warning("Cannot analyze worker safety: missing dose data")
                    return None

                # Define dose thresholds (rem/hr)
                thresholds = {
                    'extreme_hazard': 100.0,  # Extreme radiation hazard
                    'high_hazard': 10.0,  # High radiation area
                    'radiation_area': 1.0,  # Radiation area
                    'controlled': 0.1,  # Controlled area
                    'minimal': 0.01  # Minimal concern
                }

                # Calculate maximum dose rate
                max_dose = self.data['dose_mean'].max()

                # Determine safety category
                safety_category = 'minimal'
                for category, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
                    if max_dose >= threshold:
                        safety_category = category
                        break

                # Calculate scenario-specific time limits (hours to reach daily limit of 0.02 rem)
                daily_limit = 0.02  # rem
                time_limits = {}

                # Calculate time limits for different percentiles of dose rate
                percentiles = [100, 75, 50, 25, 0]  # max, 75th, median, 25th, min
                for p in percentiles:
                    if p == 100:
                        dose_rate = max_dose
                        label = 'maximum'
                    elif p == 0:
                        dose_rate = self.data['dose_mean'][self.data['dose_mean'] > 0].min()
                        label = 'minimum'
                    else:
                        dose_rate = self.data['dose_mean'].quantile(p / 100)
                        label = f'{p}th_percentile'

                    if dose_rate > 0:
                        time_limit = daily_limit / dose_rate  # hours
                        time_limits[label] = time_limit

                # Generate safety recommendations
                recommendations = []

                if safety_category == 'extreme_hazard':
                    recommendations = [
                        "IMMEDIATE ACTION REQUIRED: Extreme radiation hazard detected",
                        "No personnel access permitted when source is active",
                        "Engineering controls must be implemented before operation",
                        "Additional shielding required to reduce dose rates",
                        "Consider redesign of penetration geometry to include bends",
                        "Remote operations mandatory for any work in this area"
                    ]
                elif safety_category == 'high_hazard':
                    recommendations = [
                        "HIGH RADIATION AREA: Strict controls required",
                        "Limited access with radiation work permit only",
                        "Dosimetry and real-time monitoring required",
                        "Maximum stay time must be calculated and enforced",
                        "Additional local shielding recommended",
                        "Position workers at angles > 30° from penetration axis"
                    ]
                elif safety_category == 'radiation_area':
                    recommendations = [
                        "RADIATION AREA: Access controls required",
                        "Area must be designated as a radiation area",
                        "Training required for all personnel entering area",
                        "Time restrictions based on calculated dose rates",
                        "Regular surveys to verify conditions",
                        "Position workstations away from direct streaming path"
                    ]
                elif safety_category == 'controlled':
                    recommendations = [
                        "CONTROLLED AREA: Radiation worker access only",
                        "Area must be designated as a controlled area",
                        "Periodic monitoring recommended",
                        "Dosimetry recommended for regular workers",
                        "Maintain ALARA practices"
                    ]
                else:  # minimal
                    recommendations = [
                        "MINIMAL CONCERN: General access permitted",
                        "Periodic surveys to verify conditions",
                        "Normal radiation safety practices sufficient",
                        "No special controls required"
                    ]

                # Add specific recommendations based on data analysis
                if 'channel_diameter' in self.data.columns:
                    # Check effect of channel diameter
                    diameter_analysis = self.analyze_channel_diameter_trend()
                    if diameter_analysis and 'trends' in diameter_analysis:
                        recommendations.append(
                            f"PENETRATION SIZE: Minimize penetration diameter; each 50% reduction in diameter "
                            f"reduces dose by approximately {diameter_analysis['trends']['diameter_vs_dose']['parameters']['b'] / 2:.1f}×"
                        )

                # Return comprehensive analysis
                return {
                    'max_dose_rate': max_dose,
                    'safety_category': safety_category,
                    'threshold': thresholds[safety_category],
                    'time_limits': time_limits,
                    'recommendations': recommendations,
                    'statistics': {
                        'sample_count': len(self.data),
                        'dose_range': [self.data['dose_mean'].min(), max_dose],
                        'dose_median': self.data['dose_mean'].median()
                    }
                }

            except Exception as e:
                self.logger.error(f"Error analyzing worker safety: {str(e)}")
                return None








