#!/usr/bin/env python3
"""
Peak-to-Baseline Ratio Analysis Module
======================================

This module provides comprehensive analysis and visualization of peak-to-baseline ratios
from INS spectra analysis. It creates publication-quality plots and statistical analysis
for comparing peak-to-baseline ratios across different samples.

Features:
- Violin, box, and beeswarm plots
- Statistical analysis tables
- Density distribution plots
- Population comparison analysis
- Temperature-dependent analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, iqr, gaussian_kde, mode, ks_2samp
from scipy import signal
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Set up matplotlib for publication-quality figures
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14
})

# Use professional style
plt.style.use('default')
sns.set_style("white")
sns.set_context("talk", font_scale=1.2)

class PBRatioAnalyzer:
    """Comprehensive peak-to-baseline ratio analyzer and visualizer."""
    
    def __init__(self, output_dir: str = "pb_ratio_analysis"):
        """
        Initialize the P/B ratio analyzer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save all analysis plots and results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different plot types
        self.plots_dir = self.output_dir / "plots"
        self.stats_dir = self.output_dir / "statistics"
        self.comparison_dir = self.output_dir / "comparisons"
        
        for d in [self.plots_dir, self.stats_dir, self.comparison_dir]:
            d.mkdir(exist_ok=True)
        
        self.ratio_data = []
        self.sample_labels = []
    
    def add_ratio_data(self, ratios: np.ndarray, sample_label: str, 
                      energy_positions: Optional[np.ndarray] = None,
                      peak_amplitudes: Optional[np.ndarray] = None,
                      baseline_values: Optional[np.ndarray] = None):
        """
        Add peak-to-baseline ratio data for a sample.
        
        Parameters:
        -----------
        ratios : np.ndarray
            Peak-to-baseline ratios
        sample_label : str
            Label for the sample
        energy_positions : np.ndarray, optional
            Energy positions of the peaks
        peak_amplitudes : np.ndarray, optional
            Peak amplitudes
        baseline_values : np.ndarray, optional
            Baseline values at peak positions
        """
        # Create data dictionary
        data_dict = {
            'ratios': ratios,
            'sample_label': sample_label,
            'energy_positions': energy_positions,
            'peak_amplitudes': peak_amplitudes,
            'baseline_values': baseline_values
        }
        
        self.ratio_data.append(data_dict)
        self.sample_labels.append(sample_label)
    
    def calculate_statistics(self, ratios: np.ndarray) -> Dict:
        """
        Calculate comprehensive statistical measures for ratio data.
        
        Parameters:
        -----------
        ratios : np.ndarray
            Peak-to-baseline ratios
            
        Returns:
        --------
        dict
            Dictionary of statistical measures
        """
        if len(ratios) == 0:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0,
                'cv': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'iqr': 0.0,
                'percentile_25': 0.0,
                'percentile_75': 0.0,
                'percentile_90': 0.0,
                'percentile_95': 0.0,
                'percentile_99': 0.0
            }
        
        return {
            'count': len(ratios),
            'mean': np.mean(ratios),
            'median': np.median(ratios),
            'std': np.std(ratios),
            'min': np.min(ratios),
            'max': np.max(ratios),
            'range': np.max(ratios) - np.min(ratios),
            'cv': np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else 0,
            'skewness': skew(ratios),
            'kurtosis': kurtosis(ratios),
            'iqr': iqr(ratios),
            'percentile_25': np.percentile(ratios, 25),
            'percentile_75': np.percentile(ratios, 75),
            'percentile_90': np.percentile(ratios, 90),
            'percentile_95': np.percentile(ratios, 95),
            'percentile_99': np.percentile(ratios, 99)
        }
    
    def create_statistics_table(self) -> pd.DataFrame:
        """
        Create a comprehensive statistics table for all samples.
        
        Returns:
        --------
        pd.DataFrame
            Statistics table
        """
        stats_data = []
        
        for data_dict in self.ratio_data:
            ratios = data_dict['ratios']
            sample_label = data_dict['sample_label']
            stats = self.calculate_statistics(ratios)
            stats['Sample'] = sample_label
            stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        
        # Reorder columns for better presentation
        column_order = [
            'Sample', 'count', 'mean', 'median', 'std', 'min', 'max', 'range',
            'cv', 'skewness', 'kurtosis', 'iqr', 'percentile_25', 'percentile_75',
            'percentile_90', 'percentile_95', 'percentile_99'
        ]
        
        stats_df = stats_df[column_order]
        
        # Save statistics table
        stats_path = self.stats_dir / "pb_ratio_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        
        # Create and save statistics table plot
        self._plot_statistics_table(stats_df)
        
        return stats_df
    
    def _plot_statistics_table(self, stats_df: pd.DataFrame):
        """Create a visual statistics table."""
        # Select key statistics for visualization
        key_stats = ['Sample', 'count', 'mean', 'median', 'std', 'cv', 'skewness', 'kurtosis']
        plot_df = stats_df[key_stats].copy()
        
        # Round numerical values for better display
        for col in plot_df.columns:
            if col != 'Sample':
                plot_df[col] = plot_df[col].round(4)
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(plot_df) * 0.4)))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=plot_df.values,
            colLabels=plot_df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Statistical Measures for Peak-to-Baseline Ratios', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save the table
        table_path = self.stats_dir / "statistics_table.pdf"
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_violin_plot(self):
        """Create a violin plot of all peak-to-baseline ratios."""
        if not self.ratio_data:
            print("No ratio data available for plotting")
            return
        
        # Prepare data for plotting
        plot_data = []
        for data_dict in self.ratio_data:
            ratios = data_dict['ratios']
            sample_label = data_dict['sample_label']
            for ratio in ratios:
                plot_data.append({'Sample': sample_label, 'Peak_Baseline_Ratio': ratio})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create violin plot
        plt.figure(figsize=(12, max(8, len(plot_df['Sample'].unique()) * 0.4)))
        ax = sns.violinplot(
            data=plot_df, 
            y='Sample', 
            x='Peak_Baseline_Ratio', 
            bw_adjust=0.3, 
            inner="quart", 
            density_norm='area', 
            cut=0, 
            palette="viridis"
        )
        
        # Add individual data points
        sns.swarmplot(
            data=plot_df, 
            y='Sample', 
            x='Peak_Baseline_Ratio', 
            size=2.5, 
            color=".25", 
            alpha=0.6, 
            ax=ax
        )
        
        ax.set_title('Violin Plot of Peak-to-Baseline Ratios', fontsize=16)
        ax.set_xlabel('Peak-to-Baseline Ratio')
        ax.set_ylabel('')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "violin_plot.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_box_plot(self):
        """Create a box plot of all peak-to-baseline ratios."""
        if not self.ratio_data:
            print("No ratio data available for plotting")
            return
        
        # Prepare data for plotting
        plot_data = []
        for data_dict in self.ratio_data:
            ratios = data_dict['ratios']
            sample_label = data_dict['sample_label']
            for ratio in ratios:
                plot_data.append({'Sample': sample_label, 'Peak_Baseline_Ratio': ratio})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create box plot
        plt.figure(figsize=(12, max(8, len(plot_df['Sample'].unique()) * 0.4)))
        ax = sns.boxplot(
            data=plot_df, 
            y='Sample', 
            x='Peak_Baseline_Ratio', 
            palette="cubehelix"
        )
        
        # Add individual data points
        sns.swarmplot(
            data=plot_df, 
            y='Sample', 
            x='Peak_Baseline_Ratio', 
            size=2.5, 
            color=".25", 
            alpha=0.6, 
            ax=ax
        )
        
        ax.set_title('Box Plot of Peak-to-Baseline Ratios', fontsize=16)
        ax.set_xlabel('Peak-to-Baseline Ratio')
        ax.set_ylabel('')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "box_plot.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_beeswarm_plot(self):
        """Create a beeswarm plot of all peak-to-baseline ratios."""
        if not self.ratio_data:
            print("No ratio data available for plotting")
            return
        
        # Prepare data for plotting
        plot_data = []
        for data_dict in self.ratio_data:
            ratios = data_dict['ratios']
            sample_label = data_dict['sample_label']
            for ratio in ratios:
                plot_data.append({'Sample': sample_label, 'Peak_Baseline_Ratio': ratio})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create beeswarm plot
        plt.figure(figsize=(12, max(8, len(plot_df['Sample'].unique()) * 0.4)))
        ax = sns.swarmplot(
            data=plot_df, 
            y='Sample', 
            x='Peak_Baseline_Ratio', 
            size=3, 
            alpha=0.8, 
            palette='Set2'
        )
        
        ax.set_title('Beeswarm Plot of Peak-to-Baseline Ratios', fontsize=16)
        ax.set_xlabel('Peak-to-Baseline Ratio')
        ax.set_ylabel('')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "beeswarm_plot.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_density_plot(self):
        """Create a density plot of normalized peak-to-baseline ratios."""
        if not self.ratio_data:
            print("No ratio data available for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        
        for data_dict in self.ratio_data:
            ratios = data_dict['ratios']
            sample_label = data_dict['sample_label']
            
            if len(ratios) > 0:
                # Log normalization (add small constant to avoid log(0))
                normalized_ratios = np.log(ratios + 1e-10)
                
                # Calculate KDE
                kde = gaussian_kde(normalized_ratios)
                x_range = np.linspace(normalized_ratios.min(), normalized_ratios.max(), 200)
                density = kde(x_range)
                
                plt.plot(x_range, density, label=sample_label, alpha=0.7, linewidth=2)
        
        plt.xlabel('Log Normalized Peak-to-Baseline Ratio')
        plt.ylabel('Density')
        plt.title('Density Distribution of Normalized Peak Ratios')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "density_distribution.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_ridgeline_plot(self):
        """Create a ridgeline plot of peak-to-baseline ratios."""
        if not self.ratio_data:
            print("No ratio data available for plotting")
            return
        
        # Prepare data for plotting
        plot_data = []
        for data_dict in self.ratio_data:
            ratios = data_dict['ratios']
            sample_label = data_dict['sample_label']
            for ratio in ratios:
                plot_data.append({'Sample': sample_label, 'Ratio': ratio})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create ridgeline plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for sample_label in plot_df['Sample'].unique():
            sample_data = plot_df[plot_df['Sample'] == sample_label]['Ratio']
            if len(sample_data) > 0:
                sns.kdeplot(data=sample_data, label=sample_label, ax=ax, alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Peak-to-Baseline Ratio')
        ax.set_ylabel('Density')
        ax.set_title('Ridgeline Plot of Peak-to-Baseline Ratios')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "ridgeline_plot.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_analysis(self):
        """Create comprehensive comparison analysis between samples."""
        if len(self.ratio_data) < 2:
            print("Need at least 2 samples for comparison analysis")
            return
        
        # Calculate KS test p-values between all pairs
        samples = [data['sample_label'] for data in self.ratio_data]
        p_value_matrix = pd.DataFrame(index=samples, columns=samples)
        
        for i, sample1 in enumerate(samples):
            for j, sample2 in enumerate(samples):
                if i == j:
                    p_value_matrix.loc[sample1, sample2] = 1.0
                else:
                    ratios1 = self.ratio_data[i]['ratios']
                    ratios2 = self.ratio_data[j]['ratios']
                    
                    if len(ratios1) > 0 and len(ratios2) > 0:
                        _, p_value = ks_2samp(ratios1, ratios2)
                        p_value_matrix.loc[sample1, sample2] = p_value
                    else:
                        p_value_matrix.loc[sample1, sample2] = 1.0
        
        # Save p-value matrix
        p_value_path = self.comparison_dir / "ks_test_p_values.csv"
        p_value_matrix.to_csv(p_value_path)
        
        # Create heatmap of p-values
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            p_value_matrix.astype(float), 
            annot=True, 
            cmap='viridis_r', 
            vmin=0, 
            vmax=1,
            fmt='.3f'
        )
        plt.title('Kolmogorov-Smirnov Test P-Values\n(Values < 0.05 indicate significant differences)')
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = self.comparison_dir / "ks_test_heatmap.pdf"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return p_value_matrix
    
    def create_all_plots(self):
        """Create all available plots and analyses."""
        print("Creating comprehensive P/B ratio analysis...")
        
        # Create statistics table
        stats_df = self.create_statistics_table()
        print(f"✓ Statistics table created with {len(stats_df)} samples")
        
        # Create individual plots
        self.create_violin_plot()
        print("✓ Violin plot created")
        
        self.create_box_plot()
        print("✓ Box plot created")
        
        self.create_beeswarm_plot()
        print("✓ Beeswarm plot created")
        
        self.create_density_plot()
        print("✓ Density plot created")
        
        self.create_ridgeline_plot()
        print("✓ Ridgeline plot created")
        
        # Create comparison analysis
        if len(self.ratio_data) >= 2:
            p_value_matrix = self.create_comparison_analysis()
            print("✓ Comparison analysis created")
        
        print(f"\nAll plots saved to: {self.output_dir}")
        print(f"Statistics saved to: {self.stats_dir}")
        print(f"Comparison results saved to: {self.comparison_dir}")
    
    def save_ratio_data_to_csv(self, filename: str = "pb_ratio_data.csv"):
        """Save all ratio data to a CSV file for further analysis."""
        if not self.ratio_data:
            print("No ratio data to save")
            return
        
        # Prepare data for CSV
        csv_data = []
        for data_dict in self.ratio_data:
            ratios = data_dict['ratios']
            sample_label = data_dict['sample_label']
            energy_positions = data_dict.get('energy_positions', [None] * len(ratios))
            peak_amplitudes = data_dict.get('peak_amplitudes', [None] * len(ratios))
            baseline_values = data_dict.get('baseline_values', [None] * len(ratios))
            
            for i, ratio in enumerate(ratios):
                csv_data.append({
                    'Sample': sample_label,
                    'Peak_Baseline_Ratio': ratio,
                    'Energy_Position': energy_positions[i] if energy_positions is not None and i < len(energy_positions) else None,
                    'Peak_Amplitude': peak_amplitudes[i] if peak_amplitudes is not None and i < len(peak_amplitudes) else None,
                    'Baseline_Value': baseline_values[i] if baseline_values is not None and i < len(baseline_values) else None
                })
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / filename
        csv_df.to_csv(csv_path, index=False)
        print(f"✓ Ratio data saved to: {csv_path}")
        
        return csv_path


def analyze_pb_ratios_from_features(features_list: List[Dict], output_dir: str = "pb_ratio_analysis") -> PBRatioAnalyzer:
    """
    Analyze peak-to-baseline ratios from a list of feature dictionaries.
    
    Parameters:
    -----------
    features_list : List[Dict]
        List of feature dictionaries from ML analysis
    output_dir : str
        Directory to save analysis results
        
    Returns:
    --------
    PBRatioAnalyzer
        Configured analyzer with all data loaded
    """
    analyzer = PBRatioAnalyzer(output_dir)
    
    for features in features_list:
        molecule_name = features.get('molecule_name', 'Unknown')
        
        # Extract peak-to-baseline ratios
        if 'peak_to_baseline_ratios' in features and len(features['peak_to_baseline_ratios']) > 0:
            ratios = features['peak_to_baseline_ratios']
            
            # Extract additional data if available
            energy_positions = features.get('peak_centers', None)
            peak_amplitudes = features.get('peak_amplitudes', None)
            baseline_values = None  # Not directly available in features
            
            analyzer.add_ratio_data(
                ratios=ratios,
                sample_label=molecule_name,
                energy_positions=energy_positions,
                peak_amplitudes=peak_amplitudes,
                baseline_values=baseline_values
            )
    
    return analyzer 