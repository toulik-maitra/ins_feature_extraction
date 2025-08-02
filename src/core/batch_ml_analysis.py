#!/usr/bin/env python3
"""
Batch ML Analysis Workflow for INS Spectra
==========================================

This script provides a comprehensive workflow for analyzing INS spectra:
1. Single file analysis
2. Directory batch processing
3. Feature extraction for ML models
4. Organized output structure

Usage:
    python3 batch_ml_analysis.py --file <path_to_single_file>
    python3 batch_ml_analysis.py --directory <path_to_directory>
    python3 batch_ml_analysis.py --help
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import glob
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .ml_peak_analyzer import MLPeakAnalyzer
from ..config.output_config import create_output_structure
try:
    from ..utils.pb_ratio_analysis import PBRatioAnalyzer, analyze_pb_ratios_from_features
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from pb_ratio_analysis import PBRatioAnalyzer, analyze_pb_ratios_from_features

class BatchMLAnalyzer:
    """Comprehensive batch analyzer for INS spectra with ML feature extraction."""
    
    def __init__(self, output_dir="ml_analysis_results"):
        """
        Initialize the batch analyzer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to store all results
        """
        self.output_dir = Path(output_dir)
        
        # Create organized directory structure using centralized configuration
        dirs = create_output_structure(self.output_dir)
        
        # Assign directory paths
        self.plots_dir = dirs["plots"]
        self.features_dir = dirs["features"]
        self.summaries_dir = dirs["summaries"]
        self.logs_dir = dirs["logs"]
        self.pb_ratio_dir = dirs["pb_ratio"]
        self.main_plots_dir = dirs["main_plots"]
        self.baseline_plots_dir = dirs["baseline_plots"]
        self.peak_plots_dir = dirs["peak_plots"]
        self.kde_plots_dir = dirs["kde_plots"]
        
        # Initialize results storage
        self.all_features = []
        self.analysis_log = []
        
    def analyze_single_file(self, filepath, molecule_name=None, plot_individual=True, baseline_detector_type='physics_aware_als', is_experimental=False):
        """
        Analyze a single INS spectrum file.
        
        Parameters:
        -----------
        filepath : str
            Path to the spectrum CSV file
        molecule_name : str, optional
            Name for the molecule (extracted from filename if None)
        plot_individual : bool
            Whether to create individual plots for each file
            
        Returns:
        --------
        dict
            Analysis results and features
        """
        filepath = Path(filepath)
        
        if molecule_name is None:
            molecule_name = filepath.stem
        
        print(f"\n{'='*60}")
        print(f"ANALYZING: {molecule_name}")
        print(f"File: {filepath}")
        print(f"{'='*60}")
        
        try:
            # Always analyze in 0-3500 cm^-1
            analyzer = MLPeakAnalyzer(energy_range=(0, 3500), baseline_detector_type=baseline_detector_type, is_experimental=is_experimental)
            
            # Load spectrum data
            analyzer.load_spectrum_data(
                str(filepath),
                skiprows=0, energy_col="x", intensity_col="y"
            )
            
            # Restrict data to 0-3500 cm^-1 for peak detection
            mask = (analyzer.spectrum_data['energy'] >= 0) & (analyzer.spectrum_data['energy'] <= 3500)
            analyzer.spectrum_data['energy'] = analyzer.spectrum_data['energy'][mask]
            analyzer.spectrum_data['intensity'] = analyzer.spectrum_data['intensity'][mask]
            
            # Detect baseline
            analyzer.detect_baseline()
            
            # Determine if the file is experimental data
            is_experimental = "experimental" in filepath.name.lower() or "layered" in filepath.name.lower() or "random" in filepath.name.lower()
            
            # Use small smoothing window for both experimental and simulated data
            peak_params = dict(distance=2 if is_experimental else 5, prominence=0.005 if is_experimental else 0.02, width=1 if is_experimental else 2, smooth_window=4)
            fit_smooth_window = 4
            
            # Detect peaks with high sensitivity
            peak_positions, peak_intensities = analyzer.detect_peaks_from_spectrum(
                height=None,
                **peak_params,
                plot_detection=False
            )
            
            # Perform global Gaussian fitting
            fit_results = analyzer.fit_global_gaussians(smoothing=True, smooth_window=fit_smooth_window)
            
            if fit_results is None:
                print(f"✗ Fitting failed for {molecule_name}")
                return None
            
            # Outlier removal for peak-to-baseline ratio is handled in extract_ml_features(remove_ratio_outliers=True)
            features = analyzer.extract_ml_features(remove_ratio_outliers=True)
            features['molecule_name'] = molecule_name
            features['filename'] = filepath.name
            features['filepath'] = str(filepath)
            features['analysis_timestamp'] = datetime.now().isoformat()
            
            # Create individual plots if requested
            if plot_individual:
                # Main analysis plot
                plot_filename = f"{molecule_name}_analysis.pdf"
                plot_path = self.main_plots_dir / plot_filename
                analyzer.plot_publication_quality(
                    save_path=str(plot_path),
                    dpi=300,
                    figsize=(16, 14)
                )
                # Baseline detection plot
                baseline_plot_filename = f"{molecule_name}_baseline_detection.pdf"
                baseline_plot_path = self.baseline_plots_dir / baseline_plot_filename
                self._create_baseline_plot(analyzer, str(baseline_plot_path))
                # Peak detection plot
                peak_plot_filename = f"{molecule_name}_peak_detection.pdf"
                peak_plot_path = self.peak_plots_dir / peak_plot_filename
                self._create_peak_plot(analyzer, str(peak_plot_path))
                # KDE density plot
                kde_plot_filename = f"{molecule_name}_kde_density.pdf"
                kde_plot_path = self.kde_plots_dir / kde_plot_filename
                self._create_kde_plot(analyzer, features, str(kde_plot_path))
            
            # Save individual features
            features_filename = f"{molecule_name}_features.csv"
            features_path = self.features_dir / features_filename
            analyzer.save_features_to_csv(str(features_path))
            
            # Log analysis
            log_entry = {
                'molecule_name': molecule_name,
                'filename': filepath.name,
                'num_peaks': features['num_peaks'],
                'r_squared': features['r_squared'],
                'rmse': features['rmse'],
                'mean_fwhm': features['mean_fwhm'],
                'total_area': features['total_area'],
                'status': 'success'
            }
            
            print(f"✓ Analysis completed successfully")
            print(f"  - Peaks detected: {features['num_peaks']}")
            print(f"  - Fit quality (R²): {features['r_squared']:.4f}")
            print(f"  - Mean FWHM: {features['mean_fwhm']:.2f} cm⁻¹")
            
            return features, log_entry
            
        except Exception as e:
            print(f"✗ Error analyzing {molecule_name}: {e}")
            log_entry = {
                'molecule_name': molecule_name,
                'filename': filepath.name,
                'num_peaks': 0,
                'r_squared': 0,
                'rmse': 0,
                'mean_fwhm': 0,
                'total_area': 0,
                'status': f'error: {str(e)}'
            }
            return None, log_entry
    
    def analyze_directory(self, directory_path, file_pattern="*.csv", plot_individual=False, baseline_detector_type='physics_aware_als', is_experimental=False):
        """
        Analyze all CSV files in a directory.
        
        Parameters:
        -----------
        directory_path : str
            Path to directory containing CSV files
        file_pattern : str
            Pattern to match files (default: "*.csv")
        plot_individual : bool
            Whether to create individual plots (can be slow for many files)
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            print(f"✗ Directory not found: {directory_path}")
            return
        
        # Find all CSV files
        csv_files = list(directory_path.glob(file_pattern))
        
        if not csv_files:
            print(f"✗ No CSV files found in {directory_path}")
            return
        
        print(f"\n{'='*60}")
        print(f"BATCH ANALYSIS: {len(csv_files)} files found")
        print(f"Directory: {directory_path}")
        print(f"{'='*60}")
        
        # Process each file
        for i, filepath in enumerate(csv_files, 1):
            print(f"\nProcessing file {i}/{len(csv_files)}: {filepath.name}")
            
            # Extract molecule name from filename
            molecule_name = self._extract_molecule_name(filepath.name)
            
            # Analyze file
            result = self.analyze_single_file(
                filepath, 
                molecule_name=molecule_name,
                plot_individual=plot_individual,
                baseline_detector_type=baseline_detector_type,
                is_experimental=is_experimental
            )
            
            if result is not None:
                features, log_entry = result
                self.all_features.append(features)
                self.analysis_log.append(log_entry)
            else:
                # Add failed analysis to log
                self.analysis_log.append(log_entry)
        
        # Create comprehensive results
        self._create_batch_summary()
    
    def _extract_molecule_name(self, filename):
        """Extract meaningful molecule name from filename."""
        # Remove common prefixes and suffixes
        name = filename.replace('.csv', '')
        name = name.replace('xy_', '')
        name = name.replace('_2.00', '')
        name = name.replace('_experimental', '_exp')
        name = name.replace('_layered', '_lay')
        name = name.replace('_random', '_rand')
        
        return name
    
    def _create_baseline_plot(self, analyzer, save_path):
        """Create baseline detection plot."""
        try:
            import matplotlib.pyplot as plt
            
            if not hasattr(analyzer, 'baseline_data') or analyzer.baseline_data is None:
                print("No baseline data available for plotting")
                return
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = analyzer.spectrum_data['energy']
            y = analyzer.spectrum_data['intensity']
            baseline = analyzer.baseline_data['baseline']
            
            ax.plot(x, y, 'k-', linewidth=1, alpha=0.7, label='Spectrum')
            ax.plot(x, baseline, 'r-', linewidth=2, label='Baseline')
            ax.fill_between(x, baseline, y, alpha=0.3, color='blue', label='Peak Area')
            
            ax.set_xlabel('Energy (cm$^{-1}$)', fontsize=12)
            ax.set_ylabel('Intensity (a.u.)', fontsize=12)
            ax.set_title('Baseline Detection', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(analyzer.energy_range)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            print(f"✓ Baseline plot saved to: {save_path}")
            
        except Exception as e:
            print(f"✗ Error creating baseline plot: {e}")
    
    def _create_peak_plot(self, analyzer, save_path):
        """Create peak detection plot."""
        try:
            import matplotlib.pyplot as plt
            
            if not hasattr(analyzer, 'peak_data') or analyzer.peak_data is None:
                print("No peak data available for plotting")
                return
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = analyzer.spectrum_data['energy']
            y = analyzer.spectrum_data['intensity']
            peak_positions = analyzer.peak_data['positions']
            peak_intensities = analyzer.peak_data['intensities']
            
            ax.plot(x, y, 'k-', linewidth=1, alpha=0.7, label='Spectrum')
            ax.scatter(peak_positions, peak_intensities, c='red', s=50, alpha=0.8, 
                      edgecolors='black', linewidth=0.5, label='Detected Peaks')
            
            ax.set_xlabel('Energy (cm$^{-1}$)', fontsize=12)
            ax.set_ylabel('Intensity (a.u.)', fontsize=12)
            ax.set_title('Peak Detection', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(analyzer.energy_range)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            print(f"✓ Peak detection plot saved to: {save_path}")
            
        except Exception as e:
            print(f"✗ Error creating peak plot: {e}")
    
    def _create_kde_plot(self, analyzer, features, save_path):
        """Create KDE density plot for peak distributions."""
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import gaussian_kde
            
            if not hasattr(analyzer, 'fit_results') or analyzer.fit_results is None:
                print("No fit results available for KDE plot")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            peak_params = analyzer.fit_results['peak_params']
            centers = [p['center'] for p in peak_params]
            amplitudes = [p['amplitude'] for p in peak_params]
            fwhms = [p['fwhm'] for p in peak_params]
            areas = [p['area'] for p in peak_params]
            
            # Peak center distribution
            if len(centers) > 1:
                kde_centers = gaussian_kde(centers)
                x_centers = np.linspace(min(centers), max(centers), 100)
                ax1.plot(x_centers, kde_centers(x_centers), 'b-', linewidth=2)
                ax1.hist(centers, bins=max(3, len(centers)//3), alpha=0.3, density=True, color='blue')
            ax1.set_xlabel('Peak Center (cm$^{-1}$)', fontsize=12)
            ax1.set_ylabel('Density', fontsize=12)
            ax1.set_title('Peak Center Distribution', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Amplitude distribution
            if len(amplitudes) > 1:
                kde_amps = gaussian_kde(amplitudes)
                x_amps = np.linspace(min(amplitudes), max(amplitudes), 100)
                ax2.plot(x_amps, kde_amps(x_amps), 'g-', linewidth=2)
                ax2.hist(amplitudes, bins=max(3, len(amplitudes)//3), alpha=0.3, density=True, color='green')
            ax2.set_xlabel('Peak Amplitude (a.u.)', fontsize=12)
            ax2.set_ylabel('Density', fontsize=12)
            ax2.set_title('Peak Amplitude Distribution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # FWHM distribution
            if len(fwhms) > 1:
                kde_fwhms = gaussian_kde(fwhms)
                x_fwhms = np.linspace(min(fwhms), max(fwhms), 100)
                ax3.plot(x_fwhms, kde_fwhms(x_fwhms), 'r-', linewidth=2)
                ax3.hist(fwhms, bins=max(3, len(fwhms)//3), alpha=0.3, density=True, color='red')
            ax3.set_xlabel('Peak FWHM (cm$^{-1}$)', fontsize=12)
            ax3.set_ylabel('Density', fontsize=12)
            ax3.set_title('Peak Width Distribution', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Area distribution
            if len(areas) > 1:
                kde_areas = gaussian_kde(areas)
                x_areas = np.linspace(min(areas), max(areas), 100)
                ax4.plot(x_areas, kde_areas(x_areas), 'purple', linewidth=2)
                ax4.hist(areas, bins=max(3, len(areas)//3), alpha=0.3, density=True, color='purple')
            ax4.set_xlabel('Peak Area (a.u.)', fontsize=12)
            ax4.set_ylabel('Density', fontsize=12)
            ax4.set_title('Peak Area Distribution', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            print(f"✓ KDE density plot saved to: {save_path}")
            
        except Exception as e:
            print(f"✗ Error creating KDE plot: {e}")
    
    def _create_pb_ratio_analysis(self):
        """Create comprehensive peak-to-baseline ratio analysis."""
        if not self.all_features:
            print("No features available for P/B ratio analysis")
            return
        
        print(f"\n{'='*60}")
        print("CREATING PEAK-TO-BASELINE RATIO ANALYSIS")
        print(f"{'='*60}")
        
        try:
            # Filter features that have peak-to-baseline ratios
            features_with_ratios = []
            for features in self.all_features:
                if ('peak_to_baseline_ratios' in features and 
                    len(features['peak_to_baseline_ratios']) > 0):
                    features_with_ratios.append(features)
            
            if not features_with_ratios:
                print("No samples with peak-to-baseline ratios found")
                return
            
            print(f"Found {len(features_with_ratios)} samples with P/B ratios")
            
            # Create P/B ratio analyzer
            pb_analyzer = PBRatioAnalyzer(output_dir=str(self.pb_ratio_dir))
            
            # Add data for each sample
            for features in features_with_ratios:
                molecule_name = features.get('molecule_name', 'Unknown')
                ratios = features['peak_to_baseline_ratios']
                energy_positions = features.get('peak_centers', None)
                peak_amplitudes = features.get('peak_amplitudes', None)
                
                pb_analyzer.add_ratio_data(
                    ratios=ratios,
                    sample_label=molecule_name,
                    energy_positions=energy_positions,
                    peak_amplitudes=peak_amplitudes
                )
            
            # Create all plots and analyses
            pb_analyzer.create_all_plots()
            
            # Save ratio data to CSV
            pb_analyzer.save_ratio_data_to_csv("pb_ratio_data.csv")
            
            print(f"✓ P/B ratio analysis completed successfully")
            print(f"  - {len(features_with_ratios)} samples analyzed")
            print(f"  - Plots saved to: {self.pb_ratio_dir}/plots/")
            print(f"  - Statistics saved to: {self.pb_ratio_dir}/statistics/")
            print(f"  - Comparisons saved to: {self.pb_ratio_dir}/comparisons/")
            
        except Exception as e:
            print(f"✗ Error in P/B ratio analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_batch_summary(self):
        """Create comprehensive summary of batch analysis."""
        if not self.all_features:
            print("No successful analyses to summarize.")
            return
        
        print(f"\n{'='*60}")
        print("CREATING BATCH SUMMARY")
        print(f"{'='*60}")
        
        # Create combined features DataFrame
        features_df = pd.DataFrame(self.all_features)
        
        # Save combined features
        combined_features_path = self.features_dir / "all_molecules_features.csv"
        features_df.to_csv(combined_features_path, index=False)
        print(f"✓ Combined features saved: {combined_features_path}")
        
        # Create analysis summary
        summary_df = pd.DataFrame(self.analysis_log)
        summary_path = self.summaries_dir / "analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Analysis summary saved: {summary_path}")
        
        # Create statistical summary
        successful_analyses = summary_df[summary_df['status'] == 'success']
        
        if len(successful_analyses) > 0:
            stats_summary = {
                'total_files_processed': len(summary_df),
                'successful_analyses': len(successful_analyses),
                'failed_analyses': len(summary_df) - len(successful_analyses),
                'success_rate': len(successful_analyses) / len(summary_df) * 100,
                'avg_num_peaks': successful_analyses['num_peaks'].mean(),
                'avg_r_squared': successful_analyses['r_squared'].mean(),
                'avg_rmse': successful_analyses['rmse'].mean(),
                'avg_mean_fwhm': successful_analyses['mean_fwhm'].mean(),
                'avg_total_area': successful_analyses['total_area'].mean(),
            }
            
            stats_df = pd.DataFrame([stats_summary])
            stats_path = self.summaries_dir / "statistical_summary.csv"
            stats_df.to_csv(stats_path, index=False)
            print(f"✓ Statistical summary saved: {stats_path}")
            
            # Print summary statistics
            print(f"\nBATCH ANALYSIS SUMMARY:")
            print(f"  Total files processed: {stats_summary['total_files_processed']}")
            print(f"  Successful analyses: {stats_summary['successful_analyses']}")
            print(f"  Success rate: {stats_summary['success_rate']:.1f}%")
            print(f"  Average peaks per spectrum: {stats_summary['avg_num_peaks']:.1f}")
            print(f"  Average fit quality (R²): {stats_summary['avg_r_squared']:.4f}")
        
        # Create ML-ready dataset (remove unnecessary features)
        # Remove individual peak arrays and metadata columns (but keep sample identifier)
        columns_to_remove = [
            'filename', 'filepath', 'analysis_timestamp',
            'peak_centers', 'peak_amplitudes', 'peak_fwhms', 'peak_areas', 
            'peak_to_baseline_ratios'
        ]
        
        ml_features = features_df.drop(columns_to_remove, axis=1, errors='ignore')
        
        # Reorder columns for better ML analysis
        essential_features = [
            # Sample identifier (KEEP THIS!)
            'molecule_name',
            
            # Core spectral features
            'num_peaks', 'peak_density', 'total_spectral_area', 'peak_area_fraction',
            'energy_span', 'mean_center', 'std_center',
            
            # Amplitude features
            'mean_amplitude', 'std_amplitude', 'max_amplitude', 'min_amplitude',
            'amplitude_range', 'amplitude_cv', 'amplitude_skewness', 'amplitude_kurtosis',
            
            # Width features
            'mean_fwhm', 'std_fwhm', 'max_fwhm', 'min_fwhm', 'fwhm_range', 'fwhm_cv',
            'fwhm_skewness', 'fwhm_kurtosis',
            
            # Area features
            'total_area', 'mean_area', 'std_area', 'area_cv', 'max_area', 'min_area',
            'area_range', 'area_skewness', 'area_kurtosis', 'largest_peak_area',
            'smallest_peak_area', 'largest_peak_area_fraction', 'area_median',
            'area_percentile_25', 'area_percentile_75', 'area_iqr',
            
            # Spectral area features
            'non_peak_area', 'non_peak_area_fraction',
            
            # Baseline features
            'detected_baseline_area', 'detected_baseline_area_fraction',
            'signal_above_baseline_area', 'signal_above_baseline_fraction',
            
            # Peak-to-baseline ratio features
            'mean_peak_to_baseline_ratio', 'std_peak_to_baseline_ratio',
            'max_peak_to_baseline_ratio', 'min_peak_to_baseline_ratio',
            'median_peak_to_baseline_ratio', 'peak_to_baseline_ratio_cv',
            'num_ratio_outliers_removed',
            
            # Energy region features
            'low_energy_peaks', 'mid_energy_peaks', 'high_energy_peaks',
            
            # Peak spacing features
            'mean_peak_spacing', 'std_peak_spacing',
            
            # Fit quality features
            'r_squared', 'rmse', 'baseline'
        ]
        
        # Keep only essential features that exist in the dataset
        available_features = [col for col in essential_features if col in ml_features.columns]
        ml_features_clean = ml_features[available_features]
        
        ml_dataset_path = self.features_dir / "ml_dataset.csv"
        ml_features_clean.to_csv(ml_dataset_path, index=False)
        print(f"✓ ML-ready dataset saved: {ml_dataset_path}")
        print(f"  - {len(ml_features_clean.columns)} essential features")
        print(f"  - {len(ml_features_clean)} samples")
        
        # Create peak-to-baseline ratio analysis
        self._create_pb_ratio_analysis()
        
        print(f"\n✓ All results saved in: {self.output_dir}")
        print(f"  - Individual features: {self.features_dir}")
        print(f"  - Combined dataset: {ml_dataset_path}")
        print(f"  - Analysis summaries: {self.summaries_dir}")
        print(f"  - Individual plots: {self.plots_dir}")
        print(f"  - P/B ratio analysis: {self.pb_ratio_dir}")

def main():
    """Main function to handle command line arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Batch ML Analysis for INS Spectra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python3 batch_ml_analysis.py --file "path/to/spectrum.csv"
  
  # Analyze all CSV files in a directory
  python3 batch_ml_analysis.py --directory "path/to/spectra/"
  
  # Analyze with individual plots
  python3 batch_ml_analysis.py --directory "path/to/spectra/" --plot-individual
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Path to single CSV file to analyze')
    group.add_argument('--directory', type=str, help='Path to directory containing CSV files')
    
    parser.add_argument('--output-dir', type=str, default='ml_analysis_results',
                       help='Output directory for results (default: ml_analysis_results)')
    parser.add_argument('--plot-individual', action='store_true',
                       help='Create individual plots for each file (slower for many files)')
    parser.add_argument('--file-pattern', type=str, default='*.csv',
                       help='File pattern to match (default: *.csv)')
    
    args = parser.parse_args()
    
    # Initialize batch analyzer
    analyzer = BatchMLAnalyzer(output_dir=args.output_dir)
    
    # Run analysis based on input type
    if args.file:
        print("SINGLE FILE ANALYSIS")
        print("="*60)
        result = analyzer.analyze_single_file(args.file, plot_individual=True)
        if result is not None:
            features, log_entry = result
            analyzer.all_features.append(features)
            analyzer.analysis_log.append(log_entry)
            analyzer._create_batch_summary()
    
    elif args.directory:
        print("DIRECTORY BATCH ANALYSIS")
        print("="*60)
        analyzer.analyze_directory(
            args.directory,
            file_pattern=args.file_pattern,
            plot_individual=args.plot_individual
        )

if __name__ == "__main__":
    main() 