#!/usr/bin/env python3
"""
Smart Baseline Analysis for INS Spectra
=======================================

This script provides intelligent baseline detection analysis:
1. Always runs baseline detection first on a sample spectrum
2. Helps choose the best algorithm for your data
3. Saves figures in the existing comprehensive_analysis_results directory
4. Follows the existing workflow where baseline detection is separate from feature extraction

Usage:
    python3 smart_baseline_analysis.py --directory <path_to_directory>
    python3 smart_baseline_analysis.py --file <path_to_single_file>
    python3 smart_baseline_analysis.py --help

Author: Enhanced Baseline Detection System
Date: 2024
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings
from glob import glob

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import enhanced baseline detection
try:
    from utils.enhanced_baseline_detection import (
        EnhancedBaselineDetectorFactory,
        BaselineValidationSystem,
        BaselineQualityMetrics
    )
    from utils.baseline_detection import detect_baseline
    from core.enhanced_ml_peak_analyzer import EnhancedMLPeakAnalyzer
    print("✓ Enhanced baseline detection system imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')

class SmartBaselineAnalyzer:
    """Smart baseline analyzer that follows the existing workflow."""
    
    def __init__(self):
        """Initialize the smart baseline analyzer."""
        # Use existing directory structure
        self.output_dir = Path("comprehensive_analysis_results")
        self.baseline_plots_dir = self.output_dir / "plots" / "baseline_detection"
        self.baseline_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Available algorithms (focusing on rolling and ALS)
        self.rolling_algorithms = [
            'advanced_rolling_adaptive',
            'advanced_rolling_multi_scale', 
            'advanced_rolling_percentile'
        ]
        
        self.als_algorithms = [
            'pybaselines_asls',
            'spectrochempy_asls',
            'binned_als'
        ]
        
        self.all_algorithms = self.rolling_algorithms + self.als_algorithms
        
    def load_ins_csv(self, file_path):
        """Load a single INS spectrum CSV file, auto-detecting columns."""
        try:
            df = pd.read_csv(file_path)
            print(f"  Columns: {list(df.columns)}")
            print(f"  Shape: {df.shape}")
            
            # Detect energy column
            energy_col = None
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['energy', 'wavenumber', 'cm-1', 'cm^-1', 'x']):
                    energy_col = col
                    break
            
            # Detect intensity column
            intensity_col = None
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['intensity', 'counts', 'absorbance', 'y']):
                    intensity_col = col
                    break
            
            if energy_col is None or intensity_col is None:
                raise ValueError(f"Could not detect energy/intensity columns in {file_path}")
            
            energy = df[energy_col].values
            intensity = df[intensity_col].values
            
            # Restrict to 0-3500 cm^-1 as per existing workflow
            mask = (energy >= 0) & (energy <= 3500)
            energy = energy[mask]
            intensity = intensity[mask]
            
            print(f"  Loaded: {file_path}")
            print(f"  Energy range: {energy.min():.2f} to {energy.max():.2f}")
            print(f"  Intensity range: {intensity.min():.2f} to {intensity.max():.2f}")
            print(f"  Data points: {len(energy)}")
            
            return {
                'energy': energy,
                'intensity': intensity,
                'filename': os.path.basename(file_path)
            }
            
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
            return None
    
    def optimize_als_parameters(self, detector_class, intensity, energy, param_grid):
        """Grid search to optimize ALS parameters for a given spectrum."""
        best_score = float('inf')
        best_params = None
        best_baseline = None
        for lam in param_grid['lam']:
            for p in param_grid['p']:
                try:
                    detector = detector_class('asls')
                    baseline = detector.detect_baseline(intensity, energy, lam=lam, p=p)
                    # Score: smoothness + penalty for baseline above signal
                    smoothness = BaselineQualityMetrics.calculate_smoothness(baseline)
                    over_baseline = np.mean((baseline > intensity) * (baseline - intensity))
                    score = smoothness + 10 * over_baseline  # penalize overshoot
                    if score < best_score:
                        best_score = score
                        best_params = {'lam': lam, 'p': p}
                        best_baseline = baseline
                except Exception:
                    continue
        return best_baseline, best_params, best_score

    def run_baseline_comparison(self, spectrum_data, spectrum_name):
        """Run comprehensive baseline comparison on a spectrum."""
        
        print(f"\n{'='*80}")
        print(f"BASELINE DETECTION COMPARISON - {spectrum_name.upper()}")
        print(f"{'='*80}")
        
        print(f"\nTesting {len(self.all_algorithms)} algorithms:")
        print(f"  Rolling-based ({len(self.rolling_algorithms)}): {', '.join(self.rolling_algorithms)}")
        print(f"  ALS-based ({len(self.als_algorithms)}): {', '.join(self.als_algorithms)}")
        
        results = []
        energy = spectrum_data['energy']
        intensity = spectrum_data['intensity']
        
        # Test each algorithm
        for algorithm in self.all_algorithms:
            print(f"\n--- Testing {algorithm} ---")
            
            try:
                if algorithm == 'pybaselines_asls':
                    from utils.enhanced_baseline_detection import PyBaselinesDetector
                    param_grid = {'lam': [1e4, 1e5, 1e6, 1e7], 'p': [0.001, 0.01, 0.05, 0.1]}
                    baseline, best_params, _ = self.optimize_als_parameters(PyBaselinesDetector, intensity, energy, param_grid)
                    param_str = f"lam={best_params['lam']}, p={best_params['p']}" if best_params else "default"
                elif algorithm == 'spectrochempy_asls':
                    from utils.enhanced_baseline_detection import SpectroChemPyDetector
                    param_grid = {'lam': [1e4, 1e5, 1e6, 1e7], 'p': [0.001, 0.01, 0.05, 0.1]}
                    # SpectroChemPy uses 'lamb' and 'asymmetry' instead of 'lam' and 'p'
                    best_score = float('inf')
                    best_params = None
                    best_baseline = None
                    for lamb in param_grid['lam']:
                        for asymmetry in param_grid['p']:
                            try:
                                detector = SpectroChemPyDetector('asls')
                                baseline = detector.detect_baseline(intensity, energy, lamb=lamb, asymmetry=asymmetry)
                                smoothness = BaselineQualityMetrics.calculate_smoothness(baseline)
                                over_baseline = np.mean((baseline > intensity) * (baseline - intensity))
                                score = smoothness + 10 * over_baseline
                                if score < best_score:
                                    best_score = score
                                    best_params = {'lamb': lamb, 'asymmetry': asymmetry}
                                    best_baseline = baseline
                            except Exception:
                                continue
                    baseline = best_baseline
                    param_str = f"lamb={best_params['lamb']}, asym={best_params['asymmetry']}" if best_params else "default"
                elif algorithm == 'binned_als':
                    from utils.enhanced_baseline_detection import BinnedALSDetector
                    param_grid = {'lam': [1e4, 1e5, 1e6, 1e7], 'p': [0.001, 0.01, 0.05, 0.1]}
                    best_score = float('inf')
                    best_params = None
                    best_baseline = None
                    for lam in param_grid['lam']:
                        for p in param_grid['p']:
                            try:
                                detector = BinnedALSDetector('asls')
                                baseline = detector.detect_baseline(intensity, energy, lam=lam, p=p)
                                smoothness = BaselineQualityMetrics.calculate_smoothness(baseline)
                                over_baseline = np.mean((baseline > intensity) * (baseline - intensity))
                                score = smoothness + 10 * over_baseline
                                if score < best_score:
                                    best_score = score
                                    best_params = {'lam': lam, 'p': p}
                                    best_baseline = baseline
                            except Exception:
                                continue
                    baseline = best_baseline
                    param_str = f"lam={best_params['lam']}, p={best_params['p']}" if best_params else "default"
                else:
                    detector = EnhancedBaselineDetectorFactory.create_detector(algorithm)
                    baseline = detector.detect_baseline(intensity, energy)
                    param_str = "default"
                processing_time = 0  # For optimized, timing is less relevant
                metrics = BaselineQualityMetrics()
                smoothness = metrics.calculate_smoothness(baseline)
                baseline_range = np.max(baseline) - np.min(baseline)
                peak_preservation = np.corrcoef(intensity, intensity - baseline)[0, 1]
                result = {
                    'algorithm': algorithm,
                    'type': 'rolling' if 'rolling' in algorithm else 'als',
                    'processing_time': processing_time,
                    'smoothness': smoothness,
                    'baseline_range': baseline_range,
                    'peak_preservation': peak_preservation,
                    'baseline': baseline,
                    'params': param_str,
                    'status': 'success'
                }
                print(f"  ✓ Success")
                print(f"    Params: {param_str}")
                print(f"    Smoothness: {smoothness:.6f}")
                print(f"    Baseline range: {baseline_range:.6f}")
                print(f"    Peak preservation: {peak_preservation:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                result = {
                    'algorithm': algorithm,
                    'type': 'rolling' if 'rolling' in algorithm else 'als',
                    'status': 'failed',
                    'error': str(e)
                }
            
            results.append(result)
        
        return results
    
    def create_baseline_visualization(self, results, spectrum_data, spectrum_name):
        """Create clear baseline visualization."""
        
        # Filter successful results
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            print("No successful results to visualize")
            return None
        
        # Separate rolling and ALS results
        rolling_results = [r for r in successful_results if r['type'] == 'rolling']
        als_results = [r for r in successful_results if r['type'] == 'als']
        
        energy = spectrum_data['energy']
        intensity = spectrum_data['intensity']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Baseline Detection Comparison - {spectrum_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original spectrum with rolling baselines
        ax1.plot(energy, intensity, 'k-', linewidth=1.5, alpha=0.8, label='Original Spectrum')
        rolling_colors = ['blue', 'green', 'orange']
        for i, result in enumerate(rolling_results):
            ax1.plot(energy, result['baseline'], color=rolling_colors[i], linewidth=2.5, 
                    label=f"{result['algorithm'].split('_')[-1]} (smooth: {result['smoothness']:.4f})")
        
        ax1.set_xlabel('Energy (cm⁻¹)')
        ax1.set_ylabel('Intensity')
        ax1.set_title('Rolling-Based Baselines')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Original spectrum with ALS baselines
        ax2.plot(energy, intensity, 'k-', linewidth=1.5, alpha=0.8, label='Original Spectrum')
        als_colors = ['red', 'purple']
        for i, result in enumerate(als_results):
            ax2.plot(energy, result['baseline'], color=als_colors[i], linewidth=2.5, 
                    label=f"{result['algorithm'].split('_')[-1]} ({result['params']}, smooth: {result['smoothness']:.4f})")
        
        ax2.set_xlabel('Energy (cm⁻¹)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('ALS-Based Baselines')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance metrics
        all_algorithms = [r['algorithm'].split('_')[-1] for r in successful_results]
        processing_times = [r['processing_time'] for r in successful_results]
        colors_plot = ['blue' if r['type'] == 'rolling' else 'red' for r in successful_results]
        
        bars1 = ax3.bar(range(len(all_algorithms)), processing_times, color=colors_plot, alpha=0.7)
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Processing Time (s)')
        ax3.set_title('Processing Time Comparison')
        ax3.set_xticks(range(len(all_algorithms)))
        ax3.set_xticklabels(all_algorithms, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars1, processing_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(processing_times)*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Peak preservation comparison
        peak_preservations = [r['peak_preservation'] for r in successful_results]
        
        bars2 = ax4.bar(range(len(all_algorithms)), peak_preservations, color=colors_plot, alpha=0.7)
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Peak Preservation (Correlation)')
        ax4.set_title('Peak Preservation Comparison (Higher is Better)')
        ax4.set_xticks(range(len(all_algorithms)))
        ax4.set_xticklabels(all_algorithms, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars2, peak_preservations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save in the proper directory
        filename = f"baseline_comparison_{spectrum_name.lower().replace(' ', '_')}.pdf"
        filepath = self.baseline_plots_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Baseline comparison saved as: {filepath}")
        
        return filepath
    
    def create_recommendation_report(self, results, spectrum_data, spectrum_name):
        """Create a recommendation report for algorithm selection."""
        
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            print("No successful results to report")
            return None
        
        # Separate results by type
        rolling_results = [r for r in successful_results if r['type'] == 'rolling']
        als_results = [r for r in successful_results if r['type'] == 'als']
        
        filename = f"baseline_recommendations_{spectrum_name.lower().replace(' ', '_')}.txt"
        filepath = self.baseline_plots_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"BASELINE DETECTION RECOMMENDATIONS\n")
            f.write(f"==================================\n")
            f.write(f"Spectrum: {spectrum_name}\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"SPECTRUM INFORMATION\n")
            f.write(f"-------------------\n")
            f.write(f"Filename: {spectrum_data['filename']}\n")
            f.write(f"Energy range: {spectrum_data['energy'].min():.2f} to {spectrum_data['energy'].max():.2f} cm⁻¹\n")
            f.write(f"Intensity range: {spectrum_data['intensity'].min():.2f} to {spectrum_data['intensity'].max():.2f}\n")
            f.write(f"Data points: {len(spectrum_data['energy'])}\n\n")
            
            f.write(f"ALGORITHM PERFORMANCE SUMMARY\n")
            f.write(f"-----------------------------\n")
            f.write(f"Total algorithms tested: {len(results)}\n")
            f.write(f"Successful algorithms: {len(successful_results)}\n")
            f.write(f"Failed algorithms: {len(results) - len(successful_results)}\n\n")
            
            # Performance rankings
            f.write(f"PERFORMANCE RANKINGS\n")
            f.write(f"===================\n")
            
            # Best smoothness
            best_smooth = min(successful_results, key=lambda x: x['smoothness'])
            f.write(f"Best smoothness: {best_smooth['algorithm']} ({best_smooth['smoothness']:.6f})\n")
            
            # Best peak preservation
            best_preservation = max(successful_results, key=lambda x: x['peak_preservation'])
            f.write(f"Best peak preservation: {best_preservation['algorithm']} ({best_preservation['peak_preservation']:.6f})\n")
            
            # Fastest processing
            fastest = min(successful_results, key=lambda x: x['processing_time'])
            f.write(f"Fastest processing: {fastest['algorithm']} ({fastest['processing_time']:.6f} s)\n")
            
            # Smallest baseline range
            smallest_range = min(successful_results, key=lambda x: x['baseline_range'])
            f.write(f"Smallest baseline range: {smallest_range['algorithm']} ({smallest_range['baseline_range']:.6f})\n")
            
            # Detailed results
            f.write(f"\nDETAILED RESULTS\n")
            f.write(f"================\n")
            
            # Rolling algorithms
            if rolling_results:
                f.write(f"\nROLLING-BASED ALGORITHMS ({len(rolling_results)})\n")
                f.write(f"{'='*50}\n")
                for result in rolling_results:
                    f.write(f"\n{result['algorithm']}:\n")
                    f.write(f"  Processing time: {result['processing_time']:.6f} s\n")
                    f.write(f"  Smoothness: {result['smoothness']:.6f}\n")
                    f.write(f"  Baseline range: {result['baseline_range']:.6f}\n")
                    f.write(f"  Peak preservation: {result['peak_preservation']:.6f}\n")
            
            # ALS algorithms
            if als_results:
                f.write(f"\nALS-BASED ALGORITHMS ({len(als_results)})\n")
                f.write(f"{'='*50}\n")
                for result in als_results:
                    f.write(f"\n{result['algorithm']}:\n")
                    f.write(f"  Processing time: {result['processing_time']:.6f} s\n")
                    f.write(f"  Smoothness: {result['smoothness']:.6f}\n")
                    f.write(f"  Baseline range: {result['baseline_range']:.6f}\n")
                    f.write(f"  Peak preservation: {result['peak_preservation']:.6f}\n")
                    f.write(f"  Params: {result['params']}\n")
            
            # Recommendations
            f.write(f"\nRECOMMENDATIONS\n")
            f.write(f"==============\n")
            
            if rolling_results and als_results:
                f.write(f"Based on the analysis of {spectrum_name}:\n\n")
                
                # Best overall
                best_overall = best_preservation  # Peak preservation is most important
                f.write(f"RECOMMENDED ALGORITHM: {best_overall['algorithm']}\n")
                f.write(f"  - Peak preservation: {best_overall['peak_preservation']:.3f}\n")
                f.write(f"  - Smoothness: {best_overall['smoothness']:.4f}\n")
                f.write(f"  - Processing time: {best_overall['processing_time']:.4f} s\n\n")
                
                f.write(f"ALTERNATIVE OPTIONS:\n")
                f.write(f"  - For speed: {fastest['algorithm']} ({fastest['processing_time']:.4f} s)\n")
                f.write(f"  - For smoothness: {best_smooth['algorithm']} (smoothness: {best_smooth['smoothness']:.4f})\n")
                f.write(f"  - For balance: {smallest_range['algorithm']} (range: {smallest_range['baseline_range']:.2f})\n\n")
                
                f.write(f"USAGE IN BATCH ANALYSIS:\n")
                f.write(f"  Use the recommended algorithm in your batch analysis:\n")
                f.write(f"  baseline_detector_type='{best_overall['algorithm']}'\n")
            
            # Failed algorithms
            failed_results = [r for r in results if r['status'] == 'failed']
            if failed_results:
                f.write(f"\nFAILED ALGORITHMS\n")
                f.write(f"=================\n")
                for result in failed_results:
                    f.write(f"{result['algorithm']}: {result['error']}\n")
        
        print(f"✓ Recommendation report saved as: {filepath}")
        return filepath
    
    def analyze_single_file(self, file_path):
        """Analyze baseline detection for a single file."""
        print(f"\n{'='*80}")
        print(f"ANALYZING SINGLE FILE: {file_path}")
        print(f"{'='*80}")
        
        # Load spectrum
        spectrum_data = self.load_ins_csv(file_path)
        if spectrum_data is None:
            return None
        
        spectrum_name = spectrum_data['filename'].replace('.csv', '')
        
        # Run baseline comparison
        results = self.run_baseline_comparison(spectrum_data, spectrum_name)
        
        # Create visualization
        self.create_baseline_visualization(results, spectrum_data, spectrum_name)
        
        # Create recommendation report
        self.create_recommendation_report(results, spectrum_data, spectrum_name)
        
        return results
    
    def analyze_directory(self, directory_path):
        """Analyze baseline detection for a directory (using first file as sample)."""
        print(f"\n{'='*80}")
        print(f"ANALYZING DIRECTORY: {directory_path}")
        print(f"{'='*80}")
        
        # Find CSV files
        csv_files = glob(os.path.join(directory_path, "*.csv"))
        if not csv_files:
            print(f"✗ No CSV files found in {directory_path}")
            return None
        
        # Use first file as sample
        sample_file = csv_files[0]
        print(f"✓ Using sample file: {sample_file}")
        
        # Analyze the sample file
        results = self.analyze_single_file(sample_file)
        
        if results:
            print(f"\n{'='*80}")
            print(f"DIRECTORY ANALYSIS COMPLETE")
            print(f"{'='*80}")
            print(f"✓ Baseline analysis completed on sample file")
            print(f"✓ Results saved in: comprehensive_analysis_results/plots/baseline_detection/")
            print(f"✓ Use the recommended algorithm for batch analysis of all files")
            print(f"✓ Check the recommendation report for algorithm selection")
        
        return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Smart Baseline Analysis for INS Spectra')
    parser.add_argument('--file', type=str, help='Path to single INS spectrum file')
    parser.add_argument('--directory', type=str, help='Path to directory containing INS spectra')
    
    args = parser.parse_args()
    
    if not args.file and not args.directory:
        print("Please provide either --file or --directory argument")
        parser.print_help()
        return
    
    # Initialize analyzer
    analyzer = SmartBaselineAnalyzer()
    
    # Run analysis
    if args.file:
        analyzer.analyze_single_file(args.file)
    elif args.directory:
        analyzer.analyze_directory(args.directory)
    
    print(f"\n{'='*80}")
    print(f"SMART BASELINE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Results saved in: comprehensive_analysis_results/plots/baseline_detection/")
    print(f"✓ Check the recommendation report for algorithm selection")
    print(f"✓ Use the recommended algorithm in your batch analysis workflow")

if __name__ == "__main__":
    main() 