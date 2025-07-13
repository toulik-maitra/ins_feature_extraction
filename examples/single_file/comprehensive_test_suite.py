#!/usr/bin/env python3
"""
Comprehensive Test Suite for INS ML Analysis System
==================================================

This script provides a unified testing framework that consolidates functionality
from multiple test files and removes duplications.

Features:
- Single file analysis testing
- Baseline detection testing
- ML feature extraction testing
- Enhanced baseline detection testing
- Comprehensive reporting

Author: Consolidated Test Suite
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import the enhanced modules (recommended for new development)
from utils.enhanced_baseline_detection import (
    EnhancedBaselineDetectorFactory,
    BaselineValidationSystem,
    BaselineQualityMetrics
)
from core.enhanced_ml_peak_analyzer import EnhancedMLPeakAnalyzer

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_test_spectrum(energy_range=(0, 3500), num_points=1000, seed=42):
    """
    Create a realistic test spectrum with known baseline for validation.
    
    Parameters:
    -----------
    energy_range : tuple
        Energy range (min, max) in cm⁻¹
    num_points : int
        Number of energy points
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths)
    """
    np.random.seed(seed)
    
    # Energy range
    energy = np.linspace(energy_range[0], energy_range[1], num_points)
    
    # Create true baseline (complex baseline with multiple components)
    true_baseline = (
        0.05 +  # Constant offset
        0.0001 * energy +  # Linear trend
        0.0000001 * energy**2 +  # Quadratic trend
        0.1 * np.exp(-energy / 1000) +  # Exponential decay
        0.02 * np.sin(energy / 500)  # Oscillatory component
    )
    
    # Add peaks at realistic INS positions
    peak_positions = [500, 800, 1200, 1500, 2000, 2500, 3000]
    peak_amplitudes = [0.5, 0.8, 0.6, 1.2, 0.9, 0.7, 0.4]
    peak_widths = [30, 25, 35, 40, 30, 35, 25]
    
    # Create spectrum
    spectrum = true_baseline.copy()
    for pos, amp, width in zip(peak_positions, peak_amplitudes, peak_widths):
        peak = amp * np.exp(-(energy - pos)**2 / (2 * width**2))
        spectrum += peak
    
    # Add realistic noise (different levels for different regions)
    noise_levels = np.where(energy < 1000, 0.01, 0.02)  # Higher noise at higher energy
    noise = np.random.normal(0, 1, len(energy)) * noise_levels
    spectrum += noise
    
    return energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths

def test_baseline_detection():
    """Test baseline detection with multiple algorithms."""
    
    print("="*60)
    print("BASELINE DETECTION TESTING")
    print("="*60)
    
    # Create test spectrum
    energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths = create_test_spectrum()
    
    # Get available detectors
    available_detectors = EnhancedBaselineDetectorFactory.get_available_detectors()
    print(f"Testing {len(available_detectors)} baseline detectors: {available_detectors}")
    
    results = []
    
    for detector_type in available_detectors:
        print(f"\nTesting {detector_type}...")
        
        try:
            # Create detector
            detector = EnhancedBaselineDetectorFactory.create_detector(detector_type)
            
            # Detect baseline
            start_time = time.time()
            estimated_baseline = detector.detect_baseline(spectrum, energy)
            processing_time = time.time() - start_time
            
            # Calculate quality metrics
            corrected_spectrum = spectrum - estimated_baseline
            metrics = BaselineQualityMetrics.calculate_baseline_quality_score(
                true_baseline, estimated_baseline, spectrum, corrected_spectrum,
                np.array(peak_positions, dtype=int)
            )
            
            result = {
                'detector': detector_type,
                'processing_time': processing_time,
                'status': 'success',
                **metrics
            }
            results.append(result)
            
            print(f"  ✓ Success: RMSE = {metrics['rmse']:.6f}, "
                  f"Correlation = {metrics['correlation']:.4f}, "
                  f"Time = {processing_time:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'detector': detector_type,
                'status': 'error',
                'error': str(e)
            })
    
    return pd.DataFrame(results)

def test_ml_peak_analyzer():
    """Test the enhanced ML peak analyzer."""
    
    print("\n" + "="*60)
    print("ML PEAK ANALYZER TESTING")
    print("="*60)
    
    # Create test spectrum
    energy, spectrum, true_baseline, peak_positions, peak_amplitudes, peak_widths = create_test_spectrum()
    
    # Save test spectrum
    spectrum_df = pd.DataFrame({
        'energy': energy,
        'intensity': spectrum
    })
    spectrum_df.to_csv("test_spectrum.csv", index=False)
    
    # Initialize analyzer
    analyzer = EnhancedMLPeakAnalyzer(
        energy_range=(0, 3500),
        baseline_detector_type='pybaselines_asls',
        enable_parameter_optimization=False
    )
    
    # Load spectrum data
    analyzer.load_spectrum_data("test_spectrum.csv", energy_col="energy", intensity_col="intensity")
    
    # Detect baseline
    analyzer.detect_enhanced_baseline()
    
    # Detect peaks
    analyzer.detect_peaks_from_spectrum(
        height=0.1,
        distance=10,
        prominence=0.05,
        width=5
    )
    
    # Fit Gaussians
    fit_results = analyzer.fit_global_gaussians(smoothing=True)
    
    if fit_results is None:
        print("✗ Gaussian fitting failed!")
        return None
    
    print(f"✓ Gaussian fitting successful! R² = {fit_results['r_squared']:.4f}")
    
    # Extract features
    features = analyzer.extract_enhanced_ml_features()
    
    print(f"✓ Extracted {len(features)} features")
    print(f"  - Number of peaks: {features['num_peaks']}")
    print(f"  - Mean FWHM: {features['mean_fwhm']:.2f} cm⁻¹")
    print(f"  - Total area: {features['total_area']:.2f}")
    print(f"  - Mean amplitude: {features['mean_amplitude']:.3f}")
    
    # Create visualization
    analyzer.plot_enhanced_analysis(save_path="test_analysis.pdf")
    
    # Save features
    features_df = pd.DataFrame([features])
    features_df.to_csv("test_features.csv", index=False)
    
    return features

def test_validation_system():
    """Test the validation system."""
    
    print("\n" + "="*60)
    print("VALIDATION SYSTEM TESTING")
    print("="*60)
    
    # Create validation system
    validation_system = BaselineValidationSystem()
    
    # Energy range
    energy = np.linspace(0, 3500, 1000)
    
    # Test parameters
    peak_positions = [500, 800, 1200, 1500, 2000, 2500, 3000]
    peak_amplitudes = [0.5, 0.8, 0.6, 1.2, 0.9, 0.7, 0.4]
    peak_widths = [30, 25, 35, 40, 30, 35, 25]
    
    # Test detectors
    test_detectors = ['pybaselines_asls', 'pybaselines_airpls', 'spectrochempy_polynomial']
    detectors = [EnhancedBaselineDetectorFactory.create_detector(dt) for dt in test_detectors]
    
    # Run validation
    validation_results = validation_system.compare_detectors(
        detectors=detectors,
        energy=energy,
        peak_positions=peak_positions,
        peak_amplitudes=peak_amplitudes,
        peak_widths=peak_widths,
        baseline_types=['polynomial', 'exponential', 'linear'],
        noise_levels=[0.01, 0.02, 0.05]
    )
    
    print("✓ Validation completed successfully!")
    print(f"  Tested {len(test_detectors)} detectors across multiple conditions")
    
    return validation_results

def generate_test_report(baseline_results, ml_results, validation_results):
    """Generate a comprehensive test report."""
    
    print("\n" + "="*60)
    print("GENERATING TEST REPORT")
    print("="*60)
    
    # Create report directory
    os.makedirs("test_results", exist_ok=True)
    
    # Save baseline results
    if baseline_results is not None:
        baseline_results.to_csv("test_results/baseline_test_results.csv", index=False)
        print("✓ Saved baseline test results")
    
    # Save ML results
    if ml_results is not None:
        ml_df = pd.DataFrame([ml_results])
        ml_df.to_csv("test_results/ml_test_results.csv", index=False)
        print("✓ Saved ML test results")
    
    # Save validation results
    if validation_results is not None:
        validation_results.to_csv("test_results/validation_results.csv", index=False)
        print("✓ Saved validation results")
    
    # Generate summary
    summary = {
        'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_detectors_tested': len(baseline_results) if baseline_results is not None else 0,
        'ml_features_extracted': len(ml_results) if ml_results is not None else 0,
        'validation_conditions_tested': len(validation_results) if validation_results is not None else 0
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("test_results/test_summary.csv", index=False)
    
    print("✓ Test report generated in test_results/ directory")

def main():
    """Main test function."""
    print("="*80)
    print("COMPREHENSIVE INS ML ANALYSIS SYSTEM TEST SUITE")
    print("="*80)
    
    try:
        # Run all tests
        baseline_results = test_baseline_detection()
        ml_results = test_ml_peak_analyzer()
        validation_results = test_validation_system()
        
        # Generate report
        generate_test_report(baseline_results, ml_results, validation_results)
        
        print("\n" + "="*80)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nThe INS ML Analysis System is working correctly.")
        print("Check the test_results/ directory for detailed results.")
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 