#!/usr/bin/env python3
"""
Test script for ML Peak Analyzer
================================

This script tests the MLPeakAnalyzer with existing data files.
"""

import os
import sys
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml_peak_analyzer import MLPeakAnalyzer

def test_with_existing_data():
    """Test the analyzer with existing peak data files."""
    
    print("Testing ML Peak Analyzer with existing data...")
    
    # Find existing peak files
    peak_files = [f for f in os.listdir('..') if f.startswith('filtered_peaks_') and f.endswith('_data.csv')]
    
    if not peak_files:
        print("No existing peak files found. Creating test data...")
        create_test_data()
        peak_files = ["test_peaks.csv"]
    
    # Test with first available file
    test_file = peak_files[0]
    print(f"Testing with file: {test_file}")
    
    # Load peak data
    peaks_data = pd.read_csv(f"../{test_file}")
    print(f"Loaded {len(peaks_data)} peaks from {test_file}")
    
    # Create simulated spectrum data for testing
    create_test_spectrum()
    
    # Initialize analyzer
    analyzer = MLPeakAnalyzer(energy_range=(0, 3500))
    
    # Load data
    analyzer.load_spectrum_data("test_spectrum.csv", skiprows=0)
    analyzer.load_peak_data(f"../{test_file}")
    
    # Perform fitting
    print("Performing Gaussian fitting...")
    fit_results = analyzer.fit_global_gaussians(smoothing=True)
    
    if fit_results is None:
        print("✗ Fitting failed!")
        return False
    
    print(f"✓ Fitting successful! R² = {fit_results['r_squared']:.4f}")
    
    # Extract features
    print("Extracting ML features...")
    features = analyzer.extract_ml_features()
    
    print(f"✓ Extracted {len(features)} features")
    print(f"  - Number of peaks: {features['num_peaks']}")
    print(f"  - Mean FWHM: {features['mean_fwhm']:.2f} cm⁻¹")
    print(f"  - Total area: {features['total_area']:.2f}")
    
    # Create test plot
    print("Creating test plot...")
    analyzer.plot_publication_quality(save_path="test_analysis.pdf")
    
    # Save features
    analyzer.save_features_to_csv("test_features.csv")
    
    print("✓ All tests passed!")
    return True

def create_test_data():
    """Create test peak data."""
    np.random.seed(42)
    
    # Create realistic peak data
    num_peaks = 10
    peak_positions = np.sort(np.random.uniform(100, 3000, num_peaks))
    peak_intensities = np.random.uniform(0.5, 2.5, num_peaks)
    
    peak_data = pd.DataFrame({
        'Peak Position (Energy)': peak_positions,
        'Peak Intensity': peak_intensities
    })
    
    peak_data.to_csv("../test_peaks.csv", index=False)
    print("✓ Created test peak data")

def create_test_spectrum():
    """Create test spectrum data."""
    # Create energy axis
    energy = np.linspace(0, 3500, 3501)
    
    # Create spectrum with multiple peaks
    np.random.seed(42)
    
    # Define test peaks
    peak_params = [
        (1.5, 300, 20),
        (2.0, 600, 25),
        (1.8, 900, 18),
        (2.2, 1200, 30),
        (1.6, 1500, 22),
        (2.5, 1800, 28),
        (1.4, 2100, 16),
        (1.9, 2400, 24),
        (1.2, 2700, 14),
        (2.1, 3000, 26),
    ]
    
    # Generate spectrum
    intensity = np.zeros_like(energy)
    baseline = 0.1
    
    for amp, center, sigma in peak_params:
        intensity += amp * np.exp(-((energy - center)**2) / (2 * sigma**2))
    
    # Add baseline and noise
    intensity += baseline
    intensity += np.random.normal(0, 0.05, len(energy))
    
    # Create spectrum data
    spectrum_data = pd.DataFrame({
        'Energy': energy,
        'Intensity': intensity
    })
    
    spectrum_data.to_csv("test_spectrum.csv", index=False)
    print("✓ Created test spectrum data")

def main():
    """Main test function."""
    print("="*50)
    print("ML PEAK ANALYZER TEST SUITE")
    print("="*50)
    
    try:
        success = test_with_existing_data()
        
        if success:
            print("\n" + "="*50)
            print("✓ ALL TESTS PASSED!")
            print("="*50)
            print("\nThe ML Peak Analyzer is working correctly.")
            print("You can now use it with your actual data files.")
        else:
            print("\n" + "="*50)
            print("✗ SOME TESTS FAILED!")
            print("="*50)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 