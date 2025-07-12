#!/usr/bin/env python3
"""
Test Baseline Detection and Peak-to-Baseline Ratio Features
==========================================================

This script demonstrates the new baseline detection capabilities and
peak-to-baseline ratio features in the INS ML Analysis System.

Usage:
    python test_baseline_features.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.ml_peak_analyzer import MLPeakAnalyzer
from utils.baseline_detection import detect_baseline, calculate_peak_to_baseline_ratios

def create_test_spectrum():
    """Create a synthetic test spectrum for demonstration."""
    energy = np.linspace(0, 3500, 1000)
    
    # Create baseline with some variation
    baseline = 0.1 + 0.05 * np.sin(energy / 500) + 0.02 * np.random.normal(0, 1, len(energy))
    
    # Add peaks
    peaks = [
        (500, 0.8, 50),   # (center, amplitude, width)
        (1000, 1.2, 60),
        (1500, 0.6, 40),
        (2000, 1.5, 70),
        (2500, 0.9, 55),
        (3000, 0.4, 30)
    ]
    
    intensity = baseline.copy()
    for center, amp, width in peaks:
        intensity += amp * np.exp(-((energy - center)**2) / (2 * width**2))
    
    return energy, intensity, baseline

def test_baseline_detection():
    """Test different baseline detection methods."""
    print("="*60)
    print("TESTING BASELINE DETECTION METHODS")
    print("="*60)
    
    # Create test spectrum
    energy, intensity, true_baseline = create_test_spectrum()
    
    # Test different baseline detectors
    detectors = ['dynamic_rolling', 'polynomial', 'als', 'morphological']
    
    plt.figure(figsize=(15, 10))
    
    for i, detector_type in enumerate(detectors, 1):
        print(f"\nTesting {detector_type} baseline detector...")
        
        try:
            # Detect baseline
            detected_baseline = detect_baseline(
                intensity=intensity,
                energy=energy,
                detector_type=detector_type,
                is_experimental=False
            )
            
            # Calculate error
            error = np.mean(np.abs(detected_baseline - true_baseline))
            print(f"  ✓ Baseline detected successfully")
            print(f"  ✓ Mean absolute error: {error:.4f}")
            
            # Plot
            plt.subplot(2, 2, i)
            plt.plot(energy, intensity, 'k-', alpha=0.7, label='Spectrum')
            plt.plot(energy, true_baseline, 'g-', linewidth=2, label='True Baseline')
            plt.plot(energy, detected_baseline, 'r--', linewidth=2, label='Detected Baseline')
            plt.xlabel('Energy (cm⁻¹)')
            plt.ylabel('Intensity (a.u.)')
            plt.title(f'{detector_type.replace("_", " ").title()} Detector')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    plt.tight_layout()
    plt.savefig('baseline_detection_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Baseline comparison plot saved: baseline_detection_comparison.pdf")

def test_peak_to_baseline_ratios():
    """Test peak-to-baseline ratio feature extraction."""
    print("\n" + "="*60)
    print("TESTING PEAK-TO-BASELINE RATIO FEATURES")
    print("="*60)
    
    # Create test spectrum
    energy, intensity, true_baseline = create_test_spectrum()
    
    # Detect peaks (simplified)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(intensity, prominence=0.1, distance=50)
    peak_positions = energy[peaks]
    peak_intensities = intensity[peaks]
    
    print(f"✓ Detected {len(peaks)} peaks")
    
    # Detect baseline
    detected_baseline = detect_baseline(
        intensity=intensity,
        energy=energy,
        detector_type='dynamic_rolling',
        is_experimental=False
    )
    
    # Calculate peak-to-baseline ratios
    ratio_features = calculate_peak_to_baseline_ratios(
        intensity=intensity,
        baseline=detected_baseline,
        peak_positions=peaks,
        peak_amplitudes=peak_intensities
    )
    
    print("\nPeak-to-Baseline Ratio Features:")
    print("-" * 40)
    for feature, value in ratio_features.items():
        if isinstance(value, float):
            print(f"{feature:30s}: {value:.4f}")
        else:
            print(f"{feature:30s}: {value}")
    
    # Plot spectrum with peaks and baseline
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(energy, intensity, 'k-', linewidth=1, alpha=0.7, label='Spectrum')
    plt.plot(energy, detected_baseline, 'g-', linewidth=2, label='Detected Baseline')
    plt.plot(peak_positions, peak_intensities, 'ro', markersize=8, label='Detected Peaks')
    plt.xlabel('Energy (cm⁻¹)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Spectrum with Detected Peaks and Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot peak-to-baseline ratios
    plt.subplot(2, 1, 2)
    baseline_at_peaks = detected_baseline[peaks]
    ratios = peak_intensities / (baseline_at_peaks + 1e-10)
    
    plt.bar(range(len(peaks)), ratios, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Peak Index')
    plt.ylabel('Peak-to-Baseline Ratio')
    plt.title('Peak-to-Baseline Ratios')
    plt.grid(True, alpha=0.3)
    
    # Add ratio values as text
    for i, ratio in enumerate(ratios):
        plt.text(i, ratio + 0.1, f'{ratio:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('peak_to_baseline_ratios.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Peak-to-baseline ratio plot saved: peak_to_baseline_ratios.pdf")

def test_integrated_features():
    """Test the integrated feature extraction with baseline detection."""
    print("\n" + "="*60)
    print("TESTING INTEGRATED FEATURE EXTRACTION")
    print("="*60)
    
    # Create test spectrum and save to CSV
    energy, intensity, _ = create_test_spectrum()
    test_data = pd.DataFrame({'x': energy, 'y': intensity})
    test_data.to_csv('test_spectrum.csv', index=False)
    
    print("✓ Created test spectrum: test_spectrum.csv")
    
    # Initialize analyzer with baseline detection
    analyzer = MLPeakAnalyzer(
        energy_range=(0, 3500),
        baseline_detector_type='dynamic_rolling',
        is_experimental=False
    )
    
    # Load spectrum
    analyzer.load_spectrum_data('test_spectrum.csv', skiprows=0, energy_col="x", intensity_col="y")
    
    # Detect peaks
    analyzer.detect_peaks_from_spectrum(prominence=0.1, distance=50)
    
    # Detect baseline
    analyzer.detect_baseline()
    
    # Perform fitting
    fit_results = analyzer.fit_global_gaussians()
    
    if fit_results is not None:
        # Extract features (including peak-to-baseline ratios)
        features = analyzer.extract_ml_features()
        
        print(f"\n✓ Successfully extracted {len(features)} features")
        
        # Show peak-to-baseline ratio features
        print("\nPeak-to-Baseline Ratio Features:")
        print("-" * 40)
        ratio_features = [k for k in features.keys() if 'baseline' in k.lower() or 'ratio' in k.lower()]
        for feature in sorted(ratio_features):
            value = features[feature]
            if isinstance(value, float):
                print(f"{feature:35s}: {value:.4f}")
            else:
                print(f"{feature:35s}: {value}")
        
        # Create publication-quality plot
        analyzer.plot_publication_quality(save_path='integrated_analysis.pdf')
        print(f"\n✓ Integrated analysis plot saved: integrated_analysis.pdf")
        
        # Save features
        analyzer.save_features_to_csv('test_features.csv')
        print(f"✓ Features saved: test_features.csv")
        
    else:
        print("✗ Fitting failed")
    
    # Clean up
    if os.path.exists('test_spectrum.csv'):
        os.remove('test_spectrum.csv')

def main():
    """Main function to run all tests."""
    print("INS ML ANALYSIS SYSTEM - BASELINE FEATURE TEST")
    print("="*60)
    
    try:
        # Test baseline detection methods
        test_baseline_detection()
        
        # Test peak-to-baseline ratio features
        test_peak_to_baseline_ratios()
        
        # Test integrated feature extraction
        test_integrated_features()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- baseline_detection_comparison.pdf")
        print("- peak_to_baseline_ratios.pdf")
        print("- integrated_analysis.pdf")
        print("- test_features.csv")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 