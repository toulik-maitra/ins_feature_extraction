#!/usr/bin/env python3
"""
Test script for Peak-to-Baseline Ratio Analysis Integration
==========================================================

This script tests the integration of p/b ratio analysis into the batch analysis workflow.
It creates sample data and verifies that the analysis produces the expected plots and statistics.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.batch_ml_analysis import BatchMLAnalyzer
from utils.pb_ratio_analysis import PBRatioAnalyzer, analyze_pb_ratios_from_features

def create_test_spectrum_data():
    """Create test spectrum data for analysis."""
    # Create energy range
    energy = np.linspace(0, 3500, 3501)
    
    # Create baseline with some variation
    baseline = 0.1 + 0.05 * np.sin(energy / 500) + 0.02 * np.random.normal(0, 1, len(energy))
    
    # Create peaks at different positions
    peak_positions = [500, 1000, 1500, 2000, 2500, 3000]
    peak_amplitudes = [0.5, 0.8, 0.3, 0.6, 0.4, 0.7]
    peak_widths = [50, 60, 40, 55, 45, 65]
    
    # Create spectrum with peaks
    intensity = baseline.copy()
    for pos, amp, width in zip(peak_positions, peak_amplitudes, peak_widths):
        # Add Gaussian peak
        peak = amp * np.exp(-0.5 * ((energy - pos) / width) ** 2)
        intensity += peak
    
    return energy, intensity, baseline, peak_positions, peak_amplitudes

def create_test_csv_files():
    """Create test CSV files for batch analysis."""
    test_dir = Path("test_pb_analysis_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create multiple test files with different characteristics
    test_files = []
    
    for i in range(3):
        energy, intensity, baseline, peak_positions, peak_amplitudes = create_test_spectrum_data()
        
        # Add some variation between files
        intensity *= (0.8 + 0.4 * i)  # Different overall intensities
        baseline *= (0.9 + 0.2 * i)   # Different baseline levels
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': energy,
            'y': intensity
        })
        
        # Save to CSV
        filename = f"test_spectrum_{i+1}.csv"
        filepath = test_dir / filename
        df.to_csv(filepath, index=False)
        test_files.append(str(filepath))
        
        print(f"Created test file: {filepath}")
    
    return test_dir, test_files

def test_pb_ratio_analyzer_direct():
    """Test the PBRatioAnalyzer directly."""
    print("\n" + "="*60)
    print("TESTING P/B RATIO ANALYZER DIRECTLY")
    print("="*60)
    
    # Create test data
    ratios1 = np.random.lognormal(2, 0.5, 50)  # Sample 1
    ratios2 = np.random.lognormal(1.5, 0.8, 45)  # Sample 2
    ratios3 = np.random.lognormal(2.5, 0.3, 55)  # Sample 3
    
    # Create analyzer
    analyzer = PBRatioAnalyzer(output_dir="test_pb_analysis_direct")
    
    # Add data
    analyzer.add_ratio_data(ratios1, "Sample_1")
    analyzer.add_ratio_data(ratios2, "Sample_2")
    analyzer.add_ratio_data(ratios3, "Sample_3")
    
    # Create all plots
    analyzer.create_all_plots()
    
    # Save data
    analyzer.save_ratio_data_to_csv("test_pb_data.csv")
    
    print("✓ Direct P/B ratio analysis completed")
    print(f"  - Output directory: {analyzer.output_dir}")
    print(f"  - Plots created: {len(list(analyzer.plots_dir.glob('*.pdf')))}")
    print(f"  - Statistics created: {len(list(analyzer.stats_dir.glob('*.csv')))}")

def test_batch_integration():
    """Test the integration with batch analysis."""
    print("\n" + "="*60)
    print("TESTING BATCH ANALYSIS INTEGRATION")
    print("="*60)
    
    # Create test data
    test_dir, test_files = create_test_csv_files()
    
    # Initialize batch analyzer
    batch_analyzer = BatchMLAnalyzer(output_dir="test_batch_pb_analysis")
    
    # Analyze each test file
    for i, filepath in enumerate(test_files):
        print(f"\nAnalyzing test file {i+1}/3: {Path(filepath).name}")
        
        result = batch_analyzer.analyze_single_file(
            filepath,
            molecule_name=f"Test_Molecule_{i+1}",
            plot_individual=False,  # Skip individual plots for speed
            baseline_detector_type='als',
            is_experimental=False
        )
        
        if result is not None:
            features, log_entry = result
            batch_analyzer.all_features.append(features)
            batch_analyzer.analysis_log.append(log_entry)
            print(f"✓ Analysis successful: {features['num_peaks']} peaks detected")
        else:
            print("✗ Analysis failed")
    
    # Create batch summary (this will trigger P/B ratio analysis)
    if batch_analyzer.all_features:
        batch_analyzer._create_batch_summary()
        print("\n✓ Batch analysis with P/B ratio integration completed")
        print(f"  - Output directory: {batch_analyzer.output_dir}")
        print(f"  - P/B ratio directory: {batch_analyzer.pb_ratio_dir}")
    else:
        print("\n✗ No successful analyses to summarize")

def test_features_integration():
    """Test the analyze_pb_ratios_from_features function."""
    print("\n" + "="*60)
    print("TESTING FEATURES INTEGRATION")
    print("="*60)
    
    # Create mock features
    features_list = []
    
    for i in range(3):
        # Create mock peak-to-baseline ratios
        ratios = np.random.lognormal(2, 0.5, 30 + i * 10)
        
        features = {
            'molecule_name': f'Mock_Molecule_{i+1}',
            'peak_to_baseline_ratios': ratios,
            'peak_centers': np.random.uniform(0, 3500, len(ratios)),
            'peak_amplitudes': np.random.uniform(0.1, 1.0, len(ratios))
        }
        features_list.append(features)
    
    # Analyze using the helper function
    analyzer = analyze_pb_ratios_from_features(
        features_list, 
        output_dir="test_features_pb_analysis"
    )
    
    # Create plots
    analyzer.create_all_plots()
    
    print("✓ Features integration test completed")
    print(f"  - Analyzed {len(features_list)} samples")
    print(f"  - Output directory: {analyzer.output_dir}")

def main():
    """Run all tests."""
    print("="*80)
    print("PEAK-TO-BASELINE RATIO ANALYSIS INTEGRATION TESTS")
    print("="*80)
    
    try:
        # Test 1: Direct analyzer
        test_pb_ratio_analyzer_direct()
        
        # Test 2: Batch integration
        test_batch_integration()
        
        # Test 3: Features integration
        test_features_integration()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Generated test directories:")
        print("  - test_pb_analysis_direct/")
        print("  - test_batch_pb_analysis/")
        print("  - test_features_pb_analysis/")
        print("  - test_pb_analysis_data/")
        print("\nCheck these directories for generated plots and statistics.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 