#!/usr/bin/env python3
"""
Demo script for batch analysis of INS spectra.
This script demonstrates the batch workflow on a subset of files.
"""

import os
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.batch_ml_analysis import BatchMLAnalyzer

def run_demo():
    """Run a demo batch analysis on a subset of INS spectra."""
    
    # Path to your INS spectra directory
    spectra_dir = "/Users/toulikmaitra/Documents/UC Davis/1. Projects/Molecule Analysis/24- Structural Entropy/3- Anthracene/INS_spectra_all"
    
    # Check if directory exists
    if not os.path.exists(spectra_dir):
        print(f"✗ Directory not found: {spectra_dir}")
        print("Please update the spectra_dir path in this script.")
        return
    
    # Create a subset of files for demo (first 5 files)
    csv_files = list(Path(spectra_dir).glob("*.csv"))[:5]
    
    if not csv_files:
        print(f"✗ No CSV files found in {spectra_dir}")
        return
    
    print(f"DEMO: Analyzing {len(csv_files)} files from {spectra_dir}")
    print("Files to analyze:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file.name}")
    
    # Initialize batch analyzer
    analyzer = BatchMLAnalyzer(output_dir="demo_results")
    
    # Process each file
    for i, filepath in enumerate(csv_files, 1):
        print(f"\nProcessing file {i}/{len(csv_files)}: {filepath.name}")
        
        # Extract molecule name
        molecule_name = analyzer._extract_molecule_name(filepath.name)
        
        # Analyze file (without individual plots for speed)
        result = analyzer.analyze_single_file(
            filepath, 
            molecule_name=molecule_name,
            plot_individual=False  # Set to True if you want individual plots
        )
        
        if result is not None:
            features, log_entry = result
            analyzer.all_features.append(features)
            analyzer.analysis_log.append(log_entry)
            print(f"  ✓ Success: {features['num_peaks']} peaks, R² = {features['r_squared']:.3f}")
        else:
            print(f"  ✗ Failed")
    
    # Create comprehensive results
    analyzer._create_batch_summary()
    
    print(f"\n✓ Demo completed! Check the 'demo_results' directory for all outputs.")

if __name__ == "__main__":
    run_demo() 