#!/usr/bin/env python3
"""
Test script for the new BinnedALSBaselineDetector
==================================================

This script tests the binned ALS baseline detection on real INS spectra
to verify it works correctly without the internal state issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.baseline_detection import BinnedALSBaselineDetector, AsymmetricLeastSquaresBaselineDetector

def load_spectrum_data(file_path):
    """Load spectrum data from file, robust to delimiter and header issues."""
    import csv
    import itertools
    try:
        # Try to auto-detect delimiter and header
        with open(file_path, 'r') as f:
            # Read first 5 lines for inspection
            preview = list(itertools.islice(f, 5))
        print(f"\nPreview of {file_path}:")
        for line in preview:
            print(line.strip())
        
        # First, try CSV with header and comma delimiter
        try:
            data = pd.read_csv(file_path, delimiter=',', header=0)
            if data.shape[1] >= 2:
                energy = pd.to_numeric(data.iloc[:, 0], errors='coerce').values
                intensity = pd.to_numeric(data.iloc[:, 1], errors='coerce').values
                mask = ~np.isnan(energy) & ~np.isnan(intensity)
                energy = energy[mask]
                intensity = intensity[mask]
                if len(energy) > 1 and energy[0] > energy[-1]:
                    energy = energy[::-1]
                    intensity = intensity[::-1]
                if len(energy) > 0 and len(intensity) > 0:
                    return energy, intensity
        except Exception as e:
            print(f"  Fallback: Could not load as CSV with header: {e}")
        
        # Try common delimiters (fallback)
        for delim in [',', '\t', '\s+']:
            try:
                if delim == '\s+':
                    data = pd.read_csv(file_path, sep=delim, header=None, comment='#')
                else:
                    data = pd.read_csv(file_path, delimiter=delim, header=None, comment='#')
                # If first row is not numeric, skip it
                if not np.issubdtype(data.iloc[0,0], np.number):
                    data = data.iloc[1:,:].reset_index(drop=True)
                if data.shape[1] >= 2:
                    energy = pd.to_numeric(data.iloc[:, 0], errors='coerce').values
                    intensity = pd.to_numeric(data.iloc[:, 1], errors='coerce').values
                    # Remove NaNs
                    mask = ~np.isnan(energy) & ~np.isnan(intensity)
                    energy = energy[mask]
                    intensity = intensity[mask]
                    # Ensure energy is in ascending order
                    if len(energy) > 1 and energy[0] > energy[-1]:
                        energy = energy[::-1]
                        intensity = intensity[::-1]
                    if len(energy) > 0 and len(intensity) > 0:
                        return energy, intensity
            except Exception as e:
                continue
        raise ValueError("File must have at least 2 columns of numeric data (energy, intensity) after trying common delimiters.")
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")

def test_binned_als_detector():
    """Test the binned ALS detector on real INS spectra."""
    
    print("Testing BinnedALSBaselineDetector...")
    
    # Test data paths
    test_files = [
        "/Users/toulikmaitra/Documents/UC Davis/1. Projects/Molecule Analysis/24- Structural Entropy/3- Anthracene/INS_spectra_all/xy_experimental_500_2.00.csv",
        "/Users/toulikmaitra/Documents/UC Davis/1. Projects/Molecule Analysis/3-Peak and Background detection/Experimental_INS_files/Normalised_Data/Anthracene_INS_Experimental.csv"
    ]
    
    # Create output directory
    output_dir = Path("comprehensive_analysis_results/plots/baseline_detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detectors
    binned_als = BinnedALSBaselineDetector(
        bins=[(0, 500), (500, 2000), (2000, 3500)],
        lambda_param=1e6,
        p_param=0.01,
        max_iterations=10,
        blend_width=10
    )
    
    regular_als = AsymmetricLeastSquaresBaselineDetector(
        lambda_param=1e6,
        p_param=0.01,
        max_iterations=10
    )
    
    results = {}
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        print(f"\nProcessing: {file_path}")
        
        try:
            # Load spectrum
            energy, intensity = load_spectrum_data(file_path)
            
            # Detect baselines
            print("  Fitting binned ALS baseline...")
            binned_baseline = binned_als.detect_baseline(intensity, energy)
            
            print("  Fitting regular ALS baseline...")
            regular_baseline = regular_als.detect_baseline(intensity, energy)
            
            # Calculate metrics
            binned_corrected = intensity - binned_baseline
            regular_corrected = intensity - regular_baseline
            
            # Store results
            results[file_path] = {
                'energy': energy,
                'intensity': intensity,
                'binned_baseline': binned_baseline,
                'regular_baseline': regular_baseline,
                'binned_corrected': binned_corrected,
                'regular_corrected': regular_corrected
            }
            
            print(f"  ✓ Successfully processed {len(energy)} data points")
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path}: {e}")
            continue
    
    # Create comparison plots
    if results:
        create_comparison_plots(results, output_dir)
        print(f"\n✓ Results saved to {output_dir}")
    else:
        print("\n✗ No results to plot")

def create_comparison_plots(results, output_dir):
    """Create comparison plots for binned vs regular ALS."""
    
    fig, axes = plt.subplots(len(results), 2, figsize=(15, 5*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (file_path, data) in enumerate(results.items()):
        energy = data['energy']
        intensity = data['intensity']
        binned_baseline = data['binned_baseline']
        regular_baseline = data['regular_baseline']
        binned_corrected = data['binned_corrected']
        regular_corrected = data['regular_corrected']
        
        # Plot 1: Original spectrum with baselines
        ax1 = axes[i, 0]
        ax1.plot(energy, intensity, 'b-', label='Original', alpha=0.7, linewidth=1)
        ax1.plot(energy, binned_baseline, 'r-', label='Binned ALS', linewidth=2)
        ax1.plot(energy, regular_baseline, 'g--', label='Regular ALS', linewidth=2)
        ax1.set_xlabel('Energy (cm⁻¹)')
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'Baseline Comparison - {Path(file_path).name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Corrected spectra
        ax2 = axes[i, 1]
        ax2.plot(energy, binned_corrected, 'r-', label='Binned ALS Corrected', linewidth=1)
        ax2.plot(energy, regular_corrected, 'g-', label='Regular ALS Corrected', linewidth=1)
        ax2.set_xlabel('Energy (cm⁻¹)')
        ax2.set_ylabel('Corrected Intensity')
        ax2.set_title(f'Corrected Spectra - {Path(file_path).name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'binned_als_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed comparison for first result
    if results:
        first_file = list(results.keys())[0]
        data = results[first_file]
        create_detailed_comparison(data, first_file, output_dir)

def create_detailed_comparison(data, file_path, output_dir):
    """Create detailed comparison showing bin boundaries."""
    
    energy = data['energy']
    intensity = data['intensity']
    binned_baseline = data['binned_baseline']
    regular_baseline = data['regular_baseline']
    
    # Define bins for visualization
    bins = [(0, 500), (500, 2000), (2000, 3500)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Original with baselines and bin boundaries
    ax1.plot(energy, intensity, 'b-', label='Original', alpha=0.7, linewidth=1)
    ax1.plot(energy, binned_baseline, 'r-', label='Binned ALS', linewidth=2)
    ax1.plot(energy, regular_baseline, 'g--', label='Regular ALS', linewidth=2)
    
    # Add bin boundaries
    colors = ['orange', 'purple', 'brown']
    for i, ((start, end), color) in enumerate(zip(bins, colors)):
        ax1.axvline(x=start, color=color, linestyle=':', alpha=0.7, label=f'Bin {i+1} start')
        ax1.axvline(x=end, color=color, linestyle=':', alpha=0.7, label=f'Bin {i+1} end')
        ax1.axvspan(start, end, alpha=0.1, color=color)
    
    ax1.set_xlabel('Energy (cm⁻¹)')
    ax1.set_ylabel('Intensity')
    ax1.set_title(f'Binned ALS Baseline Detection - {Path(file_path).name}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference between binned and regular ALS
    difference = binned_baseline - regular_baseline
    ax2.plot(energy, difference, 'purple', linewidth=2, label='Binned - Regular ALS')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add bin boundaries
    for i, ((start, end), color) in enumerate(zip(bins, colors)):
        ax2.axvline(x=start, color=color, linestyle=':', alpha=0.7)
        ax2.axvline(x=end, color=color, linestyle=':', alpha=0.7)
        ax2.axvspan(start, end, alpha=0.1, color=color)
    
    ax2.set_xlabel('Energy (cm⁻¹)')
    ax2.set_ylabel('Baseline Difference')
    ax2.set_title('Difference Between Binned and Regular ALS Baselines')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'binned_als_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    test_binned_als_detector() 