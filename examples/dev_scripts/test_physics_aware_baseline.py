#!/usr/bin/env python3
"""
Test script for Physics-Aware ALS Baseline Detection
====================================================

This script tests the physics-aware ALS baseline detection that accounts for
the different nature of fundamentals (0→1) vs overtones (0→2, 0→3, etc.) in INS spectra.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.baseline_detection import (
    PhysicsAwareALSBaselineDetector, 
    BinnedALSBaselineDetector, 
    AsymmetricLeastSquaresBaselineDetector
)

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

def test_physics_aware_baseline():
    """Test physics-aware baseline detection on INS spectra."""
    
    print("Testing Physics-Aware ALS Baseline Detection...")
    
    # Test data paths
    test_files = [
        "/Users/toulikmaitra/Documents/UC Davis/1. Projects/Molecule Analysis/24- Structural Entropy/3- Anthracene/INS_spectra_all/xy_experimental_500_2.00.csv",
        "/Users/toulikmaitra/Documents/UC Davis/1. Projects/Molecule Analysis/3-Peak and Background detection/Experimental_INS_files/Normalised_Data/Anthracene_INS_Experimental.csv"
    ]
    
    # Create output directory
    output_dir = Path("comprehensive_analysis_results/plots/baseline_detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detectors
    physics_aware = PhysicsAwareALSBaselineDetector(
        fundamental_region=(0, 500),     # 0→1 transitions dominate
        overtone_region=(500, 3500),     # 0→2, 0→3, etc. dominate
        fundamental_lambda=1e5,          # Less smooth for fundamentals
        fundamental_p=0.05,              # More asymmetric for fundamentals
        overtone_lambda=1e6,             # Smoother for overtones
        overtone_p=0.01,                 # Less asymmetric for overtones
        max_iterations=10,
        blend_width=20
    )
    
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
            print("  Fitting physics-aware ALS baseline...")
            physics_baseline = physics_aware.detect_baseline(intensity, energy)
            
            print("  Fitting binned ALS baseline...")
            binned_baseline = binned_als.detect_baseline(intensity, energy)
            
            print("  Fitting regular ALS baseline...")
            regular_baseline = regular_als.detect_baseline(intensity, energy)
            
            # Calculate corrected spectra
            physics_corrected = intensity - physics_baseline
            binned_corrected = intensity - binned_baseline
            regular_corrected = intensity - regular_baseline
            
            # Store results
            results[file_path] = {
                'energy': energy,
                'intensity': intensity,
                'physics_baseline': physics_baseline,
                'binned_baseline': binned_baseline,
                'regular_baseline': regular_baseline,
                'physics_corrected': physics_corrected,
                'binned_corrected': binned_corrected,
                'regular_corrected': regular_corrected
            }
            
            print(f"  ✓ Successfully processed {len(energy)} data points")
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path}: {e}")
            continue
    
    # Create comparison plots
    if results:
        create_physics_comparison_plots(results, output_dir)
        print(f"\n✓ Results saved to {output_dir}")
    else:
        print("\n✗ No results to plot")

def create_physics_comparison_plots(results, output_dir):
    """Create comparison plots for physics-aware vs other methods."""
    
    fig, axes = plt.subplots(len(results), 3, figsize=(18, 5*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (file_path, data) in enumerate(results.items()):
        energy = data['energy']
        intensity = data['intensity']
        physics_baseline = data['physics_baseline']
        binned_baseline = data['binned_baseline']
        regular_baseline = data['regular_baseline']
        physics_corrected = data['physics_corrected']
        binned_corrected = data['binned_corrected']
        regular_corrected = data['regular_corrected']
        
        # Plot 1: Original spectrum with all baselines
        ax1 = axes[i, 0]
        ax1.plot(energy, intensity, 'b-', label='Original', alpha=0.7, linewidth=1)
        ax1.plot(energy, physics_baseline, 'r-', label='Physics-Aware ALS', linewidth=2)
        ax1.plot(energy, binned_baseline, 'g--', label='Binned ALS', linewidth=2)
        ax1.plot(energy, regular_baseline, 'orange', label='Regular ALS', linewidth=2)
        
        # Add region boundaries
        ax1.axvline(x=500, color='purple', linestyle=':', alpha=0.7, label='Fundamental/Overtone boundary')
        ax1.axvspan(0, 500, alpha=0.1, color='blue', label='Fundamentals (0→1)')
        ax1.axvspan(500, 3500, alpha=0.1, color='red', label='Higher Order (0→2, 0→3, etc.)')
        
        ax1.set_xlabel('Energy (cm⁻¹)')
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'Baseline Comparison - {Path(file_path).name}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Corrected spectra
        ax2 = axes[i, 1]
        ax2.plot(energy, physics_corrected, 'r-', label='Physics-Aware Corrected', linewidth=1)
        ax2.plot(energy, binned_corrected, 'g-', label='Binned ALS Corrected', linewidth=1)
        ax2.plot(energy, regular_corrected, 'orange', label='Regular ALS Corrected', linewidth=1)
        ax2.set_xlabel('Energy (cm⁻¹)')
        ax2.set_ylabel('Corrected Intensity')
        ax2.set_title(f'Corrected Spectra - {Path(file_path).name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Baseline differences
        ax3 = axes[i, 2]
        ax3.plot(energy, physics_baseline - regular_baseline, 'r-', label='Physics - Regular', linewidth=2)
        ax3.plot(energy, binned_baseline - regular_baseline, 'g-', label='Binned - Regular', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axvline(x=500, color='purple', linestyle=':', alpha=0.7)
        ax3.set_xlabel('Energy (cm⁻¹)')
        ax3.set_ylabel('Baseline Difference')
        ax3.set_title(f'Baseline Differences - {Path(file_path).name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'physics_aware_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed physics-aware plot for first result
    if results:
        first_file = list(results.keys())[0]
        data = results[first_file]
        create_detailed_physics_plot(data, first_file, output_dir)

def create_detailed_physics_plot(data, file_path, output_dir):
    """Create detailed plot showing physics-aware baseline detection."""
    
    energy = data['energy']
    intensity = data['intensity']
    physics_baseline = data['physics_baseline']
    regular_baseline = data['regular_baseline']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Original with physics-aware baseline and regions
    ax1.plot(energy, intensity, 'b-', label='Original INS Spectrum', alpha=0.7, linewidth=1)
    ax1.plot(energy, physics_baseline, 'r-', label='Physics-Aware Baseline', linewidth=2)
    ax1.plot(energy, regular_baseline, 'g--', label='Regular ALS Baseline', linewidth=2)
    
    # Add region boundaries and shading
    ax1.axvline(x=500, color='purple', linestyle=':', alpha=0.7, linewidth=2, label='Fundamental/Higher Order Boundary')
    ax1.axvspan(0, 500, alpha=0.1, color='blue', label='Fundamentals (0→1 transitions)')
    ax1.axvspan(500, 3500, alpha=0.1, color='red', label='Higher Order (0→2, 0→3, etc.)')
    
    ax1.set_xlabel('Energy (cm⁻¹)')
    ax1.set_ylabel('Intensity')
    ax1.set_title(f'Physics-Aware Baseline Detection for INS Spectrum\n{Path(file_path).name}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference between physics-aware and regular ALS
    difference = physics_baseline - regular_baseline
    ax2.plot(energy, difference, 'purple', linewidth=2, label='Physics-Aware - Regular ALS')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(x=500, color='purple', linestyle=':', alpha=0.7, linewidth=2)
    ax2.axvspan(0, 500, alpha=0.1, color='blue')
    ax2.axvspan(500, 3500, alpha=0.1, color='red')
    
    ax2.set_xlabel('Energy (cm⁻¹)')
    ax2.set_ylabel('Baseline Difference')
    ax2.set_title('Physics-Aware vs Regular ALS Baseline Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'physics_aware_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    test_physics_aware_baseline() 