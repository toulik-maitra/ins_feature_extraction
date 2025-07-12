#!/usr/bin/env python3
"""
Create Clean ML Dataset
=======================

This script creates a clean ML-ready dataset from the comprehensive analysis results,
removing unnecessary features like individual peak arrays.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_clean_ml_dataset():
    """Create a clean ML dataset from comprehensive analysis results."""
    
    # Path to the comprehensive results
    results_dir = Path("comprehensive_analysis_results")
    features_dir = results_dir / "features"
    
    # Read the combined features file
    combined_file = features_dir / "all_molecules_features.csv"
    
    if not combined_file.exists():
        print(f"Combined features file not found: {combined_file}")
        return
    
    print(f"Reading combined features from: {combined_file}")
    features_df = pd.read_csv(combined_file)
    
    print(f"  Original dataset: {len(features_df)} samples, {len(features_df.columns)} features")
    
    # Remove unnecessary features but KEEP sample identifiers
    columns_to_remove = [
        'filename', 'filepath', 'analysis_timestamp',
        'peak_centers', 'peak_amplitudes', 'peak_fwhms', 'peak_areas', 
        'peak_to_baseline_ratios'
    ]
    
    # Remove columns that exist in the dataset
    existing_columns_to_remove = [col for col in columns_to_remove if col in features_df.columns]
    ml_features = features_df.drop(existing_columns_to_remove, axis=1)
    
    print(f"  Removed {len(existing_columns_to_remove)} unnecessary columns")
    
    # Define essential features in order (including sample identifier)
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
    
    # Save clean ML dataset
    ml_dataset_path = features_dir / "ml_dataset_clean.csv"
    ml_features_clean.to_csv(ml_dataset_path, index=False)
    
    print(f"Clean ML dataset saved: {ml_dataset_path}")
    print(f"  - {len(ml_features_clean.columns)} essential features (including sample ID)")
    print(f"  - {len(ml_features_clean)} samples")
    
    # Print sample identifiers for reference
    print(f"\nSample Identifiers:")
    for i, sample_name in enumerate(ml_features_clean['molecule_name']):
        print(f"  Row {i+1}: {sample_name}")
    
    # Print feature summary
    print(f"\nFeature Categories:")
    print(f"  Sample identifier: 1 (molecule_name)")
    print(f"  Core spectral: {len([f for f in available_features if f in ['num_peaks', 'peak_density', 'total_spectral_area', 'peak_area_fraction', 'energy_span', 'mean_center', 'std_center']])}")
    print(f"  Amplitude: {len([f for f in available_features if 'amplitude' in f])}")
    print(f"  Width: {len([f for f in available_features if 'fwhm' in f])}")
    print(f"  Area: {len([f for f in available_features if 'area' in f and 'baseline' not in f and 'peak_area_fraction' not in f])}")
    print(f"  Baseline: {len([f for f in available_features if 'baseline' in f])}")
    print(f"  Peak-to-baseline ratios: {len([f for f in available_features if 'ratio' in f])}")
    print(f"  Energy regions: {len([f for f in available_features if 'energy_peaks' in f])}")
    print(f"  Peak spacing: {len([f for f in available_features if 'spacing' in f])}")
    print(f"  Fit quality: {len([f for f in available_features if f in ['r_squared', 'rmse', 'baseline']])}")
    
    return ml_features_clean

if __name__ == "__main__":
    create_clean_ml_dataset() 