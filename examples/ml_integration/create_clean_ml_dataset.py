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
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from config.output_config import get_analysis_results_path, get_features_file_path, get_clean_ml_dataset_path

def create_clean_ml_dataset(output_dir_name=None):
    """Create a clean ML dataset from comprehensive analysis results."""
    
    # Get paths using centralized configuration
    results_dir = get_analysis_results_path(output_dir_name)
    features_dir = results_dir / "features"
    
    # Read the combined features file
    combined_file = get_features_file_path(output_dir_name, "all_molecules_features.csv")
    
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
        
        # Energy region features (basic counts)
        'low_energy_peaks', 'mid_energy_peaks', 'high_energy_peaks',
        
        # Enhanced energy region features (comprehensive analysis)
        # Low energy region features (0-500 cm⁻¹)
        'low_energy_peak_count', 'low_energy_amplitude_mean', 'low_energy_amplitude_std',
        'low_energy_amplitude_max', 'low_energy_amplitude_min', 'low_energy_amplitude_median',
        'low_energy_amplitude_skewness', 'low_energy_amplitude_kurtosis', 'low_energy_amplitude_cv',
        'low_energy_amplitude_iqr', 'low_energy_amplitude_percentile_25', 'low_energy_amplitude_percentile_75',
        'low_energy_fwhm_mean', 'low_energy_fwhm_std', 'low_energy_fwhm_max', 'low_energy_fwhm_min',
        'low_energy_fwhm_median', 'low_energy_fwhm_skewness', 'low_energy_fwhm_kurtosis', 'low_energy_fwhm_cv',
        'low_energy_fwhm_iqr', 'low_energy_fwhm_percentile_25', 'low_energy_fwhm_percentile_75',
        'low_energy_area_mean', 'low_energy_area_std', 'low_energy_area_max', 'low_energy_area_min',
        'low_energy_area_median', 'low_energy_area_skewness', 'low_energy_area_kurtosis', 'low_energy_area_cv',
        'low_energy_area_iqr', 'low_energy_area_percentile_25', 'low_energy_area_percentile_75',
        
        # Mid energy region features (500-2000 cm⁻¹)
        'mid_energy_peak_count', 'mid_energy_amplitude_mean', 'mid_energy_amplitude_std',
        'mid_energy_amplitude_max', 'mid_energy_amplitude_min', 'mid_energy_amplitude_median',
        'mid_energy_amplitude_skewness', 'mid_energy_amplitude_kurtosis', 'mid_energy_amplitude_cv',
        'mid_energy_amplitude_iqr', 'mid_energy_amplitude_percentile_25', 'mid_energy_amplitude_percentile_75',
        'mid_energy_fwhm_mean', 'mid_energy_fwhm_std', 'mid_energy_fwhm_max', 'mid_energy_fwhm_min',
        'mid_energy_fwhm_median', 'mid_energy_fwhm_skewness', 'mid_energy_fwhm_kurtosis', 'mid_energy_fwhm_cv',
        'mid_energy_fwhm_iqr', 'mid_energy_fwhm_percentile_25', 'mid_energy_fwhm_percentile_75',
        'mid_energy_area_mean', 'mid_energy_area_std', 'mid_energy_area_max', 'mid_energy_area_min',
        'mid_energy_area_median', 'mid_energy_area_skewness', 'mid_energy_area_kurtosis', 'mid_energy_area_cv',
        'mid_energy_area_iqr', 'mid_energy_area_percentile_25', 'mid_energy_area_percentile_75',
        
        # High energy region features (2000-3500 cm⁻¹)
        'high_energy_peak_count', 'high_energy_amplitude_mean', 'high_energy_amplitude_std',
        'high_energy_amplitude_max', 'high_energy_amplitude_min', 'high_energy_amplitude_median',
        'high_energy_amplitude_skewness', 'high_energy_amplitude_kurtosis', 'high_energy_amplitude_cv',
        'high_energy_amplitude_iqr', 'high_energy_amplitude_percentile_25', 'high_energy_amplitude_percentile_75',
        'high_energy_fwhm_mean', 'high_energy_fwhm_std', 'high_energy_fwhm_max', 'high_energy_fwhm_min',
        'high_energy_fwhm_median', 'high_energy_fwhm_skewness', 'high_energy_fwhm_kurtosis', 'high_energy_fwhm_cv',
        'high_energy_fwhm_iqr', 'high_energy_fwhm_percentile_25', 'high_energy_fwhm_percentile_75',
        'high_energy_area_mean', 'high_energy_area_std', 'high_energy_area_max', 'high_energy_area_min',
        'high_energy_area_median', 'high_energy_area_skewness', 'high_energy_area_kurtosis', 'high_energy_area_cv',
        'high_energy_area_iqr', 'high_energy_area_percentile_25', 'high_energy_area_percentile_75',
        
        # Peak spacing features
        'mean_peak_spacing', 'std_peak_spacing',
        
        # Fit quality features
        'r_squared', 'rmse', 'baseline'
    ]
    
    # Keep only essential features that exist in the dataset
    available_features = [col for col in essential_features if col in ml_features.columns]
    ml_features_clean = ml_features[available_features]
    
    # Save clean ML dataset
    ml_dataset_path = get_clean_ml_dataset_path(output_dir_name)
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