#!/usr/bin/env python3
"""
Output Configuration for INS ML Analysis System
===============================================

This module provides centralized configuration for all output directories
to ensure consistency across batch analysis and ML integration workflows.
"""

import os
from pathlib import Path

# Project root directory (relative to this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default output directory names
DEFAULT_OUTPUT_DIR = "comprehensive_analysis_results"
DEFAULT_ML_OUTPUT_DIR = "ml_integration_results"

# Output directory structure
OUTPUT_STRUCTURE = {
    "plots": {
        "main_analysis": "Main spectrum + fit plots",
        "baseline_detection": "Baseline analysis plots", 
        "peak_detection": "Peak detection plots",
        "kde_density": "KDE density plots"
    },
    "features": "Extracted features CSV files",
    "summaries": "Analysis summaries and statistics",
    "logs": "Analysis logs and processing history",
    "pb_ratio_analysis": {
        "plots": "Peak-to-baseline ratio plots",
        "statistics": "P/B ratio statistics",
        "comparisons": "Statistical comparisons"
    }
}

def get_output_dir(output_dir_name=None, base_path=None):
    """
    Get the full path to the output directory.
    
    Parameters:
    -----------
    output_dir_name : str, optional
        Name of the output directory (default: DEFAULT_OUTPUT_DIR)
    base_path : str or Path, optional
        Base path for the output directory (default: PROJECT_ROOT)
        
    Returns:
    --------
    Path
        Full path to the output directory
    """
    if output_dir_name is None:
        output_dir_name = DEFAULT_OUTPUT_DIR
    
    if base_path is None:
        base_path = PROJECT_ROOT
    
    return Path(base_path) / output_dir_name

def create_output_structure(output_dir):
    """
    Create the complete output directory structure.
    
    Parameters:
    -----------
    output_dir : str or Path
        Base output directory path
        
    Returns:
    --------
    dict
        Dictionary with all created directory paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create main directories
    plots_dir = output_dir / "plots"
    features_dir = output_dir / "features"
    summaries_dir = output_dir / "summaries"
    logs_dir = output_dir / "logs"
    pb_ratio_dir = output_dir / "pb_ratio_analysis"
    
    # Create plot subdirectories
    main_plots_dir = plots_dir / "main_analysis"
    baseline_plots_dir = plots_dir / "baseline_detection"
    peak_plots_dir = plots_dir / "peak_detection"
    kde_plots_dir = plots_dir / "kde_density"
    
    # Create P/B ratio subdirectories
    pb_plots_dir = pb_ratio_dir / "plots"
    pb_stats_dir = pb_ratio_dir / "statistics"
    pb_comparisons_dir = pb_ratio_dir / "comparisons"
    
    # Create all directories
    directories = [
        plots_dir, features_dir, summaries_dir, logs_dir, pb_ratio_dir,
        main_plots_dir, baseline_plots_dir, peak_plots_dir, kde_plots_dir,
        pb_plots_dir, pb_stats_dir, pb_comparisons_dir
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
    
    return {
        "base": output_dir,
        "plots": plots_dir,
        "features": features_dir,
        "summaries": summaries_dir,
        "logs": logs_dir,
        "pb_ratio": pb_ratio_dir,
        "main_plots": main_plots_dir,
        "baseline_plots": baseline_plots_dir,
        "peak_plots": peak_plots_dir,
        "kde_plots": kde_plots_dir,
        "pb_plots": pb_plots_dir,
        "pb_stats": pb_stats_dir,
        "pb_comparisons": pb_comparisons_dir
    }

def get_ml_output_dir(output_dir_name=None, base_path=None):
    """
    Get the full path to the ML integration output directory.
    
    Parameters:
    -----------
    output_dir_name : str, optional
        Name of the ML output directory (default: DEFAULT_ML_OUTPUT_DIR)
    base_path : str or Path, optional
        Base path for the output directory (default: PROJECT_ROOT)
        
    Returns:
    --------
    Path
        Full path to the ML output directory
    """
    if output_dir_name is None:
        output_dir_name = DEFAULT_ML_OUTPUT_DIR
    
    if base_path is None:
        base_path = PROJECT_ROOT
    
    return Path(base_path) / output_dir_name

def print_output_structure(output_dir):
    """
    Print the output directory structure for user reference.
    
    Parameters:
    -----------
    output_dir : str or Path
        Output directory path
    """
    output_dir = Path(output_dir)
    
    print("="*80)
    print("OUTPUT DIRECTORY STRUCTURE")
    print("="*80)
    print(f"Base directory: {output_dir}")
    print("="*80)
    print("📁 comprehensive_analysis_results/")
    print("  ├── 📁 plots/")
    print("  │   ├── 📁 main_analysis/      - Main spectrum + fit plots")
    print("  │   ├── 📁 baseline_detection/ - Baseline analysis plots")
    print("  │   ├── 📁 peak_detection/     - Peak detection plots")
    print("  │   └── 📁 kde_density/        - KDE density plots")
    print("  ├── 📁 features/               - Extracted features CSV files")
    print("  │   ├── all_molecules_features.csv")
    print("  │   ├── ml_dataset.csv")
    print("  │   ├── ml_dataset_clean.csv")
    print("  │   └── individual_*_features.csv")
    print("  ├── 📁 summaries/              - Analysis summaries")
    print("  │   ├── analysis_summary.csv")
    print("  │   └── statistical_summary.csv")
    print("  ├── 📁 logs/                   - Analysis logs")
    print("  └── 📁 pb_ratio_analysis/      - Peak-to-baseline analysis")
    print("      ├── 📁 plots/              - P/B ratio plots")
    print("      ├── 📁 statistics/         - P/B ratio statistics")
    print("      └── 📁 comparisons/        - Statistical comparisons")
    print("="*80)

def get_analysis_results_path(output_dir_name=None):
    """
    Get the path to the analysis results directory.
    
    Parameters:
    -----------
    output_dir_name : str, optional
        Name of the output directory
        
    Returns:
    --------
    Path
        Path to the analysis results directory
    """
    return get_output_dir(output_dir_name)

def get_features_file_path(output_dir_name=None, filename="all_molecules_features.csv"):
    """
    Get the path to a specific features file.
    
    Parameters:
    -----------
    output_dir_name : str, optional
        Name of the output directory
    filename : str
        Name of the features file
        
    Returns:
    --------
    Path
        Path to the features file
    """
    output_dir = get_output_dir(output_dir_name)
    return output_dir / "features" / filename

def get_clean_ml_dataset_path(output_dir_name=None):
    """
    Get the path to the clean ML dataset.
    
    Parameters:
    -----------
    output_dir_name : str, optional
        Name of the output directory
        
    Returns:
    --------
    Path
        Path to the clean ML dataset
    """
    return get_features_file_path(output_dir_name, "ml_dataset_clean.csv") 