"""
Analysis Configuration
=====================

This file contains all configurable parameters for the INS ML analysis system.
Modify these parameters to customize the analysis for your specific needs.
"""

# Energy Range Configuration
ENERGY_RANGE = (0, 3500)  # cm⁻¹

# Peak Detection Parameters
PEAK_DETECTION_CONFIG = {
    'height': None,           # Minimum peak height (None = auto)
    'distance': 3,            # Minimum points between peaks
    'prominence': 0.01,       # Minimum prominence (sensitivity)
    'width': 1,               # Minimum peak width
    'smooth_window': 11       # Savitzky-Golay smoothing window
}

# Gaussian Fitting Parameters
FITTING_CONFIG = {
    'smoothing': True,        # Apply smoothing before fitting
    'smooth_window': 21,      # Smoothing window size
    'max_iterations': 1000,   # Maximum fitting iterations
    'tolerance': 1e-6,        # Fitting tolerance
    'baseline_fit': True      # Include baseline in fit
}

# Parameter Bounds for Fitting
PARAMETER_BOUNDS = {
    'amplitude': (0, None),   # Amplitude bounds (min, max)
    'center': (0, 3500),      # Center bounds (energy range)
    'sigma': (0.1, 100),      # Width bounds (FWHM/2.355)
    'baseline': (-1, 1)       # Baseline bounds
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    'include_individual_peaks': True,  # Include individual peak arrays
    'energy_regions': {                # Energy region boundaries
        'low': 1000,                   # Low energy cutoff
        'mid': 2000                    # Mid energy cutoff
    },
    'statistical_features': True,      # Include skewness, kurtosis, etc.
    'area_features': True,             # Include area-related features
    'quality_features': True           # Include fit quality features
}

# Plotting Configuration
PLOT_CONFIG = {
    'dpi': 300,               # Figure resolution
    'figsize': (12, 10),      # Figure size (width, height)
    'save_format': 'pdf',     # Save format (pdf, png, jpg)
    'style': 'default',       # Matplotlib style
    'color_scheme': 'viridis' # Color scheme for plots
}

# Output Configuration
OUTPUT_CONFIG = {
    'create_individual_plots': True,   # Create plots for each file
    'create_summary_plots': True,      # Create summary plots
    'save_features': True,             # Save extracted features
    'save_summaries': True,            # Save analysis summaries
    'create_logs': True,               # Create processing logs
    'output_format': 'csv'             # Output format for data
}

# Quality Control Parameters
QUALITY_CONFIG = {
    'min_r_squared': 0.8,     # Minimum R² for acceptable fits
    'max_rmse': 0.1,          # Maximum RMSE for acceptable fits
    'min_peaks': 1,           # Minimum number of peaks
    'max_peaks': 100,         # Maximum number of peaks
    'min_peak_amplitude': 0.001,  # Minimum peak amplitude
    'max_peak_amplitude': 10.0    # Maximum peak amplitude
}

# ML Dataset Configuration
ML_CONFIG = {
    'include_metadata': True,          # Include file metadata
    'normalize_features': False,       # Normalize features
    'remove_outliers': False,          # Remove statistical outliers
    'feature_selection': None,         # Feature selection method
    'cross_validation_folds': 5        # CV folds for ML evaluation
}

# Batch Processing Configuration
BATCH_CONFIG = {
    'file_pattern': '*.csv',           # File pattern to match
    'recursive_search': False,         # Search subdirectories
    'max_files': None,                 # Maximum files to process
    'parallel_processing': False,      # Use parallel processing
    'n_jobs': 1                       # Number of parallel jobs
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',                   # Logging level
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'analysis.log',            # Log file name
    'console_output': True             # Output to console
}

# Advanced Configuration
ADVANCED_CONFIG = {
    'use_gpu': False,                  # Use GPU acceleration (if available)
    'memory_limit': None,              # Memory limit in GB
    'cache_results': True,             # Cache intermediate results
    'cleanup_temp_files': True,        # Clean up temporary files
    'backup_results': False            # Create backup of results
}

# Custom Feature Definitions
CUSTOM_FEATURES = {
    'peak_ratios': True,               # Calculate peak amplitude ratios
    'energy_weighted_features': True,  # Energy-weighted statistics
    'spectral_moments': True,          # Spectral moment calculations
    'peak_clustering': False,          # Peak clustering analysis
    'wavelet_features': False          # Wavelet-based features
}

# Validation Configuration
VALIDATION_CONFIG = {
    'validate_input_data': True,       # Validate input data format
    'check_energy_range': True,        # Check energy range consistency
    'validate_peaks': True,            # Validate detected peaks
    'check_fit_quality': True,         # Check fit quality metrics
    'generate_validation_report': True # Generate validation report
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'optimize_memory': True,           # Optimize memory usage
    'use_fast_math': True,             # Use fast math operations
    'vectorize_operations': True,      # Vectorize calculations
    'profile_performance': False,      # Profile performance
    'cache_intermediate': True         # Cache intermediate results
}

# Export Configuration
EXPORT_CONFIG = {
    'export_formats': ['csv', 'json'], # Export formats
    'include_plots': True,             # Include plots in export
    'compress_output': False,          # Compress output files
    'create_archive': False,           # Create archive of results
    'export_metadata': True            # Export metadata
}

# Documentation Configuration
DOC_CONFIG = {
    'generate_feature_docs': True,     # Generate feature documentation
    'create_analysis_report': True,    # Create analysis report
    'include_statistics': True,        # Include statistical summaries
    'export_plots': True,              # Export all plots
    'create_summary_table': True       # Create summary table
}

def get_config():
    """Get the complete configuration dictionary."""
    return {
        'energy_range': ENERGY_RANGE,
        'peak_detection': PEAK_DETECTION_CONFIG,
        'fitting': FITTING_CONFIG,
        'parameter_bounds': PARAMETER_BOUNDS,
        'features': FEATURE_CONFIG,
        'plotting': PLOT_CONFIG,
        'output': OUTPUT_CONFIG,
        'quality': QUALITY_CONFIG,
        'ml': ML_CONFIG,
        'batch': BATCH_CONFIG,
        'logging': LOGGING_CONFIG,
        'advanced': ADVANCED_CONFIG,
        'custom_features': CUSTOM_FEATURES,
        'validation': VALIDATION_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'export': EXPORT_CONFIG,
        'documentation': DOC_CONFIG
    }

def validate_config(config):
    """Validate configuration parameters."""
    errors = []
    
    # Check energy range
    if config['energy_range'][0] >= config['energy_range'][1]:
        errors.append("Energy range: min must be less than max")
    
    # Check peak detection parameters
    if config['peak_detection']['distance'] < 1:
        errors.append("Peak distance must be >= 1")
    
    if config['peak_detection']['prominence'] <= 0:
        errors.append("Peak prominence must be > 0")
    
    # Check fitting parameters
    if config['fitting']['max_iterations'] <= 0:
        errors.append("Max iterations must be > 0")
    
    if config['fitting']['tolerance'] <= 0:
        errors.append("Tolerance must be > 0")
    
    # Check quality parameters
    if not (0 <= config['quality']['min_r_squared'] <= 1):
        errors.append("Min R² must be between 0 and 1")
    
    if config['quality']['max_rmse'] <= 0:
        errors.append("Max RMSE must be > 0")
    
    return errors

def print_config_summary():
    """Print a summary of the current configuration."""
    config = get_config()
    
    print("="*60)
    print("INS ML ANALYSIS CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"Energy Range: {config['energy_range'][0]} - {config['energy_range'][1]} cm⁻¹")
    print(f"Peak Detection Sensitivity: prominence = {config['peak_detection']['prominence']}")
    print(f"Fitting: smoothing = {config['fitting']['smoothing']}, max_iter = {config['fitting']['max_iterations']}")
    print(f"Quality Control: min R² = {config['quality']['min_r_squared']}, max RMSE = {config['quality']['max_rmse']}")
    print(f"Output: individual_plots = {config['output']['create_individual_plots']}")
    print(f"ML Features: {len(config['features'])} feature categories enabled")
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        print("\n⚠ Configuration Warnings:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ Configuration is valid")
    
    print("="*60)

if __name__ == "__main__":
    print_config_summary() 