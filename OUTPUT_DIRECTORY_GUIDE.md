# Output Directory Configuration Guide

## Overview

The INS ML Analysis System now uses a **centralized output configuration** to ensure consistent directory structure and prevent confusion between batch analysis and ML integration workflows.

### 🚀 Enhanced Energy Region Features

The system now provides **comprehensive energy region analysis** with 93 enhanced features across three energy regions:

- **Low Energy (0-500 cm⁻¹)**: 31 features for fundamental vibrations
- **Mid Energy (500-2000 cm⁻¹)**: 31 features for combination bands  
- **High Energy (2000-3500 cm⁻¹)**: 31 features for overtone vibrations

**Enhanced Features Include:**
- **Amplitude Statistics**: Mean, std, max, min, median, skewness, kurtosis, CV, IQR, percentiles
- **Width Statistics**: FWHM analysis with full statistical distribution
- **Area Statistics**: Integrated intensities with comprehensive analysis
- **Cross-Region Correlations**: Energy-dependent structural relationships

**Benefits:**
- **Better Structure Discrimination**: Detailed energy-dependent analysis
- **Improved Temperature Studies**: Enhanced sensitivity to temperature changes
- **Advanced ML Models**: 90+ new features for better predictive performance
- **Comprehensive Analysis**: 151 total features vs. original 61 features

## Centralized Configuration

All output directories are managed through `src/config/output_config.py`, which provides:

- **Consistent directory structure** across all analysis workflows
- **Centralized path management** to prevent hardcoded paths
- **Automatic directory creation** with proper organization
- **Clear documentation** of where each file type is saved

## Directory Structure

```
📁 comprehensive_analysis_results/
  ├── 📁 plots/
  │   ├── 📁 main_analysis/      - Main spectrum + fit plots
  │   ├── 📁 baseline_detection/ - Baseline analysis plots
  │   ├── 📁 peak_detection/     - Peak detection plots
  │   └── 📁 kde_density/        - KDE density plots
  ├── 📁 features/               - Extracted features CSV files
  │   ├── all_molecules_features.csv (202 features)
  │   ├── ml_dataset.csv
  │   ├── ml_dataset_clean.csv (151 features)
  │   └── individual_*_features.csv
  ├── 📁 summaries/              - Analysis summaries
  │   ├── analysis_summary.csv
  │   └── statistical_summary.csv
  ├── 📁 logs/                   - Analysis logs
  └── 📁 pb_ratio_analysis/      - Peak-to-baseline analysis
      ├── 📁 plots/              - P/B ratio plots
      ├── 📁 statistics/         - P/B ratio statistics
      └── 📁 comparisons/        - Statistical comparisons
```

## Key Files and Their Locations

### Batch Analysis Results
- **Main plots**: `comprehensive_analysis_results/plots/main_analysis/`
- **Baseline plots**: `comprehensive_analysis_results/plots/baseline_detection/`
- **Peak detection plots**: `comprehensive_analysis_results/plots/peak_detection/`
- **KDE density plots**: `comprehensive_analysis_results/plots/kde_density/`
- **All features**: `comprehensive_analysis_results/features/all_molecules_features.csv`
- **Clean ML dataset**: `comprehensive_analysis_results/features/ml_dataset_clean.csv`

### ML Integration Results
- **Clean dataset**: `comprehensive_analysis_results/features/ml_dataset_clean.csv`
- **Analysis summaries**: `comprehensive_analysis_results/summaries/`
- **P/B ratio analysis**: `comprehensive_analysis_results/pb_ratio_analysis/`

## Usage Guidelines

### For Batch Analysis
1. **Run batch analysis**: `python3 examples/batch_processing/run_batch_analysis.py`
2. **Results automatically saved to**: `comprehensive_analysis_results/`
3. **All plots and features organized** in proper subdirectories
4. **Clean ML dataset automatically created** in `features/ml_dataset_clean.csv`

### For ML Integration
1. **Clean dataset already available** at `comprehensive_analysis_results/features/ml_dataset_clean.csv`
2. **Run ML analysis**: `python3 examples/ml_integration/ml_example.py`
3. **All results saved** in the same `comprehensive_analysis_results/` directory

## Configuration Functions

### Core Functions
- `get_output_dir(output_dir_name)` - Get output directory path
- `create_output_structure(output_dir)` - Create complete directory structure
- `print_output_structure(output_dir)` - Display directory structure

### Path Functions
- `get_analysis_results_path(output_dir_name)` - Get analysis results path
- `get_features_file_path(output_dir_name, filename)` - Get features file path
- `get_clean_ml_dataset_path(output_dir_name)` - Get clean ML dataset path

## Benefits

### ✅ **No More Confusion**
- All results saved in one consistent location
- Clear directory structure with descriptive names
- Centralized configuration prevents path conflicts

### ✅ **Automatic Organization**
- Plots automatically sorted by type
- Features files properly named and organized
- Logs and summaries in dedicated directories

### ✅ **Easy Access**
- Clean ML dataset always at predictable location
- All analysis results in one place
- Clear documentation of file locations

### ✅ **Future-Proof**
- Easy to modify directory structure in one place
- Consistent across all analysis workflows
- Scalable for additional analysis types

## Troubleshooting

### If Results Are Not Found
1. Check that `comprehensive_analysis_results/` exists in the project root
2. Verify the directory structure matches the guide above
3. Ensure batch analysis completed successfully
4. Check logs in `comprehensive_analysis_results/logs/`

### If ML Integration Fails
1. Verify `ml_dataset_clean.csv` exists in `comprehensive_analysis_results/features/`
2. Check that the file contains the expected features
3. Ensure the file path is correctly referenced in ML scripts

## Migration Notes

- **Old results**: Previous analysis results may be in different locations
- **New workflow**: All future analyses will use this centralized structure
- **Backward compatibility**: Existing scripts updated to use new configuration
- **No data loss**: All existing results preserved, just reorganized

---

**Last Updated**: Current session
**Configuration File**: `src/config/output_config.py`
**Test Status**: ✅ Verified working correctly 