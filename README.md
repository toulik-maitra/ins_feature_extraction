# INS Spectrum ML Feature Extraction

A comprehensive, automated workflow for analyzing Inelastic Neutron Scattering (INS) spectra and extracting machine learning-ready features. This system was developed to streamline the analysis of INS data and provide researchers with robust, reproducible feature extraction capabilities.

## Quick Start

### System Overview
```bash
python3 main.py                    # View system overview
python3 quick_start.py             # Quick start demonstration
```

### Complete Analysis Workflow (Recommended)
```bash
python3 run_complete_analysis.py   # Runs everything automatically
```

### Individual Steps
```bash
# Single file analysis
python3 examples/single_file/comprehensive_test_suite.py

# Batch analysis (includes automatic ML dataset creation)
python3 examples/batch_processing/run_batch_analysis.py

# Manual ML dataset creation (if needed)
python3 examples/ml_integration/create_clean_ml_dataset.py
```

## Project Structure

```
INS_ML_Analysis_System/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                           # Installation script
├── main.py                            # Main entry point
├── quick_start.py                     # Quick start demonstration
├── run_complete_analysis.py           # Complete workflow runner
├── src/                               # Source code
│   ├── core/                          # Core analysis modules
│   │   ├── enhanced_ml_peak_analyzer.py # Enhanced analyzer class (recommended)
│   │   ├── ml_peak_analyzer.py        # Legacy analyzer class (deprecated)
│   │   └── batch_ml_analysis.py       # Batch processing workflow
│   ├── utils/                         # Utility functions
│   │   ├── enhanced_baseline_detection.py # Enhanced baseline detection (recommended)
│   │   ├── baseline_detection.py      # Legacy baseline detection (deprecated)
│   │   └── run_single_INS_analysis.py # Single file runner
│   ├── visualization/                 # Plotting utilities
│   └── config/                        # Configuration files
│       ├── __init__.py
│       └── analysis_config.py         # Configuration settings
├── examples/                          # Example scripts
│   ├── single_file/                   # Single file analysis examples
│   │   ├── comprehensive_test_suite.py # Unified test suite (recommended)
│   │   ├── test_ml_analyzer.py        # Legacy test file (deprecated)
│   │   └── test_baseline_features.py  # Legacy baseline detection example
│   ├── batch_processing/              # Batch processing examples
│   │   ├── run_batch_analysis.py      # Main batch analysis script
│   │   └── run_batch_demo.py          # Batch processing demo
│   ├── batch_analysis/                # Batch analysis scripts
│   ├── dev_scripts/                   # Development and test scripts
│   └── ml_integration/                # ML integration examples (gitignored)
├── docs/                              # Documentation
│   ├── USAGE_GUIDE.md                 # Detailed usage instructions
│   ├── TECHNICAL_DOCS.md              # Technical documentation
│   └── CHANGELOG.md                   # Version history
└── comprehensive_analysis_results/    # Analysis results (created during runtime)
    ├── features/                      # Extracted features
    ├── plots/                         # Generated plots
    ├── summaries/                     # Analysis summaries
    └── logs/                          # Processing logs
```

## Output Organization

The system creates a comprehensive, organized output structure. Note that all results, features, and ML integration scripts are excluded from version control via `.gitignore` to keep the repository clean. Do not upload any files from these directories:

- `comprehensive_analysis_results/` (all subdirectories)
- `ml_integration_results/`
- `examples/ml_integration/`

See `.gitignore` for complete details.

```
comprehensive_analysis_results/
├── features/                          # Extracted features
│   ├── ml_dataset_clean.csv          # Clean ML dataset (151 features, 56 samples)
│   ├── all_molecules_features.csv    # Complete dataset with all features (202 features)
│   └── [molecule]_features.csv       # Individual feature files
├── plots/                            # Publication-quality plots
│   ├── main_analysis/                # Main spectrum analysis plots
│   ├── baseline_detection/           # Baseline detection plots
│   ├── peak_detection/               # Peak detection plots
│   └── kde_density/                  # KDE density distribution plots
├── summaries/                        # Analysis summaries
│   └── analysis_summary.csv          # Summary statistics
└── logs/                             # Processing logs
ml_integration_results/               # ML analysis results (excluded from git)
examples/ml_integration/              # ML integration scripts and outputs (excluded from git)
```

## Version Control and GitHub

- **Repository:** https://github.com/toulik-maitra/ins_feature_extraction
- **Important:** All ML integration scripts and results are excluded from version control. Only core analysis code, documentation, and configuration should be committed.
- **.gitignore:** Updated to ensure no results or ML integration scripts are uploaded.

## Code Organization and Module Recommendations

### Recommended Modules (Use These)
- **`src/core/enhanced_ml_peak_analyzer.py`** - Enhanced ML peak analyzer with optimization capabilities
- **`src/utils/enhanced_baseline_detection.py`** - Enhanced baseline detection with validation and parameter optimization
- **`examples/single_file/comprehensive_test_suite.py`** - Unified test suite that consolidates multiple test files

### Legacy Modules (Deprecated)
- **`src/core/ml_peak_analyzer.py`** - Basic ML peak analyzer (deprecated, use enhanced version)
- **`src/utils/baseline_detection.py`** - Basic baseline detection (deprecated, use enhanced version)
- **`examples/single_file/test_ml_analyzer.py`** - Basic test file (replaced by comprehensive test suite)
- **`examples/single_file/enhanced_baseline_demo.py`** - Standalone demo (functionality integrated into comprehensive test suite)

### Code Consolidation
The codebase has been consolidated to remove duplications and improve maintainability:
- **Unified test suite** replaces multiple test files with overlapping functionality
- **Enhanced modules** provide all functionality of legacy modules plus additional features
- **Deprecation warnings** guide users to recommended modules
- **Consolidated examples** reduce code repetition and improve organization

### Migration Guide
If you're using legacy modules, migrate to:
- `ml_peak_analyzer.py` → `enhanced_ml_peak_analyzer.py`
- `baseline_detection.py` → `enhanced_baseline_detection.py`
- Individual test files → `comprehensive_test_suite.py`

## Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Pandas, Matplotlib, Scikit-learn, Seaborn

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Development Setup
```bash
pip install -e .
```

## Key Features

### 🚀 Enhanced Energy Region Analysis (NEW!)
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

### Comprehensive Feature Extraction (151 ML-Ready Features)
- **Sample Identification**: Each row includes `molecule_name` for clear sample tracking
- **Peak Analysis**: Count, density, amplitude, width, area statistics
- **Baseline Analysis**: Detected baseline areas and signal quality metrics
- **Peak-to-Baseline Ratios**: With outlier removal (1st-99th percentile)
- **Enhanced Energy Region Analysis**: 93 comprehensive features per energy region (0-500, 500-2000, 2000-3500 cm⁻¹)
- **Fit Quality Metrics**: R², RMSE, baseline values

### Advanced Peak Detection
- **Shoulder Detection**: Sensitive detection of shoulder peaks
- **Strict Energy Range**: All analysis within 0-3500 cm⁻¹
- **Configurable Sensitivity**: Different settings for experimental vs simulated data
- **Robust Fitting**: Gaussian peak fitting with quality metrics

### Baseline Detection
- **Multiple Algorithms**: Dynamic rolling, polynomial, and statistical methods
- **Peak-to-Baseline Ratios**: Signal quality assessment
- **Outlier Removal**: Automatic removal of extreme ratio values
- **Visualization**: Dedicated baseline detection plots

### Instrument Peak Width Analysis
- **Peak Width vs Energy Plots**: Include instrument resolution line
- **Instrument Equation**: `0.25 + 0.005*energy + 0.0000001*(energy^2)`
- **Trend Analysis**: Data trend line vs instrument line comparison
- **Quality Assessment**: Peak width relative to instrument resolution

## Comprehensive Feature Documentation

The system extracts 151 essential features from INS spectra, organized into the following categories:

### 1. Sample Identification
| Feature | Description | ML Relevance |
|---------|-------------|--------------|
| `molecule_name` | Sample identifier | Sample tracking and grouping |

### 2. Core Spectral Features (7 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `num_peaks` | Total number of detected peaks | count | Molecular complexity indicator |
| `peak_density` | Peaks per unit energy | peaks/cm⁻¹ | Spectral complexity density |
| `total_spectral_area` | Total integrated area of spectrum | a.u.·cm⁻¹ | Complete spectral intensity |
| `peak_area_fraction` | Fraction of spectrum in peaks | dimensionless | Peak dominance |
| `energy_span` | Total energy range covered | cm⁻¹ | Spectral breadth |
| `mean_center` | Average peak position | cm⁻¹ | Central energy region |
| `std_center` | Standard deviation of positions | cm⁻¹ | Position spread |

### 3. Amplitude Features (8 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `mean_amplitude` | Average peak amplitude | a.u. | Overall spectral intensity |
| `std_amplitude` | Standard deviation of amplitudes | a.u. | Amplitude variability |
| `max_amplitude` | Maximum peak amplitude | a.u. | Strongest spectral feature |
| `min_amplitude` | Minimum peak amplitude | a.u. | Weakest spectral feature |
| `amplitude_range` | Range of amplitudes | a.u. | Dynamic range |
| `amplitude_cv` | Coefficient of variation | dimensionless | Relative amplitude spread |
| `amplitude_skewness` | Skewness of amplitude distribution | dimensionless | Amplitude asymmetry |
| `amplitude_kurtosis` | Kurtosis of amplitude distribution | dimensionless | Amplitude peakedness |

### 4. Width Features (8 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `mean_width` | Average peak width (FWHM) | cm⁻¹ | Typical peak sharpness |
| `std_width` | Standard deviation of widths | cm⁻¹ | Width variability |
| `max_width` | Maximum peak width | cm⁻¹ | Broadest feature |
| `min_width` | Minimum peak width | cm⁻¹ | Sharpest feature |
| `width_range` | Range of peak widths | cm⁻¹ | Width diversity |
| `width_cv` | Coefficient of variation | dimensionless | Relative width spread |
| `width_skewness` | Skewness of width distribution | dimensionless | Width asymmetry |
| `width_kurtosis` | Kurtosis of width distribution | dimensionless | Width peakedness |

### 5. Area Features (8 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `mean_area` | Average peak area | a.u.·cm⁻¹ | Typical peak intensity |
| `std_area` | Standard deviation of areas | a.u.·cm⁻¹ | Area variability |
| `max_area` | Maximum peak area | a.u.·cm⁻¹ | Strongest integrated feature |
| `min_area` | Minimum peak area | a.u.·cm⁻¹ | Weakest integrated feature |
| `area_range` | Range of peak areas | a.u.·cm⁻¹ | Area diversity |
| `area_cv` | Coefficient of variation | dimensionless | Relative area spread |
| `area_skewness` | Skewness of area distribution | dimensionless | Area asymmetry |
| `area_kurtosis` | Kurtosis of area distribution | dimensionless | Area peakedness |

### 6. Enhanced Energy Region Analysis (93 features)
**Low Energy Region (0-500 cm⁻¹) - 31 features:**
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `low_energy_peaks` | Peak count in 0-500 cm⁻¹ | count | Fundamental vibrations |
| `low_energy_peak_count` | Enhanced peak count | count | Detailed fundamental analysis |
| `low_energy_amplitude_mean/std/max/min/median` | Amplitude statistics | a.u. | Fundamental vibration intensities |
| `low_energy_amplitude_skewness/kurtosis/cv` | Amplitude distribution | dimensionless | Fundamental vibration patterns |
| `low_energy_amplitude_iqr/percentile_25/75` | Amplitude percentiles | a.u. | Fundamental vibration ranges |
| `low_energy_fwhm_mean/std/max/min/median` | Width statistics | cm⁻¹ | Fundamental peak sharpness |
| `low_energy_fwhm_skewness/kurtosis/cv` | Width distribution | dimensionless | Fundamental peak shapes |
| `low_energy_fwhm_iqr/percentile_25/75` | Width percentiles | cm⁻¹ | Fundamental peak width ranges |
| `low_energy_area_mean/std/max/min/median` | Area statistics | a.u.·cm⁻¹ | Fundamental integrated intensities |
| `low_energy_area_skewness/kurtosis/cv` | Area distribution | dimensionless | Fundamental intensity patterns |
| `low_energy_area_iqr/percentile_25/75` | Area percentiles | a.u.·cm⁻¹ | Fundamental intensity ranges |

**Mid Energy Region (500-2000 cm⁻¹) - 31 features:**
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `mid_energy_peaks` | Peak count in 500-2000 cm⁻¹ | count | Combination bands |
| `mid_energy_peak_count` | Enhanced peak count | count | Detailed combination analysis |
| `mid_energy_amplitude_mean/std/max/min/median` | Amplitude statistics | a.u. | Combination band intensities |
| `mid_energy_amplitude_skewness/kurtosis/cv` | Amplitude distribution | dimensionless | Combination band patterns |
| `mid_energy_amplitude_iqr/percentile_25/75` | Amplitude percentiles | a.u. | Combination band ranges |
| `mid_energy_fwhm_mean/std/max/min/median` | Width statistics | cm⁻¹ | Combination peak sharpness |
| `mid_energy_fwhm_skewness/kurtosis/cv` | Width distribution | dimensionless | Combination peak shapes |
| `mid_energy_fwhm_iqr/percentile_25/75` | Width percentiles | cm⁻¹ | Combination peak width ranges |
| `mid_energy_area_mean/std/max/min/median` | Area statistics | a.u.·cm⁻¹ | Combination integrated intensities |
| `mid_energy_area_skewness/kurtosis/cv` | Area distribution | dimensionless | Combination intensity patterns |
| `mid_energy_area_iqr/percentile_25/75` | Area percentiles | a.u.·cm⁻¹ | Combination intensity ranges |

**High Energy Region (2000-3500 cm⁻¹) - 31 features:**
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `high_energy_peaks` | Peak count in 2000-3500 cm⁻¹ | count | Overtone vibrations |
| `high_energy_peak_count` | Enhanced peak count | count | Detailed overtone analysis |
| `high_energy_amplitude_mean/std/max/min/median` | Amplitude statistics | a.u. | Overtone vibration intensities |
| `high_energy_amplitude_skewness/kurtosis/cv` | Amplitude distribution | dimensionless | Overtone vibration patterns |
| `high_energy_amplitude_iqr/percentile_25/75` | Amplitude percentiles | a.u. | Overtone vibration ranges |
| `high_energy_fwhm_mean/std/max/min/median` | Width statistics | cm⁻¹ | Overtone peak sharpness |
| `high_energy_fwhm_skewness/kurtosis/cv` | Width distribution | dimensionless | Overtone peak shapes |
| `high_energy_fwhm_iqr/percentile_25/75` | Width percentiles | cm⁻¹ | Overtone peak width ranges |
| `high_energy_area_mean/std/max/min/median` | Area statistics | a.u.·cm⁻¹ | Overtone integrated intensities |
| `high_energy_area_skewness/kurtosis/cv` | Area distribution | dimensionless | Overtone intensity patterns |
| `high_energy_area_iqr/percentile_25/75` | Area percentiles | a.u.·cm⁻¹ | Overtone intensity ranges |

### 7. Peak-to-Baseline Ratios (8 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `mean_peak_baseline_ratio` | Average peak-to-baseline ratio | dimensionless | Signal quality |
| `std_peak_baseline_ratio` | Standard deviation of ratios | dimensionless | Ratio variability |
| `max_peak_baseline_ratio` | Maximum ratio | dimensionless | Best signal quality |
| `min_peak_baseline_ratio` | Minimum ratio | dimensionless | Worst signal quality |
| `ratio_range` | Range of ratios | dimensionless | Signal quality spread |
| `ratio_cv` | Coefficient of variation | dimensionless | Relative ratio spread |
| `ratio_skewness` | Skewness of ratio distribution | dimensionless | Ratio asymmetry |
| `ratio_kurtosis` | Kurtosis of ratio distribution | dimensionless | Ratio peakedness |

### 8. Baseline Analysis (6 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `baseline_area` | Total baseline area | a.u.·cm⁻¹ | Background intensity |
| `baseline_mean` | Average baseline value | a.u. | Background level |
| `baseline_std` | Standard deviation of baseline | a.u. | Background variability |
| `baseline_slope` | Linear baseline slope | a.u./cm⁻¹ | Background trend |
| `baseline_r2` | Baseline fit R² | dimensionless | Baseline fit quality |
| `baseline_rmse` | Baseline fit RMSE | a.u. | Baseline fit error |

### 9. Fit Quality Metrics (4 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `mean_r2` | Average peak fit R² | dimensionless | Overall fit quality |
| `mean_rmse` | Average peak fit RMSE | a.u. | Overall fit error |
| `fit_quality_score` | Composite fit quality | dimensionless | Combined quality metric |
| `failed_fits` | Number of failed peak fits | count | Fit reliability |

### 10. Advanced Statistical Features (6 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `spectral_complexity` | Spectral complexity index | dimensionless | Overall complexity |
| `peak_clustering` | Peak clustering coefficient | dimensionless | Peak distribution pattern |
| `energy_efficiency` | Energy distribution efficiency | dimensionless | Energy utilization |
| `spectral_entropy` | Spectral entropy | dimensionless | Information content |
| `peak_symmetry` | Peak symmetry index | dimensionless | Spectral symmetry |
| `spectral_regularity` | Spectral regularity index | dimensionless | Pattern regularity |

## Usage Examples

### Single File Analysis
```python
from src.core.enhanced_ml_peak_analyzer import EnhancedMLPeakAnalyzer
from src.utils.enhanced_baseline_detection import EnhancedBaselineDetector

# Initialize analyzer
analyzer = EnhancedMLPeakAnalyzer()

# Load and analyze spectrum
spectrum_data = analyzer.load_spectrum('path/to/spectrum.csv')
features = analyzer.extract_features(spectrum_data)

# Save results
analyzer.save_features(features, 'output_features.csv')
analyzer.save_plots('output_plots/')
```

### Batch Processing
```python
from src.core.batch_ml_analysis import BatchMLAnalysis

# Initialize batch processor
batch_processor = BatchMLAnalysis()

# Process all spectra in directory
batch_processor.process_directory('input_spectra/')

# Results saved to comprehensive_analysis_results/
```

### Baseline Detection
```python
from src.utils.enhanced_baseline_detection import EnhancedBaselineDetector

# Initialize detector
detector = EnhancedBaselineDetector()

# Detect baseline with physics-aware method
baseline = detector.detect_baseline_physics_aware(spectrum_data)

# Compare multiple methods
comparison = detector.compare_methods(spectrum_data)
```

## Configuration

The system uses configuration files in `src/config/` to control analysis parameters:

- **Peak Detection**: Sensitivity, minimum height, width constraints
- **Baseline Detection**: Algorithm selection, parameter optimization
- **Energy Ranges**: Analysis boundaries and region definitions
- **Output Settings**: Plot styles, file formats, directory structure


