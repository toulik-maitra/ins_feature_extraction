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
│   ├── ml_dataset_clean.csv          # Clean ML dataset (61 features, 56 samples)
│   ├── all_molecules_features.csv    # Complete dataset with all features
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

### Comprehensive Feature Extraction (61 ML-Ready Features)
- **Sample Identification**: Each row includes `molecule_name` for clear sample tracking
- **Peak Analysis**: Count, density, amplitude, width, area statistics
- **Baseline Analysis**: Detected baseline areas and signal quality metrics
- **Peak-to-Baseline Ratios**: With outlier removal (1st-99th percentile)
- **Energy Region Analysis**: Low/mid/high energy peak distributions
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

The system extracts 61 essential features from INS spectra, organized into the following categories:

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

