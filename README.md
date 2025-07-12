# INS Spectrum ML Feature Extraction

A comprehensive, automated workflow for analyzing Inelastic Neutron Scattering (INS) spectra and extracting machine learning-ready features. This system provides publication-quality analysis, robust peak detection, baseline detection, and extensive feature extraction optimized for ML applications.

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
python3 examples/single_file/test_ml_analyzer.py

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
│   │   ├── ml_peak_analyzer.py        # Main analyzer class
│   │   └── batch_ml_analysis.py       # Batch processing workflow
│   ├── utils/                         # Utility functions
│   │   ├── baseline_detection.py      # Baseline detection algorithms
│   │   └── run_single_INS_analysis.py # Single file runner
│   ├── visualization/                 # Plotting utilities
│   └── config/                        # Configuration files
│       ├── __init__.py
│       └── analysis_config.py         # Configuration settings
├── examples/                          # Example scripts
│   ├── single_file/                   # Single file analysis examples
│   │   ├── test_ml_analyzer.py        # Single file analysis example
│   │   └── test_baseline_features.py  # Baseline detection example
│   ├── batch_processing/              # Batch processing examples
│   │   ├── run_batch_analysis.py      # Main batch analysis script
│   │   └── run_batch_demo.py          # Batch processing demo
│   └── ml_integration/                # ML integration examples
│       ├── create_clean_ml_dataset.py # Clean ML dataset generator
│       └── ml_example.py              # ML integration example
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

## Output Organization

The system creates a comprehensive, organized output structure:

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

### Publication-Quality Visualization
- **Organized Plot Structure**: Separate directories for each plot type
- **High-Resolution Output**: PDF format for publication
- **Comprehensive Analysis**: 4 plot types per spectrum
- **Clean Layout**: Professional appearance with proper labels

## Comprehensive Feature Documentation

The system extracts **61 essential features** from INS spectra, organized into the following categories:

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
| `max_amplitude` | Maximum peak amplitude | a.u. | Strongest vibrational mode |
| `min_amplitude` | Minimum peak amplitude | a.u. | Weakest vibrational mode |
| `amplitude_range` | Range of amplitudes | a.u. | Spectral contrast |
| `amplitude_cv` | Coefficient of variation | dimensionless | Relative amplitude spread |
| `amplitude_skewness` | Distribution skewness | dimensionless | Amplitude asymmetry |
| `amplitude_kurtosis` | Distribution kurtosis | dimensionless | Amplitude peakedness |

### 4. Width Features (8 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `mean_fwhm` | Average peak width | cm⁻¹ | Typical vibrational broadening |
| `std_fwhm` | Standard deviation of widths | cm⁻¹ | Width variability |
| `max_fwhm` | Maximum peak width | cm⁻¹ | Broadest vibrational mode |
| `min_fwhm` | Minimum peak width | cm⁻¹ | Sharpest vibrational mode |
| `fwhm_range` | Range of peak widths | cm⁻¹ | Width diversity |
| `fwhm_cv` | Coefficient of variation | dimensionless | Relative width spread |
| `fwhm_skewness` | Distribution skewness | dimensionless | Width asymmetry |
| `fwhm_kurtosis` | Distribution kurtosis | dimensionless | Width peakedness |

### 5. Area Features (17 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `total_area` | Sum of all peak areas | a.u.·cm⁻¹ | Total spectral intensity |
| `mean_area` | Average peak area | a.u.·cm⁻¹ | Typical peak contribution |
| `std_area` | Standard deviation of areas | a.u.·cm⁻¹ | Area variability |
| `area_cv` | Coefficient of variation | dimensionless | Relative area spread |
| `max_area` | Maximum peak area | a.u.·cm⁻¹ | Dominant vibrational mode |
| `min_area` | Minimum peak area | a.u.·cm⁻¹ | Minor vibrational mode |
| `area_range` | Range of peak areas | a.u.·cm⁻¹ | Area diversity |
| `area_skewness` | Distribution skewness | dimensionless | Area asymmetry |
| `area_kurtosis` | Distribution kurtosis | dimensionless | Area peakedness |
| `largest_peak_area` | Largest individual peak area | a.u.·cm⁻¹ | Primary vibrational mode |
| `smallest_peak_area` | Smallest individual peak area | a.u.·cm⁻¹ | Minor vibrational mode |
| `largest_peak_area_fraction` | Fraction in largest peak | dimensionless | Primary mode dominance |
| `area_median` | Median peak area | a.u.·cm⁻¹ | Central tendency |
| `area_percentile_25` | 25th percentile area | a.u.·cm⁻¹ | Lower quartile |
| `area_percentile_75` | 75th percentile area | a.u.·cm⁻¹ | Upper quartile |
| `area_iqr` | Interquartile range | a.u.·cm⁻¹ | Area spread |
| `non_peak_area` | Area not in peaks | a.u.·cm⁻¹ | Background contribution |
| `non_peak_area_fraction` | Fraction in baseline | dimensionless | Background level |

### 6. Baseline Features (11 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `detected_baseline_area` | Area under detected baseline | a.u.·cm⁻¹ | Baseline contribution |
| `detected_baseline_area_fraction` | Baseline area fraction | dimensionless | Baseline level |
| `signal_above_baseline_area` | Area above baseline | a.u.·cm⁻¹ | Net signal intensity |
| `signal_above_baseline_fraction` | Signal fraction above baseline | dimensionless | Signal quality |
| `baseline` | Baseline offset | a.u. | Background level |

### 7. Peak-to-Baseline Ratio Features (7 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `mean_peak_to_baseline_ratio` | Average peak-to-baseline ratio | dimensionless | Signal quality indicator |
| `std_peak_to_baseline_ratio` | Standard deviation of ratios | dimensionless | Ratio variability |
| `max_peak_to_baseline_ratio` | Maximum peak-to-baseline ratio | dimensionless | Strongest signal peak |
| `min_peak_to_baseline_ratio` | Minimum peak-to-baseline ratio | dimensionless | Weakest signal peak |
| `median_peak_to_baseline_ratio` | Median peak-to-baseline ratio | dimensionless | Central ratio tendency |
| `peak_to_baseline_ratio_cv` | Coefficient of variation of ratios | dimensionless | Relative ratio spread |
| `num_ratio_outliers_removed` | Number of outliers removed | count | Data quality indicator |

### 8. Energy Region Features (3 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `low_energy_peaks` | Peaks below 1000 cm⁻¹ | count | Low-frequency modes |
| `mid_energy_peaks` | Peaks 1000-2000 cm⁻¹ | count | Mid-frequency modes |
| `high_energy_peaks` | Peaks above 2000 cm⁻¹ | count | High-frequency modes |

### 9. Peak Spacing Features (2 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `mean_peak_spacing` | Average distance between peaks | cm⁻¹ | Peak distribution |
| `std_peak_spacing` | Standard deviation of spacings | cm⁻¹ | Spacing regularity |

### 10. Fit Quality Features (3 features)
| Feature | Description | Units | ML Relevance |
|---------|-------------|-------|--------------|
| `r_squared` | R-squared value of fit | dimensionless | Fit quality |
| `rmse` | Root mean square error | a.u. | Fit accuracy |
| `baseline` | Baseline offset | a.u. | Background level |

## Usage Examples

### Single File Analysis
```python
from src.core.ml_peak_analyzer import MLPeakAnalyzer

# Initialize analyzer
analyzer = MLPeakAnalyzer()

# Analyze a single file
results = analyzer.analyze_file('path/to/spectrum.txt')
```

### Batch Analysis
```python
from src.core.batch_ml_analysis import BatchMLAnalyzer

# Initialize batch analyzer
batch_analyzer = BatchMLAnalyzer(output_dir="results")

# Analyze directory of spectra
batch_analyzer.analyze_directory("path/to/spectra/")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{ins_feature_extraction,
  title={INS Spectrum ML Feature Extraction},
  author={Toulik Maitra},
  year={2025},
  url={https://github.com/toulik-maitra/ins_feature_extraction.git}
}
```

