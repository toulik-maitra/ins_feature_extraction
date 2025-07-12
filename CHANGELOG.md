# Changelog

All notable changes to the INS ML Analysis System will be documented in this file.

## [2.0.0] - 2024-12-19

### 🎉 Major Release - Complete System Overhaul

#### ✨ Added
- **Baseline Detection System**
  - Multiple baseline detection algorithms (dynamic_rolling, polynomial, statistical)
  - Peak-to-baseline ratio calculations with outlier removal
  - Dedicated baseline detection plots
  - Baseline area and signal quality metrics

- **Instrument Peak Width Analysis**
  - Instrument resolution line: `0.25 + 0.005*energy + 0.0000001*(energy^2)`
  - Peak width vs energy plots with instrument comparison
  - Quality assessment relative to instrument resolution
  - Trend analysis between data and instrument lines

- **Clean ML Dataset Generation**
  - `ml_dataset_clean.csv` with 61 essential features
  - Sample identification with `molecule_name` column
  - Removal of unnecessary individual peak arrays
  - Organized feature categories for ML analysis

- **Organized Output Structure**
  - Structured directory organization for all outputs
  - Separate plot directories: main_analysis, baseline_detection, peak_detection, kde_density
  - Individual feature files for each spectrum
  - Comprehensive analysis summaries

- **Enhanced Feature Extraction**
  - Peak-to-baseline ratio features with outlier removal (1st-99th percentile)
  - Baseline area and signal quality features
  - Non-peak area calculations independent of baseline
  - Enhanced statistical features for all peak properties

#### 🔧 Changed
- **Spectral Area Calculations**
  - Total spectral area now calculated from raw spectrum data (independent of baseline)
  - Peak area fraction based on raw spectrum integration
  - Non-peak area calculated as total area minus peak areas
  - Baseline area features added separately for baseline analysis

- **Peak Detection Sensitivity**
  - Improved shoulder detection for experimental data
  - Configurable parameters for experimental vs simulated data
  - Enhanced prominence and distance settings
  - Better handling of close peaks and shoulders

- **Plot Organization**
  - All plots now saved in organized subdirectories
  - Publication-quality PDF output
  - Consistent naming conventions
  - Professional layout with proper labels

- **Error Handling**
  - Robust error handling for baseline detection
  - Graceful fallback for missing baseline detection
  - Better handling of edge cases in peak detection
  - Comprehensive logging of analysis steps

#### 🐛 Fixed
- **Import Issues**
  - Fixed module import paths for baseline detection
  - Corrected relative imports in batch analyzer
  - Resolved path issues for utility modules

- **Method Signature Errors**
  - Added `remove_ratio_outliers` parameter to `extract_ml_features`
  - Fixed missing `_create_baseline_plot` method
  - Added missing plotting methods in batch analyzer

- **Plot Generation Issues**
  - Fixed `plt.show()` causing interactive windows
  - Implemented proper plot saving with `plt.close()`
  - Resolved directory creation issues

- **Data Processing Issues**
  - Fixed Savitzky-Golay filter window size errors
  - Corrected energy range restrictions (0-3500 cm⁻¹)
  - Fixed baseline data initialization issues

#### 📊 Performance
- **Processing Speed**
  - Optimized peak detection algorithms
  - Improved memory management for large datasets
  - Better handling of batch processing

- **Memory Usage**
  - Efficient array operations with numpy
  - Reduced memory footprint for large datasets
  - Better garbage collection

#### 📚 Documentation
- **Comprehensive README Update**
  - Complete feature documentation (61 features)
  - Usage examples and best practices
  - Troubleshooting guide
  - ML integration examples

- **Technical Documentation**
  - Algorithm descriptions and implementation details
  - Performance metrics and optimization strategies
  - Scientific validation and accuracy metrics

- **Usage Guide**
  - Step-by-step instructions
  - Advanced configuration options
  - ML integration examples
  - Troubleshooting section

#### 🎯 ML Features
- **Sample Identification**
  - Clear molecule names for all samples
  - Easy tracking of which spectrum each row represents
  - Organized sample metadata

- **Feature Categories**
  - Core spectral: 7 features
  - Amplitude: 8 features
  - Width: 8 features
  - Area: 17 features
  - Baseline: 11 features
  - Peak-to-baseline ratios: 7 features
  - Energy regions: 3 features
  - Peak spacing: 2 features
  - Fit quality: 3 features

#### 🔬 Scientific Improvements
- **Baseline Analysis**
  - Multiple baseline detection methods
  - Signal quality assessment
  - Background contribution analysis
  - Peak-to-baseline ratio statistics

- **Instrument Analysis**
  - Comparison with instrument resolution
  - Quality assessment metrics
  - Trend analysis capabilities
  - Resolution-dependent analysis

- **Data Quality**
  - Outlier removal for ratio calculations
  - Comprehensive quality metrics
  - Robust error handling
  - Validation checks

## [1.0.0] - 2024-12-18

### 🎉 Initial Release

#### ✨ Added
- **Basic Peak Detection**
  - Gaussian peak fitting
  - Peak parameter extraction
  - Basic feature calculation

- **Feature Extraction**
  - Amplitude, width, and area features
  - Statistical moments and distributions
  - Energy region analysis

- **Basic Plotting**
  - Main analysis plots
  - Peak detection visualization
  - KDE density plots

- **Batch Processing**
  - Directory processing capabilities
  - Basic output organization
  - Individual file analysis

#### 🔧 Core Features
- Peak detection using SciPy
- Gaussian fitting with curve_fit
- Basic statistical feature extraction
- Simple plotting capabilities
- CSV output for features

#### 📊 Initial Features
- ~70 features per spectrum
- Basic peak analysis
- Simple area calculations
- Energy region features
- Fit quality metrics

---

## Versioning

This project uses [Semantic Versioning](http://semver.org/). For the versions available, see the [tags on this repository](https://github.com/your-repo/ins-ml-analysis-system/tags).

## Release Notes

### Version 2.0.0 Highlights
- **Major Feature Addition**: Baseline detection and analysis
- **ML Optimization**: Clean dataset with sample identification
- **Scientific Enhancement**: Instrument peak width analysis
- **User Experience**: Organized output structure and comprehensive documentation
- **Performance**: Improved processing speed and memory efficiency

### Migration from 1.0.0
- Update import statements for new module structure
- Use new `ml_dataset_clean.csv` for ML analysis
- Take advantage of baseline detection features
- Utilize organized plot structure for better analysis

### Breaking Changes
- Changed output directory structure
- Updated feature names and organization
- Modified plot file naming conventions
- Enhanced error handling and logging

### Deprecated Features
- Old individual peak array features (replaced with clean dataset)
- Simple baseline calculations (replaced with advanced baseline detection)
- Basic plotting (replaced with organized plot structure)

---

## Contributing

When contributing to this project, please update this changelog with a new entry under the appropriate version section. Follow the format:

```
### ✨ Added
- New feature description

### 🔧 Changed
- Changed feature description

### 🐛 Fixed
- Bug fix description
```

## Future Releases

### Planned for 2.1.0
- Advanced baseline methods using machine learning
- Non-Gaussian peak fitting capabilities
- Real-time processing features
- Enhanced visualization options

### Planned for 3.0.0
- Cloud-based analysis platform
- GPU acceleration for large datasets
- Advanced spectral decomposition
- Machine learning model integration 