# Enhanced Baseline Detection System

## Overview

The Enhanced Baseline Detection System provides a comprehensive solution for baseline correction in INS spectra with multiple algorithms, systematic parameter optimization, and validation capabilities. This system addresses the limitations of the original rolling minimum algorithm by offering scientifically justified alternatives with quantitative quality metrics.

## Key Features

### 🔬 Multiple Baseline Algorithms
- **PyBaselines Integration**: Access to the entire ALS family (asls, airpls, arpls, drpls, iarpls, aspls, psalsa)
- **SpectroChemPy Integration**: Polynomial, pchip, asls, and snip methods
- **Advanced Rolling Methods**: Adaptive, multi-scale, and percentile-based rolling baselines
- **Legacy Support**: Fallback to original dynamic rolling method

### ⚙️ Systematic Parameter Optimization
- **Automated Optimization**: Systematic parameter search with validation
- **Quality-Based Selection**: Objective metrics to select optimal parameters
- **Comprehensive Ranges**: Well-defined parameter ranges for each algorithm
- **Validation with Ground Truth**: Use of simulated data with known baselines

### 📊 Quantitative Quality Metrics
- **RMSE**: Root Mean Square Error between true and estimated baseline
- **Correlation**: Pearson correlation coefficient
- **Smoothness**: Baseline smoothness assessment
- **Peak Preservation**: How well peaks are preserved after correction
- **Combined Quality Score**: Weighted combination of all metrics

### 🔄 Integration with ML Pipeline
- **Seamless Integration**: Works with existing MLPeakAnalyzer
- **Enhanced Features**: Additional baseline-related features for ML
- **Comprehensive Reporting**: Detailed analysis reports with recommendations
- **Visualization**: Publication-quality plots and comparisons

## Installation

### Dependencies

The enhanced system requires additional dependencies beyond the base requirements:

```bash
pip install pybaselines>=0.6.0
pip install spectrochempy>=0.6.0
pip install lmfit>=1.2.0
```

### Verification

To verify the installation, run:

```python
from src.utils.enhanced_baseline_detection import EnhancedBaselineDetectorFactory

# Check available detectors
available = EnhancedBaselineDetectorFactory.get_available_detectors()
print(f"Available detectors: {available}")
```

## Usage

### Basic Usage

```python
from src.core.enhanced_ml_peak_analyzer import analyze_spectrum_with_enhanced_baseline

# Analyze spectrum with enhanced baseline detection
analyzer = analyze_spectrum_with_enhanced_baseline(
    filepath="your_spectrum.csv",
    baseline_detector_type="pybaselines_asls",
    enable_parameter_optimization=True,
    distance=3,
    prominence=0.01,
    width=1
)

# Extract enhanced features
features = analyzer.extract_enhanced_ml_features()

# Generate report
analyzer.generate_enhanced_report("analysis_report.md")

# Create visualization
analyzer.plot_enhanced_analysis("analysis_plot.pdf")
```

### Advanced Usage with Validation

```python
from src.utils.enhanced_baseline_detection import (
    detect_enhanced_baseline,
    BaselineValidationSystem
)

# Create validation data (if you have ground truth)
validation_data = {
    'intensity': your_spectrum_intensity,
    'energy': your_spectrum_energy,
    'true_baseline': known_baseline,
    'peak_positions': known_peak_positions
}

# Detect baseline with optimization
result = detect_enhanced_baseline(
    intensity=your_spectrum_intensity,
    energy=your_spectrum_energy,
    detector_type="pybaselines_asls",
    optimize_parameters=True,
    validation_data=validation_data
)

print(f"Baseline detected using: {result['detector_name']}")
print(f"Quality metrics: {result['quality_metrics']}")
```

### Parameter Optimization

```python
from src.utils.enhanced_baseline_detection import EnhancedBaselineDetectorFactory

# Create detector
detector = EnhancedBaselineDetectorFactory.create_detector("pybaselines_asls")

# Optimize parameters
optimization_result = detector.optimize_parameters(
    intensity=spectrum_intensity,
    energy=spectrum_energy,
    true_baseline=known_baseline,
    peak_positions=known_peak_positions,
    max_iterations=50
)

print(f"Best parameters: {optimization_result['best_parameters']}")
print(f"Best quality score: {optimization_result['best_score']}")
```

## Available Detectors

### PyBaselines Detectors

| Detector | Description | Key Parameters |
|----------|-------------|----------------|
| `pybaselines_asls` | Asymmetric Least Squares | `lambda_`, `p`, `diff_order` |
| `pybaselines_airpls` | Adaptive Iteratively Reweighted Penalized Least Squares | `lambda_`, `diff_order` |
| `pybaselines_arpls` | Asymmetric Reweighted Penalized Least Squares | `lambda_`, `diff_order` |
| `pybaselines_drpls` | Doubly Reweighted Penalized Least Squares | `lambda_`, `eta`, `diff_order` |
| `pybaselines_iarpls` | Improved Asymmetric Reweighted Penalized Least Squares | `lambda_`, `diff_order` |
| `pybaselines_aspls` | Adaptive Smoothness Penalized Least Squares | `lambda_`, `p`, `diff_order` |
| `pybaselines_psalsa` | Penalized Spline Approximation | `lambda_`, `p`, `diff_order` |

### SpectroChemPy Detectors

| Detector | Description | Key Parameters |
|----------|-------------|----------------|
| `spectrochempy_polynomial` | Polynomial baseline | `degree` |
| `spectrochempy_pchip` | Piecewise Cubic Hermite Interpolation | None |
| `spectrochempy_asls` | Asymmetric Least Squares | `lambda_`, `p` |
| `spectrochempy_snip` | Sensitive Nonlinear Iterative Peak clipping | None |

### Advanced Rolling Detectors

| Detector | Description | Key Parameters |
|----------|-------------|----------------|
| `advanced_rolling_adaptive` | Adaptive rolling based on local characteristics | `window_size`, `sensitivity` |
| `advanced_rolling_multi_scale` | Multi-scale rolling with weighted combination | `window_sizes`, `weights` |
| `advanced_rolling_percentile` | Percentile-based rolling | `window_size`, `percentile` |

## Quality Metrics

### RMSE (Root Mean Square Error)
- **Definition**: `sqrt(mean((true_baseline - estimated_baseline)²))`
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Measures the average deviation from the true baseline

### Correlation
- **Definition**: Pearson correlation coefficient between true and estimated baseline
- **Range**: -1 to 1 (higher is better)
- **Interpretation**: Measures how well the estimated baseline follows the true baseline trend

### Smoothness
- **Definition**: `mean(abs(diff(estimated_baseline)))`
- **Range**: 0 to ∞ (lower is better)
- **Interpretation**: Measures how smooth the estimated baseline is

### Peak Preservation
- **Definition**: Average ratio of corrected to original peak heights
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: Measures how well peaks are preserved after baseline correction

### Quality Score
- **Definition**: Combined metric: `RMSE + 0.1*smoothness - 0.5*correlation + 0.2*(1-peak_preservation)`
- **Range**: -∞ to ∞ (lower is better)
- **Interpretation**: Overall quality assessment

## Parameter Optimization

### Optimization Process

1. **Parameter Range Definition**: Each detector defines its parameter ranges
2. **Random Search**: Systematic random sampling within parameter ranges
3. **Quality Assessment**: Each parameter set is evaluated using quality metrics
4. **Best Selection**: Parameters with the best quality score are selected
5. **Validation**: Optimized parameters are validated on test data

### Example Optimization

```python
# Define validation data
validation_data = {
    'intensity': spectrum_intensity,
    'energy': spectrum_energy,
    'true_baseline': known_baseline,
    'peak_positions': known_peak_positions
}

# Optimize parameters
detector = EnhancedBaselineDetectorFactory.create_detector("pybaselines_asls")
result = detector.optimize_parameters(
    intensity=spectrum_intensity,
    energy=spectrum_energy,
    true_baseline=known_baseline,
    peak_positions=known_peak_positions,
    max_iterations=50
)

print(f"Best lambda_: {result['best_parameters']['lambda_']}")
print(f"Best p: {result['best_parameters']['p']}")
print(f"Best quality score: {result['best_score']}")
```

## Validation System

### Synthetic Data Generation

The validation system can create synthetic spectra with known baselines:

```python
from src.utils.enhanced_baseline_detection import BaselineValidationSystem

validation_system = BaselineValidationSystem()

# Create synthetic spectrum
energy = np.linspace(0, 3500, 1000)
peak_positions = [500, 800, 1200, 1500, 2000, 2500, 3000]
peak_amplitudes = [0.5, 0.8, 0.6, 1.2, 0.9, 0.7, 0.4]
peak_widths = [30, 25, 35, 40, 30, 35, 25]

spectrum, true_baseline = validation_system.create_synthetic_spectrum(
    energy, peak_positions, peak_amplitudes, peak_widths,
    baseline_type='polynomial', noise_level=0.01
)
```

### Detector Comparison

```python
# Compare multiple detectors
detectors = [
    EnhancedBaselineDetectorFactory.create_detector("pybaselines_asls"),
    EnhancedBaselineDetectorFactory.create_detector("pybaselines_airpls"),
    EnhancedBaselineDetectorFactory.create_detector("spectrochempy_polynomial")
]

comparison_results = validation_system.compare_detectors(
    detectors, energy, peak_positions, peak_amplitudes, peak_widths,
    baseline_types=['polynomial', 'exponential', 'linear'],
    noise_levels=[0.01, 0.05, 0.1]
)

print(comparison_results)
```

## Integration with ML Pipeline

### Enhanced Features

The enhanced system adds several new features to the ML analysis:

- `baseline_detector_used`: Name of the detector used
- `baseline_processing_time`: Processing time for baseline detection
- `baseline_rmse`: RMSE of baseline detection (if validation data available)
- `baseline_correlation`: Correlation of baseline detection
- `baseline_smoothness`: Smoothness of detected baseline
- `baseline_peak_preservation`: Peak preservation metric
- `baseline_quality_score`: Overall quality score
- `parameter_optimization_enabled`: Whether optimization was used

### Example ML Analysis

```python
from src.core.enhanced_ml_peak_analyzer import EnhancedMLPeakAnalyzer

# Create analyzer with enhanced baseline detection
analyzer = EnhancedMLPeakAnalyzer(
    baseline_detector_type="pybaselines_asls",
    enable_parameter_optimization=True,
    validation_data=validation_data
)

# Load and analyze spectrum
analyzer.load_spectrum_data("spectrum.csv")
analyzer.detect_enhanced_baseline()
analyzer.detect_peaks_from_spectrum()
analyzer.fit_global_gaussians()
features = analyzer.extract_enhanced_ml_features()

# Access enhanced features
print(f"Baseline detector: {features['baseline_detector_used']}")
print(f"Baseline RMSE: {features['baseline_rmse']}")
print(f"Baseline quality score: {features['baseline_quality_score']}")
```

## Best Practices

### 1. Algorithm Selection

**For High Accuracy:**
- Use `pybaselines_asls` or `pybaselines_airpls`
- Enable parameter optimization
- Use validation data if available

**For Fast Processing:**
- Use `spectrochempy_polynomial` or `advanced_rolling_percentile`
- Disable parameter optimization for speed

**For Balanced Performance:**
- Use `pybaselines_asls` with moderate parameter optimization
- Consider `spectrochempy_asls` for good balance

### 2. Parameter Optimization

- **Use validation data**: Always provide validation data when possible
- **Adequate iterations**: Use at least 30-50 iterations for reliable optimization
- **Monitor convergence**: Check optimization history for convergence
- **Validate results**: Test optimized parameters on independent data

### 3. Quality Assessment

- **Multiple metrics**: Don't rely on a single metric
- **Context matters**: Consider your specific application requirements
- **Visual inspection**: Always visually inspect baseline detection results
- **Peak preservation**: Ensure peaks are not over-corrected

### 4. Documentation

- **Record parameters**: Document all parameters used for baseline detection
- **Justify choices**: Be able to explain why specific algorithms and parameters were chosen
- **Report quality**: Include quality metrics in your analysis reports
- **Version control**: Track changes to baseline detection methods

## Troubleshooting

### Common Issues

**1. Import Errors**
```python
# If pybaselines is not available
try:
    from src.utils.enhanced_baseline_detection import detect_enhanced_baseline
except ImportError:
    print("Install pybaselines: pip install pybaselines")
```

**2. Parameter Optimization Fails**
```python
# Reduce iterations or check parameter ranges
result = detector.optimize_parameters(
    max_iterations=10,  # Reduce from default 50
    # ... other parameters
)
```

**3. Poor Baseline Quality**
```python
# Try different algorithms
detectors_to_try = [
    "pybaselines_asls",
    "pybaselines_airpls", 
    "spectrochempy_polynomial",
    "advanced_rolling_adaptive"
]

for detector_type in detectors_to_try:
    try:
        result = detect_enhanced_baseline(
            intensity, energy, detector_type=detector_type
        )
        print(f"{detector_type}: RMSE = {result['quality_metrics']['rmse']}")
    except Exception as e:
        print(f"{detector_type}: Failed - {e}")
```

### Performance Optimization

**1. Reduce Processing Time**
```python
# Use faster algorithms
fast_detectors = ["spectrochempy_polynomial", "advanced_rolling_percentile"]

# Disable parameter optimization
result = detect_enhanced_baseline(
    intensity, energy, 
    detector_type="spectrochempy_polynomial",
    optimize_parameters=False
)
```

**2. Memory Optimization**
```python
# Process data in chunks for large datasets
chunk_size = 1000
for i in range(0, len(energy), chunk_size):
    chunk_energy = energy[i:i+chunk_size]
    chunk_intensity = intensity[i:i+chunk_size]
    # Process chunk...
```

## Examples

### Complete Analysis Example

```python
import numpy as np
import pandas as pd
from src.core.enhanced_ml_peak_analyzer import analyze_spectrum_with_enhanced_baseline

# Load your data
data = pd.read_csv("your_spectrum.csv")
energy = data['energy'].values
intensity = data['intensity'].values

# Create validation data (if you have ground truth)
validation_data = {
    'intensity': intensity,
    'energy': energy,
    'true_baseline': known_baseline,  # Your known baseline
    'peak_positions': known_peaks     # Your known peak positions
}

# Analyze with enhanced baseline detection
analyzer = analyze_spectrum_with_enhanced_baseline(
    filepath="your_spectrum.csv",
    baseline_detector_type="pybaselines_asls",
    enable_parameter_optimization=True,
    validation_data=validation_data,
    distance=3,
    prominence=0.01,
    width=1
)

# Extract features
features = analyzer.extract_enhanced_ml_features()

# Generate comprehensive report
analyzer.generate_enhanced_report("enhanced_analysis_report.md")

# Create visualization
analyzer.plot_enhanced_analysis("enhanced_analysis_plot.pdf")

# Print key results
print(f"Baseline detector: {features['baseline_detector_used']}")
print(f"Baseline quality score: {features['baseline_quality_score']:.6f}")
print(f"Number of peaks: {features['num_peaks']}")
print(f"Fit R²: {features['fit_r_squared']:.4f}")
```

### Comparison Study Example

```python
from src.utils.enhanced_baseline_detection import (
    EnhancedBaselineDetectorFactory,
    BaselineValidationSystem
)

# Create validation system
validation_system = BaselineValidationSystem()

# Test multiple detectors
detectors_to_test = [
    "pybaselines_asls",
    "pybaselines_airpls",
    "spectrochempy_polynomial",
    "advanced_rolling_adaptive"
]

# Create synthetic test data
energy = np.linspace(0, 3500, 1000)
peak_positions = [500, 800, 1200, 1500, 2000, 2500, 3000]
peak_amplitudes = [0.5, 0.8, 0.6, 1.2, 0.9, 0.7, 0.4]
peak_widths = [30, 25, 35, 40, 30, 35, 25]

spectrum, true_baseline = validation_system.create_synthetic_spectrum(
    energy, peak_positions, peak_amplitudes, peak_widths,
    baseline_type='polynomial', noise_level=0.01
)

# Compare detectors
detectors = [EnhancedBaselineDetectorFactory.create_detector(dt) for dt in detectors_to_test]
comparison_results = validation_system.compare_detectors(
    detectors, energy, peak_positions, peak_amplitudes, peak_widths,
    baseline_types=['polynomial', 'exponential'],
    noise_levels=[0.01, 0.05]
)

# Print results
print("Detector Comparison Results:")
print(comparison_results.groupby('detector')['rmse_mean'].mean().sort_values())
```

## Conclusion

The Enhanced Baseline Detection System provides a scientifically rigorous approach to baseline correction with:

1. **Multiple Algorithm Options**: Choose from state-of-the-art baseline detection methods
2. **Systematic Parameter Optimization**: Avoid "magic numbers" with objective optimization
3. **Quantitative Quality Assessment**: Measure and compare baseline detection quality
4. **Comprehensive Validation**: Test algorithms on synthetic data with known ground truth
5. **Seamless Integration**: Works with existing ML analysis pipeline
6. **Complete Documentation**: Track and justify all baseline detection choices

This system addresses the limitations of the original rolling minimum approach by providing scientifically justified alternatives with quantitative validation, ensuring that baseline correction decisions are transparent, reproducible, and defensible. 