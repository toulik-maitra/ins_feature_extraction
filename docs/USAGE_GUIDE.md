# INS ML Analysis System - Usage Guide

This guide provides step-by-step instructions for using the INS ML Analysis System.

## 📋 Prerequisites

Before starting, ensure you have:
- Python 3.8+ installed
- Required dependencies: `pip install -r requirements.txt`
- Your INS spectra data in CSV format with columns: `x` (energy) and `y` (intensity)

## 🚀 Quick Start

### Step 1: Prepare Your Data

Ensure your CSV files have the correct format:
```csv
x,y
0.0,0.001
1.0,0.002
...
3500.0,0.001
```

### Step 2: Run Batch Analysis

```bash
# For a single file
python3 run_batch_analysis.py --file "path/to/spectrum.csv" --output "my_results" --plot-individual

# For a directory of files
python3 run_batch_analysis.py --directory "path/to/spectra/" --output "comprehensive_results" --plot-individual
```

### Step 3: Generate Clean ML Dataset

```bash
python3 create_clean_ml_dataset.py
```

## 📊 Understanding the Output

### Directory Structure
```
comprehensive_analysis_results/
├── features/
│   ├── ml_dataset_clean.csv          # ← Use this for ML
│   ├── all_molecules_features.csv    # Complete dataset
│   └── [molecule]_features.csv       # Individual files
├── plots/
│   ├── main_analysis/                # Main spectrum plots
│   ├── baseline_detection/           # Baseline plots
│   ├── peak_detection/               # Peak detection plots
│   └── kde_density/                  # Density plots
├── summaries/
│   └── analysis_summary.csv          # Summary statistics
└── logs/                             # Processing logs
```

### Key Files for ML Analysis

1. **`ml_dataset_clean.csv`** - Your main ML dataset
   - 151 features per sample (including 93 enhanced energy region features)
   - Includes sample identifiers (`molecule_name`)
   - Ready for machine learning

2. **`all_molecules_features.csv`** - Complete dataset
   - 202 total features including individual peak arrays
   - Enhanced energy region features with comprehensive statistics
   - Useful for detailed analysis

## 🚀 Enhanced Energy Region Features

### Overview
The system now provides **comprehensive energy region analysis** with 93 enhanced features across three energy regions:

- **Low Energy (0-500 cm⁻¹)**: 31 features for fundamental vibrations
- **Mid Energy (500-2000 cm⁻¹)**: 31 features for combination bands  
- **High Energy (2000-3500 cm⁻¹)**: 31 features for overtone vibrations

### Enhanced Feature Categories

**For Each Energy Region:**
- **Amplitude Features (11)**: Mean, std, max, min, median, skewness, kurtosis, CV, IQR, 25th/75th percentiles
- **Width Features (11)**: FWHM statistics with full distribution analysis
- **Area Features (11)**: Integrated intensities with comprehensive statistics

### Benefits
- **Better Structure Discrimination**: Detailed energy-dependent analysis
- **Improved Temperature Studies**: Enhanced sensitivity to temperature changes
- **Advanced ML Models**: 90+ new features for better predictive performance
- **Cross-Region Correlations**: Energy-dependent structural relationships

### Example Usage
```python
# Enhanced energy region features are automatically included
# in the feature extraction process
features = analyzer.extract_features(spectrum_data)

# Access enhanced energy region features
low_energy_features = [col for col in features.columns if 'low_energy' in col]
mid_energy_features = [col for col in features.columns if 'mid_energy' in col]
high_energy_features = [col for col in features.columns if 'high_energy' in col]

print(f"Low energy features: {len(low_energy_features)}")
print(f"Mid energy features: {len(mid_energy_features)}")
print(f"High energy features: {len(high_energy_features)}")
```

## 🔧 Advanced Configuration

### Custom Peak Detection Parameters

```python
from src.core.ml_peak_analyzer import MLPeakAnalyzer

analyzer = MLPeakAnalyzer(
    energy_range=(0, 3500),           # Analysis range
    baseline_detector_type='dynamic_rolling',  # Baseline method
    is_experimental=True              # Experimental data settings
)

# Custom peak detection
peaks = analyzer.detect_peaks_from_spectrum(
    prominence=0.005,    # Peak prominence threshold
    distance=2,          # Minimum distance between peaks
    width=1,            # Minimum peak width
    smooth_window=5     # Smoothing window
)
```

### Baseline Detection Methods

```python
from src.utils.baseline_detection import detect_baseline

# Available methods:
baseline_rolling = detect_baseline(x, y, method='dynamic_rolling')
baseline_poly = detect_baseline(x, y, method='polynomial')
baseline_stat = detect_baseline(x, y, method='statistical')
```

## 📈 ML Integration Examples

### Basic Classification

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load clean dataset
df = pd.read_csv("comprehensive_analysis_results/features/ml_dataset_clean.csv")

# Prepare features (exclude sample identifier)
X = df.drop('molecule_name', axis=1)
y = df['molecule_name']  # Or your target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Feature Importance Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

# Get feature importance
feature_importance = clf.feature_importances_
feature_names = X.columns

# Sort by importance
indices = np.argsort(feature_importance)[::-1]

# Plot top 20 features
plt.figure(figsize=(12, 8))
plt.title("Feature Importance")
plt.bar(range(20), feature_importance[indices[:20]])
plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=45)
plt.tight_layout()
plt.show()
```

### Clustering Analysis

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add clusters to dataframe
df['cluster'] = clusters
print(df.groupby('cluster')['molecule_name'].value_counts())
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Low R² Values (< 0.9)
**Symptoms**: Poor fit quality in main analysis plots
**Solutions**:
- Increase smoothing window: `smooth_window=7` or higher
- Adjust peak detection sensitivity: `prominence=0.01`
- Check data quality and format
- Verify energy range is 0-3500 cm⁻¹

#### 2. Too Many/Few Peaks
**Symptoms**: Unrealistic peak counts
**Solutions**:
- For too many peaks: Increase `prominence` (0.01-0.05)
- For too few peaks: Decrease `prominence` (0.001-0.01)
- Adjust `distance` between peaks (2-10)
- Use different settings for experimental vs simulated data

#### 3. Baseline Detection Issues
**Symptoms**: Poor baseline fits or missing baseline plots
**Solutions**:
- Try different baseline methods: `dynamic_rolling`, `polynomial`, `statistical`
- Check for data artifacts or noise
- Verify data preprocessing

#### 4. Memory Issues with Large Datasets
**Symptoms**: System crashes or slow performance
**Solutions**:
- Process files in smaller batches
- Use `plot_individual=False` for faster processing
- Increase system memory or use cloud computing

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with debug output
analyzer = MLPeakAnalyzer()
analyzer.load_spectrum_data("spectrum.csv")
analyzer.detect_peaks_from_spectrum(plot_detection=True)
analyzer.detect_baseline()
```

## 📊 Performance Optimization

### For Large Datasets (>1000 files)

1. **Batch Processing**:
```bash
# Process in smaller batches
python3 run_batch_analysis.py --directory "batch1/" --output "results1"
python3 run_batch_analysis.py --directory "batch2/" --output "results2"
```

2. **Parallel Processing**:
```python
from multiprocessing import Pool
import os

def process_file(filepath):
    # Your processing logic here
    pass

# Process files in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_file, file_list)
```

3. **Memory Management**:
```python
# Clear memory after processing each file
import gc

for file in files:
    analyzer = MLPeakAnalyzer()
    analyzer.load_spectrum_data(file)
    # Process file
    del analyzer
    gc.collect()
```

## 🎯 Best Practices

### Data Preparation
- Ensure consistent energy range (0-3500 cm⁻¹)
- Check for NaN or infinite values
- Normalize intensity if needed
- Use consistent file naming conventions

### Analysis Settings
- Use `is_experimental=True` for experimental data
- Use `is_experimental=False` for simulated data
- Adjust peak detection sensitivity based on data quality
- Use appropriate baseline detection method

### ML Pipeline
- Always scale features before ML
- Use cross-validation for robust evaluation
- Check for class imbalance in classification
- Monitor feature importance for interpretability

### Quality Control
- Review R² values for fit quality
- Check peak-to-baseline ratios for signal quality
- Verify peak counts are reasonable
- Examine baseline detection plots

## 📚 Additional Resources

- **API Documentation**: See `docs/api/` for detailed function documentation
- **Examples**: Check `examples/` for specific use cases
- **Tutorials**: Visit `docs/tutorials/` for step-by-step guides
- **GitHub Issues**: Report bugs or request features

## 🆘 Getting Help

If you encounter issues:

1. Check this usage guide
2. Review the troubleshooting section
3. Enable debug mode for detailed output
4. Check the example scripts
5. Open an issue on GitHub with:
   - Error message
   - Data format description
   - System specifications
   - Steps to reproduce 