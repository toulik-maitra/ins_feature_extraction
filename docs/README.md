# INS Spectrum ML Analysis Workflow

A comprehensive, automated workflow for analyzing INS (Inelastic Neutron Scattering) spectra and extracting ML-ready features.

## 🚀 Quick Start

### Single File Analysis
```bash
python3 batch_ml_analysis.py --file "path/to/spectrum.csv"
```

### Batch Directory Analysis
```bash
python3 batch_ml_analysis.py --directory "path/to/spectra/"
```

### Demo (5 files)
```bash
python3 run_batch_demo.py
```

## 📁 Organized Output Structure

The workflow creates a well-organized output directory:

```
ml_analysis_results/
├── features/                    # Individual and combined feature files
│   ├── molecule1_features.csv
│   ├── molecule2_features.csv
│   ├── all_molecules_features.csv    # Combined features
│   └── ml_dataset.csv               # ML-ready dataset
├── plots/                      # Publication-quality plots
│   ├── molecule1_analysis.pdf
│   └── molecule2_analysis.pdf
├── summaries/                  # Analysis summaries
│   ├── analysis_summary.csv    # Per-file results
│   └── statistical_summary.csv # Overall statistics
└── logs/                       # Processing logs
```

## 🔧 Configuration

Edit `analysis_config.py` to customize:

- **Peak Detection Sensitivity**: Adjust `prominence`, `distance`, `width`
- **Fitting Parameters**: Modify smoothing, iterations, bounds
- **Output Settings**: Change DPI, figure size, format
- **Quality Thresholds**: Set minimum R², maximum RMSE

## 📊 Extracted ML Features

### Peak Count Features
- `num_peaks`: Total detected peaks
- `peak_density`: Peaks per unit energy

### Amplitude Features
- `mean_amplitude`, `std_amplitude`: Statistical measures
- `max_amplitude`, `min_amplitude`: Range analysis
- `amplitude_cv`: Coefficient of variation
- `amplitude_skewness`, `amplitude_kurtosis`: Distribution shape

### Width Features
- `mean_fwhm`, `std_fwhm`: Full-width half-maximum statistics
- `fwhm_range`, `fwhm_cv`: Width variability
- `fwhm_skewness`, `fwhm_kurtosis`: Width distribution shape

### Area Features
- `total_area`: Sum of all peak areas
- `mean_area`, `std_area`: Area statistics
- `area_cv`: Area coefficient of variation

### Position Features
- `mean_center`, `std_center`: Peak position statistics
- `energy_span`: Total energy range covered
- `mean_peak_spacing`: Average distance between peaks

### Energy Region Features
- `low_energy_peaks`: Peaks below 1000 cm⁻¹
- `mid_energy_peaks`: Peaks 1000-2000 cm⁻¹
- `high_energy_peaks`: Peaks above 2000 cm⁻¹

### Fit Quality Features
- `r_squared`: Goodness of fit
- `rmse`: Root mean square error
- `baseline`: Baseline offset

## 🤖 ML Integration

### Load ML Dataset
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the ML-ready dataset
features_df = pd.read_csv("ml_analysis_results/features/ml_dataset.csv")

# Prepare for ML (remove any non-numeric columns)
X = features_df.select_dtypes(include=[np.number])
y = features_df['target_column']  # Your target variable

# Split for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Classification Example
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
importances = clf.feature_importances_
feature_names = X.columns
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")
```

### Regression Example
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Train regressor
reg = LinearRegression()
reg.fit(X_train, y_train)

# Evaluate
y_pred = reg.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
```

## 📈 Advanced Usage

### Custom Peak Detection
```python
from ml_peak_analyzer import MLPeakAnalyzer

analyzer = MLPeakAnalyzer(energy_range=(0, 3500))
analyzer.load_spectrum_data("spectrum.csv", skiprows=0, energy_col="x", intensity_col="y")

# High sensitivity for shoulders
peaks = analyzer.detect_peaks_from_spectrum(
    prominence=0.005,  # Very sensitive
    distance=2,        # Close peaks
    width=1           # Narrow peaks
)
```

### Batch Processing with Custom Settings
```python
from batch_ml_analysis import BatchMLAnalyzer

# Initialize with custom output directory
analyzer = BatchMLAnalyzer(output_dir="my_analysis_results")

# Process directory with individual plots
analyzer.analyze_directory(
    "path/to/spectra/",
    plot_individual=True,  # Create individual plots
    file_pattern="*.csv"
)
```

### Quality Control
```python
# Filter results by quality
summary_df = pd.read_csv("ml_analysis_results/summaries/analysis_summary.csv")
good_fits = summary_df[summary_df['r_squared'] > 0.9]
print(f"High quality fits: {len(good_fits)}/{len(summary_df)}")
```

## 🔍 Troubleshooting

### Common Issues

1. **Low R² values**: 
   - Increase `smooth_window` in fitting
   - Adjust peak detection sensitivity
   - Check data quality

2. **Too many/few peaks**:
   - Adjust `prominence` parameter
   - Modify `distance` between peaks
   - Change `width` requirements

3. **Fitting failures**:
   - Check data format (x,y columns)
   - Verify energy range
   - Increase `max_iterations`

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with detailed output
analyzer = MLPeakAnalyzer()
analyzer.load_spectrum_data("spectrum.csv")
analyzer.detect_peaks_from_spectrum(plot_detection=True)  # Visual confirmation
```

## 📊 Performance

- **Speed**: ~2-5 seconds per spectrum
- **Memory**: Efficient for 1000+ spectra
- **Accuracy**: R² > 0.95 typical for good data
- **Scalability**: Handles directories with 1000+ files

## 🎯 Use Cases

### Molecular Classification
- Distinguish between different molecular structures
- Identify functional groups
- Classify crystal phases

### Property Prediction
- Predict molecular properties from spectral features
- Correlate spectral patterns with physical properties
- Quantitative structure-property relationships

### Quality Control
- Identify anomalous spectra
- Monitor experimental conditions
- Validate data quality

### Research Applications
- High-throughput screening
- Structure-property relationships
- Material characterization

## 📚 Citation

If you use this workflow in your research:

```bibtex
@software{ins_ml_workflow,
  title={INS Spectrum ML Analysis Workflow},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/ins-ml-workflow}
}
```

---

**Built for Science, Optimized for ML** 🧬🔬🤖 