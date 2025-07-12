# Quick Start Guide

## Installation

### 1. Clone or Download the Repository
```bash
git clone https://github.com/your-repo/ins-ml-analysis-system.git
cd ins-ml-analysis-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import numpy, scipy, pandas, matplotlib, sklearn; print('All dependencies installed successfully!')"
```

## Basic Usage

### Single File Analysis

1. **Prepare your data**: Ensure your CSV file has columns named 'x' (energy) and 'y' (intensity)

2. **Run analysis**:
```bash
python src/core/batch_ml_analysis.py --file "path/to/your/spectrum.csv"
```

3. **Check results**: Look in the `ml_analysis_results/` folder for:
   - `features/`: Extracted ML features
   - `plots/`: Publication-quality plots
   - `summaries/`: Analysis summaries

### Batch Directory Analysis

1. **Organize your data**: Place all CSV files in a directory

2. **Run batch analysis**:
```bash
python src/core/batch_ml_analysis.py --directory "path/to/your/spectra/"
```

3. **Review results**: Check the organized output structure

## Example Workflows

### Example 1: Single Spectrum Analysis
```python
from src.core.ml_peak_analyzer import MLPeakAnalyzer

# Initialize analyzer
analyzer = MLPeakAnalyzer(energy_range=(0, 3500))

# Load data
analyzer.load_spectrum_data("spectrum.csv", skiprows=0, energy_col="x", intensity_col="y")

# Detect peaks
peaks = analyzer.detect_peaks_from_spectrum(prominence=0.01)

# Fit Gaussians
fit_results = analyzer.fit_global_gaussians()

# Extract features
features = analyzer.extract_ml_features()

# Create plot
analyzer.plot_publication_quality(save_path="analysis.pdf")

# Print summary
analyzer.print_summary()
```

### Example 2: Batch Processing
```python
from src.core.batch_ml_analysis import BatchMLAnalyzer

# Initialize batch analyzer
batch_analyzer = BatchMLAnalyzer(output_dir="my_results")

# Process directory
batch_analyzer.analyze_directory("path/to/spectra/", plot_individual=True)

# Results are automatically organized in my_results/
```

### Example 3: ML Integration
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load features
features_df = pd.read_csv("ml_analysis_results/features/ml_dataset.csv")

# Prepare for ML
X = features_df.select_dtypes(include=[np.number])
y = features_df['target_column']  # Your target variable

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
```

## Data Format Requirements

### CSV File Format
Your CSV files should have the following structure:
```csv
x,y
0.0,0.1
1.0,0.2
2.0,0.3
...
```

Where:
- `x`: Energy values in cm⁻¹
- `y`: Intensity values in arbitrary units

### Alternative Column Names
If your data uses different column names, specify them:
```python
analyzer.load_spectrum_data("spectrum.csv", 
                           skiprows=0, 
                           energy_col="energy", 
                           intensity_col="intensity")
```

## Output Structure

After running analysis, you'll find:

```
ml_analysis_results/
├── features/                    # ML-ready features
│   ├── molecule1_features.csv
│   ├── molecule2_features.csv
│   ├── all_molecules_features.csv
│   └── ml_dataset.csv
├── plots/                      # Publication-quality plots
│   ├── molecule1_analysis.pdf
│   └── molecule2_analysis.pdf
├── summaries/                  # Analysis summaries
│   ├── analysis_summary.csv
│   └── statistical_summary.csv
└── logs/                       # Processing logs
```

## Key Features Extracted

The system extracts **50+ features** including:

- **Peak Count**: Number of peaks, peak density
- **Amplitude**: Mean, std, range, skewness, kurtosis
- **Width**: FWHM statistics and distributions
- **Area**: Peak areas, total spectral area, fractions
- **Position**: Peak positions, energy span, spacing
- **Quality**: R², RMSE, baseline

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
2. **File not found**: Check file paths and permissions
3. **Low R² values**: Adjust peak detection parameters
4. **Too many/few peaks**: Modify prominence and distance parameters

### Getting Help

- Check the full documentation in `docs/`
- Review example scripts in `examples/`
- Check the troubleshooting section in the main README

## Next Steps

1. **Explore examples**: Run the demo scripts in `examples/`
2. **Read documentation**: Check `docs/features.md` for detailed feature explanations
3. **Customize parameters**: Adjust peak detection and fitting parameters
4. **Integrate with ML**: Use the extracted features for your ML models

## Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation
- Review the example scripts 