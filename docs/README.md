# INS Spectrum ML Feature Extraction Workflow

A comprehensive, automated workflow for analyzing INS (Inelastic Neutron Scattering) spectra and extracting ML-ready features. This tool was designed to bridge the gap between raw spectral data and machine learning applications, making it easier to extract meaningful features from your INS spectra.

## Getting Started

### Single File Analysis
If you have just one spectrum to analyze, you can run:
```bash
python3 batch_ml_analysis.py --file "path/to/spectrum.csv"
```

### Batch Directory Analysis
For processing multiple files at once, use:
```bash
python3 batch_ml_analysis.py --directory "path/to/spectra/"
```

### Try the Demo
To see how it works with sample data, run:
```bash
python3 run_batch_demo.py
```

## Output Organization

The workflow organizes your results in a clean, logical structure that makes it easy to find what you need:

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

## Configuration Options

You can customize the analysis by editing `analysis_config.py`. Here's what you can adjust:

- **Peak Detection Sensitivity**: Fine-tune `prominence`, `distance`, and `width` parameters to match your data characteristics
- **Fitting Parameters**: Modify smoothing, iterations, and bounds to improve convergence
- **Output Settings**: Change DPI, figure size, and format to match your publication requirements
- **Quality Thresholds**: Set minimum R² and maximum RMSE values to filter results

## What Features Are Extracted

The workflow extracts a comprehensive set of features that capture different aspects of your spectral data:

### Peak Count Features
- `num_peaks`: Total number of detected peaks
- `peak_density`: Number of peaks per unit energy

### Amplitude Features
- `mean_amplitude`, `std_amplitude`: Statistical measures of peak heights
- `max_amplitude`, `min_amplitude`: Range analysis of peak intensities
- `amplitude_cv`: Coefficient of variation for amplitude consistency
- `amplitude_skewness`, `amplitude_kurtosis`: Distribution shape characteristics

### Width Features
- `mean_fwhm`, `std_fwhm`: Full-width half-maximum statistics
- `fwhm_range`, `fwhm_cv`: Width variability measures
- `fwhm_skewness`, `fwhm_kurtosis`: Width distribution shape

### Area Features
- `total_area`: Sum of all peak areas
- `mean_area`, `std_area`: Area statistics
- `area_cv`: Area coefficient of variation

### Position Features
- `mean_center`, `std_center`: Peak position statistics
- `energy_span`: Total energy range covered by peaks
- `mean_peak_spacing`: Average distance between adjacent peaks

### Energy Region Features
- `low_energy_peaks`: Peaks below 500 cm⁻¹
- `mid_energy_peaks`: Peaks between 500-2000 cm⁻¹
- `high_energy_peaks`: Peaks above 2000 cm⁻¹

### Fit Quality Features
- `r_squared`: Goodness of fit measure
- `rmse`: Root mean square error
- `baseline`: Baseline offset value


## Troubleshooting

### Common Issues and Solutions

1. **Low R² values**: 
   - Try increasing the `smooth_window` in fitting parameters
   - Adjust peak detection sensitivity
   - Check your data quality - noisy data can cause poor fits

2. **Too many or too few peaks**:
   - Adjust the `prominence` parameter - higher values detect fewer peaks
   - Modify the `distance` between peaks
   - Change the `width` requirements

3. **Fitting failures**:
   - Check your data format (make sure you have x,y columns)
   - Verify the energy range matches your data
   - Increase `max_iterations` if the fit doesn't converge

### Debug Mode
When things aren't working as expected, enable debug mode:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with detailed output
analyzer = MLPeakAnalyzer()
analyzer.load_spectrum_data("spectrum.csv")
analyzer.detect_peaks_from_spectrum(plot_detection=True)  # Visual confirmation
```

## Performance Characteristics

- **Speed**: Typically 2-5 seconds per spectrum
- **Memory**: Efficient for processing 1000+ spectra
- **Accuracy**: R² > 0.95 is typical for good quality data
- **Scalability**: Handles directories with 1000+ files without issues



