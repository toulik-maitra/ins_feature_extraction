#!/usr/bin/env python3
"""
Run full INS spectrum analysis workflow on a single file.
"""
from ..core.ml_peak_analyzer import MLPeakAnalyzer

# Path to the INS spectrum file
spectrum_file = "/Users/toulikmaitra/Documents/UC Davis/1. Projects/Molecule Analysis/24- Structural Entropy/3- Anthracene/INS_spectra_all/xy_experimental_100_2.00.csv"

# 1. Initialize analyzer
analyzer = MLPeakAnalyzer(energy_range=(0, 3500))

# 2. Load spectrum (columns are 'x' and 'y')
analyzer.load_spectrum_data(
    spectrum_file,
    skiprows=0, energy_col="x", intensity_col="y"
)

# 3. Detect peaks (high sensitivity, plot for confirmation)
analyzer.detect_peaks_from_spectrum(
    height=None, distance=3, prominence=0.01, width=1, smooth_window=11, plot_detection=True
)

# 4. Fit global Gaussians
analyzer.fit_global_gaussians(smoothing=True, smooth_window=21)

# 5. Extract ML features
features = analyzer.extract_ml_features()

# 6. Print summary and plot
analyzer.print_summary()
analyzer.plot_publication_quality(save_path="INS_analysis_publication.pdf", dpi=300)

# 7. Save features
analyzer.save_features_to_csv("INS_ml_features.csv")

print("\nAnalysis complete. Check INS_analysis_publication.pdf and INS_ml_features.csv for results.") 