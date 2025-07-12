# Feature Documentation

## Overview

The INS ML Analysis System extracts **50+ features** from INS spectra, providing comprehensive characterization for machine learning applications. Each feature is designed to capture different aspects of molecular vibrational behavior and spectral characteristics.

## Feature Categories

### 1. Peak Count Features

#### `num_peaks`
- **Description**: Total number of detected peaks in the spectrum
- **Units**: count
- **Range**: 0 to ∞
- **Physical Meaning**: Indicates molecular complexity and number of distinct vibrational modes
- **ML Relevance**: 
  - Higher values suggest more complex molecular structures
  - Useful for molecular classification and complexity assessment
  - Correlates with molecular size and symmetry

#### `peak_density`
- **Description**: Number of peaks per unit energy
- **Units**: peaks/cm⁻¹
- **Calculation**: `num_peaks / (energy_range[1] - energy_range[0])`
- **Physical Meaning**: Spectral complexity density across the energy range
- **ML Relevance**:
  - Indicates how densely packed vibrational modes are
  - Useful for distinguishing between simple and complex molecules
  - Helps identify molecular symmetry and structure

### 2. Amplitude Features

#### `mean_amplitude`
- **Description**: Average amplitude of all detected peaks
- **Units**: a.u. (arbitrary units)
- **Calculation**: `np.mean(amplitudes)`
- **Physical Meaning**: Typical intensity of vibrational modes
- **ML Relevance**:
  - Indicates overall spectral intensity
  - Correlates with sample concentration and experimental conditions
  - Useful for normalization and quality control

#### `std_amplitude`
- **Description**: Standard deviation of peak amplitudes
- **Units**: a.u.
- **Calculation**: `np.std(amplitudes)`
- **Physical Meaning**: Variability in vibrational mode intensities
- **ML Relevance**:
  - Indicates spectral heterogeneity
  - Higher values suggest diverse vibrational modes
  - Useful for identifying molecular complexity

#### `max_amplitude`
- **Description**: Maximum peak amplitude
- **Units**: a.u.
- **Calculation**: `np.max(amplitudes)`
- **Physical Meaning**: Intensity of the strongest vibrational mode
- **ML Relevance**:
  - Identifies dominant vibrational modes
  - Useful for molecular fingerprinting
  - Correlates with functional group presence

#### `min_amplitude`
- **Description**: Minimum peak amplitude
- **Units**: a.u.
- **Calculation**: `np.min(amplitudes)`
- **Physical Meaning**: Intensity of the weakest vibrational mode
- **ML Relevance**:
  - Indicates detection sensitivity
  - Useful for quality assessment
  - Helps identify minor vibrational modes

#### `amplitude_range`
- **Description**: Range of peak amplitudes
- **Units**: a.u.
- **Calculation**: `max_amplitude - min_amplitude`
- **Physical Meaning**: Dynamic range of vibrational intensities
- **ML Relevance**:
  - Indicates spectral contrast
  - Higher values suggest diverse vibrational modes
  - Useful for molecular complexity assessment

#### `amplitude_cv`
- **Description**: Coefficient of variation of amplitudes
- **Units**: dimensionless
- **Calculation**: `std_amplitude / mean_amplitude`
- **Physical Meaning**: Relative variability in peak intensities
- **ML Relevance**:
  - Normalized measure of amplitude diversity
  - Useful for comparing different molecules
  - Indicates spectral homogeneity

#### `amplitude_skewness`
- **Description**: Skewness of amplitude distribution
- **Units**: dimensionless
- **Calculation**: `E[((X - μ) / σ)³]`
- **Physical Meaning**: Asymmetry of amplitude distribution
- **ML Relevance**:
  - Indicates whether most peaks are weak or strong
  - Useful for molecular classification
  - Correlates with molecular symmetry

#### `amplitude_kurtosis`
- **Description**: Kurtosis of amplitude distribution
- **Units**: dimensionless
- **Calculation**: `E[((X - μ) / σ)⁴] - 3`
- **Physical Meaning**: Peakedness of amplitude distribution
- **ML Relevance**:
  - Indicates concentration of amplitudes around mean
  - Useful for identifying characteristic patterns
  - Correlates with molecular structure

### 3. Width Features (FWHM - Full Width Half Maximum)

#### `mean_fwhm`
- **Description**: Average full-width half-maximum of peaks
- **Units**: cm⁻¹
- **Calculation**: `np.mean(fwhms)`
- **Physical Meaning**: Typical vibrational mode broadening
- **ML Relevance**:
  - Indicates molecular environment and interactions
  - Correlates with temperature and phase
  - Useful for material characterization

#### `std_fwhm`
- **Description**: Standard deviation of peak widths
- **Units**: cm⁻¹
- **Calculation**: `np.std(fwhms)`
- **Physical Meaning**: Variability in vibrational broadening
- **ML Relevance**:
  - Indicates heterogeneity in molecular environment
  - Useful for identifying mixed phases
  - Correlates with structural disorder

#### `max_fwhm`
- **Description**: Maximum peak width
- **Units**: cm⁻¹
- **Calculation**: `np.max(fwhms)`
- **Physical Meaning**: Broadest vibrational mode
- **ML Relevance**:
  - Indicates highly broadened modes
  - Useful for identifying specific interactions
  - Correlates with molecular dynamics

#### `min_fwhm`
- **Description**: Minimum peak width
- **Units**: cm⁻¹
- **Calculation**: `np.min(fwhms)`
- **Physical Meaning**: Sharpest vibrational mode
- **ML Relevance**:
  - Indicates well-defined vibrational modes
  - Useful for molecular fingerprinting
  - Correlates with molecular rigidity

#### `fwhm_range`
- **Description**: Range of peak widths
- **Units**: cm⁻¹
- **Calculation**: `max_fwhm - min_fwhm`
- **Physical Meaning**: Diversity in vibrational broadening
- **ML Relevance**:
  - Indicates variety of molecular environments
  - Useful for complexity assessment
  - Correlates with structural diversity

#### `fwhm_cv`
- **Description**: Coefficient of variation of widths
- **Units**: dimensionless
- **Calculation**: `std_fwhm / mean_fwhm`
- **Physical Meaning**: Relative variability in peak widths
- **ML Relevance**:
  - Normalized measure of width diversity
  - Useful for comparing different molecules
  - Indicates width homogeneity

#### `fwhm_skewness`
- **Description**: Skewness of width distribution
- **Units**: dimensionless
- **Calculation**: `E[((X - μ) / σ)³]`
- **Physical Meaning**: Asymmetry of width distribution
- **ML Relevance**:
  - Indicates whether most peaks are narrow or broad
  - Useful for molecular classification
  - Correlates with molecular environment

#### `fwhm_kurtosis`
- **Description**: Kurtosis of width distribution
- **Units**: dimensionless
- **Calculation**: `E[((X - μ) / σ)⁴] - 3`
- **Physical Meaning**: Peakedness of width distribution
- **ML Relevance**:
  - Indicates concentration of widths around mean
  - Useful for identifying characteristic patterns
  - Correlates with structural uniformity

### 4. Area Features

#### `total_area`
- **Description**: Sum of all peak areas
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.sum(areas)`
- **Physical Meaning**: Total integrated intensity of vibrational modes
- **ML Relevance**:
  - Indicates overall molecular vibrational activity
  - Correlates with sample concentration
  - Useful for normalization

#### `mean_area`
- **Description**: Average peak area
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.mean(areas)`
- **Physical Meaning**: Typical contribution of individual peaks
- **ML Relevance**:
  - Indicates typical vibrational mode strength
  - Useful for molecular characterization
  - Correlates with molecular structure

#### `std_area`
- **Description**: Standard deviation of peak areas
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.std(areas)`
- **Physical Meaning**: Variability in peak contributions
- **ML Relevance**:
  - Indicates area heterogeneity
  - Higher values suggest diverse vibrational modes
  - Useful for complexity assessment

#### `area_cv`
- **Description**: Coefficient of variation of areas
- **Units**: dimensionless
- **Calculation**: `std_area / mean_area`
- **Physical Meaning**: Relative variability in peak areas
- **ML Relevance**:
  - Normalized measure of area diversity
  - Useful for comparing different molecules
  - Indicates area homogeneity

#### `max_area`
- **Description**: Maximum peak area
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.max(areas)`
- **Physical Meaning**: Area of the most intense vibrational mode
- **ML Relevance**:
  - Identifies dominant vibrational modes
  - Useful for molecular fingerprinting
  - Correlates with functional group presence

#### `min_area`
- **Description**: Minimum peak area
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.min(areas)`
- **Physical Meaning**: Area of the weakest vibrational mode
- **ML Relevance**:
  - Indicates detection sensitivity
  - Useful for quality assessment
  - Helps identify minor vibrational modes

#### `area_range`
- **Description**: Range of peak areas
- **Units**: a.u.·cm⁻¹
- **Calculation**: `max_area - min_area`
- **Physical Meaning**: Dynamic range of peak contributions
- **ML Relevance**:
  - Indicates area contrast
  - Higher values suggest diverse vibrational modes
  - Useful for complexity assessment

#### `area_skewness`
- **Description**: Skewness of area distribution
- **Units**: dimensionless
- **Calculation**: `E[((X - μ) / σ)³]`
- **Physical Meaning**: Asymmetry of area distribution
- **ML Relevance**:
  - Indicates whether most peaks are weak or strong
  - Useful for molecular classification
  - Correlates with molecular symmetry

#### `area_kurtosis`
- **Description**: Kurtosis of area distribution
- **Units**: dimensionless
- **Calculation**: `E[((X - μ) / σ)⁴] - 3`
- **Physical Meaning**: Peakedness of area distribution
- **ML Relevance**:
  - Indicates concentration of areas around mean
  - Useful for identifying characteristic patterns
  - Correlates with molecular structure

#### `largest_peak_area`
- **Description**: Largest individual peak area
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.max(areas)`
- **Physical Meaning**: Contribution of the primary vibrational mode
- **ML Relevance**:
  - Identifies the most important vibrational mode
  - Useful for molecular fingerprinting
  - Correlates with dominant functional groups

#### `smallest_peak_area`
- **Description**: Smallest individual peak area
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.min(areas)`
- **Physical Meaning**: Contribution of the weakest vibrational mode
- **ML Relevance**:
  - Indicates detection sensitivity
  - Useful for quality assessment
  - Helps identify minor vibrational modes

#### `area_median`
- **Description**: Median peak area
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.median(areas)`
- **Physical Meaning**: Central tendency of peak areas
- **ML Relevance**:
  - Robust measure of typical peak contribution
  - Less sensitive to outliers than mean
  - Useful for molecular characterization

#### `area_percentile_25`
- **Description**: 25th percentile of peak areas
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.percentile(areas, 25)`
- **Physical Meaning**: Lower quartile of peak contributions
- **ML Relevance**:
  - Indicates lower range of vibrational mode strengths
  - Useful for identifying weak modes
  - Helps assess spectral quality

#### `area_percentile_75`
- **Description**: 75th percentile of peak areas
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.percentile(areas, 75)`
- **Physical Meaning**: Upper quartile of peak contributions
- **ML Relevance**:
  - Indicates upper range of vibrational mode strengths
  - Useful for identifying strong modes
  - Helps assess spectral intensity

#### `area_iqr`
- **Description**: Interquartile range of peak areas
- **Units**: a.u.·cm⁻¹
- **Calculation**: `area_percentile_75 - area_percentile_25`
- **Physical Meaning**: Spread of middle 50% of peak areas
- **ML Relevance**:
  - Robust measure of area variability
  - Less sensitive to outliers than range
  - Useful for molecular characterization

### 5. Spectral Area Features

#### `total_spectral_area`
- **Description**: Total integrated area of the entire spectrum
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.trapz(y, x)` (numerical integration)
- **Physical Meaning**: Complete spectral intensity including baseline
- **ML Relevance**:
  - Indicates total spectral activity
  - Useful for normalization
  - Correlates with sample concentration

#### `peak_area_fraction`
- **Description**: Fraction of spectrum area contained in peaks
- **Units**: dimensionless
- **Calculation**: `total_area / total_spectral_area`
- **Physical Meaning**: Proportion of spectrum in vibrational modes
- **ML Relevance**:
  - Indicates peak dominance
  - Higher values suggest well-defined vibrational modes
  - Useful for quality assessment

#### `baseline_area`
- **Description**: Area not contained in peaks
- **Units**: a.u.·cm⁻¹
- **Calculation**: `total_spectral_area - total_area`
- **Physical Meaning**: Background contribution to spectrum
- **ML Relevance**:
  - Indicates baseline level
  - Useful for quality assessment
  - Correlates with experimental conditions

#### `baseline_area_fraction`
- **Description**: Fraction of spectrum area in baseline
- **Units**: dimensionless
- **Calculation**: `baseline_area / total_spectral_area`
- **Physical Meaning**: Proportion of spectrum in background
- **ML Relevance**:
  - Indicates baseline dominance
  - Lower values suggest better quality data
  - Useful for quality control

#### `largest_peak_area_fraction`
- **Description**: Fraction of spectrum area in largest peak
- **Units**: dimensionless
- **Calculation**: `largest_peak_area / total_spectral_area`
- **Physical Meaning**: Dominance of primary vibrational mode
- **ML Relevance**:
  - Indicates primary mode importance
  - Useful for molecular fingerprinting
  - Correlates with dominant functional groups

### 6. Position Features

#### `mean_center`
- **Description**: Average peak position
- **Units**: cm⁻¹
- **Calculation**: `np.mean(centers)`
- **Physical Meaning**: Central energy region of vibrational modes
- **ML Relevance**:
  - Indicates typical vibrational energy
  - Useful for molecular classification
  - Correlates with molecular structure

#### `std_center`
- **Description**: Standard deviation of peak positions
- **Units**: cm⁻¹
- **Calculation**: `np.std(centers)`
- **Physical Meaning**: Spread of vibrational energies
- **ML Relevance**:
  - Indicates energy diversity
  - Higher values suggest diverse vibrational modes
  - Useful for complexity assessment

#### `center_range`
- **Description**: Range of peak positions
- **Units**: cm⁻¹
- **Calculation**: `np.max(centers) - np.min(centers)`
- **Physical Meaning**: Energy coverage of vibrational modes
- **ML Relevance**:
  - Indicates spectral breadth
  - Useful for molecular characterization
  - Correlates with molecular complexity

#### `energy_span`
- **Description**: Total energy range covered by peaks
- **Units**: cm⁻¹
- **Calculation**: `np.max(centers) - np.min(centers)`
- **Physical Meaning**: Spectral breadth of vibrational modes
- **ML Relevance**:
  - Indicates molecular vibrational diversity
  - Useful for complexity assessment
  - Correlates with molecular size and structure

#### `mean_peak_spacing`
- **Description**: Average distance between adjacent peaks
- **Units**: cm⁻¹
- **Calculation**: `np.mean(np.diff(sorted_centers))`
- **Physical Meaning**: Typical separation between vibrational modes
- **ML Relevance**:
  - Indicates peak distribution
  - Useful for molecular classification
  - Correlates with molecular symmetry

#### `std_peak_spacing`
- **Description**: Standard deviation of peak spacings
- **Units**: cm⁻¹
- **Calculation**: `np.std(np.diff(sorted_centers))`
- **Physical Meaning**: Regularity of peak distribution
- **ML Relevance**:
  - Indicates spacing uniformity
  - Lower values suggest regular patterns
  - Useful for molecular classification

### 7. Energy Region Features

#### `low_energy_peaks`
- **Description**: Number of peaks below 1000 cm⁻¹
- **Units**: count
- **Calculation**: `len([c for c in centers if c < 1000])`
- **Physical Meaning**: Low-frequency vibrational modes
- **ML Relevance**:
  - Indicates molecular framework vibrations
  - Useful for molecular classification
  - Correlates with molecular size and structure

#### `mid_energy_peaks`
- **Description**: Number of peaks between 1000-2000 cm⁻¹
- **Units**: count
- **Calculation**: `len([c for c in centers if 1000 <= c < 2000])`
- **Physical Meaning**: Mid-frequency vibrational modes
- **ML Relevance**:
  - Indicates functional group vibrations
  - Useful for molecular fingerprinting
  - Correlates with chemical composition

#### `high_energy_peaks`
- **Description**: Number of peaks above 2000 cm⁻¹
- **Units**: count
- **Calculation**: `len([c for c in centers if c >= 2000])`
- **Physical Meaning**: High-frequency vibrational modes
- **ML Relevance**:
  - Indicates bond stretching vibrations
  - Useful for molecular fingerprinting
  - Correlates with specific functional groups

### 8. Fit Quality Features

#### `r_squared`
- **Description**: Coefficient of determination (goodness of fit)
- **Units**: dimensionless
- **Range**: 0 to 1
- **Calculation**: `1 - (SS_res / SS_tot)`
- **Physical Meaning**: How well the Gaussian model fits the data
- **ML Relevance**:
  - Indicates fit quality
  - Higher values suggest better fits
  - Useful for quality control

#### `rmse`
- **Description**: Root mean square error
- **Units**: a.u.
- **Calculation**: `sqrt(mean((y - y_fit)²))`
- **Physical Meaning**: Average deviation from the fit
- **ML Relevance**:
  - Indicates fit accuracy
  - Lower values suggest better fits
  - Useful for quality assessment

#### `baseline`
- **Description**: Baseline offset from fitting
- **Units**: a.u.
- **Calculation**: Fitted baseline parameter
- **Physical Meaning**: Background level in the spectrum
- **ML Relevance**:
  - Indicates baseline level
  - Useful for quality assessment
  - Correlates with experimental conditions

### 9. Individual Peak Features (Arrays)

#### `peak_centers`
- **Description**: Individual peak positions
- **Units**: cm⁻¹
- **Type**: numpy array
- **Physical Meaning**: Energy positions of each vibrational mode
- **ML Relevance**:
  - Provides detailed peak analysis
  - Useful for molecular fingerprinting
  - Enables pattern recognition

#### `peak_amplitudes`
- **Description**: Individual peak amplitudes
- **Units**: a.u.
- **Type**: numpy array
- **Physical Meaning**: Intensity of each vibrational mode
- **ML Relevance**:
  - Provides detailed amplitude analysis
  - Useful for molecular characterization
  - Enables intensity pattern analysis

#### `peak_fwhms`
- **Description**: Individual peak widths
- **Units**: cm⁻¹
- **Type**: numpy array
- **Physical Meaning**: Width of each vibrational mode
- **ML Relevance**:
  - Provides detailed width analysis
  - Useful for molecular characterization
  - Enables broadening pattern analysis

#### `peak_areas`
- **Description**: Individual peak areas
- **Units**: a.u.·cm⁻¹
- **Type**: numpy array
- **Physical Meaning**: Integrated intensity of each vibrational mode
- **ML Relevance**:
  - Provides detailed area analysis
  - Useful for molecular characterization
  - Enables contribution pattern analysis

### 10. Peak-to-Baseline Ratio Features

#### `mean_peak_to_baseline_ratio`
- **Description**: Average peak-to-baseline ratio
- **Units**: dimensionless
- **Calculation**: `np.mean(peak_amplitudes / baseline_at_peaks)`
- **Physical Meaning**: Typical signal quality relative to background
- **ML Relevance**:
  - Indicates overall signal quality
  - Higher values suggest better signal-to-noise
  - Useful for quality assessment and classification

#### `std_peak_to_baseline_ratio`
- **Description**: Standard deviation of peak-to-baseline ratios
- **Units**: dimensionless
- **Calculation**: `np.std(peak_amplitudes / baseline_at_peaks)`
- **Physical Meaning**: Variability in signal quality across peaks
- **ML Relevance**:
  - Indicates consistency of signal quality
  - Lower values suggest uniform signal quality
  - Useful for quality control

#### `max_peak_to_baseline_ratio`
- **Description**: Maximum peak-to-baseline ratio
- **Units**: dimensionless
- **Calculation**: `np.max(peak_amplitudes / baseline_at_peaks)`
- **Physical Meaning**: Strongest signal peak relative to background
- **ML Relevance**:
  - Identifies dominant vibrational modes
  - Useful for molecular fingerprinting
  - Correlates with functional group presence

#### `min_peak_to_baseline_ratio`
- **Description**: Minimum peak-to-baseline ratio
- **Units**: dimensionless
- **Calculation**: `np.min(peak_amplitudes / baseline_at_peaks)`
- **Physical Meaning**: Weakest signal peak relative to background
- **ML Relevance**:
  - Indicates detection sensitivity
  - Useful for quality assessment
  - Helps identify minor vibrational modes

#### `peak_to_baseline_ratio_range`
- **Description**: Range of peak-to-baseline ratios
- **Units**: dimensionless
- **Calculation**: `max_ratio - min_ratio`
- **Physical Meaning**: Dynamic range of signal quality
- **ML Relevance**:
  - Indicates signal quality contrast
  - Higher values suggest diverse signal strengths
  - Useful for complexity assessment

#### `peak_to_baseline_ratio_cv`
- **Description**: Coefficient of variation of peak-to-baseline ratios
- **Units**: dimensionless
- **Calculation**: `std_ratio / mean_ratio`
- **Physical Meaning**: Relative variability in signal quality
- **ML Relevance**:
  - Normalized measure of ratio diversity
  - Useful for comparing different molecules
  - Indicates signal quality homogeneity

#### `peak_to_baseline_ratio_skewness`
- **Description**: Skewness of peak-to-baseline ratio distribution
- **Units**: dimensionless
- **Calculation**: `E[((X - μ) / σ)³]`
- **Physical Meaning**: Asymmetry of signal quality distribution
- **ML Relevance**:
  - Indicates whether most peaks are strong or weak
  - Useful for molecular classification
  - Correlates with molecular structure

#### `peak_to_baseline_ratio_kurtosis`
- **Description**: Kurtosis of peak-to-baseline ratio distribution
- **Units**: dimensionless
- **Calculation**: `E[((X - μ) / σ)⁴] - 3`
- **Physical Meaning**: Peakedness of signal quality distribution
- **ML Relevance**:
  - Indicates concentration of ratios around mean
  - Useful for identifying characteristic patterns
  - Correlates with molecular structure

#### `baseline_ratio_percentile_25`
- **Description**: 25th percentile of peak-to-baseline ratios
- **Units**: dimensionless
- **Calculation**: `np.percentile(ratios, 25)`
- **Physical Meaning**: Lower quartile of signal quality
- **ML Relevance**:
  - Indicates lower range of signal strengths
  - Useful for identifying weak signals
  - Helps assess spectral quality

#### `baseline_ratio_percentile_75`
- **Description**: 75th percentile of peak-to-baseline ratios
- **Units**: dimensionless
- **Calculation**: `np.percentile(ratios, 75)`
- **Physical Meaning**: Upper quartile of signal quality
- **ML Relevance**:
  - Indicates upper range of signal strengths
  - Useful for identifying strong signals
  - Helps assess spectral intensity

#### `baseline_ratio_iqr`
- **Description**: Interquartile range of peak-to-baseline ratios
- **Units**: dimensionless
- **Calculation**: `percentile_75 - percentile_25`
- **Physical Meaning**: Spread of middle 50% of signal qualities
- **ML Relevance**:
  - Robust measure of ratio variability
  - Less sensitive to outliers than range
  - Useful for molecular characterization

#### `baseline_ratio_median`
- **Description**: Median peak-to-baseline ratio
- **Units**: dimensionless
- **Calculation**: `np.median(ratios)`
- **Physical Meaning**: Central tendency of signal quality
- **ML Relevance**:
  - Robust measure of typical signal quality
  - Less sensitive to outliers than mean
  - Useful for molecular characterization

#### `high_ratio_peaks`
- **Description**: Number of peaks with ratio > 10
- **Units**: count
- **Calculation**: `np.sum(ratios > 10)`
- **Physical Meaning**: Number of strong signal peaks
- **ML Relevance**:
  - Indicates number of dominant vibrational modes
  - Useful for molecular fingerprinting
  - Correlates with functional group abundance

#### `medium_ratio_peaks`
- **Description**: Number of peaks with ratio 3-10
- **Units**: count
- **Calculation**: `np.sum((ratios >= 3) & (ratios <= 10))`
- **Physical Meaning**: Number of medium signal peaks
- **ML Relevance**:
  - Indicates number of moderate vibrational modes
  - Useful for molecular characterization
  - Correlates with molecular complexity

#### `low_ratio_peaks`
- **Description**: Number of peaks with ratio < 3
- **Units**: count
- **Calculation**: `np.sum(ratios < 3)`
- **Physical Meaning**: Number of weak signal peaks
- **ML Relevance**:
  - Indicates number of minor vibrational modes
  - Useful for quality assessment
  - Helps identify detection sensitivity

#### `total_baseline_area`
- **Description**: Total integrated baseline area
- **Units**: a.u.·cm⁻¹
- **Calculation**: `np.trapz(baseline, energy)`
- **Physical Meaning**: Total background contribution
- **ML Relevance**:
  - Indicates overall background level
  - Useful for quality assessment
  - Correlates with experimental conditions

#### `baseline_intensity_mean`
- **Description**: Average baseline intensity
- **Units**: a.u.
- **Calculation**: `np.mean(baseline)`
- **Physical Meaning**: Typical background level
- **ML Relevance**:
  - Indicates typical background intensity
  - Useful for quality assessment
  - Correlates with experimental conditions

#### `baseline_intensity_std`
- **Description**: Standard deviation of baseline intensity
- **Units**: a.u.
- **Calculation**: `np.std(baseline)`
- **Physical Meaning**: Background variability
- **ML Relevance**:
  - Indicates background stability
  - Lower values suggest stable background
  - Useful for quality control

#### `baseline_intensity_range`
- **Description**: Range of baseline intensities
- **Units**: a.u.
- **Calculation**: `np.max(baseline) - np.min(baseline)`
- **Physical Meaning**: Background contrast
- **ML Relevance**:
  - Indicates background variation
  - Higher values suggest variable background
  - Useful for quality assessment

#### `signal_to_baseline_ratio`
- **Description**: Overall signal-to-baseline ratio
- **Units**: dimensionless
- **Calculation**: `total_spectral_area / total_baseline_area`
- **Physical Meaning**: Total signal quality
- **ML Relevance**:
  - Indicates overall spectral quality
  - Higher values suggest better signal-to-noise
  - Useful for quality assessment and classification

## Feature Selection Guidelines

### For Molecular Classification
- **Primary**: `num_peaks`, `peak_density`, `energy_span`, `low_energy_peaks`, `mid_energy_peaks`, `high_energy_peaks`
- **Secondary**: `mean_center`, `std_center`, `amplitude_cv`, `fwhm_cv`

### For Property Prediction
- **Primary**: `total_area`, `mean_amplitude`, `mean_fwhm`, `peak_area_fraction`, `amplitude_skewness`
- **Secondary**: `area_cv`, `fwhm_skewness`, `mean_peak_spacing`

### For Quality Control
- **Primary**: `r_squared`, `rmse`, `baseline`, `peak_area_fraction`, `signal_to_baseline_ratio`, `mean_peak_to_baseline_ratio`
- **Secondary**: `amplitude_cv`, `fwhm_cv`, `energy_span`, `baseline_intensity_std`

### For Anomaly Detection
- **Primary**: All statistical features (skewness, kurtosis, CV), peak-to-baseline ratio features
- **Secondary**: `num_peaks`, `peak_density`, `energy_span`

### For Signal Quality Assessment
- **Primary**: `mean_peak_to_baseline_ratio`, `signal_to_baseline_ratio`, `high_ratio_peaks`, `low_ratio_peaks`
- **Secondary**: `peak_to_baseline_ratio_cv`, `baseline_intensity_mean`, `total_baseline_area`

## Feature Engineering Tips

1. **Normalization**: Use `total_spectral_area` for intensity normalization
2. **Scaling**: Apply standard scaling for ML algorithms
3. **Feature Selection**: Use correlation analysis to remove redundant features
4. **Dimensionality Reduction**: Consider PCA for high-dimensional datasets
5. **Feature Importance**: Use tree-based models to assess feature importance

## References

1. "Inelastic Neutron Scattering Spectroscopy" - T. J. Udovic
2. "Machine Learning for Spectroscopy" - A. Rinnan
3. "Feature Engineering for Machine Learning" - A. Zheng 