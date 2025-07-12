import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import gaussian_kde
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add utils to path for baseline detection
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from baseline_detection import detect_baseline
except ImportError:
    print("Warning: baseline_detection module not found. Baseline features will not be available.")
    detect_baseline = None

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MLPeakAnalyzer:
    """
    Advanced Gaussian-based peak analyzer optimized for machine learning feature extraction
    and publication-quality visualization.
    """
    
    def __init__(self, energy_range=(0, 3500), baseline_detector_type='dynamic_rolling', is_experimental=False):
        self.energy_range = energy_range
        self.baseline_detector_type = baseline_detector_type
        self.is_experimental = is_experimental
        self.spectrum_data = None
        self.peak_data = None
        self.baseline_data = None
        self.fit_results = None
        self.features = {}
        
    def load_spectrum_data(self, filepath: str, skiprows: int = 1, 
                          energy_col: int = 0, intensity_col: int = 1) -> None:
        """
        Load experimental INS spectrum data.
        
        Parameters:
        -----------
        filepath : str
            Path to the spectrum CSV file
        skiprows : int
            Number of header rows to skip
        energy_col : int or str
            Column index or name for energy values
        intensity_col : int or str
            Column index or name for intensity values
        """
        try:
            if skiprows == 0:
                spectrum_data = pd.read_csv(filepath, header=0)
            else:
                spectrum_data = pd.read_csv(filepath, skiprows=skiprows, header=None)
            self.spectrum_data = {
                'energy': spectrum_data[energy_col].values,
                'intensity': spectrum_data[intensity_col].values
            }
            print(f"✓ Loaded spectrum data: {len(self.spectrum_data['energy'])} points")
        except Exception as e:
            print(f"✗ Error loading spectrum data: {e}")
    
    def detect_baseline(self) -> None:
        """
        Detect baseline using the specified baseline detector.
        """
        if self.spectrum_data is None:
            raise ValueError("Spectrum data must be loaded first")
        
        if detect_baseline is None:
            print("Warning: Baseline detection not available. Using zero baseline.")
            self.baseline_data = {
                'baseline': np.zeros_like(self.spectrum_data['intensity'])
            }
            return
        
        try:
            x = self.spectrum_data['energy']
            y = self.spectrum_data['intensity']
            
            # Detect baseline using the specified method
            baseline = detect_baseline(
                intensity=y,
                energy=x,
                detector_type=self.baseline_detector_type,
                is_experimental=self.is_experimental
            )
            
            self.baseline_data = {
                'baseline': baseline
            }
            
            print(f"✓ Baseline detected using {self.baseline_detector_type} method")
            
        except Exception as e:
            print(f"✗ Error detecting baseline: {e}")
            # Fallback to zero baseline
            self.baseline_data = {
                'baseline': np.zeros_like(self.spectrum_data['intensity'])
            }
            
    def load_peak_data(self, filepath: str, energy_col: str = None,
                      intensity_col: str = None) -> None:
        """
        Load pre-detected peak data.
        
        Parameters:
        -----------
        filepath : str
            Path to the peaks CSV file
        energy_col : str, optional
            Column name for peak positions (auto-detected if None)
        intensity_col : str, optional
            Column name for peak intensities (auto-detected if None)
        """
        try:
            peaks_data = pd.read_csv(filepath)
            
            # Auto-detect column names if not provided
            if energy_col is None:
                if "Peak Position (Energy)" in peaks_data.columns:
                    energy_col = "Peak Position (Energy)"
                elif "Fitted X" in peaks_data.columns:
                    energy_col = "Fitted X"
                else:
                    energy_col = peaks_data.columns[0]  # Use first column
            
            if intensity_col is None:
                if "Peak Intensity" in peaks_data.columns:
                    intensity_col = "Peak Intensity"
                elif "Fitted Y" in peaks_data.columns:
                    intensity_col = "Fitted Y"
                else:
                    intensity_col = peaks_data.columns[1]  # Use second column
            
            self.peak_data = {
                'positions': peaks_data[energy_col].values,
                'intensities': peaks_data[intensity_col].values
            }
            print(f"✓ Loaded peak data: {len(self.peak_data['positions'])} peaks")
            print(f"  Using columns: {energy_col}, {intensity_col}")
        except Exception as e:
            print(f"✗ Error loading peak data: {e}")
    
    def gaussian_function(self, x: np.ndarray, amplitude: float, 
                         center: float, sigma: float) -> np.ndarray:
        """
        Normalized Gaussian function.
        
        Parameters:
        -----------
        x : np.ndarray
            Energy values
        amplitude : float
            Peak amplitude
        center : float
            Peak center position
        sigma : float
            Gaussian width parameter
            
        Returns:
        --------
        np.ndarray
            Gaussian function values
        """
        return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))
    
    def global_gaussian_model(self, x: np.ndarray, *params: float) -> np.ndarray:
        """
        Global model: baseline + sum of Gaussian peaks.
        
        Parameters:
        -----------
        x : np.ndarray
            Energy values
        *params : float
            Parameters: [baseline, amp1, center1, sigma1, amp2, center2, sigma2, ...]
            
        Returns:
        --------
        np.ndarray
            Model function values
        """
        baseline = params[0]
        model = baseline * np.ones_like(x)
        
        num_peaks = (len(params) - 1) // 3
        for i in range(num_peaks):
            amp = params[1 + i*3]
            center = params[1 + i*3 + 1]
            sigma = params[1 + i*3 + 2]
            model += self.gaussian_function(x, amp, center, sigma)
            
        return model
    
    def fit_global_gaussians(self, smoothing: bool = True, 
                           smooth_window: int = 51) -> Dict:
        """
        Perform global Gaussian fitting to the spectrum.
        
        Parameters:
        -----------
        smoothing : bool
            Whether to smooth data before fitting
        smooth_window : int
            Savitzky-Golay filter window size
            
        Returns:
        --------
        Dict
            Fitting results and parameters
        """
        if self.spectrum_data is None or self.peak_data is None:
            raise ValueError("Spectrum and peak data must be loaded first")
        
        x = self.spectrum_data['energy']
        y = self.spectrum_data['intensity']
        
        # Smooth data if requested
        if smoothing:
            if smooth_window % 2 == 0:
                smooth_window += 1
            y_smooth = savgol_filter(y, smooth_window, 3)
        else:
            y_smooth = y
        
        # Set up initial parameters and bounds
        num_peaks = len(self.peak_data['positions'])
        p0 = [0.0]  # baseline
        lb = [0.0]  # baseline lower bound
        ub = [np.inf]  # baseline upper bound
        
        for i in range(num_peaks):
            center0 = self.peak_data['positions'][i]
            amp0 = self.peak_data['intensities'][i]
            sigma0 = 5.0  # Initial guess for width
            
            p0.extend([amp0, center0, sigma0])
            lb.extend([0, max(0, center0 - 10), 0.1])
            ub.extend([np.inf, center0 + 10, 50])
        
        # Perform global fit
        try:
            popt, pcov = curve_fit(
                self.global_gaussian_model, x, y_smooth,
                p0=p0, bounds=(lb, ub), maxfev=100000
            )
            
            # Calculate fit quality metrics
            y_fit = self.global_gaussian_model(x, *popt)
            residuals = y_smooth - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_smooth - np.mean(y_smooth))**2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Extract individual peak parameters
            baseline = popt[0]
            peak_params = []
            for i in range(num_peaks):
                amp = popt[1 + i*3]
                center = popt[1 + i*3 + 1]
                sigma = popt[1 + i*3 + 2]
                fwhm = 2.355 * sigma
                peak_params.append({
                    'amplitude': amp,
                    'center': center,
                    'sigma': sigma,
                    'fwhm': fwhm,
                    'area': amp * sigma * np.sqrt(2 * np.pi)
                })
            
            self.fit_results = {
                'parameters': popt,
                'covariance': pcov,
                'baseline': baseline,
                'peak_params': peak_params,
                'r_squared': r_squared,
                'rmse': rmse,
                'y_fit': y_fit,
                'residuals': residuals
            }
            
            print(f"✓ Global fit completed: R² = {r_squared:.4f}, RMSE = {rmse:.4f}")
            return self.fit_results
            
        except Exception as e:
            print(f"✗ Fitting failed: {e}")
            return None
    
    def extract_ml_features(self, remove_ratio_outliers=True) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract comprehensive features for machine learning.
        
        Parameters:
        -----------
        remove_ratio_outliers : bool, optional
            If True, remove outliers from peak-to-baseline ratio calculations
            using 1st-99th percentile range (default: True)
        
        Returns:
        --------
        Dict
            Dictionary of extracted features
        """
        if self.fit_results is None:
            raise ValueError("Must perform fitting before feature extraction")
        
        peak_params = self.fit_results['peak_params']
        num_peaks = len(peak_params)
        
        # Basic peak statistics
        amplitudes = [p['amplitude'] for p in peak_params]
        centers = [p['center'] for p in peak_params]
        fwhms = [p['fwhm'] for p in peak_params]
        areas = [p['area'] for p in peak_params]
        
        # Calculate total integrated area of the entire spectrum
        x = self.spectrum_data['energy']
        y = self.spectrum_data['intensity']
        total_spectral_area = np.trapz(y, x)  # Numerical integration of the spectrum
        
        # Statistical features
        features = {
            # Peak count features
            'num_peaks': num_peaks,
            'peak_density': num_peaks / (self.energy_range[1] - self.energy_range[0]),
            
            # Amplitude features
            'mean_amplitude': np.mean(amplitudes),
            'std_amplitude': np.std(amplitudes),
            'max_amplitude': np.max(amplitudes),
            'min_amplitude': np.min(amplitudes),
            'amplitude_range': np.max(amplitudes) - np.min(amplitudes),
            'amplitude_cv': np.std(amplitudes) / np.mean(amplitudes) if np.mean(amplitudes) > 0 else 0,
            
            # Width features
            'mean_fwhm': np.mean(fwhms),
            'std_fwhm': np.std(fwhms),
            'max_fwhm': np.max(fwhms),
            'min_fwhm': np.min(fwhms),
            'fwhm_range': np.max(fwhms) - np.min(fwhms),
            'fwhm_cv': np.std(fwhms) / np.mean(fwhms) if np.mean(fwhms) > 0 else 0,
            
            # Area features
            'total_area': np.sum(areas),
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'area_cv': np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0,
            'max_area': np.max(areas),
            'min_area': np.min(areas),
            'area_range': np.max(areas) - np.min(areas),
            'area_skewness': self._calculate_skewness(areas),
            'area_kurtosis': self._calculate_kurtosis(areas),
            
            # Spectral area features (calculated from raw spectrum data)
            'total_spectral_area': total_spectral_area,
            'peak_area_fraction': np.sum(areas) / total_spectral_area if total_spectral_area > 0 else 0,
            'non_peak_area': total_spectral_area - np.sum(areas),
            'non_peak_area_fraction': (total_spectral_area - np.sum(areas)) / total_spectral_area if total_spectral_area > 0 else 0,
            
            # Position features
            'mean_center': np.mean(centers),
            'std_center': np.std(centers),
            'center_range': np.max(centers) - np.min(centers),
            'energy_span': np.max(centers) - np.min(centers),
            
            # Fit quality features
            'r_squared': self.fit_results['r_squared'],
            'rmse': self.fit_results['rmse'],
            'baseline': self.fit_results['baseline'],
            
            # Distribution features
            'amplitude_skewness': self._calculate_skewness(amplitudes),
            'amplitude_kurtosis': self._calculate_kurtosis(amplitudes),
            'fwhm_skewness': self._calculate_skewness(fwhms),
            'fwhm_kurtosis': self._calculate_kurtosis(fwhms),
            
            # Peak spacing features
            'mean_peak_spacing': self._calculate_mean_spacing(centers),
            'std_peak_spacing': self._calculate_std_spacing(centers),
            
            # Energy region features
            'low_energy_peaks': len([c for c in centers if c < 500]),
            'mid_energy_peaks': len([c for c in centers if 500 <= c < 2000]),
            'high_energy_peaks': len([c for c in centers if c >= 2000]),
            
            # Individual peak features (for detailed analysis)
            'peak_centers': np.array(centers),
            'peak_amplitudes': np.array(amplitudes),
            'peak_fwhms': np.array(fwhms),
            'peak_areas': np.array(areas),
            
            # Enhanced area analysis features
            'largest_peak_area': np.max(areas),
            'smallest_peak_area': np.min(areas),
            'largest_peak_area_fraction': np.max(areas) / total_spectral_area if total_spectral_area > 0 else 0,
            'area_median': np.median(areas),
            'area_percentile_25': np.percentile(areas, 25),
            'area_percentile_75': np.percentile(areas, 75),
            'area_iqr': np.percentile(areas, 75) - np.percentile(areas, 25)
        }
        
        # Calculate baseline area features (if baseline is detected)
        if hasattr(self, 'baseline_data') and self.baseline_data is not None:
            baseline_values = self.baseline_data['baseline']
            baseline_area = np.trapz(baseline_values, x)  # Area under the detected baseline
            
            features.update({
                'detected_baseline_area': baseline_area,
                'detected_baseline_area_fraction': baseline_area / total_spectral_area if total_spectral_area > 0 else 0,
                'signal_above_baseline_area': total_spectral_area - baseline_area,
                'signal_above_baseline_fraction': (total_spectral_area - baseline_area) / total_spectral_area if total_spectral_area > 0 else 0
            })
        else:
            # No baseline data available
            features.update({
                'detected_baseline_area': 0.0,
                'detected_baseline_area_fraction': 0.0,
                'signal_above_baseline_area': total_spectral_area,
                'signal_above_baseline_fraction': 1.0
            })
        
        # Calculate peak-to-baseline ratios with optional outlier removal
        if hasattr(self, 'baseline_data') and self.baseline_data is not None:
            baseline_values = self.baseline_data['baseline']
            peak_to_baseline_ratios = []
            
            for i, params in enumerate(peak_params):
                # Find baseline value at peak center
                center_idx = np.argmin(np.abs(x - params['center']))
                baseline_at_peak = baseline_values[center_idx]
                
                if baseline_at_peak > 0:
                    ratio = params['amplitude'] / baseline_at_peak
                    peak_to_baseline_ratios.append(ratio)
            
            if peak_to_baseline_ratios:
                ratios_array = np.array(peak_to_baseline_ratios)
                
                # Remove outliers if requested
                if remove_ratio_outliers and len(ratios_array) > 4:
                    q1 = np.percentile(ratios_array, 1)
                    q99 = np.percentile(ratios_array, 99)
                    filtered_ratios = ratios_array[(ratios_array >= q1) & (ratios_array <= q99)]
                else:
                    filtered_ratios = ratios_array
                
                # Add ratio features
                features.update({
                    'mean_peak_to_baseline_ratio': np.mean(filtered_ratios),
                    'std_peak_to_baseline_ratio': np.std(filtered_ratios),
                    'max_peak_to_baseline_ratio': np.max(filtered_ratios),
                    'min_peak_to_baseline_ratio': np.min(filtered_ratios),
                    'median_peak_to_baseline_ratio': np.median(filtered_ratios),
                    'peak_to_baseline_ratio_cv': np.std(filtered_ratios) / np.mean(filtered_ratios) if np.mean(filtered_ratios) > 0 else 0,
                    'peak_to_baseline_ratios': filtered_ratios,
                    'num_ratio_outliers_removed': len(ratios_array) - len(filtered_ratios) if remove_ratio_outliers and len(ratios_array) > 4 else 0
                })
            else:
                # No valid ratios found
                features.update({
                    'mean_peak_to_baseline_ratio': 0.0,
                    'std_peak_to_baseline_ratio': 0.0,
                    'max_peak_to_baseline_ratio': 0.0,
                    'min_peak_to_baseline_ratio': 0.0,
                    'median_peak_to_baseline_ratio': 0.0,
                    'peak_to_baseline_ratio_cv': 0.0,
                    'peak_to_baseline_ratios': np.array([]),
                    'num_ratio_outliers_removed': 0
                })
        else:
            # No baseline data available
            features.update({
                'mean_peak_to_baseline_ratio': 0.0,
                'std_peak_to_baseline_ratio': 0.0,
                'max_peak_to_baseline_ratio': 0.0,
                'min_peak_to_baseline_ratio': 0.0,
                'median_peak_to_baseline_ratio': 0.0,
                'peak_to_baseline_ratio_cv': 0.0,
                'peak_to_baseline_ratios': np.array([]),
                'num_ratio_outliers_removed': 0
            })
        
        self.features = features
        return features
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_mean_spacing(self, centers: List[float]) -> float:
        """Calculate mean spacing between peaks."""
        if len(centers) < 2:
            return 0.0
        sorted_centers = np.sort(centers)
        spacings = np.diff(sorted_centers)
        return np.mean(spacings)
    
    def _calculate_std_spacing(self, centers: List[float]) -> float:
        """Calculate standard deviation of peak spacings."""
        if len(centers) < 2:
            return 0.0
        sorted_centers = np.sort(centers)
        spacings = np.diff(sorted_centers)
        return np.std(spacings)
    
    def plot_publication_quality(self, save_path: Optional[str] = None,
                               dpi: int = 300, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Create publication-quality plots.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        dpi : int
            Figure resolution
        figsize : tuple
            Figure size (width, height)
        """
        if self.fit_results is None:
            raise ValueError("Must perform fitting before plotting")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        
        x = self.spectrum_data['energy']
        y = self.spectrum_data['intensity']
        y_fit = self.fit_results['y_fit']
        peak_params = self.fit_results['peak_params']
        
        # Main spectrum plot
        ax1.plot(x, y, 'k-', linewidth=1, alpha=0.7, label='Experimental')
        ax1.plot(x, y_fit, 'r-', linewidth=2, label='Gaussian Fit')
        
        # Plot individual peaks
        for i, params in enumerate(peak_params):
            x_peak = np.linspace(params['center'] - 3*params['sigma'], 
                               params['center'] + 3*params['sigma'], 100)
            y_peak = self.gaussian_function(x_peak, params['amplitude'], 
                                          params['center'], params['sigma'])
            ax1.plot(x_peak, y_peak, '--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel('Energy (cm$^{-1}$)', fontsize=12)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax1.set_title('INS Spectrum with Gaussian Fits', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(self.energy_range)
        
        # Peak width vs energy
        centers = [p['center'] for p in peak_params]
        fwhms = [p['fwhm'] for p in peak_params]
        
        ax2.scatter(centers, fwhms, c='blue', s=60, alpha=0.7, edgecolors='black', linewidth=0.5, label='Peak Widths')
        
        # Instrument peak width line
        x_instrument = np.linspace(0, 3500, 100)
        y_instrument = 0.25 + 0.005 * x_instrument + 0.0000001 * (x_instrument**2)
        ax2.plot(x_instrument, y_instrument, 'r-', linewidth=3, alpha=0.8, label='Instrument Width')
        
        # Fit trend line to data
        if len(centers) > 2:
            z = np.polyfit(centers, fwhms, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(centers), max(centers), 100)
            ax2.plot(x_trend, p(x_trend), 'g--', linewidth=2, alpha=0.8, label='Data Trend')
        
        ax2.set_xlabel('Peak Energy (cm$^{-1}$)', fontsize=12)
        ax2.set_ylabel('Peak Width (FWHM, cm$^{-1}$)', fontsize=12)
        ax2.set_title('Peak Width vs Energy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Amplitude distribution
        amplitudes = [p['amplitude'] for p in peak_params]
        ax3.hist(amplitudes, bins=max(3, len(amplitudes)//3), alpha=0.7, 
                color='green', edgecolor='black', linewidth=0.5)
        ax3.axvline(np.mean(amplitudes), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(amplitudes):.2f}')
        ax3.set_xlabel('Peak Amplitude (a.u.)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Peak Amplitude Distribution', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = self.fit_results['residuals']
        ax4.plot(x, residuals, 'k-', linewidth=1, alpha=0.7)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax4.set_xlabel('Energy (cm$^{-1}$)', fontsize=12)
        ax4.set_ylabel('Residuals', fontsize=12)
        ax4.set_title(f'Fit Residuals (R² = {self.fit_results["r_squared"]:.4f})', 
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(self.energy_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✓ Figure saved to: {save_path}")
        plt.close()
    
    def save_features_to_csv(self, filepath: str) -> None:
        """
        Save extracted features to CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the features CSV
        """
        if not self.features:
            raise ValueError("Must extract features before saving")
        
        # Convert features to DataFrame
        feature_data = {}
        for key, value in self.features.items():
            if isinstance(value, np.ndarray):
                # For array features, create separate columns
                for i, val in enumerate(value):
                    feature_data[f"{key}_{i+1}"] = val
            else:
                feature_data[key] = value
        
        df = pd.DataFrame([feature_data])
        df.to_csv(filepath, index=False)
        print(f"✓ Features saved to: {filepath}")
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of the analysis."""
        if self.fit_results is None:
            print("No analysis results available.")
            return
        
        print("\n" + "="*60)
        print("PEAK ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Number of peaks analyzed: {len(self.fit_results['peak_params'])}")
        print(f"Fit quality (R²): {self.fit_results['r_squared']:.4f}")
        print(f"Root mean square error: {self.fit_results['rmse']:.4f}")
        print(f"Baseline offset: {self.fit_results['baseline']:.4f}")
        
        if self.features:
            print(f"\nKey Features:")
            print(f"  Mean peak width: {self.features['mean_fwhm']:.2f} cm⁻¹")
            print(f"  Mean peak amplitude: {self.features['mean_amplitude']:.2f}")
            print(f"  Total peak area: {self.features['total_area']:.2f}")
            print(f"  Total spectral area: {self.features['total_spectral_area']:.2f}")
            print(f"  Peak area fraction: {self.features['peak_area_fraction']:.3f}")
            print(f"  Energy span: {self.features['energy_span']:.1f} cm⁻¹")
            print(f"  Peak density: {self.features['peak_density']:.4f} peaks/cm⁻¹")
        
        print("="*60) 

    def detect_peaks_from_spectrum(self, height=None, distance=3, prominence=0.01, width=1, smooth_window=11, plot_detection=False):
        """
        Detect peaks directly from the loaded spectrum with high sensitivity (shoulders and all peaks).
        Assumes low/no noise in the spectrum (as in INS spectra).

        Parameters:
        -----------
        height : float, optional
            Minimum peak height (None = auto)
        distance : int, optional
            Minimum number of points between peaks (default: 3)
        prominence : float, optional
            Minimum prominence of peaks (default: 0.01)
        width : int, optional
            Minimum width of peaks (default: 1)
        smooth_window : int, optional
            Window size for Savitzky-Golay smoothing (default: 11)
        plot_detection : bool, optional
            If True, plot the detected peaks for visual confirmation
        """
        if self.spectrum_data is None:
            raise ValueError("Spectrum data must be loaded first")
        x = self.spectrum_data['energy']
        y = self.spectrum_data['intensity']
        # Smoothing (mild, just to help with numerical stability)
        if smooth_window > 1:
            if smooth_window % 2 == 0:
                smooth_window += 1
            from scipy.signal import savgol_filter
            y_smooth = savgol_filter(y, smooth_window, 3)
        else:
            y_smooth = y
        # Peak detection (high sensitivity)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(
            y_smooth,
            height=height,
            distance=distance,
            prominence=prominence,
            width=width
        )
        peak_positions = x[peaks]
        peak_intensities = y[peaks]
        self.peak_data = {
            'positions': peak_positions,
            'intensities': peak_intensities
        }
        print(f"✓ Detected {len(peak_positions)} peaks (including shoulders)")
        if plot_detection:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 5))
            plt.plot(x, y, label='Spectrum')
            plt.plot(peak_positions, peak_intensities, 'ro', label='Detected Peaks')
            plt.xlabel('Energy (cm$^{-1}$)')
            plt.ylabel('Intensity (a.u.)')
            plt.title('Peak Detection (High Sensitivity)')
            plt.legend()
            plt.show()
        return peak_positions, peak_intensities 