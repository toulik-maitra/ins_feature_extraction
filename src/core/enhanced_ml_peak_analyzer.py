"""
Enhanced ML Peak Analyzer
=========================

This module provides an enhanced version of the MLPeakAnalyzer that integrates
the new comprehensive baseline detection system with parameter optimization
and validation capabilities.

Author: Enhanced Baseline Detection System
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import gaussian_kde
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import sys
import os
import time
warnings.filterwarnings('ignore')

# Add utils to path for enhanced baseline detection
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
try:
    from enhanced_baseline_detection import (
        detect_enhanced_baseline,
        EnhancedBaselineDetectorFactory,
        BaselineValidationSystem,
        BaselineQualityMetrics
    )
    ENHANCED_BASELINE_AVAILABLE = True
except ImportError:
    ENHANCED_BASELINE_AVAILABLE = False
    print("Warning: enhanced_baseline_detection module not found. Using fallback baseline detection.")
    try:
        from baseline_detection import detect_baseline
    except ImportError:
        print("Warning: baseline_detection module not found. Baseline features will not be available.")
        detect_baseline = None

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class EnhancedMLPeakAnalyzer:
    """
    Enhanced Gaussian-based peak analyzer with comprehensive baseline detection
    and parameter optimization capabilities.
    """
    
    def __init__(self, energy_range=(0, 3500), 
                 baseline_detector_type='pybaselines_asls',
                 enable_parameter_optimization=False,
                 validation_data=None,
                 is_experimental=False):
        """
        Initialize the enhanced peak analyzer.
        
        Parameters:
        -----------
        energy_range : tuple
            Energy range for analysis (min, max) in cm⁻¹
        baseline_detector_type : str
            Type of baseline detector to use
        enable_parameter_optimization : bool
            Whether to enable parameter optimization
        validation_data : dict, optional
            Validation data for parameter optimization
        is_experimental : bool
            Whether the data is experimental (for legacy compatibility)
        """
        self.energy_range = energy_range
        self.baseline_detector_type = baseline_detector_type
        self.enable_parameter_optimization = enable_parameter_optimization
        self.validation_data = validation_data
        self.is_experimental = is_experimental
        
        # Data storage
        self.spectrum_data = None
        self.peak_data = None
        self.baseline_data = None
        self.fit_results = None
        self.features = {}
        
        # Enhanced baseline detection results
        self.baseline_optimization_results = None
        self.baseline_quality_metrics = None
        
        # Analysis metadata
        self.analysis_metadata = {
            'baseline_detector_used': baseline_detector_type,
            'parameter_optimization_enabled': enable_parameter_optimization,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'energy_range': energy_range
        }
    
    def load_spectrum_data(self, filepath: str, skiprows: int = 0, 
                          energy_col: str = "x", intensity_col: str = "y") -> None:
        """
        Load spectrum data from file.
        
        Parameters:
        -----------
        filepath : str
            Path to the spectrum file
        skiprows : int
            Number of rows to skip
        energy_col : str
            Name of energy column
        intensity_col : str
            Name of intensity column
        """
        try:
            # Load data
            data = pd.read_csv(filepath, skiprows=skiprows)
            
            # Extract energy and intensity
            energy = data[energy_col].values
            intensity = data[intensity_col].values
            
            # Filter to energy range
            mask = (energy >= self.energy_range[0]) & (energy <= self.energy_range[1])
            energy = energy[mask]
            intensity = intensity[mask]
            
            self.spectrum_data = {
                'energy': energy,
                'intensity': intensity,
                'filepath': filepath
            }
            
            print(f"✓ Spectrum data loaded: {len(energy)} points in {self.energy_range[0]}-{self.energy_range[1]} cm⁻¹")
            
        except Exception as e:
            print(f"✗ Error loading spectrum data: {e}")
            raise
    
    def detect_enhanced_baseline(self) -> None:
        """
        Detect baseline using the enhanced baseline detection system.
        """
        if self.spectrum_data is None:
            raise ValueError("Spectrum data must be loaded first")
        
        if not ENHANCED_BASELINE_AVAILABLE:
            print("Warning: Enhanced baseline detection not available. Using fallback method.")
            self._detect_fallback_baseline()
            return
        
        try:
            x = self.spectrum_data['energy']
            y = self.spectrum_data['intensity']
            
            print(f"Detecting baseline using {self.baseline_detector_type}...")
            
            # Detect baseline using enhanced system
            baseline_result = detect_enhanced_baseline(
                intensity=y,
                energy=x,
                detector_type=self.baseline_detector_type,
                optimize_parameters=self.enable_parameter_optimization,
                validation_data=self.validation_data
            )
            
            # Store results
            self.baseline_data = {
                'baseline': baseline_result['baseline'],
                'detector_name': baseline_result['detector_name'],
                'detector_type': baseline_result['detector_type'],
                'parameters': baseline_result['parameters'],
                'processing_time': baseline_result['processing_time'],
                'optimization_history': baseline_result['optimization_history']
            }
            
            # Store quality metrics if available
            if baseline_result['quality_metrics']:
                self.baseline_quality_metrics = baseline_result['quality_metrics']
            
            # Store optimization results if available
            if baseline_result['optimization_history']:
                self.baseline_optimization_results = baseline_result['optimization_history']
            
            print(f"✓ Baseline detected successfully using {baseline_result['detector_name']}")
            print(f"  Processing time: {baseline_result['processing_time']:.3f}s")
            
            if baseline_result['quality_metrics']:
                metrics = baseline_result['quality_metrics']
                print(f"  Quality metrics - RMSE: {metrics['rmse']:.6f}, "
                      f"Correlation: {metrics['correlation']:.4f}")
            
        except Exception as e:
            print(f"✗ Error detecting enhanced baseline: {e}")
            print("Falling back to basic baseline detection...")
            self._detect_fallback_baseline()
    
    def _detect_fallback_baseline(self) -> None:
        """Fallback baseline detection using original method."""
        if detect_baseline is None:
            print("Warning: No baseline detection available. Using zero baseline.")
            self.baseline_data = {
                'baseline': np.zeros_like(self.spectrum_data['intensity']),
                'detector_name': 'zero_baseline',
                'detector_type': 'fallback',
                'parameters': {},
                'processing_time': 0.0
            }
            return
        
        try:
            x = self.spectrum_data['energy']
            y = self.spectrum_data['intensity']
            
            baseline = detect_baseline(
                intensity=y,
                energy=x,
                detector_type='dynamic_rolling',
                is_experimental=self.is_experimental
            )
            
            self.baseline_data = {
                'baseline': baseline,
                'detector_name': 'dynamic_rolling_fallback',
                'detector_type': 'fallback',
                'parameters': {},
                'processing_time': 0.0
            }
            
            print("✓ Fallback baseline detection completed")
            
        except Exception as e:
            print(f"✗ Fallback baseline detection failed: {e}")
            self.baseline_data = {
                'baseline': np.zeros_like(self.spectrum_data['intensity']),
                'detector_name': 'zero_baseline',
                'detector_type': 'fallback',
                'parameters': {},
                'processing_time': 0.0
            }
    
    def detect_peaks_from_spectrum(self, height=None, distance=3, prominence=0.01, 
                                 width=1, smooth_window=11, plot_detection=False):
        """
        Detect peaks directly from the loaded spectrum with high sensitivity.
        
        Parameters:
        -----------
        height : float, optional
            Minimum peak height (None = auto)
        distance : int, optional
            Minimum number of points between peaks
        prominence : float, optional
            Minimum prominence of peaks
        width : int, optional
            Minimum width of peaks
        smooth_window : int, optional
            Window size for Savitzky-Golay smoothing
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
            y_smooth = savgol_filter(y, smooth_window, 3)
        else:
            y_smooth = y
        
        # Peak detection (high sensitivity)
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
            'intensities': peak_intensities,
            'peak_indices': peaks,
            'properties': properties
        }
        
        print(f"✓ Detected {len(peak_positions)} peaks (including shoulders)")
        
        if plot_detection:
            self._plot_peak_detection(x, y, y_smooth, peaks)
    
    def _plot_peak_detection(self, x, y, y_smooth, peaks):
        """Plot peak detection results."""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x, y, 'k-', alpha=0.7, label='Original Spectrum')
        plt.plot(x, y_smooth, 'r-', alpha=0.5, label='Smoothed Spectrum')
        plt.plot(x[peaks], y[peaks], 'ro', markersize=8, label='Detected Peaks')
        plt.xlabel('Energy (cm⁻¹)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Peak Detection Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(x, y - y_smooth, 'g-', alpha=0.7, label='Residuals')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Energy (cm⁻¹)')
        plt.ylabel('Residuals (a.u.)')
        plt.title('Fitting Residuals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def gaussian(self, x, amplitude, center, sigma):
        """Gaussian function for peak fitting."""
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    
    def global_gaussian_model(self, x, baseline, *params):
        """Global Gaussian model with baseline."""
        result = baseline * np.ones_like(x)
        for i in range(0, len(params), 3):
            if i + 2 < len(params):
                amplitude, center, sigma = params[i], params[i+1], params[i+2]
                result += self.gaussian(x, amplitude, center, sigma)
        return result
    
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
    
    def extract_enhanced_ml_features(self, remove_ratio_outliers=True) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract enhanced ML features including baseline quality metrics.
        
        Parameters:
        -----------
        remove_ratio_outliers : bool
            Whether to remove outliers from peak-to-baseline ratios
            
        Returns:
        --------
        Dict[str, Union[float, np.ndarray]]
            Dictionary of extracted features
        """
        if self.spectrum_data is None or self.peak_data is None:
            raise ValueError("Spectrum and peak data must be loaded first")
        
        x = self.spectrum_data['energy']
        y = self.spectrum_data['intensity']
        
        # Basic peak features
        peak_positions = self.peak_data['positions']
        peak_intensities = self.peak_data['intensities']
        
        # Extract peak parameters from fitting if available
        if self.fit_results is not None:
            peak_params = self.fit_results['peak_params']
            amplitudes = [p['amplitude'] for p in peak_params]
            centers = [p['center'] for p in peak_params]
            sigmas = [p['sigma'] for p in peak_params]
            fwhms = [p['fwhm'] for p in peak_params]
            areas = [p['area'] for p in peak_params]
        else:
            # Use detected peak data
            amplitudes = peak_intensities
            centers = peak_positions
            sigmas = [5.0] * len(peak_positions)  # Default width
            fwhms = [11.775] * len(peak_positions)  # Default FWHM
            areas = [amp * sig * np.sqrt(2 * np.pi) for amp, sig in zip(amplitudes, sigmas)]
        
        # Convert to numpy arrays
        amplitudes = np.array(amplitudes)
        centers = np.array(centers)
        sigmas = np.array(sigmas)
        fwhms = np.array(fwhms)
        areas = np.array(areas)
        
        # Basic features
        features = {
            'num_peaks': len(peak_positions),
            'peak_density': len(peak_positions) / (self.energy_range[1] - self.energy_range[0]),
            
            # Amplitude features
            'mean_amplitude': np.mean(amplitudes),
            'std_amplitude': np.std(amplitudes),
            'max_amplitude': np.max(amplitudes),
            'min_amplitude': np.min(amplitudes),
            'amplitude_range': np.max(amplitudes) - np.min(amplitudes),
            'amplitude_cv': np.std(amplitudes) / np.mean(amplitudes) if np.mean(amplitudes) > 0 else 0,
            
            # Position features
            'mean_center': np.mean(centers),
            'std_center': np.std(centers),
            'center_range': np.max(centers) - np.min(centers),
            
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
            'max_area': np.max(areas),
            'min_area': np.min(areas),
            'area_range': np.max(areas) - np.min(areas),
            'area_cv': np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0,
        }
        
        # Add enhanced baseline features
        if self.baseline_data is not None:
            baseline_values = self.baseline_data['baseline']
            baseline_area = np.trapz(baseline_values, x)
            total_spectral_area = np.trapz(y, x)
            
            features.update({
                'detected_baseline_area': baseline_area,
                'detected_baseline_area_fraction': baseline_area / total_spectral_area if total_spectral_area > 0 else 0,
                'signal_above_baseline_area': total_spectral_area - baseline_area,
                'signal_above_baseline_fraction': (total_spectral_area - baseline_area) / total_spectral_area if total_spectral_area > 0 else 0,
                'baseline_detector_used': self.baseline_data['detector_name'],
                'baseline_processing_time': self.baseline_data['processing_time']
            })
            
            # Add baseline quality metrics if available
            if self.baseline_quality_metrics:
                features.update({
                    'baseline_rmse': self.baseline_quality_metrics['rmse'],
                    'baseline_correlation': self.baseline_quality_metrics['correlation'],
                    'baseline_smoothness': self.baseline_quality_metrics['smoothness'],
                    'baseline_peak_preservation': self.baseline_quality_metrics['peak_preservation'],
                    'baseline_quality_score': self.baseline_quality_metrics['quality_score']
                })
        
        # Add fit quality features if available
        if self.fit_results is not None:
            features.update({
                'fit_r_squared': self.fit_results['r_squared'],
                'fit_rmse': self.fit_results['rmse'],
                'fit_baseline': self.fit_results['baseline']
            })
        
        # Add analysis metadata
        features.update({
            'analysis_timestamp': self.analysis_metadata['analysis_timestamp'],
            'energy_range_min': self.energy_range[0],
            'energy_range_max': self.energy_range[1],
            'parameter_optimization_enabled': self.enable_parameter_optimization
        })
        
        self.features = features
        return features
    
    def plot_enhanced_analysis(self, save_path: str = None):
        """
        Create comprehensive visualization of the enhanced analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.spectrum_data is None:
            raise ValueError("Spectrum data must be loaded first")
        
        x = self.spectrum_data['energy']
        y = self.spectrum_data['intensity']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced ML Peak Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Original spectrum with baseline
        ax1 = axes[0, 0]
        ax1.plot(x, y, 'k-', alpha=0.7, label='Original Spectrum', linewidth=1)
        
        if self.baseline_data is not None:
            baseline = self.baseline_data['baseline']
            ax1.plot(x, baseline, 'r--', linewidth=2, 
                    label=f"Baseline ({self.baseline_data['detector_name']})")
        
        if self.peak_data is not None:
            peak_positions = self.peak_data['positions']
            peak_intensities = self.peak_data['intensities']
            ax1.plot(peak_positions, peak_intensities, 'bo', markersize=6, label='Detected Peaks')
        
        ax1.set_xlabel('Energy (cm⁻¹)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.set_title('Spectrum with Baseline and Peaks')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Baseline-corrected spectrum with fit
        ax2 = axes[0, 1]
        if self.baseline_data is not None:
            corrected_spectrum = y - self.baseline_data['baseline']
            ax2.plot(x, corrected_spectrum, 'g-', alpha=0.7, label='Baseline-Corrected Spectrum', linewidth=1)
            
            if self.fit_results is not None:
                y_fit = self.fit_results['y_fit']
                ax2.plot(x, y_fit, 'r-', linewidth=2, 
                        label=f"Gaussian Fit (R² = {self.fit_results['r_squared']:.4f})")
        
        ax2.set_xlabel('Energy (cm⁻¹)')
        ax2.set_ylabel('Intensity (a.u.)')
        ax2.set_title('Baseline-Corrected Spectrum with Fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Quality metrics
        ax3 = axes[1, 0]
        if self.baseline_quality_metrics:
            metrics = ['rmse', 'correlation', 'smoothness', 'peak_preservation']
            values = [self.baseline_quality_metrics[m] for m in metrics]
            colors = ['red', 'blue', 'green', 'orange']
            
            bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
            ax3.set_ylabel('Metric Value')
            ax3.set_title('Baseline Quality Metrics')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
        
        # Plot 4: Parameter optimization history (if available)
        ax4 = axes[1, 1]
        if self.baseline_optimization_results:
            iterations = [h['iteration'] for h in self.baseline_optimization_results]
            scores = [h['quality_score'] for h in self.baseline_optimization_results]
            ax4.plot(iterations, scores, 'o-', color='purple', alpha=0.7)
            ax4.set_xlabel('Optimization Iteration')
            ax4.set_ylabel('Quality Score (Lower is Better)')
            ax4.set_title('Parameter Optimization Progress')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No optimization data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Parameter Optimization')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Enhanced analysis plot saved: {save_path}")
        
        plt.show()
    
    def generate_enhanced_report(self, save_path: str = None) -> str:
        """
        Generate a comprehensive report of the enhanced analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the report
            
        Returns:
        --------
        str
            Report content
        """
        report = []
        report.append("# Enhanced ML Peak Analysis Report")
        report.append("")
        report.append(f"**Analysis Date:** {self.analysis_metadata['analysis_timestamp']}")
        report.append(f"**Energy Range:** {self.energy_range[0]}-{self.energy_range[1]} cm⁻¹")
        report.append("")
        
        # Spectrum information
        if self.spectrum_data:
            report.append("## Spectrum Information")
            report.append(f"- Data points: {len(self.spectrum_data['energy'])}")
            report.append(f"- Energy range: {self.spectrum_data['energy'].min():.1f}-{self.spectrum_data['energy'].max():.1f} cm⁻¹")
            report.append(f"- Intensity range: {self.spectrum_data['intensity'].min():.4f}-{self.spectrum_data['intensity'].max():.4f}")
            report.append("")
        
        # Peak detection results
        if self.peak_data:
            report.append("## Peak Detection Results")
            report.append(f"- Number of peaks detected: {len(self.peak_data['positions'])}")
            report.append(f"- Peak density: {self.features.get('peak_density', 0):.4f} peaks/cm⁻¹")
            report.append("")
        
        # Baseline detection results
        if self.baseline_data:
            report.append("## Baseline Detection Results")
            report.append(f"- Detector used: {self.baseline_data['detector_name']}")
            report.append(f"- Processing time: {self.baseline_data['processing_time']:.3f}s")
            report.append(f"- Parameter optimization: {'Enabled' if self.enable_parameter_optimization else 'Disabled'}")
            
            if self.baseline_quality_metrics:
                report.append("### Quality Metrics")
                report.append(f"- RMSE: {self.baseline_quality_metrics['rmse']:.6f}")
                report.append(f"- Correlation: {self.baseline_quality_metrics['correlation']:.4f}")
                report.append(f"- Smoothness: {self.baseline_quality_metrics['smoothness']:.6f}")
                report.append(f"- Peak preservation: {self.baseline_quality_metrics['peak_preservation']:.4f}")
                report.append(f"- Quality score: {self.baseline_quality_metrics['quality_score']:.6f}")
            
            report.append("")
        
        # Fitting results
        if self.fit_results:
            report.append("## Gaussian Fitting Results")
            report.append(f"- R²: {self.fit_results['r_squared']:.4f}")
            report.append(f"- RMSE: {self.fit_results['rmse']:.4f}")
            report.append(f"- Number of fitted peaks: {len(self.fit_results['peak_params'])}")
            report.append("")
        
        # Feature summary
        if self.features:
            report.append("## Feature Summary")
            report.append("### Peak Features")
            report.append(f"- Mean amplitude: {self.features.get('mean_amplitude', 0):.4f}")
            report.append(f"- Mean FWHM: {self.features.get('mean_fwhm', 0):.2f} cm⁻¹")
            report.append(f"- Total area: {self.features.get('total_area', 0):.4f}")
            report.append("")
            
            if 'detected_baseline_area' in self.features:
                report.append("### Baseline Features")
                report.append(f"- Baseline area: {self.features['detected_baseline_area']:.4f}")
                report.append(f"- Signal above baseline: {self.features['signal_above_baseline_area']:.4f}")
                report.append(f"- Baseline fraction: {self.features['detected_baseline_area_fraction']:.4f}")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if self.baseline_quality_metrics:
            if self.baseline_quality_metrics['quality_score'] < 0.1:
                report.append("- Baseline detection quality is excellent")
            elif self.baseline_quality_metrics['quality_score'] < 0.3:
                report.append("- Baseline detection quality is good")
            else:
                report.append("- Consider trying different baseline detection parameters")
        
        if self.fit_results and self.fit_results['r_squared'] < 0.8:
            report.append("- Consider adjusting peak detection parameters for better fitting")
        
        report_content = '\n'.join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            print(f"✓ Enhanced analysis report saved: {save_path}")
        
        return report_content

def analyze_spectrum_with_enhanced_baseline(filepath: str, 
                                          baseline_detector_type: str = 'pybaselines_asls',
                                          enable_parameter_optimization: bool = False,
                                          validation_data: Dict = None,
                                          **kwargs) -> EnhancedMLPeakAnalyzer:
    """
    Convenience function to analyze a spectrum with enhanced baseline detection.
    
    Parameters:
    -----------
    filepath : str
        Path to the spectrum file
    baseline_detector_type : str
        Type of baseline detector to use
    enable_parameter_optimization : bool
        Whether to enable parameter optimization
    validation_data : dict, optional
        Validation data for parameter optimization
    **kwargs : dict
        Additional parameters for peak detection
        
    Returns:
    --------
    EnhancedMLPeakAnalyzer
        Analyzer with complete analysis results
    """
    
    # Create analyzer
    analyzer = EnhancedMLPeakAnalyzer(
        baseline_detector_type=baseline_detector_type,
        enable_parameter_optimization=enable_parameter_optimization,
        validation_data=validation_data
    )
    
    # Load spectrum
    analyzer.load_spectrum_data(filepath)
    
    # Detect enhanced baseline
    analyzer.detect_enhanced_baseline()
    
    # Detect peaks
    analyzer.detect_peaks_from_spectrum(**kwargs)
    
    # Fit Gaussians
    analyzer.fit_global_gaussians()
    
    # Extract features
    analyzer.extract_enhanced_ml_features()
    
    return analyzer 