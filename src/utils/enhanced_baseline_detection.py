"""
Enhanced Baseline Detection Module
==================================

This module provides comprehensive baseline detection algorithms for INS spectra
with systematic parameter optimization and validation capabilities.

Features:
- Multiple baseline algorithms (pybaselines, spectrochempy, custom)
- Systematic parameter optimization
- Validation with simulated data
- Quantitative quality metrics
- Comprehensive documentation of choices

Author: Enhanced Baseline Detection System
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union, Any
from abc import ABC, abstractmethod
import warnings
import time
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import external libraries
try:
    import pybaselines
    from pybaselines import Baseline
    PYBASELINES_AVAILABLE = True
except ImportError:
    PYBASELINES_AVAILABLE = False
    warnings.warn("pybaselines not available. Install with: pip install pybaselines")

try:
    import spectrochempy as scp
    SPECTROCHEMPY_AVAILABLE = True
except ImportError:
    SPECTROCHEMPY_AVAILABLE = False
    warnings.warn("spectrochempy not available. Install with: pip install spectrochempy")

class BaselineQualityMetrics:
    """Class for calculating baseline quality metrics."""
    
    @staticmethod
    def calculate_rmse(true_baseline: np.ndarray, estimated_baseline: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(np.mean((true_baseline - estimated_baseline) ** 2))
    
    @staticmethod
    def calculate_mae(true_baseline: np.ndarray, estimated_baseline: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(true_baseline - estimated_baseline))
    
    @staticmethod
    def calculate_correlation(true_baseline: np.ndarray, estimated_baseline: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        return pearsonr(true_baseline, estimated_baseline)[0]
    
    @staticmethod
    def calculate_smoothness(baseline: np.ndarray) -> float:
        """Calculate baseline smoothness (lower is smoother)."""
        return np.mean(np.abs(np.diff(baseline)))
    
    @staticmethod
    def calculate_peak_preservation(original_spectrum: np.ndarray, 
                                  corrected_spectrum: np.ndarray,
                                  peak_positions: np.ndarray) -> float:
        """Calculate how well peaks are preserved after baseline correction."""
        if len(peak_positions) == 0:
            return 0.0
        
        original_peaks = original_spectrum[peak_positions]
        corrected_peaks = corrected_spectrum[peak_positions]
        
        # Calculate peak height preservation
        peak_ratios = corrected_peaks / (original_peaks + 1e-10)
        return np.mean(peak_ratios)
    
    @staticmethod
    def calculate_baseline_quality_score(true_baseline: np.ndarray, 
                                       estimated_baseline: np.ndarray,
                                       original_spectrum: np.ndarray,
                                       corrected_spectrum: np.ndarray,
                                       peak_positions: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive quality score for baseline detection."""
        
        # Accuracy metrics
        rmse = BaselineQualityMetrics.calculate_rmse(true_baseline, estimated_baseline)
        mae = BaselineQualityMetrics.calculate_mae(true_baseline, estimated_baseline)
        correlation = BaselineQualityMetrics.calculate_correlation(true_baseline, estimated_baseline)
        
        # Smoothness metric
        smoothness = BaselineQualityMetrics.calculate_smoothness(estimated_baseline)
        
        # Peak preservation metric
        peak_preservation = BaselineQualityMetrics.calculate_peak_preservation(
            original_spectrum, corrected_spectrum, peak_positions
        )
        
        # Combined quality score (lower is better)
        quality_score = rmse + 0.1 * smoothness - 0.5 * correlation + 0.2 * (1 - peak_preservation)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'smoothness': smoothness,
            'peak_preservation': peak_preservation,
            'quality_score': quality_score
        }

class EnhancedBaselineDetector(ABC):
    """Abstract base class for enhanced baseline detection algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.optimization_history = []
    
    @abstractmethod
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """Detect baseline for given intensity and energy arrays."""
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges for optimization."""
        pass
    
    def optimize_parameters(self, intensity: np.ndarray, energy: np.ndarray,
                          true_baseline: np.ndarray, peak_positions: np.ndarray,
                          max_iterations: int = 50) -> Dict[str, Any]:
        """Optimize parameters using simulated data with known ground truth."""
        
        print(f"Optimizing parameters for {self.name}...")
        
        best_score = float('inf')
        best_params = None
        optimization_history = []
        
        param_ranges = self.get_parameter_ranges()
        
        for iteration in range(max_iterations):
            # Generate random parameters within ranges
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            try:
                # Detect baseline with current parameters
                estimated_baseline = self.detect_baseline(intensity, energy, **params)
                corrected_spectrum = intensity - estimated_baseline
                
                # Calculate quality metrics
                metrics = BaselineQualityMetrics.calculate_baseline_quality_score(
                    true_baseline, estimated_baseline, intensity, corrected_spectrum, peak_positions
                )
                
                optimization_history.append({
                    'iteration': iteration,
                    'parameters': params.copy(),
                    'quality_score': metrics['quality_score'],
                    'rmse': metrics['rmse'],
                    'correlation': metrics['correlation']
                })
                
                # Update best parameters if better
                if metrics['quality_score'] < best_score:
                    best_score = metrics['quality_score']
                    best_params = params.copy()
                    print(f"  Iteration {iteration}: New best score = {best_score:.6f}")
                
            except Exception as e:
                print(f"  Iteration {iteration}: Error - {e}")
                continue
        
        # Set best parameters
        if best_params:
            self.parameters.update(best_params)
            print(f"✓ Optimization completed. Best score: {best_score:.6f}")
        
        self.optimization_history = optimization_history
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_history': optimization_history
        }

class PyBaselinesDetector(EnhancedBaselineDetector):
    """Baseline detector using pybaselines library."""
    
    def __init__(self, method: str = 'asls'):
        super().__init__(f"pybaselines_{method}")
        self.method = method
        self.baseline_fitter = None
        
        if not PYBASELINES_AVAILABLE:
            raise ImportError("pybaselines is required for this detector")
        
        # Initialize baseline fitter
        self.baseline_fitter = Baseline()
    
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """Detect baseline using pybaselines."""
        if self.baseline_fitter is None:
            raise RuntimeError("Baseline fitter not initialized")
        method_params = self._get_method_parameters(**kwargs)
        try:
            # Use correct argument names for pybaselines >=1.2.0
            # All methods: y, lam, p, niter, diff_order, etc.
            # Only pass params that are valid for the method
            method_func = getattr(self.baseline_fitter, self.method)
            # Remove None values
            method_params = {k: v for k, v in method_params.items() if v is not None}
            baseline, params = method_func(intensity, **method_params)
            return baseline
        except Exception as e:
            raise RuntimeError(f"PyBaselines {self.method} failed: {e}")
    
    def _get_method_parameters(self, **kwargs) -> Dict[str, Any]:
        # Map old names to new pybaselines names
        # Only include params that are valid for the method
        if self.method in ['asls', 'airpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa']:
            return {
                'lam': kwargs.get('lam', 1e6),
                'p': kwargs.get('p', 0.01),
                'max_iter': kwargs.get('max_iter', 50),
                'diff_order': kwargs.get('diff_order', 2),
                # 'eta' only for drpls
                'eta': kwargs.get('eta', 0.5) if self.method == 'drpls' else None
            }
        else:
            return kwargs
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        if self.method in ['asls', 'airpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa']:
            return {
                'lam': (1e3, 1e8),
                'p': (0.001, 0.1),
                'niter': (10, 100),
                'diff_order': (1, 3)
            }
        else:
            return {}

class SpectroChemPyDetector(EnhancedBaselineDetector):
    """Baseline detector using spectrochempy library."""
    
    def __init__(self, method: str = 'polynomial'):
        super().__init__(f"spectrochempy_{method}")
        self.method = method
        if not SPECTROCHEMPY_AVAILABLE:
            raise ImportError("spectrochempy is required for this detector")
    
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        try:
            # Create Baseline object with correct parameters
            baseline_obj = scp.Baseline()
            
            if self.method == 'polynomial':
                order = kwargs.get('order', 3)
                baseline_obj.model = 'polynomial'
                baseline_obj.order = order
            elif self.method == 'pchip':
                baseline_obj.model = 'polynomial'
                baseline_obj.order = 'pchip'
            elif self.method == 'asls':
                baseline_obj.model = 'asls'
                baseline_obj.lamb = kwargs.get('lamb', 1e6)
                baseline_obj.asymmetry = kwargs.get('asymmetry', 0.05)
                baseline_obj.max_iter = kwargs.get('max_iter', 50)
            elif self.method == 'snip':
                baseline_obj.model = 'snip'
                baseline_obj.snip_width = kwargs.get('snip_width', 0)
            else:
                baseline_obj.model = self.method
            
            # Create dataset and fit baseline
            dataset = scp.NDDataset(intensity, coordset=[energy])
            baseline_obj.fit(dataset)
            baseline = baseline_obj.baseline.data
            
            return baseline
        except Exception as e:
            raise RuntimeError(f"SpectroChemPy {self.method} failed: {e}")
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        if self.method == 'polynomial':
            return {'order': (1, 6)}
        elif self.method == 'asls':
            return {'lamb': (1e3, 1e8), 'asymmetry': (0.001, 0.1), 'max_iter': (10, 100)}
        elif self.method == 'snip':
            return {'snip_width': (0, 50)}
        else:
            return {}

class AdvancedRollingBaselineDetector(EnhancedBaselineDetector):
    """Advanced rolling baseline detector with multiple window strategies."""
    
    def __init__(self, strategy: str = 'adaptive'):
        super().__init__(f"advanced_rolling_{strategy}")
        self.strategy = strategy
    
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """Detect baseline using advanced rolling strategies."""
        
        if self.strategy == 'adaptive':
            return self._adaptive_rolling(intensity, energy, **kwargs)
        elif self.strategy == 'multi_scale':
            return self._multi_scale_rolling(intensity, energy, **kwargs)
        elif self.strategy == 'percentile':
            return self._percentile_rolling(intensity, energy, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _adaptive_rolling(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """Adaptive rolling baseline based on local signal characteristics."""
        
        window_size = kwargs.get('window_size', 50)
        sensitivity = kwargs.get('sensitivity', 0.1)
        
        baseline = np.zeros_like(intensity)
        
        for i in range(len(intensity)):
            # Calculate local window
            start = max(0, i - window_size // 2)
            end = min(len(intensity), i + window_size // 2 + 1)
            
            local_intensity = intensity[start:end]
            local_energy = energy[start:end]
            
            # Calculate local statistics
            local_std = np.std(local_intensity)
            local_mean = np.mean(local_intensity)
            
            # Adaptive threshold based on local noise
            threshold = local_mean - sensitivity * local_std
            
            # Find baseline points (below threshold)
            baseline_mask = local_intensity <= threshold
            
            if np.sum(baseline_mask) > 0:
                baseline[i] = np.mean(local_intensity[baseline_mask])
            else:
                baseline[i] = np.min(local_intensity)
        
        return baseline
    
    def _multi_scale_rolling(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """Multi-scale rolling baseline using different window sizes."""
        
        window_sizes = kwargs.get('window_sizes', [20, 50, 100])
        weights = kwargs.get('weights', [0.5, 0.3, 0.2])
        
        if len(window_sizes) != len(weights):
            weights = [1.0 / len(window_sizes)] * len(window_sizes)
        
        baselines = []
        
        for window_size in window_sizes:
            baseline = pd.Series(intensity).rolling(
                window=window_size, min_periods=1, center=True
            ).min().values
            baselines.append(baseline)
        
        # Weighted combination
        combined_baseline = np.zeros_like(intensity)
        for baseline, weight in zip(baselines, weights):
            combined_baseline += weight * baseline
        
        return combined_baseline
    
    def _percentile_rolling(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """Percentile-based rolling baseline."""
        
        window_size = kwargs.get('window_size', 50)
        percentile = kwargs.get('percentile', 10)
        
        baseline = pd.Series(intensity).rolling(
            window=window_size, min_periods=1, center=True
        ).quantile(percentile / 100.0).values
        
        return baseline
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter ranges for optimization."""
        
        if self.strategy == 'adaptive':
            return {
                'window_size': (10, 200),
                'sensitivity': (0.01, 0.5)
            }
        elif self.strategy == 'multi_scale':
            return {
                'window_sizes': [(10, 50), (50, 150), (100, 300)],
                'weights': [(0.1, 0.9), (0.1, 0.9), (0.1, 0.9)]
            }
        elif self.strategy == 'percentile':
            return {
                'window_size': (10, 200),
                'percentile': (1, 25)
            }
        else:
            return {}

class BinnedALSDetector(EnhancedBaselineDetector):
    """ALS baseline fitted in bins and stitched together."""
    def __init__(self, bins=None):
        super().__init__("binned_als")
        self.bins = bins if bins is not None else [(0, 500), (500, 2000), (2000, 3500)]
        if not PYBASELINES_AVAILABLE:
            raise ImportError("pybaselines is required for this detector")
        self.baseline_fitter = Baseline()
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        baseline = np.zeros_like(intensity)
        valid_bin_idxs = []
        valid_bin_vals = []
        n_bins = len(self.bins)
        for i, (start, end) in enumerate(self.bins):
            if i == n_bins - 1:
                mask = (energy >= start) & (energy <= end)
            else:
                mask = (energy >= start) & (energy < end)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                print(f"BinnedALS: Bin {i} ({start}-{end}) is empty.")
                continue
            e_bin = energy[idx]
            y_bin = intensity[idx]
            y_bin = np.asarray(y_bin, dtype=float).flatten()
            valid_mask = np.isfinite(y_bin)
            if not np.all(valid_mask):
                print(f"  Bin {i} contains NaN/Inf, removing invalid points.")
                y_bin = y_bin[valid_mask]
                idx = idx[valid_mask]
                e_bin = e_bin[valid_mask]
            print(f"BinnedALS: Bin {i} ({start}-{end}), points: {len(idx)}, energy range: {e_bin.min():.2f}-{e_bin.max():.2f}, shape: {y_bin.shape}, dtype: {y_bin.dtype}, min: {y_bin.min():.2f}, max: {y_bin.max():.2f}")
            if len(y_bin) < 2:
                print(f"  Skipping bin {i} (too few valid points)")
                continue
            lam = kwargs.get('lam', 1e5)
            p = kwargs.get('p', 0.01)
            try:
                b_bin, _ = Baseline().asls(y_bin, lam=lam, p=p)
            except Exception as e:
                print(f"  ALS failed in bin {i}: {e}\n  y_bin: {y_bin}\n  idx: {idx}\n  e_bin: {e_bin}")
                b_bin = np.min(y_bin) * np.ones_like(y_bin)
            baseline[idx] = b_bin
            valid_bin_idxs.extend(idx)
            valid_bin_vals.extend(b_bin)
        # Interpolate over empty bins
        empty = (baseline == 0)
        if np.any(empty):
            from scipy.interpolate import interp1d
            valid_idx = np.array(valid_bin_idxs)
            valid_vals = np.array(valid_bin_vals)
            if len(valid_idx) > 1:
                interp_func = interp1d(energy[valid_idx], valid_vals, kind='linear', fill_value='extrapolate')
                baseline[empty] = interp_func(energy[empty])
            else:
                baseline[empty] = np.min(intensity)
        # Smooth at bin edges
        from scipy.ndimage import gaussian_filter1d
        baseline = gaussian_filter1d(baseline, sigma=5)
        return baseline
    def get_parameter_ranges(self):
        return {'lam': (1e3, 1e7), 'p': (0.001, 0.1)}

class BaselineValidationSystem:
    """System for validating baseline detection algorithms."""
    
    def __init__(self):
        self.validation_results = {}
    
    def create_synthetic_spectrum(self, energy: np.ndarray, 
                                 peak_positions: List[float],
                                 peak_amplitudes: List[float],
                                 peak_widths: List[float],
                                 baseline_type: str = 'polynomial',
                                 noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic spectrum with known baseline for validation."""
        
        # Create true baseline
        if baseline_type == 'polynomial':
            coeffs = np.random.uniform(-0.1, 0.1, 4)
            true_baseline = np.polyval(coeffs, energy)
        elif baseline_type == 'exponential':
            true_baseline = 0.1 * np.exp(-energy / 1000)
        elif baseline_type == 'linear':
            true_baseline = 0.05 + 0.0001 * energy
        else:
            true_baseline = np.zeros_like(energy)
        
        # Create peaks
        spectrum = true_baseline.copy()
        for pos, amp, width in zip(peak_positions, peak_amplitudes, peak_widths):
            peak = amp * np.exp(-(energy - pos)**2 / (2 * width**2))
            spectrum += peak
        
        # Add noise
        noise = np.random.normal(0, noise_level, len(energy))
        spectrum += noise
        
        return spectrum, true_baseline
    
    def validate_detector(self, detector: EnhancedBaselineDetector,
                         energy: np.ndarray,
                         peak_positions: List[float],
                         peak_amplitudes: List[float],
                         peak_widths: List[float],
                         baseline_type: str = 'polynomial',
                         noise_level: float = 0.01,
                         n_trials: int = 10) -> Dict[str, Any]:
        """Validate a baseline detector with synthetic data."""
        
        print(f"Validating {detector.name}...")
        
        all_metrics = []
        
        for trial in range(n_trials):
            # Create synthetic spectrum
            spectrum, true_baseline = self.create_synthetic_spectrum(
                energy, peak_positions, peak_amplitudes, peak_widths,
                baseline_type, noise_level
            )
            
            try:
                # Detect baseline
                estimated_baseline = detector.detect_baseline(spectrum, energy)
                corrected_spectrum = spectrum - estimated_baseline
                
                # Calculate metrics
                metrics = BaselineQualityMetrics.calculate_baseline_quality_score(
                    true_baseline, estimated_baseline, spectrum, corrected_spectrum,
                    np.array(peak_positions, dtype=int)
                )
                
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"  Trial {trial}: Error - {e}")
                continue
        
        if not all_metrics:
            return {'error': 'All trials failed'}
        
        # Aggregate results
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated_metrics[f'{key}_mean'] = np.mean(values)
            aggregated_metrics[f'{key}_std'] = np.std(values)
            aggregated_metrics[f'{key}_min'] = np.min(values)
            aggregated_metrics[f'{key}_max'] = np.max(values)
        
        return aggregated_metrics
    
    def compare_detectors(self, detectors: List[EnhancedBaselineDetector],
                         energy: np.ndarray,
                         peak_positions: List[float],
                         peak_amplitudes: List[float],
                         peak_widths: List[float],
                         baseline_types: List[str] = None,
                         noise_levels: List[float] = None) -> pd.DataFrame:
        """Compare multiple baseline detectors."""
        
        if baseline_types is None:
            baseline_types = ['polynomial', 'exponential', 'linear']
        
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1]
        
        results = []
        
        for detector in detectors:
            for baseline_type in baseline_types:
                for noise_level in noise_levels:
                    metrics = self.validate_detector(
                        detector, energy, peak_positions, peak_amplitudes, peak_widths,
                        baseline_type, noise_level
                    )
                    
                    if 'error' not in metrics:
                        result = {
                            'detector': detector.name,
                            'baseline_type': baseline_type,
                            'noise_level': noise_level,
                            **metrics
                        }
                        results.append(result)
        
        return pd.DataFrame(results)

class EnhancedBaselineDetectorFactory:
    """Factory for creating enhanced baseline detectors."""
    
    @staticmethod
    def create_detector(detector_type: str, **kwargs) -> EnhancedBaselineDetector:
        """Create an enhanced baseline detector."""
        
        if detector_type.startswith('pybaselines_'):
            method = detector_type.split('_', 1)[1]
            return PyBaselinesDetector(method, **kwargs)
        
        elif detector_type.startswith('spectrochempy_'):
            method = detector_type.split('_', 1)[1]
            return SpectroChemPyDetector(method, **kwargs)
        
        elif detector_type.startswith('advanced_rolling_'):
            strategy = detector_type.split('_', 2)[2]
            return AdvancedRollingBaselineDetector(strategy, **kwargs)
        
        elif detector_type == 'binned_als':
            return BinnedALSDetector(**kwargs)
        
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    @staticmethod
    def get_available_detectors() -> List[str]:
        """Get list of available detector types."""
        
        detectors = []
        
        # PyBaselines detectors
        if PYBASELINES_AVAILABLE:
            pybaselines_methods = ['asls', 'airpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa']
            detectors.extend([f'pybaselines_{method}' for method in pybaselines_methods])
        
        # SpectroChemPy detectors
        if SPECTROCHEMPY_AVAILABLE:
            spectrochempy_methods = ['polynomial', 'pchip', 'asls', 'snip']
            detectors.extend([f'spectrochempy_{method}' for method in spectrochempy_methods])
        
        # Advanced rolling detectors
        rolling_strategies = ['adaptive', 'multi_scale', 'percentile']
        detectors.extend([f'advanced_rolling_{strategy}' for strategy in rolling_strategies])
        
        detectors.append('binned_als')
        
        return detectors

def detect_enhanced_baseline(intensity: np.ndarray, energy: np.ndarray,
                           detector_type: str = 'pybaselines_asls',
                           optimize_parameters: bool = False,
                           validation_data: Dict = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Enhanced baseline detection with optional parameter optimization and validation.
    
    Parameters:
    -----------
    intensity : np.ndarray
        Intensity values
    energy : np.ndarray
        Energy values (cm⁻¹)
    detector_type : str
        Type of baseline detector
    optimize_parameters : bool
        Whether to optimize parameters using validation data
    validation_data : dict, optional
        Validation data with true baseline and peak positions
    **kwargs : dict
        Additional parameters for the detector
        
    Returns:
    --------
    dict
        Dictionary containing baseline, detector info, and quality metrics
    """
    
    # Create detector
    detector = EnhancedBaselineDetectorFactory.create_detector(detector_type, **kwargs)
    
    # Optimize parameters if requested
    if optimize_parameters and validation_data is not None:
        optimization_result = detector.optimize_parameters(
            intensity=validation_data['intensity'],
            energy=validation_data['energy'],
            true_baseline=validation_data['true_baseline'],
            peak_positions=validation_data['peak_positions']
        )
        print(f"Parameter optimization completed: {optimization_result['best_score']:.6f}")
    
    # Detect baseline
    start_time = time.time()
    baseline = detector.detect_baseline(intensity, energy)
    processing_time = time.time() - start_time
    
    # Calculate quality metrics if validation data is available
    quality_metrics = None
    if validation_data is not None:
        corrected_spectrum = intensity - baseline
        quality_metrics = BaselineQualityMetrics.calculate_baseline_quality_score(
            validation_data['true_baseline'], baseline, intensity, corrected_spectrum,
            validation_data['peak_positions']
        )
    
    return {
        'baseline': baseline,
        'detector_name': detector.name,
        'detector_type': detector_type,
        'parameters': detector.parameters,
        'processing_time': processing_time,
        'quality_metrics': quality_metrics,
        'optimization_history': detector.optimization_history
    } 