"""
Baseline Detection Module
=========================

This module provides various baseline detection algorithms for INS spectra.
The modular design allows easy swapping between different baseline detection methods.

Author: Your Name
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from abc import ABC, abstractmethod
import warnings

class BaselineDetector(ABC):
    """Abstract base class for baseline detection algorithms."""
    
    @abstractmethod
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """
        Detect baseline for given intensity and energy arrays.
        
        Parameters:
        -----------
        intensity : np.ndarray
            Intensity values
        energy : np.ndarray
            Energy values (cm⁻¹)
        **kwargs : dict
            Additional parameters specific to the algorithm
            
        Returns:
        --------
        np.ndarray
            Baseline values
        """
        pass
    
    def validate_inputs(self, intensity: np.ndarray, energy: np.ndarray) -> None:
        """Validate input arrays."""
        if len(intensity) != len(energy):
            raise ValueError("Intensity and energy arrays must have the same length")
        if len(intensity) == 0:
            raise ValueError("Input arrays cannot be empty")
        if np.any(np.isnan(intensity)) or np.any(np.isnan(energy)):
            raise ValueError("Input arrays cannot contain NaN values")

class DynamicRollingBaselineDetector(BaselineDetector):
    """
    Dynamic rolling minimum baseline detector with configurable window sizes
    for different energy ranges.
    """
    
    def __init__(self, 
                 experimental_ranges: Optional[List[Tuple[Tuple[float, float], int]]] = None,
                 simulation_ranges: Optional[List[Tuple[Tuple[float, float], int]]] = None):
        """
        Initialize the detector with energy range configurations.
        
        Parameters:
        -----------
        experimental_ranges : list, optional
            List of (energy_range, window_size) tuples for experimental data
        simulation_ranges : list, optional
            List of (energy_range, window_size) tuples for simulation data
        """
        # Default configurations
        self.experimental_ranges = experimental_ranges or [
            ((0, 500), 80),
            ((500, 2000), 50),  # Special window for experimental data
            ((2000, 3500), 600)
        ]
        
        self.simulation_ranges = simulation_ranges or [
            ((0, 300), 70),
            ((300, 2000), 160),  # Larger window for simulation data
            ((2000, 3500), 600)
        ]
    
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, 
                       is_experimental: bool = False, **kwargs) -> np.ndarray:
        """
        Apply dynamic rolling minimum baseline correction.
        
        Parameters:
        -----------
        intensity : np.ndarray
            Intensity values
        energy : np.ndarray
            Energy values (cm⁻¹)
        is_experimental : bool
            Whether the data is experimental (affects window sizes)
        **kwargs : dict
            Additional parameters (ignored for this method)
            
        Returns:
        --------
        np.ndarray
            Baseline values
        """
        self.validate_inputs(intensity, energy)
        
        baseline = np.zeros_like(intensity)
        ranges_windows = self.experimental_ranges if is_experimental else self.simulation_ranges
        
        for (start, end), window_size in ranges_windows:
            mask = (energy >= start) & (energy < end)
            if np.any(mask):
                baseline[mask] = pd.Series(intensity[mask]).rolling(
                    window=window_size, min_periods=1, center=True).min().values
        
        return baseline

class PolynomialBaselineDetector(BaselineDetector):
    """
    Polynomial baseline detector using iterative fitting.
    """
    
    def __init__(self, degree: int = 3, max_iterations: int = 10, tolerance: float = 1e-6):
        """
        Initialize polynomial baseline detector.
        
        Parameters:
        -----------
        degree : int
            Polynomial degree
        max_iterations : int
            Maximum iterations for iterative fitting
        tolerance : float
            Convergence tolerance
        """
        self.degree = degree
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """
        Detect baseline using polynomial fitting.
        
        Parameters:
        -----------
        intensity : np.ndarray
            Intensity values
        energy : np.ndarray
            Energy values (cm⁻¹)
        **kwargs : dict
            Additional parameters (ignored for this method)
            
        Returns:
        --------
        np.ndarray
            Baseline values
        """
        self.validate_inputs(intensity, energy)
        
        # Normalize energy to prevent numerical issues
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min())
        
        # Initial baseline estimate
        baseline = np.polyval(np.polyfit(energy_norm, intensity, self.degree), energy_norm)
        
        # Iterative refinement
        for _ in range(self.max_iterations):
            # Calculate residuals
            residuals = intensity - baseline
            
            # Identify points below baseline (likely baseline points)
            baseline_mask = residuals < 0
            
            if np.sum(baseline_mask) < self.degree + 1:
                break  # Not enough points for fitting
            
            # Fit polynomial to baseline points
            baseline_energy = energy_norm[baseline_mask]
            baseline_intensity = intensity[baseline_mask]
            
            try:
                coeffs = np.polyfit(baseline_energy, baseline_intensity, self.degree)
                new_baseline = np.polyval(coeffs, energy_norm)
                
                # Check convergence
                if np.max(np.abs(new_baseline - baseline)) < self.tolerance:
                    break
                
                baseline = new_baseline
            except np.RankWarning:
                break
        
        return baseline

class AsymmetricLeastSquaresBaselineDetector(BaselineDetector):
    """
    Asymmetric Least Squares (ALS) baseline detector.
    """
    
    def __init__(self, lambda_param: float = 1e6, p_param: float = 0.01, max_iterations: int = 10):
        """
        Initialize ALS baseline detector.
        
        Parameters:
        -----------
        lambda_param : float
            Smoothness parameter
        p_param : float
            Asymmetry parameter
        max_iterations : int
            Maximum iterations
        """
        self.lambda_param = lambda_param
        self.p_param = p_param
        self.max_iterations = max_iterations
    
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """
        Detect baseline using Asymmetric Least Squares.
        
        Parameters:
        -----------
        intensity : np.ndarray
            Intensity values
        energy : np.ndarray
            Energy values (cm⁻¹)
        **kwargs : dict
            Additional parameters (ignored for this method)
            
        Returns:
        --------
        np.ndarray
            Baseline values
        """
        self.validate_inputs(intensity, energy)
        
        n = len(intensity)
        
        # Create difference matrix
        D = np.zeros((n-2, n))
        for i in range(n-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
        
        # Initialize baseline
        baseline = intensity.copy()
        
        # Iterative ALS algorithm
        for _ in range(self.max_iterations):
            # Calculate weights
            residuals = intensity - baseline
            weights = np.where(residuals > 0, self.p_param, 1 - self.p_param)
            
            # Solve weighted least squares
            W = np.diag(weights)
            A = W + self.lambda_param * D.T @ D
            b = W @ intensity
            
            try:
                new_baseline = np.linalg.solve(A, b)
                
                # Check convergence
                if np.max(np.abs(new_baseline - baseline)) < 1e-6:
                    break
                
                baseline = new_baseline
            except np.linalg.LinAlgError:
                warnings.warn("ALS baseline detection failed due to singular matrix")
                break
        
        return baseline

class MorphologicalBaselineDetector(BaselineDetector):
    """
    Morphological baseline detector using opening operation.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize morphological baseline detector.
        
        Parameters:
        -----------
        window_size : int
            Size of the morphological window
        """
        self.window_size = window_size
    
    def detect_baseline(self, intensity: np.ndarray, energy: np.ndarray, **kwargs) -> np.ndarray:
        """
        Detect baseline using morphological opening.
        
        Parameters:
        -----------
        intensity : np.ndarray
            Intensity values
        energy : np.ndarray
            Energy values (cm⁻¹)
        **kwargs : dict
            Additional parameters (ignored for this method)
            
        Returns:
        --------
        np.ndarray
            Baseline values
        """
        self.validate_inputs(intensity, energy)
        
        # Create structural element
        half_window = self.window_size // 2
        baseline = np.zeros_like(intensity)
        
        # Apply morphological opening (erosion followed by dilation)
        for i in range(len(intensity)):
            start = max(0, i - half_window)
            end = min(len(intensity), i + half_window + 1)
            
            # Erosion (minimum)
            eroded = np.min(intensity[start:end])
            
            # Dilation (maximum) - but we want the minimum after erosion
            # This effectively gives us the baseline
            baseline[i] = eroded
        
        return baseline

class BaselineDetectorFactory:
    """Factory class for creating baseline detectors."""
    
    @staticmethod
    def create_detector(detector_type: str, **kwargs) -> BaselineDetector:
        """
        Create a baseline detector of the specified type.
        
        Parameters:
        -----------
        detector_type : str
            Type of detector ('dynamic_rolling', 'polynomial', 'als', 'morphological')
        **kwargs : dict
            Parameters for the detector
            
        Returns:
        --------
        BaselineDetector
            Configured baseline detector
        """
        detector_map = {
            'dynamic_rolling': DynamicRollingBaselineDetector,
            'polynomial': PolynomialBaselineDetector,
            'als': AsymmetricLeastSquaresBaselineDetector,
            'morphological': MorphologicalBaselineDetector
        }
        
        if detector_type not in detector_map:
            raise ValueError(f"Unknown detector type: {detector_type}. "
                           f"Available types: {list(detector_map.keys())}")
        
        return detector_map[detector_type](**kwargs)

def detect_baseline(intensity: np.ndarray, energy: np.ndarray, 
                   detector_type: str = 'dynamic_rolling', 
                   is_experimental: bool = False, **kwargs) -> np.ndarray:
    """
    Convenience function for baseline detection.
    
    Parameters:
    -----------
    intensity : np.ndarray
        Intensity values
    energy : np.ndarray
        Energy values (cm⁻¹)
    detector_type : str
        Type of baseline detector to use
    is_experimental : bool
        Whether the data is experimental (for dynamic rolling detector)
    **kwargs : dict
        Additional parameters for the detector
        
    Returns:
    --------
    np.ndarray
        Baseline values
    """
    detector = BaselineDetectorFactory.create_detector(detector_type, **kwargs)
    
    if detector_type == 'dynamic_rolling':
        return detector.detect_baseline(intensity, energy, is_experimental=is_experimental)
    else:
        return detector.detect_baseline(intensity, energy)

def calculate_peak_to_baseline_ratios(intensity: np.ndarray, baseline: np.ndarray, 
                                    peak_positions: np.ndarray, peak_amplitudes: np.ndarray) -> Dict:
    """
    Calculate peak-to-baseline ratios and related features.
    
    Parameters:
    -----------
    intensity : np.ndarray
        Intensity values
    baseline : np.ndarray
        Baseline values
    peak_positions : np.ndarray
        Peak positions (indices)
    peak_amplitudes : np.ndarray
        Peak amplitudes
        
    Returns:
    --------
    dict
        Dictionary containing peak-to-baseline ratio features
    """
    if len(peak_positions) == 0:
        return {
            'mean_peak_to_baseline_ratio': 0.0,
            'std_peak_to_baseline_ratio': 0.0,
            'max_peak_to_baseline_ratio': 0.0,
            'min_peak_to_baseline_ratio': 0.0,
            'peak_to_baseline_ratio_range': 0.0,
            'peak_to_baseline_ratio_cv': 0.0,
            'peak_to_baseline_ratio_skewness': 0.0,
            'peak_to_baseline_ratio_kurtosis': 0.0,
            'baseline_ratio_percentile_25': 0.0,
            'baseline_ratio_percentile_75': 0.0,
            'baseline_ratio_iqr': 0.0,
            'baseline_ratio_median': 0.0,
            'high_ratio_peaks': 0,
            'medium_ratio_peaks': 0,
            'low_ratio_peaks': 0,
            'total_baseline_area': 0.0,
            'baseline_intensity_mean': 0.0,
            'baseline_intensity_std': 0.0,
            'baseline_intensity_range': 0.0,
            'signal_to_baseline_ratio': 0.0
        }
    
    # Calculate peak-to-baseline ratios
    baseline_at_peaks = baseline[peak_positions]
    peak_to_baseline_ratios = peak_amplitudes / (baseline_at_peaks + 1e-10)  # Avoid division by zero
    
    # Basic statistics
    mean_ratio = np.mean(peak_to_baseline_ratios)
    std_ratio = np.std(peak_to_baseline_ratios)
    max_ratio = np.max(peak_to_baseline_ratios)
    min_ratio = np.min(peak_to_baseline_ratios)
    ratio_range = max_ratio - min_ratio
    ratio_cv = std_ratio / mean_ratio if mean_ratio > 0 else 0
    
    # Distribution statistics
    def calculate_skewness(data):
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_kurtosis(data):
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    ratio_skewness = calculate_skewness(peak_to_baseline_ratios)
    ratio_kurtosis = calculate_kurtosis(peak_to_baseline_ratios)
    
    # Percentile statistics
    ratio_percentile_25 = np.percentile(peak_to_baseline_ratios, 25)
    ratio_percentile_75 = np.percentile(peak_to_baseline_ratios, 75)
    ratio_iqr = ratio_percentile_75 - ratio_percentile_25
    ratio_median = np.median(peak_to_baseline_ratios)
    
    # Ratio categories
    high_ratio_peaks = np.sum(peak_to_baseline_ratios > 10)
    medium_ratio_peaks = np.sum((peak_to_baseline_ratios >= 3) & (peak_to_baseline_ratios <= 10))
    low_ratio_peaks = np.sum(peak_to_baseline_ratios < 3)
    
    # Baseline statistics
    total_baseline_area = np.trapz(baseline)
    baseline_intensity_mean = np.mean(baseline)
    baseline_intensity_std = np.std(baseline)
    baseline_intensity_range = np.max(baseline) - np.min(baseline)
    
    # Overall signal-to-baseline ratio
    signal_to_baseline_ratio = np.trapz(intensity) / (total_baseline_area + 1e-10)
    
    return {
        'mean_peak_to_baseline_ratio': mean_ratio,
        'std_peak_to_baseline_ratio': std_ratio,
        'max_peak_to_baseline_ratio': max_ratio,
        'min_peak_to_baseline_ratio': min_ratio,
        'peak_to_baseline_ratio_range': ratio_range,
        'peak_to_baseline_ratio_cv': ratio_cv,
        'peak_to_baseline_ratio_skewness': ratio_skewness,
        'peak_to_baseline_ratio_kurtosis': ratio_kurtosis,
        'baseline_ratio_percentile_25': ratio_percentile_25,
        'baseline_ratio_percentile_75': ratio_percentile_75,
        'baseline_ratio_iqr': ratio_iqr,
        'baseline_ratio_median': ratio_median,
        'high_ratio_peaks': high_ratio_peaks,
        'medium_ratio_peaks': medium_ratio_peaks,
        'low_ratio_peaks': low_ratio_peaks,
        'total_baseline_area': total_baseline_area,
        'baseline_intensity_mean': baseline_intensity_mean,
        'baseline_intensity_std': baseline_intensity_std,
        'baseline_intensity_range': baseline_intensity_range,
        'signal_to_baseline_ratio': signal_to_baseline_ratio
    } 