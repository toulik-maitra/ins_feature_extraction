"""
INS ML Analysis System - Source Package
=======================================

This package contains the core analysis modules for INS spectrum processing.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.ml_peak_analyzer import MLPeakAnalyzer
from .core.batch_ml_analysis import BatchMLAnalyzer

__all__ = ["MLPeakAnalyzer", "BatchMLAnalyzer"] 