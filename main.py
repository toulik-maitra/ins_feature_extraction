#!/usr/bin/env python3
"""
Main entry point for INS ML Analysis System.

This script provides easy access to the main functionality of the system.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.ml_peak_analyzer import MLPeakAnalyzer
from src.core.batch_ml_analysis import BatchMLAnalyzer

def main():
    """Main entry point for the INS ML Analysis System."""
    print("INS ML Analysis System")
    print("=" * 50)
    print()
    print("Available modules:")
    print("1. MLPeakAnalyzer - Single file analysis")
    print("2. BatchMLAnalyzer - Batch processing")
    print()
    print("For examples, see the 'examples/' directory:")
    print("- examples/single_file/ - Single file analysis examples")
    print("- examples/batch_processing/ - Batch processing examples")
    print("- examples/ml_integration/ - ML dataset creation examples")
    print()
    print("For documentation, see the 'docs/' directory.")
    print()
    print("To run batch analysis:")
    print("  python3 examples/batch_processing/run_batch_analysis.py")
    print()
    print("To create clean ML dataset:")
    print("  python3 examples/ml_integration/create_clean_ml_dataset.py")
    print()
    print("For complete workflow:")
    print("  python3 run_complete_analysis.py")

if __name__ == "__main__":
    main() 