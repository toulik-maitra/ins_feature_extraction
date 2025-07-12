#!/usr/bin/env python3
"""
Complete INS Analysis Workflow
=============================

This script runs the complete INS analysis workflow:
1. Batch analysis of all spectra
2. Automatic creation of clean ML dataset
3. Summary of results

Usage:
    python3 run_complete_analysis.py
"""

import os
import sys
import subprocess

def run_complete_analysis():
    """Run the complete INS analysis workflow."""
    
    print("="*80)
    print("INS ML ANALYSIS SYSTEM - COMPLETE WORKFLOW")
    print("="*80)
    print()
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the batch analysis script
    batch_script = os.path.join(script_dir, "examples", "batch_processing", "run_batch_analysis.py")
    
    if not os.path.exists(batch_script):
        print(f"Batch analysis script not found: {batch_script}")
        return False
    
    print("Starting complete analysis workflow...")
    print()
    
    try:
        # Run the batch analysis script
        print("Running batch analysis...")
        result = subprocess.run([sys.executable, batch_script], 
                              capture_output=False, 
                              text=True, 
                              cwd=script_dir)
        
        if result.returncode == 0:
            print()
            print("Complete analysis workflow finished successfully!")
            print()
            print("Results available in:")
            print("   comprehensive_analysis_results/")
            print("   ├── features/ml_dataset_clean.csv    (Clean ML dataset)")
            print("   ├── features/all_molecules_features.csv (Complete features)")
            print("   ├── plots/                          (All analysis plots)")
            print("   ├── summaries/                      (Analysis summaries)")
            print("   └── logs/                           (Processing logs)")
            print()
            print("Next steps:")
            print("   1. Use ml_dataset_clean.csv for machine learning")
            print("   2. Review plots in the plots/ directory")
            print("   3. Check summaries for analysis statistics")
            print()
            return True
        else:
            print(f"Batch analysis failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"Error running complete analysis: {e}")
        return False

if __name__ == "__main__":
    success = run_complete_analysis()
    if not success:
        sys.exit(1) 