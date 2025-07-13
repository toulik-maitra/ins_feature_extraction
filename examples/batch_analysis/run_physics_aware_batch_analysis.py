#!/usr/bin/env python3
"""
Non-interactive batch analysis launcher for INS spectra with physics-aware baseline detection.
Organizes results in structured directories with physics-aware baseline detection.
Automatically creates clean ML dataset after analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.batch_ml_analysis import BatchMLAnalyzer

# Import the clean ML dataset creation function
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples', 'ml_integration'))
from create_clean_ml_dataset import create_clean_ml_dataset

# Define paths
simulated_dir = "/Users/toulikmaitra/Documents/UC Davis/1. Projects/Molecule Analysis/24- Structural Entropy/3- Anthracene/INS_spectra_all"
experimental_file = "/Users/toulikmaitra/Documents/UC Davis/1. Projects/Molecule Analysis/3-Peak and Background detection/Experimental_INS_files/Normalised_Data/Anthracene_INS_Experimental.csv"

print("="*80)
print("BATCH ANALYSIS WITH PHYSICS-AWARE BASELINE DETECTION")
print("="*80)
print("Peak detection parameters:")
print("  - Distance: 2 (exp), 5 (sim)")
print("  - Prominence: 0.005 (exp), 0.02 (sim)")
print("  - Width: 1 (exp), 2 (sim)")
print("  - Smooth window: 3 (all)")
print("="*80)

# Use physics-aware baseline method
baseline_method = "physics_aware_als"
print(f"\nUsing baseline method: {baseline_method}")
print("  - 0-500 cm⁻¹: Fundamentals (0→1 transitions) with less smooth baseline")
print("  - 500-3500 cm⁻¹: Higher order transitions (0→2, 0→3, etc.) with smoother baseline")
print("="*80)

# Initialize batch analyzer
batch_analyzer = BatchMLAnalyzer(output_dir="comprehensive_analysis_results")

# Analyze experimental spectrum
if os.path.exists(experimental_file):
    print("\nAnalyzing experimental spectrum...")
    batch_analyzer.analyze_single_file(
        experimental_file,
        molecule_name="Anthracene_Experimental",
        plot_individual=True,  # Create all plots for experimental
        baseline_detector_type=baseline_method,
        is_experimental=True
    )

# Analyze all simulated spectra
print("\nAnalyzing simulated spectra...")
batch_analyzer.analyze_directory(
    simulated_dir,
    file_pattern="*.csv",
    plot_individual=True,  # Create all plots for each simulated spectrum
    baseline_detector_type=baseline_method,
    is_experimental=False
)

print("\n" + "="*80)
print("ANALYSIS COMPLETED!")
print("="*80)
print("Results organized in:")
print("  comprehensive_analysis_results/")
print("  ├── plots/")
print("  │   ├── main_analysis/      - Main spectrum + fit plots")
print("  │   ├── baseline_detection/ - Baseline analysis plots")
print("  │   ├── peak_detection/     - Peak detection plots")
print("  │   └── kde_density/        - KDE density plots")
print("  ├── features/               - Combined features CSV")
print("  ├── summaries/              - Analysis summaries")
print("  └── logs/                   - Analysis logs")
print("="*80)

# Automatically create clean ML dataset
print("\n" + "="*80)
print("CREATING CLEAN ML DATASET...")
print("="*80)

try:
    # Change to the project root directory for the ML dataset creation
    project_root = os.path.dirname(__file__)
    original_dir = os.getcwd()
    os.chdir(project_root)
    
    # Create clean ML dataset
    clean_dataset = create_clean_ml_dataset()
    
    if clean_dataset is not None:
        print("\n" + "="*80)
        print("CLEAN ML DATASET CREATED SUCCESSFULLY!")
        print("="*80)
        print("File: comprehensive_analysis_results/features/ml_dataset_clean.csv")
        print(f"Features: {len(clean_dataset.columns)}")
        print(f"Samples: {len(clean_dataset)}")
        print("="*80)
    else:
        print("\nFailed to create clean ML dataset")
    
    # Change back to original directory
    os.chdir(original_dir)
    
except Exception as e:
    print(f"\nError creating clean ML dataset: {e}")
    print("You can manually run: python3 examples/ml_integration/create_clean_ml_dataset.py")

print("\n" + "="*80)
print("BATCH ANALYSIS WORKFLOW COMPLETE!")
print("="*80)
print("Next steps:")
print("1. Check comprehensive_analysis_results/features/ml_dataset_clean.csv")
print("2. Review plots in comprehensive_analysis_results/plots/")
print("3. Use the clean dataset for machine learning analysis")
print("="*80) 