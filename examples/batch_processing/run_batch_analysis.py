#!/usr/bin/env python3
"""
Batch analysis launcher for INS spectra (experimental and simulated).
Organizes results in structured directories with physics-aware baseline detection.
Automatically creates clean ML dataset after analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.batch_ml_analysis import BatchMLAnalyzer
from config.output_config import get_output_dir, create_output_structure, print_output_structure

# Import the clean ML dataset creation function
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_integration'))
from create_clean_ml_dataset import create_clean_ml_dataset

def get_baseline_method():
    """Get baseline detection method from user input."""
    print("\n" + "="*80)
    print("BASELINE DETECTION METHOD SELECTION")
    print("="*80)
    print("Available baseline detection methods:")
    print("1. physics_aware_als    - Physics-aware ALS (0-500: fundamentals, 500-3500: higher order) [DEFAULT]")
    print("2. binned_als          - Binned ALS (0-500, 500-2000, 2000-3500 cm⁻¹)")
    print("3. als                 - Regular ALS (global)")
    print("4. dynamic_rolling     - Dynamic rolling minimum")
    print("5. polynomial          - Polynomial fitting")
    print("6. morphological       - Morphological operations")
    print("="*80)
    
    while True:
        try:
            choice = input("Enter baseline method number (1-6) or press Enter for default (1): ").strip()
            if choice == "":
                return "physics_aware_als"
            
            choice = int(choice)
            if choice == 1:
                return "physics_aware_als"
            elif choice == 2:
                return "binned_als"
            elif choice == 3:
                return "als"
            elif choice == 4:
                return "dynamic_rolling"
            elif choice == 5:
                return "polynomial"
            elif choice == 6:
                return "morphological"
            else:
                print("Invalid choice. Please enter a number between 1-6.")
        except ValueError:
            print("Invalid input. Please enter a number between 1-6.")

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

# Get baseline method from user
baseline_method = get_baseline_method()

print(f"\nSelected baseline method: {baseline_method}")
print("="*80)

# Get output directory and create structure
output_dir = get_output_dir("comprehensive_analysis_results")
dirs = create_output_structure(output_dir)

# Initialize batch analyzer
batch_analyzer = BatchMLAnalyzer(output_dir=str(output_dir))

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
print_output_structure(output_dir)

# Automatically create clean ML dataset
print("\n" + "="*80)
print("CREATING CLEAN ML DATASET...")
print("="*80)

try:
    # Change to the project root directory for the ML dataset creation
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    original_dir = os.getcwd()
    os.chdir(project_root)
    
    # Create clean ML dataset
    clean_dataset = create_clean_ml_dataset(output_dir_name="comprehensive_analysis_results")
    
    if clean_dataset is not None:
        print("\n" + "="*80)
        print("CLEAN ML DATASET CREATED SUCCESSFULLY!")
        print("="*80)
        print(f"File: {output_dir}/features/ml_dataset_clean.csv")
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
print(f"1. Check {output_dir}/features/ml_dataset_clean.csv")
print(f"2. Review plots in {output_dir}/plots/")
print("3. Use the clean dataset for machine learning analysis")
print("="*80) 