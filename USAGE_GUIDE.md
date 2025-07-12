# INS ML Analysis System - Usage Guide

## Quick Start Options

### Option 1: Complete Workflow (Recommended)
Run everything automatically - batch analysis + clean ML dataset creation:
```bash
python3 run_complete_analysis.py
```

### Option 2: Step-by-Step
Run individual components as needed:

#### A. System Overview
```bash
python3 main.py                    # View system overview
python3 quick_start.py             # Quick start demonstration
```

#### B. Batch Analysis (with automatic ML dataset creation)
```bash
python3 examples/batch_processing/run_batch_analysis.py
```

#### C. Single File Analysis
```bash
python3 examples/single_file/test_ml_analyzer.py
```

#### D. Manual ML Dataset Creation (if needed)
```bash
python3 examples/ml_integration/create_clean_ml_dataset.py
```

## What Each Option Does

### `run_complete_analysis.py` (Recommended)
- Runs batch analysis on all spectra
- Automatically creates clean ML dataset
- Provides complete workflow summary
- One command does everything

### `run_batch_analysis.py`
- Analyzes experimental spectrum
- Analyzes all simulated spectra
- Creates all plots and features
- Automatically creates clean ML dataset
- Organizes results in structured directories

### `create_clean_ml_dataset.py`
- Reads comprehensive analysis results
- Removes unnecessary features
- Creates ML-ready dataset with 61 features
- Includes sample identifiers
- Applies outlier removal

## Output Structure

After running any analysis, you'll get:

```
comprehensive_analysis_results/
├── features/
│   ├── ml_dataset_clean.csv          # Clean ML dataset (61 features, 56 samples)
│   ├── all_molecules_features.csv    # Complete dataset with all features
│   └── [molecule]_features.csv       # Individual feature files
├── plots/
│   ├── main_analysis/                # Main spectrum analysis plots
│   ├── baseline_detection/           # Baseline detection plots
│   ├── peak_detection/               # Peak detection plots
│   └── kde_density/                  # KDE density distribution plots
├── summaries/                        # Analysis summaries
└── logs/                             # Processing logs
```

## Recommended Workflow

1. **Start with complete workflow**: `python3 run_complete_analysis.py`
2. **Check results**: Review `ml_dataset_clean.csv` and plots
3. **Use for ML**: The clean dataset is ready for machine learning
4. **Customize if needed**: Run individual scripts for specific needs

## Troubleshooting

### If complete workflow fails:
1. Try running batch analysis directly: `python3 examples/batch_processing/run_batch_analysis.py`
2. Check if all dependencies are installed: `pip3 install -r requirements.txt`
3. Verify file paths in the batch analysis script

### If ML dataset creation fails:
1. Ensure batch analysis completed successfully
2. Check that `comprehensive_analysis_results/features/all_molecules_features.csv` exists
3. Run manually: `python3 examples/ml_integration/create_clean_ml_dataset.py`

## Next Steps

After successful analysis:
1. Use `ml_dataset_clean.csv` for machine learning analysis
2. Review plots in the `plots/` directory
3. Check analysis summaries for statistics
4. Customize analysis parameters if needed

The system is designed to be robust and provide comprehensive results with minimal user intervention. 