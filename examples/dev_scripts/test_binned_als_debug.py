import numpy as np
import pandas as pd
from pybaselines import Baseline
import traceback

# File path
csv_file = "/Users/toulikmaitra/Documents/UC Davis/1. Projects/Molecule Analysis/3-Peak and Background detection/Experimental_INS_files/Normalised_Data/Anthracene_INS_Experimental.csv"

# Load data
print(f"Loading: {csv_file}")
df = pd.read_csv(csv_file)
energy = np.asarray(df['x'], dtype=float)
intensity = np.asarray(df['y'], dtype=float)

bins = [(0, 500), (500, 2000), (2000, 3500)]

baseline_fitter = Baseline()

for i, (start, end) in enumerate(bins):
    if i == len(bins) - 1:
        mask = (energy >= start) & (energy <= end)
    else:
        mask = (energy >= start) & (energy < end)
    idx = np.where(mask)[0]
    e_bin = energy[idx]
    y_bin = intensity[idx]
    y_bin = np.asarray(y_bin, dtype=float).flatten()
    valid_mask = np.isfinite(y_bin)
    y_bin = y_bin[valid_mask]
    e_bin = e_bin[valid_mask]
    print(f"\nBin {i} ({start}-{end}):")
    print(f"  Points: {len(y_bin)}")
    print(f"  Energy range: {e_bin.min() if len(e_bin)>0 else 'NA'} - {e_bin.max() if len(e_bin)>0 else 'NA'}")
    print(f"  y_bin shape: {y_bin.shape}, dtype: {y_bin.dtype}, min: {y_bin.min() if len(y_bin)>0 else 'NA'}, max: {y_bin.max() if len(y_bin)>0 else 'NA'}")
    if len(y_bin) < 2:
        print("  Skipping (too few valid points)")
        continue
    try:
        b_bin, _ = baseline_fitter.asls(y_bin, lam=1e5, p=0.01)
        print(f"  ALS baseline computed successfully. b_bin shape: {b_bin.shape}")
    except Exception as e:
        print(f"  ALS failed: {e}")
        traceback.print_exc()
        print(f"  y_bin: {y_bin}")
        print(f"  e_bin: {e_bin}") 