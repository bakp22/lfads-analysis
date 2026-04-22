import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression

# 1. Load Behavior
behavior = loadmat('/Users/berenekpinar/Desktop/lfads-analysis/bilbo_20250430_lfads_trialparams_chk.mat')
reaction_times = behavior['trial_RTs'].flatten()

# 2. Load LFADS Factors
def load_factors(path):
    with h5py.File(path, 'r') as f:
        train = f['train_factors'][:]
        valid = f['valid_factors'][:]
    return np.concatenate((train, valid), axis=0)

path_single = '/Users/berenekpinar/Desktop/lfads-analysis/output/04302025/lfads_output_bilbo_CHKDLAY_DLPFC_20250430_20ms_LFADS (1).h5'
path_multi = '/Users/berenekpinar/Desktop/lfads-analysis/output/04302025/lfads_output_bilbo_CHKDLAY_DLPFC_20250430_20ms_LFADS (2).h5'

factors_single = load_factors(path_single)
factors_multi = load_factors(path_multi)

# 3. Exhaustive Window Search Function
def exhaustive_window_search(factors, target, min_bins=3):
    """
    Tests every possible slice (start to end) and returns R^2 scores.
    """
    n_trials, n_time, n_factors = factors.shape
    # Matrix to store results: Rows = Start Bin, Cols = End Bin
    r2_matrix = np.full((n_time, n_time), np.nan)
    
    best_r2 = -np.inf
    best_coords = (0, 0)

    print(f"Scanning {n_time} time bins...")
    
    for start in range(n_time):
        for end in range(start + min_bins, n_time + 1):
            # Extract slice and average across time
            # Resulting shape: (Trials, Factors)
            x_data = factors[:, start:end, :].mean(axis=1)
            
            # Fit and score
            model = LinearRegression().fit(x_data, target)
            r2 = model.score(x_data, target)
            
            # Store in matrix (end-1 for 0-indexing visualization)
            r2_matrix[start, end-1] = r2
            
            if r2 > best_r2:
                best_r2 = r2
                best_coords = (start, end)
                
    return r2_matrix, best_r2, best_coords

# 4. Run Analysis
# Note: min_bins=5 assumes at least 100ms of data for a stable estimate
res_single, r2_s, slice_s = exhaustive_window_search(factors_single, reaction_times)
res_multi, r2_m, slice_m = exhaustive_window_search(factors_multi, reaction_times)

# 5. Output Results
print("-" * 30)
print(f"SINGLE FILE BEST:")
print(f"  Window: Bins {slice_s[0]} to {slice_s[1]}")
print(f"  R-squared: {r2_s:.4f}")
print("-" * 30)
print(f"MULTI FILE BEST:")
print(f"  Window: Bins {slice_m[0]} to {slice_m[1]}")
print(f"  R-squared: {r2_m:.4f}")

# 6. Visualization
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

im1 = ax[0].imshow(res_single, origin='lower', cmap='viridis')
ax[0].set_title(f'Single File R² Heatmap\nBest: {r2_s:.3f}')
ax[0].set_xlabel('End Bin')
ax[0].set_ylabel('Start Bin')
plt.colorbar(im1, ax=ax[0], label='R²')

im2 = ax[1].imshow(res_multi, origin='lower', cmap='viridis')
ax[1].set_title(f'Multi File R² Heatmap\nBest: {r2_m:.3f}')
ax[1].set_xlabel('End Bin')
ax[1].set_ylabel('Start Bin')
plt.colorbar(im2, ax=ax[1], label='R²')

plt.tight_layout()
plt.show()