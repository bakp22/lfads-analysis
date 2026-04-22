import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score

def get_factors_at_time(factors, t_prime_ms, start_time_ms=-400, bin_ms=20):
    """
    Slices the factors at a specific time point t' (in ms).
    Maps real-world time (ms) to the correct bin index in the factors array.
    """
    # Calculate bin index relative to the start of the recording (e.g., -400ms)
    # If t_prime_ms is 0, and start is -400, index will be (0 - (-400)) / 20 = 20
    bin_index = int((t_prime_ms - start_time_ms) / bin_ms)
    
    # Bound the index to the actual data range (Tr)
    Tr = factors.shape[1]
    bin_index = max(0, min(bin_index, Tr - 1))
    
    return factors[:, bin_index, :]

def load_3d_factors(file_path):
    """Concatenates train and valid factors into one (K, Tr, L) array."""
    with h5py.File(file_path, "r") as f:
        train_f = f['train_factors'][:]
        valid_f = f['valid_factors'][:]
        return np.concatenate((train_f, valid_f), axis=0)

# 1. File Paths
path_multi = "output/04302025/lfads_output_bilbo_CHKDLAY_DLPFC_20250430_20ms_LFADS (2).h5"
path_beh   = "/Users/berenakpinar/lfads-torch/bilbo_20250430_lfads_trialparams_chk.mat"

# 2. Load Data
factors_multi = load_3d_factors(path_multi)
behavior = loadmat(path_beh)

# Extract behavioral targets
rt = behavior["trial_RTs"].flatten()
actions = behavior["trial_action_choices"].flatten()
coherence = behavior["trial_coherences"].flatten()
colors = behavior["trial_color_choices"].flatten()

# 3. Align Trials (K)
k_min = min(len(factors_multi), len(rt), len(actions), len(coherence), len(colors))
X_all = factors_multi[:k_min]
y_rt = rt[:k_min]
y_act = actions[:k_min]
y_coh = coherence[:k_min]
y_col = colors[:k_min]

# 4. Temporal Snapshot Loop: 0ms to 800ms
print(f"{'Time (ms)':<10} | {'RT R2':<10} | {'Action Acc':<12} | {'Color Acc':<10} | {'Coh R2':<10}")
print("-" * 65)

# We start at 0ms and move forward in 40ms increments
for t_ms in range(0, 801, 40):
    # Slice the neural state at this specific moment: X_{t=t'}
    X_t = get_factors_at_time(X_all, t_ms, start_time_ms=-400, bin_ms=20)
    
    # RT Regression
    reg_rt = LinearRegression().fit(X_t, y_rt)
    r2_rt = r2_score(y_rt, reg_rt.predict(X_t))
    
    # Action Choice Classification
    clf_act = LogisticRegression(max_iter=1000).fit(X_t, y_act)
    acc_act = clf_act.score(X_t, y_act)
    
    # Color Choice Classification
    clf_col = LogisticRegression(max_iter=1000).fit(X_t, y_col)
    acc_col = clf_col.score(X_t, y_col)
    
    # Coherence Regression
    reg_coh = LinearRegression().fit(X_t, y_coh)
    r2_coh = r2_score(y_coh, reg_coh.predict(X_t))
    
    print(f"{t_ms:<10} | {r2_rt:<10.4f} | {acc_act:<12.4f} | {acc_col:<10.4f} | {r2_coh:<10.4f}")