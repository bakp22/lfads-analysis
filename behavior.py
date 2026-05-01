import h5py
import numpy as np
from scipy.io import loadmat
from scipy.stats import f as f_dist
import matplotlib.pyplot as plt
import time 

def get_factors_at_certain_time_bin(factors, t_prime_ms, start_time_ms=-400, bin_ms=20):
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

def load_3d_factors(file_path, max_trials=None):
    """
    Concatenate train and valid factors into one (K, Tr, L) array.

    If max_trials is provided, only load up to that many total trials.
    This avoids reading the full H5 file when behavior arrays are shorter.
    """
    with h5py.File(file_path, "r") as f:
        train_ds = f["train_factors"]
        valid_ds = f["valid_factors"]

        if max_trials is None:
            train_f = train_ds[:]
            valid_f = valid_ds[:]
            return np.concatenate((train_f, valid_f), axis=0)

        n_train = train_ds.shape[0]
        n_valid = valid_ds.shape[0]
        n_total = n_train + n_valid
        n_keep = int(min(max_trials, n_total))

        n_from_train = min(n_keep, n_train)
        n_from_valid = max(0, n_keep - n_from_train)

        train_f = train_ds[:n_from_train] if n_from_train > 0 else np.empty((0, *train_ds.shape[1:]))
        valid_f = valid_ds[:n_from_valid] if n_from_valid > 0 else np.empty((0, *valid_ds.shape[1:]))
        return np.concatenate((train_f, valid_f), axis=0)

def run_rt_timecourse_regression(
    factors,
    rt,
    start_time_ms=0,
    bin_ms=20,
    alpha=0.05,
    min_r2=0.0,
):
    """
    Fit one linear model per time bin: RT ~ latent state at that bin.

    Returns:
        times_ms: (Tr,) time for each bin
        r2_vals: (Tr,) R^2 values
        p_vals: (Tr,) model F-test p-values
        earliest_sig_time_ms: earliest time satisfying p < alpha and R^2 > min_r2
    """
    K, Tr, _ = factors.shape
    rt = rt.flatten()[:K]

    # Keep only finite RT trials once, then apply same subset across all bins.
    valid_rt = np.isfinite(rt)
    rt_clean = rt[valid_rt]
    factors_clean = factors[valid_rt]

    times_ms = start_time_ms + np.arange(Tr) * bin_ms
    r2_vals = np.full(Tr, np.nan)
    p_vals = np.full(Tr, np.nan)

    for t_idx in range(Tr):
        X_t = factors_clean[:, t_idx, :]
        valid_rows = np.all(np.isfinite(X_t), axis=1) & np.isfinite(rt_clean)

        if valid_rows.sum() <= X_t.shape[1] + 1:
            # Not enough data to fit a full linear model.
            continue

        X_fit = X_t[valid_rows]
        y_fit = rt_clean[valid_rows]

        # OLS via least squares, then compute model-level F-test p-value.
        X_design = np.column_stack([np.ones(X_fit.shape[0]), X_fit])
        beta, _, _, _ = np.linalg.lstsq(X_design, y_fit, rcond=None)
        y_pred = X_design @ beta

        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        if ss_tot <= 0:
            continue

        r2 = 1.0 - (ss_res / ss_tot)
        # Clamp numeric noise to valid R^2 range used in F-stat computation.
        r2 = float(np.clip(r2, 0.0, 0.999999999))

        n = X_fit.shape[0]
        p = X_fit.shape[1]
        if n <= p + 1:
            continue

        f_stat = (r2 / p) / ((1.0 - r2) / (n - p - 1))
        p_val = f_dist.sf(f_stat, p, n - p - 1)

        r2_vals[t_idx] = r2
        p_vals[t_idx] = p_val

    sig_mask = (p_vals < alpha) & (r2_vals > min_r2)
    earliest_sig_time_ms = times_ms[np.where(sig_mask)[0][0]] if np.any(sig_mask) else None

    return times_ms, r2_vals, p_vals, earliest_sig_time_ms



def predict_rt_per_trial_over_time(factors, rt):
    """
    Build a trial-by-time matrix of RT predictions.

    For each time bin and each trial, fit OLS on all OTHER trials
    (leave-one-trial-out) and predict held-out trial RT.
    """
    K, Tr, _ = factors.shape
    y = rt.flatten()[:K]
    y_hat = np.full((K, Tr), np.nan)

    valid_rt = np.isfinite(y)
    for t_idx in range(Tr):
        X_t = factors[:, t_idx, :]
        valid_x = np.all(np.isfinite(X_t), axis=1)
        valid_all = valid_rt & valid_x

        idx_valid = np.where(valid_all)[0]
        if len(idx_valid) < 3:
            continue

        for i in idx_valid:
            train_idx = idx_valid[idx_valid != i]
            if len(train_idx) <= X_t.shape[1] + 1:
                continue

            X_train = X_t[train_idx]
            y_train = y[train_idx]
            X_design_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
            beta, _, _, _ = np.linalg.lstsq(X_design_train, y_train, rcond=None)

            x_test = np.concatenate(([1.0], X_t[i]))
            y_hat[i, t_idx] = x_test @ beta

    return y_hat

def mark_pt(axis, timepoint, p_values, r2_values, p_threshold=0.05, r2_threshold=0.1):
    is_high = (r2_values > r2_threshold) & (p_values < p_threshold)


    #DEBUG
    print(f"Found {np.sum(is_high)} points matching criteria.")
    
    if np.any(is_high):
        axis.scatter(timepoint[is_high], r2_values[is_high], color="red", s=40, edgecolor="white", zorder=5, label="high r2")
    else:
      print("no points that meet such criteria")



def plot_rt_timecourse(times_ms, r2_vals, p_vals, alpha=0.05, out_path="rt_predictability_timecourse.png"):
    """Plot R^2 and p-values across trial time and save a PNG figure."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(times_ms, r2_vals, color="tab:blue", linewidth=2, label="RT R^2")

    mark_pt(ax1, times_ms, r2_vals, p_vals)

    ax1.set_xlabel("Time in trial (ms)")
    ax1.set_ylabel("R^2", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axhline(0, color="tab:blue", linestyle="--", alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(times_ms, p_vals, color="tab:red", linewidth=1.5, label="Model p-value")
    ax2.axhline(alpha, color="tab:red", linestyle="--", alpha=0.7, label=f"alpha={alpha}")
    ax2.set_ylabel("p-value", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(bottom=0)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    # 2. Combine them and create one legend
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
           loc='upper right', frameon=True, fontsize='small')

    fig.suptitle("When does latent state predict reaction time?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)




# 1. File Paths
path_multi = "output/04302025/lfads_output_bilbo_CHKDLAY_DLPFC_20250430_20ms_LFADS (2).h5"
path_beh   = "/Users/berenakpinar/lfads-torch/bilbo_20250430_lfads_trialparams_chk.mat"

# 2. Load Data
print("Loading behavior .mat ...")
t0 = time.time()
behavior = loadmat(path_beh)
print(f"Behavior loaded in {time.time() - t0:.2f}s")

# Extract behavioral targets
rt = behavior["trial_RTs"].flatten()
actions = behavior["trial_action_choices"].flatten()
coherence = behavior["trial_coherences"].flatten()
colors = behavior["trial_color_choices"].flatten()

# Load only as many neural trials as needed to match behavior length.
max_trials_needed = min(len(rt), len(actions), len(coherence), len(colors))
print(f"Loading LFADS factors (up to {max_trials_needed} trials) ...")
t0 = time.time()
factors_multi = load_3d_factors(path_multi, max_trials=max_trials_needed)
print(f"LFADS factors loaded in {time.time() - t0:.2f}s")

# 3. Align Trials (K)
k_min = min(len(factors_multi), len(rt), len(actions), len(coherence), len(colors))
X_all = factors_multi[:k_min]
y_rt = rt[:k_min]

# 4. Time-resolved RT prediction analysis
times_ms, r2_vals, p_vals, earliest_sig_time_ms = run_rt_timecourse_regression(
    factors=X_all,
    rt=y_rt,
    start_time_ms=0,
    bin_ms=20,
    alpha=0.05,
    min_r2=0.0,
)

print(f"{'Time (ms)':<10} | {'RT R2':<10} | {'Model p-value':<14}")
print("-" * 42)
for t_ms, r2_t, p_t in zip(times_ms, r2_vals, p_vals):
    if np.isnan(r2_t) or np.isnan(p_t):
        print(f"{int(t_ms):<10} | {'nan':<10} | {'nan':<14}")
    else:
        print(f"{int(t_ms):<10} | {r2_t:<10.4f} | {p_t:<14.4g}")

if earliest_sig_time_ms is None:
    print("\nNo significant RT-predictive timepoint found (p < 0.05 and R^2 > 0).")
else:
    print(
        f"\nEarliest significant RT-predictive timepoint: {int(earliest_sig_time_ms)} ms "
        "(p < 0.05 and R^2 > 0)."
    )

plot_rt_timecourse(times_ms, r2_vals, p_vals, alpha=0.05, out_path="rt_predictability_timecourse.png")
print("Saved plot: rt_predictability_timecourse.png")

# 5. Per-trial RT predictions across time (trial x time matrix)
print("Computing per-trial RT predictions across time (LOO) ...")
t0 = time.time()
rt_pred_by_trial_time = predict_rt_per_trial_over_time(X_all, y_rt)
print(f"Per-trial prediction matrix computed in {time.time() - t0:.2f}s")

np.save("rt_pred_by_trial_time.npy", rt_pred_by_trial_time)
np.save("rt_true.npy", y_rt)
np.save("times_ms.npy", times_ms)
print("Saved per-trial outputs: rt_pred_by_trial_time.npy, rt_true.npy, times_ms.npy")

# Also save text-friendly versions for quick inspection in the editor.
np.savetxt("rt_pred_by_trial_time.csv", rt_pred_by_trial_time, delimiter=",")
np.savetxt("rt_true.csv", y_rt, delimiter=",")
np.savetxt("times_ms.csv", times_ms, delimiter=",")
print("Saved CSV outputs: rt_pred_by_trial_time.csv, rt_true.csv, times_ms.csv")
