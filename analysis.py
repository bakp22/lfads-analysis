import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import f as f_dist

# 1. Load Behavior
behavior = loadmat('/Users/berenakpinar/Desktop/lfads-analysis/bilbo_20250430_lfads_trialparams_chk.mat')
reaction_times = behavior['trial_RTs'].flatten()

# 2. Load LFADS Factors
def load_factors(path):
    with h5py.File(path, 'r') as f:
        train = f['train_factors'][:]
        valid = f['valid_factors'][:]
    return np.concatenate((train, valid), axis=0)

path_single = '/Users/berenakpinar/Desktop/lfads-analysis/output/04302025/lfads_output_bilbo_CHKDLAY_DLPFC_20250430_20ms_LFADS (1).h5'
path_multi = '/Users/berenakpinar/Desktop/lfads-analysis/output/04302025/lfads_output_bilbo_CHKDLAY_DLPFC_20250430_20ms_LFADS (2).h5'

factors_single = load_factors(path_single)
factors_multi = load_factors(path_multi)

# 3. Exhaustive Window Search Function
def _r2_ols_with_intercept(X, y):
    """Same R² as sklearn LinearRegression(fit_intercept=True).score on (X, y)."""
    X = np.asarray(X, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64).ravel()
    X_ = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    y_hat = X_ @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _r2_global_model_pvalue(r2, n, k):
    """
    F-test p-value for H0: all k predictor slopes are zero (standard 'global' R² test).
    k = number of features (not counting intercept). Requires n > k + 1.
    """
    r2 = np.asarray(r2, dtype=np.float64)
    df2 = n - k - 1
    out = np.full(np.shape(r2), np.nan, dtype=np.float64)
    if df2 <= 0:
        return out
    r2c = np.clip(r2, 0.0, 1.0)
    interior = np.isfinite(r2c) & (r2c > 0.0) & (r2c < 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (r2c / k) / ((1.0 - r2c) / df2)
    out[interior] = f_dist.sf(F[interior], k, df2)
    out[np.isfinite(r2c) & (r2c >= 1.0)] = 0.0
    out[np.isfinite(r2c) & (r2c <= 0.0)] = 1.0
    return out


def per_start_best_window(r2_matrix, p_matrix, min_bins, n_time):
    """For each valid start bin, pick the end that maximizes R²; return 1D series."""
    starts, r2s, ps, end_bins = [], [], [], []
    for s in range(n_time):
        j0 = s + min_bins - 1
        if j0 >= n_time:
            break
        row = r2_matrix[s, j0:n_time]
        if not np.any(np.isfinite(row)):
            continue
        j = j0 + int(np.nanargmax(row))
        starts.append(s)
        r2s.append(float(r2_matrix[s, j]))
        ps.append(float(p_matrix[s, j]))
        end_bins.append(j)
    return (
        np.asarray(starts, dtype=int),
        np.asarray(r2s, dtype=np.float64),
        np.asarray(ps, dtype=np.float64),
        np.asarray(end_bins, dtype=int),
    )


def exhaustive_window_search(factors, target, min_bins=3, label=""):
    """
    Tests every possible slice (start to end) and returns R^2 scores.
    """
    n_trials, n_time, n_factors = factors.shape
    r2_matrix = np.full((n_time, n_time), np.nan)

    best_r2 = -np.inf
    best_coords = (0, 0)

    # Prefix sums so window means are O(1) per (start, end)
    cs = np.zeros((n_trials, n_time + 1, n_factors), dtype=np.float64)
    cs[:, 1:] = np.cumsum(factors.astype(np.float64, copy=False), axis=1)
    y = np.asarray(target, dtype=np.float64).ravel()

    prefix = f"{label}: " if label else ""
    L = n_time - min_bins
    n_windows = (L + 1) * (L + 2) // 2
    print(f"{prefix}Scanning {n_time} time bins ({n_windows} windows)...", flush=True)
    t0 = time.monotonic()

    for start in range(n_time):
        if start > 0 and start % 10 == 0:
            elapsed = time.monotonic() - t0
            rate = start / elapsed
            eta = (n_time - start) / rate if rate > 0 else float("nan")
            print(f"{prefix}  start bin {start}/{n_time} (~{eta:.0f}s left)", flush=True)
        for end in range(start + min_bins, n_time + 1):
            w = end - start
            x_data = (cs[:, end] - cs[:, start]) / w
            r2 = _r2_ols_with_intercept(x_data, y)
            r2_matrix[start, end - 1] = r2
            if r2 > best_r2:
                best_r2 = r2
                best_coords = (start, end)

    print(f"{prefix}Done ({time.monotonic() - t0:.1f}s).", flush=True)
    p_matrix = _r2_global_model_pvalue(r2_matrix, n_trials, n_factors)
    return r2_matrix, p_matrix, best_r2, best_coords


MIN_BINS = 3

# 4. Run Analysis
# Note: min_bins=5 assumes at least 100ms of data for a stable estimate
res_single, p_single, r2_s, slice_s = exhaustive_window_search(
    factors_single, reaction_times, min_bins=MIN_BINS, label="single"
)
res_multi, p_multi, r2_m, slice_m = exhaustive_window_search(
    factors_multi, reaction_times, min_bins=MIN_BINS, label="multi"
)

n_trials_s, n_time_s, n_fac_s = factors_single.shape
n_trials_m, n_time_m, n_fac_m = factors_multi.shape
p_best_s = float(_r2_global_model_pvalue(r2_s, n_trials_s, n_fac_s).reshape(-1)[0])
p_best_m = float(_r2_global_model_pvalue(r2_m, n_trials_m, n_fac_m).reshape(-1)[0])

st_s, r2_curve_s, p_curve_s, end_s = per_start_best_window(
    res_single, p_single, MIN_BINS, n_time_s
)
st_m, r2_curve_m, p_curve_m, end_m = per_start_best_window(
    res_multi, p_multi, MIN_BINS, n_time_m
)

# 5. Output Results
print("-" * 30)
print(f"SINGLE FILE BEST:")
print(f"  Window: Bins {slice_s[0]} to {slice_s[1]} (end exclusive)")
print(f"  R-squared: {r2_s:.4f}")
print(f"  Global model p-value (F-test): {p_best_s:.4e}")
print("-" * 30)
print(f"MULTI FILE BEST:")
print(f"  Window: Bins {slice_m[0]} to {slice_m[1]} (end exclusive)")
print(f"  R-squared: {r2_m:.4f}")
print(f"  Global model p-value (F-test): {p_best_m:.4e}")

# 6. Visualization: per start bin, best-window R² and p-value
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)

def _plot_r2_p(ax, start_bins, r2s, ps, title):
    c_r2 = "C0"
    c_p = "C1"
    ax.plot(start_bins, r2s, color=c_r2, lw=1.5, label="R²")
    ax.set_xlabel("Time bin")
    ax.set_ylabel("R²", color=c_r2)
    ax.tick_params(axis="y", labelcolor=c_r2)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(start_bins, ps, color=c_p, lw=1.5, alpha=0.9, label="p-value")
    ax2.set_ylabel("p-value (global F-test)", color=c_p)
    ax2.tick_params(axis="y", labelcolor=c_p)
    ax2.set_yscale("log")
    ax2.axhline(0.05, color="gray", ls="--", lw=1, alpha=0.8)
    ax.set_title(title)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)


_plot_r2_p(
    axes[0],
    st_s,
    r2_curve_s,
    p_curve_s,
    f"Single file — best end per start\nGlobal best: R²={r2_s:.3f}, p={p_best_s:.2e}",
)
_plot_r2_p(
    axes[1],
    st_m,
    r2_curve_m,
    p_curve_m,
    f"Multi file — best end per start\nGlobal best: R²={r2_m:.3f}, p={p_best_m:.2e}",
)

plt.tight_layout()
plt.show()