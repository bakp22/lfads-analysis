# lfads-analysis

Behavioral decoding utilities for LFADS factors.

## What `behavior.py` does

`behavior.py` loads LFADS latent factors from an HDF5 output file and trial-level behavior from a MATLAB trial-params file, then asks:

- How predictive are latent factors of **reaction time**?
- How predictive are they of **action choice** and **color choice**?
- How predictive are they of **stimulus coherence**?

It evaluates this at multiple time snapshots (0 ms to 800 ms, step 40 ms), printing one line of decoding performance per timepoint.

## Required inputs

### 1) LFADS output (`.h5`)

Expected keys in the HDF5 file:

- `train_factors`
- `valid_factors`

These arrays are concatenated along trial dimension to make a `(trials, time_bins, factors)` tensor.

### 2) Trial params (`.mat`)

Expected variables:

- `trial_RTs`
- `trial_action_choices`
- `trial_coherences`
- `trial_color_choices`

Each is flattened to a 1D target vector.

## How trial params and LFADS outputs are matched

`behavior.py` aligns by trial index after loading:

- Let `K_factors = number of concatenated LFADS trials`
- Let `K_behavior = number of trials in each behavioral vector`
- Use `k_min = min(K_factors, K_behavior, ...)`
- Truncate all arrays to first `k_min` trials

This protects against shape mismatches, but it assumes trial order is already consistent between LFADS output and the trial-params file.

## Time mapping used in `behavior.py`

The script maps real time to LFADS bins with:

- `start_time_ms = -400`
- `bin_ms = 20`
- `bin_index = int((t_ms - start_time_ms) / bin_ms)`

So:

- `t_ms = 0` corresponds to bin index `20`
- Higher `t_ms` moves forward in time
- Index is clamped to valid range `[0, Tr - 1]`

## How to run

From repo root:

```bash
python behavior.py
```

Before running, set paths in `behavior.py`:

- `path_multi` -> LFADS `.h5` file to evaluate
- `path_beh` -> matching trial-params `.mat` file

If you want portability across machines, prefer relative paths inside the repo instead of absolute user paths.

## Output columns and interpretation

The script prints:

- `Time (ms)`
- `RT R2`
- `Action Acc`
- `Color Acc`
- `Coh R2`

Interpretation:

- `RT R2`: Fraction of reaction-time variance explained by a linear readout from factors at that time. Higher is better.
- `Action Acc`: Classification accuracy for action choice at that time. Higher is better.
- `Color Acc`: Classification accuracy for color choice at that time. Higher is better.
- `Coh R2`: Fraction of coherence variance explained by linear regression at that time. Higher is better.

Practical reading:

- Rising `Action Acc`/`Color Acc` over time suggests increasing choice information in latent state.
- Peaks in `RT R2` suggest moments where latent dynamics best encode reaction-time variability.
- Peaks in `Coh R2` suggest strongest encoding of stimulus strength.

## Important caveats

- Metrics are currently computed on the same trials used for fitting (no held-out split), so values are optimistic.
- If class labels are imbalanced, raw accuracy may be misleading; consider balanced accuracy or class weighting.
- The approach depends on correct trial ordering between `.h5` and `.mat`.

## Suggested next improvements

- Add train/test split or cross-validation per timepoint.
- Save results to CSV for plotting.
- Add baseline metrics (chance accuracy, shuffled-label controls).