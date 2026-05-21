import h5py
import numpy as np

# Load times_ms
times_ms = np.load('times_ms.npy')
print(f"times_ms: {len(times_ms)} bins, {times_ms[0]:.0f} to {times_ms[-1]:.0f} ms, {times_ms[1]-times_ms[0]:.0f}ms per bin")

# Load LFADS data
with h5py.File("output/04302025/lfads_output_bilbo_CHKDLAY_DLPFC_20250430_20ms_LFADS (2).h5", "r") as f:
    train_factors = f['train_factors'][:]
    valid_factors = f['valid_factors'][:]
    print(f"train_factors: {train_factors.shape} = ({train_factors.shape[0]} trials, {train_factors.shape[1]} timepoints, {train_factors.shape[2]} factors)")
    print(f"valid_factors: {valid_factors.shape} = ({valid_factors.shape[0]} trials, {valid_factors.shape[1]} timepoints, {valid_factors.shape[2]} factors)")
    print(f"\nT in LFADS factors: {train_factors.shape[1]}")
    print(f"Expected T from times_ms: {len(times_ms)}")
    print(f"MATCH: {train_factors.shape[1] == len(times_ms)}")
