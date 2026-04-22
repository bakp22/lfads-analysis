import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_3d_pca_trajectory(file_path):
    with h5py.File(file_path, "r") as f:
        # Load factors: (Trials, Time, Factors)
        train_f = f['train_factors'][:]
        valid_f = f['valid_factors'][:]
        factors = np.concatenate((train_f, valid_f), axis=0)
        
    K, Tr, L = factors.shape
    
    # 1. Reshape to "Long-Form" (Total Timepoints x Factors)
    # PCA needs a 2D matrix where every row is a moment in time
    factors_2d = factors.reshape(K * Tr, L)
    
    # 2. Run PCA to get 3 components
    pca = PCA(n_components=3)
    factors_pca = pca.fit_transform(factors_2d)
    
    # 3. Reshape back to (Trials, Time, 3) to keep trial structure
    trajectories = factors_pca.reshape(K, Tr, 3)
    
    return trajectories, pca.explained_variance_ratio_

# --- Execute and Plot ---
path_multi = "output/04302025/lfads_output_bilbo_CHKDLAY_DLPFC_20250430_20ms_LFADS (2).h5"
trajectories, variance = get_3d_pca_trajectory(path_multi)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot a subset of trials (e.g., first 50) to avoid a "spaghetti" mess
for i in range(50):
    # trajectories[trial_index, :, dimension]
    ax.plot(trajectories[i, :, 0], trajectories[i, :, 1], trajectories[i, :, 2], 
            alpha=0.4, linewidth=1)

# Highlight the start and end points
ax.scatter(trajectories[:50, 0, 0], trajectories[:50, 0, 1], trajectories[:50, 0, 2], 
           color='green', s=10, label='Trial Start')
ax.scatter(trajectories[:50, -1, 0], trajectories[:50, -1, 1], trajectories[:50, -1, 2], 
           color='red', s=10, label='Trial End')

ax.set_title(f"3D Neural Trajectory (PCA)\nVariance Explained: {np.sum(variance)*100:.2f}%")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
plt.legend()
plt.show()


#PCAs:

'''
    PC1 - This is the direction in neural space that shows the most variation. 
          In many neuroscience tasks, PC 1 often represents Time or Task Phase 
          (e.g., stimulus onset vs. movement). TIME
    PC2 - Stimulus or Rule

    PC3 - Action/Choice 
'''