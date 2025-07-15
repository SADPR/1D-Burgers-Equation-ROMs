import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
from matplotlib.animation import FuncAnimation, PillowWriter

# Load the KMeans model, local bases, and global U matrix
kmeans = joblib.load('kmeans_model.pkl')
local_bases = np.load('local_bases_overlap.npy', allow_pickle=True).item()
U_global = np.load('U_global.npy')

# Function to reconstruct snapshot using local basis
def reconstruct_snapshot_with_local_basis(snapshot, U):
    return U @ (U.T @ snapshot)

# Load the specific snapshot
snapshot_file = '../fem_training_data/fem_simulation_mu1_4.250_mu2_0.0150.npy'
snapshot = np.load(snapshot_file)

# Project snapshot onto the global basis to determine the cluster
q_global_snapshot = (U_global.T @ snapshot).T
snapshot_clusters = kmeans.predict(q_global_snapshot)

# Reconstruct each column of the snapshot using the appropriate local basis
reconstructed_snapshot = np.zeros_like(snapshot)

for i in range(snapshot.shape[1]):
    single_column = snapshot[:, i]

    # Get the local basis for the identified cluster
    cluster_id = snapshot_clusters[i]
    U = local_bases[cluster_id]

    # Reconstruct the column using the local basis
    reconstructed_column = reconstruct_snapshot_with_local_basis(single_column, U)
    
    reconstructed_snapshot[:, i] = reconstructed_column

np.save("local_pod_reconstruction.npy", reconstructed_snapshot)

# Plot the original and reconstructed snapshots
X = np.linspace(0, 100, snapshot.shape[0])

# Create a combined GIF with original and reconstructed snapshots
fig, ax = plt.subplots(figsize=(10, 6))
line_original, = ax.plot(X, snapshot[:, 0], 'b-', label='Original Snapshot')
line_reconstructed, = ax.plot(X, reconstructed_snapshot[:, 0], 'r--', label='Reconstructed Snapshot (Local Basis)')
ax.set_xlim(X[0], X[-1])
ax.set_ylim(np.min(snapshot), np.max(snapshot))
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.legend()

def update(frame):
    line_original.set_ydata(snapshot[:, frame])
    line_reconstructed.set_ydata(reconstructed_snapshot[:, frame])
    ax.set_title(f'Snapshot Comparison at t = {frame * 0.07:.2f}\nCluster ID: {snapshot_clusters[frame]}')
    return line_original, line_reconstructed

ani = FuncAnimation(fig, update, frames=snapshot.shape[1], blit=True)

# Save animation as GIF
ani.save("reconstructed_local_pod.gif", writer=PillowWriter(fps=10))

plt.show()





