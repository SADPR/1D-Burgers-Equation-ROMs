import numpy as np
import os
from sklearn.cluster import KMeans
import joblib

# Load snapshot data
data_path = '../training_data/'
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)

# Perform global SVD to obtain a reduced representation
U_global, S_global, VT_global = np.linalg.svd(all_snapshots, full_matrices=False)

# Choose number of global modes to retain (adjust based on your data)
num_global_modes = 301
q_global = (U_global[:, :num_global_modes]).T @ all_snapshots

# Perform clustering on the reduced representation
n_clusters = 18  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
snapshot_labels = kmeans.fit_predict(q_global.T)

# Create a dictionary to store snapshots for each cluster
clustered_snapshots = {i: [] for i in range(n_clusters)}
for i, label in enumerate(snapshot_labels):
    clustered_snapshots[label].append(all_snapshots[:, i])

# Add overlapping snapshots to adjacent clusters
overlap_threshold = 0.1  # Define a threshold for overlap
for i, label in enumerate(snapshot_labels):
    distances = np.linalg.norm(kmeans.cluster_centers_ - q_global[:, i].reshape(1, -1), axis=1)
    close_clusters = np.where(distances < overlap_threshold)[0]
    for cluster in close_clusters:
        clustered_snapshots[cluster].append(all_snapshots[:, i])

# Convert lists to numpy arrays
for key in clustered_snapshots:
    clustered_snapshots[key] = np.array(clustered_snapshots[key]).T

# Perform SVD for each cluster to obtain the local basis
local_bases = {}
for key in clustered_snapshots:
    U, S, VT = np.linalg.svd(clustered_snapshots[key], full_matrices=False)
    local_bases[key] = U[:, :16]  # Keep only the first 16 modes

# Save the KMeans model, local bases, and global U matrix
joblib.dump(kmeans, 'kmeans_model.pkl')
np.save('local_bases_overlap.npy', local_bases)
np.save('U_global.npy', U_global[:, :num_global_modes])

print("Clustering and local basis computation with overlap completed.")



