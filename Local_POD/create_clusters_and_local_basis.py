import numpy as np
import os
from sklearn.cluster import KMeans
import joblib

def get_number_of_singular_values_for_given_tolerance(M, N, s, epsilon):
    """
    Compute the number of singular values (modes) to retain based on the given tolerance.
    """
    dimMATRIX = max(M, N)
    tol = dimMATRIX * np.finfo(float).eps * max(s) / 2
    R = np.sum(s > tol)  # Definition of numerical rank
    if epsilon == 0:
        K = R
    else:
        SingVsq = np.multiply(s, s)
        SingVsq.sort()
        normEf2 = np.sqrt(np.cumsum(SingVsq))
        epsilon = epsilon * normEf2[-1]  # relative tolerance
        T = sum(normEf2 < epsilon)
        K = len(s) - T
    K = min(R, K)
    return K

# Load snapshot data
data_path = '../FEM/training_data/'
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)

# Perform global SVD to obtain a reduced representation
U_global, S_global, VT_global = np.linalg.svd(all_snapshots, full_matrices=False)

# Choose a tolerance level (e.g., 1e-4) and compute the number of global modes to retain
tolerance = 1e-4
M_global, N_global = U_global.shape
num_global_modes = get_number_of_singular_values_for_given_tolerance(M_global, N_global, S_global, tolerance)
q_global = (U_global[:, :num_global_modes]).T @ all_snapshots

# Test different numbers of clusters
n_clusters_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # Example values; can be adjusted
best_n_clusters = None
best_error = float('inf'clusters/)

for n_clusters in n_clusters_list:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    snapshot_labels = kmeans.fit_predict(q_global.T)
    
    # Save cluster labels and centers
    np.save(f'clusters/cluster_labels_{n_clusters}.npy', snapshot_labels)
    np.save(f'clusters/cluster_centers_{n_clusters}.npy', kmeans.cluster_centers_)
    
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
    
    # Convert lists to numpy arrays and perform SVD for each cluster
    local_bases = {}
    total_error = 0
    
    for key in clustered_snapshots:
        clustered_snapshots[key] = np.array(clustered_snapshots[key]).T
        M_local, N_local = clustered_snapshots[key].shape
        U_local, S_local, VT_local = np.linalg.svd(clustered_snapshots[key], full_matrices=False)
        
        # Compute the number of modes to retain based on the same tolerance
        num_local_modes = get_number_of_singular_values_for_given_tolerance(M_local, N_local, S_local, tolerance)
        local_bases[key] = U_local[:, :num_local_modes]
        
        # Save the size of each cluster's basis
        np.save(f'clusters/local_bases_cluster_{key}_nclusters_{n_clusters}.npy', local_bases[key])
        
        # Compute the reconstruction error for this cluster
        reconstructed = U_local[:, :num_local_modes] @ np.diag(S_local[:num_local_modes]) @ VT_local[:num_local_modes, :]
        error = np.linalg.norm(clustered_snapshots[key] - reconstructed) / np.linalg.norm(clustered_snapshots[key])
        total_error += error

        # Save the reconstruction error for each cluster
        # np.save(f'clusters/reconstruction_error_cluster_{key}_nclusters_{n_clusters}.npy', error)
    
    # Average error over all clusters
    average_error = total_error / n_clusters
    print(f"Number of clusters: {n_clusters}, Average reconstruction error: {average_error}")
    
    # Track the best number of clusters based on the lowest error
    if average_error < best_error:
        best_error = average_error
        best_n_clusters = n_clusters
        best_local_bases = local_bases
        best_kmeans = kmeans

# Save the best KMeans model, local bases, and global U matrix
joblib.dump(best_kmeans, 'best_kmeans_model.pkl')
np.save('best_local_bases.npy', best_local_bases)
np.save('U_global.npy', U_global[:, :num_global_modes])

print(f"Best number of clusters: {best_n_clusters}, with average error: {best_error}")
print("Clustering and local basis computation with overlap completed.")




