import numpy as np
import os

def count_modes_in_clusters(cluster_bases_dir, n_clusters_list):
    for n_clusters in n_clusters_list:
        print(f"\nClustering with {n_clusters} clusters:")
        total_modes = 0
        num_clusters = 0
        for cluster_index in range(n_clusters):
            # Load the local basis for this cluster
            basis_file = os.path.join(cluster_bases_dir, f'local_bases_cluster_{cluster_index}_nclusters_{n_clusters}.npy')
            if os.path.exists(basis_file):
                local_basis = np.load(basis_file)
                num_modes = local_basis.shape[1]  # Number of columns corresponds to the number of modes
                print(f"Cluster {cluster_index}: {num_modes} modes")
                total_modes += num_modes
                num_clusters += 1
            else:
                print(f"Basis file for Cluster {cluster_index} in {n_clusters} clusters not found.")

        if num_clusters > 0:
            average_modes = total_modes / num_clusters
            print(f"Average number of modes for {n_clusters} clusters: {average_modes:.2f}")
        else:
            print(f"No clusters found for {n_clusters} clusters.")

# Define the directory where the local bases are saved
cluster_bases_dir = '.'  # Modify this path if needed

# Define the list of cluster configurations you tested
n_clusters_list = [10, 15, 18, 20, 25]

# Count and print the number of modes for each cluster in each clustering
count_modes_in_clusters(cluster_bases_dir, n_clusters_list)

