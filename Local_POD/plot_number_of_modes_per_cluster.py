import numpy as np
import os
import matplotlib.pyplot as plt

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def count_modes_in_clusters(cluster_bases_dir, n_clusters_list):
    average_modes_list = []  # List to store average number of modes for each clustering
    all_cluster_modes = []  # List to store the number of modes for each cluster

    for n_clusters in n_clusters_list:
        cluster_modes = []  # Store modes for this specific clustering
        print(f"\nClustering with {n_clusters} clusters:")
        total_modes = 0
        num_clusters = 0
        for cluster_index in range(n_clusters):
            # Load the local basis for this cluster
            basis_file = os.path.join(cluster_bases_dir, f'clusters/local_bases_cluster_{cluster_index}_nclusters_{n_clusters}.npy')
            if os.path.exists(basis_file):
                local_basis = np.load(basis_file)
                num_modes = local_basis.shape[1]  # Number of columns corresponds to the number of modes
                print(f"Cluster {cluster_index}: {num_modes} modes")
                total_modes += num_modes
                num_clusters += 1
                cluster_modes.append(num_modes)
            else:
                print(f"Basis file for Cluster {cluster_index} in {n_clusters} clusters not found.")

        all_cluster_modes.append(cluster_modes)

        if num_clusters > 0:
            average_modes = total_modes / num_clusters
            print(f"Average number of modes for {n_clusters} clusters: {average_modes:.2f}")
            average_modes_list.append(average_modes)
        else:
            print(f"No clusters found for {n_clusters} clusters.")
            average_modes_list.append(np.nan)  # Use NaN to represent missing data

    return average_modes_list, all_cluster_modes

# Define the directory where the local bases are saved
cluster_bases_dir = '.'  # Modify this path if needed

# Define the list of cluster configurations you tested
n_clusters_list = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]

# Count and get the average number of modes for each clustering
average_modes_list, all_cluster_modes = count_modes_in_clusters(cluster_bases_dir, n_clusters_list)

# Plotting the average number of modes for each clustering
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_list, average_modes_list, marker='o', linestyle='-', color='b', label='Average Modes')

# Plotting the range of modes per cluster as vertical bars
for i, cluster_modes in enumerate(all_cluster_modes):
    if len(cluster_modes) > 0:
        plt.vlines(x=n_clusters_list[i], ymin=min(cluster_modes), ymax=max(cluster_modes), color='r', alpha=0.5, label='Mode Range' if i == 0 else "")  # Add label only once

# Define y-axis ticks with more granularity from 0 to 50
yticks = list(range(0, 60, 10)) + list(range(100, 350, 50))
plt.yticks(yticks)  # Set y-axis ticks to include 0,10,20,30,40,50,100,150,200,250,300

plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)  # Grid lines for y-axis

plt.xlabel(r'Number of Clusters')
plt.ylabel(r'Number of Modes')
plt.title(r'Average Number of Modes for Different Clustering Configurations with Mode Range')
plt.legend()
plt.savefig("average_modes_per_clustering_with_range.pdf", format='pdf')
plt.show()






