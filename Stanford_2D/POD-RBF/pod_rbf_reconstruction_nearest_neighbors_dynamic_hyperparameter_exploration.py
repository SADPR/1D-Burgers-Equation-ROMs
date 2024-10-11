import numpy as np
import os
import pickle
from scipy.spatial import KDTree
import time
import csv

# Define various RBF kernels
def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)

def multiquadric_rbf(r, epsilon):
    return np.sqrt(1 + (epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

def linear_rbf(r, epsilon):
    return r

def cubic_rbf(r, epsilon):
    return r ** 3

def thin_plate_spline_rbf(r, epsilon):
    return r ** 2 * np.log(r + np.finfo(float).eps)

def power_rbf(r, epsilon, p=2):
    return r ** p

def exponential_rbf(r, epsilon):
    return np.exp(-epsilon * r)

def polyharmonic_spline_rbf(r, epsilon):
    return r ** 2 * np.log(r + np.finfo(float).eps)

def rbf_kernel(r, epsilon, kernel_type):
    """Selects the RBF kernel function based on the specified kernel type."""
    if kernel_type == "gaussian":
        return gaussian_rbf(r, epsilon)
    elif kernel_type == "multiquadric":
        return multiquadric_rbf(r, epsilon)
    elif kernel_type == "inverse_multiquadric":
        return inverse_multiquadric_rbf(r, epsilon)
    elif kernel_type == "linear":
        return linear_rbf(r, epsilon)
    elif kernel_type == "cubic":
        return cubic_rbf(r, epsilon)
    elif kernel_type == "thin_plate_spline":
        return thin_plate_spline_rbf(r, epsilon)
    elif kernel_type == "power":
        return power_rbf(r, epsilon, p=2)
    elif kernel_type == "exponential":
        return exponential_rbf(r, epsilon)
    elif kernel_type == "polyharmonic_spline":
        return polyharmonic_spline_rbf(r, epsilon)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def interpolate_on_the_fly(kdtree, q_p_train, q_s_train, x_new, epsilon, neighbors, kernel_type):
    """Interpolate at new points using nearest neighbors."""
    dist, idx = kdtree.query(x_new, k=neighbors)
    X_neighbors = q_p_train[idx].reshape(neighbors, -1)
    Y_neighbors = q_s_train[idx, :].reshape(neighbors, -1)
    dists_neighbors = np.linalg.norm(X_neighbors[:, np.newaxis] - X_neighbors[np.newaxis, :], axis=-1)
    Phi_neighbors = rbf_kernel(dists_neighbors, epsilon, kernel_type)
    Phi_neighbors += np.eye(neighbors) * 1e-8  # Regularization for stability
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)
    rbf_values = rbf_kernel(dist, epsilon, kernel_type)
    f_new = rbf_values @ W_neighbors
    return f_new

def reconstruct_snapshot_with_pod_rbf_neighbors(snapshot, U_p, U_s, q_p_train, q_s_train, kdtree, r, epsilon, neighbors, kernel_type):
    """Reconstruct the snapshot using POD-RBF with neighbors."""
    q = U_p.T @ snapshot
    q_p = q[:r, :]
    reconstructed_snapshots_rbf = []
    for i in range(q_p.shape[1]):
        q_p_sample = np.array(q_p[:, i].reshape(1, -1))
        q_s_pred = interpolate_on_the_fly(kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors, kernel_type).T
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred.reshape(-1)
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)
    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).squeeze().T
    return reconstructed_snapshots_rbf

def explore_combinations(epsilon_values, neighbor_values, kernel_types):
    # Load the saved KDTree and training data (q_p and q_s)
    with open('modes/training_data.pkl', 'rb') as f:
        data = pickle.load(f)
        kdtree = data['KDTree']
        q_p_train = data['q_p']
        q_s_train = data['q_s']

    # Load a random snapshot from the training_data directory
    snapshot_file = '../POD/FOM_vs_ROM_Results/U_FOM_mu1_4.560_mu2_0.0190_mesh_250x250.npy'
    snapshot = np.load(snapshot_file).T

    # Load U_p, U_s, and the full U matrix
    U_p = np.load('modes/U_p.npy')
    U_s = np.load('modes/U_s.npy')

    r = U_p.shape[1]  # Number of primary modes

    results = []

    # Iterate over all combinations
    for epsilon in epsilon_values:
        for neighbors in neighbor_values:
            for kernel_type in kernel_types:
                print(f"Testing epsilon={epsilon}, neighbors={neighbors}, kernel_type={kernel_type}")
                try:
                    start_time = time.time()
                    # Reconstruct the snapshot using POD-RBF with current parameters
                    pod_rbf_reconstructed = reconstruct_snapshot_with_pod_rbf_neighbors(
                        snapshot, U_p, U_s, q_p_train, q_s_train, kdtree, r, epsilon, neighbors, kernel_type
                    )

                    # Calculate reconstruction error
                    pod_rbf_error = np.linalg.norm(snapshot - pod_rbf_reconstructed) / np.linalg.norm(snapshot)
                    elapsed_time = time.time() - start_time
                    print(f"Reconstruction error: {pod_rbf_error:.6e}, Time: {elapsed_time:.2f} seconds")

                    # Save the result
                    results.append({
                        'epsilon': epsilon,
                        'neighbors': neighbors,
                        'kernel_type': kernel_type,
                        'reconstruction_error': pod_rbf_error,
                        'time': elapsed_time
                    })
                except Exception as e:
                    print(f"Failed for epsilon={epsilon}, neighbors={neighbors}, kernel_type={kernel_type}: {e}")

    # Save results to CSV
    results_file = 'FOM_vs_POD-RBF_Exploration_Results.csv'
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['epsilon', 'neighbors', 'kernel_type', 'reconstruction_error', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Exploration results saved to {results_file}")

if __name__ == '__main__':
    # RBF parameter ranges
    epsilon_values = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    neighbor_values = [5, 10, 20, 50, 100]

    # RBF kernels to test
    kernel_types = [
        "gaussian", "multiquadric", "inverse_multiquadric", "linear",
        "cubic", "thin_plate_spline", "power", "exponential", "polyharmonic_spline"
    ]

    # Run the exploration
    explore_combinations(epsilon_values, neighbor_values, kernel_types)
