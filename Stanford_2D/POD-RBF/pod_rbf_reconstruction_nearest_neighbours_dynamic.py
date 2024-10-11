import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.spatial import KDTree
import time

# Function to compute the Gaussian RBF kernel
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

# Function to dynamically interpolate at new points using nearest neighbors
def interpolate_on_the_fly(kdtree, q_p_train, q_s_train, x_new, epsilon, neighbors, print_times=False):
    """Interpolate at new points using nearest neighbors and solving the system on the fly."""
    start_time = time.time()
    dist, idx = kdtree.query(x_new, k=neighbors)
    kdtree_time = time.time()
    if print_times:
        print(f"KDTree query took: {kdtree_time - start_time:.6f} seconds")

    X_neighbors = q_p_train[idx].reshape(neighbors, -1)
    Y_neighbors = q_s_train[idx, :].reshape(neighbors, -1)

    extract_time = time.time()
    if print_times:
        print(f"Extracting neighbors took: {extract_time - kdtree_time:.6f} seconds")

    dists_neighbors = np.linalg.norm(X_neighbors[:, np.newaxis] - X_neighbors[np.newaxis, :], axis=-1)
    Phi_neighbors = gaussian_rbf(dists_neighbors, epsilon)
    rbf_matrix_time = time.time()
    if print_times:
        print(f"RBF matrix computation took: {rbf_matrix_time - extract_time:.6f} seconds")

    Phi_neighbors += np.eye(neighbors) * 1e-8
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)
    solve_time = time.time()
    if print_times:
        print(f"Solving the linear system took: {solve_time - rbf_matrix_time:.6f} seconds")

    rbf_values = gaussian_rbf(dist, epsilon)
    rbf_eval_time = time.time()
    if print_times:
        print(f"RBF evaluation for new point took: {rbf_eval_time - solve_time:.6f} seconds")

    f_new = rbf_values @ W_neighbors
    total_time = time.time() - start_time
    if print_times:
        print(f"Total interpolation process took: {total_time:.6f} seconds")

    return f_new

# Function to reconstruct a snapshot using the POD-RBF model with nearest neighbors and dynamic interpolation
def reconstruct_snapshot_with_pod_rbf_neighbors(snapshot, U_p, U_s, q_p_train, q_s_train, kdtree, r, epsilon, neighbors, print_times=False):
    start_total_time = time.time()
    q = U_p.T @ snapshot
    q_p = q[:r, :]

    reconstructed_snapshots_rbf = []
    for i in range(q_p.shape[1]):
        if print_times:
            print(f"Time step {i+1} of {q_p.shape[1]}")
        q_p_sample = np.array(q_p[:, i].reshape(1, -1))

        q_s_pred = interpolate_on_the_fly(kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors, print_times).T
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred.reshape(-1)
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)

    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).squeeze().T
    print(f"Total reconstruction process took: {time.time() - start_total_time:.6f} seconds")

    return reconstructed_snapshots_rbf

# Function to reconstruct using normal POD
def reconstruct_snapshot_with_pod(snapshot, U, num_modes, print_times=False):
    """Reconstruct a snapshot using standard POD."""
    start_time = time.time()
    U_modes = U[:, :num_modes]
    q_pod = U_modes.T @ snapshot
    reconstructed_pod = U_modes @ q_pod
    if print_times:
        print(f"POD reconstruction took: {time.time() - start_time:.6f} seconds")
    return reconstructed_pod

if __name__ == '__main__':
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
    U_full = np.hstack((U_p, U_s))  # Full U matrix with 150 modes

    epsilon = 0.0001
    neighbors = 50
    num_modes = 150

    # Boolean to control whether to perform POD comparison
    compare_pod = False
    # Boolean to control whether to print times for each step
    print_times = False

    # Reconstruct the snapshot using dynamic RBF interpolation with nearest neighbors
    pod_rbf_reconstructed = reconstruct_snapshot_with_pod_rbf_neighbors(
        snapshot, U_p, U_s, q_p_train, q_s_train, kdtree, 10, epsilon, neighbors, print_times
    )

    # Reconstruct the snapshot using standard POD with 150 modes if compare_pod is True
    if compare_pod:
        pod_reconstructed = reconstruct_snapshot_with_pod(snapshot, U_full, num_modes, print_times)

    # Save the reconstructed data
    results_dir = "FOM_vs_POD-RBF_Reconstruction_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    pod_rbf_file_path = os.path.join(results_dir, "reconstructed_snapshot_pod_rbf.npy")
    np.save(pod_rbf_file_path, pod_rbf_reconstructed)
    print(f"POD-RBF reconstructed snapshot saved successfully to {pod_rbf_file_path}")

    if compare_pod:
        pod_file_path = os.path.join(results_dir, "reconstructed_snapshot_pod.npy")
        np.save(pod_file_path, pod_reconstructed)
        print(f"POD reconstructed snapshot saved successfully to {pod_file_path}")

    # Calculate and compare reconstruction errors
    pod_rbf_error = np.linalg.norm(snapshot - pod_rbf_reconstructed) / np.linalg.norm(snapshot)
    print(f"POD-RBF Reconstruction error: {pod_rbf_error:.6e}")

    if compare_pod:
        pod_error = np.linalg.norm(snapshot - pod_reconstructed) / np.linalg.norm(snapshot)
        print(f"POD Reconstruction error: {pod_error:.6e}")
