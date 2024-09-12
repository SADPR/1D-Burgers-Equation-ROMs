import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import os
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
import time

# Function to compute the Gaussian RBF kernel
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

# Function to dynamically interpolate at new points using nearest neighbors
def interpolate_on_the_fly(kdtree, q_p_train, q_s_train, x_new, epsilon, neighbors):
    """Interpolate at new points using nearest neighbors and solving the system on the fly."""

    start_time = time.time()

    # Find the nearest neighbors for the new point
    dist, idx = kdtree.query(x_new, k=neighbors)
    kdtree_time = time.time()
    print(f"KDTree query took: {kdtree_time - start_time:.6f} seconds")

    # Extract the neighbor points and ensure they have the correct shape (neighbors, 28)
    X_neighbors = q_p_train[idx].reshape(neighbors, -1)  # Reshape to (neighbors, 28)
    Y_neighbors = q_s_train[idx, :].reshape(neighbors, -1)  # Y_neighbors corresponding to q_s_train

    extract_time = time.time()
    print(f"Extracting neighbors took: {extract_time - kdtree_time:.6f} seconds")

    # Compute pairwise distances between neighbors using pdist
    dists_neighbors = squareform(pdist(X_neighbors))
    dist_calc_time = time.time()
    print(f"Distance calculations took: {dist_calc_time - extract_time:.6f} seconds")

    # Compute the RBF matrix for the neighbors (Phi matrix)
    Phi_neighbors = gaussian_rbf(dists_neighbors, epsilon)
    rbf_matrix_time = time.time()
    print(f"RBF matrix computation took: {rbf_matrix_time - dist_calc_time:.6f} seconds")

    # Regularize the Phi_neighbors matrix by adding a small value to its diagonal
    Phi_neighbors += np.eye(neighbors) * 1e-8  # Regularization for numerical stability

    # Solve the system Phi_neighbors * W_neighbors = Y_neighbors to find the interpolation weights
    solve_start_time = time.time()
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)
    solve_time = time.time()
    print(f"Solving the linear system took: {solve_time - solve_start_time:.6f} seconds")

    # Compute RBF values between the new point and its neighbors
    rbf_values = gaussian_rbf(dist, epsilon)  # This is between x_new and its neighbors
    rbf_eval_time = time.time()
    print(f"RBF evaluation for new point took: {rbf_eval_time - solve_time:.6f} seconds")

    # Interpolate the new value by multiplying the RBF values with the computed weights
    f_new = rbf_values @ W_neighbors  # RBF interpolation step
    interpolation_time = time.time()
    print(f"Final interpolation step took: {interpolation_time - rbf_eval_time:.6f} seconds")

    total_time = time.time() - start_time
    print(f"Total interpolation process took: {total_time:.6f} seconds")

    return f_new

# Function to reconstruct a snapshot using the POD-RBF model with nearest neighbors and dynamic interpolation
def reconstruct_snapshot_with_pod_rbf_neighbors(snapshot_file, U_p, U_s, q_p_train, q_s_train, kdtree, r, epsilon, neighbors):
    start_total_time = time.time()

    # Load the snapshot file
    snapshots = np.load(snapshot_file)
    load_snapshot_time = time.time()
    print(f"Loading snapshot took: {load_snapshot_time - start_total_time:.6f} seconds")

    # Project onto the POD basis to get q_p
    q = U_p.T @ snapshots
    q_p = q[:r, :]
    project_pod_time = time.time()
    print(f"Projection onto POD basis took: {project_pod_time - load_snapshot_time:.6f} seconds")

    # Ensure q_p_train and q_s_train are NumPy arrays before entering the loop
    q_p_train_array = np.array(q_p_train)
    q_s_train_array = np.array(q_s_train)

    array_conversion_time = time.time()
    print(f"Initial np.array conversion took: {array_conversion_time - project_pod_time:.6f} seconds")

    # Reconstruct the snapshots using dynamic RBF interpolation with neighbors
    reconstructed_snapshots_rbf = []
    for i in range(q_p.shape[1]):
        print(f"Time step {i+1} of {q_p.shape[1]}")

        # Sample q_p for this time step
        q_p_sample = np.array(q_p[:, i].reshape(1, -1))  # Reshape to match input format for RBF

        # Time the interpolation process before and after
        start_interp_time = time.time()

        # Perform the interpolation
        q_s_pred = interpolate_on_the_fly(kdtree, q_p_train_array, q_s_train_array, q_p_sample, epsilon, neighbors)

        interp_time_before_transpose = time.time()
        print(f"Interpolation (before transpose) for time step {i+1} took: {interp_time_before_transpose - start_interp_time:.6f} seconds")

        # Now perform the transpose and measure its time
        q_s_pred = q_s_pred.T

        interp_time = time.time()
        print(f"Transpose operation for time step {i+1} took: {interp_time - interp_time_before_transpose:.6f} seconds")
        print(f"Total interpolation (with transpose) for time step {i+1} took: {interp_time - start_interp_time:.6f} seconds")

        # Reconstruct the snapshot using the POD-RBF model
        start_reconstruction_time = time.time()
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred.reshape(-1)
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)
        reconstruction_time = time.time()
        print(f"Reconstruction for time step {i+1} took: {reconstruction_time - interp_time:.6f} seconds")

    # Convert list to array and return
    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).squeeze().T
    final_conversion_time = time.time()
    print(f"Final conversion to array took: {final_conversion_time - reconstruction_time:.6f} seconds")
    print(f"Final shape of reconstructed_snapshots_rbf: {reconstructed_snapshots_rbf.shape}")

    total_time = time.time() - start_total_time
    print(f"Total reconstruction process took: {total_time:.6f} seconds")

    return reconstructed_snapshots_rbf

# Function to create gif with all snapshots overlaid
def create_combined_gif(X, original_snapshot, rbf_reconstructed, nTimeSteps, At, latent_dim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 8)

    line_original, = ax.plot(X, original_snapshot[:, 0], 'b-', label='Original Snapshot')
    line_rbf, = ax.plot(X, rbf_reconstructed[:, 0], 'g--', label=f'POD-RBF Reconstructed (inf modes={latent_dim}, sup modes={301})')

    ax.set_title('Snapshot Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_original.set_ydata(original_snapshot[:, frame])
        line_rbf.set_ydata(rbf_reconstructed[:, frame])
        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')
        return line_original, line_rbf

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)
    plt.show()

if __name__ == '__main__':
    # Load the saved KDTree and training data (q_p and q_s)
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)
        kdtree = data['KDTree']
        q_p_train = data['q_p']
        q_s_train = data['q_s']

    # Load a random snapshot from the training_data directory
    # snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot_file = '../FEM/testing_data/simulations/simulation_mu1_4.85_mu2_0.0222.npy'
    snapshot = np.load(snapshot_file)

    # Load U_p and U_s
    U_p = np.load('U_p.npy')
    U_s = np.load('U_s.npy')

    epsilon = 0.1
    neighbors = 10  # Use the nearest 100 neighbors for interpolation

    # Reconstruct the snapshot using dynamic RBF interpolation with nearest neighbors
    start = time.time()
    pod_rbf_reconstructed = reconstruct_snapshot_with_pod_rbf_neighbors(
        snapshot_file, U_p, U_s, q_p_train, q_s_train, kdtree, 28, epsilon, neighbors
    )
    end = time.time()
    print(f"Time {start-end}, for neighbours: {neighbors}")
    print(f"Error: {np.linalg.norm(snapshot-pod_rbf_reconstructed)/np.linalg.norm(snapshot)}")

    np.save("pod_rbf_reconstruction_neighbors.npy", pod_rbf_reconstructed)

    # Domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization and numerical diffusion
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)

    # Create the combined GIF
    create_combined_gif(X, snapshot, pod_rbf_reconstructed, nTimeSteps, At, 28)

