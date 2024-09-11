import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import os
from scipy.spatial import KDTree

# Function to compute Euclidean distances between two sets of points
def compute_distances(X1, X2):
    """Compute pairwise Euclidean distances between two sets of points."""
    dists = np.sqrt(np.sum((X1 - X2) ** 2, axis=1))
    return dists

# Function to compute the Gaussian RBF kernel
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

# Function to interpolate using precomputed weights and nearest neighbors
def interpolate_with_weights_neighbors(X_train, W, x_new, epsilon, neighbors):
    """Interpolate at new points using precomputed weights and nearest neighbors."""
    # Build a KDTree for finding nearest neighbors efficiently
    tree = KDTree(X_train)

    # Find the nearest neighbors for the new point
    dist, idx = tree.query(x_new, k=neighbors)

    # Compute RBF values for the neighbors
    rbf_values = gaussian_rbf(dist, epsilon)

    # Interpolate the new value using the precomputed weights
    W_neighbors = W[idx, :]  # Get the weights for the neighbors
    f_new = rbf_values @ W_neighbors  # Compute the interpolated value

    return f_new

# Function to reconstruct snapshot using precomputed RBF weights and nearest neighbors
def reconstruct_snapshot_with_pod_rbf_neighbors(snapshot_file, U_p, U_s, q_p_train, W, r, epsilon, neighbors):
    # Load the snapshot file
    snapshots = np.load(snapshot_file)

    # Project onto the POD basis to get q_p
    q = U_p.T @ snapshots
    q_p = q[:r, :]

    # Reconstruct the snapshots using precomputed RBF weights
    reconstructed_snapshots_rbf = []
    for i in range(q_p.shape[1]):
        q_p_sample = np.array(q_p[:, i].reshape(1, -1))  # Reshape to match input format for RBF

        # Use nearest neighbors interpolation
        q_s_pred = interpolate_with_weights_neighbors(np.array(q_p_train), np.array(W), q_p_sample, epsilon, neighbors).T

        # Reconstruct the snapshot using the POD-RBF model
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred.reshape(-1)
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)

    # Convert list to array and return
    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).squeeze().T
    print(f"Final shape of reconstructed_snapshots_rbf: {reconstructed_snapshots_rbf.shape}")
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
    # Load the RBF model data (q_p_train and precomputed weights W)
    with open('rbf_weights.pkl', 'rb') as f:
        q_p_train, W = pickle.load(f)

    # Load a random snapshot from the training_data directory
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Load U_p and U_s
    U_p = np.load('U_p.npy')
    U_s = np.load('U_s.npy')

    epsilon = 1.0
    neighbors = 100  # Use the nearest 100 neighbors for interpolation

    # Reconstruct the snapshot using precomputed RBF weights and nearest neighbors
    pod_rbf_reconstructed = reconstruct_snapshot_with_pod_rbf_neighbors(
        snapshot_file, U_p, U_s, q_p_train, W, 28, epsilon, neighbors
    )

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
