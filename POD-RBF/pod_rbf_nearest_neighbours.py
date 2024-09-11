import os
import numpy as np
import pickle
from scipy.spatial import KDTree

# Function to compute the Gaussian RBF kernel
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

# Function to solve for weights w for each training point using nearest neighbors
def solve_for_weights_neighbors(X_train, Y_train, epsilon, neighbors):
    """Solve for weights w using the neighbor-based Phi matrix, without storing the full matrix."""
    W = np.zeros_like(Y_train)  # Initialize weights matrix

    # Build a KDTree for finding nearest neighbors efficiently
    tree = KDTree(X_train)

    # Solve for each point's weights separately using its neighbors
    for i in range(len(X_train)):
        print(f"Solving for point {i} of {len(X_train)}")

        # Find the nearest neighbors (including the point itself)
        dist, idx = tree.query(X_train[i], k=neighbors)

        # Extract the neighbor points
        X_neighbors = X_train[idx]
        Y_neighbors = Y_train[idx, :]

        # Compute pairwise distances between the neighbors
        dists_neighbors = np.linalg.norm(X_neighbors[:, None, :] - X_neighbors[None, :, :], axis=-1)

        # Compute the RBF matrix for the neighbors
        Phi_neighbors = gaussian_rbf(dists_neighbors, epsilon)  # Now Phi_neighbors is (100, 100)

        # Solve the small linear system for the current point
        Phi_neighbors += np.eye(neighbors) * 1e-8  # Regularization
        W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)

        # Store only the weight for the current point (corresponding to i-th point)
        W[i, :] = W_neighbors[0]

    return W

# Function to interpolate using the precomputed weights with neighbors
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

# Load snapshot data
data_path = '../FEM/training_data/'
all_snapshots = []

# Hardcoded list of filenames
hardcoded_files = [
    'simulation_mu1_4.75_mu2_0.0164.npy',
    'simulation_mu1_4.75_mu2_0.0225.npy',
    'simulation_mu1_4.75_mu2_0.0243.npy',
    'simulation_mu1_4.75_mu2_0.0282.npy',
    'simulation_mu1_4.76_mu2_0.0181.npy',
    'simulation_mu1_4.76_mu2_0.0182.npy',
    'simulation_mu1_4.76_mu2_0.0240.npy',
    'simulation_mu1_4.76_mu2_0.0290.npy',
    'simulation_mu1_4.77_mu2_0.0244.npy',
    'simulation_mu1_4.77_mu2_0.0250.npy',
    'simulation_mu1_4.77_mu2_0.0293.npy',
    'simulation_mu1_4.78_mu2_0.0156.npy',
    'simulation_mu1_4.78_mu2_0.0170.npy',
    'simulation_mu1_4.78_mu2_0.0212.npy',
    'simulation_mu1_4.78_mu2_0.0274.npy',
    'simulation_mu1_4.78_mu2_0.0279.npy',
    'simulation_mu1_4.79_mu2_0.0210.npy',
    'simulation_mu1_4.79_mu2_0.0248.npy',
    'simulation_mu1_4.79_mu2_0.0284.npy',
    'simulation_mu1_4.79_mu2_0.0287.npy',
    'simulation_mu1_4.80_mu2_0.0186.npy',
    'simulation_mu1_4.80_mu2_0.0191.npy',
    'simulation_mu1_4.80_mu2_0.0221.npy'
]

# Load only the snapshots from the hardcoded list
for filename in hardcoded_files:
    file_path = os.path.join(data_path, filename)
    snapshots = np.load(file_path)
    all_snapshots.append(snapshots)

# # Load snapshot data
# data_path = '../FEM/training_data/'
# files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
# all_snapshots = []

# for file in files:
#     snapshots = np.load(file)
#     all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)
print(f"All snapshots shape: {all_snapshots.shape}")

# # Perform SVD on the snapshots without normalization
U, S, VT = np.linalg.svd(all_snapshots, full_matrices=False)

# # Set the number of modes for principal and secondary modes
r = 28  # Number of principal modes
R = 301  # Total number of modes

U_p = U[:, :r]
U_s = U[:, r:R]
U_combined = np.hstack((U_p, U_s))  # Combine U_p and U_s into a single matrix

# Save U_p and U_s
np.save('U_p.npy', U_p)
np.save('U_s.npy', U_s)

# # Load U_p and U_s
# U_p = np.load('U_p.npy')
# U_s = np.load('U_s.npy')
# U_combined = np.hstack((U_p, U_s))

# Prepare training data
q = U_combined.T @ all_snapshots  # Project snapshots onto the combined POD basis
q_p = q[:r, :]  # Principal modes
q_s = q[r:R, :]  # Secondary modes
print(f"q_p shape: {q_p.shape}, q_s shape: {q_s.shape}")

# Set the number of neighbors to use
neighbors = 100
epsilon = 1.0  # Set the epsilon parameter for the Gaussian RBF

# Solve for the weights using the neighbor-based Phi matrix (without storing Phi)
W = solve_for_weights_neighbors(np.array(q_p.T), np.array(q_s.T), epsilon=epsilon, neighbors=neighbors)
print(f"Final weights W shape: {W.shape}")

# Save the weights and training inputs
with open('rbf_weights.pkl', 'wb') as f:
    pickle.dump((q_p.T, W), f)

print("RBF model weights have been saved successfully.")

