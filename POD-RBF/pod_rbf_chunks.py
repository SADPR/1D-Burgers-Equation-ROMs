import os
import numpy as np
import pickle

# Function to compute the Gaussian RBF kernel
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

# Function to compute pairwise Euclidean distances between two sets of points in chunks
def compute_distances_in_chunks(X1, X2, chunk_size=1000):
    """Compute pairwise Euclidean distances between two sets of points in chunks to avoid memory issues."""
    num_samples_X1 = X1.shape[0]
    num_samples_X2 = X2.shape[0]

    # Initialize the distance matrix
    dists = np.zeros((num_samples_X1, num_samples_X2))

    # Compute distances in chunks
    for i in range(0, num_samples_X1, chunk_size):
        print(f"Chunk: {i}")
        end_i = min(i + chunk_size, num_samples_X1)
        for j in range(0, num_samples_X2, chunk_size):
            end_j = min(j + chunk_size, num_samples_X2)
            dists[i:end_i, j:end_j] = np.sqrt(np.sum((X1[i:end_i, None, :] - X2[None, j:end_j, :]) ** 2, axis=-1))

    return dists

# Function to compute the interpolation matrix Phi
def compute_interpolation_matrix(X_train, epsilon, chunk_size=100):
    """Compute the interpolation matrix Phi based on the distances between training points."""
    dists = compute_distances_in_chunks(X_train, X_train, chunk_size)
    Phi = gaussian_rbf(dists, epsilon)
    return Phi

# Function to solve for weights w in the system Phi * w = Y_train
def solve_for_weights(Phi, Y_train):
    """Solve for weights w in the system Phi * w = Y_train."""
    # Add a small regularization term to the diagonal for numerical stability
    Phi += np.eye(Phi.shape[0]) * 1e-8
    # Solve the system
    W = np.linalg.solve(Phi, Y_train)
    return W

# Function to interpolate using the precomputed weights
def interpolate_with_weights(X_train, W, x_new, epsilon):
    """Interpolate at new points using precomputed weights."""
    # Compute distances between the new point and training points
    dists = compute_distances_in_chunks(x_new[None, :], X_train)
    # Compute RBF values
    rbf_values = gaussian_rbf(dists, epsilon)
    # Compute the interpolated value
    f_new = rbf_values @ W  # Shape: (output_dim,)
    return f_new

# # Load snapshot data
# data_path = '../FEM/training_data/'
# all_snapshots = []

# # Hardcoded list of filenames
# hardcoded_files = [
#     'simulation_mu1_4.75_mu2_0.0164.npy',
#     'simulation_mu1_4.75_mu2_0.0225.npy',
#     'simulation_mu1_4.75_mu2_0.0243.npy',
#     'simulation_mu1_4.75_mu2_0.0282.npy',
#     'simulation_mu1_4.76_mu2_0.0181.npy',
#     'simulation_mu1_4.76_mu2_0.0182.npy',
#     'simulation_mu1_4.76_mu2_0.0240.npy',
#     'simulation_mu1_4.76_mu2_0.0290.npy',
#     'simulation_mu1_4.77_mu2_0.0244.npy',
#     'simulation_mu1_4.77_mu2_0.0250.npy',
#     'simulation_mu1_4.77_mu2_0.0293.npy',
#     'simulation_mu1_4.78_mu2_0.0156.npy',
#     'simulation_mu1_4.78_mu2_0.0170.npy',
#     'simulation_mu1_4.78_mu2_0.0212.npy',
#     'simulation_mu1_4.78_mu2_0.0274.npy',
#     'simulation_mu1_4.78_mu2_0.0279.npy',
#     'simulation_mu1_4.79_mu2_0.0210.npy',
#     'simulation_mu1_4.79_mu2_0.0248.npy',
#     'simulation_mu1_4.79_mu2_0.0284.npy',
#     'simulation_mu1_4.79_mu2_0.0287.npy',
#     'simulation_mu1_4.80_mu2_0.0186.npy',
#     'simulation_mu1_4.80_mu2_0.0191.npy',
#     'simulation_mu1_4.80_mu2_0.0221.npy'
# ]

# # Load only the snapshots from the hardcoded list
# for filename in hardcoded_files:
#     file_path = os.path.join(data_path, filename)
#     snapshots = np.load(file_path)
#     all_snapshots.append(snapshots)

# # Combine the loaded snapshots into a single array
# all_snapshots = np.hstack(all_snapshots)

# Load snapshot data
data_path = '../FEM/training_data/'
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)

# Perform SVD on the snapshots without normalization
U, S, VT = np.linalg.svd(all_snapshots, full_matrices=False)

# Set the number of modes for principal and secondary modes
r = 28  # Number of principal modes
R = 301  # Total number of modes

U_p = U[:, :r]
U_s = U[:, r:R]
U_combined = np.hstack((U_p, U_s))  # Combine U_p and U_s into a single matrix

# Save U_p and U_s
np.save('U_p.npy', U_p)
np.save('U_s.npy', U_s)

# Prepare training data
q = U_combined.T @ all_snapshots  # Project snapshots onto the combined POD basis
q_p = q[:r, :]  # Principal modes
q_s = q[r:R, :]  # Secondary modes

# Compute the interpolation matrix Phi for the training data
epsilon = 1.0  # Set the epsilon parameter for the Gaussian RBF
Phi = compute_interpolation_matrix(np.array(q_p.T), epsilon=epsilon)

# Solve for the weights W
W = solve_for_weights(Phi, np.array(q_s.T))  # W has shape (n_samples, output_dim)

# Save the weights and training inputs
with open('rbf_weights.pkl', 'wb') as f:
    pickle.dump((q_p.T, W), f)

print("RBF model weights have been saved successfully.")

