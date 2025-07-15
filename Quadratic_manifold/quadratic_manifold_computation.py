import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from getH_alpha_functions import getQ, getE, getH

# Load the entire snapshot dataset
data_path = '../FEM/fem_training_data/'
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('fem_simulation_mu1_4.250_mu2_0.0150.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (N, Ns)

# Initialize parameters
epsilon_s = 1e-4  # Tolerance
zeta = 0.1  # Correction factor
omega = 0.01  # Regularization parameter, if not specified, we will determine alpha*
N, Ns = all_snapshots.shape  # N is the number of spatial points, Ns is the number of snapshots

print(f"Loaded snapshot data with dimensions: {all_snapshots.shape}")

# Step 2: Singular Value Decomposition (SVD)

# Uncomment these lines if you need to compute and save the global SVD
# U_S, Sigma_S, Y_S_T = np.linalg.svd(all_snapshots, full_matrices=False)
# np.save('U_global.npy', U_S)
# np.save('Sigma_global.npy', Sigma_S)
# np.save('VT_global.npy', Y_S_T)

# Load the precomputed SVD matrices
U_S = np.load('U_global.npy')
Sigma_S = np.load('Sigma_global.npy')

# Determine n_tra
sigma_cumulative = np.cumsum(Sigma_S) / np.sum(Sigma_S)
n_tra = np.searchsorted(sigma_cumulative, 1 - epsilon_s)

# Compute the dimension n_qua for the quadratic approximation
n_qua = int((np.sqrt(9 + 8 * n_tra) - 3) / 2 * (1 + zeta))

# Final dimension n
n_final = min(n_qua, int((np.sqrt(1 + 8 * Ns) - 1) / 2))

# Truncate U to retain n_final modes
U = U_S[:, :n_final]
q = U.T @ all_snapshots

# Save the truncated U matrix for future use
np.save('U_truncated.npy', U)

# Regularization term
alpha = 10

# Construct matrices Q and E using the provided functions
Q = getQ(n_final, Ns, q)
E = getE(N, Ns, all_snapshots, U, q)

# Compute the coefficient matrix H
H = getH(Q, E, n_final, N, alpha)

# Save the H matrix for future use
np.save('H_quadratic.npy', H)

print("Quadratic approximation manifold computation completed.")







