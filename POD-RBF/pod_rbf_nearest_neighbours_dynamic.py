import os
import numpy as np
import pickle
from scipy.spatial import KDTree

# Load snapshot data
data_path = '../FEM/training_data/'  # Adjust this path as necessary
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

# Combine the loaded snapshots into a single array
all_snapshots = np.hstack(all_snapshots)
print(f"Loaded {len(hardcoded_files)} snapshot files.")

# Perform SVD on the snapshots without normalization
U, S, VT = np.linalg.svd(all_snapshots, full_matrices=False)

# Set the number of modes for principal and secondary modes
r = 28  # Number of principal modes
R = 301  # Total number of modes

# Extract the principal and secondary modes
U_p = U[:, :r]
U_s = U[:, r:R]

# Save U_p and U_s for future use
np.save('U_p.npy', U_p)
np.save('U_s.npy', U_s)

# Prepare training data (projection onto the combined POD basis)
U_combined = np.hstack((U_p, U_s))
q = U_combined.T @ all_snapshots  # Project snapshots onto the combined POD basis
q_p = q[:r, :]  # Principal modes
q_s = q[r:R, :]  # Secondary modes

# Build a KDTree for q_p (principal modes)
kdtree = KDTree(q_p.T)

# Save the KDTree and q_p, q_s data for future interpolation
with open('training_data.pkl', 'wb') as f:
    pickle.dump({'KDTree': kdtree, 'q_p': q_p.T, 'q_s': q_s.T}, f)

print("Training data and KDTree have been saved successfully.")



