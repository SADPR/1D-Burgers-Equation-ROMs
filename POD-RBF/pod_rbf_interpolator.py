import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist, squareform
import os
import pickle

# # Load snapshot data
# data_path = '../FEM/training_data/'
# files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
# all_snapshots = []

# Now you can iterate through the 'files' list and load the relevant snapshots
# for file in files:
#     snapshots = np.load(file)
#     all_snapshots.append(snapshots)

# all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)

import os
import numpy as np

# Define the hardcoded list of filenames
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

# hardcoded_files = [
#     'simulation_mu1_4.76_mu2_0.0181.npy',
#     'simulation_mu1_4.76_mu2_0.0182.npy',
#     'simulation_mu1_4.76_mu2_0.0240.npy',
#     'simulation_mu1_4.76_mu2_0.0290.npy'
# ]

# Load snapshot data
data_path = '../FEM/training_data/'  # Adjust this path as necessary
all_snapshots = []

# Load only the snapshots from the hardcoded list
for filename in hardcoded_files:
    file_path = os.path.join(data_path, filename)
    snapshots = np.load(file_path)
    all_snapshots.append(snapshots)

# Combine the loaded snapshots into a single array if needed
all_snapshots = np.hstack(all_snapshots)

print(f"Loaded {len(hardcoded_files)} snapshot files.")

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

def remove_duplicates_optimized(X, threshold=1e-8):
    """Remove near-duplicate rows in matrix X based on a distance threshold in an optimized way."""
    unique_indices = []  # Store indices of unique rows
    unique_rows = []  # Store unique rows for easy comparison

    for i in range(X.shape[0]):
        row = X[i]
        is_unique = True

        # Compare this row with all previously stored unique rows
        for unique_row in unique_rows:
            if np.linalg.norm(row - unique_row) < threshold:  # Check distance threshold
                is_unique = False
                break

        if is_unique:
            unique_indices.append(i)  # Mark the index as unique
            unique_rows.append(row)  # Add this row to the list of unique rows

    return X[unique_indices], np.array(unique_indices)

# Remove redundant points from q_p
q_p_train_unique, unique_indices = remove_duplicates_optimized(q_p.T)
q_s_train_unique = q_s.T[unique_indices]  # Match the secondary data to the unique principal modes
print("Hey")

# Train the RBF model on the entire (cleaned) training dataset
# Use all unique training data without regularization
rbf_interp = RBFInterpolator(q_p_train_unique, q_s_train_unique, kernel='gaussian', epsilon=2.0, neighbors=100)

# Save the trained RBF model to a file
with open('rbf_model.pkl', 'wb') as f:
    pickle.dump(rbf_interp, f)

print("RBF model has been trained and saved successfully.")


