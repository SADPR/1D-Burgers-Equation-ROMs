import os
import numpy as np
import time
import pickle
from sklearn.utils.extmath import randomized_svd
from scipy.spatial import KDTree

def perform_rsvd_and_extract_modes(all_snapshots, primary_modes=10, total_modes=150):
    """
    Perform randomized SVD and extract primary and secondary modes.
    
    Parameters:
    all_snapshots - Snapshot matrix
    primary_modes - Number of primary modes
    total_modes - Total number of modes to retain
    
    Returns:
    U_p - Primary modes
    U_s - Secondary modes
    q - Projected coordinates onto the combined POD basis
    q_p - Projected primary coordinates
    q_s - Projected secondary coordinates
    """
    # Step 1: Perform randomized SVD with 150 modes
    print("Performing RSVD...")
    start_time = time.time()
    U, s, Vh = randomized_svd(all_snapshots, n_components=total_modes)
    print(f"RSVD took {time.time() - start_time:.2f} seconds.")
    
    # Step 2: Extract primary and secondary modes
    U_p = U[:, :primary_modes]  # Primary modes
    U_s = U[:, primary_modes:total_modes]  # Secondary modes

    # Step 3: Project the snapshot matrix onto the combined POD basis
    print("Projecting snapshots onto the combined POD basis...")
    projection_start_time = time.time()
    U_combined = np.hstack((U_p, U_s))
    q = U_combined.T @ all_snapshots  # Project snapshots onto the combined POD basis
    q_p = q[:primary_modes, :]  # Primary mode projections
    q_s = q[primary_modes:total_modes, :]  # Secondary mode projections
    print(f"Projection took {time.time() - projection_start_time:.2f} seconds.")
    
    return U_p, U_s, q, q_p, q_s

def remove_duplicates(data, tolerance=1e-8):
    """Remove near-duplicate rows from data."""
    _, unique_indices = np.unique(np.round(data / tolerance) * tolerance, axis=0, return_index=True)
    return unique_indices

def build_kdtree_and_save_data(q_p, q_s, filename):
    """
    Build a KDTree using unique q_p points and save it along with q_p and q_s.
    
    Parameters:
    q_p - Projected primary mode coordinates
    q_s - Projected secondary mode coordinates
    filename - Filename for saving the KDTree and training data
    """
    # Step 1: Remove duplicates
    unique_indices = remove_duplicates(q_p.T)
    q_p = q_p[:, unique_indices]
    q_s = q_s[:, unique_indices]
    
    # Step 2: Build the KDTree using unique q_p points
    print("Building KDTree...")
    kdtree_start_time = time.time()
    kdtree = KDTree(q_p.T)
    print(f"KDTree construction took {time.time() - kdtree_start_time:.2f} seconds.")
    
    # Step 3: Save the KDTree and q_p, q_s data for future interpolation
    with open(filename, 'wb') as f:
        pickle.dump({'KDTree': kdtree, 'q_p': q_p.T, 'q_s': q_s.T}, f)

    print(f"Training data and KDTree have been saved successfully in {filename}.")

if __name__ == '__main__':
    # Create directory for modes if it doesn't exist
    modes_dir = "modes"
    if not os.path.exists(modes_dir):
        os.makedirs(modes_dir)

    # Load snapshot data
    fom_directory = "../Burgers_2D/FOM_Solutions"
    snapshot_files = [f for f in os.listdir(fom_directory) if f.endswith('.npy') and f.startswith('U_FOM')]
    
    load_start_time = time.time()
    all_snapshots = []
    for file in snapshot_files:
        U_FOM = np.load(os.path.join(fom_directory, file))  
        SnapshotMatrix = U_FOM.T  # Transpose to have shape (total_dofs, n_time_steps)
        all_snapshots.append(SnapshotMatrix)

    # Concatenate all snapshot matrices
    all_snapshots = np.hstack(all_snapshots)
    print(f"Loading and reshaping snapshots took {time.time() - load_start_time:.2f} seconds.")
    print(f"Snapshot matrix shape: {all_snapshots.shape}")

    # Perform RSVD and extract primary and secondary modes
    primary_modes = 10
    total_modes = 150
    U_p, U_s, q, q_p, q_s = perform_rsvd_and_extract_modes(all_snapshots, primary_modes=primary_modes, total_modes=total_modes)

    # Save the modes and projected data for future use
    np.save(os.path.join(modes_dir, 'U_p.npy'), U_p)
    np.save(os.path.join(modes_dir, 'U_s.npy'), U_s)
    np.save(os.path.join(modes_dir, 'q.npy'), q)
    np.save(os.path.join(modes_dir, 'q_p.npy'), q_p)
    np.save(os.path.join(modes_dir, 'q_s.npy'), q_s)
    print("Primary and secondary modes, as well as projected data (q, q_p, q_s), saved successfully.")

    # Build KDTree and save training data
    training_data_filename = os.path.join(modes_dir, 'training_data.pkl')
    build_kdtree_and_save_data(q_p, q_s, training_data_filename)

    print("Processing complete.")
