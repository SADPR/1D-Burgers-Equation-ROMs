import numpy as np
import os
import time
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_number_of_modes_for_tolerance(U_FOM, U, s, tolerance):
    """
    Find the minimum number of modes required such that the reconstruction error is within the given tolerance.
    
    Parameters:
    U_FOM - Original snapshot matrix
    U - Matrix of left singular vectors from the SVD
    s - Singular values from the SVD
    tolerance - Desired reconstruction tolerance
    
    Returns:
    num_modes - Number of modes required to match the tolerance
    """
    total_energy = np.sum(s ** 2)
    cumulative_energy = np.cumsum(s ** 2)
    
    for num_modes in range(1, len(s) + 1):
        # Reconstruct the snapshots using the selected number of modes
        Phi = U[:, :num_modes]
        U_approx = Phi @ (Phi.T @ U_FOM)
        
        # Calculate the relative error: norm(original - approximation) / norm(original)
        relative_error = np.linalg.norm(U_FOM - U_approx) / np.linalg.norm(U_FOM)
        
        if relative_error <= tolerance:
            print(f"Tolerance {tolerance:.1e}: {num_modes} modes yield an error of {relative_error:.6e}")
            return num_modes, relative_error

    # Return the maximum number of modes if the tolerance is not met
    return len(s), relative_error

def plot_singular_values(s, SnapshotsMatrix, modes_for_tolerances):
    M, N = np.shape(SnapshotsMatrix)

    plt.figure(figsize=(8, 6))
    plt.plot(s, marker='s', markevery=5, linewidth=1)  
    plt.yscale('log')
    colours = ['k', 'm', 'g', 'b', 'c', 'r', 'y']
    plt.ylabel(r'$\sigma_i$ (log scale)', size=15)
    plt.xlabel(r'Index of $\sigma_i$', size=15)

    for idx, (tol, num_modes) in enumerate(modes_for_tolerances):
        plt.axvline(x=num_modes - 1, ymin=0.05, ymax=0.95, c=colours[idx % len(colours)],
                    label=r'$\epsilon$ = ' + "{:.0e}".format(tol))

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("modes/singular_value_decay.pdf", format='pdf')
    plt.close()

def save_modes(U, s, tolerances, snapshots, modes_for_tolerances):
    """
    Save the modes based on the correct number of modes for each tolerance.
        
    Parameters:
    U - Left singular vectors (POD modes) from SVD
    tolerances - List of tolerances
    snapshots - Snapshot matrix (for shape reference)
    modes_for_tolerances - List of tuples (tolerance, num_modes) for saving modes
    """
    if not os.path.exists("modes"):
        os.makedirs("modes")

    for tol, num_modes in modes_for_tolerances:
        start_time = time.time()
        U_modes = U[:, :num_modes]
        np.save(f"modes/U_modes_tol_{tol:.0e}.npy", U_modes)
        print(f'Saved modes for tolerance {tol:.0e} with {num_modes} modes. Time taken: {time.time() - start_time:.2f} seconds.')

if __name__ == '__main__':
    overall_start_time = time.time()
    
    # Ensure the 'modes' directory exists
    if not os.path.exists("modes"):
        os.makedirs("modes")

    # Directory where the FOM solutions are saved
    fom_directory = "../Burgers_2D/FOM_Solutions"
    snapshot_files = [f for f in os.listdir(fom_directory) if f.endswith('.npy') and f.startswith('U_FOM')]
    snapshot_files = [snapshot_files[0]]  # Use only the first snapshot file

    load_start_time = time.time()
    all_snapshots = []
    for file in snapshot_files:
        # Load the .npy file
        U_FOM = np.load(os.path.join(fom_directory, file))  # Shape: (nTimeSteps + 1, total_dofs)
        n_time_steps_plus_one, total_dofs = U_FOM.shape

        # Transpose U_FOM to get snapshot matrix of shape (total_dofs, nTimeSteps + 1)
        SnapshotMatrix = U_FOM.T  # Shape: (total_dofs, nTimeSteps + 1)
        all_snapshots.append(SnapshotMatrix)

    # Stack all snapshots horizontally (if multiple files are used)
    all_snapshots = np.hstack(all_snapshots)
    print(f"Loading and reshaping snapshots took {time.time() - load_start_time:.2f} seconds.")
    print(f"Snapshot matrix shape: {all_snapshots.shape}")

    # Perform SVD on the snapshot matrix
    svd_start_time = time.time()
    U, s, Vh = np.linalg.svd(all_snapshots, full_matrices=False)
    print(f"SVD computation took {time.time() - svd_start_time:.2f} seconds.")

    # Print singular values
    print("Singular values:")
    print(s)

    # Define the tolerances
    tols = [1e-3, 1e-4, 1e-6, 1e-10]

    # List to store the number of modes for each tolerance
    modes_for_tolerances = []

    # Perform reconstruction-based mode selection
    for tol in tols:
        tol_start_time = time.time()
        num_modes, error = get_number_of_modes_for_tolerance(all_snapshots, U, s, tol)
        modes_for_tolerances.append((tol, num_modes))
        print(f"For tolerance {tol}, the reconstruction error is {error:.6e} with {num_modes} modes. Time taken: {time.time() - tol_start_time:.2f} seconds.")

    # Save the modes based on the calculated number of modes for each tolerance
    save_start_time = time.time()
    save_modes(U, s, tols, all_snapshots, modes_for_tolerances)
    print(f"Saving modes took {time.time() - save_start_time:.2f} seconds.")

    # Plot the singular value decay with the vertical lines for the number of modes
    plot_start_time = time.time()
    plot_singular_values(s, all_snapshots, modes_for_tolerances)
    print(f"Plotting the singular value decay took {time.time() - plot_start_time:.2f} seconds.")

    print(f"Total time taken: {time.time() - overall_start_time:.2f} seconds.")
