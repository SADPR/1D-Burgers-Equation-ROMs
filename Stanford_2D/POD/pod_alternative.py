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
    plt.ylabel(r'$\frac{\sigma_i }{\sigma_{\max}}$ (log scale)', size=15)
    plt.xlabel(r'Index of $\sigma_i$', size=15)

    for idx, (tol, num_modes) in enumerate(modes_for_tolerances):
        plt.axvline(x=num_modes, ymin=0.05, ymax=0.95, c=colours[idx % len(colours)],
                    label=r'$\epsilon$ = ' + "{:.0e}".format(tol))

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("modes/singular_value_decay_corrected.pdf", format='pdf')
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
    
    # Add this boolean flag to control reconstruction-based mode selection
    perform_reconstruction = False  # Set to False to skip mode selection by reconstruction
    
    # Ensure the 'modes' directory exists
    if not os.path.exists("modes"):
        os.makedirs("modes")

    # Directory where the FOM solutions are saved
    fom_directory = "../Burgers_2D/FOM_Solutions"
    snapshot_files = [f for f in os.listdir(fom_directory) if f.endswith('.npz') and f.startswith('FOM_solution')]
    # You can choose to process all snapshot files or select specific ones
    snapshot_files = [snapshot_files[0]]  # Uncomment to use only the first snapshot file

    load_start_time = time.time()
    all_snapshots = []
    for file in snapshot_files:
        # Load the .npz file
        data = np.load(os.path.join(fom_directory, file))
        U_FOM = data['U_FOM']  # Shape: (n_nodes, n_time_steps + 1, 2)
        U0 = U_FOM[:, 0, :]    # Initial condition at t=0

        # Compute delta U's: delta_U_n = U_n - U_{n-1}
        # U_FOM[:, 1:, :] are the solutions from time step 1 to the end
        # U_FOM[:, :-1, :] are the solutions from time step 0 to the second last
        delta_U = U_FOM[:, 1:, :] - U_FOM[:, :-1, :]  # Shape: (n_nodes, n_time_steps, 2)

        # Flatten delta_U for both velocity components u_x and u_y for each time step
        n_nodes, n_time_steps, _ = delta_U.shape
        delta_U_flattened = delta_U.reshape(n_nodes * 2, n_time_steps)

        # Collect all delta_U snapshots across all parameter points
        all_snapshots.append(delta_U_flattened)

    # Stack all snapshots horizontally
    all_snapshots = np.hstack(all_snapshots)
    print(f"Loading and computing delta_U snapshots took {time.time() - load_start_time:.2f} seconds.")

    # Perform SVD on the snapshot matrix
    svd_start_time = time.time()
    U, s, _ = np.linalg.svd(all_snapshots, full_matrices=False)
    print(f"SVD computation took {time.time() - svd_start_time:.2f} seconds.")

    # Define the tolerances
    tols = [1e-10]

    # List to store the number of modes for each tolerance
    modes_for_tolerances = []

    # If perform_reconstruction is True, we calculate modes based on tolerance
    if perform_reconstruction:
        # Loop through each tolerance and calculate the number of modes for it
        for tol in tols:
            tol_start_time = time.time()
            num_modes, error = get_number_of_modes_for_tolerance(all_snapshots, U, s, tol)
            modes_for_tolerances.append((tol, num_modes))
            print(f"For tolerance {tol}, the reconstruction error is {error:.6e} with {num_modes} modes. Time taken: {time.time() - tol_start_time:.2f} seconds.")
    else:
        # If reconstruction is disabled, use singular values to estimate number of modes
        for tol in tols:
            total_energy = np.sum(s ** 2)
            cumulative_energy = np.cumsum(s ** 2)
            num_modes = np.searchsorted(cumulative_energy / total_energy, 1 - tol) + 1
            modes_for_tolerances.append((tol, num_modes))
            print(f"Estimated {num_modes} modes for tolerance {tol} based on singular values (without reconstruction).")

    # Save the modes based on the calculated number of modes for each tolerance
    save_start_time = time.time()
    save_modes(U, s, tols, all_snapshots, modes_for_tolerances)
    print(f"Saving modes took {time.time() - save_start_time:.2f} seconds.")

    # Plot the singular value decay with the vertical lines for the number of modes
    plot_start_time = time.time()
    plot_singular_values(s, all_snapshots, modes_for_tolerances)
    print(f"Plotting the singular value decay took {time.time() - plot_start_time:.2f} seconds.")

    print(f"Total time taken: {time.time() - overall_start_time:.2f} seconds.")
