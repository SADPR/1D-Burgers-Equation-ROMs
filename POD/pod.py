import numpy as np
from matplotlib import pyplot as plt
import os

def get_number_of_singular_values_for_given_tolerance(M, N, s, epsilon):
    dimMATRIX = max(M, N)
    tol = dimMATRIX * np.finfo(float).eps * max(s) / 2
    R = np.sum(s > tol)  # Definition of numerical rank
    if epsilon == 0:
        K = R
    else:
        SingVsq = np.multiply(s, s)
        SingVsq.sort()
        normEf2 = np.sqrt(np.cumsum(SingVsq))
        epsilon = epsilon * normEf2[-1]  # relative tolerance
        T = sum(normEf2 < epsilon)
        K = len(s) - T
    K = min(R, K)
    return K

def get_list_of_number_of_singular_values_for_list_of_tolerances(tolerances, s, M, N):
    number_of_singular_values = []
    for tol in tolerances:
        number_of_singular_values.append(get_number_of_singular_values_for_given_tolerance(M, N, s, tol))
    return number_of_singular_values

def plot_singular_values(s, SnapshotsMatrix, tolerances=None):
    M, N = np.shape(SnapshotsMatrix)

    plt.plot(s, marker='s', markevery=5, linewidth=1)  # alpha=0.9
    plt.yscale('log')
    colours = ['k', 'm', 'g', 'b', 'c', 'r', 'y']
    plt.ylabel(r'$\frac{\sigma_i }{\sigma_1}$ log scale', size=15)
    if tolerances is not None:
        singular_values_taken = get_list_of_number_of_singular_values_for_list_of_tolerances(tolerances, s, M, N)
        counter = 0
        for tol, sigmas in zip(tolerances, singular_values_taken):
            print(f'for a tolerance of {tol}, {sigmas} modes are required')
            plt.axvline(x=sigmas, ymin=0.05, ymax=0.95, c=colours[counter], label=r'$\epsilon$ = ' + "{:.0e}".format(tol))
            counter += 1
    plt.grid()
    plt.legend()
    plt.savefig("modes/singular_value_decay.png")
    plt.close()

def save_modes(U, s, tolerances, snapshots):
    M, N = np.shape(snapshots)
    singular_values_taken = get_list_of_number_of_singular_values_for_list_of_tolerances(tolerances, s, M, N)
    mean_snapshot = np.mean(snapshots, axis=1)

    if not os.path.exists("modes"):
        os.makedirs("modes")

    for tol, num_modes in zip(tolerances, singular_values_taken):
        U_modes = U[:, :num_modes]
        # reconstructed_snapshots = U_modes @ np.diag(S_modes) @ V_modes #+ mean_snapshot[:, np.newaxis]

        np.save(f"modes/U_modes_tol_{tol:.0e}.npy", U_modes)
        # np.save(f"modes/reconstructed_snapshots_tol_{tol:.0e}.npy", reconstructed_snapshots)

        print(f'Saved modes for tolerance {tol:.0e} to U_modes_tol_{tol:.0e}.npy, S_modes_tol_{tol:.0e}.npy, V_modes_tol_{tol:.0e}.npy, and reconstructed_snapshots_tol_{tol:.0e}.npy')

if __name__ == '__main__':
    # Load all snapshot files from the training_data directory
    snapshot_files = [f for f in os.listdir("training_data") if f.endswith('.npy') and f.startswith('simulation_')]
    
    all_snapshots = []
    for file in snapshot_files:
        snapshots = np.load(os.path.join("training_data", file))
        all_snapshots.append(snapshots)
    
    # Stack all snapshots
    all_snapshots = np.hstack(all_snapshots)

    # Compute the mean of the snapshots
    mean_snapshot = np.mean(all_snapshots, axis=1)

    # Subtract the mean from each snapshot to get the fluctuation field
    fluctuations = all_snapshots - mean_snapshot[:, np.newaxis]

    # Compute the SVD of the fluctuation field
    U, s, _ = np.linalg.svd(all_snapshots, full_matrices=False)

    # Define the tolerances
    tols = [1e-1, 5e-2, 2e-2, 1e-2, 1e-3, 1e-4]

    # Plot the singular value decay with tolerance lines
    plot_singular_values(s, all_snapshots, tols)

    # Save the modes for each tolerance
    save_modes(U, tols, all_snapshots)
