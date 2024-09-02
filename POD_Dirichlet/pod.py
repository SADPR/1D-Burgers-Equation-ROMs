import numpy as np
from matplotlib import pyplot as plt
import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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

def plot_singular_values(s, SnapshotsMatrix, tolerances=None, label_suffix=""):
    M, N = np.shape(SnapshotsMatrix)

    plt.plot(s, marker='s', markevery=5, linewidth=1)  # alpha=0.9
    plt.yscale('log')
    colours = ['k', 'm', 'g', 'b', 'c', 'r', 'y']
    plt.ylabel(r'$\frac{\sigma_i }{\sigma_{max}}$ (log scale)', size=15)
    plt.xlabel(r'Index of $\sigma_i$', size=15)
    if tolerances is not None:
        singular_values_taken = get_list_of_number_of_singular_values_for_list_of_tolerances(tolerances, s, M, N)
        counter = 0
        for tol, sigmas in zip(tolerances, singular_values_taken):
            print(f'For a tolerance of {tol}, {sigmas} modes are required')
            plt.axvline(x=sigmas, ymin=0.05, ymax=0.95, c=colours[counter], label=r'$\epsilon$ = ' + "{:.0e}".format(tol))
            counter += 1
    plt.grid()
    plt.legend()
    plt.savefig(f"modes/singular_value_decay{label_suffix}.pdf", format='pdf')
    plt.close()

def save_modes(U, s, tolerances, snapshots, label_suffix=""):
    M, N = np.shape(snapshots)
    singular_values_taken = get_list_of_number_of_singular_values_for_list_of_tolerances(tolerances, s, M, N)

    if not os.path.exists("modes"):
        os.makedirs("modes")

    for tol, num_modes in zip(tolerances, singular_values_taken):
        U_modes = U[:, :num_modes]
        np.save(f"modes/U_modes{label_suffix}_tol_{tol:.0e}.npy", U_modes)
        print(f'Saved modes for tolerance {tol:.0e} to U_modes{label_suffix}_tol_{tol:.0e}.npy')

if __name__ == '__main__':
    # Define Dirichlet nodes
    dirichlet_nodes = [0]  # Example: first node is a Dirichlet node
    free_nodes = np.setdiff1d(np.arange(256 * 2 + 1), dirichlet_nodes)  # Adjusted for your mesh settings

    # Load all snapshot files from the training_data directory
    snapshot_files = [f for f in os.listdir("../FEM/training_data") if f.endswith('.npy') and f.startswith('simulation_')]
    
    all_snapshots = []
    for file in snapshot_files:
        snapshots = np.load(os.path.join("../FEM/training_data", file))
        all_snapshots.append(snapshots)
    
    # Stack all snapshots
    all_snapshots = np.hstack(all_snapshots)

    # Separate snapshots for free and Dirichlet nodes
    snapshots_free_nodes = all_snapshots[free_nodes, :]
    snapshots_dirichlet_nodes = all_snapshots[dirichlet_nodes, :]

    # Compute the SVD for the free nodes
    U_free, s_free, _ = np.linalg.svd(snapshots_free_nodes, full_matrices=False)

    # Compute the SVD for the Dirichlet nodes
    U_dirichlet, s_dirichlet, _ = np.linalg.svd(snapshots_dirichlet_nodes, full_matrices=False)

    # Define the tolerances
    tols = [1e-1, 5e-2, 2e-2, 1e-2, 1e-3, 1e-4]

    # Plot the singular value decay with tolerance lines for free nodes
    plot_singular_values(s_free, snapshots_free_nodes, tols, label_suffix="_free")

    # Plot the singular value decay with tolerance lines for Dirichlet nodes
    plot_singular_values(s_dirichlet, snapshots_dirichlet_nodes, tols, label_suffix="_dirichlet")

    # Save the modes for each tolerance for free nodes
    save_modes(U_free, s_free, tols, snapshots_free_nodes, label_suffix="_free")

    # Save the modes for each tolerance for Dirichlet nodes
    save_modes(U_dirichlet, s_dirichlet, tols, snapshots_dirichlet_nodes, label_suffix="_dirichlet")



