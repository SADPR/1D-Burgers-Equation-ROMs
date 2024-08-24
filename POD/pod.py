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

def plot_singular_values(s, SnapshotsMatrix, tolerances=None):
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
    plt.savefig("modes/singular_value_decay.pdf", format='pdf')
    plt.close()

def plot_modes(U, num_modes=6):
    # Domain and mesh settings
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    fig, axs = plt.subplots(3, 2, figsize=(12, 9))
    axs = axs.flatten()

    for i in range(num_modes):
        mode = U[:, i]
        ax = axs[i]
        ax.plot(X, mode)  # Use X as the x-axis values
        ax.set_xlim([a, b])  # Set the x-axis limits from 0 to 100
        ax.set_title(rf'Mode {i+1}')
        ax.grid(True)

    plt.suptitle(r'First 6 POD Modes', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("modes/first_6_modes.pdf", format='pdf')
    plt.close()


def save_modes(U, s, tolerances, snapshots):
    M, N = np.shape(snapshots)
    singular_values_taken = get_list_of_number_of_singular_values_for_list_of_tolerances(tolerances, s, M, N)

    if not os.path.exists("modes"):
        os.makedirs("modes")

    for tol, num_modes in zip(tolerances, singular_values_taken):
        U_modes = U[:, :num_modes]
        np.save(f"modes/U_modes_tol_{tol:.0e}.npy", U_modes)
        print(f'Saved modes for tolerance {tol:.0e} to U_modes_tol_{tol:.0e}.npy')

if __name__ == '__main__':
    # Load all snapshot files from the training_data directory
    snapshot_files = [f for f in os.listdir("../FEM/training_data") if f.endswith('.npy') and f.startswith('simulation_')]
    
    all_snapshots = []
    for file in snapshot_files:
        snapshots = np.load(os.path.join("../FEM/training_data", file))
        all_snapshots.append(snapshots)
    
    # Stack all snapshots
    all_snapshots = np.hstack(all_snapshots)

    # Compute the SVD of the fluctuation field
    U, s, _ = np.linalg.svd(all_snapshots, full_matrices=False)

    # Define the tolerances
    tols = [1e-1, 5e-2, 2e-2, 1e-2, 1e-3, 1e-4]

    # Plot the singular value decay with tolerance lines
    plot_singular_values(s, all_snapshots, tols)

    # Plot the first 6 modes
    plot_modes(U, num_modes=6)

    # Save the modes for each tolerance
    save_modes(U, s, tols, all_snapshots)

