import numpy as np
from matplotlib import pyplot as plt
import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_number_of_singular_values_for_given_tolerance(M, N, s, epsilon_squared):
    s_sorted = np.sort(s)[::-1]
    squared_cumsum = np.cumsum(s_sorted ** 2)
    squared_total = squared_cumsum[-1]
    squared_relative_loss = 1.0 - (squared_cumsum / squared_total)
    K = np.argmax(squared_relative_loss <= epsilon_squared) + 1
    return K

def get_list_of_number_of_singular_values_for_list_of_tolerances(tolerances_squared, s, M, N):
    return [get_number_of_singular_values_for_given_tolerance(M, N, s, eps2) for eps2 in tolerances_squared]

def plot_singular_values(s, SnapshotsMatrix, tolerances_squared=None):
    M, N = np.shape(SnapshotsMatrix)

    linear_cumsum = np.cumsum(s)
    total = np.sum(s)
    linear_relative_loss = 1.0 - (linear_cumsum / total)

    plt.figure(figsize=(8, 5))
    plt.plot(linear_relative_loss, linewidth=2)
    plt.yscale('log')
    colours = ['k', 'm', 'g', 'b', 'r', 'c', 'y']
    plt.ylabel(r'$1 - \frac{\sum_{i=1}^{n} \sigma_i}{\sum_{i=1}^{r} \sigma_i}$', size=15)
    plt.xlabel(r'Singular value index $n$', size=15)

    if tolerances_squared is not None:
        singular_values_taken = get_list_of_number_of_singular_values_for_list_of_tolerances(tolerances_squared, s, M, N)
        for counter, (eps2, n_modes) in enumerate(zip(tolerances_squared, singular_values_taken)):
            eps_percent = eps2 * 100
            plt.axvline(x=n_modes, ymin=0.05, ymax=0.95, c=colours[counter % len(colours)],
                        label=rf'$\epsilon^2 = {eps_percent:.4g}\%$')

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    os.makedirs("modes", exist_ok=True)
    plt.savefig("modes/singular_value_decay_linear_loss.pdf", format='pdf')
    plt.close()


def plot_modes(U, num_modes=6):
    a, b = 0, 100
    m = 511
    X = np.linspace(a, b, m + 1)

    fig, axs = plt.subplots(3, 2, figsize=(12, 9))
    axs = axs.flatten()

    for i in range(num_modes):
        axs[i].plot(X, U[:, i])
        axs[i].set_xlim([a, b])
        axs[i].set_title(rf'Mode {i+1}')
        axs[i].grid(True)

    plt.suptitle(r'First 6 POD Modes', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(top=0.9)
    # plt.savefig("modes/first_6_modes.pdf", format='pdf')
    plt.close()

def save_modes(U, s, tolerances_squared, snapshots):
    M, N = snapshots.shape
    singular_values_taken = get_list_of_number_of_singular_values_for_list_of_tolerances(tolerances_squared, s, M, N)
    os.makedirs("modes", exist_ok=True)

    for eps2, num_modes in zip(tolerances_squared, singular_values_taken):
        U_modes = U[:, :num_modes]
        np.save(f"modes/U_modes_tol_{eps2:.0e}.npy", U_modes)
        np.save(f"modes/Singular_values_modes_tol_{eps2:.0e}.npy", s[:num_modes])
        print(f'Saved modes for tolerance epsilon^2 = {eps2:.0e} to U_modes_tol_{eps2:.0e}.npy')

if __name__ == '__main__':
    snapshot_files = [f for f in os.listdir("../FEM/fem_training_data") if f.endswith('.npy') and f.startswith('fem_simulation_')]
    all_snapshots = [np.load(os.path.join("../FEM/fem_training_data", file)) for file in snapshot_files]
    all_snapshots = np.hstack(all_snapshots)

    U, s, _ = np.linalg.svd(all_snapshots, full_matrices=False)

    eps2_tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]  

    plot_singular_values(s, all_snapshots, eps2_tols)
    plot_modes(U, num_modes=6)
    save_modes(U, s, eps2_tols, all_snapshots)

