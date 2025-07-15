import numpy as np
import matplotlib.pyplot as plt
import os

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def compute_l2_norm_error(U_FOM, U_ROM):
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)

def plot_fom_rom_comparison():
    # Load domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.07
    times_of_interest = [7, 14, 21]  # seconds
    time_indices = [int(t / At) for t in times_of_interest]

    # Directory where ROM results are saved
    save_dir = "."

    # Load FOM results
    fom_filename = "../../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy"
    U_FOM = np.load(fom_filename)

    # Load ROM results for Local POD with 20 clusters (LSPG)
    rom_filename_local_lspg = f"local_PROM_20_clusters_lspg.npy"
    U_ROM_local_lspg = np.load(os.path.join(save_dir, rom_filename_local_lspg))

    # Load ROM results for Global POD with 28 modes (LSPG)
    rom_filename_global_lspg = f"../../POD/Results_thesis/rom_solutions/U_PROM_tol_4e-03_mu1_4.750_mu2_0.0200_lspg.npy"
    U_ROM_global_lspg = np.load(os.path.join(save_dir, rom_filename_global_lspg))

    # Plotting
    plt.figure(figsize=(7, 6))

    # Plot FOM results (with label only once)
    plt.plot(X, U_FOM[:, time_indices[0]], color='black', linestyle='-', linewidth=2, label='FOM')
    for t_index in time_indices[1:]:
        plt.plot(X, U_FOM[:, t_index], color='black', linestyle='-', linewidth=2)

    # Plot Local POD results (LSPG) (with label only once)
    plt.plot(X, U_ROM_local_lspg[:, time_indices[0]], color='blue', linestyle='-', linewidth=2, label='Local POD LSPG (20 Clusters, Avg 17.5 Modes)')
    for t_index in time_indices[1:]:
        plt.plot(X, U_ROM_local_lspg[:, t_index], color='blue', linestyle='-', linewidth=2)

    # Plot Global POD results (LSPG) (with label only once)
    plt.plot(X, U_ROM_global_lspg[:, time_indices[0]], color='red', linestyle='-', linewidth=2, label='Global POD LSPG (17 Modes)')
    for t_index in time_indices[1:]:
        plt.plot(X, U_ROM_global_lspg[:, t_index], color='red', linestyle='-', linewidth=2)

    # Calculate and print L2 norm errors
    l2_error_local_lspg = compute_l2_norm_error(U_FOM, U_ROM_local_lspg)
    l2_error_global_lspg = compute_l2_norm_error(U_FOM, U_ROM_global_lspg)

    print(f"Local POD (20 clusters, Avg 17.5 Modes), LSPG L2 Error = {l2_error_local_lspg:.6f}")
    print(f"Global POD (17 modes), LSPG L2 Error = {l2_error_global_lspg:.6f}")

    # Annotate the plot
    plt.title(f'FOM vs ROM (LSPG) Comparison\nLocal POD (20 Clusters, Avg 17.5 Modes) vs Global POD (17 Modes)')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.ylim(0, 7)
    plt.xlim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(True)

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the figure as a PDF with a POD-related name
    plt.savefig(f"fom_local_vs_global_pod_comparison_17_modes.pdf", format='pdf')

    plt.show()

# Run the comparison plot
plot_fom_rom_comparison()
