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
    fom_filename = "simulation_mu1_4.76_mu2_0.0182.npy"
    U_FOM = np.load(fom_filename)

    # Load ROM results for Local POD with 28 modes (LSPG)
    rom_filename_local_lspg = "../../Local_POD/RESULTS_TEMPORARY/local_PROM_20_clusters_lspg.npy"
    U_ROM_local_lspg = np.load(os.path.join(save_dir, rom_filename_local_lspg))

    # Load ROM results for Global POD with 28 modes (LSPG)
    rom_filename_global_lspg = "../../Local_POD/RESULTS_TEMPORARY/PROM_solution_28_modes.npy"
    U_ROM_global_lspg = np.load(os.path.join(save_dir, rom_filename_global_lspg))

    # Load ROM results for POD-ANN
    rom_filename_pod_ann = "U_POD_ANN_PROM_solution.npy"
    U_POD_ANN_PROM = np.load(os.path.join(save_dir, rom_filename_pod_ann))[:,:-1]

    # Plotting
    plt.figure(figsize=(7, 6))

    # Plot FOM results (with label only once)
    plt.plot(X, U_FOM[:, time_indices[0]], color='black', linestyle='-', linewidth=3, label='FOM')
    for t_index in time_indices[1:]:
        plt.plot(X, U_FOM[:, t_index], color='black', linestyle='-', linewidth=3)

    # Plot Global POD results (LSPG) (with label only once)
    # plt.plot(X, U_ROM_global_lspg[:, time_indices[0]], color='blue', linestyle='-', linewidth=3, label='Global POD LSPG (28 Modes)')
    # for t_index in time_indices[1:]:
    #     plt.plot(X, U_ROM_global_lspg[:, t_index], color='blue', linestyle='-', linewidth=3)

    # Plot Local POD results (LSPG) (with label only once)
    plt.plot(X, U_ROM_local_lspg[:, time_indices[0]], color='green', linestyle='-', linewidth=3, label='Local POD LSPG (28 Modes)')
    for t_index in time_indices[1:]:
        plt.plot(X, U_ROM_local_lspg[:, t_index], color='green', linestyle='-', linewidth=3)

    # Plot POD-ANN results (with label only once)
    plt.plot(X, U_POD_ANN_PROM[:, time_indices[0]], color='red', linestyle='--', linewidth=3, label='POD-ANN (28 primary modes, 301 secondary modes)')
    for t_index in time_indices[1:]:
        plt.plot(X, U_POD_ANN_PROM[:, t_index], color='red', linestyle='--', linewidth=3)

    # Calculate and print L2 norm errors
    l2_error_local_lspg = compute_l2_norm_error(U_FOM, U_ROM_local_lspg)
    l2_error_global_lspg = compute_l2_norm_error(U_FOM, U_ROM_global_lspg)
    l2_error_pod_ann = compute_l2_norm_error(U_FOM, U_POD_ANN_PROM)

    print(f"Local POD (20 clusters, Avg 27.5 Modes), LSPG L2 Error = {l2_error_local_lspg:.6f}")
    print(f"Global POD (28 modes), LSPG L2 Error = {l2_error_global_lspg:.6f}")
    print(f"POD-ANN (28 primary modes, 301 secondary modes) L2 Error = {l2_error_pod_ann:.6f}")

    # Annotate the plot
    plt.title(f'FOM vs ROM (LSPG) Comparison\nLocal POD (20 clusters, Avg 27.5 Modes) vs POD-ANN (28 primary modes, 301 secondary modes)')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.ylim(0, 7)
    plt.xlim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(True)

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the figure as a PDF
    plt.savefig(f"fom_local_pod_ann_comparison_28_modes.pdf", format='pdf')

    plt.show()

# Run the comparison plot
plot_fom_rom_comparison()
