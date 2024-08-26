import numpy as np
import matplotlib.pyplot as plt
import os

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def compute_l2_norm_error(U_FOM, U_ROM):
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)

def plot_fom_rom_comparison_local(num_clusters):
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
    fom_filename = "../../POD/RESULTS_TEMPORARY/rom_solutions/simulation_mu1_4.76_mu2_0.0182.npy"
    U_FOM = np.load(fom_filename)

    # Load ROM results for the given number of clusters and tolerance
    rom_filename_lspg = f"local_PROM_{num_clusters}_clusters_lspg.npy"
    rom_filename_galerkin = f"local_PROM_{num_clusters}_clusters_galerkin.npy"
    U_ROM_lspg = np.load(os.path.join(save_dir, rom_filename_lspg))
    U_ROM_galerkin = np.load(os.path.join(save_dir, rom_filename_galerkin))

    # Plotting
    plt.figure(figsize=(7, 6))

    # Plot FOM results (with label only once)
    plt.plot(X, U_FOM[:, time_indices[0]], color='black', linestyle='-', linewidth=2, label='FOM')
    for t_index in time_indices[1:]:
        plt.plot(X, U_FOM[:, t_index], color='black', linestyle='-', linewidth=2)

    # Plot ROM results (LSPG) (with label only once)
    plt.plot(X, U_ROM_lspg[:, time_indices[0]], color='blue', linestyle='-', linewidth=2, label=f'LSPG ROM')
    for t_index in time_indices[1:]:
        plt.plot(X, U_ROM_lspg[:, t_index], color='blue', linestyle='-', linewidth=2)

    # Plot ROM results (Galerkin) (with label only once)
    plt.plot(X, U_ROM_galerkin[:, time_indices[0]], color='red', linestyle='-', linewidth=2, label=f'Galerkin ROM')
    for t_index in time_indices[1:]:
        plt.plot(X, U_ROM_galerkin[:, t_index], color='red', linestyle='-', linewidth=2)

    # Calculate and print L2 norm errors
    l2_error_galerkin = compute_l2_norm_error(U_FOM, U_ROM_galerkin)
    l2_error_lspg = compute_l2_norm_error(U_FOM, U_ROM_lspg)

    print(f"Number of clusters {num_clusters}, Galerkin L2 Error = {l2_error_galerkin:.6f}, LSPG L2 Error = {l2_error_lspg:.6f}")

    # Annotate the plot with the number of clusters and tolerance
    plt.title(f'FOM vs Local ROM (LSPG \& Galerkin) at Different Times\nLocal POD-Based ROM ({num_clusters} Clusters, Tolerance = 1e-4)')
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
    plt.savefig(f"fom_local_pod_comparison_{num_clusters}_clusters.pdf", format='pdf')

    plt.show()

# Example usage for 20 clusters and a tolerance of 1e-4
plot_fom_rom_comparison_local(num_clusters=20)
