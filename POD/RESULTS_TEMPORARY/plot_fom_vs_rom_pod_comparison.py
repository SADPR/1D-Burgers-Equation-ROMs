import numpy as np
import matplotlib.pyplot as plt
import os

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_num_modes(tol):
    # Load the modes file corresponding to the tolerance
    modes_file = f"../modes/U_modes_tol_{tol:.0e}.npy"
    U_modes = np.load(modes_file)
    return U_modes.shape[1]  # Number of columns corresponds to the number of modes

def compute_l2_norm_error(U_FOM, U_ROM):
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)

def plot_fom_rom_comparison(tol):
    # Domain and mesh
    a, b = 0, 100

    # Mesh
    m = 511
    h = (b - a) / m
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.05
    times_of_interest = [5]  # seconds
    time_indices = [int(t / At) for t in times_of_interest]

    # Directory where ROM results are saved
    save_dir = "rom_solutions"

    # Load FOM results
    fom_filename = "../../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy"
    U_FOM = np.load(fom_filename)

    # Load ROM results for the given tolerance
    rom_filename_lspg = f"U_PROM_tol_{tol:.0e}_lspg.npy"
    rom_filename_galerkin = f"U_PROM_tol_{tol:.0e}_galerkin.npy"
    U_ROM_lspg = np.load(os.path.join(save_dir, rom_filename_lspg))
    U_ROM_galerkin = np.load(os.path.join(save_dir, rom_filename_galerkin))

    # Get the number of modes
    num_modes = get_num_modes(tol)

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

    print(f"Tolerance {tol:.0e}: Galerkin L2 Error = {l2_error_galerkin:.6f}, LSPG L2 Error = {l2_error_lspg:.6f}")

    # Annotate the plot with the number of modes
    plt.title(f'FOM vs ROM (LSPG \& Galerkin) at Different Times\nPOD-Based ROM (Tolerance = {tol:.0e}, Modes = {num_modes})')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.ylim(0, 7)
    plt.xlim(0,100)
    plt.legend(loc='upper right')
    plt.grid(True)

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the figure as a PDF with a POD-related name
    plt.savefig(f"fom_pod_comparison_tol_{tol:.0e}_modes_{num_modes}.pdf", format='pdf')

    plt.show()

# Example usage for different tolerances:
tolerances = [1e-5]#[1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for tol in tolerances:
    plot_fom_rom_comparison(tol)








