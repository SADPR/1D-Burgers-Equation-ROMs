import numpy as np
import matplotlib.pyplot as plt
import os

# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def compute_relative_errors(U_FOM, U_ROM):
    errors = []
    for i in range(U_FOM.shape[1]):
        rel_error = np.linalg.norm(U_FOM[:, i] - U_ROM[:, i]) / np.linalg.norm(U_FOM[:, i])
        errors.append(rel_error)
    return np.array(errors)

def get_num_modes(tol, modes_dir):
    modes_file = os.path.join(modes_dir, f"U_modes_tol_{tol:.0e}.npy")
    return np.load(modes_file).shape[1]

def plot_errors_per_timestep(tolerances, fom_path, rom_dir, modes_dir):
    U_FOM = np.load(fom_path)
    time_steps = np.arange(U_FOM.shape[1])
    colours = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']

    plt.figure(figsize=(9, 6))

    for i, tol in enumerate(tolerances):
        num_modes = get_num_modes(tol, modes_dir)

        # Load ROMs
        rom_lspg = np.load(os.path.join(rom_dir, f"U_PROM_tol_{tol:.0e}_lspg.npy"))
        rom_galerkin = np.load(os.path.join(rom_dir, f"U_PROM_tol_{tol:.0e}_galerkin.npy"))

        # Compute per-timestep errors
        rel_err_lspg = compute_relative_errors(U_FOM, rom_lspg)
        rel_err_galerkin = compute_relative_errors(U_FOM, rom_galerkin)

        # Plot
        plt.plot(time_steps, rel_err_lspg, linestyle='-', color=colours[i % len(colours)],
                 label=rf'LSPG ($\epsilon^2={tol:.0e}$, $n={num_modes}$)')
        plt.plot(time_steps, rel_err_galerkin, linestyle='--', color=colours[i % len(colours)],
                 label=rf'Galerkin ($\epsilon^2={tol:.0e}$, $n={num_modes}$)')

        # Print overall L2 norm relative errors
        global_err_lspg = 100*np.linalg.norm(U_FOM - rom_lspg) / np.linalg.norm(U_FOM)
        global_err_galerkin = 100*np.linalg.norm(U_FOM - rom_galerkin) / np.linalg.norm(U_FOM)
        print(f"[{tol:.0e}] Overall relative error:")
        print(f"   Galerkin: {global_err_galerkin:.4e}")
        print(f"   LSPG    : {global_err_lspg:.4e}")

    plt.xlabel('Time step index', fontsize=14)
    plt.ylabel(r'Relative $\ell^2$ Error', fontsize=14)
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rom_relative_errors_over_time.pdf", format='pdf')
    plt.close()

if __name__ == '__main__':
    tolerances = [1e-3, 1e-4, 1e-5]
    fom_path = "../../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy"
    rom_dir = "rom_solutions"
    modes_dir = "../modes"

    plot_errors_per_timestep(tolerances, fom_path, rom_dir, modes_dir)


