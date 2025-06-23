import numpy as np
import matplotlib.pyplot as plt
import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_num_modes(tol):
    modes_file = f"../modes/U_modes_tol_{tol:.0e}.npy"
    U_modes = np.load(modes_file)
    return U_modes.shape[1]

def compute_rel_error(U_FOM, U_ROM):
    """Compute relative error according to Eq.~(X): summed squared 2-norm over time steps."""
    num_steps = U_FOM.shape[1]
    num = 0.0
    denom = 0.0
    for m in range(num_steps):
        diff = U_FOM[:, m] - U_ROM[:, m]
        num += np.linalg.norm(diff, ord=2) ** 2
        denom += np.linalg.norm(U_FOM[:, m], ord=2) ** 2
    return np.sqrt(num / denom)

def plot_fom_rom_comparison(tol, mu1, mu2):
    # Domain and mesh
    a, b = 0, 100
    m = 511
    X = np.linspace(a, b, m + 1)

    # Time points to plot
    At = 0.05
    times_of_interest = [5, 10, 15, 20, 25]
    time_indices = [int(t / At) for t in times_of_interest]

    # Filenames and tags
    tag_mu1 = f"{mu1:.3f}"
    tag_mu2 = f"{mu2:.4f}"
    save_dir = "rom_solutions"
    fom_filename = f"../../FEM/fem_testing_data/fem_simulation_mu1_{tag_mu1}_mu2_{tag_mu2}.npy"
    rom_filename_lspg = f"U_PROM_tol_{tol:.0e}_mu1_{tag_mu1}_mu2_{tag_mu2}_lspg.npy"
    rom_filename_galerkin = f"U_PROM_tol_{tol:.0e}_mu1_{tag_mu1}_mu2_{tag_mu2}_galerkin.npy"

    # Load solutions
    U_FOM = np.load(fom_filename)
    U_ROM_lspg = np.load(os.path.join(save_dir, rom_filename_lspg))
    U_ROM_galerkin = np.load(os.path.join(save_dir, rom_filename_galerkin))

    # Mode count
    num_modes = get_num_modes(tol)

    # Plot
    plt.figure(figsize=(7, 6))
    for idx, t_index in enumerate(time_indices):
        label_fom = 'FOM' if idx == 0 else None
        label_lspg = 'LSPG PROM' if idx == 0 else None
        label_galerkin = 'Galerkin PROM' if idx == 0 else None

        plt.plot(X, U_FOM[:, t_index], 'k-', linewidth=3, label=label_fom)
        plt.plot(X, U_ROM_lspg[:, t_index], 'b--', linewidth=2, label=label_lspg, alpha=0.8)
        plt.plot(X, U_ROM_galerkin[:, t_index], 'r-.', linewidth=2, label=label_galerkin, alpha=0.8)

    # Print relative errors
    err_gal = compute_rel_error(U_FOM, U_ROM_galerkin)
    err_lspg = compute_rel_error(U_FOM, U_ROM_lspg)
    print(f"[mu1={mu1:.3f}, mu2={mu2:.4f}, tol={tol:.0e}] Galerkin Rel Error = {err_gal:.6f}, LSPG Rel Error = {err_lspg:.6f}")

    # Styling
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u(x,t)$')
    plt.title(rf'$\mu_1 = {mu1:.3f}$, $\mu_2 = {mu2:.4f}$')
    plt.ylim(0, 7)
    plt.xlim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save
    fig_name = f"fom_pod_mu1_{tag_mu1}_mu2_{tag_mu2}_tol_{tol:.0e}_modes_{num_modes}.pdf"
    plt.savefig(fig_name, format='pdf')
    plt.close()

# Main loop
test_points = [
    (4.75, 0.0200),
    (4.56, 0.0190),
    (5.19, 0.0260)
]
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

max_errors = []

for tol in tolerances:
    rel_errors_galerkin = []
    rel_errors_lspg = []

    for mu1, mu2 in test_points:
        tag_mu1 = f"{mu1:.3f}"
        tag_mu2 = f"{mu2:.4f}"
        fom_filename = f"../../FEM/fem_testing_data/fem_simulation_mu1_{tag_mu1}_mu2_{tag_mu2}.npy"
        rom_gal_filename = f"rom_solutions/U_PROM_tol_{tol:.0e}_mu1_{tag_mu1}_mu2_{tag_mu2}_galerkin.npy"
        rom_lspg_filename = f"rom_solutions/U_PROM_tol_{tol:.0e}_mu1_{tag_mu1}_mu2_{tag_mu2}_lspg.npy"

        try:
            U_FOM = np.load(fom_filename)
            U_GAL = np.load(rom_gal_filename)
            U_LSPG = np.load(rom_lspg_filename)

            err_gal = compute_rel_error(U_FOM, U_GAL)
            err_lspg = compute_rel_error(U_FOM, U_LSPG)

            rel_errors_galerkin.append(err_gal * 100)
            rel_errors_lspg.append(err_lspg * 100)

            plot_fom_rom_comparison(tol, mu1, mu2)

        except FileNotFoundError as e:
            print(f"Missing file for mu=({mu1}, {mu2}), tol={tol:.0e}: {e}")
            rel_errors_galerkin.append(np.nan)
            rel_errors_lspg.append(np.nan)

    max_gal = np.nanmax(rel_errors_galerkin)
    max_lspg = np.nanmax(rel_errors_lspg)
    max_errors.append((f"{tol:.0e}", max_gal, max_lspg))

# Summary table
print("\n=== Maximum Relative Errors Across Test Points ===")
print(f"{'Tolerance':<10} {'Galerkin Max (%)':<20} {'LSPG Max (%)':<20}")
for tol_str, max_gal, max_lspg in max_errors:
    print(f"{tol_str:<10} {max_gal:<20.6f} {max_lspg:<20.6f}")











