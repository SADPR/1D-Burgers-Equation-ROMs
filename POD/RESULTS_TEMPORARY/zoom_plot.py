import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def my_mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)
    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)

    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2

def get_num_modes(tol):
    modes_file = f"../modes/U_modes_tol_{tol:.0e}.npy"
    U_modes = np.load(modes_file)
    return U_modes.shape[1]

def plot_fom_rom_comparison(tol, mu1=4.75, mu2=0.0200):
    # Parameters
    a, b = 0, 100
    m = 511
    X = np.linspace(a, b, m + 1)
    At = 0.05
    time_idx = int(5 / At)  # time = 5s
    tag_mu1 = f"{mu1:.3f}"
    tag_mu2 = f"{mu2:.4f}"

    # File paths
    save_dir = "rom_solutions"
    fom_filename = f"../../FEM/fem_testing_data/fem_simulation_mu1_{tag_mu1}_mu2_{tag_mu2}.npy"
    rom_filename_lspg = os.path.join(save_dir, f"U_PROM_tol_{tol:.0e}_mu1_{tag_mu1}_mu2_{tag_mu2}_lspg.npy")
    rom_filename_galerkin = os.path.join(save_dir, f"U_PROM_tol_{tol:.0e}_mu1_{tag_mu1}_mu2_{tag_mu2}_galerkin.npy")

    # Load solutions
    U_FOM = np.load(fom_filename)
    U_ROM_lspg = np.load(rom_filename_lspg)
    U_ROM_galerkin = np.load(rom_filename_galerkin)
    num_modes = get_num_modes(tol)

    # Main plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(X, U_FOM[:, time_idx], 'k-', linewidth=3, label='FOM')
    ax.plot(X, U_ROM_lspg[:, time_idx], 'b-', linewidth=2, label='LSPG PROM', alpha=0.8)
    ax.plot(X, U_ROM_galerkin[:, time_idx], 'r-', linewidth=2, label='Galerkin PROM', alpha=0.8)

    ax.set_title(rf'$\mu_1 = {mu1:.3f}$, $\mu_2 = {mu2:.4f}$, $t = 5$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x,t)$')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 7)
    ax.legend(loc='upper right')
    ax.grid(True)

    # Inset zoom plot
    axins = ax.inset_axes([0.3, 0.4, 0.6, 0.4])
    axins.plot(X, U_FOM[:, time_idx], 'k-', linewidth=3)
    axins.plot(X, U_ROM_lspg[:, time_idx], 'b-', linewidth=2, alpha=0.8)
    axins.plot(X, U_ROM_galerkin[:, time_idx], 'r-', linewidth=2, alpha=0.8)

    axins.set_xlim(0, 15)
    axins.set_ylim(4.575, 5)
    axins.grid(True, linestyle='--', linewidth=0.4)

    my_mark_inset(ax, axins, loc1a=2, loc1b=1, loc2a=3, loc2b=4, fc="none", ec="black")

    # Save figure
    fig_name = f"fom_pod_comparison_tol_{tol:.0e}_mu1_{tag_mu1}_mu2_{tag_mu2}_modes_{num_modes}.pdf"
    plt.tight_layout()
    plt.savefig(fig_name, format='pdf')
    plt.show()

# Example usage
plot_fom_rom_comparison(tol=1e-4)


