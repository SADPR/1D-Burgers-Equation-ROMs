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

def compute_l2_norm_error(U_FOM, U_ROM):
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)

def plot_fom_rom_comparison(tol):
    a, b = 0, 100
    m = 511
    X = np.linspace(a, b, m + 1)
    At = 0.05
    time_idx = int(5 / At)  # t = 5

    save_dir = "rom_solutions"
    fom_filename = "../../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy"
    U_FOM = np.load(fom_filename)
    U_ROM_lspg = np.load(os.path.join(save_dir, f"U_PROM_tol_{tol:.0e}_lspg.npy"))
    U_ROM_galerkin = np.load(os.path.join(save_dir, f"U_PROM_tol_{tol:.0e}_galerkin.npy"))
    num_modes = get_num_modes(tol)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(X, U_FOM[:, time_idx], color='black', linestyle='-', linewidth=2, label='FOM')
    ax.plot(X, U_ROM_lspg[:, time_idx], color='blue', linestyle='-', linewidth=2, label='LSPG PROM')
    ax.plot(X, U_ROM_galerkin[:, time_idx], color='red', linestyle='-', linewidth=2, label='Galerkin PROM')

    ax.set_title(f'FOM vs PROM at $t=5$ s\nPOD-Based PROM ($\\epsilon^2$={tol:.0e}, $n$={num_modes})')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 7)
    ax.legend(loc='upper right')
    ax.grid(True)

    # Inset + custom connector
    axins = ax.inset_axes([0.3, 0.4, 0.6, 0.4])
    axins.plot(X, U_FOM[:, time_idx], color='black', linewidth=2)
    axins.plot(X, U_ROM_lspg[:, time_idx], color='blue', linewidth=2)
    axins.plot(X, U_ROM_galerkin[:, time_idx], color='red', linewidth=2)

    axins.set_xlim(0, 15)
    axins.set_ylim(4.575, 5)
    axins.grid(True, linestyle='--', linewidth=0.4)

    # Use your custom mark_inset
    my_mark_inset(ax, axins, loc1a=2, loc1b=1, loc2a=3, loc2b=4, fc="none", ec="black")

    plt.tight_layout()
    plt.savefig(f"fom_pod_comparison_tol_{tol:.0e}_modes_{num_modes}.pdf", format='pdf')
    plt.show()

# Example usage
plot_fom_rom_comparison(1e-4)

