import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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

def plot_fom_rom_comparison_gif(tol, fps=30):
    # Load domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.07
    total_time_steps = 500  # Adjust according to your total time steps

    # Directory where ROM results are saved
    save_dir = "rom_solutions"
    
    # Load FOM results
    fom_filename = "rom_solutions/simulation_mu1_4.76_mu2_0.0182.npy"
    U_FOM = np.load(fom_filename)

    # Load ROM results for the given tolerance
    rom_filename_lspg = f"U_PROM_tol_{tol:.0e}_lspg.npy"
    rom_filename_galerkin = f"U_PROM_tol_{tol:.0e}_galerkin.npy"
    U_ROM_lspg = np.load(os.path.join(save_dir, rom_filename_lspg))
    U_ROM_galerkin = np.load(os.path.join(save_dir, rom_filename_galerkin))

    # Get the number of modes
    num_modes = get_num_modes(tol)

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(7, 6))

    def update(t_index):
        ax.clear()

        # Plot FOM results
        ax.plot(X, U_FOM[:, t_index], color='black', linestyle='-', linewidth=2, label='FOM')

        # Plot ROM results (LSPG)
        ax.plot(X, U_ROM_lspg[:, t_index], color='blue', linestyle='-', linewidth=2, label='LSPG ROM')

        # Plot ROM results (Galerkin)
        ax.plot(X, U_ROM_galerkin[:, t_index], color='red', linestyle='-', linewidth=2, label='Galerkin ROM')

        # Annotate the plot with the number of modes and time step
        # Correct the title by escaping the & character
        ax.set_title(f'FOM vs ROM (LSPG \\& Galerkin)\nTolerance = {tol:.0e}, Modes = {num_modes}, Time Step = {t_index}')

        # ax.set_title(f'FOM vs ROM (LSPG & Galerkin)\nTolerance = {tol:.0e}, Modes = {num_modes}, Time Step = {t_index}')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
        ax.set_ylim(0, 7)
        ax.set_xlim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(True)

    # Create the animation
    anim = FuncAnimation(fig, update, frames=total_time_steps, interval=1000 / fps)

    # Save the animation as a GIF using PillowWriter
    gif_filename = f"fom_pod_comparison_tol_{tol:.0e}_modes_{num_modes}.gif"
    anim.save(gif_filename, writer=PillowWriter(fps=fps))

    plt.close()

# Example usage for different tolerances:
tolerances = [1e-1, 5e-2, 2e-2, 1e-2, 1e-3, 1e-4]
for tol in tolerances:
    plot_fom_rom_comparison_gif(tol, fps=30)
