import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Enable LaTeX text rendering (optional)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def reconstruct_snapshot_quadratic(U, H, q, modes):
    Q_single = get_single_Q(modes, q)
    return U @ q + H @ Q_single

def reconstruct_snapshot_pod(U, q):
    return U @ q

def plot_comparison_gif(fps=30):
    # Load domain
    a = 0
    b = 100
    m = int(256 * 2)  # Adjust grid size
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.07
    total_time_steps = 500  # Adjust based on your time steps

    # Load reconstructed data and FOM snapshots
    Phi_p = np.load("U_truncated.npy")
    H = np.load("H_quadratic.npy")
    snapshot_file = '../FEM/training_data/simulation_mu1_4.61_mu2_0.0200.npy'
    snapshot = np.load(snapshot_file)

    reconstructed_snapshots_pod = np.load("reconstructed_snapshots_pod.npy")
    reconstructed_snapshots_quadratic = np.load("reconstructed_snapshots_quadratic.npy")

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot setup
    def update(t_index):
        ax.clear()

        # Plot FOM Snapshot (Black)
        ax.plot(X, snapshot[:, t_index], color='black', linestyle='-', linewidth=2, label='FOM')

        # Plot POD Reconstructed (Red)
        ax.plot(X, reconstructed_snapshots_pod[:, t_index], color='red', linestyle='--', linewidth=2, label='POD Reconstructed')

        # Plot Quadratic Reconstructed (Blue)
        ax.plot(X, reconstructed_snapshots_quadratic[:, t_index], color='blue', linestyle='-.', linewidth=2, label='Quadratic Reconstructed')

        # Title and labels with LaTeX rendering if needed
        ax.set_title(f'FOM vs POD \\& Quadratic Reconstructed\nTime Step = {t_index}, t = {t_index * At:.2f}s')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
        ax.set_ylim(np.min(snapshot), np.max(snapshot))
        ax.set_xlim(a, b)
        ax.legend(loc='upper right')
        ax.grid(True)

    # Create the animation
    anim = FuncAnimation(fig, update, frames=total_time_steps, interval=1000 / fps)

    # Save the animation as a GIF using PillowWriter
    gif_filename = f"comparison_pod_vs_quadratic.gif"
    anim.save(gif_filename, writer=PillowWriter(fps=fps))

    plt.close()

# Example usage:
plot_comparison_gif(fps=30)





