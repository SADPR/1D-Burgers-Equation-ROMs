import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Function to reconstruct snapshot using POD modes
def reconstruct_snapshot_with_pod(snapshot, U):
    return U @ (U.T @ snapshot)

# Function to create gif with all snapshots overlaid
def create_gif(X, original_snapshot, pod_reconstructed, nTimeSteps, At, tol, num_modes, latent_dim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 8)
    
    line_original, = ax.plot(X, original_snapshot[:, 0], 'b-', label='Original Snapshot')
    line_pod, = ax.plot(X, pod_reconstructed[:, 0], 'k-.', label=f'POD Reconstructed (tol={tol}, modes={num_modes})')
    ax.set_title('Snapshot Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_original.set_ydata(original_snapshot[:, frame])
        line_pod.set_ydata(pod_reconstructed[:, frame])
        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')
        return line_original, line_pod

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    ani.save("pod_reconstruction.gif", writer=PillowWriter(fps=10))

    plt.show()

if __name__ == '__main__':

    # Load a random snapshot from the training_data directory
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Load modes for POD with a specific tolerance
    tol = 5e-02
    U = np.load(f"modes/U_modes_tol_{tol:.0e}.npy")

    # Reconstruct the snapshot using POD
    pod_reconstructed = reconstruct_snapshot_with_pod(snapshot, U)

    np.save("pod_reconstruction.npy", pod_reconstructed)

    # Domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization and numerical diffusion
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)

    # Create the combined GIF
    create_gif(X, snapshot, pod_reconstructed, nTimeSteps, At, tol, U.shape[1], 16)





