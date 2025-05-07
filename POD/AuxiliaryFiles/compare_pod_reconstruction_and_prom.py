import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Function to create gif with all snapshots overlaid
def create_combined_gif(X, original_snapshot, pod_prom, pod_reconstructed, nTimeSteps, At, tol, num_modes):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 8)
    
    line_original, = ax.plot(X, original_snapshot[:, 0], 'b-', label='Original Snapshot')
    line_pod_reconstructed, = ax.plot(X, pod_reconstructed[:, 0], 'k-.', label=f'POD Reconstructed (tol={tol}, modes={num_modes})')
    line_pod_prom, = ax.plot(X, pod_prom[:, 0], 'r--', label=f'POD PROM (tol={tol}, modes={num_modes})')
    
    ax.set_title('Snapshot Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_original.set_ydata(original_snapshot[:, frame])
        line_pod_reconstructed.set_ydata(pod_reconstructed[:, frame])
        line_pod_prom.set_ydata(pod_prom[:, frame])
        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')
        return line_original, line_pod_reconstructed, line_pod_prom

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    ani.save("pod_vs_prom_reconstruction.gif", writer=PillowWriter(fps=10))

    plt.show()

if __name__ == '__main__':
    snapshot_prom_pod = np.load("pod_prom.npy")
    snapshot_reconstruction_pod = np.load("pod_reconstruction.npy")
    original_snapshot = np.load("../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy")

    # Domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Tolerance
    tol = 5e-02
    modes = 16

    # Time discretization and numerical diffusion
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)

    # Calculate and print relative errors
    rel_error_pod_reconstructed = np.linalg.norm(original_snapshot - snapshot_reconstruction_pod) / np.linalg.norm(original_snapshot)
    rel_error_pod_prom = np.linalg.norm(original_snapshot - snapshot_prom_pod) / np.linalg.norm(original_snapshot)

    print(f"Relative Error for POD Reconstructed: {rel_error_pod_reconstructed:.4f}")
    print(f"Relative Error for POD PROM: {rel_error_pod_prom:.4f}")

    # Create the combined GIF
    create_combined_gif(X, original_snapshot, snapshot_prom_pod, snapshot_reconstruction_pod, nTimeSteps, At, tol, modes)






