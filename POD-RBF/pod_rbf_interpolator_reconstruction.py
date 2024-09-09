import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import pickle

# Function to create gif with all snapshots overlaid
def create_combined_gif(X, original_snapshot, rbf_reconstructed, nTimeSteps, At, latent_dim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 8)

    line_original, = ax.plot(X, original_snapshot[:, 0], 'b-', label='Original Snapshot')
    line_rbf, = ax.plot(X, rbf_reconstructed[:, 0], 'g--', label=f'POD-RBF Reconstructed (inf modes={latent_dim}, sup modes={301})')

    ax.set_title('Snapshot Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_original.set_ydata(original_snapshot[:, frame])
        line_rbf.set_ydata(rbf_reconstructed[:, frame])
        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')
        return line_original, line_rbf

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    # ani.save("pod_rbf_reconstruction.gif", writer=PillowWriter(fps=10))

    plt.show()

# Function to reconstruct snapshot using POD-RBF
def reconstruct_snapshot_with_pod_rbf(snapshot_file, U_p, U_s, rbf_model, r):
    # Load the snapshot file
    snapshots = np.load(snapshot_file)

    # Project onto the POD basis to get q_p
    q = U_p.T @ snapshots
    q_p = q[:r, :]

    # Reconstruct the snapshots using the RBF model
    reconstructed_snapshots_rbf = []
    for i in range(q_p.shape[1]):
        q_p_sample = q_p[:, i].reshape(1, -1)  # Reshape to match input format for RBF
        q_s_pred = rbf_model(q_p_sample).T      # Use RBF model to predict secondary modes
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred.reshape(-1)
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)

    # Convert list to array and return
    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).squeeze().T
    return reconstructed_snapshots_rbf

if __name__ == '__main__':
    # Load the saved RBF model
    with open('rbf_model.pkl', 'rb') as f:
        rbf_model = pickle.load(f)

    # Load a random snapshot from the training_data directory
    # snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot_file = 'simulation_mu1_4.77_mu2_0.0200.npy'
    snapshot = np.load(snapshot_file)

    # Load U_p and U_s
    U_p = np.load('U_p.npy')
    U_s = np.load('U_s.npy')

    # Reconstruct the snapshot using POD-RBF
    pod_rbf_reconstructed = reconstruct_snapshot_with_pod_rbf(
        snapshot_file, U_p, U_s, rbf_model, 28
    )

    np.save("pod_rbf_reconstruction.npy", pod_rbf_reconstructed)

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
    create_combined_gif(X, snapshot, pod_rbf_reconstructed, nTimeSteps, At, 28)
