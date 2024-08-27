import numpy as np
import os
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from getH_alpha_functions import getQ, getE, getH, get_single_Q

# Function to reconstruct a snapshot using the quadratic approximation
def reconstruct_snapshot_quadratic(U, H, q, modes):
    Q_single = get_single_Q(modes, q)
    return U @ q + H @ Q_single

# Function to reconstruct a snapshot using POD
def reconstruct_snapshot_pod(U, q):
    return U @ q

# Main function
if __name__ == "__main__":
    # Load the precomputed basis and H matrix
    Phi_p = np.load("U_truncated.npy")
    H = np.load("H_quadratic.npy")

    # Load a specific snapshot for reconstruction
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)
    Ns = snapshot.shape[1]
    num_modes = Phi_p.shape[1]

    # Initialize arrays to hold the relative errors
    relative_errors_pod = np.zeros(Ns)
    relative_errors_quadratic = np.zeros(Ns)

    # Initialize arrays to store the reconstructed snapshots
    reconstructed_snapshots_pod = np.zeros_like(snapshot)
    reconstructed_snapshots_quadratic = np.zeros_like(snapshot)

    # Calculate relative errors for each snapshot
    for i in range(Ns):
        original_snapshot = snapshot[:, i]

        # Project the snapshot onto the reduced basis
        q_snapshot = Phi_p.T @ original_snapshot

        # Reconstruct using POD
        reconstructed_pod = reconstruct_snapshot_pod(Phi_p, q_snapshot)
        reconstructed_snapshots_pod[:, i] = reconstructed_pod

        # Reconstruct using Quadratic Approximation
        reconstructed_quadratic = reconstruct_snapshot_quadratic(Phi_p, H, q_snapshot, num_modes)
        reconstructed_snapshots_quadratic[:, i] = reconstructed_quadratic

        # Calculate relative errors
        relative_errors_pod[i] = np.linalg.norm(original_snapshot - reconstructed_pod) / np.linalg.norm(original_snapshot)
        relative_errors_quadratic[i] = np.linalg.norm(original_snapshot - reconstructed_quadratic) / np.linalg.norm(original_snapshot)

    # Save the reconstructed snapshots and relative errors
    np.save("reconstructed_snapshots_pod.npy", reconstructed_snapshots_pod)
    np.save("reconstructed_snapshots_quadratic.npy", reconstructed_snapshots_quadratic)
    np.save("relative_errors_pod.npy", relative_errors_pod)
    np.save("relative_errors_quadratic.npy", relative_errors_quadratic)

    # Plot the relative errors and save the figure
    plt.figure(figsize=(10, 6))
    plt.plot(range(Ns), relative_errors_pod, label='POD Relative Error', marker='o')
    plt.plot(range(Ns), relative_errors_quadratic, label='Quadratic Approximation Relative Error', marker='x')
    plt.xlabel('Snapshot Index')
    plt.ylabel('Relative Error')
    plt.title('Comparison of Relative Reconstruction Errors')
    plt.legend()
    plt.grid(True)
    plt.savefig("relative_error_comparison.png")
    plt.show()

    # Create a combined GIF to visualize the comparison
    X = np.linspace(0, 100, snapshot.shape[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    line_original, = ax.plot(X, snapshot[:, 0], 'b-', label='Original Snapshot')
    line_pod, = ax.plot(X, reconstructed_snapshots_pod[:, 0], 'g--', label='POD Reconstructed')
    line_quadratic, = ax.plot(X, reconstructed_snapshots_quadratic[:, 0], 'r-.', label='Quadratic Reconstructed')
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(np.min(snapshot), np.max(snapshot))
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_original.set_ydata(snapshot[:, frame])
        line_pod.set_ydata(reconstructed_snapshots_pod[:, frame])
        line_quadratic.set_ydata(reconstructed_snapshots_quadratic[:, frame])
        ax.set_title(f'Snapshot Comparison at t = {frame * 0.07:.2f}')
        return line_original, line_pod, line_quadratic

    ani = FuncAnimation(fig, update, frames=Ns, blit=True)

    # Save animation as GIF
    ani.save("comparison_pod_vs_quadratic.gif", writer=PillowWriter(fps=10))

    plt.show()



