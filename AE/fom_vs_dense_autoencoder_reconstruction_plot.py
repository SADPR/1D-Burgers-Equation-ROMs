import numpy as np
import matplotlib.pyplot as plt
import os

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def compute_l2_norm_error(U_FOM, U_ROM):
    """Compute the L2 norm error between the FOM and ROM over the entire matrix."""
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)

def plot_reconstruction_comparison():
    # Load domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.07
    times_of_interest = [7, 14, 21]  # seconds
    time_indices = [int(t / At) for t in times_of_interest]

    # Load the FOM snapshot data
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Load the Dense Autoencoder reconstructed data
    reconstructed_snapshots_ae = np.load("reconstructed_snapshots_latent_16.npy")

    # Calculate and print L2 norm errors over the entire matrices
    l2_error_ae = compute_l2_norm_error(snapshot, reconstructed_snapshots_ae)
    print(f"Total Dense Autoencoder L2 Error = {l2_error_ae:.6f}")

    # Plotting
    plt.figure(figsize=(7, 6))

    # Plot original snapshots (FOM) for the selected times
    plt.plot(X, snapshot[:, time_indices[0]], color='black', linestyle='-', linewidth=2, label='FOM')
    for t_index in time_indices[1:]:
        plt.plot(X, snapshot[:, t_index], color='black', linestyle='-', linewidth=2)

    # Plot Dense Autoencoder reconstructed snapshots
    plt.plot(X, reconstructed_snapshots_ae[:, time_indices[0]], color='blue', linestyle='--', linewidth=2, label='Dense Autoencoder')
    for t_index in time_indices[1:]:
        plt.plot(X, reconstructed_snapshots_ae[:, t_index], color='blue', linestyle='--', linewidth=2)

    # Annotate the plot with the reconstruction methods
    plt.title(f'FOM vs Dense Autoencoder Reconstruction at Different Times')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.ylim(0, 7)
    plt.xlim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(True)

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the figure as a PDF
    plt.savefig("fom_dense_autoencoder_comparison.pdf", format='pdf')

    plt.show()

# Call the plotting function
plot_reconstruction_comparison()

