import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Define the DenseAutoencoder class (same as before)
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DenseAutoencoder, self).__init__()

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

# Function to create a GIF with all snapshots overlaid
def create_combined_gif(X, original_snapshot, autoencoder_reconstructed, nTimeSteps, At, latent_dim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 8)

    line_original, = ax.plot(X, original_snapshot[:, 0], 'b-', label='Original Snapshot')
    line_autoencoder, = ax.plot(X, autoencoder_reconstructed[:, 0], 'r--', label=f'Autoencoder Reconstructed (latent dim={latent_dim})')

    ax.set_title('Snapshot Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_original.set_ydata(original_snapshot[:, frame])
        line_autoencoder.set_ydata(autoencoder_reconstructed[:, frame])
        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')
        return line_original, line_autoencoder

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    ani.save(f"ae_reconstruction_latent_{latent_dim}.gif", writer=PillowWriter(fps=10))

    plt.show()

if __name__ == '__main__':
    latent_dim = 16
    input_dim = 513

    # Load the trained autoencoder model
    autoencoder_model = torch.load(f'dense_autoencoder_complete_latent_{latent_dim}.pth')
    autoencoder_model.eval()

    # Load the mean and std values
    mean = np.load('data_mean.npy')
    std = np.load('data_std.npy')

    # Load a specific snapshot for reconstruction
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Convert to PyTorch tensor and normalize
    snapshot = torch.tensor(snapshot.T, dtype=torch.float32)  # Transpose to match input dimension
    snapshot_normalized = (snapshot - mean) / std

    # Reconstruct the snapshot using the autoencoder
    with torch.no_grad():
        reconstructed_snapshot_normalized = autoencoder_model(snapshot_normalized)
        reconstructed_snapshot = reconstructed_snapshot_normalized * std + mean

    # Convert back to numpy array for plotting
    snapshot = snapshot.numpy().T
    reconstructed_snapshot = reconstructed_snapshot.numpy().T

    # Save the reconstructed snapshots
    np.save(f'reconstructed_snapshots_latent_{latent_dim}.npy', reconstructed_snapshot)

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
    create_combined_gif(X, snapshot, reconstructed_snapshot, nTimeSteps, At, latent_dim)






