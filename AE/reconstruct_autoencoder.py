import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import random
import torch
import torch.nn as nn

# Define the DenseAutoencoder class
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed



# Function to create gif with all snapshots overlaid
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
    ani.save("ae_reconstruction.gif", writer=PillowWriter(fps=10))

    plt.show()

if __name__ == '__main__':
    # Load the trained autoencoder model
    autoencoder_model = DenseAutoencoder(513, 16)
    autoencoder_model = torch.load('dense_autoencoder_complete.pth')
    autoencoder_model.eval()

    # Load the mean and std values
    data_path = '../training_data/'  # Replace with your data folder
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
    all_snapshots = []

    for file in files:
        snapshots = np.load(file)
        all_snapshots.append(snapshots)

    all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)
    mean = np.mean(all_snapshots)
    std = np.std(all_snapshots)

    # Load a random snapshot from the training_data directory
    snapshot_file = '../training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)

    # Convert to PyTorch tensor and normalize using PyTorch's operations
    snapshot = torch.tensor(snapshot.T, dtype=torch.float32)  # Transpose to match input dimension
    snapshot_normalized = (snapshot - mean) / std

    # Reconstruct the snapshot using the autoencoder
    with torch.no_grad():
        reconstructed_snapshot_normalized = autoencoder_model(snapshot_normalized)
        reconstructed_snapshot = reconstructed_snapshot_normalized * std + mean

    # Convert back to numpy array for plotting
    snapshot = snapshot.numpy().T
    reconstructed_snapshot = reconstructed_snapshot.numpy().T

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
    create_combined_gif(X, snapshot, reconstructed_snapshot, nTimeSteps, At, 16)





