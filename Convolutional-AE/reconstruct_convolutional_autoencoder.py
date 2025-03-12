import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import os

# Define the ConvAutoencoder class (make sure this matches your trained model's architecture)
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),  # (batch_size, 16, 512)
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),  # Downsample by 2 (batch_size, 16, 256)
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # (batch_size, 32, 256)
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),  # Downsample by 2 (batch_size, 32, 128)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # (batch_size, 64, 128)
            nn.ELU(),
            nn.MaxPool1d(2, stride=2)  # Downsample by 2 (batch_size, 64, 64)
        )

        # Fully connected layers for latent space representation
        self.fc1 = nn.Linear(64 * 64, latent_dim)  # Compress to latent space
        self.fc2 = nn.Linear(latent_dim, 64 * 64)  # Expand back

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # Upsample by 2 (batch_size, 32, 128)
            nn.ELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # Upsample by 2 (batch_size, 16, 256)
            nn.ELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),  # Upsample by 2 (batch_size, 1, 512)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc1(x)  # Compress to latent space
        x = self.fc2(x)  # Expand back
        x = x.view(x.size(0), 64, 64)  # Reshape to match the last Conv layer output shape
        x = self.decoder(x)
        return x

# Function to create gif with all snapshots overlaid
def create_combined_gif(X, original_snapshot, autoencoder_reconstructed, nTimeSteps, At, latent_dim):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 8)

    line_original, = ax.plot(X, original_snapshot[:, 0], 'b-', label='Original Snapshot')
    line_autoencoder, = ax.plot(X, autoencoder_reconstructed[:, 0], 'r--', label=f'Convolutional AE Reconstructed (latent dim={latent_dim})')

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
    ani.save("convolutional_ae_reconstruction.gif", writer=PillowWriter(fps=10))

    plt.show()

if __name__ == '__main__':
    # Load the trained autoencoder model
    autoencoder_model = ConvAutoencoder(latent_dim=16)
    autoencoder_model.load_state_dict(torch.load('best_conv_autoencoder.pth'))
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
    snapshot = torch.tensor(snapshot.T[:, :-1], dtype=torch.float32).unsqueeze(1)  # Transpose and remove last node
    snapshot_normalized = (snapshot - mean) / std

    # Reconstruct the snapshot using the autoencoder
    with torch.no_grad():
        reconstructed_snapshot_normalized = autoencoder_model(snapshot_normalized)
        reconstructed_snapshot = reconstructed_snapshot_normalized * std + mean

    # Convert back to numpy array for plotting
    snapshot = snapshot.squeeze().numpy().T  # Remove channel dimension and transpose
    reconstructed_snapshot = reconstructed_snapshot.squeeze().numpy().T  # Remove channel dimension and transpose

    # Domain
    a = 0
    b = 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)[:-1]

    # Time discretization and numerical diffusion
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)

    # Create the combined GIF
    create_combined_gif(X, snapshot, reconstructed_snapshot, nTimeSteps, At, 16)
