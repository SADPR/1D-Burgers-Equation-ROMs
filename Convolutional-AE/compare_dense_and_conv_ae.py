import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import os

# Define the DenseAutoencoder class
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DenseAutoencoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.ELU(),
        #     nn.Linear(512, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 128),
        #     nn.ELU(),
        #     nn.Linear(128, 64),
        #     nn.ELU(),
        #     nn.Linear(64, latent_dim),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, 64),
        #     nn.ELU(),
        #     nn.Linear(64, 128),
        #     nn.ELU(),
        #     nn.Linear(128, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 512),
        #     nn.ELU(),
        #     nn.Linear(512, input_dim),
        # )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

# Define the ConvAutoencoder class
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvAutoencoder, self).__init__()

    #     # Encoder
    #     self.encoder = nn.Sequential(
    #         nn.Conv1d(1, 16, kernel_size=3, padding=1),  # (batch_size, 16, 512)
    #         nn.ELU(),
    #         nn.MaxPool1d(2, stride=2),  # Downsample by 2 (batch_size, 16, 256)
    #         nn.Conv1d(16, 32, kernel_size=3, padding=1),  # (batch_size, 32, 256)
    #         nn.ELU(),
    #         nn.MaxPool1d(2, stride=2),  # Downsample by 2 (batch_size, 32, 128)
    #         nn.Conv1d(32, 64, kernel_size=3, padding=1),  # (batch_size, 64, 128)
    #         nn.ELU(),
    #         nn.MaxPool1d(2, stride=2)  # Downsample by 2 (batch_size, 64, 64)
    #     )

    #     # Fully connected layers for latent space representation
    #     self.fc1 = nn.Linear(64 * 64, latent_dim)  # Compress to latent space
    #     self.fc2 = nn.Linear(latent_dim, 64 * 64)  # Expand back

    #     # Decoder
    #     self.decoder = nn.Sequential(
    #         nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # Upsample by 2 (batch_size, 32, 128)
    #         nn.ELU(),
    #         nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # Upsample by 2 (batch_size, 16, 256)
    #         nn.ELU(),
    #         nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),  # Upsample by 2 (batch_size, 1, 512)
    #     )

    def forward(self, x):
        x = self.encoder(x)  # Encode the input
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer (batch_size, 64*2)
        x = self.fc1(x)  # Compress to latent space (batch_size, latent_dim)
        x = self.fc2(x)  # Expand back (batch_size, 64*2)
        x = x.view(x.size(0), 64, 2)  # Reshape to match the last Conv layer output shape
        x = self.decoder(x)  # Decode the latent representation
        return x

# Function to create gif with all snapshots overlaid
def create_combined_gif(X_dense, X_conv, original_snapshot_dense, original_snapshot_conv, dense_reconstructed, conv_reconstructed, nTimeSteps, At, latent_dim_dense, latent_dim_conv):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(min(X_dense[0], X_conv[0]), max(X_dense[-1], X_conv[-1]))
    ax.set_ylim(0, 8)

    line_original_dense, = ax.plot(X_dense, original_snapshot_dense[:, 0], 'b-', label='Original Snapshot')
    line_dense, = ax.plot(X_dense, dense_reconstructed[:, 0], 'g--', label=f'Dense Autoencoder Reconstructed (latent dim={latent_dim_dense})')
    # line_original_conv, = ax.plot(X_conv, original_snapshot_conv[:, 0], 'c-', label='Original Snapshot (Conv)')
    line_conv, = ax.plot(X_conv, conv_reconstructed[:, 0], 'r--', label=f'Conv Autoencoder Reconstructed (latent dim={latent_dim_conv})')

    ax.set_title('Snapshot Comparison')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line_original_dense.set_ydata(original_snapshot_dense[:, frame])
        line_dense.set_ydata(dense_reconstructed[:, frame])
        # line_original_conv.set_ydata(original_snapshot_conv[:, frame])
        line_conv.set_ydata(conv_reconstructed[:, frame])
        ax.set_title(f'Snapshot Comparison at t = {frame * At:.2f}')
        return line_original_dense, line_dense, line_conv

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    ani.save("ae_comparison.gif", writer=PillowWriter(fps=10))

    plt.show()

if __name__ == '__main__':
    # Load the trained autoencoder models
    dense_autoencoder_model = torch.load('../AE/dense_autoencoder_complete.pth')
    dense_autoencoder_model.eval()

    conv_autoencoder_model = torch.load('conv_autoencoder_complete_carlberg.pth')
    conv_autoencoder_model.eval()

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

    # Preprocess the snapshot for Dense Autoencoder
    snapshot_dense = torch.tensor(snapshot.T, dtype=torch.float32)  # Transpose for dense autoencoder
    snapshot_dense_normalized = (snapshot_dense - mean) / std

    # Preprocess the snapshot for Conv Autoencoder
    snapshot_conv = torch.tensor(snapshot.T, dtype=torch.float32).unsqueeze(1)  # Transpose and remove last node for conv autoencoder
    snapshot_conv_normalized = (snapshot_conv - mean) / std

    # Reconstruct the snapshot using the dense autoencoder
    with torch.no_grad():
        dense_reconstructed_normalized = dense_autoencoder_model(snapshot_dense_normalized)
        dense_reconstructed = dense_reconstructed_normalized * std + mean

    # Reconstruct the snapshot using the convolutional autoencoder
    with torch.no_grad():
        conv_reconstructed_normalized = conv_autoencoder_model(snapshot_conv_normalized)
        conv_reconstructed = conv_reconstructed_normalized * std + mean

    # Save the reconstructed snapshots to .npy files
    np.save('dense_reconstructed_snapshot.npy', dense_reconstructed.numpy())
    np.save('conv_reconstructed_snapshot.npy', conv_reconstructed.numpy())

    # Convert back to numpy array for plotting
    snapshot_dense = snapshot_dense.numpy().T  # Transpose for plotting
    dense_reconstructed = dense_reconstructed.numpy().T  # Transpose for plotting
    snapshot_conv = snapshot_conv.squeeze().numpy().T  # Remove channel dimension and transpose
    conv_reconstructed = conv_reconstructed.squeeze().numpy().T  # Remove channel dimension and transpose

    # Domain
    a = 0
    b = 100
    m_dense = int(256 * 2)
    m_conv = int(256 * 2)  # Remove one node for conv
    X_dense = np.linspace(a, b, m_dense + 1)
    X_conv = np.linspace(a, b, m_conv + 1)

    # Time discretization and numerical diffusion
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)

    # Create the combined GIF
    create_combined_gif(X_dense, X_conv, snapshot_dense, snapshot_conv, dense_reconstructed, conv_reconstructed, nTimeSteps, At, 16, 16)

