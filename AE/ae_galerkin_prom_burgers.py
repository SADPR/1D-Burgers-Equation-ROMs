import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import sys

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../FEM/'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module
from fem_burgers import FEMBurgers


# Define the DenseAutoencoder class
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DenseAutoencoder, self).__init__()

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

if __name__ == "__main__":
    # Domain
    a = 0
    b = 100

    # Mesh
    m = int(256 * 2)
    h = (b - a) / m
    X = np.linspace(a, b, m + 1)
    T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

    # Initial condition
    u0 = np.ones_like(X)

    umax = np.max(u0) + 0.1
    umin = np.min(u0) - 0.1

    # Boundary conditions
    uxa = 4.76  # u(0,t) = 4.3

    # Time discretization and numerical diffusion
    Tf = 0.49
    At = 0.07
    nTimeSteps = int(round(Tf / At)) + 1
    E = 0.01

    # Parameter mu2
    mu2 = 0.0182

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Load the trained autoencoder
    input_dim = 513
    latent_dim = 16
    model = torch.load(f'dense_autoencoder_complete_latent_{latent_dim}.pth')
    model.eval()

    # Load the mean and std values
    mean = np.load('data_mean.npy')
    std = np.load('data_std.npy')

    # Solution using AE-based PROM
    print('AE-PROM method (Picard)...')
    U_AEPROM = fem_burgers.ae_prom(At, nTimeSteps, u0, uxa, E, mu2, model, mean, std)

    # Plotting each time step in a separate subplot
    fig, axes = plt.subplots(nrows=(nTimeSteps // 4) + 1, ncols=4, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(nTimeSteps + 1):
        ax = axes[i]
        ax.plot(X, U_AEPROM[:, i])
        ax.set_xlim(a, 3)
        ax.set_ylim(0, 6)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f't = {i * At:.2f}')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


