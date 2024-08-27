import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from fem_burgers import FEMBurgers  # Import the FEMBurgers class
import torch
import torch.nn as nn
import os


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
    uxa = 4.3  # u(0,t) = 4.3

    # Time discretization and numerical diffusion
    Tf = 0.7
    At = 0.07
    nTimeSteps = int(Tf / At)
    E = 0.01

    # Parameter mu2
    mu2 = 0.021

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Load the trained autoencoder
    input_dim = 513
    latent_dim = 16
    model = DenseAutoencoder(input_dim, latent_dim)
    model = torch.load('dense_autoencoder_complete.pth')
    model.eval()

    # Load the mean and std values for normalization
    data_path = 'training_data/'  # Replace with your data folder
    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
    all_snapshots = []

    for file in files:
        snapshots = np.load(file)
        all_snapshots.append(snapshots)

    all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)
    mean = np.mean(all_snapshots)
    std = np.std(all_snapshots)

    # Solution using AE-based PROM
    print('AE-PROM method (Picard)...')
    U_AEPROM = fem_burgers.ae_prom(At, nTimeSteps, u0, uxa, E, mu2, model, mean, std)

    # Postprocess
    npasplot = round(nTimeSteps / 10)
    ind = list(range(0, nTimeSteps + 1, npasplot)) + [nTimeSteps]

    U = U_AEPROM
    fig, ax = plt.subplots()
    line, = ax.plot(X, U[:, 0], label='Solution over time')
    ax.set_xlim(a, b)
    ax.set_ylim(0, 6)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line.set_ydata(U[:, frame])
        ax.set_title(f't = {frame * At:.2f}')
        return line,

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    # ani.save("burgers_equation_ae_prom.gif", writer=PillowWriter(fps=10))

    plt.show()
