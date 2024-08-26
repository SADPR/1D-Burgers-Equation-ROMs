import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os
import joblib

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../FEM'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module
from fem_burgers import FEMBurgers

def main(n_clusters):
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
    uxa = 4.76  # u(0,t) = 4.76

    # Time discretization and numerical diffusion
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)
    E = 0.01

    # Parameter mu2
    mu2 = 0.0182

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Load k-means model and local bases based on the number of clusters
    kmeans_filename = f'clusters/kmeans_model_{n_clusters}_clusters.pkl'
    local_bases_filename = f'clusters/local_bases_overlap_{n_clusters}_clusters.npy'
    kmeans = joblib.load(kmeans_filename)
    local_bases = np.load(local_bases_filename, allow_pickle=True).item()
    U_global = np.load('clusters/U_global.npy')

    # Number of global modes to use for clustering
    num_global_modes = 301  # Adjust based on your data

    # Solution using Local PROM
    print('Local PROM method (Picard)...')
    projection="LSPG"
    U_PROM = fem_burgers.local_prom_burgers(At, nTimeSteps, u0, uxa, E, mu2, kmeans, local_bases, U_global, num_global_modes, projection=projection)

    np.save(f"local_PROM_{n_clusters}_clusters_{projection}.npy", U_PROM)

    # Postprocess
    npasplot = round(nTimeSteps / 10)
    ind = list(range(0, nTimeSteps + 1, npasplot)) + [nTimeSteps]

    U = U_PROM
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

    # ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # # Save animation as GIF
    # ani.save(f"local_burgers_equation_prom_{n_clusters}_clusters.gif", writer=PillowWriter(fps=10))

    # plt.show()

if __name__ == "__main__":
    # Set the number of clusters here
    n_clusters = 20  # Example: set this to the number of clusters you want to use
    main(n_clusters)


