import numpy as np
import os
import sys
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../FEM/'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module
from fem_burgers import FEMBurgers

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

    # Boundary conditions
    uxa = 4.76  # u(0,t) = 4.3

    # Time discretization and numerical diffusion
    Tf = 3
    At = 0.07
    nTimeSteps = int(Tf / At)
    E = 0.01

    # Parameter mu2
    mu2 = 0.0182

    # Number of modes to use (for quadratic manifold)
    num_modes = 28

    Phi_p = np.load("U_truncated.npy")
    H = np.load("H_quadratic.npy")

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Directory to save results
    save_dir = "quadratic_rom_solutions"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f'Running Quadratic PROM method (LSPG) with {num_modes} modes...')
    
    # Compute the Quadratic PROM solution
    U_PROM = fem_burgers.pod_quadratic_manifold(At, nTimeSteps, u0, uxa, E, mu2, Phi_p, H, num_modes, projection="Galerkin")

    # Save the solution
    np.save(os.path.join(save_dir, f"U_PROM_quadratic_{num_modes}_modes.npy"), U_PROM)

    print("Quadratic PROM simulation completed and saved.")

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(X, U_PROM[:, 0], color='b')
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(np.min(U_PROM), np.max(U_PROM))
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Solution at t = 0.00')

    # Function to update the plot at each time step
    def update(frame):
        line.set_ydata(U_PROM[:, frame])
        ax.set_title(f'Solution at t = {frame * At:.2f}')
        return line,

    # Create animation
    ani = FuncAnimation(fig, update, frames=nTimeSteps, blit=True)
    plt.show()

