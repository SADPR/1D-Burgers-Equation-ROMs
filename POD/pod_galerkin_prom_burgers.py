import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os

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

    umax = np.max(u0) + 0.1
    umin = np.min(u0) - 0.1

    # Boundary conditions
    uxa = 4.76  # u(0,t) = 4.3

    # Time discretization and numerical diffusion
    Tf = 5
    At = 0.07
    nTimeSteps = int(Tf / At)
    E = 0.01

    # Parameter mu2
    mu2 = 0.0182

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Load reduced basis (example, replace with actual basis)
    tol = 1e-03
    Phi = np.load(f"modes/U_modes_tol_{tol:.0e}.npy")

    # Solution using PROM
    print('PROM method (Picard)...')
    U_PROM = fem_burgers.pod_prom_burgers(At, nTimeSteps, u0, uxa, E, mu2, Phi, projection="LSPG")

    np.save("pod_prom.npy", U_PROM)

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

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # Save animation as GIF
    ani.save("burgers_equation_prom.gif", writer=PillowWriter(fps=10))

    plt.show()
