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
    # Domain and mesh
    a, b = 0, 100

    # Mesh
    m = 511
    h = (b - a) / m
    X = np.linspace(a, b, m + 1)
    T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

    # Initial condition
    u0 = np.ones_like(X)

    # Time discretization
    Tf = 25
    At = 0.05
    nTimeSteps = int(Tf / At)
    E = 0.00

    # Boundary conditions
    mu1 = 4.750  # u(0,t) = 4.750

    # Parameter mu2
    mu2 = 0.0200

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Load reduced basis (example, replace with actual basis)
    tol = 1e-04
    Phi = np.load(f"modes/U_modes_tol_{tol:.0e}.npy")

    # Solution using PROM
    print('PROM method (Picard)...')
    U_PROM = fem_burgers.pod_prom_burgers(At, nTimeSteps, u0, mu1, E, mu2, Phi, projection="LSPG")

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
