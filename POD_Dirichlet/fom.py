import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys

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

    # Time discretization and numerical diffusion
    Tf = 5
    At = 0.07
    nTimeSteps = int(Tf / At)
    E = 0.01

    # Parameters
    mu1 = 4.76
    mu2 = 0.0182

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Run the simulation
    U_FOM = fem_burgers.fom_burgers_dirichlet(At, nTimeSteps, u0, mu1, E, mu2)

    # Construct filename with parameters
    filename = f"simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.npy"

    # Save the result
    np.save(filename, U_FOM)

    # Create and save animation as GIF
    U = U_FOM
    fig, ax = plt.subplots()
    line, = ax.plot(X, U[:, 0], label='Solution over time')
    ax.set_xlim(a, b)
    ax.set_ylim(0, 8)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line.set_ydata(U[:, frame])
        ax.set_title(f't = {frame * At:.2f}')
        return line,

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)
    plt.show()

    # Save animation as GIF
    # gif_filename = f"simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.gif"
    # ani.save(gif_filename, writer=PillowWriter(fps=10))
    # plt.close(fig)  # Close the figure to avoid displaying it
    



