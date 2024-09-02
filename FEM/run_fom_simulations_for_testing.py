import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from fem_burgers import FEMBurgers  # Import the FEMBurgers class
import os

# Load the selected testing samples
testing_samples = np.load('testing_data/testing_samples.npy')

# Create directories for saving the testing data and gifs
os.makedirs("testing_data/simulations", exist_ok=True)
os.makedirs("testing_data/gifs", exist_ok=True)

# Running the simulations for each testing sample
for i, (mu1, mu2) in enumerate(testing_samples):
    print(f"Running simulation {i+1}/{len(testing_samples)} with mu1={mu1:.2f}, mu2={mu2:.4f}")

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
    Tf = 35
    At = 0.07
    nTimeSteps = int(Tf / At)
    E = 0.01

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Run the simulation
    U_FOM = fem_burgers.fom_burgers(At, nTimeSteps, u0, mu1, E, mu2)

    # Construct filename with parameters
    filename = f"testing_data/simulations/simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.npy"

    # Save the result
    np.save(filename, U_FOM)

    # # Create and save animation as GIF
    # U = U_FOM
    # fig, ax = plt.subplots()
    # line, = ax.plot(X, U[:, 0], label='Solution over time')
    # ax.set_xlim(a, b)
    # ax.set_ylim(0, 8)
    # ax.set_xlabel('x')
    # ax.set_ylabel('u')
    # ax.legend()

    # def update(frame):
    #     line.set_ydata(U[:, frame])
    #     ax.set_title(f't = {frame * At:.2f}')
    #     return line,

    # ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)

    # # Save animation as GIF
    # gif_filename = f"testing_data/gifs/simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.gif"
    # ani.save(gif_filename, writer=PillowWriter(fps=10))
    # plt.close(fig)  # Close the figure to avoid displaying it

print("Simulations completed and results saved.")
