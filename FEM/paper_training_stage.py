import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from fem_burgers import FEMBurgers
import os

# Define 3x3 grid for training
mu1_values = np.linspace(4.25, 5.50, 3)  # [4.25, 4.875, 5.50]
mu2_values = np.linspace(0.015, 0.03, 3)  # [0.015, 0.0225, 0.03]
all_samples = np.array([[mu1, mu2] for mu1 in mu1_values for mu2 in mu2_values])

# Create directories
os.makedirs("training_data", exist_ok=True)
os.makedirs("training_gifs", exist_ok=True)

# Save parameter combinations
np.save("training_data/parameter_combinations.npy", all_samples)

# Visualize the sampling
plt.scatter(all_samples[:, 0], all_samples[:, 1])
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.title('Training Grid Sampling ($\mu_1$, $\mu_2$)')
plt.savefig("training_data/training_grid_plot.pdf")
plt.close()

# Run simulations
for i, (mu1, mu2) in enumerate(all_samples):
    print(f"Training sample {i+1}/{len(all_samples)}: mu1={mu1:.3f}, mu2={mu2:.4f}")

    # Domain and mesh
    a, b = 0, 100
    m = 512
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

    # Instantiate and simulate
    fem_burgers = FEMBurgers(X, T)
    U_FOM = fem_burgers.fom_burgers(At, nTimeSteps, u0, mu1, E, mu2)

    # Save solution
    file_name = f"training_data/simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    np.save(file_name, U_FOM)

    # Save GIF
    fig, ax = plt.subplots()
    line, = ax.plot(X, U_FOM[:, 0], label='Solution over time')
    ax.set_xlim(a, b)
    ax.set_ylim(0, 8)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line.set_ydata(U_FOM[:, frame])
        ax.set_title(f't = {frame * At:.2f}')
        return line,

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)
    gif_file = f"training_gifs/simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif"
    ani.save(gif_file, writer=PillowWriter(fps=10))
    plt.close(fig)

