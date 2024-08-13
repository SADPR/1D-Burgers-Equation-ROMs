import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from fem_burgers import FEMBurgers  # Import the FEMBurgers class
from scipy.stats import qmc
import os

# Latin Hypercube Sampling
def generate_lhs_samples(n_samples, param_ranges):
    """
    Generate Latin Hypercube Samples for given parameter ranges.
    
    Args:
        n_samples: Number of samples to generate.
        param_ranges: List of tuples containing (min, max) for each parameter.
    
    Returns:
        samples: Array of shape (n_samples, n_params) with LHS samples.
    """
    n_params = len(param_ranges)
    sampler = qmc.LatinHypercube(d=n_params)
    lhs_samples = sampler.random(n=n_samples)
    
    samples = np.zeros_like(lhs_samples)
    for i in range(n_params):
        min_val, max_val = param_ranges[i]
        samples[:, i] = lhs_samples[:, i] * (max_val - min_val) + min_val
    
    return samples

# Parameter ranges for mu1 and mu2
param_ranges = [(4.25, 5.5), (0.015, 0.03)]
n_lhs_samples = 4  # Number of LHS samples excluding the 4 corners

# Generate LHS samples
lhs_samples = generate_lhs_samples(n_lhs_samples, param_ranges)

# Define the corner points
corners = np.array([
    [4.25, 0.015],
    [4.25, 0.03],
    [5.5, 0.015],
    [5.5, 0.03]
])

# Combine the LHS samples with the corners
all_samples = np.vstack([lhs_samples, corners])

# Create directories for saving the training data and gifs
os.makedirs("../new_training_data", exist_ok=True)
os.makedirs("../new_training_gifs", exist_ok=True)

# Save the parameter combinations
np.save("../new_training_data/parameter_combinations.npy", all_samples)

# Visualize the sampling and save the plot as PNG
plt.scatter(all_samples[:, 0], all_samples[:, 1])
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.title('LHS Sampling of $\mu_1$ and $\mu_2$')
plt.savefig("../new_training_data/lhs_sampling_plot.png")
plt.close()  # Close the plot to avoid displaying it

# Create directories for saving the training data and gifs
os.makedirs("../new_training_data", exist_ok=True)
os.makedirs("../new_training_gifs", exist_ok=True)

# Running the simulations for each sample
for i, (mu1, mu2) in enumerate(all_samples):
    print(f"Running simulation {i+1}/{len(all_samples)} with mu1={mu1:.2f}, mu2={mu2:.4f}")

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
    Tf = 2
    At = 0.07
    nTimeSteps = int(Tf / At)
    E = 0.01

    # Create an instance of the FEMBurgers class
    fem_burgers = FEMBurgers(X, T)

    # Run the simulation
    U_FOM = fem_burgers.fom_burgers(At, nTimeSteps, u0, mu1, E, mu2)

    # Construct filename with parameters
    filename = f"../new_training_data/simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.npy"

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

    # Save animation as GIF
    gif_filename = f"../new_training_gifs/simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.gif"
    ani.save(gif_filename, writer=PillowWriter(fps=10))
    plt.close(fig)  # Close the figure to avoid displaying it

# Save the samples for reference
np.save("../new_training_data/lhs_samples.npy", all_samples)



