import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from fv_burgers import FVBurgers  # Your FV solver class
import os

# Define training parameter grid
mu1_values = np.linspace(4.25, 5.50, 3)
mu2_values = np.linspace(0.015, 0.03, 3)
all_samples = np.array([[mu1, mu2] for mu1 in mu1_values for mu2 in mu2_values])

# Directories
os.makedirs("fv_training_data", exist_ok=True)
os.makedirs("fv_training_gifs", exist_ok=True)

# Save grid and plot
np.save("fv_training_data/parameter_combinations.npy", all_samples)

plt.scatter(all_samples[:, 0], all_samples[:, 1])
plt.xlabel('$\mu_1$')
plt.ylabel('$\mu_2$')
plt.title('Training Grid Sampling ($\mu_1$, $\mu_2$)')
plt.savefig("fv_training_data/training_grid_plot.pdf")
plt.close()

# Run each sample
for i, (mu1, mu2) in enumerate(all_samples):
    print(f"FV Training Sample {i+1}/{len(all_samples)}: mu1 = {mu1:.3f}, mu2 = {mu2:.4f}")

    # Domain
    a, b = 0.0, 100.0
    N = 512
    dx = (b - a) / N
    x_centers = np.linspace(a + dx / 2, b - dx / 2, N)
    u0 = np.ones_like(x_centers)

    # Time settings
    Tf = 25
    dt = 0.05
    n_steps = int(Tf / dt)

    # Solve using FV
    fv_solver = FVBurgers(a, b, N)
    U_FOM = fv_solver.fom_burgers_newton(dt, n_steps, u0, mu1, mu2)

    # Save solution
    file_name = f"fv_training_data/fv_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    np.save(file_name, U_FOM)

    # Save GIF
    fig, ax = plt.subplots()
    line, = ax.plot(x_centers, U_FOM[:, 0], label='Solution over time')
    ax.set_xlim(a, b)
    ax.set_ylim(0, 8)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

    def update(frame):
        line.set_ydata(U_FOM[:, frame])
        ax.set_title(f't = {frame * dt:.2f}')
        return line,

    ani = FuncAnimation(fig, update, frames=n_steps + 1, blit=True)
    gif_file = f"fv_training_gifs/fv_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif"
    ani.save(gif_file, writer=PillowWriter(fps=10))
    plt.close(fig)
