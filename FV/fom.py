import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from fv_burgers import FVBurgers  # Import your FV implementation class

if __name__ == "__main__":

    # Domain
    a = 0.0
    b = 100.0

    # Mesh (cell centers)
    N = 512
    dx = (b - a) / N
    x_centers = np.linspace(a + dx/2, b - dx/2, N)  # physical cells only

    # Initial condition
    u0 = np.ones_like(x_centers)

    # Time parameters
    Tf = 5.0
    dt = 0.05
    n_steps = int(Tf / dt)

    # Parameters
    mu1 = 4.75     # Left Dirichlet BC
    mu2 = 0.0300   # Source term exponential parameter

    # Create FVBurgers instance
    fv_burgers = FVBurgers(a, b, N)

    # Run simulation
    U_FOM = fv_burgers.fom_burgers_newton(dt, n_steps, u0, mu1, mu2)

    # Save result
    filename = f"fv_simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.npy"
    np.save(filename, U_FOM)

    # Animate
    fig, ax = plt.subplots()
    line, = ax.plot(x_centers, U_FOM[:, 0], label='Solution')
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
    plt.show()

    gif_filename = f"fv_simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.gif"
    ani.save(gif_filename, writer=PillowWriter(fps=10))
    plt.close(fig)