import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Import your FD implementation class
from fd_burgers import FDBurgers

if __name__ == "__main__":
    # --- Domain Setup ---
    a = 0.0
    b = 100.0
    N = 512

    # Grid
    dx = (b - a) / (N - 1)
    x_nodes = np.linspace(a, b, N)  # This matches the definition in FDBurgers

    # Initial condition (u0 has shape N, same as x_nodes)
    u0 = np.ones(N)  # or any other function of x_nodes

    # --- Time Parameters ---
    Tf = 5.0
    dt = 0.05
    n_steps = int(Tf / dt)

    # --- Physical Parameters ---
    mu1 = 4.75    # Dirichlet BC at left boundary
    mu2 = 0.03    # Source exponent

    # --- Instantiate the FD solver ---
    fd_burgers = FDBurgers(a, b, N)

    # Run the simulation
    U_FOM = fd_burgers.fom_burgers_newton(
        dt=dt,
        n_steps=n_steps,
        U0=u0,
        mu1=mu1,
        mu2=mu2,
        max_iter=30,
        tol=1e-10,
        use_fd_jacobian=False  # set True to use FD Jacobian, else uses analytical
    )

    # --- Save final result to file ---
    filename = f"fd_simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.npy"
    np.save(filename, U_FOM)

    # --- Create Animation ---
    fig, ax = plt.subplots()
    line, = ax.plot(x_nodes, U_FOM[:, 0], label="FD Solution", color='blue')
    ax.set_xlim(a, b)
    ax.set_ylim(0, 8)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Finite Difference Burgers")
    ax.legend()

    def update(frame):
        # Update solution line
        line.set_ydata(U_FOM[:, frame])
        ax.set_title(f"t = {frame * dt:.2f}")
        return [line]

    ani = FuncAnimation(
        fig, update, frames=n_steps + 1, interval=100, blit=True
    )
    plt.show()

    # --- Save GIF ---
    gif_filename = f"fd_simulation_mu1_{mu1:.2f}_mu2_{mu2:.4f}.gif"
    ani.save(gif_filename, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Simulation complete. Results:\n  - {filename}\n  - {gif_filename}")
