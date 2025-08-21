import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# LaTeX formatting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_param_gif(mu1=5.500, mu2=0.0300, save_dir="simulation_results"):
    # Domain
    a, b = 0, 100
    m = int(256 * 2)
    X = np.linspace(a, b, m + 1)

    # Time step (should match simulation)
    At = 0.05

    # Load solution
    filename = f"{save_dir}/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    U_FOM = np.load(filename)
    nTimeSteps = U_FOM.shape[1] - 1

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))
    line, = ax.plot(X, U_FOM[:, 0], color='blue', lw=2)
    ax.set_xlim(a, b)
    ax.set_ylim(0.5, 7.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x,t)$')
    ax.grid(True)

    def update(frame):
        line.set_ydata(U_FOM[:, frame])
        ax.set_title(rf'$\mu_1={mu1:.3f},\ \mu_2={mu2:.4f},\ t={frame*At:.2f}\ \mathrm{{s}}$')
        return line,

    ani = FuncAnimation(fig, update, frames=nTimeSteps + 1, blit=True)
    gif_file = f"simulation_results/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif"
    ani.save(gif_file, writer=PillowWriter(fps=50))
    plt.close(fig)

if __name__ == "__main__":
    plot_param_gif()
