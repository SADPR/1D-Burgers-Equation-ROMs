#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# LaTeX formatting (tick numbers will use TeX)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_param_gif(mu1, mu2, data_dir="simulation_results", out_dir="simulation_results",
                   At=0.05, a=0.0, b=100.0, m=int(256*2), ylim=(0.5, 7.5),
                   fps=30, dpi=90, line_color='blue'):
    """Make a single GIF for given (mu1, mu2) with grid + numeric ticks only."""
    os.makedirs(out_dir, exist_ok=True)

    # Spatial grid
    X = np.linspace(a, b, m + 1)

    # Load solution
    filename = f"{data_dir}/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    if not os.path.isfile(filename):
        print(f"[skip] file not found: {filename}")
        return
    U_FOM = np.load(filename)
    n_frames = U_FOM.shape[1]

    # Figure
    fig, ax = plt.subplots(figsize=(7, 5))
    (line,) = ax.plot(X, U_FOM[:, 0], color=line_color, lw=2)

    # Limits
    ax.set_xlim(a, b)
    ax.set_ylim(*ylim)

    # Grid + numeric ticks only
    ax.grid(True, which='both', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('')  # no label
    ax.set_ylabel('')  # no label
    # keep ticks & numbers (default); ensure they’re visible
    ax.tick_params(direction='out', length=4, width=0.8)

    # Keep spines (default). Trim margins but leave room for tick numbers
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    ax.margins(x=0)

    def update(frame):
        line.set_ydata(U_FOM[:, frame])
        return line,

    ani = FuncAnimation(fig, update, frames=n_frames, blit=True)
    gif_file = f"{out_dir}/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}_grid.gif"
    ani.save(gif_file, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[ok] saved → {gif_file}")

def plot_all_training_gifs(data_dir="simulation_results", out_dir="simulation_results",
                           mu1_values=None, mu2_values=None, At=0.05, fps=50, dpi=90):
    """Loop over a grid of (mu1, mu2) and render a GIF for each (grid + ticks only)."""
    if mu1_values is None:
        mu1_values = np.linspace(4.25, 5.50, 3)      # [4.250, 4.875, 5.500]
    if mu2_values is None:
        mu2_values = np.linspace(0.0150, 0.0300, 3)  # [0.0150, 0.0225, 0.0300]

    print(f"Rendering {len(mu1_values)*len(mu2_values)} GIFs…")
    for mu1 in mu1_values:
        for mu2 in mu2_values:
            plot_param_gif(mu1, mu2,
                           data_dir=data_dir, out_dir=out_dir, At=At,
                           fps=fps, dpi=dpi, line_color='blue')

if __name__ == "__main__":
    plot_all_training_gifs(
        data_dir="simulation_results",
        out_dir="fem_training_gifs",
        At=0.05,
        fps=50,
        dpi=90
    )
