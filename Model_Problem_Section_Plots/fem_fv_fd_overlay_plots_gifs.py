#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# LaTeX-friendly plot settings
plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"],
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rc('font', size=14)

def load_three(mu1, mu2, a=0.0, b=100.0, base="../"):
    U_FEM = np.load(f"{base}FEM/fem_training_data/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
    N_fem = U_FEM.shape[0]; x_fem = np.linspace(a, b, N_fem)

    U_FV  = np.load(f"{base}FV/fv_training_data/fv_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
    N_fv = U_FV.shape[0]; dx_fv = (b - a) / N_fv; x_fv = np.linspace(a + dx_fv/2, b - dx_fv/2, N_fv)

    U_FD  = np.load(f"{base}FD/fd_training_data/fd_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
    N_fd = U_FD.shape[0]; x_fd = np.linspace(a, b, N_fd)

    return (x_fem, U_FEM), (x_fv, U_FV), (x_fd, U_FD)

def overlay_gif_for_combo(mu1, mu2,
                          a=0.0, b=100.0, dt=0.05,
                          out_dir="overlay_comparisons_FEM_FV_FD",
                          fps=30, dpi=90,
                          times_to_plot=None, frame_stride=1,
                          fig_size=(7, 5)):   # ← match your other GIFs
    os.makedirs(out_dir, exist_ok=True)

    (x_fem, U_FEM), (x_fv, U_FV), (x_fd, U_FD) = load_three(mu1, mu2, a=a, b=b, base="../")

    n_frames_all = min(U_FEM.shape[1], U_FV.shape[1], U_FD.shape[1])
    if times_to_plot is None:
        frames = np.arange(0, n_frames_all, frame_stride)
    else:
        idxs = [int(t / dt) for t in times_to_plot if 0 <= int(t / dt) < n_frames_all]
        frames = np.array(idxs, dtype=int)
        if frames.size == 0:
            raise ValueError("times_to_plot produced no valid frames.")

    u_min = min(U_FEM[:, frames].min(), U_FV[:, frames].min(), U_FD[:, frames].min())
    u_max = max(U_FEM[:, frames].max(), U_FV[:, frames].max(), U_FD[:, frames].max())
    pad = 0.10 * max(1e-12, u_max - u_min)
    y_min, y_max = u_min - pad, u_max + pad

    # --- FIGURE: match look & margins of other 1D GIFs
    fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    ax.set_xlim(a, b); ax.set_ylim(y_min, y_max)
    ax.grid(True)
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$u(x,t)$')
    ax.margins(x=0)

    fem_line, = ax.plot(x_fem, U_FEM[:, frames[0]], color='black', linestyle='-',  lw=2, label='FEM')
    fv_line,  = ax.plot(x_fv,  U_FV[:,  frames[0]], color='green', linestyle='--', lw=2, label='FV')
    fd_line,  = ax.plot(x_fd,  U_FD[:,  frames[0]], color='red',   linestyle='-.', lw=2, label='FD')

    ax.legend(loc='upper right')

    def set_title_for_frame(k):
        t = k * dt
        ax.set_title(rf'$\bm{{\mu}}=[{mu1:.3f},\,{mu2:.4f}],\ t={t:.2f}\,\mathrm{{s}}$')

    set_title_for_frame(frames[0])

    def update(i):
        k = frames[i]
        fem_line.set_ydata(U_FEM[:, k])
        fv_line.set_ydata(U_FV[:, k])
        fd_line.set_ydata(U_FD[:, k])
        set_title_for_frame(k)
        return fem_line, fv_line, fd_line

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out_path = os.path.join(out_dir, f"overlay_mu1_{mu1:.3f}_mu_2_{mu2:.4f}.gif")
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[ok] saved → {out_path}")

# Example usage
if __name__ == "__main__":
    overlay_gif_for_combo(mu1=4.250, mu2=0.0225,
                          dt=0.05,
                          out_dir="overlay_comparisons_FEM_FV_FD",
                          fps=50, dpi=90,
                          times_to_plot=None,   # or [5,10,15,20,25]
                          frame_stride=1,
                          fig_size=(7, 5))      # ← same shape as your other 1D GIFs
