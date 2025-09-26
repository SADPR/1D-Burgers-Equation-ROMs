#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ----------------- utils -----------------
def get_num_modes(tol, modes_dir="../modes"):
    modes_file = os.path.join(modes_dir, f"U_modes_tol_{tol:.0e}.npy")
    U_modes = np.load(modes_file)
    return U_modes.shape[1]

def compute_rel_error(U_FOM, U_ROM):
    """Relative error: sqrt( sum_t ||FOM-ROM||^2 / sum_t ||FOM||^2 )."""
    num_steps = U_FOM.shape[1]
    num = 0.0
    denom = 0.0
    for m in range(num_steps):
        diff = U_FOM[:, m] - U_ROM[:, m]
        num += np.linalg.norm(diff, 2) ** 2
        denom += np.linalg.norm(U_FOM[:, m], 2) ** 2
    return np.sqrt(num / denom)

# --------------- GIF maker ---------------
def gif_fom_vs_lspg_for_case(
    tol, mu1, mu2,
    fom_base="../../FEM/fem_testing_data",
    rom_dir="rom_solutions",
    out_dir="rom_solutions/gifs",
    At=0.05, a=0.0, b=100.0,
    fps=30, dpi=90, frame_stride=1,
    fig_size=(7, 5)
):
    """Create one GIF comparing FOM vs ROM(LSPG) for (mu1,mu2) at a given tol."""
    os.makedirs(out_dir, exist_ok=True)

    tag_mu1 = f"{mu1:.3f}"
    tag_mu2 = f"{mu2:.4f}"

    # Load data
    fom_filename = os.path.join(fom_base, f"fem_simulation_mu1_{tag_mu1}_mu2_{tag_mu2}.npy")
    rom_filename = os.path.join(rom_dir, f"U_PROM_tol_{tol:.0e}_mu1_{tag_mu1}_mu2_{tag_mu2}_lspg.npy")

    U_FOM = np.load(fom_filename)             # shape (Nx, Nt+1)
    U_ROM = np.load(rom_filename)

    # Grid from FOM
    Nx = U_FOM.shape[0]
    X = np.linspace(a, b, Nx)

    # Frames
    n_frames_all = min(U_FOM.shape[1], U_ROM.shape[1])
    frames = np.arange(0, n_frames_all, frame_stride, dtype=int)

    # y-lims across both solutions (chosen frames)
    u_min = min(U_FOM[:, frames].min(), U_ROM[:, frames].min())
    u_max = max(U_FOM[:, frames].max(), U_ROM[:, frames].max())
    pad = 0.10 * max(1e-12, u_max - u_min)
    y_min, y_max = u_min - pad, u_max + pad

    # Diagnostics
    modes = get_num_modes(tol)
    rel_err = compute_rel_error(U_FOM, U_ROM)
    print(f"[tol={tol:.0e}, mu1={mu1:.3f}, mu2={mu2:.4f}] modes={modes}  rel.err={rel_err:.6e}")

    # Figure
    fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    ax.set_xlim(a, b)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")

    # Lines
    fom_line, = ax.plot(X, U_FOM[:, frames[0]], color='black', lw=2, label="FOM")
    rom_line, = ax.plot(X, U_ROM[:, frames[0]], color='blue',  lw=2, label="PROM")

    ax.legend(loc='upper right')

    def set_title_for_k(k):
        t = k * At
        ax.set_title(
            rf"$\mu_1={mu1:.3f},\ \mu_2={mu2:.4f},\ \mathrm{{tol}}={tol:.0e},\ \mathrm{{modes}}={modes}"
            rf"\quad t={t:.2f}\,\mathrm{{s}}$"
        )

    set_title_for_k(frames[0])

    def update(i):
        k = frames[i]
        fom_line.set_ydata(U_FOM[:, k])
        rom_line.set_ydata(U_ROM[:, k])
        set_title_for_k(k)
        return fom_line, rom_line

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out_path = os.path.join(out_dir, f"FOM_ROM_tol_{tol:.0e}_mu1_{tag_mu1}_mu2_{tag_mu2}.gif")
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[ok] saved → {out_path}")

# ------------------ main ------------------
if __name__ == "__main__":
    test_points = [
        # (4.75, 0.0200),
        (4.56, 0.0190)
        # (5.19, 0.0260),
    ]
    tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    for tol in tolerances:
        for mu1, mu2 in test_points:
            try:
                gif_fom_vs_lspg_for_case(
                    tol=tol, mu1=mu1, mu2=mu2,
                    fom_base="../../FEM/fem_testing_data",
                    rom_dir="rom_solutions",
                    out_dir="rom_solutions/gifs",
                    At=0.05,
                    fps=50, dpi=90,
                    frame_stride=1,       # increase to 2 or 3 to shrink file size
                    fig_size=(7, 5)
                )
            except FileNotFoundError as e:
                print(f"[skip] missing data for tol={tol:.0e}, mu=({mu1:.3f},{mu2:.4f}): {e}")
