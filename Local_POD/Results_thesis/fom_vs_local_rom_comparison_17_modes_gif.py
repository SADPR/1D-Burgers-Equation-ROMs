#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def compute_l2_norm_error(U_FOM, U_ROM):
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)

def gif_fom_local_global(
    mu1=5.190, mu2=0.0260,
    fem_base="../../FEM/fem_testing_data",
    local_file="local_PROM_20_clusters_LSPG_mu1_5.190_mu2_0.0260.npy",
    global_file="../../POD/Results_thesis/rom_solutions/U_PROM_tol_4e-03_mu1_5.190_mu2_0.0260_lspg.npy",
    out_path="fom_local_vs_global_pod.gif",
    a=0.0, b=100.0, m=511,
    At=0.05, fps=30, dpi=90, frame_stride=1,
    fig_size=(7, 6)
):
    # ---- Load data ----
    tag_mu1 = f"{mu1:.3f}"
    tag_mu2 = f"{mu2:.4f}"

    fom_filename = os.path.join(fem_base, f"fem_simulation_mu1_{tag_mu1}_mu2_{tag_mu2}.npy")
    U_FOM = np.load(fom_filename)                                # (Nx, Nt+1)
    U_LOC = np.load(local_file)                                   # Local PROM (LSPG)
    U_GLO = np.load(global_file)                                  # Global PROM (LSPG)

    # Spatial grid (from m)
    X = np.linspace(a, b, m + 1)

    # Ensure all have same time length
    n_frames_all = min(U_FOM.shape[1], U_LOC.shape[1], U_GLO.shape[1])
    frames = np.arange(0, n_frames_all, frame_stride, dtype=int)

    # Dynamic y-limits across all three over chosen frames
    u_min = min(U_FOM[:, frames].min(), U_LOC[:, frames].min(), U_GLO[:, frames].min())
    u_max = max(U_FOM[:, frames].max(), U_LOC[:, frames].max(), U_GLO[:, frames].max())
    pad = 0.10 * max(1e-12, (u_max - u_min))
    y_min, y_max = u_min - pad, u_max + pad

    # Errors (optional print)
    err_loc = compute_l2_norm_error(U_FOM, U_LOC)
    err_glo = compute_l2_norm_error(U_FOM, U_GLO)
    print(f"[mu1={mu1:.3f}, mu2={mu2:.4f}]  L2 Local={err_loc:.6e}  L2 Global={err_glo:.6e}")

    # ---- Figure & artists ----
    fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    ax.set_xlim(a, b); ax.set_ylim(y_min, y_max)
    ax.grid(True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")

    # Lines
    line_fom, = ax.plot(X, U_FOM[:, frames[0]], color='black', lw=2, label="FOM")
    line_loc, = ax.plot(X, U_LOC[:, frames[0]], color='blue',  lw=2,
                        label=r'Local PROM (20 clusters, avg $n\approx17.5$)')
    line_glo, = ax.plot(X, U_GLO[:, frames[0]], color='red',   lw=2,
                        label=r'Global PROM ($n=17$)')

    ax.legend(loc='upper right')

    def set_title_for_k(k):
        t = k * At
        ax.set_title(rf'$\mu_1={mu1:.3f},\ \mu_2={mu2:.4f}\quad t={t:.2f}\,\mathrm{{s}}$')

    set_title_for_k(frames[0])

    def update(i):
        k = frames[i]
        line_fom.set_ydata(U_FOM[:, k])
        line_loc.set_ydata(U_LOC[:, k])
        line_glo.set_ydata(U_GLO[:, k])
        set_title_for_k(k)
        return line_fom, line_loc, line_glo

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[ok] saved → {out_path}")

if __name__ == "__main__":
    gif_fom_local_global(
        mu1=5.190, mu2=0.0260,
        # paths below already match your script; change if needed:
        fem_base="../../FEM/fem_testing_data",
        local_file="local_PROM_20_clusters_LSPG_mu1_5.190_mu2_0.0260.npy",
        global_file="../../POD/Results_thesis/rom_solutions/U_PROM_tol_4e-03_mu1_5.190_mu2_0.0260_lspg.npy",
        out_path="fom_local_vs_global_pod_mu1_5.190_mu2_0.0260.gif",
        At=0.05,
        fps=50, dpi=90,
        frame_stride=1,   # try 2–3 to shrink file size
        fig_size=(7, 6)
    )
