#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------ STYLE ------------------
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# ------------------ HELPERS ----------------
def rel_err(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(A)

def _find_ann_file(ann_dir, mu1, mu2, n_ret, n_bar):
    pat = rf"POD_ANN_PROM_U_n{n_ret}_nb{n_bar}_mu1_{mu1:.3f}_mu2_{mu2:.4f}\.npy"
    for f in os.listdir(ann_dir):
        if re.match(pat, f):
            return os.path.join(ann_dir, f)
    return None

def _dynamic_ylim(arrs, frames):
    vmin = min(a[:, frames].min() for a in arrs)
    vmax = max(a[:, frames].max() for a in arrs)
    pad  = 0.10 * max(1e-12, (vmax - vmin))
    return vmin - pad, vmax + pad

def _make_one_gif(X, U_fom, U_ann, U_glob, mu1, mu2,
                  n_ret, n_bar, n_glob,
                  out_gif, At=0.05, fps=40, dpi=90, frame_stride=1,
                  fig_size=(7, 6)):
    # Align lengths
    n_frames_all = min(U_fom.shape[1], U_ann.shape[1], U_glob.shape[1])
    frames = np.arange(0, n_frames_all, frame_stride, dtype=int)

    # Y limits across all three
    y_min, y_max = _dynamic_ylim([U_fom, U_ann, U_glob], frames)

    # Colors / styles (as in your figure script)
    fom_color = "k"
    ann_color = "b"
    if n_glob in (17, 5):
        glob_color, glob_ls = "g", "--"  # same-n compared to PROM-ANN
    else:
        glob_color, glob_ls = "g", "--"  # n=96

    # --- Figure & artists ---
    fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(y_min, y_max)
    ax.grid(True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")

    # Initial lines
    line_fom, = ax.plot(X, U_fom[:, frames[0]], color=fom_color, lw=2, label="FOM")
    line_ann, = ax.plot(X, U_ann[:, frames[0]], color=ann_color, lw=2,
                        label=rf"PROM-ANN ($n={n_ret},\;\bar{{n}}={n_bar}$)")
    line_glb, = ax.plot(X, U_glob[:, frames[0]], color=glob_color, ls=glob_ls, lw=2,
                        label=rf"Global PROM ($n={n_glob}$)")

    def _title_for_k(k):
        t = k * At
        ax.set_title(rf"$\mu_1={mu1:.3f},\ \mu_2={mu2:.4f}\quad t={t:.2f}\,\mathrm{{s}}$")

    _title_for_k(frames[0])
    ax.legend(loc="upper right")

    # --- Animator ---
    def update(i):
        k = frames[i]
        line_fom.set_ydata(U_fom[:, k])
        line_ann.set_ydata(U_ann[:, k])
        line_glb.set_ydata(U_glob[:, k])
        _title_for_k(k)
        return line_fom, line_ann, line_glb

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[ok] saved → {out_gif}")

# ------------------ MAIN DRIVER ------------------
def make_prom_ann_gifs(
    mu1, mu2,
    # data locations (match your repo layout)
    fom_dir="../../FEM/fem_testing_data",
    ann_dir="../pod_ann_prom_solutions",
    glob_patterns={
        96: "../../POD/Results_thesis/rom_solutions/U_PROM_tol_1e-04_mu1_{:.3f}_mu2_{:.4f}_lspg.npy",
        17: "../../POD/Results_thesis/rom_solutions/U_PROM_tol_4e-03_mu1_{:.3f}_mu2_{:.4f}_lspg.npy",
        5:  "../../POD/Results_thesis/rom_solutions/U_PROM_tol_2e-02_mu1_{:.3f}_mu2_{:.4f}_lspg.npy",
    },
    # simulation/time info
    a=0.0, b=100.0, m=None,  # if m is None, infer from FOM
    At=0.05,
    # animation options
    fps=40, dpi=90, frame_stride=1,
    fig_size=(7, 6),
    out_dir="prom_ann_gifs",
    # the four comparisons you want
    ann_settings=((17, 79), (5, 91)),
    glob_to_compare={ (17,79): (17, 96), (5,91): (5, 96) },
):
    os.makedirs(out_dir, exist_ok=True)

    # Load FOM
    fom_path = os.path.join(fom_dir, f"fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
    U_fom = np.load(fom_path)  # shape (Nx, Nt+1)

    # Spatial grid
    Nx = U_fom.shape[0]
    if m is None:
        m = Nx - 1
    X = np.linspace(a, b, m + 1)

    # Loop PROM-ANN settings and global references
    for (n_ret, n_bar) in ann_settings:
        ann_path = _find_ann_file(ann_dir, mu1, mu2, n_ret, n_bar)
        if ann_path is None:
            print(f"[warn] PROM-ANN file not found for n={n_ret}, nbar={n_bar} (mu1={mu1:.3f}, mu2={mu2:.4f})")
            continue

        U_ann = np.load(ann_path)
        # Align time length to FOM
        U_ann = U_ann[:, :U_fom.shape[1]]

        # Errors vs FOM (optional print)
        print(f"PROM-ANN (n={n_ret}, nbar={n_bar})  L2 error = {rel_err(U_fom, U_ann):.3e}")

        for n_glob in glob_to_compare[(n_ret, n_bar)]:
            glob_path = glob_patterns[n_glob].format(mu1, mu2)
            if not os.path.isfile(glob_path):
                print(f"[warn] Global PROM (n={n_glob}) not found for mu=({mu1:.3f},{mu2:.4f})")
                continue

            U_glb = np.load(glob_path)[:, :U_fom.shape[1]]
            print(f"Global PROM (n={n_glob:2d}) L2 error = {rel_err(U_fom, U_glb):.3e}")

            out_gif = os.path.join(
                out_dir,
                f"fom_vs_ann{n_ret}_nb{n_bar}_vs_global{n_glob}_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif"
            )

            _make_one_gif(
                X=X, U_fom=U_fom, U_ann=U_ann, U_glob=U_glb,
                mu1=mu1, mu2=mu2,
                n_ret=n_ret, n_bar=n_bar, n_glob=n_glob,
                out_gif=out_gif, At=At, fps=fps, dpi=dpi,
                frame_stride=frame_stride, fig_size=fig_size
            )

if __name__ == "__main__":
    # Example: make the four GIFs for μ = (5.190, 0.0260).
    # Change mu1, mu2 or call this multiple times for other test points.
    make_prom_ann_gifs(
        mu1=5.190, mu2=0.0260,
        At=0.05,
        fps=50, dpi=90,
        frame_stride=1,            # increase to 2–3 for smaller files
        fig_size=(7, 6),
        out_dir="prom_ann_gifs"
    )
