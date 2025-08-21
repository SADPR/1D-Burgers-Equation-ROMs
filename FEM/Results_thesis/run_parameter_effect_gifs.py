#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# LaTeX formatting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ---------- helpers ----------
def _load_case(mu1, mu2, data_dir):
    fn = f"{data_dir}/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    if not os.path.isfile(fn):
        print(f"[skip] missing: {fn}")
        return None
    return np.load(fn)

def _animate_comparison(X, U_list, labels, colors, title_prefix, out_path,
                        At=0.05, ylim=(0.5, 7.5), fps=30, dpi=90):
    # All arrays shape: (Nx, Nt+1). Use min common frames just in case.
    n_frames = min(U.shape[1] for U in U_list)

    fig, ax = plt.subplots(figsize=(7, 5))
    lines = []
    for lab, col, U in zip(labels, colors, U_list):
        (ln,) = ax.plot(X, U[:, 0], lw=2, color=col, label=lab)
        lines.append(ln)

    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x,t)$')
    ax.grid(True)
    ax.legend(loc='upper right')

    def update(frame):
        for ln, U in zip(lines, U_list):
            ln.set_ydata(U[:, frame])
        t = frame * At
        ax.set_title(rf'{title_prefix} \quad $t={t:.2f}\,\mathrm{{s}}$')  # ← wrap in $...$
        return tuple(lines)

    ani = FuncAnimation(fig, update, frames=n_frames, blit=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[ok] saved → {out_path}")

# ---------- main API ----------
def compare_mu2_gif(mu1_fixed=4.875,
                    mu2_values=(0.0150, 0.0225, 0.0300),
                    data_dir="fem_training_data", out_dir="fem_training_gifs",
                    At=0.05, a=0.0, b=100.0, m=int(256*2),
                    colors=('red', 'blue', 'green'), fps=30, dpi=90):
    X = np.linspace(a, b, m + 1)
    U_list, labels, cols = [], [], []
    for mu2, col in zip(mu2_values, colors):
        U = _load_case(mu1_fixed, mu2, data_dir)
        if U is None: continue
        U_list.append(U); labels.append(rf'$\mu_2={mu2:.4f}$'); cols.append(col)
    if not U_list:
        print("[warn] no data found for compare_mu2_gif"); return
    out = f"{out_dir}/compare_mu2_mu1_{mu1_fixed:.3f}.gif"
    title = rf'Fixed $\mu_1={mu1_fixed:.3f}$; varying $\mu_2$'
    _animate_comparison(X, U_list, labels, cols, title, out, At=At, fps=fps, dpi=dpi)

def compare_mu1_gif(mu2_fixed=0.0225,
                    mu1_values=(4.250, 4.875, 5.500),
                    data_dir="fem_training_data", out_dir="fem_training_gifs",
                    At=0.05, a=0.0, b=100.0, m=int(256*2),
                    colors=('red', 'blue', 'green'), fps=30, dpi=90):
    X = np.linspace(a, b, m + 1)
    U_list, labels, cols = [], [], []
    for mu1, col in zip(mu1_values, colors):
        U = _load_case(mu1, mu2_fixed, data_dir)
        if U is None: continue
        U_list.append(U); labels.append(rf'$\mu_1={mu1:.3f}$'); cols.append(col)
    if not U_list:
        print("[warn] no data found for compare_mu1_gif"); return
    out = f"{out_dir}/compare_mu1_mu2_{mu2_fixed:.4f}.gif"
    title = rf'Fixed $\mu_2={mu2_fixed:.4f}$; varying $\mu_1$'
    _animate_comparison(X, U_list, labels, cols, title, out, At=At, fps=fps, dpi=dpi)

# ---------- run both (3×3 defaults) ----------
if __name__ == "__main__":
    # Defaults match your training grid and directories:
    compare_mu2_gif(mu1_fixed=4.250,
                    mu2_values=(0.0150, 0.0225, 0.0300),
                    data_dir="simulation_results",
                    out_dir="fem_training_gifs",
                    At=0.05, fps=50, dpi=90)

    compare_mu1_gif(mu2_fixed=0.0150,
                    mu1_values=(4.250, 4.875, 5.500),
                    data_dir="simulation_results",
                    out_dir="fem_training_gifs",
                    At=0.05, fps=50, dpi=90)
