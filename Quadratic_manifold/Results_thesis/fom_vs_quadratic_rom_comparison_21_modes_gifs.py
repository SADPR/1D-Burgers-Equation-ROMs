#!/usr/bin/env python3
"""
compare_quadratic_proms_gif.py
-------------------------------------------------------
FOM vs Quadratic PROM (+ Local & Global POD, optional)
Generates one GIF per parameter pair.
"""

import numpy as np, matplotlib.pyplot as plt, os, re
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# --------------------------------------------------- #
# 1. user settings
# --------------------------------------------------- #
cases = [
    (4.750, 0.0200),
    (4.560, 0.0190),
    (5.190, 0.0260),
]

At = 0.05
fom_dir  = "../../FEM/fem_testing_data"
qrom_dir = "../quadratic_rom_solutions"
local_pattern = "local_PROM_20_clusters_LSPG_mu1_{:.3f}_mu2_{:.4f}.npy"
global_file   = "../../POD/Results_thesis/rom_solutions/U_PROM_tol_3e-03_mu1_{:.3f}_mu2_{:.4f}_lspg.npy"

out_dir = "comparison_gifs"
os.makedirs(out_dir, exist_ok=True)

# --------------------------------------------------- #
# 2. helpers
# --------------------------------------------------- #
def load_qprom(mu1, mu2):
    pat = rf"quadratic_PROM_U_PROM_(\d+)_modes_mu1_{mu1:.3f}_mu2_{mu2:.4f}\.npy"
    for f in os.listdir(qrom_dir):
        m = re.match(pat, f)
        if m:
            n = int(m.group(1))
            return np.load(os.path.join(qrom_dir, f)), n
    raise FileNotFoundError("No QPROM file found for this parameter set")

# --------------------------------------------------- #
# 3. main loop
# --------------------------------------------------- #
for mu1, mu2 in cases:
    print(f"\n=== case μ1={mu1:.3f}, μ2={mu2:.4f} ===")

    # load data
    fom = np.load(os.path.join(fom_dir,
        f"fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"))
    qrom, n_q = load_qprom(mu1, mu2)

    try:
        local = np.load(local_pattern.format(mu1, mu2))
    except FileNotFoundError:
        local = None

    try:
        global_pod = np.load(global_file.format(mu1, mu2))
    except FileNotFoundError:
        global_pod = None

    # grid
    X = np.linspace(0, 100, fom.shape[0])
    n_frames = fom.shape[1]

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    line_fom, = ax.plot(X, fom[:, 0], "k-", lw=2, label="FOM")
    line_qrom, = ax.plot(X, qrom[:, 0], "g-", lw=2, label=rf"QPROM ($n={n_q}$)")
    line_local, = ax.plot([], [], "b-", lw=2, label="Local PROM") if local is not None else (None,)
    line_global, = ax.plot([], [], "r-", lw=2, label=rf"Global PROM ($n={n_q}$)") if global_pod is not None else (None,)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$")
    ax.set_title(rf"$\mu_1={mu1:.3f},\;\mu_2={mu2:.4f}$")
    ax.grid(True)
    ax.legend(loc="upper right")

    def update(frame):
        line_fom.set_ydata(fom[:, frame])
        line_qrom.set_ydata(qrom[:, frame])
        if local is not None: line_local.set_data(X, local[:, frame])
        if global_pod is not None: line_global.set_data(X, global_pod[:, frame])
        return line_fom, line_qrom, line_local, line_global

    ani = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=50)
    gif_name = os.path.join(out_dir, f"fom_vs_qprom_mu1_{mu1:.3f}_mu2_{mu2:.4f}.gif")
    ani.save(gif_name, writer=PillowWriter(fps=50), dpi=90)
    plt.close(fig)

    print(" saved gif →", gif_name)
