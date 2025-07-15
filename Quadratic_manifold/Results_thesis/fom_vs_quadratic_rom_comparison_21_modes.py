#!/usr/bin/env python3
"""
compare_quadratic_proms.py
-------------------------------------------------------
FOM  vs  Quadratic PROM  (+ Local & Global POD, optional)
Generates one PDF per parameter pair and prints L2 errors.
"""

import numpy as np, matplotlib.pyplot as plt, os, re, sys
plt.rc("text", usetex=True);  plt.rc("font", family="serif")


# --------------------------------------------------- #
# 1.  user settings
# --------------------------------------------------- #
cases = [               # three test points
    (4.750, 0.0200),
    (4.560, 0.0190),
    (5.190, 0.0260),
]

At  = 0.05
times_to_plot = [5, 10, 15, 20, 25]           # [s]

# directories / naming patterns ----------------------
fom_dir   = "../../FEM/fem_testing_data"
qrom_dir  = "../quadratic_rom_solutions"

# optional (set to None if not available)
local_pattern  = "local_PROM_20_clusters_LSPG_mu1_{:.3f}_mu2_{:.4f}.npy"
global_file    = "../../POD/Results_thesis/rom_solutions/U_PROM_tol_3e-03_mu1_{:.3f}_mu2_{:.4f}_lspg.npy"

# pdf output folder
out_dir = "comparison_figures"
os.makedirs(out_dir, exist_ok=True)

# --------------------------------------------------- #
# 2.  helper funcs
# --------------------------------------------------- #
def rel_error(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(A)

def load_qprom(mu1, mu2):
    pat = rf"quadratic_PROM_U_PROM_(\d+)_modes_mu1_{mu1:.3f}_mu2_{mu2:.4f}\.npy"
    for f in os.listdir(qrom_dir):
        m = re.match(pat, f)
        if m:
            n = int(m.group(1))
            return np.load(os.path.join(qrom_dir, f)), n
    raise FileNotFoundError("No QPROM file found for this parameter set")

# --------------------------------------------------- #
# 3.  main loop over (mu1,mu2)
# --------------------------------------------------- #
for mu1, mu2 in cases:
    print(f"\n=== case  μ1={mu1:.3f}, μ2={mu2:.4f}  ===")

    # ---------- load data ----------
    fom = np.load(os.path.join(
        fom_dir, f"fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"))

    qrom, n_q = load_qprom(mu1, mu2)

    # Local POD
    try:
        local = np.load(local_pattern.format(mu1, mu2))
    except FileNotFoundError:
        local = None

    # Global POD
    try:
        gfile = global_file.format(mu1, mu2)
        global_pod = np.load(gfile)
    except FileNotFoundError:
        global_pod = None

    # ---------- common stuff ----------
    X = np.linspace(0, 100, fom.shape[0])
    idx = [int(t/At) for t in times_to_plot]

    # ---------- plot ----------
    plt.figure(figsize=(7,6))

    # FOM
    plt.plot(X, fom[:, idx[0]], "k-", lw=2, label="FOM")
    for j in idx[1:]: plt.plot(X, fom[:, j], "k-", lw=2)

    # QPROM
    plt.plot(X, qrom[:, idx[0]], "g-", lw=2,
             label=rf"QPROM ($n={n_q}$)")
    for j in idx[1:]: plt.plot(X, qrom[:, j], "g-", lw=2)

    # Local POD
    if local is not None:
        plt.plot(X, local[:, idx[0]], "b-", lw=2,
                 label="Local PROM")
        for j in idx[1:]: plt.plot(X, local[:, j], "b-", lw=2)

    # Global POD
    if global_pod is not None:
        plt.plot(X, global_pod[:, idx[0]], "r-", lw=2,
                 label=rf"Global PROM ($n={n_q}$)")
        for j in idx[1:]: plt.plot(X, global_pod[:, j], "r-", lw=2)

    plt.xlabel(r"$x$");  plt.ylabel(r"$u$")
    plt.title(rf"$\mu_1={mu1:.3f},\;\mu_2={mu2:.4f}$")
    plt.xlim(0,100); plt.ylim(0,8); plt.grid()
    plt.legend()
    plt.tight_layout()

    pdf_name = os.path.join(out_dir,
        f"fom_vs_qprom_mu1_{mu1:.3f}_mu2_{mu2:.4f}.pdf")
    plt.savefig(pdf_name)
    print(" saved figure →", pdf_name)
    plt.close()

    # ---------- errors ----------
    print(f"   Quadratic PROM  L2 error = {rel_error(fom, qrom):.3e}")
    if local is not None:
        print(f"   Local POD PROM  L2 error = {rel_error(fom, local):.3e}")
    if global_pod is not None:
        print(f"   Global POD PROM L2 error = {rel_error(fom, global_pod):.3e}")
