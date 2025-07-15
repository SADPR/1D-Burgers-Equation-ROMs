#!/usr/bin/env python3
"""
compare_pod_ann_proms.py
-------------------------------------------------------
For every (mu1, mu2) test case it creates four plots:

   • FOM | PROM–ANN (n=17, nbar=79) | Global PROM (n=17)
   • FOM | PROM–ANN (n=17, nbar=79) | Global PROM (n=96)
   • FOM | PROM–ANN (n=5,  nbar=91) | Global PROM (n=5)
   • FOM | PROM–ANN (n=5,  nbar=91) | Global PROM (n=96)

Colours / linestyles
------------------------------
FOM            : black  solid
PROM–ANN       : blue   solid
Global-POD same-n : green  dashed
Global-POD 96     : red    dashed
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# ------------------------------------------------------------------ #
# 1. user settings
# ------------------------------------------------------------------ #
cases = [
    (4.750, 0.0200),
    (4.560, 0.0190),
    (5.190, 0.0260),
]

At = 0.05
times_to_plot = [5, 10, 15, 20, 25]  # seconds → indices via At

fom_dir = "../../FEM/fem_testing_data"
ann_dir = "../pod_ann_prom_solutions"

glob_patterns = {
    96: "../../POD/Results_thesis/rom_solutions/U_PROM_tol_1e-04_mu1_{:.3f}_mu2_{:.4f}_lspg.npy",
    17: "../../POD/Results_thesis/rom_solutions/U_PROM_tol_4e-03_mu1_{:.3f}_mu2_{:.4f}_lspg.npy",
    5:  "../../POD/Results_thesis/rom_solutions/U_PROM_tol_2e-02_mu1_{:.3f}_mu2_{:.4f}_lspg.npy",
}

# PROM-ANN settings to find
ann_cases = [
    (17, 79),
    (5, 91)
]

# Global PROM n's to compare for each PROM-ANN
compare_pairs = {
    (17, 79): [17, 96],
    (5, 91): [5, 96],
}

# Line styles for Global PROMs
line_styles = {
    17: ("g", "--"),  # same-n: green dashed
    5:  ("g", "--"),
    96: ("r", "--"),  # n=96: red dashed
}

out_dir = "pod_ann_comparison_figures"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# 2. helpers
# ------------------------------------------------------------------ #
def rel_err(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(A)

# ------------------------------------------------------------------ #
# 3. main loop
# ------------------------------------------------------------------ #
for mu1, mu2 in cases:
    print(f"\n=== μ1={mu1:.3f}, μ2={mu2:.4f} ===")

    # Load FOM
    fom_path = os.path.join(fom_dir, f"fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
    fom = np.load(fom_path)

    # Space and time setup
    X = np.linspace(0, 100, fom.shape[0])
    idx = [int(t / At) for t in times_to_plot]

    for n_ret, n_bar in ann_cases:
        # Find PROM-ANN file
        pat = rf"POD_ANN_PROM_U_n{n_ret}_nb{n_bar}_mu1_{mu1:.3f}_mu2_{mu2:.4f}\.npy"
        found = False
        for f in os.listdir(ann_dir):
            if re.match(pat, f):
                ann = np.load(os.path.join(ann_dir, f))[:, :fom.shape[1]]
                found = True
                break
        if not found:
            print(f"  ( PROM-ANN n={n_ret} nbar={n_bar} not found )")
            continue

        # For this PROM-ANN, make two comparisons
        for n_glob in compare_pairs[(n_ret, n_bar)]:
            try:
                glob = np.load(glob_patterns[n_glob].format(mu1, mu2))[:, :fom.shape[1]]
            except FileNotFoundError:
                print(f"  ( Global PROM n={n_glob} not found )")
                continue

            col_g, ls_g = line_styles[n_glob]

            # ---- plot ----
            plt.figure(figsize=(7, 6))

            # FOM
            plt.plot(X, fom[:, idx[0]], "k-", lw=2, label="FOM")
            for j in idx[1:]:
                plt.plot(X, fom[:, j], "k-", lw=2)

            # PROM-ANN
            plt.plot(X, ann[:, idx[0]], "b-", lw=2,
                     label=rf"PROM-ANN ($n={n_ret},\;\bar{{n}}={n_bar}$)")
            for j in idx[1:]:
                plt.plot(X, ann[:, j], "b-", lw=2)

            # Global PROM
            plt.plot(X, glob[:, idx[0]], ls_g, color=col_g, lw=2,
                     label=rf"Global PROM ($n={n_glob}$)")
            for j in idx[1:]:
                plt.plot(X, glob[:, j], ls_g, color=col_g, lw=2)

            plt.xlabel(r"$x$")
            plt.ylabel(r"$u$")
            plt.title(rf"$\mu_1={mu1:.3f},\;\mu_2={mu2:.4f}$")
            plt.xlim(0, 100)
            plt.ylim(0, 8)
            plt.grid()
            plt.legend()
            plt.tight_layout()

            pdf = os.path.join(
                out_dir,
                f"fom_vs_ann{n_ret}_nb{n_bar}_vs_global{n_glob}_mu1_{mu1:.3f}_mu2_{mu2:.4f}.pdf"
            )
            plt.savefig(pdf)
            plt.close()
            print("  saved →", pdf)

            # ---- errors ----
            print(f"   Global PROM {n_glob:2d}   L2 error = {rel_err(fom, glob):.3e}")

        print(f"   PROM-ANN   n={n_ret:2d} nb={n_bar:2d}  L2 error = {rel_err(fom, ann):.3e}")

