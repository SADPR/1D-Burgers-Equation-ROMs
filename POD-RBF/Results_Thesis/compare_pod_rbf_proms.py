
#!/usr/bin/env python3
"""
compare_pod_rbf_prom_single.py
-------------------------------------------------------
Single-case comparison for:

    μ1 = 4.560, μ2 = 0.0190
    POD–RBF PROM (n=17, nbar=79)
    Global PROM (n=17 and n=96)

Colours / linestyles
------------------------------
FOM                : black  solid
POD–RBF PROM       : blue   solid
Global-POD same-n  : green  dashed (n=17)
Global-POD (n=96)  : red    dashed
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# ------------------------------------------------------------------ #
# 1. user settings
# ------------------------------------------------------------------ #
mu1 = 4.560
mu2 = 0.0190

At = 0.05
times_to_plot = [5, 10, 15, 20, 25]  # seconds → indices via At

# FOM data (from FEM)
fom_dir = "../../FEM/fem_testing_data"
fom_file = f"fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"

# POD–RBF PROM solution
rbf_dir = "../pod_rbf_prom_solutions"
rbf_file = f"POD_RBF_PROM_U_n17_nb79_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"

# Global PROM solutions
glob_files = {
    17: "../../POD/Results_thesis/rom_solutions/U_PROM_tol_4e-03_mu1_{:.3f}_mu2_{:.4f}_lspg.npy",
    96: "../../POD/Results_thesis/rom_solutions/U_PROM_tol_1e-04_mu1_{:.3f}_mu2_{:.4f}_lspg.npy",
}

# Line styles for Global PROMs
line_styles = {
    17: ("g", "--"),  # same-n: green dashed
    96: ("r", "--"),  # n=96: red dashed
}

out_dir = "pod_rbf_comparison_figures"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# 2. helpers
# ------------------------------------------------------------------ #
def rel_err(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(A)


# ------------------------------------------------------------------ #
# 3. load data
# ------------------------------------------------------------------ #
print(f"\n=== μ1={mu1:.3f}, μ2={mu2:.4f} ===")

# FOM
fom_path = os.path.join(fom_dir, fom_file)
if not os.path.isfile(fom_path):
    raise FileNotFoundError(f"FOM file not found: {fom_path}")
fom = np.load(fom_path)  # shape: (Nx, Nt)

# Space and time setup
Nx, Nt = fom.shape
X = np.linspace(0, 100, Nx)
idx = [min(int(t / At), Nt - 1) for t in times_to_plot]

# POD–RBF PROM
rbf_path = os.path.join(rbf_dir, rbf_file)
if not os.path.isfile(rbf_path):
    raise FileNotFoundError(f"POD–RBF PROM file not found: {rbf_path}")
rbf = np.load(rbf_path)[:, :Nt]

print(f"   POD–RBF PROM: {rbf.shape}, FOM: {fom.shape}")

# ------------------------------------------------------------------ #
# 4. loop over global PROM n = 17,96
# ------------------------------------------------------------------ #
for n_glob in (17, 96):
    glob_path = glob_files[n_glob].format(mu1, mu2)
    if not os.path.isfile(glob_path):
        print(f"  ( Global PROM n={n_glob} not found: {glob_path} )")
        continue

    glob = np.load(glob_path)[:, :Nt]
    print(f"   Global PROM n={n_glob}: {glob.shape}")

    col_g, ls_g = line_styles[n_glob]

    # ---- plot ----
    plt.figure(figsize=(7, 6))

    # FOM
    plt.plot(X, fom[:, idx[0]], "k-", lw=2, label="FOM")
    for j in idx[1:]:
        plt.plot(X, fom[:, j], "k-", lw=2)

    # POD–RBF PROM
    plt.plot(
        X, rbf[:, idx[0]], "b-", lw=2,
        label=r"POD--RBF PROM ($n=17,\;\bar{n}=79$)"
    )
    for j in idx[1:]:
        plt.plot(X, rbf[:, j], "b-", lw=2)

    # Global PROM
    plt.plot(
        X, glob[:, idx[0]], ls_g, color=col_g, lw=2,
        label=rf"Global PROM ($n={n_glob}$)"
    )
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
        f"fom_vs_rbf17_nb79_vs_global{n_glob}_mu1_{mu1:.3f}_mu2_{mu2:.4f}.pdf"
    )
    plt.savefig(pdf)
    plt.close()
    print("  saved →", pdf)

    # ---- errors ----
    print(f"   Global PROM {n_glob:2d}   L2 error = {rel_err(fom, glob):.3e}")

# ------------------------------------------------------------------ #
# 5. print POD–RBF PROM error
# ------------------------------------------------------------------ #
print(f"   POD–RBF PROM  n=17 nb=79  L2 error = {rel_err(fom, rbf):.3e}")

