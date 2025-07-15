import numpy as np
import os

# ---------- helper: integrated (root-sum-squared) relative error ----------
def integrated_rel_error(U_FOM, U_ROM):
    """
    Integrated relative L2 error:
    || U_FOM - U_ROM ||_F / || U_FOM ||_F   (Frobenius norms).
    """
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)

# ---------- test points (mu1, mu2) ----------
test_points = [
    (4.750, 0.0200),
    (4.560, 0.0190),
    (5.190, 0.0260)
]

# ---------- table header ----------
print(f"{'mu1':>8} {'mu2':>8} {'Global LSPG Int. L2 (%)':>28} {'Local LSPG Int. L2 (%)':>27}")
print("-" * 78)

# ---------- loop ----------
for mu1, mu2 in test_points:
    t1, t2 = f"{mu1:.3f}", f"{mu2:.4f}"

    # FOM path
    fom_path = f"../../FEM/fem_testing_data/fem_simulation_mu1_{t1}_mu2_{t2}.npy"
    U_FOM = np.load(fom_path)

    # Global PROM path
    global_path = f"../../POD/Results_thesis/rom_solutions/U_PROM_tol_4e-03_mu1_{t1}_mu2_{t2}_lspg.npy"
    U_ROM_global = np.load(global_path)

    # Local PROM path
    local_path = f"local_PROM_20_clusters_LSPG_mu1_{t1}_mu2_{t2}.npy"
    U_ROM_local = np.load(local_path)

    # Integrated errors (percentage)
    int_err_global = integrated_rel_error(U_FOM, U_ROM_global) * 100
    int_err_local  = integrated_rel_error(U_FOM, U_ROM_local)  * 100

    # print one row
    print(f"{mu1:8.3f} {mu2:8.4f} {int_err_global:28.2f} {int_err_local:27.2f}")
