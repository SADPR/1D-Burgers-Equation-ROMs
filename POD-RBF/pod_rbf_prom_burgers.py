#!/usr/bin/env python3
# POD–RBF PROM online solver for 1D Burgers (same style as POD-ANN script)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.animation import FuncAnimation, PillowWriter

# ----------------- FEM path (same as POD-ANN script) -----------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../FEM/'))
sys.path.append(parent_dir)

from fem_burgers import FEMBurgers  # noqa: E402

# ----------------- RBF artifact loading utilities --------------------
def load_rbf_artifacts(rbf_dir):
    # 1) POD bases (primary / secondary)
    U_p = np.load(os.path.join(rbf_dir, "Phi_primary.npy"))
    U_s = np.load(os.path.join(rbf_dir, "Phi_secondary.npy"))

    # 2) X_train (scaled)
    xtrain_txt = os.path.join(rbf_dir, "rbf_xTrain.txt")
    with open(xtrain_txt, "r") as f:
        Ns, n = map(int, f.readline().split())
    X_train = np.loadtxt(xtrain_txt, skiprows=1)
    if X_train.ndim == 1:
        X_train = X_train[None, :]
    assert X_train.shape == (Ns, n), f"X_train shape {X_train.shape} != ({Ns}, {n})"

    # 3) RBF weights W
    precomp_txt = os.path.join(rbf_dir, "rbf_precomputations.txt")
    with open(precomp_txt, "r") as f:
        W_rows, W_cols = map(int, f.readline().split())
    W = np.loadtxt(precomp_txt, skiprows=1)
    if W.ndim == 1:
        W = W[:, None]
    assert W.shape == (W_rows, W_cols), f"W shape {W.shape} != ({W_rows}, {W_cols})"

    # 4) Scaling parameters
    stdscale_txt = os.path.join(rbf_dir, "rbf_stdscaling.txt")
    with open(stdscale_txt, "r") as f:
        in_size, out_size = map(int, f.readline().split())
        scalingMethod = int(f.readline().strip())
    if scalingMethod != 1:
        raise ValueError("Expected min–max scaling method (1) in rbf_stdscaling.txt.")

    x_min = np.loadtxt(stdscale_txt, skiprows=2, max_rows=1)
    x_max = np.loadtxt(stdscale_txt, skiprows=3, max_rows=1)
    y_min = np.loadtxt(stdscale_txt, skiprows=4, max_rows=1)
    y_max = np.loadtxt(stdscale_txt, skiprows=5, max_rows=1)

    assert in_size == X_train.shape[1], "in_size mismatch in scaling file"
    assert out_size == W.shape[1], "out_size mismatch in scaling file"

    # 5) Kernel + epsilon
    hyper_txt = os.path.join(rbf_dir, "rbf_hyper.txt")
    with open(hyper_txt, "r") as f:
        _hdr = f.readline().strip()
        kernel_name = f.readline().strip()
        epsilon = float(f.readline().strip())

    return U_p, U_s, X_train, W, epsilon, x_min, x_max, y_min, y_max, kernel_name


if __name__ == "__main__":
    # ----------------- Domain and mesh (same as POD-ANN) ---------------
    a, b = 0.0, 100.0
    m = 511
    h = (b - a) / m
    X = np.linspace(a, b, m + 1)
    T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

    # Initial condition
    u0 = np.ones_like(X)

    # Boundary conditions
    mu1 = 4.750  # u(0,t) = mu1

    # Time discretization and numerical diffusion
    Tf = 25.0
    At = 0.05
    nTimeSteps = int(Tf / At) 
    E = 0.00

    # Parameter mu2
    mu2 = 0.0200

    # Create FEM solver
    fem_burgers = FEMBurgers(X, T)

    # ----------------- Load RBF artifacts ------------------------------
    rbf_dir = "rbf_training_simple"   # same out_dir as in train_prom_rbf.py
    print(f"[info] Loading RBF artifacts from: {rbf_dir}")

    (U_p, U_s,
     X_train, W, epsilon,
     x_min, x_max, y_min, y_max,
     kernel_name) = load_rbf_artifacts(rbf_dir)

    print(f"[info] U_p: {U_p.shape}, U_s: {U_s.shape}")
    print(f"[info] X_train: {X_train.shape}, W: {W.shape}")
    print(f"[info] kernel={kernel_name}, epsilon={epsilon:.6e}")

    # ----------------- POD–RBF PROM solve ------------------------------
    print("POD–RBF PROM method...")
    import time
    start = time.time()

    U_POD_RBF_PROM = fem_burgers.pod_rbf_prom(
        At, nTimeSteps,
        u0, mu1, E, mu2,
        U_p, U_s,
        X_train, W, epsilon,
        x_min, x_max, y_min, y_max,
        projection="LSPG",
        kernel=kernel_name,
        tol_newton=1e-6,
        max_newton=20
    )

    end = time.time()
    print(f"[timing] POD–RBF PROM time: {end - start:.3f} s")

        # ----------------- Save solution -----------------------------------
    out_dir = "pod_rbf_prom_solutions"
    os.makedirs(out_dir, exist_ok=True)

    n_ret = U_p.shape[1]   # retained modes  (n)
    n_dis = U_s.shape[1]   # discarded modes (n̄)

    fname = (
        f"POD_RBF_PROM_U_n{n_ret}_nb{n_dis}"
        f"_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    )
    out_path = os.path.join(out_dir, fname)
    np.save(out_path, U_POD_RBF_PROM)

    print(f"[saved] POD–RBF PROM solution -> {out_path}")

    # --------------------------------------------------------------
    # FOM vs POD–RBF PROM: overlay plot + error (NO second ROM file)
    # --------------------------------------------------------------
    fom_dir = os.path.join(parent_dir, "fem_testing_data")
    fom_file = f"fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    fom_path = os.path.join(fom_dir, fom_file)

    if not os.path.isfile(fom_path):
        print(f"[warn] FOM file not found, skipping overlay plot: {fom_path}")
    else:
        U_FOM = np.load(fom_path)   # shape (N, T_FOM)

        # Align time horizon
        min_T = min(U_FOM.shape[1], U_POD_RBF_PROM.shape[1])
        U_FOM = U_FOM[:, :min_T]
        U_ROM = U_POD_RBF_PROM[:, :min_T]

        # 1) Frobenius trajectory error
        num = np.linalg.norm(U_FOM - U_ROM, ord='fro')
        den = np.linalg.norm(U_FOM, ord='fro')
        rel_frob = (num / den) if den > 0 else np.nan

        print(f"\n=== POD–RBF PROM metrics (mu1={mu1:.3f}, mu2={mu2:.4f}) ===")
        print(f"  Projection: LSPG, kernel: {kernel_name}, ε={epsilon:.4f}")
        print(f"  Frobenius relative error (trajectory): {100*rel_frob:.4f}%")

        # 2) Overlay at multiple times
        snapshot_times = [5, 10, 15, 20, 25]  # seconds
        snapshot_indices = [
            min(int(t / At), min_T - 1) for t in snapshot_times
        ]

        os.makedirs("figs_rom_vs_fom", exist_ok=True)

        plt.figure(figsize=(8, 5))
        for t_idx, t_val in zip(snapshot_indices, snapshot_times):
            plt.plot(
                X, U_FOM[:, t_idx], color='k', linewidth=1.5,
                label="HDM" if t_idx == snapshot_indices[0] else ""
            )
            plt.plot(
                X, U_ROM[:, t_idx], '--', color='b', linewidth=1.5,
                label="POD–RBF PROM (LSPG)"
                if t_idx == snapshot_indices[0] else ""
            )
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(
            f"Overlay (mu1={mu1:.3f}, mu2={mu2:.4f}) | "
            f"{kernel_name}, ε={epsilon:.3f}"
        )
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(ncol=2)

        multi_pdf = os.path.join(
            "figs_rom_vs_fom",
            f"overlay_times_PODRBF_lspg_{kernel_name}_eps_{epsilon:.4f}_"
            f"mu1_{mu1:.3f}_mu2_{mu2:.4f}.pdf"
        )
        plt.savefig(multi_pdf, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"[saved] Overlay figure -> {multi_pdf}")


