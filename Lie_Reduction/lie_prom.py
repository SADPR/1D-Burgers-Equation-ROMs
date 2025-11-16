#!/usr/bin/env python3
# Lie PROM online solver for 1D Burgers (same style as POD–RBF script)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from matplotlib.animation import FuncAnimation, PillowWriter  # kept for consistency, not strictly needed

# ----------------- FEM path (same as POD-ANN / POD-RBF) -----------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../FEM/'))
sys.path.append(parent_dir)

from fem_burgers import FEMBurgers  # noqa: E402


# ----------------- Lie artifact loading utilities -----------------------
def load_lie_artifacts(lie_dir):
    """
    Load all offline artifacts needed for the Lie PROM.

    Expected files (from build_lie_prom_artifacts.py):
        - refs_indices.npy          : reference snapshot index per cluster
        - U_global.npy              : global POD basis (N x r_g) used for clustering
        - kmeans_lie.pkl            : trained sklearn KMeans model
        - u_ref_cluster_c.npy       : reference snapshot for cluster c
    """
    # 1) Reference indices per cluster
    refs_path = os.path.join(lie_dir, "refs_indices.npy")
    if not os.path.isfile(refs_path):
        raise FileNotFoundError(f"refs_indices.npy not found in {lie_dir}")
    refs_indices = np.load(refs_path, allow_pickle=True)

    # 2) Global POD basis
    Uglob_path = os.path.join(lie_dir, "U_global.npy")
    if not os.path.isfile(Uglob_path):
        raise FileNotFoundError(f"U_global.npy not found in {lie_dir}")
    U_global = np.load(Uglob_path)

    # 3) KMeans object
    kmeans_path = os.path.join(lie_dir, "kmeans_lie.pkl")
    if not os.path.isfile(kmeans_path):
        raise FileNotFoundError(f"kmeans_lie.pkl not found in {lie_dir}")
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    # 4) Reference snapshot per cluster
    u_refs = []
    for c, idx in enumerate(refs_indices):
        ref_file = os.path.join(lie_dir, f"u_ref_cluster_{c}.npy")
        if idx is None or (not os.path.isfile(ref_file)):
            print(f"[info] Cluster {c}: no reference file (idx={idx}), setting u_refs[{c}] = None")
            u_refs.append(None)
        else:
            u_ref_c = np.load(ref_file)
            u_refs.append(u_ref_c)

    print(f"[info] Loaded Lie artifacts from: {lie_dir}")
    print(f"       refs_indices: len={len(refs_indices)}")
    print(f"       U_global:    {U_global.shape}")
    print(f"       u_refs:      {sum(ur is not None for ur in u_refs)} non-empty clusters")

    return kmeans, refs_indices, u_refs, U_global


if __name__ == "__main__":
    # ----------------- Domain and mesh (same as POD-ANN / POD-RBF) ---------------
    a, b = 0.0, 100.0
    m = 511
    h = (b - a) / m
    X = np.linspace(a, b, m + 1)
    T = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

    # Initial condition
    u0 = np.ones_like(X)

    # Boundary condition mu1 = u(0,t)
    mu1 = 4.750

    # Time discretization and numerical diffusion
    Tf = 25.0
    At = 0.05
    nTimeSteps = int(Tf / At)
    E = 0.00

    # Parameter mu2
    mu2 = 0.0200

    # Create FEM solver
    fem_burgers = FEMBurgers(X, T)

    # ----------------- Load Lie artifacts ---------------------------------------
    lie_dir = "lie_cluster_GN_full5"   # same OUT_DIR as build_lie_prom_artifacts.py
    print(f"[info] Loading Lie artifacts from: {lie_dir}")

    kmeans, refs_indices, u_refs, U_global = load_lie_artifacts(lie_dir)

    # How many global modes to use for cluster assignment (you can tune this)
    num_global_modes = min(10, U_global.shape[1])

    # ----------------- Lie PROM solve ------------------------------------------
    print("Lie PROM method...")
    import time
    start = time.time()

    # NOTE: lie_prom signature should now take u_refs instead of S_snapshots
    U_LIE_PROM, g_hist = fem_burgers.lie_prom(
        At=At,
        nTimeSteps=nTimeSteps,
        u0=u0,
        mu1=mu1,
        E=E,
        mu2=mu2,
        kmeans=kmeans,
        refs_indices=refs_indices,
        u_refs=u_refs,
        U_global=U_global,
        num_global_modes=num_global_modes,
        projection="LSPG",    # or "Galerkin"
        tol_newton=1e-6,
        max_newton=20
    )

    end = time.time()
    print(f"[timing] Lie PROM time: {end - start:.3f} s")

    # ----------------- Save solution & Lie parameters --------------------------
    out_dir = "lie_prom_solutions"
    os.makedirs(out_dir, exist_ok=True)

    fname_u = f"Lie_PROM_U_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    fname_g = f"Lie_PROM_g_hist_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"

    out_u_path = os.path.join(out_dir, fname_u)
    out_g_path = os.path.join(out_dir, fname_g)

    np.save(out_u_path, U_LIE_PROM)
    np.save(out_g_path, g_hist)

    print(f"[saved] Lie PROM solution     -> {out_u_path}")
    print(f"[saved] Lie parameters (g(t)) -> {out_g_path}")

    # ----------------- FOM vs Lie PROM: overlay plot + error -------------------
    fom_dir = os.path.join(parent_dir, "fem_testing_data")
    fom_file = f"fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
    fom_path = os.path.join(fom_dir, fom_file)

    if not os.path.isfile(fom_path):
        print(f"[warn] FOM file not found, skipping overlay plot: {fom_path}")
    else:
        U_FOM = np.load(fom_path)   # shape (N, T_FOM)

        # Align time horizon
        min_T = min(U_FOM.shape[1], U_LIE_PROM.shape[1])
        U_FOM = U_FOM[:, :min_T]
        U_ROM = U_LIE_PROM[:, :min_T]

        # 1) Frobenius trajectory error
        num = np.linalg.norm(U_FOM - U_ROM, ord='fro')
        den = np.linalg.norm(U_FOM, ord='fro')
        rel_frob = (num / den) if den > 0 else np.nan

        print(f"\n=== Lie PROM metrics (mu1={mu1:.3f}, mu2={mu2:.4f}) ===")
        print(f"  Projection: LSPG")
        print(f"  Frobenius relative error (trajectory): {100*rel_frob:.4f}%")

        # 2) Overlay at multiple times
        snapshot_times = [5, 10, 15, 20, 25]  # seconds
        snapshot_indices = [
            min(int(t / At), min_T - 1) for t in snapshot_times
        ]

        os.makedirs("figs_lie_rom_vs_fom", exist_ok=True)

        plt.figure(figsize=(8, 5))
        for t_idx, t_val in zip(snapshot_indices, snapshot_times):
            plt.plot(
                X, U_FOM[:, t_idx], color='k', linewidth=1.5,
                label="HDM" if t_idx == snapshot_indices[0] else ""
            )
            plt.plot(
                X, U_ROM[:, t_idx], '--', linewidth=1.5,
                label="Lie PROM (LSPG)"
                if t_idx == snapshot_indices[0] else ""
            )
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(
            f"Overlay (mu1={mu1:.3f}, mu2={mu2:.4f}) | Lie PROM (multi-reference)"
        )
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend(ncol=2)

        multi_pdf = os.path.join(
            "figs_lie_rom_vs_fom",
            f"overlay_times_LiePROM_lspg_mu1_{mu1:.3f}_mu2_{mu2:.4f}.pdf"
        )
        plt.savefig(multi_pdf, format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"[saved] Overlay figure -> {multi_pdf}")
