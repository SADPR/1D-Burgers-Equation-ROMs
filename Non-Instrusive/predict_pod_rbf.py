#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# ----------------- Style (safe default: no LaTeX to avoid TeX errors) -----------------
USE_TEX = False
plt.rc('text', usetex=USE_TEX)
plt.rc('font', family='serif')

# =========================
# CONFIG (edit as needed)
# =========================
# Points to evaluate
TEST_POINTS = [
    (4.250, 0.0150),   # your requested case
    # (4.560, 0.0190),
    # (5.190, 0.0260),
]

# Time labeling step (seconds/frame for titles and index mapping)
AT = 0.05
TIMES_OF_INTEREST = [5, 10, 15, 20, 25]             # seconds
TIME_INDICES = [int(t/AT) for t in TIMES_OF_INTEREST]

# Paths
FOM_TEST_DIR   = "../FEM/fem_training_data"       # held-out FOM trajectories
RBF_MODELS_DIR = "rbf_models"                       # artifacts from training
OUT_DIR        = "rbf_outputs/plots_static"         # where PDFs go

# Model artifact files
UMODES_PATH = os.path.join(RBF_MODELS_DIR, "U_modes.npy")
SCALER_PATH = os.path.join(RBF_MODELS_DIR, "scaler.pkl")
RBF_PATH    = os.path.join(RBF_MODELS_DIR, "rbf.pkl")
INFO_PATH   = os.path.join(RBF_MODELS_DIR, "train_info.json")  # optional

# Spatial domain (for axis labeling)
A, B = 0.0, 100.0

# =========================
# Utilities
# =========================
def compute_rel_error(U_FOM, U_ROM):
    """Relative error over time: ||FOM-ROM|| / ||FOM|| (match min Nt)."""
    Nt = min(U_FOM.shape[1], U_ROM.shape[1])
    diff = U_FOM[:, :Nt] - U_ROM[:, :Nt]
    num = np.linalg.norm(diff)
    den = np.linalg.norm(U_FOM[:, :Nt]) + 1e-16
    return num / den

def load_model():
    U_modes = np.load(UMODES_PATH)           # [M, n]
    scaler  = load(SCALER_PATH)
    rbf     = load(RBF_PATH)
    info = None
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, "r") as f:
            info = json.load(f)
    return U_modes, scaler, rbf, info

def predict_on_fom_grid(mu1, mu2, U_modes, scaler, rbf, U_FOM):
    """Predict POD–RBF on the same Nt as FOM using tau = linspace(0,1,Nt)."""
    Nt = U_FOM.shape[1]
    tau = np.linspace(0.0, 1.0, Nt)
    Zstar = np.column_stack([
        np.full(Nt, mu1, dtype=float),
        np.full(Nt, mu2, dtype=float),
        tau
    ])                         # [Nt, 3]
    Zs = scaler.transform(Zstar)
    Qhat = rbf(Zs)             # [Nt, n]
    Uhat = U_modes @ Qhat.T    # [M, Nt]
    return Uhat

def plot_overlay_static(mu1, mu2, U_FOM, U_PODRBF, modes_count, kernel_name="gaussian"):
    """One figure with overlays at t = 5,10,15,20,25 s."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Safety: clamp indices to available range
    Nt = min(U_FOM.shape[1], U_PODRBF.shape[1])
    idxs = [i for i in TIME_INDICES if 0 <= i < Nt]
    if not idxs:
        raise RuntimeError("None of the requested TIME_INDICES are within trajectory length.")

    Nx = U_FOM.shape[0]
    X  = np.linspace(A, B, Nx)

    # Figure
    plt.figure(figsize=(7, 6))
    first = True
    for k in idxs:
        label_fom = r"FOM" if first else None
        label_rbf = r"POD--RBF" if first else None
        plt.plot(X, U_FOM[:, k], 'k-', linewidth=3, label=label_fom)
        plt.plot(X, U_PODRBF[:, k], 'b--', linewidth=2, alpha=0.9, label=label_rbf)
        first = False

    rel_err = compute_rel_error(U_FOM, U_PODRBF)
    print(f"[mu1={mu1:.3f}, mu2={mu2:.4f}] modes={modes_count}  POD-RBF rel.err={rel_err:.6e}")

    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$")
    times_str = ", ".join([str(TIMES_OF_INTEREST[i]) for i, k in enumerate(TIME_INDICES) if 0 <= k < Nt])
    title = rf"$\mu_1={mu1:.3f}$, $\mu_2={mu2:.4f}$  |  modes={modes_count}, kernel={kernel_name}" if USE_TEX else f"mu1={mu1:.3f}, mu2={mu2:.4f} | modes={modes_count}, kernel={kernel_name}"
    plt.title(title)
    plt.xlim(A, B)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    tag_mu1 = f"{mu1:.3f}"
    tag_mu2 = f"{mu2:.4f}"
    out_path = os.path.join(OUT_DIR, f"FOM_PODRBF_static_mu1_{tag_mu1}_mu2_{tag_mu2}.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"[ok] saved → {out_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    U_modes, scaler, rbf, info = load_model()
    modes_count = U_modes.shape[1]
    kernel_name = "gaussian"
    if info is not None:
        kernel_name = info.get("kernel", kernel_name)

    for (mu1, mu2) in TEST_POINTS:
        tag_mu1 = f"{mu1:.3f}"
        tag_mu2 = f"{mu2:.4f}"
        fom_path = os.path.join(FOM_TEST_DIR, f"fem_simulation_mu1_{tag_mu1}_mu2_{tag_mu2}.npy")
        if not os.path.exists(fom_path):
            print(f"[skip] FOM not found for (mu1,mu2)=({tag_mu1},{tag_mu2}): {fom_path}")
            continue

        U_FOM = np.load(fom_path)  # [M, Nt]

        # Ensure spatial DoFs match
        assert U_FOM.shape[0] == U_modes.shape[0], \
            f"Spatial DoFs mismatch: FOM {U_FOM.shape[0]} vs POD modes {U_modes.shape[0]}"

        # Predict on the same Nt as FOM
        Uhat = predict_on_fom_grid(mu1, mu2, U_modes, scaler, rbf, U_FOM)

        # Static overlay at requested times
        plot_overlay_static(mu1, mu2, U_FOM, Uhat, modes_count, kernel_name=kernel_name)
