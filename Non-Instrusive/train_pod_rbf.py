#!/usr/bin/env python3
import os, re, json
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RBFInterpolator

# =========================
# CONFIG (edit as needed)
# =========================
MODES_PATH      = "modes/U_modes_tol_1e-06.npy"   # or your chosen file
DATA_DIR        = "../FEM/fem_training_data"
PREFIX          = "fem_simulation_"
TIME_NORM       = "index"   # "index" or "vector"
T_VECTOR_PATH   = None      # path to a .npy with times if TIME_NORM="vector"

KERNEL          = "gaussian"  # ["gaussian","linear","cubic","quintic","thin_plate_spline","multiquadric","inverse_multiquadric"]
EPSILON         = 1.0         # used by gaussian/mq/imq
NEIGHBORS       = 100         # local kNN; set 0 or None for global (can be heavy)
SMOOTHING       = 1e-8

# =========================
# Helpers
# =========================
MU_RE = re.compile(r"mu1_([0-9]+(?:\.[0-9]+)?)_mu2_([0-9]+(?:\.[0-9]+)?)")

def parse_mus(fname: str):
    m = MU_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse mu1/mu2 from '{fname}'. Expected ..._mu1_<float>_mu2_<float>...")
    return float(m.group(1)), float(m.group(2))

def build_Z_Q(modes_path: str, data_dir: str, prefix: str, time_normalization: str, t_vector_path: str):
    U_modes = np.load(modes_path)  # [M, n]
    M, n = U_modes.shape

    # deterministic order (match your POD stacking if you saved it)
    if os.path.exists("modes/stack_order.json"):
        with open("modes/stack_order.json","r") as f:
            files = json.load(f)["files"]
    else:
        files = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith(".npy")])

    # time normalization
    t_norm = None
    if time_normalization == "vector":
        if not (t_vector_path and os.path.exists(t_vector_path)):
            raise ValueError("TIME_NORM='vector' requires a valid T_VECTOR_PATH")
        t_full = np.load(t_vector_path).ravel()
        if t_full[0] != 0.0:
            t_full = t_full - t_full[0]
        T = t_full[-1] if t_full[-1] != 0 else 1.0
        t_norm = t_full / T

    Z_list, Q_list = [], []
    for fname in files:
        mu1, mu2 = parse_mus(fname)
        S = np.load(os.path.join(data_dir, fname))  # [M, Nt]
        if S.ndim != 2 or S.shape[0] != M:
            raise ValueError(f"{fname}: expected [M,Nt] with M={M}, got {S.shape}")
        Nt = S.shape[1]
        tau = (np.linspace(0.0,1.0,Nt) if time_normalization == "index" else t_norm)
        if len(tau) != Nt:
            raise ValueError(f"{fname}: tau length {len(tau)} != Nt {Nt}")

        Z_traj = np.column_stack([np.full(Nt, mu1), np.full(Nt, mu2), tau])   # [Nt,3]
        Q_traj = (S.T @ U_modes)                                              # [Nt,n]

        Z_list.append(Z_traj); Q_list.append(Q_traj)

    Z = np.vstack(Z_list)   # [Ns,3]
    Q = np.vstack(Q_list)   # [Ns,n]
    return U_modes, Z, Q, files

def train_and_save():
    U_modes, Z, Q, files = build_Z_Q(MODES_PATH, DATA_DIR, PREFIX, TIME_NORM, T_VECTOR_PATH)

    # scale inputs
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)

    neigh = None if (NEIGHBORS is None or NEIGHBORS <= 0) else min(NEIGHBORS, Zs.shape[0]-1)
    rbf = RBFInterpolator(
        Zs, Q,
        kernel=KERNEL,
        epsilon=(EPSILON if KERNEL in ["gaussian","multiquadric","inverse_multiquadric"] else None),
        neighbors=neigh,
        smoothing=SMOOTHING
    )

    os.makedirs("rbf_models", exist_ok=True)
    np.save("rbf_models/U_modes.npy", U_modes)
    dump(scaler, "rbf_models/scaler.pkl")
    dump(rbf,    "rbf_models/rbf.pkl")
    with open("rbf_models/train_info.json","w") as f:
        json.dump({
            "files": files,
            "Ns": int(Z.shape[0]),
            "n_modes": int(U_modes.shape[1]),
            "kernel": KERNEL,
            "epsilon": EPSILON,
            "neighbors": neigh,
            "smoothing": SMOOTHING,
            "inputs": "Z=[mu1,mu2,tau in 0..1]",
            "time_normalization": TIME_NORM
        }, f, indent=2)

    print(f"[OK] Trained RBF on {Z.shape[0]} samples; n_modes={U_modes.shape[1]}; neighbors={neigh}")
    print("Saved to rbf_models/")

if __name__ == "__main__":
    train_and_save()

