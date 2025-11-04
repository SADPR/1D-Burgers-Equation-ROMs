#!/usr/bin/env python3
import os, json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------- style (off by default to avoid TeX errors) ----------
USE_TEX = False
plt.rc('text', usetex=USE_TEX)
plt.rc('font', family='serif')

# =========================
# CONFIG
# =========================
TEST_POINTS = [(4.750, 0.0200)]  # you can add more pairs
AT = 0.05
TIMES_OF_INTEREST = [5, 10, 15, 20, 25]
TIME_INDICES = [int(t/AT) for t in TIMES_OF_INTEREST]

FOM_TEST_DIR = "../FEM/fem_testing_data"
ANN_DIR      = "ann_models"
OUT_DIR      = "ann_outputs/plots_static"

UMODES_PATH  = os.path.join(ANN_DIR, "U_modes.npy")
SCALER_Z_NPZ = os.path.join(ANN_DIR, "scaler_z.npz")
MODEL_PT     = os.path.join(ANN_DIR, "ann_model.pt")
CONFIG_JSON  = os.path.join(ANN_DIR, "config.json")

A, B = 0.0, 100.0

# =========================
# Helper bits
# =========================
def load_artifacts():
    U_modes = np.load(UMODES_PATH)  # [M, n]
    M, n = U_modes.shape
    sc = np.load(SCALER_Z_NPZ)
    mean, std = sc["mean"], sc["std"]
    cfg = json.load(open(CONFIG_JSON))
    # build model
    hidden = cfg["hidden"]; act = cfg["activation"]; dropout = cfg["dropout"]
    model = make_mlp(3, n, hidden, act, dropout)
    state = torch.load(MODEL_PT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return U_modes, model, mean, std, cfg

def make_mlp(in_dim, out_dim, hidden, act, dropout):
    if act == "relu": A = nn.ReLU
    elif act == "gelu": A = nn.GELU
    else: A = nn.ELU
    layers=[]; last=in_dim
    for h in hidden:
        layers += [nn.Linear(last,h), A()]
        if dropout>0: layers += [nn.Dropout(dropout)]
        last=h
    layers += [nn.Linear(last,out_dim)]
    return nn.Sequential(*layers)

def standardize(Z, mean, std):
    std = std.copy()
    std[std==0]=1.0
    return (Z - mean)/std

def compute_rel_error(U_FOM, U_ROM):
    Nt = min(U_FOM.shape[1], U_ROM.shape[1])
    diff = U_FOM[:, :Nt] - U_ROM[:, :Nt]
    num = np.linalg.norm(diff)
    den = np.linalg.norm(U_FOM[:, :Nt]) + 1e-16
    return num/den

def predict_on_fom_grid(mu1, mu2, U_modes, model, mean, std, U_FOM):
    Nt = U_FOM.shape[1]
    tau = np.linspace(0.0, 1.0, Nt)
    Z = np.column_stack([np.full(Nt, mu1), np.full(Nt, mu2), tau])  # [Nt,3]
    Zs = standardize(Z, mean, std)
    with torch.no_grad():
        Qhat = model(torch.from_numpy(Zs).float()).numpy()  # [Nt,n]
    Uhat = U_modes @ Qhat.T
    return Uhat

def plot_overlay(mu1, mu2, U_FOM, U_ANN, modes_count, model_name="ANN"):
    os.makedirs(OUT_DIR, exist_ok=True)
    Nt = min(U_FOM.shape[1], U_ANN.shape[1])
    idxs = [i for i in TIME_INDICES if 0 <= i < Nt]
    if not idxs: raise RuntimeError("Requested time indices out of range.")

    Nx = U_FOM.shape[0]
    X = np.linspace(A, B, Nx)

    plt.figure(figsize=(7,6))
    first=True
    for k in idxs:
        lab_f = "FOM" if first else None
        lab_r = "POD–ANN" if first else None
        plt.plot(X, U_FOM[:,k], 'k-', lw=3, label=lab_f)
        plt.plot(X, U_ANN[:,k], 'b--', lw=2, alpha=0.9, label=lab_r)
        first=False

    rel = compute_rel_error(U_FOM, U_ANN)
    print(f"[mu1={mu1:.3f}, mu2={mu2:.4f}] modes={modes_count}  POD-ANN rel.err={rel:.6e}")

    title = (rf"$\mu_1={mu1:.3f}$, $\mu_2={mu2:.4f}$  |  modes={modes_count}, {model_name}"
             if USE_TEX else f"mu1={mu1:.3f}, mu2={mu2:.4f} | modes={modes_count}, {model_name}")
    plt.title(title)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$")
    plt.xlim(A, B)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    tag_mu1=f"{mu1:.3f}"; tag_mu2=f"{mu2:.4f}"
    out_path = os.path.join(OUT_DIR, f"FOM_PODANN_static_mu1_{tag_mu1}_mu2_{tag_mu2}.pdf")
    plt.savefig(out_path, format="pdf"); plt.close()
    print(f"[ok] saved → {out_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    U_modes, model, mean, std, cfg = load_artifacts()
    modes_count = U_modes.shape[1]

    for (mu1, mu2) in TEST_POINTS:
        tag_mu1=f"{mu1:.3f}"; tag_mu2=f"{mu2:.4f}"
        fom_path = os.path.join(FOM_TEST_DIR, f"fem_simulation_mu1_{tag_mu1}_mu2_{tag_mu2}.npy")
        if not os.path.exists(fom_path):
            print(f"[skip] FOM not found for (mu1,mu2)=({tag_mu1},{tag_mu2})"); continue

        U_FOM = np.load(fom_path)
        assert U_FOM.shape[0] == U_modes.shape[0], \
            f"DoF mismatch: FOM {U_FOM.shape[0]} vs POD modes {U_modes.shape[0]}"

        Uhat = predict_on_fom_grid(mu1, mu2, U_modes, model, mean, std, U_FOM)
        plot_overlay(mu1, mu2, U_FOM, Uhat, modes_count, model_name="POD–ANN")
