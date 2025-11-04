#!/usr/bin/env python3
import os, json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------- style ----------
USE_TEX = False                 # set True if your LaTeX is installed & working
plt.rc('text', usetex=USE_TEX)
plt.rc('font', family='serif')

# =========================
# CONFIG
# =========================
TEST_POINTS = [(6.20, 0.0400)]#[(4.750, 0.0200)]   # add more pairs if desired

# FOM location
FOM_TEST_DIR = "../FEM/fem_testing_data"

# POD–ANN artifacts
ANN_DIR      = "ann_models"
UMODES_PATH  = os.path.join(ANN_DIR, "U_modes.npy")
SCALER_Z_NPZ = os.path.join(ANN_DIR, "scaler_z.npz")
MODEL_PT     = os.path.join(ANN_DIR, "ann_model.pt")
CONFIG_JSON  = os.path.join(ANN_DIR, "config.json")

# PROM (LSPG) file template
TOL = 1e-05
PROM_ROM_PATH_TMPL = "../POD/Results_thesis/rom_solutions/U_PROM_tol_{tol:.0e}_mu1_{mu1}_mu2_{mu2}_lspg.npy"

# plotting/time
AT = 0.05
TIMES_OF_INTEREST = [5, 10, 15, 20, 25]
TIME_INDICES = [int(t/AT) for t in TIMES_OF_INTEREST]
A, B = 0.0, 100.0

# outputs
STATIC_OUT_DIR = "ann_outputs/plots_static_with_prom"
GIF_OUT_DIR    = "ann_outputs/gifs"
STATIC_FMT     = "png"           # "png" or "pdf"

# GIF controls
MAKE_GIF     = True
FPS          = 30
DPI          = 90
FRAME_STRIDE = 1                  # increase (2,3,...) to shrink GIF

# =========================
# helpers
# =========================
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

def load_ann():
    U_modes = np.load(UMODES_PATH)  # [M, n]
    sc = np.load(SCALER_Z_NPZ)
    mean, std = sc["mean"], sc["std"]
    cfg = json.load(open(CONFIG_JSON))
    model = make_mlp(3, U_modes.shape[1], cfg["hidden"], cfg["activation"], cfg["dropout"])
    state = torch.load(MODEL_PT, map_location="cpu")
    model.load_state_dict(state); model.eval()
    return U_modes, model, mean, std, cfg

def standardize(Z, mean, std):
    s = std.copy(); s[s==0]=1.0
    return (Z - mean)/s

def predict_ann_on_fom(mu1, mu2, U_modes, model, mean, std, U_FOM):
    Nt = U_FOM.shape[1]
    tau = np.linspace(0.0, 1.0, Nt)
    Z = np.column_stack([np.full(Nt, mu1), np.full(Nt, mu2), tau])  # [Nt,3]
    Zs = standardize(Z, mean, std)
    with torch.no_grad():
        Qhat = model(torch.from_numpy(Zs).float()).numpy()          # [Nt,n]
    Uhat = U_modes @ Qhat.T                                         # [M,Nt]
    return Uhat

def compute_rel_error(U_FOM, U_ROM):
    Nt = min(U_FOM.shape[1], U_ROM.shape[1])
    diff = U_FOM[:, :Nt] - U_ROM[:, :Nt]
    num = np.linalg.norm(diff)
    den = np.linalg.norm(U_FOM[:, :Nt]) + 1e-16
    return num/den

def load_prom_rom(mu1, mu2, tol=TOL):
    tag_mu1 = f"{mu1:.3f}"
    tag_mu2 = f"{mu2:.4f}"
    path = PROM_ROM_PATH_TMPL.format(tol=tol, mu1=tag_mu1, mu2=tag_mu2)
    if not os.path.exists(path):
        # try 1e-05 vs 1e-5 naming variant
        alt = path.replace("1e-05", "1e-5")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"PROM file not found: {path}")
    U_prom = np.load(path)  # [M, Nt_prom]
    return U_prom, path

def plot_overlay_with_errors(mu1, mu2, U_FOM, U_ANN, U_PROM, modes_count):
    os.makedirs(STATIC_OUT_DIR, exist_ok=True)

    # match available time range
    Nt = min(U_FOM.shape[1], U_ANN.shape[1], U_PROM.shape[1])
    idxs = [i for i in TIME_INDICES if 0 <= i < Nt]
    if not idxs:
        raise RuntimeError("Requested time indices out of range.")

    # compute relative errors (global across 0..Nt-1)
    rel_ann  = compute_rel_error(U_FOM[:, :Nt], U_ANN[:, :Nt]) * 100.0
    rel_prom = compute_rel_error(U_FOM[:, :Nt], U_PROM[:, :Nt]) * 100.0
    print(f"[mu1={mu1:.3f}, mu2={mu2:.4f}] modes={modes_count}  "
          f"POD–ANN rel.err={rel_ann:.3f}%  |  PROM–LSPG rel.err={rel_prom:.3f}%")

    Nx = U_FOM.shape[0]
    X  = np.linspace(A, B, Nx)

    plt.figure(figsize=(7,6))
    first=True
    for k in idxs:
        lab_f = "FOM" if first else None
        lab_a = f"POD–ANN ({rel_ann:.2f}%)" if first else None
        lab_p = f"PROM–LSPG ({rel_prom:.2f}%)" if first else None
        plt.plot(X, U_FOM[:,k],  'k-',  lw=3, label=lab_f)
        plt.plot(X, U_ANN[:,k],  'b-', lw=2, alpha=0.9, label=lab_a)
        plt.plot(X, U_PROM[:,k], 'r-', lw=2, alpha=0.9, label=lab_p)
        first=False

    title = (rf"$\mu_1={mu1:.3f}$, $\mu_2={mu2:.4f}$  |  modes={modes_count} (ANN vs PROM)"
             if USE_TEX else f"mu1={mu1:.3f}, mu2={mu2:.4f} | modes={modes_count} (ANN vs PROM)")
    plt.title(title)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,t)$")
    plt.xlim(A, B)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    tag_mu1=f"{mu1:.3f}"; tag_mu2=f"{mu2:.4f}"
    out_path = os.path.join(STATIC_OUT_DIR, f"FOM_ANN_PROM_static_mu1_{tag_mu1}_mu2_{tag_mu2}.{STATIC_FMT}")
    plt.savefig(out_path, format=STATIC_FMT); plt.close()
    print(f"[ok] saved → {out_path}")

def make_gif_comparison(mu1, mu2, U_FOM, U_ANN, U_PROM, modes_count):
    os.makedirs(GIF_OUT_DIR, exist_ok=True)

    Nt = min(U_FOM.shape[1], U_ANN.shape[1], U_PROM.shape[1])
    frames = np.arange(0, Nt, FRAME_STRIDE, dtype=int)
    if len(frames) == 0:
        raise RuntimeError("No frames to animate (Nt==0).")

    Nx = U_FOM.shape[0]
    X  = np.linspace(A, B, Nx)

    # y-lims
    umin = min(U_FOM.min(), U_ANN.min(), U_PROM.min())
    umax = max(U_FOM.max(), U_ANN.max(), U_PROM.max())
    pad  = 0.05*(umax-umin) if umax>umin else 1.0

    fig, ax = plt.subplots(figsize=(7,5))
    ax.set_xlim(A, B)
    ax.set_ylim(umin-pad, umax+pad)
    ax.grid(True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")

    fom_line,  = ax.plot(X, U_FOM[:,frames[0]],  'k-',  lw=3,  label="FOM")
    ann_line,  = ax.plot(X, U_ANN[:,frames[0]],  'b--', lw=2,  label="POD–ANN")
    prom_line, = ax.plot(X, U_PROM[:,frames[0]], 'r-.', lw=2,  label="PROM–LSPG")
    ax.legend(loc='upper right')

    def update(i):
        k = frames[i]
        fom_line.set_ydata(U_FOM[:,k])
        ann_line.set_ydata(U_ANN[:,k])
        prom_line.set_ydata(U_PROM[:,k])
        ax.set_title(f"mu1={mu1:.3f}, mu2={mu2:.4f}  |  t={k*AT:.2f}s  | modes={modes_count}")
        return fom_line, ann_line, prom_line

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    tag_mu1=f"{mu1:.3f}"; tag_mu2=f"{mu2:.4f}"
    out_path = os.path.join(GIF_OUT_DIR, f"ANN_PROM_mu1_{tag_mu1}_mu2_{tag_mu2}.gif")
    ani.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"[ok] saved GIF → {out_path}")

# =========================
# main
# =========================
if __name__ == "__main__":
    U_modes, model, mean, std, cfg = load_ann()
    modes_count = U_modes.shape[1]

    for (mu1, mu2) in TEST_POINTS:
        tag_mu1=f"{mu1:.3f}"; tag_mu2=f"{mu2:.4f}"
        fom_path = os.path.join(FOM_TEST_DIR, f"fem_simulation_mu1_{tag_mu1}_mu2_{tag_mu2}.npy")
        if not os.path.exists(fom_path):
            print(f"[skip] FOM not found for (mu1,mu2)=({tag_mu1},{tag_mu2})"); continue

        U_FOM = np.load(fom_path)
        assert U_FOM.shape[0] == U_modes.shape[0], \
            f"DoF mismatch: FOM {U_FOM.shape[0]} vs POD modes {U_modes.shape[0]}"

        # ANN prediction
        U_ANN = predict_ann_on_fom(mu1, mu2, U_modes, model, mean, std, U_FOM)

        # PROM load
        U_PROM, prom_path = load_prom_rom(mu1, mu2, tol=TOL)
        assert U_PROM.shape[0] == U_modes.shape[0], \
            f"DoF mismatch: PROM {U_PROM.shape[0]} vs POD modes {U_modes.shape[0]} (file: {prom_path})"

        # Static overlay with errors in legend
        plot_overlay_with_errors(mu1, mu2, U_FOM, U_ANN, U_PROM, modes_count)

        # Optional GIF
        if MAKE_GIF:
            make_gif_comparison(mu1, mu2, U_FOM, U_ANN, U_PROM, modes_count)
