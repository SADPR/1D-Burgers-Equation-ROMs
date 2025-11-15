# ---- POD–RBF PROM test (single point) with FOM reference, metrics & GIF ----
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Matplotlib styling
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# -------------------- configuration --------------------
a, b = 0.0, 100.0
m = 511
X = np.linspace(a, b, m + 1)
Tconn = np.array([np.arange(1, m + 1), np.arange(2, m + 2)]).T

Tf = 25.0
At = 0.05
nTimeSteps = int(Tf / At)
E = 0.0
u0 = np.ones_like(X)

# Test point (use same mu1 for FOM and PROM!)
test_mu1, test_mu2 = (4.75, 0.0200)
uxa = test_mu1  # <<< keep BC consistent

# Projection type
projection = "LSPG"  # or "Galerkin"

# -------------------- artifact paths (from SIMPLE trainer) --------------------
rbf_dir     = "rbf_training_simple"
Phi_p_path  = os.path.join(rbf_dir, "Phi_primary.npy")        # Φ (N, n)
Phi_s_path  = os.path.join(rbf_dir, "Phi_secondary.npy")      # Φ̄ (N, nbar)
xtrain_txt  = os.path.join(rbf_dir, "rbf_xTrain.txt")         # scaled inputs X (Ns × n)
precomp_txt = os.path.join(rbf_dir, "rbf_precomputations.txt")# weights W (Ns × nbar)
stdscale_txt= os.path.join(rbf_dir, "rbf_stdscaling.txt")     # x_min/x_max/y_min/y_max
hyper_txt   = os.path.join(rbf_dir, "rbf_hyper.txt")          # kernel + epsilon

# Output dirs
os.makedirs("rom_solutions", exist_ok=True)
os.makedirs("fom_solutions", exist_ok=True)
os.makedirs("figs_rom_vs_fom", exist_ok=True)
os.makedirs("gifs_rom_vs_fom", exist_ok=True)

# -------------------- instantiate FEM --------------------
fem_burgers = FEMBurgers(X, Tconn)

# -------------------- FOM: check or run --------------------
fom_file = f"U_FOM_mu1_{test_mu1:.3f}_mu2_{test_mu2:.4f}.npy"
fom_path = os.path.join("fom_solutions", fom_file)

if os.path.exists(fom_path):
    print(f"[FOM] Loading cached FOM: mu1={test_mu1:.3f}, mu2={test_mu2:.4f}")
    U_FOM = np.load(fom_path)[:, :]
else:
    print(f"[FOM] Running FOM: mu1={test_mu1:.3f}, mu2={test_mu2:.4f}")
    U_FOM = fem_burgers.fom_burgers(At, nTimeSteps, u0, test_mu1, E, test_mu2)
    np.save(fom_path, U_FOM)
    print(f"[FOM] Saved → {fom_path}")

# -------------------- load PROM artifacts (scaled pipeline) -------------------
missing = [p for p in (Phi_p_path, Phi_s_path, xtrain_txt, precomp_txt, stdscale_txt, hyper_txt)
           if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(f"Missing RBF artifacts in '{rbf_dir}': {missing}")

U_p  = np.load(Phi_p_path)    # (N, n)
U_s  = np.load(Phi_s_path)    # (N, nbar)

# Load scaling (expects min–max method = 1)
with open(stdscale_txt, "r") as f:
    in_size, out_size = map(int, f.readline().split())
    scalingMethod = int(f.readline().strip())
    if scalingMethod != 1:
        raise ValueError("Expected min–max scaling (1) in rbf_stdscaling.txt.")
x_min = np.loadtxt(stdscale_txt, skiprows=2, max_rows=1)
x_max = np.loadtxt(stdscale_txt, skiprows=3, max_rows=1)
y_min = np.loadtxt(stdscale_txt, skiprows=4, max_rows=1)
y_max = np.loadtxt(stdscale_txt, skiprows=5, max_rows=1)

# Load X_train (scaled) and W
with open(xtrain_txt, "r") as f:
    Ns, n = map(int, f.readline().split())
X_train = np.loadtxt(xtrain_txt, skiprows=1)  # (Ns, n)
if X_train.ndim == 1:
    X_train = X_train[None, :]

with open(precomp_txt, "r") as f:
    W_rows, W_cols = map(int, f.readline().split())
W = np.loadtxt(precomp_txt, skiprows=1)       # (Ns, nbar)
if W.ndim == 1:
    W = W[:, None]

# Hyperparams
with open(hyper_txt, "r") as f:
    _hdr = f.readline().strip()  # "2 1"
    kernel = f.readline().strip()
    epsilon = float(f.readline().strip())

# -------------------- shape & option checks --------------------
N = X.size
if U_FOM.shape[0] != N:
    raise ValueError(f"U_FOM has {U_FOM.shape[0]} rows, but N={N}.")
if U_p.shape[0] != N or U_s.shape[0] != N:
    raise ValueError("U_p/U_s must have N rows.")
nbar = U_s.shape[1]
if U_p.shape[1] != n:
    raise ValueError(f"Primary dim mismatch: U_p has n={U_p.shape[1]} vs X_train n={n}")
if W.shape != (X_train.shape[0], nbar):
    raise ValueError(f"W shape {W.shape} expected {(X_train.shape[0], nbar)}")
if kernel not in ("gaussian", "imq"):
    raise ValueError("kernel must be 'gaussian' or 'imq'.")

print(f"[artifacts] Φ {U_p.shape}, Φ̄ {U_s.shape}, X_train {X_train.shape}, W {W.shape}")
print(f"[hyper] kernel={kernel}, ε={epsilon:g}, projection={projection}")
print(f"[sanity] FOM mu1={test_mu1}, PROM mu1 passed={uxa}")

# -------------------- run POD–RBF PROM (scaling-aware) -----------------------
print(f"[PROM–RBF] Running {projection} with kernel={kernel}, ε={epsilon:g}")
U_PROM_RBF = fem_burgers.pod_rbf_prom(
    At, nTimeSteps, u0, uxa, E, test_mu2,
    U_p, U_s,
    X_train, W, epsilon,
    x_min, x_max, y_min, y_max,
    projection=projection, kernel=kernel
)

rom_file = (
    f"U_PROM_RBF_{projection.lower()}_{kernel}_eps_{epsilon:.4f}_"
    f"mu1_{test_mu1:.3f}_mu2_{test_mu2:.4f}.npy"
)
rom_path = os.path.join("rom_solutions", rom_file)
np.save(rom_path, U_PROM_RBF)

# -------------------- errors --------------------
num = np.linalg.norm(U_FOM - U_PROM_RBF, ord='fro')
den = np.linalg.norm(U_FOM, ord='fro')
rel_frob = (num / den) if den > 0 else np.nan

print(f"\n=== POD–RBF PROM metrics (mu1={test_mu1:.3f}, mu2={test_mu2:.4f}) ===")
print(f"  Projection: {projection}, kernel: {kernel}, ε={epsilon:.4f}")
print(f"  Frobenius relative error (trajectory): {100*rel_frob:.4f}%")

# -------------------- overlay at multiple times --------------------
snapshot_times = [5, 10, 15, 20, 25]  # seconds
snapshot_indices = [int(t / At) for t in snapshot_times]

plt.figure(figsize=(8,5))
for t_idx, t_val in zip(snapshot_indices, snapshot_times):
    plt.plot(X, U_FOM[:, t_idx], color='k', linewidth=1.5,
             label="HDM" if t_idx == snapshot_indices[0] else "")
    plt.plot(X, U_PROM_RBF[:, t_idx], '--', color='b',
             linewidth=1.5,
             label=f"POD–RBF PROM ({projection})" if t_idx == snapshot_indices[0] else "")
plt.xlabel("x"); plt.ylabel("u")
plt.title(f"Overlay (mu1={test_mu1:.3f}, mu2={test_mu2:.4f}) | {kernel}, ε={epsilon:.3f}")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(ncol=2)
multi_pdf = os.path.join(
    "figs_rom_vs_fom",
    f"overlay_times_PODRBF_{projection.lower()}_{kernel}_eps_{epsilon:.4f}_"
    f"mu1_{test_mu1:.3f}_mu2_{test_mu2:.4f}.pdf"
)
plt.savefig(multi_pdf, format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# -------------------- GIF: time evolution overlay --------------------
fig, ax = plt.subplots(figsize=(8,5))
line_fom,  = ax.plot(X, U_FOM[:, 0],  color='k', label="HDM")
line_prom, = ax.plot(X, U_PROM_RBF[:, 0], '--', color='b',
                     label=f"POD–RBF PROM ({projection})")
ax.set_xlim(a, b)
ymin = min(U_FOM.min(), U_PROM_RBF.min())
ymax = max(U_FOM.max(), U_PROM_RBF.max())
pad = 0.05 * (ymax - ymin + 1e-12)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlabel("x"); ax.set_ylabel("u")
title = ax.set_title("t = 0.00 s")
ax.grid(True, linestyle="--", linewidth=0.5); ax.legend()

def _update(frame):
    line_fom.set_ydata(U_FOM[:, frame])
    line_prom.set_ydata(U_PROM_RBF[:, frame])
    title.set_text(f"t = {frame * At:.2f} s")
    return line_fom, line_prom, title

frames = nTimeSteps + 1
ani = FuncAnimation(fig, _update, frames=frames, blit=True)
gif_path = os.path.join(
    "gifs_rom_vs_fom",
    f"podrbf_prom_vs_fom_{projection.lower()}_{kernel}_eps_{epsilon:.4f}_"
    f"mu1_{test_mu1:.3f}_mu2_{test_mu2:.4f}.gif"
)
ani.save(gif_path, writer=PillowWriter(fps=12))
plt.close(fig)

print(f"\nSaved:")
print(f"  FOM        → {fom_path}")
print(f"  PROM–RBF   → {rom_path}")
print(f"  Overlay    → {multi_pdf}")
print(f"  GIF        → {gif_path}")