#!/usr/bin/env python3
# ---- Pure POD–RBF reconstruction (no PROM dynamics) -----------------
# Uses r = ||x - x'|| distances, loads min–max scaling, and UN-SCALES Qbar_hat.

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# -------------------- user config --------------------
a, b = 0.0, 100.0
m = 511
Xgrid = np.linspace(a, b, m + 1)

At = 0.05
Tf = 25.0
nTimeSteps = int(Tf / At)

# choose a test point (may be a training point)
test_mu1, test_mu2 = (4.750, 0.0200)

# Artifacts dir from training
rbf_dir = "rbf_training_simple"
Phi_p_path   = os.path.join(rbf_dir, "Phi_primary.npy")
Phi_s_path   = os.path.join(rbf_dir, "Phi_secondary.npy")
xtrain_txt   = os.path.join(rbf_dir, "rbf_xTrain.txt")          # scaled X (Ns × n)
precomp_txt  = os.path.join(rbf_dir, "rbf_precomputations.txt")  # W (Ns × nbar)
stdscale_txt = os.path.join(rbf_dir, "rbf_stdscaling.txt")       # min–max params
hyper_txt    = os.path.join(rbf_dir, "rbf_hyper.txt")            # kernel + epsilon

# FOM file (assumed precomputed)
fom_dir  = "../FEM/fem_testing_data"
fom_file = f"fem_simulation_mu1_{test_mu1:.3f}_mu2_{test_mu2:.4f}.npy"
fom_path = os.path.join(fom_dir, fom_file)

# -------------------- kernels k(r, eps) with r = Euclidean distance -----------
def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)

rbf_kernels = {
    "gaussian": gaussian_rbf,
    "imq": inverse_multiquadric_rbf,
}

# -------------------- 0) load FOM --------------------------------------------
if not os.path.isfile(fom_path):
    raise FileNotFoundError(f"FOM not found: {fom_path}")
print(f"[FOM] Loading {fom_path}")
U_FOM = np.load(fom_path)   # (N, T) with T = nTimeSteps+1

# -------------------- 1) load artifacts --------------------------------------
U_p = np.load(Phi_p_path)   # (N, n)
U_s = np.load(Phi_s_path)   # (N, nbar)

# scaling params (min–max)
with open(stdscale_txt, "r") as f:
    in_size, out_size = map(int, f.readline().split())
    scalingMethod = int(f.readline().strip())
    if scalingMethod != 1:
        raise ValueError("Expected min–max scaling method (1).")
x_min = np.loadtxt(stdscale_txt, skiprows=2, max_rows=1)  # (n,)
x_max = np.loadtxt(stdscale_txt, skiprows=3, max_rows=1)  # (n,)
y_min = np.loadtxt(stdscale_txt, skiprows=4, max_rows=1)  # (nbar,)
y_max = np.loadtxt(stdscale_txt, skiprows=5, max_rows=1)  # (nbar,)

# X_train (scaled) and W
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

# kernel + epsilon
with open(hyper_txt, "r") as f:
    _hdr = f.readline().strip()
    kernel_name = f.readline().strip()
    epsilon = float(f.readline().strip())
kernel_func = rbf_kernels[kernel_name]

# -------------------- sanity checks -------------------------------------------
N = Xgrid.size
if U_FOM.shape[0] != N:
    raise ValueError(f"U_FOM rows {U_FOM.shape[0]} != N={N}")
if U_p.shape[0] != N or U_s.shape[0] != N:
    raise ValueError("U_p and U_s must have N rows.")
if U_p.shape[1] != n:
    raise ValueError(f"Primary dim mismatch: U_p has n={U_p.shape[1]} vs X_train n={n}")
if W.shape != (X_train.shape[0], U_s.shape[1]):
    raise ValueError(f"W shape {W.shape} expected {(X_train.shape[0], U_s.shape[1])}")

print(f"[artifacts] Φ {U_p.shape}, Φ̄ {U_s.shape}, X_train {X_train.shape}, W {W.shape}")
print(f"[params] kernel={kernel_name}, ε={epsilon:.6f}")

# -------------------- 2) project FOM onto primary coords ----------------------
Tsteps = U_FOM.shape[1]
Qp = (U_p.T @ U_FOM).T      # (T, n), rows = time samples

# -------------------- 3) scale q_p(t) to [-1,1] ------------------------------
dx = (x_max - x_min).copy()
dx[dx < 1e-15] = 1.0
X_query = 2.0 * ((Qp - x_min) / dx) - 1.0    # (T, n)

# -------------------- 4) RBF prediction of secondary coords -------------------
# Distances r between each query and each training input
dists_q = np.linalg.norm(
    X_query[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
)                                            # (T, Ns)
Phi_q = kernel_func(dists_q, epsilon)        # (T, Ns)
Qbar_hat_scaled_rows = Phi_q @ W             # (T, nbar)  <-- scaled secondary coords

# -------- UN-SCALE Y back to original secondary coordinates (CRUCIAL) --------
dy = (y_max - y_min).copy()
dy[dy < 1e-15] = 1.0
Qbar_hat_rows = 0.5 * (Qbar_hat_scaled_rows + 1.0) * dy[None, :] + y_min[None, :]  # (T, nbar)

# Transpose to (nbar, T) for reconstruction
Qbar_hat = Qbar_hat_rows.T

# -------------------- 5) reconstruct in HDM space -----------------------------
U_hat = (U_p @ Qp.T) + (U_s @ Qbar_hat)       # (N, T)

# -------------------- 6) errors ----------------------------------------------
num = np.linalg.norm(U_FOM - U_hat)
den = np.linalg.norm(U_FOM)
rel_frob = num / (den + 1e-14)
rel_L2_time = np.linalg.norm(U_FOM - U_hat, axis=0) / (np.linalg.norm(U_FOM, axis=0) + 1e-14)

print("\n=== Pure POD–RBF reconstruction (no PROM) ===")
print(f"Frobenius relative error over trajectory: {100*rel_frob:.4f}%")

# -------------------- 7) plots (no saving) ------------------------------------
snapshot_times = [5, 10, 15, 20, 25]
snapshot_idx   = [min(int(t / At), Tsteps-1) for t in snapshot_times]

plt.figure(figsize=(8,5))
for t_idx, t_val in zip(snapshot_idx, snapshot_times):
    plt.plot(Xgrid, U_FOM[:, t_idx], 'k-', linewidth=1.5,
             label="HDM" if t_idx == snapshot_idx[0] else "")
    plt.plot(Xgrid, U_hat[:, t_idx], 'b--', linewidth=1.5,
             label="POD–RBF" if t_idx == snapshot_idx[0] else "")
plt.xlabel("x"); plt.ylabel("u")
plt.title(f"Pure POD–RBF reconstruction\n(mu1={test_mu1:.3f}, mu2={test_mu2:.4f})")
plt.legend(ncol=2)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

