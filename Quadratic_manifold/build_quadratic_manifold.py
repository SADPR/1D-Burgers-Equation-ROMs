# build_manifold.py
# ------------------------------------------------------------------
# Offline builder:   reads *.npy snapshots, outputs Φ.npy  H.npy
# ------------------------------------------------------------------
import glob, sys, os
import numpy as np
from quad_utils import build_Q, build_E, compute_H, rel_error

# ---------------- configuration ---------------- #
snap_dir = sys.argv[1] if len(sys.argv) > 1 else "../FEM/fem_training_data/"
alpha     = 1e-2          # <-- choose your favourite regularisation value
eps_s     = 1e-6          # POD energy tolerance
zeta      = 0.1          # quadratic-dimension padding
u_ref     = None          # or np.load('u_ref.npy')  if you have one

# ---------------- load snapshots ---------------- #
files = sorted(glob.glob(os.path.join(snap_dir, "*.npy")))
if not files:
    raise RuntimeError(f"No .npy files found in {snap_dir}")
S = np.hstack([np.load(f) for f in files])          # (N,Ns)
N, Ns = S.shape
print(f"S: {S.shape},  loaded from {len(files)} files")

# ---------------- POD --------------------------- #
U, s, _ = np.linalg.svd(S, full_matrices=False)
energy = np.cumsum(s**2) / np.sum(s**2)
n_tra   = np.searchsorted(energy, 1 - eps_s)
n_qua   = int((np.sqrt(9+8*n_tra) - 3)/2 * (1+zeta))
n_max   = int((np.sqrt(1+8*Ns) - 1)/2)
n = min(n_qua, n_max)
print(f"n_tra={n_tra},  choose n={n}")

Phi = U[:, :n]
q   = Phi.T @ S                                   # (n,Ns)

# ---------------- E, Q, H ----------------------- #
Q = build_Q(q)                                    # (k,Ns)
E = build_E(S, Phi, q, u_ref)                     # (N,Ns)
H = compute_H(Q, E, alpha)                        # (N,k)

# ---------------- diagnostics ------------------- #
S_hat = Phi @ q + H @ Q + (0 if u_ref is None else u_ref[:,None])
print(f"training relative error = {rel_error(S, S_hat):.3e}")

# ---------------- save -------------------------- #
np.save("Phi.npy", Phi)
np.save("H.npy",   H)
np.save("q_train.npy", q)
print("saved  Phi.npy  H.npy  q_train.npy")