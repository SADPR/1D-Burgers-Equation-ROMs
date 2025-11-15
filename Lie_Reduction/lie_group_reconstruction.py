#!/usr/bin/env python3
# Reconstruct ONLY the snapshots belonging to DATA_FILE using lie_params.npz
# (params were trained across multiple .npy files).
# Uses the exact per-snapshot template recorded during reduction.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- user config ----------------
#DATA_FILE   = Path("../FEM/fem_training_data/fem_simulation_mu1_5.500_mu2_0.0300.npy")
DATA_FILE   = Path("../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy")
PARAMS_FILE = Path("lie_params.npz")
PLOT_SAMPLES = 5         # how many snapshots from this file to plot
# ---------------------------------------------

assert DATA_FILE.exists(), f"Data file not found: {DATA_FILE.resolve()}"
assert PARAMS_FILE.exists(), f"Params file not found: {PARAMS_FILE.resolve()}"

# -------- load params (trained across ALL files) --------
z = np.load(PARAMS_FILE, allow_pickle=True)

x          = z["x"]
PERIODIC   = bool(z["periodic"])
files_list = list(z["files"])
shapes     = z["shapes_per_file"]      # array of shape (nfiles, 2): (N, Nt_i)
S_shape    = tuple(z["S_shape"])
tpl_file_idx = z["tpl_file_idx"].astype(int)
tpl_local_col = z["tpl_local_col"].astype(int)
shifts     = z["shifts"].astype(int)
dils       = z["dilations"].astype(float)
alphas     = z["alphas"].astype(float)
betas      = z["betas"].astype(float)

# sanity
N_all, Ns_all = S_shape
N_file = int(shapes[0, 0])
assert all(int(s[0]) == N_file for s in shapes), "All files must share same spatial size N"
assert N_all == N_file, "Mismatch between saved N and per-file N"

# -------- identify the slice (global column range) for DATA_FILE --------
try:
    fi_sel = files_list.index(DATA_FILE.name)
except ValueError:
    raise RuntimeError(f"{DATA_FILE.name} not present in params 'files' list.")

Nt_list = [int(s[1]) for s in shapes]
offsets = np.cumsum([0] + Nt_list[:-1])      # global start column per file
g_start = int(offsets[fi_sel])
g_end   = g_start + Nt_list[fi_sel]           # exclusive
assert g_end <= Ns_all

# -------- load ONLY the selected file's data --------
S_sel = np.load(DATA_FILE)                    # shape (N, Nt_sel)
N, Nt_sel = S_sel.shape
assert N == N_all and Nt_sel == Nt_list[fi_sel], "Selected file shape mismatch."

# -------- helper ops (shift/dilate) --------
def shift_periodic(u, k): return np.roll(u, int(k))

def shift_clamped(u, k):
    k = int(k)
    if k == 0: return u.copy()
    w = np.empty_like(u)
    if k > 0:
        w[:k] = u[0];  w[k:] = u[:-k]
    else:
        k = -k
        w[-k:] = u[-1];  w[:-k] = u[k:]
    return w

def shift_op(u, k):
    return shift_periodic(u, k) if PERIODIC else shift_clamped(u, k)

def periodic_interp(u, y):
    Nloc = u.size
    z = (y * Nloc) % Nloc
    i0 = np.floor(z).astype(int)
    i1 = (i0 + 1) % Nloc
    w = z - i0
    return (1 - w) * u[i0] + w * u[i1]

def dilate(u, s):
    if PERIODIC:
        y = (x / s) % 1.0
        return periodic_interp(u, y)
    else:
        xi = np.clip(x / s, 0.0, 1.0 - 1e-12)
        z  = xi * (N - 1)
        i0 = np.floor(z).astype(int)
        i1 = np.minimum(i0 + 1, N - 1)
        w  = z - i0
        return (1 - w) * u[i0] + w * u[i1]

# -------- lazy loader for template files --------
_cache = {}
def get_file_matrix(fname):
    """Load snapshot matrix for template file fname and cache it."""
    if fname not in _cache:
        path = DATA_FILE.parent / fname
        assert path.exists(), f"Template source file missing: {path}"
        _cache[fname] = np.load(path)
        assert _cache[fname].shape[0] == N, f"N mismatch in template file {fname}"
    return _cache[fname]

# -------- reconstruct only the selected file's snapshots --------
X_lie_sel = np.zeros_like(S_sel)

for j_local in range(Nt_sel):
    k_global = g_start + j_local

    # template location used during reduction
    fi_t  = int(tpl_file_idx[k_global])
    jl_t  = int(tpl_local_col[k_global])
    fname_t = files_list[fi_t]

    # fetch template column
    S_t = get_file_matrix(fname_t)          # (N, Nt_template)
    u_base = S_t[:, jl_t]

    # apply learned (s, shift, alpha, beta) for this snapshot
    u_s  = dilate(u_base, dils[k_global])
    u_sh = shift_op(u_s, shifts[k_global])
    X_lie_sel[:, j_local] = alphas[k_global] * u_sh + betas[k_global]

# -------- errors and plots for the subset only --------
rel_frob = np.linalg.norm(S_sel - X_lie_sel, 'fro') / np.linalg.norm(S_sel, 'fro')
print(f"[Subset: {DATA_FILE.name}] Relative Frobenius error: {rel_frob:.6f}")

# pick evenly spaced local columns to visualize
idxs = np.linspace(0, Nt_sel - 1, PLOT_SAMPLES, dtype=int)

plt.figure(figsize=(7, 4))
for j in idxs:
    plt.plot(x, S_sel[:, j],       linewidth=1.2, label=f"S[{j}] true")
    plt.plot(x, X_lie_sel[:, j], '--', linewidth=1.2, label=f"S[{j}] Lie")
plt.xlabel("x"); plt.ylabel("u"); plt.title(f"Ground truth vs Lie (subset)\n{DATA_FILE.name}")
plt.legend(ncol=2, fontsize=9)
plt.tight_layout(); plt.show()
