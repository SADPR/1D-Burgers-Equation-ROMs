#!/usr/bin/env python3
# Lie-group reconstruction for a single DATA_FILE using lie_params.npz.
# - If DATA_FILE was part of training (present in params 'files'), we REPLAY
#   the stored per-snapshot parameters (template index, shift, dilation, α, β).
# - If DATA_FILE was NOT part of training, we do ON-THE-FLY FITTING against a
#   template bank rebuilt from the training files using the saved tpl_stride.
#
# Matching math as in reduction:
#   - clamped/periodic shift (must match reduction setting)
#   - optional dilation
#   - closed-form (α, β) and SSD_min (fast)
#
# Outputs relative error and optional saved reconstruction.

import json
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- user config ----------------
DATA_FILE    = Path("../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy")
PARAMS_FILE  = Path("lie_params.npz")
TRAIN_DIR    = Path("../FEM/fem_training_data")   # where training .npy live (templates)
PLOT_SAMPLES = 8
SAVE_NPY     = True
# Inference settings for UNSEEN files:
USE_DILATE   = True
COARSE       = 8                  # local refine half-width (±COARSE)
S_FACTORS    = np.linspace(0.90, 1.15, 9)
# ---------------------------------------------

assert DATA_FILE.exists(), f"Data file not found: {DATA_FILE.resolve()}"
assert PARAMS_FILE.exists(), f"Params file not found: {PARAMS_FILE.resolve()}"
assert TRAIN_DIR.exists(), f"TRAIN_DIR not found: {TRAIN_DIR.resolve()}"

# -------- load params (trained across ALL files) --------
z = np.load(PARAMS_FILE, allow_pickle=True)

x              = z["x"]
PERIODIC       = bool(z["periodic"])
files_list     = list(z["files"])
shapes         = z["shapes_per_file"]      # (nfiles, 2): (N, Nt_i)
S_shape        = tuple(z["S_shape"])
tpl_file_idx   = z["tpl_file_idx"].astype(int)
tpl_local_col  = z["tpl_local_col"].astype(int)
shifts_saved   = z["shifts"].astype(int)
dils_saved     = z["dilations"].astype(float)
alphas_saved   = z["alphas"].astype(float)
betas_saved    = z["betas"].astype(float)
meta           = json.loads(str(z["meta"]))

N_all, Ns_all = S_shape
N_file = int(shapes[0, 0])
assert all(int(s[0]) == N_file for s in shapes), "All files must share same spatial size N"
assert N_all == N_file, "Mismatch between saved N and per-file N"

# -------- load ONLY the selected file's data --------
S_sel = np.load(DATA_FILE)                    # shape (N, Nt_sel)
N, Nt_sel = S_sel.shape
assert N == N_all, "Selected file spatial size N mismatch."

# -------- helper ops (shift/dilate) --------
def shift_periodic(u, k):
    return np.roll(u, int(k))

# Clamped shift that duplicates edge value (must match reduction behavior!)
def shift_clamped(u, k):
    k = int(k)
    if k == 0:
        return u.copy()
    w = np.empty_like(u)
    if k > 0:
        w[:k] = u[0];  w[k:] = u[:-k]
    else:
        k = -k
        w[-k:] = u[-1];  w[:-k] = u[k:]
    return w

# --- If reduction used ZERO-FILL clamping, swap the function to this: ---
# def shift_clamped(u, k):
#     k = int(k)
#     if k == 0:
#         return u.copy()
#     w = np.empty_like(u)
#     if k > 0:
#         w[:k] = 0.0;  w[k:] = u[:-k]
#     else:
#         k = -k
#         w[-k:] = 0.0; w[:-k] = u[k:]
#     return w

def shift_op(u, k):
    return shift_periodic(u, k) if PERIODIC else shift_clamped(u, k)

def periodic_interp(u, y):
    Nloc = u.size
    z = (y * Nloc) % Nloc
    i0 = np.floor(z).astype(int)
    i1 = (i0 + 1) % Nloc
    w = z - i0
    return (1 - w) * u[i0] + w * u[i1]

def dilate(u, s, x_grid):
    if PERIODIC:
        y = (x_grid / s) % 1.0
        return periodic_interp(u, y)
    else:
        Nloc = u.size
        xi = np.clip(x_grid / s, 0.0, 1.0 - 1e-12)
        z  = xi * (Nloc - 1)
        i0 = np.floor(z).astype(int)
        i1 = np.minimum(i0 + 1, Nloc - 1)
        w  = z - i0
        return (1 - w) * u[i0] + w * u[i1]

# Closed-form (α, β, SSD_min) for candidate u vs target y
def alpha_beta_ssdmin(u, y, c, e, yy):
    a = float(u @ u)
    b = float(u.sum())
    d = float(u @ y)
    det = a * c - b * b
    if abs(det) < 1e-14:
        alpha = d / (a + 1e-14)
        beta  = 0.0
        ssd   = 1e300
    else:
        alpha = (d * c - b * e) / det
        beta  = (-d * b + a * e) / det
        ssd   = yy - (c * d * d - 2.0 * b * d * e + a * e * e) / det
    return alpha, beta, ssd

# -------- fast replay path if file was SEEN in training --------
if DATA_FILE.name in files_list:
    print(f"Replay mode (seen in training): {DATA_FILE.name}")
    fi_sel = files_list.index(DATA_FILE.name)

    Nt_list = [int(s[1]) for s in shapes]
    offsets = np.cumsum([0] + Nt_list[:-1])
    g_start = int(offsets[fi_sel])
    g_end   = g_start + Nt_list[fi_sel]
    assert Nt_sel == Nt_list[fi_sel], "Selected file time length mismatch vs training."

    # lazy loader for template files (from TRAIN_DIR)
    _cache = {}
    def get_file_matrix(fname):
        if fname not in _cache:
            path = TRAIN_DIR / fname
            assert path.exists(), f"Template source missing: {path}"
            _cache[fname] = np.load(path)
            assert _cache[fname].shape[0] == N, f"N mismatch in template file {fname}"
        return _cache[fname]

    X_lie_sel = np.zeros_like(S_sel)
    for j_local in range(Nt_sel):
        k_global = g_start + j_local

        fi_t  = int(tpl_file_idx[k_global])
        jl_t  = int(tpl_local_col[k_global])
        fname_t = files_list[fi_t]

        S_t = get_file_matrix(fname_t)
        u_base = S_t[:, jl_t]

        u_s  = dilate(u_base, dils_saved[k_global], x)
        u_sh = shift_op(u_s, shifts_saved[k_global])
        X_lie_sel[:, j_local] = alphas_saved[k_global] * u_sh + betas_saved[k_global]

else:
    # -------- inference mode for UNSEEN file: build template bank and fit ----------
    print(f"Inference mode (unseen file): {DATA_FILE.name}")

    # Rebuild template bank using saved tpl_stride / tpl_max (if present)
    tpl_stride = int(meta.get("tpl_stride", 10))
    tpl_max    = int(meta.get("tpl_max", 400))

    # Load all training files listed in params (from TRAIN_DIR)
    # and sample every tpl_stride-th column to form the bank.
    train_arrays = []
    shapes_per_file = []
    for fname in files_list:
        path = TRAIN_DIR / fname
        assert path.exists(), f"Training file missing for templates: {path}"
        A = np.load(path)                     # (N, Nt_i)
        assert A.shape[0] == N, f"N mismatch in template file {fname}"
        train_arrays.append(A)
        shapes_per_file.append((A.shape[0], A.shape[1]))

    # Build list of (file_idx, local_col) pairs for the template bank
    tpl_pairs = []
    for fi, (_, Tfi) in enumerate(shapes_per_file):
        cols = list(range(0, Tfi, max(1, tpl_stride)))
        tpl_pairs.extend((fi, jl) for jl in cols)

    if len(tpl_pairs) > tpl_max:
        step = max(1, len(tpl_pairs) // tpl_max)
        tpl_pairs = tpl_pairs[::step]

    # Convenience to fetch a particular template column
    _cache = {}
    def get_train_matrix(i_file):
        fname = files_list[i_file]
        if fname not in _cache:
            _cache[fname] = np.load(TRAIN_DIR / fname)
            assert _cache[fname].shape[0] == N
        return _cache[fname]

    # Coarse shift set (same logic as reduction)
    if COARSE <= 1:
        coarse_shifts = np.arange(N)
    else:
        ntry = max(8, COARSE)
        coarse_shifts = np.unique(np.round(np.linspace(0, N - 1, ntry)).astype(int))

    X_lie_sel = np.zeros_like(S_sel)

    # Fit each snapshot y in the unseen file
    for j_local in range(Nt_sel):
        y = S_sel[:, j_local]

        # precompute invariants for closed-form SSD
        c  = float(N)
        e  = float(y.sum())
        yy = float(y @ y)

        best_err = np.inf
        best = None   # (tpl_idx, shift, s, alpha, beta)

        for idx_tpl, (fi_t, jl_t) in enumerate(tpl_pairs):
            S_t = get_train_matrix(fi_t)     # (N, Nt_train)
            u_ref = S_t[:, jl_t]

            s_grid = (S_FACTORS if USE_DILATE else np.array([1.0]))
            for s in s_grid:
                u_rs = dilate(u_ref, s, x)

                # coarse sweep
                best_local_err = np.inf
                best_local = None
                for sh in coarse_shifts:
                    cand = shift_op(u_rs, sh)
                    a_, b_, err = alpha_beta_ssdmin(cand, y, c, e, yy)
                    if err < best_local_err:
                        best_local_err = err
                        best_local = (sh, a_, b_)

                # local refine around best coarse shift
                sh0, a0, b0 = best_local
                win = np.arange(sh0 - COARSE + 1, sh0 + COARSE)
                if PERIODIC:
                    win = win % N
                else:
                    win = win[(win >= 0) & (win < N)]

                for sh in win:
                    cand = shift_op(u_rs, sh)
                    a_, b_, err = alpha_beta_ssdmin(cand, y, c, e, yy)
                    if err < best_err:
                        best_err = err
                        best = (fi_t, jl_t, sh, s, a_, b_)

        # Apply best fit to reconstruct this snapshot
        fi_t, jl_t, sh, s, a, b = best
        S_t = get_train_matrix(fi_t)
        u_base = S_t[:, jl_t]
        u_s  = dilate(u_base, s, x)
        u_sh = shift_op(u_s, sh)
        X_lie_sel[:, j_local] = a * u_sh + b

        if (j_local + 1) % 50 == 0 or j_local == Nt_sel - 1:
            print(f"[{j_local+1}/{Nt_sel}] best_tpl=(file {fi_t}, col {jl_t}) "
                  f"shift={sh} s={s:.3f} α={a:.3f} β={b:.3f} ssd={best_err:.3e}")

# -------- errors and plots --------
rel_frob = np.linalg.norm(S_sel - X_lie_sel, 'fro') / np.linalg.norm(S_sel, 'fro')
print(f"[{DATA_FILE.name}] Relative Frobenius error: {rel_frob:.6f}")

if SAVE_NPY:
    out_path = DATA_FILE.with_name(DATA_FILE.stem + "_lie_recon.npy")
    np.save(out_path, X_lie_sel)
    print(f"Saved reconstruction → {out_path}")

# pick evenly spaced local columns to visualize
PLOT_SAMPLES = max(1, min(PLOT_SAMPLES, Nt_sel))
idxs = np.linspace(0, Nt_sel - 1, PLOT_SAMPLES, dtype=int)

plt.figure(figsize=(8, 4.6))
for j in idxs:
    plt.plot(x, S_sel[:, j],       linewidth=1.2, label=f"true[{j}]")
    plt.plot(x, X_lie_sel[:, j], '--', linewidth=1.2, label=f"lie[{j}]")
plt.xlabel("x"); plt.ylabel("u")
plt.title(f"Ground truth vs Lie\n{DATA_FILE.name}")
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()


