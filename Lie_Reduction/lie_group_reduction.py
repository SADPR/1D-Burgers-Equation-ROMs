#!/usr/bin/env python3
# Lie-group reduction (non-intrusive) over ALL .npy snapshot files in a folder.
# Live timing prints per snapshot: total Δt, dilation, coarse, refine.
# Fits, for each snapshot column (across all files):
#   - translation (integer shift; periodic or clamped)
#   - optional dilation s (width change)
#   - closed-form α, β (intensity rescale/offset)
# against the best template chosen from a bank sampled across ALL files.
#
# Saves lie_params.npz with everything needed for reconstruction:
#   x, periodic, files, shapes_per_file, S_shape
#   tpl_file_idx[k], tpl_local_col[k], shifts[k], dilation[k], alpha[k], beta[k]
#   s_grid, meta

import os, re, json, time
import numpy as np
from pathlib import Path

# ---------------- user config ----------------
DATA_DIR     = Path("../FEM/fem_training_data")             # folder with .npy files
FILE_REGEX   = r"^fem_simulation_mu1_\d+\.\d{3}_mu2_\d+\.\d{4}\.npy$"
PERIODIC     = False                                        # True if spatially periodic
USE_DILATE   = True                                         # include dilation in search
S_FACTORS    = np.linspace(0.90, 1.15, 9)                   # dilation candidates
COARSE       = 8                                            # local refine half-width (±COARSE)
TPL_STRIDE   = 10                                           # pick every TPL_STRIDE-th snapshot as template
TPL_MAX      = 400                                          # cap #templates across all files
PRINT_EVERY  = 1                                            # live-print every N snapshots
# ---------------------------------------------

assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR.resolve()}"
PATTERN = re.compile(FILE_REGEX)

# ---------- load all files & stack snapshots ----------
t0 = time.perf_counter()
files = sorted([f for f in os.listdir(DATA_DIR) if PATTERN.match(f)])
assert files, f"No files matched in {DATA_DIR} with pattern {FILE_REGEX}"

arrays = []
shapes_per_file = []
for fname in files:
    A = np.load(DATA_DIR / fname)            # (N, Nt_i)
    arrays.append(A)
    shapes_per_file.append((A.shape[0], A.shape[1]))
N = arrays[0].shape[0]
assert all(s[0] == N for s in shapes_per_file), "All files must share the same spatial size N"
S = np.hstack(arrays)                        # (N, Ns_total)
N, Ns = S.shape
t_load = time.perf_counter() - t0
print(f"Loaded {len(files)} files → stacked S shape = {S.shape}  (load {t_load:.2f}s)")

# Map global column -> (file_idx, local_col)
col_offsets = np.cumsum([0] + [s[1] for s in shapes_per_file[:-1]])
def to_global_col(fi, jl): return int(col_offsets[fi] + jl)

# ---------- 1D grid for dilation interpolation ----------
x = np.linspace(0.0, 1.0, N, endpoint=False)

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
    # Periodic / non-periodic linear interpolation
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

def best_alpha_beta(u_cand, y):
    # Solve min_{α,β} || α u_cand + β 1 - y ||_2^2
    a = float(u_cand @ u_cand)
    b = float(u_cand.sum())
    c = float(y.size)
    d = float(u_cand @ y)
    e = float(y.sum())
    det = a * c - b * b
    if abs(det) < 1e-14:
        alpha = d / (a + 1e-14)
        beta  = 0.0
    else:
        alpha = (d * c - b * e) / det
        beta  = (-d * b + a * e) / det
    return alpha, beta

# ---------- template bank across all files ----------
tpl_pairs = []  # list of (file_idx, local_col)
for fi, (_, Tfi) in enumerate(shapes_per_file):
    cols = list(range(0, Tfi, max(1, TPL_STRIDE)))
    tpl_pairs.extend((fi, jl) for jl in cols)

if len(tpl_pairs) > TPL_MAX:
    step = max(1, len(tpl_pairs) // TPL_MAX)
    tpl_pairs = tpl_pairs[::step]

tpl_gcols = np.array([to_global_col(fi, jl) for (fi, jl) in tpl_pairs], dtype=int)
templates = [S[:, g].copy() for g in tpl_gcols]
print(f"Template bank size: {len(templates)}  (across {len(files)} files)")

# ---------- coarse shift set ----------
if COARSE <= 1:
    coarse_shifts = np.arange(N)
else:
    ntry = max(8, COARSE)  # try ~ntry evenly spaced coarse shifts
    coarse_shifts = np.unique(np.round(np.linspace(0, N - 1, ntry)).astype(int))

# ---------- outputs (per global snapshot) ----------
tpl_file_idx = np.zeros(Ns, dtype=int)    # which file the template comes from
tpl_local_col = np.zeros(Ns, dtype=int)   # column inside that file
shifts  = np.zeros(Ns, dtype=int)
dils    = np.ones(Ns, dtype=float)
alphas  = np.ones(Ns, dtype=float)
betas   = np.zeros(Ns, dtype=float)

# ---------- main fit loop with live timing prints ----------
for k in range(Ns):
    t_snap0 = time.perf_counter()
    y = S[:, k]
    best_err = np.inf
    best = None   # (tpl_i, shift, s, a, b)

    # timers per snapshot
    t_dilate = 0.0
    t_coarse = 0.0
    t_refine = 0.0

    for i_tpl, u_ref in enumerate(templates):
        for s in (S_FACTORS if USE_DILATE else np.array([1.0])):
            t0 = time.perf_counter()
            u_rs = dilate(u_ref, s)
            t_dilate += time.perf_counter() - t0

            # coarse sweep
            t0 = time.perf_counter()
            best_local_err = np.inf
            best_local = None
            for sh in coarse_shifts:
                cand = shift_op(u_rs, sh)
                a, b = best_alpha_beta(cand, y)
                err = np.linalg.norm(a * cand + b - y)
                if err < best_local_err:
                    best_local_err = err
                    best_local = (sh, a, b)
            t_coarse += time.perf_counter() - t0

            # local refine around best coarse shift
            sh0, a0, b0 = best_local
            win = np.arange(sh0 - COARSE + 1, sh0 + COARSE)
            if PERIODIC: win = win % N
            else:        win = win[(win >= 0) & (win < N)]

            t0 = time.perf_counter()
            for sh in win:
                cand = shift_op(u_rs, sh)
                a, b = best_alpha_beta(cand, y)
                err = np.linalg.norm(a * cand + b - y)
                if err < best_err:
                    best_err = err
                    best = (i_tpl, sh, s, a, b)
            t_refine += time.perf_counter() - t0

    # store best
    i_tpl, sh, s, a, b = best
    fi, jl = tpl_pairs[i_tpl]
    tpl_file_idx[k] = fi
    tpl_local_col[k] = jl
    shifts[k]  = int(sh)
    dils[k]    = float(s)
    alphas[k]  = float(a)
    betas[k]   = float(b)

    # live print
    t_snap = time.perf_counter() - t_snap0
    if (k + 1) % PRINT_EVERY == 0 or k == Ns - 1:
        print(
            f"[{k+1:5d}/{Ns}] tpl=({files[fi]}, col {jl:4d}) | "
            f"Δt: {t_snap:6.2f}s  "
            f"dil: {t_dilate:5.2f}s  coarse: {t_coarse:5.2f}s  refine: {t_refine:5.2f}s  "
            f"s={s:.3f} α={a:.3f} β={b:.3f}  err={best_err:.3e}"
        )

# ---------- save everything needed for reconstruction ----------
meta = dict(
    periodic=PERIODIC,
    use_dilate=USE_DILATE,
    s_factors=S_FACTORS.tolist(),
    coarse=COARSE,
    tpl_stride=TPL_STRIDE,
    tpl_max=TPL_MAX,
    file_regex=FILE_REGEX,
)

np.savez(
    "lie_params.npz",
    x=x, periodic=np.array(PERIODIC),
    files=np.array(files, dtype=object),
    shapes_per_file=np.array(shapes_per_file, dtype=int),
    S_shape=S.shape,
    tpl_file_idx=tpl_file_idx,
    tpl_local_col=tpl_local_col,
    shifts=shifts,
    dilations=dils,
    alphas=alphas,
    betas=betas,
    s_grid=(S_FACTORS if USE_DILATE else np.array([1.0])),
    meta=json.dumps(meta),
)

print("Saved → lie_params.npz")
