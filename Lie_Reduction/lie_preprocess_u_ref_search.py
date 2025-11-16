#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-reference Lie manifold on a SINGLE snapshot matrix
with discrete search + 1D warp (NO extra nonlinear amplitude term).

Model per cluster:
    y(x) ≈ α * u_ref( ξ_γ(x) shifted by k ) + β,

where:
    ξ      = x / s
    ξ_γ    = ξ + γ ξ (1 - ξ)

Parameters per snapshot:
    g = (α, β, k, s, γ)  → 5 DOFs.

Pipeline:
1) Load one snapshot file S (N x Nt).
2) POD → project into r_POD dimensions.
3) K-means on POD coords → K clusters.
4) For each cluster:
   - pick a medoid snapshot as u_ref^(c)
   - For each snapshot y_j in that cluster, find best (α, β, k, s, γ)
     via discrete search over:
         γ ∈ GAMMAS
         s ∈ S_FACTORS
         k: coarse + refine
     using closed-form α, β and SSD_min.

5) Save results and make plots per cluster:
   - Mean relative error per cluster (bar plot).
   - Error vs snapshot index.
   - TRUE vs FIT vs REF for:
       • "typical" snapshot (error closest to cluster mean),
       • worst snapshot (max error),
       • median-error snapshot (if distinct).
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------- user config --------------------

DATA_FILE   = "../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy"
OUT_DIR     = "lie_cluster_phase1_discrete_warp"
os.makedirs(OUT_DIR, exist_ok=True)

N_CLUSTERS   = 20        # number of clusters
R_POD        = 5        # POD dimension for clustering
SEED         = 1234

# Lie discrete-search settings
PERIODIC     = False
USE_DILATE   = True
S_FACTORS    = np.linspace(0.90, 1.15, 9)          # dilation candidates (s)
GAMMAS       = np.linspace(-0.4, 0.4, 7)           # warp strength γ
COARSE       = 8                                   # refine half-width (±COARSE shift window)

# -----------------------------------------------------


# ========== Lie helper functions (discrete search + warp) ==========

def shift_clamped(u, k):
    """Clamped integer shift, duplicating edge values."""
    k = int(k)
    if k == 0:
        return u.copy()
    w = np.empty_like(u)
    if k > 0:
        w[:k] = u[0]
        w[k:] = u[:-k]
    else:
        k = -k
        w[-k:] = u[-1]
        w[:-k] = u[k:]
    return w

def shift_op(u, k):
    return np.roll(u, int(k)) if PERIODIC else shift_clamped(u, k)

def periodic_interp(u, y):
    """Helper for periodic interpolation (if PERIODIC=True)."""
    Nloc = u.size
    z = (y * Nloc) % Nloc
    i0 = np.floor(z).astype(int)
    i1 = (i0 + 1) % Nloc
    w = z - i0
    return (1 - w) * u[i0] + w * u[i1]

def dilate_warp(u, s, gamma, x):
    """
    Dilate + warp u(x) in 1D using linear interpolation.

    Baseline:
        ξ = x / s
    Warp:
        ξ_γ = ξ + γ ξ (1 - ξ)
    (with clamping to [0, 1] if non-periodic).
    """
    N = u.size
    if PERIODIC:
        xi = (x / s) % 1.0
        xi_gamma = xi + gamma * xi * (1.0 - xi)
        xi_gamma = np.mod(xi_gamma, 1.0)
        return periodic_interp(u, xi_gamma)
    else:
        eps = 1e-12
        xi = np.clip(x / s, 0.0, 1.0 - eps)
        xi_gamma = xi + gamma * xi * (1.0 - xi)
        xi_gamma = np.clip(xi_gamma, 0.0, 1.0 - eps)
        z  = xi_gamma * (N - 1)
        i0 = np.floor(z).astype(int)
        i1 = np.minimum(i0 + 1, N - 1)
        w  = z - i0
        return (1.0 - w) * u[i0] + w * u[i1]

def alpha_beta_ssdmin(u, y, c, e, yy):
    """
    Closed-form α, β, and SSD_min between candidate u and target y.
    Given invariants:
        c = N, e = sum(y), yy = y·y.
    """
    a = float(u @ u)
    b = float(u.sum())
    d = float(u @ y)
    det = a * c - b * b
    if abs(det) < 1e-14:
        alpha = d / (a + 1e-14)
        beta  = 0.0
        ssd   = 1e300  # mark as bad
    else:
        alpha = (d * c - b * e) / det
        beta  = (-d * b + a * e) / det
        ssd   = yy - (c * d * d - 2.0 * b * d * e + a * e * e) / det
    return alpha, beta, ssd

def fit_snapshot_discrete_warp(y, u_ref, x):
    """
    Given target y and reference u_ref, find best (α, β, k, s, γ)
    by discrete search over:
        γ ∈ GAMMAS
        s ∈ S_FACTORS (if USE_DILATE else [1.0])
        k ∈ Z (coarse + refine),
    using closed-form α,β on the candidate vector:

        u_sg = dilate_warp(u_ref, s, γ, x)
        cand = shift_op(u_sg, k)

    Returns:
        g_opt: (5,) array [alpha, beta, k, s, gamma]
        best_err: scalar SSD
        rel: relative error ||y - u_fit|| / ||y||
    """
    N = y.size

    # Precompute invariants for closed-form SSD
    c  = float(N)
    e  = float(y.sum())
    yy = float(y @ y)

    best_err = np.inf
    best = None  # (k, s, alpha, beta, gamma)

    # Coarse shift set
    if COARSE <= 1:
        coarse_shifts = np.arange(N)
    else:
        ntry = max(8, COARSE)
        coarse_shifts = np.unique(
            np.round(np.linspace(0, N - 1, ntry)).astype(int)
        )

    s_grid = (S_FACTORS if USE_DILATE else np.array([1.0]))

    for gamma in GAMMAS:
        for s in s_grid:
            # dilate + warp once for this (s, gamma)
            u_sg = dilate_warp(u_ref, s, gamma, x)

            # coarse sweep over shifts
            best_local_err = np.inf
            best_local = None
            for sh in coarse_shifts:
                cand = shift_op(u_sg, sh)
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
                cand = shift_op(u_sg, sh)
                a_, b_, err = alpha_beta_ssdmin(cand, y, c, e, yy)
                if err < best_err:
                    best_err = err
                    best = (sh, s, a_, b_, gamma)

    k_best, s_best, alpha_best, beta_best, gamma_best = best

    # build final reconstruction and relative error
    u_sg  = dilate_warp(u_ref, s_best, gamma_best, x)
    u_sh  = shift_op(u_sg, k_best)
    u_fit = alpha_best * u_sh + beta_best
    rel   = np.linalg.norm(y - u_fit) / (np.linalg.norm(y) + 1e-14)

    g_opt = np.array([alpha_best, beta_best, k_best, s_best, gamma_best], dtype=float)
    return g_opt, best_err, rel


# ========================= main =========================

def main():
    # 1) Load one snapshot file
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}")
    S = np.load(DATA_FILE)  # (N, Nt)
    if S.ndim != 2:
        raise ValueError("DATA_FILE must be 2D (N x Nt)")
    N, Nt = S.shape
    print(f"[data] S shape = {S.shape} (N={N}, Nt={Nt})")

    # Spatial grid [0,1]
    x = np.linspace(0.0, 1.0, N)

    # 2) POD for clustering
    print("[POD] computing SVD...")
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    r_eff = min(R_POD, U.shape[1])
    U_r = U[:, :r_eff]
    Q = (U_r.T @ S).T  # (Nt, r_eff) snapshots-first in POD space

    print(f"[POD] using r={r_eff} modes for clustering")

    # 3) K-means in POD space
    print(f"[cluster] KMeans with K={N_CLUSTERS}")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(Q)  # (Nt,)

    # 4) Pick medoid reference per cluster (closest to cluster centroid in POD space)
    refs_idx = []
    for c in range(N_CLUSTERS):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            print(f"[warn] cluster {c} is empty")
            refs_idx.append(None)
            continue
        Q_c = Q[idx_c]  # (Nc, r_eff)
        centroid = Q_c.mean(axis=0)
        dists = np.linalg.norm(Q_c - centroid[None, :], axis=1)
        medoid_local = np.argmin(dists)
        medoid_global = idx_c[medoid_local]
        refs_idx.append(medoid_global)
        print(f"[cluster {c}] size={idx_c.size}, ref index={medoid_global}")

    # Save reference indices
    np.save(os.path.join(OUT_DIR, "refs_indices.npy"), np.array(refs_idx, dtype=object))

    # 5) Discrete Lie + warp fit per snapshot (per cluster)
    g_params = np.zeros((Nt, 5))   # [alpha, beta, k, s, gamma]
    ssd_arr  = np.zeros(Nt)
    rel_arr  = np.zeros(Nt)

    t0 = time.time()
    for j in range(Nt):
        c = int(labels[j])
        ref_idx = refs_idx[c]
        if ref_idx is None:
            g_params[j] = [1.0, 0.0, 0.0, 1.0, 0.0]
            ssd_arr[j] = 0.0
            rel_arr[j] = 0.0
            continue

        u_ref = S[:, ref_idx]
        y = S[:, j]

        g_opt, ssd, rel = fit_snapshot_discrete_warp(y, u_ref, x)

        g_params[j, :] = g_opt
        ssd_arr[j] = ssd
        rel_arr[j] = rel

        if (j + 1) % 50 == 0 or j == Nt - 1:
            alpha, beta, k, s_best, gamma = g_opt
            print(f"[fit] snapshot {j+1}/{Nt} | cluster={c} | "
                  f"rel={rel:.3e}, "
                  f"g=[α={alpha:.3f}, β={beta:.3f}, k={k:.1f}, "
                  f"s={s_best:.3f}, γ={gamma:.3f}]")

    t1 = time.time()
    print(f"[done] discrete+warp fits for all snapshots in {t1 - t0:.2f} s")
    print(f"[error] global mean rel error = {rel_arr.mean():.3e}, max = {rel_arr.max():.3e}")

    # 6) Save numeric results
    np.save(os.path.join(OUT_DIR, "g_params.npy"), g_params)
    np.save(os.path.join(OUT_DIR, "ssd.npy"), ssd_arr)
    np.save(os.path.join(OUT_DIR, "rel_errors.npy"), rel_arr)
    np.save(os.path.join(OUT_DIR, "labels.npy"), labels)

    meta = {
        "DATA_FILE": DATA_FILE,
        "N": int(N),
        "Nt": int(Nt),
        "N_CLUSTERS": N_CLUSTERS,
        "R_POD": R_POD,
        "PERIODIC": PERIODIC,
        "USE_DILATE": USE_DILATE,
        "S_FACTORS": S_FACTORS.tolist(),
        "GAMMAS": GAMMAS.tolist(),
        "COARSE": COARSE,
        "SEED": SEED,
        "warp_form": "xi_gamma = xi + gamma * xi * (1 - xi)",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 7) Plot: mean relative error per cluster
    mean_rel_per_cluster = []
    for c in range(N_CLUSTERS):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            mean_rel_per_cluster.append(0.0)
        else:
            mean_rel_per_cluster.append(rel_arr[idx_c].mean())
    mean_rel_per_cluster = np.array(mean_rel_per_cluster)

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(N_CLUSTERS), mean_rel_per_cluster)
    plt.xlabel("Cluster")
    plt.ylabel("Mean relative error")
    plt.title("Mean relative error per cluster (discrete Lie + warp)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mean_rel_per_cluster.png"), dpi=200)
    plt.close()

    # 8) Per-cluster plots:
    #    - error vs snapshot index
    #    - TRUE vs FIT vs REF for typical, worst, median snapshots
    for c in range(N_CLUSTERS):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue

        ref_idx = refs_idx[c]
        u_ref = S[:, ref_idx]

        # (a) error vs snapshot index for this cluster
        plt.figure(figsize=(7, 4))
        plt.plot(idx_c, rel_arr[idx_c], "o-", linewidth=1.0)
        plt.xlabel("snapshot index j")
        plt.ylabel("relative error")
        plt.title(f"Cluster {c}: relative error per snapshot")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"cluster_{c}_rel_error_curve.png"), dpi=200)
        plt.close()

        # (b) choose: typical (closest to mean), worst, median
        rel_c = rel_arr[idx_c]
        mean_c = rel_c.mean()
        # typical: closest to mean
        typical_local = np.argmin(np.abs(rel_c - mean_c))
        typical_idx = idx_c[typical_local]
        # worst: max error
        worst_local = np.argmax(rel_c)
        worst_idx = idx_c[worst_local]
        # median: closest to median
        median_val = np.median(rel_c)
        median_local = np.argmin(np.abs(rel_c - median_val))
        median_idx = idx_c[median_local]

        chosen_indices = []
        for j in [typical_idx, worst_idx, median_idx]:
            if j not in chosen_indices:
                chosen_indices.append(j)

        for j in chosen_indices:
            y = S[:, j]
            alpha, beta, k, s_best, gamma = g_params[j]
            u_sg  = dilate_warp(u_ref, s_best, gamma, x)
            u_sh  = shift_op(u_sg, k)
            u_fit = alpha * u_sh + beta

            if j == typical_idx:
                kind = "typical"
            elif j == worst_idx:
                kind = "worst"
            else:
                kind = "median"

            plt.figure(figsize=(7, 4))
            plt.plot(x, y,      label=f"true (snap {j})", linewidth=1.8)
            plt.plot(x, u_fit,  "--", label="fit", linewidth=1.5)
            plt.plot(x, u_ref,  ":", label=f"u_ref (idx={ref_idx})", linewidth=1.2)
            plt.xlabel("x")
            plt.ylabel("u")
            plt.title(
                f"Cluster {c} | snapshot {j} ({kind})\n"
                f"rel={rel_arr[j]:.3e}, "
                f"g=[α={alpha:.3f}, β={beta:.3f}, k={k:.1f}, "
                f"s={s_best:.3f}, γ={gamma:.3f}]"
            )
            plt.legend(fontsize=8)
            plt.tight_layout()
            fname = os.path.join(OUT_DIR, f"cluster_{c}_snap_{j}_{kind}_true_vs_fit.png")
            plt.savefig(fname, dpi=200)
            plt.close()

    print(f"[saved] results and plots in: {OUT_DIR}")

if __name__ == "__main__":
    main()
