#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-reference Lie manifold on a SINGLE snapshot matrix
with 5 parameters, all solved by Gauss–Newton.

Model per snapshot (in cluster c):

    y(x) ≈ α * u_model(x; s, γ, κ) + β

where
    u_model(x; s, γ, κ) = Lie-transform(u_ref^(c); s, γ, κ),

and Lie-transform is:
    1) Dilate + warp:
           ξ_raw = x / s
           ξ     = clip(ξ_raw, 0, 1 - eps)
           ξγ    = ξ + γ ξ (1 - ξ)
           ξγ    = clip(ξγ, 0, 1 - eps)
           z     = ξγ * (N - 1)
           (linear interpolation of u_ref at z)
    2) Continuous shift in index-space:
           i_grid = [0, 1, ..., N-1]
           z_shift = i_grid - κ
           (linear interpolation of u_sg at z_shift, clamped)

Parameters per snapshot:
    g = (α, β, s, γ, κ)  → 5 DOFs

We minimize:
    J(g) = 1/2 || α u_model(g) + β - y ||^2

Gauss–Newton:
    r(g) = u_fit - y
    For each iteration:
        - Build Jacobian J (N x 5)
          * ∂r/∂α, ∂r/∂β analytically
          * ∂r/∂s, ∂r/∂γ, ∂r/∂κ via finite differences
        - Solve (J^T J + λI) δ = -J^T r
        - Damped update + clamping of (s, γ, κ).

Pipeline:
1) Load one snapshot file S (N x Nt).
2) POD → project into r_POD dimensions.
3) K-means on POD coords → K clusters.
4) For each cluster:
   - pick a medoid snapshot as u_ref^(c)
5) For each snapshot y_j:
   - initialize g0 = (α0, β0, s0, γ0, κ0)
     (α0, β0 from LS on untransformed u_ref; s0=1, γ0=0, κ0=0)
   - run Gauss–Newton on all 5 params.
6) Save results and plots.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------- user config --------------------

DATA_FILE   = "../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy"
OUT_DIR     = "lie_cluster_GN_full5"
os.makedirs(OUT_DIR, exist_ok=True)

N_CLUSTERS   = 4        # number of clusters
R_POD        = 5        # POD dimension for clustering
SEED         = 1234

PERIODIC     = False    # we treat domain as non-periodic here

# Gauss–Newton settings
MAX_IT_GN    = 20
TOL_GN       = 1e-8
LAMBDA_REG   = 1e-10   # small regularization for (J^T J)
DAMPING_INIT = 1.0
DAMPING_MIN  = 1e-3

# Bounds / clamps for stability
S_MIN, S_MAX = 0.75, 1.25
G_MIN, G_MAX = -0.8, 0.8
K_MIN_FRAC   = -0.5     # kappa in [-0.5*N, 0.5*N]
K_MAX_FRAC   = 0.5

# Finite-difference steps for s, gamma, kappa
FD_EPS_S     = 1e-3
FD_EPS_GAMMA = 1e-3
FD_EPS_KAPPA = 1e-2

# -----------------------------------------------------


# ========== Lie helper functions (warp + continuous shift) ==========

def dilate_warp(u, s, gamma, x):
    """
    Dilate + warp u(x) in 1D using linear interpolation.

    Baseline:
        ξ_raw = x / s
        ξ     = clip(ξ_raw, 0, 1-eps)
    Warp:
        ξγ = ξ + γ ξ (1 - ξ)
        ξγ = clip(ξγ, 0, 1-eps)

    Returns:
        u_sg: shape (N,)
    """
    N = u.size
    eps = 1e-12

    xi_raw = x / s
    xi = np.clip(xi_raw, 0.0, 1.0 - eps)

    xi_gamma = xi + gamma * xi * (1.0 - xi)
    xi_gamma = np.clip(xi_gamma, 0.0, 1.0 - eps)

    z  = xi_gamma * (N - 1)
    i0 = np.floor(z).astype(int)
    i1 = np.minimum(i0 + 1, N - 1)
    w  = z - i0

    u0 = u[i0]
    u1 = u[i1]

    return (1.0 - w) * u0 + w * u1


def shift_continuous_clamped(u, kappa):
    """
    Continuous shift in index space, with clamping.

    Given u[i], we define, at each index i:

        src = i - kappa
        clamp src in [0, N-1]
        u_shift[i] = linear interpolation of u at src

    This generalizes integer shift (kappa integer) to continuous values.
    """
    N = u.size
    idx = np.arange(N, dtype=float)
    z = idx - kappa
    z = np.clip(z, 0.0, N - 1.0 - 1e-12)

    i0 = np.floor(z).astype(int)
    i1 = np.minimum(i0 + 1, N - 1)
    w  = z - i0

    u0 = u[i0]
    u1 = u[i1]

    return (1.0 - w) * u0 + w * u1


def lie_transform(u_ref, s, gamma, kappa, x):
    """
    Full Lie transform:
        u_sg  = dilate_warp(u_ref; s, gamma)
        u_mod = shift_continuous_clamped(u_sg; kappa)

    Returns:
        u_mod: shape (N,)
    """
    u_sg = dilate_warp(u_ref, s, gamma, x)
    u_mod = shift_continuous_clamped(u_sg, kappa)
    return u_mod


# ========== Utility: initial α,β (LS) given a "shape" u0 = u_ref ===========

def alpha_beta_ls(u, y):
    """
    Closed-form LS for α, β in:
        y ≈ α u + β 1

    Returns:
        alpha, beta
    """
    N = u.size
    c  = float(N)
    e  = float(y.sum())
    yy = float(y @ y)  # not needed for α,β but kept for consistency

    a = float(u @ u)
    b = float(u.sum())
    d = float(u @ y)

    det = a * c - b * b
    if abs(det) < 1e-14:
        alpha = d / (a + 1e-14)
        beta  = 0.0
    else:
        alpha = (d * c - b * e) / det
        beta  = (-d * b + a * e) / det

    return alpha, beta


# ========== Gauss–Newton on all 5 parameters ==========

def gauss_newton_full5(y, u_ref, x,
                       alpha0=1.0, beta0=0.0,
                       s0=1.0, gamma0=0.0, kappa0=0.0):
    """
    Gauss–Newton refinement of g = (α, β, s, γ, κ):

        u_mod(x; s, γ, κ) = Lie-transform(u_ref; s, γ, κ)
        u_fit(x) = α u_mod(x) + β
        r = u_fit - y

    Minimize 1/2 ||r||^2.

    J columns:
        ∂r/∂α     = u_mod
        ∂r/∂β     = 1
        ∂r/∂s     ≈ [r(g + δ_s e_s) - r(g)] / δ_s
        ∂r/∂γ     ≈ ...
        ∂r/∂κ     ≈ ...
    """
    N = y.size

    # Initialize parameters
    alpha = float(alpha0)
    beta  = float(beta0)
    s     = float(s0)
    gamma = float(gamma0)
    kappa = float(kappa0)

    # Clamp initial shape parameters
    s     = np.clip(s, S_MIN, S_MAX)
    gamma = np.clip(gamma, G_MIN, G_MAX)
    kappa = np.clip(kappa, K_MIN_FRAC * N, K_MAX_FRAC * N)

    # Helper for cost and residual
    def residual_and_u(alpha, beta, s, gamma, kappa):
        u_mod = lie_transform(u_ref, s, gamma, kappa, x)
        u_fit = alpha * u_mod + beta
        r = u_fit - y
        cost = 0.5 * (r @ r)
        return r, u_mod, cost

    # Initial residual and cost
    r, u_mod, old_cost = residual_and_u(alpha, beta, s, gamma, kappa)

    for it in range(MAX_IT_GN):
        # Build Jacobian (N x 5)
        J = np.empty((N, 5), dtype=float)

        # Analytic columns for α, β
        # r = α u_mod + β - y
        J[:, 0] = u_mod     # ∂r/∂α
        J[:, 1] = 1.0       # ∂r/∂β

        # Finite differences for s, gamma, kappa
        # Base residual r(g)
        # r_s ≈ (r(g + δ_s e_s) - r(g))/δ_s, etc.

        # s
        s_plus = np.clip(s + FD_EPS_S, S_MIN, S_MAX)
        r_s_plus, _, _ = residual_and_u(alpha, beta, s_plus, gamma, kappa)
        J[:, 2] = (r_s_plus - r) / FD_EPS_S

        # gamma
        gamma_plus = np.clip(gamma + FD_EPS_GAMMA, G_MIN, G_MAX)
        r_g_plus, _, _ = residual_and_u(alpha, beta, s, gamma_plus, kappa)
        J[:, 3] = (r_g_plus - r) / FD_EPS_GAMMA

        # kappa
        kappa_plus = np.clip(kappa + FD_EPS_KAPPA, K_MIN_FRAC * N, K_MAX_FRAC * N)
        r_k_plus, _, _ = residual_and_u(alpha, beta, s, gamma, kappa_plus)
        J[:, 4] = (r_k_plus - r) / FD_EPS_KAPPA

        # Normal equations: (J^T J + λI) δ = -J^T r
        JTJ = J.T @ J
        JTJ += LAMBDA_REG * np.eye(5)
        g_vec = J.T @ r
        rhs = -g_vec

        try:
            delta = np.linalg.solve(JTJ, rhs)
        except np.linalg.LinAlgError:
            print("[GN] Singularity in JTJ, stopping.")
            break

        # Damped update
        lam = DAMPING_INIT
        success = False
        for _ in range(10):
            alpha_t = alpha + lam * delta[0]
            beta_t  = beta  + lam * delta[1]
            s_t     = s     + lam * delta[2]
            gamma_t = gamma + lam * delta[3]
            kappa_t = kappa + lam * delta[4]

            # Clamp
            s_t     = np.clip(s_t, S_MIN, S_MAX)
            gamma_t = np.clip(gamma_t, G_MIN, G_MAX)
            kappa_t = np.clip(kappa_t, K_MIN_FRAC * N, K_MAX_FRAC * N)

            r_t, u_mod_t, cost_t = residual_and_u(alpha_t, beta_t, s_t, gamma_t, kappa_t)

            if cost_t < old_cost:
                alpha, beta, s, gamma, kappa = alpha_t, beta_t, s_t, gamma_t, kappa_t
                r, u_mod, old_cost = r_t, u_mod_t, cost_t
                success = True
                break
            else:
                lam *= 0.5
                if lam < DAMPING_MIN:
                    break

        if not success:
            # no improvement
            break

        # Stopping on cost change
        if it > 0 and np.abs(old_cost - cost_t) < TOL_GN * (1.0 + old_cost):
            break

    # Final relative error
    rel = np.linalg.norm(r) / (np.linalg.norm(y) + 1e-14)

    return (alpha, beta, s, gamma, kappa), rel


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
    U, svals, Vt = np.linalg.svd(S, full_matrices=False)
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

    np.save(os.path.join(OUT_DIR, "refs_indices.npy"), np.array(refs_idx, dtype=object))

    # 5) Fit per snapshot: full GN on (α, β, s, γ, κ)
    g_params = np.zeros((Nt, 5))   # [alpha, beta, s, gamma, kappa]
    rel_arr  = np.zeros(Nt)

    t0 = time.time()
    for j in range(Nt):
        c = int(labels[j])
        ref_idx = refs_idx[c]
        if ref_idx is None:
            g_params[j] = [1.0, 0.0, 1.0, 0.0, 0.0]
            rel_arr[j] = 0.0
            continue

        u_ref = S[:, ref_idx]
        y = S[:, j]

        # Initial α, β from LS on raw u_ref (no transform).
        alpha0, beta0 = alpha_beta_ls(u_ref, y)
        s0 = 1.0
        gamma0 = 0.0
        kappa0 = 0.0  # no initial shift

        (alpha, beta, s_best, gamma, kappa), rel = gauss_newton_full5(
            y, u_ref, x,
            alpha0=alpha0, beta0=beta0,
            s0=s0, gamma0=gamma0, kappa0=kappa0
        )

        g_params[j, :] = [alpha, beta, s_best, gamma, kappa]
        rel_arr[j] = rel

        if (j + 1) % 50 == 0 or j == Nt - 1:
            print(f"[fit] snapshot {j+1}/{Nt} | cluster={c} | "
                  f"rel={rel:.3e}, "
                  f"g=[α={alpha:.3f}, β={beta:.3f}, s={s_best:.3f}, "
                  f"γ={gamma:.3f}, κ={kappa:.3f}]")

    t1 = time.time()
    print(f"[done] GN-based fits for all snapshots in {t1 - t0:.2f} s")
    print(f"[error] global mean rel error = {rel_arr.mean():.3e}, max = {rel_arr.max():.3e}")

    # 6) Save numeric results
    np.save(os.path.join(OUT_DIR, "g_params.npy"), g_params)
    np.save(os.path.join(OUT_DIR, "rel_errors.npy"), rel_arr)
    np.save(os.path.join(OUT_DIR, "labels.npy"), labels)

    meta = {
        "DATA_FILE": DATA_FILE,
        "N": int(N),
        "Nt": int(Nt),
        "N_CLUSTERS": N_CLUSTERS,
        "R_POD": R_POD,
        "PERIODIC": PERIODIC,
        "GN": {
            "MAX_IT_GN": MAX_IT_GN,
            "TOL_GN": TOL_GN,
            "LAMBDA_REG": LAMBDA_REG,
            "DAMPING_INIT": DAMPING_INIT,
            "DAMPING_MIN": DAMPING_MIN,
            "S_MIN": S_MIN,
            "S_MAX": S_MAX,
            "G_MIN": G_MIN,
            "G_MAX": G_MAX,
            "K_MIN_FRAC": K_MIN_FRAC,
            "K_MAX_FRAC": K_MAX_FRAC,
            "FD_EPS_S": FD_EPS_S,
            "FD_EPS_GAMMA": FD_EPS_GAMMA,
            "FD_EPS_KAPPA": FD_EPS_KAPPA,
        },
        "warp_form": "xi_gamma = xi + gamma * xi * (1 - xi)",
        "shift": "continuous clamped index shift κ",
        "note": "All five parameters solved by Gauss–Newton (α,β,s,γ,κ).",
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
    plt.title("Mean relative error per cluster (Lie + full GN on 5 params)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mean_rel_per_cluster.png"), dpi=200)
    plt.close()

    # 8) Per-cluster plots: error curve + typical/worst/median fits
    for c in range(N_CLUSTERS):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue

        ref_idx = refs_idx[c]
        u_ref = S[:, ref_idx]

        # (a) error vs snapshot index
        plt.figure(figsize=(7, 4))
        plt.plot(idx_c, rel_arr[idx_c], "o-", linewidth=1.0)
        plt.xlabel("snapshot index j")
        plt.ylabel("relative error")
        plt.title(f"Cluster {c}: relative error per snapshot")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"cluster_{c}_rel_error_curve.png"), dpi=200)
        plt.close()

        # (b) typical, worst, median
        rel_c = rel_arr[idx_c]
        mean_c = rel_c.mean()
        typical_local = np.argmin(np.abs(rel_c - mean_c))
        typical_idx = idx_c[typical_local]
        worst_local = np.argmax(rel_c)
        worst_idx = idx_c[worst_local]
        median_val = np.median(rel_c)
        median_local = np.argmin(np.abs(rel_c - median_val))
        median_idx = idx_c[median_local]

        chosen_indices = []
        for j in [typical_idx, worst_idx, median_idx]:
            if j not in chosen_indices:
                chosen_indices.append(j)

        for j in chosen_indices:
            y = S[:, j]
            alpha, beta, s_best, gamma, kappa = g_params[j]
            u_mod = lie_transform(u_ref, s_best, gamma, kappa, x)
            u_fit = alpha * u_mod + beta

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
                f"g=[α={alpha:.3f}, β={beta:.3f}, s={s_best:.3f}, "
                f"γ={gamma:.3f}, κ={kappa:.3f}]"
            )
            plt.legend(fontsize=8)
            plt.tight_layout()
            fname = os.path.join(OUT_DIR, f"cluster_{c}_snap_{j}_{kind}_true_vs_fit.png")
            plt.savefig(fname, dpi=200)
            plt.close()

    print(f"[saved] results and plots in: {OUT_DIR}")

if __name__ == "__main__":
    main()

