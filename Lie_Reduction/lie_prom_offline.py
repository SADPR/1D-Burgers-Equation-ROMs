#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline artifact builder for the multi-reference Lie PROM.

What this script does (OFFLINE ONLY):

1) Load one snapshot file S (N x Nt).
2) POD → project into r_POD dimensions (for clustering only).
3) K-means on POD coords → K clusters.
4) For each cluster:
   - pick a medoid snapshot index as ref idx: refs_indices[c]
   - save the corresponding reference field u_ref^(c) = S[:, ref_idx]
5) Save:
   - refs_indices.npy          (list of medoid indices per cluster)
   - u_ref_cluster_c.npy       (one file per cluster with its reference)
   - U_global.npy              (global POD basis used for clustering)
   - kmeans_lie.pkl            (trained KMeans object)
   - meta.json                 (basic info: N, Nt, etc.)

These are the ONLY things needed by the online Lie PROM launcher.
"""

import os
import json
import numpy as np
from sklearn.cluster import KMeans
import pickle

# -------------------- user config --------------------

DATA_FILE   = "../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy"
OUT_DIR     = "lie_cluster_GN_full5"   # same folder used later by the online Lie PROM
os.makedirs(OUT_DIR, exist_ok=True)

N_CLUSTERS   = 4        # number of clusters
R_POD        = 5        # POD dimension for clustering
SEED         = 1234

# -----------------------------------------------------


def main():
    # 1) Load one snapshot file
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}")
    S = np.load(DATA_FILE)  # (N, Nt)
    if S.ndim != 2:
        raise ValueError("DATA_FILE must be 2D (N x Nt)")
    N, Nt = S.shape
    print(f"[data] S shape = {S.shape} (N={N}, Nt={Nt})")

    # 2) POD for clustering (global POD basis)
    print("[POD] computing SVD...")
    U, svals, Vt = np.linalg.svd(S, full_matrices=False)
    r_eff = min(R_POD, U.shape[1])
    U_r = U[:, :r_eff]   # reduced global basis for clustering
    print(f"[POD] using r={r_eff} modes for clustering")

    # For completeness, also save the full U as U_global (or U_r, your choice).
    # Here we store U_r because that is all we need for cluster assignment.
    U_global = U_r
    np.save(os.path.join(OUT_DIR, "U_global.npy"), U_global)
    print(f"[save] U_global.npy with shape {U_global.shape}")

    # 3) K-means in POD space
    #    Q: (Nt, r_eff) snapshots-first in POD space
    Q = (U_r.T @ S).T
    print(f"[cluster] KMeans with K={N_CLUSTERS}")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(Q)  # (Nt,)

    # Save KMeans object
    kmeans_path = os.path.join(OUT_DIR, "kmeans_lie.pkl")
    with open(kmeans_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"[save] kmeans_lie.pkl")

    # 4) Pick medoid reference per cluster (closest point to cluster centroid in POD space)
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

        # save reference snapshot for this cluster
        u_ref_c = S[:, medoid_global]
        ref_file = os.path.join(OUT_DIR, f"u_ref_cluster_{c}.npy")
        np.save(ref_file, u_ref_c)
        print(f"         saved u_ref_cluster_{c}.npy with shape {u_ref_c.shape}")

    # Save refs_indices separately
    refs_indices_path = os.path.join(OUT_DIR, "refs_indices.npy")
    np.save(refs_indices_path, np.array(refs_idx, dtype=object))
    print(f"[save] refs_indices.npy")

    # Optional: meta file with basic info
    meta = {
        "DATA_FILE": DATA_FILE,
        "N": int(N),
        "Nt": int(Nt),
        "N_CLUSTERS": N_CLUSTERS,
        "R_POD": R_POD,
        "U_global_shape": list(U_global.shape),
        "note": "Lie PROM offline artifacts (no GN fits stored here).",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] meta.json")

    print(f"\n[done] Offline Lie PROM artifacts written to: {OUT_DIR}")
    print("       - U_global.npy")
    print("       - kmeans_lie.pkl")
    print("       - refs_indices.npy")
    print("       - u_ref_cluster_c.npy for each non-empty cluster")


if __name__ == "__main__":
    main()
