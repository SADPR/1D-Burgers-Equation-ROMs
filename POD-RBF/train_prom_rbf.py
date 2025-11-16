#!/usr/bin/env python3
# --- POD–RBF PROM training (simple) with min–max scaling to [-1,1] ------------
# Style mimics the "old" script: r = ||x - x'||, same variable names.

import os, sys, pickle, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ------------------ user config ------------------
data_dir  = "../FEM/fem_training_data"
mu1_vals  = [4.250, 4.875, 5.500]
mu2_vals  = [0.0150, 0.0225, 0.0300]
fname_tpl = "fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"

# POD dimensions
n_primary = 17
n_total   = 17 + 79

# kernels and epsilon grid
def gaussian_rbf(r, epsilon):            # exp(-(eps*r)^2)
    return np.exp(-(epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):# 1/sqrt(1 + (eps*r)^2)
    return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)

rbf_kernels = {
    "gaussian": gaussian_rbf
    # "imq": inverse_multiquadric_rbf,
}

kernel_names = list(rbf_kernels.keys())   # e.g., ["gaussian","imq"]
epsilon_values = np.logspace(np.log10(0.2), np.log10(5.0), 10)
lambda_reg = 1e-6                         # small jitter like old script

# I/O
out_dir = "rbf_training_simple"
os.makedirs(out_dir, exist_ok=True)

# ------------------ data loading ------------------
def load_snapshots(data_dir, mu1_vals, mu2_vals, tpl):
    snaps, used = [], []
    for mu1 in mu1_vals:
        for mu2 in mu2_vals:
            fn = tpl.format(mu1=mu1, mu2=mu2)
            path = os.path.join(data_dir, fn)
            if not os.path.isfile(path):
                print(f"[warn] missing: {fn}")
                continue
            A = np.load(path)
            if A.ndim != 2:
                raise ValueError(f"{fn} must be 2D (N x Ns_block).")
            snaps.append(A); used.append(fn)
    if not snaps:
        raise FileNotFoundError("No snapshot files found.")
    S = np.hstack(snaps)  # (N, Ns)
    return S, used

# ------------------ main ------------------
def main():
    # 1) Load snapshots and POD
    S, used_files = load_snapshots(data_dir, mu1_vals, mu2_vals, fname_tpl)
    N, Ns = S.shape
    print(f"[data] Loaded S: {S.shape} from {len(used_files)} files")

    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    if n_total > U.shape[1]:
        print(f"[note] n_total={n_total} > rank={U.shape[1]} → clipping")
        n_total_clipped = U.shape[1]
    else:
        n_total_clipped = n_total
    if n_primary >= n_total_clipped:
        raise ValueError("Need n_primary < n_total.")
    Phi    = U[:, :n_primary]
    Phibar = U[:, n_primary:n_total_clipped]
    nbar   = Phibar.shape[1]
    print(f"[pod] Φ: {Phi.shape}, Φ̄: {Phibar.shape} (n={n_primary}, n̄={nbar})")

    # 2) Reduced coordinates (samples-first)
    Q    = (Phi.T    @ S).T      # (Ns, n)
    Qbar = (Phibar.T @ S).T      # (Ns, nbar)

    # 3) Remove exact duplicate samples in Q (keep first)
    print("[clean] Removing exact duplicate samples in Q (exact match)...")
    _, unique_idx = np.unique(Q, axis=0, return_index=True)
    unique_idx.sort()
    Q, Qbar = Q[unique_idx], Qbar[unique_idx]
    Ns_eff = Q.shape[0]
    print(f"[clean] Kept {Ns_eff} unique samples out of {Ns}.")

    # 4) Min–max scaling to [-1,1] (per feature)
    x_min = Q.min(axis=0); x_max = Q.max(axis=0)
    y_min = Qbar.min(axis=0); y_max = Qbar.max(axis=0)
    dx = x_max - x_min; dx[dx < 1e-15] = 1.0
    dy = y_max - y_min; dy[dy < 1e-15] = 1.0
    X = 2.0 * ((Q    - x_min) / dx) - 1.0     # (Ns_eff, n)
    Y = 2.0 * ((Qbar - y_min) / dy) - 1.0     # (Ns_eff, nbar)

    # 5) Train/val split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    # 6) Grid-search (mimic old style naming and flow)
    best_eps, best_kernel, best_err, best_W = None, None, np.inf, None
    start_time = time.time()
    print("Grid search over epsilon and kernel...")

    for eps in epsilon_values:
        for kn in kernel_names:
            kernel_func = rbf_kernels[kn]

            # Compute pairwise distances for training set
            dists_train = np.linalg.norm(
                X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
            )
            Phi_train = kernel_func(dists_train, eps)
            Phi_train += lambda_reg * np.eye(len(X_train))  # Regularization

            try:
                W = np.linalg.solve(Phi_train, Y_train)     # (Ntr, nbar)
            except np.linalg.LinAlgError:
                print(f"LinAlgError at eps={eps}, kernel={kn}, skipping.")
                continue

            # Validation set
            dists_val = np.linalg.norm(
                X_val[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
            )
            Phi_val = kernel_func(dists_val, eps)
            Y_val_pred = Phi_val @ W
            mse = mean_squared_error(Y_val, Y_val_pred)
            rel_error = np.linalg.norm(Y_val - Y_val_pred) / (np.linalg.norm(Y_val) + 1e-14)

            print(f"eps={eps:.5g}, kernel={kn:8s}, "
                  f"val MSE={mse:.6e}, RelErr={100*rel_error:.2f}%")

            if mse < best_err:
                best_err = mse
                best_eps = eps
                best_kernel = kn
                best_W = W.copy()

    if best_W is None:
        print("[Error] No feasible solution found in grid search. Exiting.")
        sys.exit(1)

    print(f"[Best] kernel={best_kernel}, eps={best_eps}, val MSE={best_err:.6e}")

    # 7) Retrain on full dataset
    dists_all = np.linalg.norm(
        X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2
    )
    Phi_all = rbf_kernels[best_kernel](dists_all, best_eps)
    Phi_all += lambda_reg * np.eye(len(X))

    W_final = np.linalg.solve(Phi_all, Y)
    total_time = time.time() - start_time
    print("Trained final W on entire dataset with best kernel & eps.")
    print(f"Training complete! Total training time: {total_time:.2f} seconds")

    # (optional) quick sanity on training fit (scaled space)
    Y_fit = Phi_all @ W_final
    fit_rel = np.linalg.norm(Y_fit - Y) / (np.linalg.norm(Y) + 1e-14)
    print(f"[sanity] training rel. fit (scaled Y): {100*fit_rel:.4f}%")

    # 8) Save everything (same simple text format as old script)
    with open(os.path.join(out_dir, "rbf_precomputations.txt"), "w") as f:
        f.write(f"{W_final.shape[0]} {W_final.shape[1]}\n")
        np.savetxt(f, W_final, fmt="%.7f")

    with open(os.path.join(out_dir, "rbf_xTrain.txt"), "w") as f:
        f.write(f"{X.shape[0]} {X.shape[1]}\n")
        np.savetxt(f, X, fmt="%.7f")

    # scaling params (min–max)
    scalingMethod = 1
    with open(os.path.join(out_dir, "rbf_stdscaling.txt"), "w") as f:
        f.write(f"{X.shape[1]} {Y.shape[1]}\n")   # input_size, output_size
        f.write(str(scalingMethod) + "\n")
        np.savetxt(f, x_min[None, :], fmt="%.7f")
        np.savetxt(f, x_max[None, :], fmt="%.7f")
        np.savetxt(f, y_min[None, :], fmt="%.7f")
        np.savetxt(f, y_max[None, :], fmt="%.7f")

    with open(os.path.join(out_dir, "rbf_hyper.txt"), "w") as f:
        f.write("2 1\n")              # "kernel name + epsilon" header
        f.write(f"{best_kernel}\n")
        f.write(f"{best_eps:.7f}\n")

    # POD & raw reduced (post-duplicate, unscaled)
    np.save(os.path.join(out_dir, "Phi_primary.npy"),  Phi)
    np.save(os.path.join(out_dir, "Phi_secondary.npy"), Phibar)
    np.save(os.path.join(out_dir, "Q_train.npy"),      Q)
    np.save(os.path.join(out_dir, "Qbar_train.npy"),   Qbar)

    # bundle (optional)
    with open(os.path.join(out_dir, "rbf_model.pkl"), "wb") as f:
        pickle.dump({
            "Phi": Phi, "Phibar": Phibar,
            "Q": Q, "Qbar": Qbar,
            "X": X, "Y": Y,
            "W": W_final,
            "kernel": best_kernel, "epsilon": best_eps,
            "lambda_reg": lambda_reg,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "n_primary": n_primary, "n_total": n_total,
            "used_files": used_files
        }, f)

    print("\n[saved]")
    print(f"  Weights:         {out_dir}/rbf_precomputations.txt")
    print(f"  X (scaled):      {out_dir}/rbf_xTrain.txt")
    print(f"  Scaling params:  {out_dir}/rbf_stdscaling.txt  (min–max, [-1,1])")
    print(f"  Hyperparams:     {out_dir}/rbf_hyper.txt")
    print(f"  POD bases:       Phi_primary.npy, Phi_secondary.npy")
    print(f"  Unscaled Q,Qbar: Q_train.npy, Qbar_train.npy")
    print(f"  Pickle bundle:   rbf_model.pkl")

if __name__ == "__main__":
    main()
