# quad_utils.py
# ------------------------------------------------------------------
# Helper functions for quadratic-manifold ROM (POD + explicit H).
# ------------------------------------------------------------------

import numpy as np
from typing import Tuple

# ------------------------------------------------------------- #
# 1.  Symmetric Kronecker helpers
# ------------------------------------------------------------- #
def get_sym(q: np.ndarray) -> np.ndarray:
    """
    Unique monomials q_i q_j  (j >= i). Length k = n(n+1)/2.
    """
    n = q.size
    idx_i, idx_j = np.triu_indices(n)
    return (q[idx_i] * q[idx_j])


def build_Q(q: np.ndarray) -> np.ndarray:
    """
    Construct Q ∈ ℝ^{k×Ns}   with k = n(n+1)/2  from reduced coordinates q.
    """
    _, Ns = q.shape
    k = q.shape[0] * (q.shape[0] + 1) // 2
    Q = np.empty((k, Ns))
    for s in range(Ns):
        Q[:, s] = get_sym(q[:, s])
    return Q


# ------------------------------------------------------------- #
# 2.  Error matrix  E = S − Φ q − u_ref
# ------------------------------------------------------------- #
def build_E(S: np.ndarray,
            Phi: np.ndarray,
            q: np.ndarray,
            u_ref: np.ndarray | None = None) -> np.ndarray:
    """
    E = S − (Φ q + u_ref)     with broadcasting-safe handling of u_ref.

    u_ref :  None  → treated as 0  
             scalar → added as constant shift  
             (N,)   → full vector reference state
    """
    if u_ref is None:
        return S - (Phi @ q)

    # ensure column broadcast
    if np.isscalar(u_ref):
        shift = u_ref
    else:
        shift = u_ref[:, None]          # (N,1)

    return S - (Phi @ q + shift)



# ------------------------------------------------------------- #
# 3.  Compute H  (vectorised, Tikhonov α)
# ------------------------------------------------------------- #
def compute_H(Q: np.ndarray,
              E: np.ndarray,
              alpha: float) -> np.ndarray:
    """
    Solve  min_H || E − H Q ||_F^2  + α² ||H||_F²   (ridge row-by-row)
    Closed-form via thin SVD of Q    (Barnett-Farhat 2022, Eq.(28))
    """
    Uq, s, VqT = np.linalg.svd(Q, full_matrices=False)   # Q = U Σ Vᵀ
    s2 = s**2                                           # (r,)

    # filter factors  f_l = σ_l² / (σ_l² + α²)
    f = s2 / (s2 + alpha**2)                            # (r,)

    # Pre-compute     Γ = V Σ⁻¹  Eᵀ    (shape r×N)
    Gamma = (VqT @ E.T) / s[:, None]                    # divide each row by σ_l

    # Assemble H   (N,k)  =  (U diag(f))  Γᵀ
    H = (Uq * f) @ Gamma                               # (k,N)
    return H.T                                         # (N,k)


# ------------------------------------------------------------- #
# 4.  Reconstruction helper + error
# ------------------------------------------------------------- #
def reconstruct(Phi: np.ndarray,
                H: np.ndarray,
                q: np.ndarray,
                u_ref: np.ndarray | None = None) -> np.ndarray:
    """
    Reconstruct snapshots from (Φ,H) and reduced coordinates q.
    """
    Q = build_Q(q)
    if u_ref is None:
        u_ref = 0.0
    return Phi @ q + H @ Q + u_ref[:, None]


def rel_error(S_exact: np.ndarray,
              S_rec: np.ndarray) -> float:
    """
    Relative Frobenius error.
    """
    return np.linalg.norm(S_exact - S_rec) / np.linalg.norm(S_exact)

