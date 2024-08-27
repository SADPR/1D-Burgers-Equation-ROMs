# import numpy as np
# import scipy.linalg
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Step 1: Load Snapshot Data and Initialize Parameters

# # Load the snapshot data
# snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
# S = np.load(snapshot_file)  # S is the snapshot matrix with dimensions (N, Ns)

# # Initialize parameters
# epsilon_s = 1e-4  # Tolerance
# zeta = 0.1  # Correction factor
# omega = 0.01  # Regularization parameter, if not specified, we will determine alpha*
# N, Ns = S.shape  # N is the number of spatial points, Ns is the number of snapshots

# print(f"Loaded snapshot data with dimensions: {S.shape}")

# # Step 2: Singular Value Decomposition (SVD)

# # Compute SVD of the snapshot matrix S
# U_S, Sigma_S, Y_S_T = np.linalg.svd(S, full_matrices=False)  # SVD of the snapshot matrix S
# Sigma_S = np.diag(Sigma_S)

# # Determine n_tra
# sigma_cumulative = np.cumsum(np.diag(Sigma_S)) / np.sum(np.diag(Sigma_S))
# n_tra = np.searchsorted(sigma_cumulative, 1 - epsilon_s)

# # Compute the dimension n_qua for the quadratic approximation
# n_qua = int((np.sqrt(9 + 8 * n_tra) - 3) / 2 * (1 + zeta))

# # Final dimension n
# n_final = min(n_qua, int((np.sqrt(1 + 8 * Ns) - 1) / 2))

# # Truncate V such that V ∈ R^{N x n}
# V = U_S[:, :n_final]

# print(f"Determined n_tra: {n_tra}")
# print(f"Computed n_qua: {n_qua}")
# print(f"Final reduced dimension n: {n_final}")
# print(f"Shape of truncated V: {V.shape}")

# # Step 3: Construction of Error Matrix E and Matrix Q

# E = np.zeros((N, Ns))
# Q = np.zeros((n_final * (n_final + 1) // 2, Ns))

# for i in range(Ns):
#     q_i = V.T @ S[:, i]  # Generalized coordinates
#     E[:, i] = S[:, i] - V @ q_i  # Compute the error
#     q_kron = np.kron(q_i, q_i)  # Kronecker product of q_i with itself
#     n = q_i.size
#     q_kron = np.kron(q_i, q_i)  # Compute the Kronecker product

#     # The symmetric matrix corresponding to q_kron (without duplicates) can be extracted by:
#     q_unique = []
#     for i in range(n):
#         for j in range(i, n):  # Only take the upper triangle and diagonal elements
#             q_unique.append(q_i[i] * q_i[j])

#     q_unique = np.array(q_unique)  # Extract unique values due to symmetry # Unique values from the Kronecker product
#     Q[:, i] = q_unique

# print(f"Computed error matrix E with shape: {E.shape}")
# print(f"Constructed matrix Q with shape: {Q.shape}")

# # Step 4: Compute Thin SVD of Q

# U_Q, Sigma_Q, Y_Q_T = np.linalg.svd(Q, full_matrices=False)
# # Sigma_Q = np.diag(Sigma_Q)

# print(f"SVD of Q computed. Shapes -> U_Q: {U_Q.shape}, Sigma_Q: {Sigma_Q.shape}, Y_Q_T: {Y_Q_T.shape}")

# # # Step 5: Determine Regularization Parameter α
# # if 'alpha_star' not in locals():
# #     alpha_best = np.zeros(N, dtype=float)  # Initialize alpha_best as a float array
# #     n_smp = int(np.floor(omega * U_Q.shape[1]))  # Determine the number of samples

# #     # Create a vector of trial α values uniformly distributed in log scale
# #     alpha_smp = np.logspace(np.log10(Sigma_Q[-1]), np.log10(Sigma_Q[0]), num=n_smp)
# #     print("Trial alpha values (alpha_smp):", alpha_smp)

# #     def gcv_function(alpha, Sigma_Q, E_i):
# #         # GCV function to compute the criterion
# #         gcv_sum = 0
# #         for l in range(Sigma_Q.shape[0]):
# #             numerator = (Sigma_Q[l]**2) / (Sigma_Q[l]**2 + alpha**2)
# #             term = numerator * (E_i[l] / Sigma_Q[l])
            
# #             # Print intermediate values
# #             print(f"alpha: {alpha}, Sigma_Q[{l}]: {Sigma_Q[l]}, E_i[{l}]: {E_i[l]}, term: {term}")
            
# #             gcv_sum += term**2

# #         if np.isnan(gcv_sum):
# #             print(f"NaN encountered in GCV function for alpha: {alpha}")

# #         return np.sum(gcv_sum)  # This should return a scalar

# #     for i in range(N):
# #         gcv_scores = np.zeros(n_smp)
# #         for k in range(n_smp):
# #             gcv_scores[k] = gcv_function(alpha_smp[k], Sigma_Q, E[i, :])
        
# #         print(f"gcv_scores for row {i}: {gcv_scores}")

# #         if np.any(np.isnan(gcv_scores)):
# #             print(f"NaN detected in gcv_scores for row {i}")

# #         # Ensure that the returned value is a scalar
# #         alpha_value = alpha_smp[np.argmin(gcv_scores)]
# #         if isinstance(alpha_value, (list, np.ndarray)):
# #             raise ValueError(f"alpha_value is not a scalar: {alpha_value}")
# #         else:
# #             alpha_best[i] = alpha_value

# #     # Set alpha_star to the most frequently selected alpha
# #     alpha_star = alpha_best[np.argmax(np.bincount(alpha_best.astype(int)))]

# # print(f"Determined alpha*: {alpha_star}")


# # Assume we skip the regularization parameter step for now

# # Step 6: Compute the Coefficient Matrix H without any modifications

# H = np.zeros((N, U_Q.shape[0]))  # Initialize H matrix

# print("Sigma_Q values:", Sigma_Q)

# alpha_star = 2000000

# for i in range(N):
#     h_i = np.zeros(U_Q.shape[0])
#     for l in range(Sigma_Q.shape[0]):
#         numerator = Sigma_Q[l]**2
#         denominator = numerator + alpha_star**2  # Use the original denominator without any epsilon
        
#         term = (numerator / denominator) * (Y_Q_T[l, :] @ E[i, :]) * U_Q[:, l]
        
#         h_i += term

#     H[i, :] = h_i

# print(f"Computed coefficient matrix H with shape: {H.shape}")

# # Step 7: Reconstruction of Snapshot Data
# S_reconstructed = np.zeros((N, Ns))  # Initialize reconstructed snapshot matrix

# for i in range(Ns):
#     q_i = V.T @ S[:, i]  # Generalized coordinates for the i-th snapshot
#     q_outer = np.outer(q_i, q_i)
#     q_unique = q_outer[np.triu_indices_from(q_outer)]  # Unique values from the Kronecker product
    
#     S_reconstructed[:, i] = V @ q_i + H @ q_unique  # Reconstruction using V and H

# print(f"Reconstructed snapshot data with shape: {S_reconstructed.shape}")

# # Optionally, calculate the reconstruction error
# reconstruction_error = np.linalg.norm(S - S_reconstructed) / np.linalg.norm(S)
# print(f"Reconstruction error: {reconstruction_error:.4f}")


# # Step 8: Create and Save the GIF Animation

# fig, ax = plt.subplots()

# # Set the x and y limits based on the data
# ax.set_xlim(0, N - 1)
# y_min = S.min()-1
# y_max = S.max()+1
# ax.set_ylim(y_min, y_max)

# # Initialize the lines for the original and reconstructed data
# line1, = ax.plot(S[:, 0], label="Original")
# line2, = ax.plot(S_reconstructed[:, 0], label="Reconstructed")
# ax.legend()

# def update(frame):
#     line1.set_ydata(S[:, frame])
#     line2.set_ydata(S_reconstructed[:, frame])
#     ax.set_title(f"Snapshot {frame+1}")
#     return line1, line2

# ani = FuncAnimation(fig, update, frames=Ns, blit=True, repeat=True)

# plt.show()

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from getH_alpha_functions import get_dQ_dq, get_single_Q, get_sym, getE, getH, getQ

# Step 1: Load Snapshot Data and Initialize Parameters

# Load the snapshot data
snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
S = np.load(snapshot_file)  # S is the snapshot matrix with dimensions (N, Ns)

# Initialize parameters
epsilon_s = 1e-4  # Tolerance
zeta = 0.1  # Correction factor
omega = 0.01  # Regularization parameter, if not specified, we will determine alpha*
N, Ns = S.shape  # N is the number of spatial points, Ns is the number of snapshots

print(f"Loaded snapshot data with dimensions: {S.shape}")

# Step 2: Singular Value Decomposition (SVD)

# Compute SVD of the snapshot matrix S
U_S, Sigma_S, Y_S_T = np.linalg.svd(S, full_matrices=False)  # SVD of the snapshot matrix S
Sigma_S = np.diag(Sigma_S)

# Determine n_tra
sigma_cumulative = np.cumsum(np.diag(Sigma_S)) / np.sum(np.diag(Sigma_S))
n_tra = np.searchsorted(sigma_cumulative, 1 - epsilon_s)

# Compute the dimension n_qua for the quadratic approximation
n_qua = int((np.sqrt(9 + 8 * n_tra) - 3) / 2 * (1 + zeta))

# Final dimension n
n_final = min(n_qua, int((np.sqrt(1 + 8 * Ns) - 1) / 2))

# Truncate V such that V ∈ R^{N x n}
V = U_S[:, :n_final]

print(f"Determined n_tra: {n_tra}")
print(f"Computed n_qua: {n_qua}")
print(f"Final reduced dimension n: {n_final}")
print(f"Shape of truncated V: {V.shape}")

# Step 3: Construction of Error Matrix E and Matrix Q

# Calculate the reduced coordinates matrix q
q = V.T @ S

# Use provided functions to calculate Q and E
E = getE(N, Ns, S, V, q)
Q = getQ(n_final, Ns, q)

print(f"Computed error matrix E with shape: {E.shape}")
print(f"Constructed matrix Q with shape: {Q.shape}")

# Step 4: Compute Thin SVD of Q

U_Q, Sigma_Q, Y_Q_T = np.linalg.svd(Q, full_matrices=False)

print(f"SVD of Q computed. Shapes -> U_Q: {U_Q.shape}, Sigma_Q: {Sigma_Q.shape}, Y_Q_T: {Y_Q_T.shape}")

# Step 6: Compute the Coefficient Matrix H without any modifications

# alpha_star = 0.1  # Small regularization parameter

alphas = [10**i for i in range(-11, -10)]

for alpha_star in alphas:
    # Use the provided getH function to calculate H
    H = getH(Q, E, n_final, N, alpha_star)

    print(f"Computed coefficient matrix H with shape: {H.shape}")

    # Step 7: Reconstruction of Snapshot Data
    S_reconstructed = np.zeros((N, Ns))  # Initialize reconstructed snapshot matrix

    for i in range(Ns):
        q_i = q[:, i]  # Generalized coordinates for the i-th snapshot
        q_unique = get_sym(q_i)  # Use get_sym to calculate the unique quadratic terms
        
        S_reconstructed[:, i] = V @ q_i + H @ q_unique  # Reconstruction using V and H

    print(f"Reconstructed snapshot data with shape: {S_reconstructed.shape}")

    # Optionally, calculate the reconstruction error
    reconstruction_error = np.linalg.norm(S - S_reconstructed) / np.linalg.norm(S)
    print(f"Alpha: {alpha_star}, Reconstruction error: {reconstruction_error:.4f}")

    # Step 8: Create and Save the GIF Animation

    fig, ax = plt.subplots()

    # Set the x and y limits based on the data
    ax.set_xlim(0, N - 1)
    y_min = min(S.min(), S_reconstructed.min())
    y_max = max(S.max(), S_reconstructed.max())
    ax.set_ylim(y_min, y_max)

    # Initialize the lines for the original and reconstructed data
    line1, = ax.plot(S[:, 0], label="Original")
    line2, = ax.plot(S_reconstructed[:, 0], label="Reconstructed")
    ax.legend()

    def update(frame):
        line1.set_ydata(S[:, frame])
        line2.set_ydata(S_reconstructed[:, frame])
        ax.set_title(f"Snapshot {frame+1}")
        return line1, line2

    ani = FuncAnimation(fig, update, frames=Ns, blit=True, repeat=True)

    plt.show()
