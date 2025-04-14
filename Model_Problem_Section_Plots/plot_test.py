import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu1 = 5.500
mu2 = 0.0225
dt = 0.05
times_to_plot = [0, 5, 10, 15, 20, 25]
frame_indices = [int(t / dt) for t in times_to_plot]
a, b = 0.0, 100.0

# Load FEM data
U_FEM = np.load(f"../FEM/fem_training_data/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
N_fem = U_FEM.shape[0]
dx_fem = (b - a) / (N_fem - 1)
x_fem = np.linspace(a, b, N_fem)

# Load FV data
U_FV = np.load(f"../FV/fv_training_data/fv_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
N_fv = U_FV.shape[0]
dx_fv = (b - a) / N_fv
x_fv = np.linspace(a + dx_fv / 2, b - dx_fv / 2, N_fv)

# Load FD data
U_FD = np.load(f"../FD/fd_training_data/fd_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
N_fd = U_FD.shape[0]
dx_fd = (b - a) / (N_fd - 1)
x_fd = np.linspace(a, b, N_fd)

# Plot
plt.figure(figsize=(10, 5))
for idx in frame_indices:
    plt.plot(x_fem, U_FEM[:, idx], color='blue', linestyle='-', label='FEM' if idx == frame_indices[0] else "")
    plt.plot(x_fv, U_FV[:, idx], color='green', linestyle='--', label='FV' if idx == frame_indices[0] else "")
    plt.plot(x_fd, U_FD[:, idx], color='red', linestyle='-.', label='FD' if idx == frame_indices[0] else "")

plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title(f'Overlay Comparison for mu1 = {mu1:.3f}, mu2 = {mu2:.4f}\nTimes: {times_to_plot} s')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"comparison_overlay_mu1_{mu1:.3f}_mu2_{mu2:.4f}.png")
