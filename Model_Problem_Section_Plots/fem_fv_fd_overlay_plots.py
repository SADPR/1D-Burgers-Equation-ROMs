import numpy as np
import matplotlib.pyplot as plt
import os

# Grid of parameter combinations
mu1_values = np.linspace(4.25, 5.50, 3)
mu2_values = np.linspace(0.0150, 0.0300, 3)
parameter_combinations = [(mu1, mu2) for mu1 in mu1_values for mu2 in mu2_values]

# Domain and time setup
a, b = 0.0, 100.0
dt = 0.05
times_to_plot = [5, 10, 15, 20, 25]  # Removed t=0
frame_indices = [int(t / dt) for t in times_to_plot]

# Output directory
output_dir = "overlay_comparisons_FEM_FV_FD"
os.makedirs(output_dir, exist_ok=True)

# Generate plots for all 9 parameter combinations
for mu1, mu2 in parameter_combinations:
    try:
        # Load FEM
        U_FEM = np.load(f"../FEM/fem_training_data/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
        N_fem = U_FEM.shape[0]
        x_fem = np.linspace(a, b, N_fem)

        # Load FV
        U_FV = np.load(f"../FV/fv_training_data/fv_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
        N_fv = U_FV.shape[0]
        dx_fv = (b - a) / N_fv
        x_fv = np.linspace(a + dx_fv / 2, b - dx_fv / 2, N_fv)

        # Load FD
        U_FD = np.load(f"../FD/fd_training_data/fd_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy")
        N_fd = U_FD.shape[0]
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
        plt.ylim(0, 8)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_overlay_mu1_{mu1:.3f}_mu2_{mu2:.4f}.png"))
        plt.close()

    except FileNotFoundError as e:
        print(f"Missing file for mu1 = {mu1:.3f}, mu2 = {mu2:.4f}: {e}")

