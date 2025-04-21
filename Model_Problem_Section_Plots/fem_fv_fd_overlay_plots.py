import numpy as np
import matplotlib.pyplot as plt
import os

# LaTeX-friendly plot settings
plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"],
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rc('font', size=14)

# Grid of parameter combinations
mu1_values = np.linspace(4.25, 5.50, 3)
mu2_values = np.linspace(0.0150, 0.0300, 3)
parameter_combinations = [(mu1, mu2) for mu1 in mu1_values for mu2 in mu2_values]

# Domain and time setup
a, b = 0.0, 100.0
dt = 0.05
times_to_plot = [5, 10, 15, 20, 25]
frame_indices = [int(t / dt) for t in times_to_plot]

# Output directory
output_dir = "overlay_comparisons_FEM_FV_FD"
os.makedirs(output_dir, exist_ok=True)

# Generate plots
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

        # Determine y-axis range dynamically
        u_min = min(U_FEM.min(), U_FV.min(), U_FD.min())
        u_max = max(U_FEM.max(), U_FV.max(), U_FD.max())
        margin = 0.1 * (u_max - u_min)
        y_min, y_max = u_min - margin, u_max + margin

        # Plot
        plt.figure(figsize=(10, 5))
        for idx in frame_indices:
            plt.plot(x_fem, U_FEM[:, idx], color='black', linestyle='-', label='FEM' if idx == frame_indices[0] else "")
            plt.plot(x_fv, U_FV[:, idx], color='green', linestyle='--', label='FV' if idx == frame_indices[0] else "")
            plt.plot(x_fd, U_FD[:, idx], color='red', linestyle='-.', label='FD' if idx == frame_indices[0] else "")

        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x,t)$')
        plt.title(r'$\bm{\mu} = [%.3f,\; %.4f]$' % (mu1, mu2), fontsize=14)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        fname = f"discretization_comparison_overlay_mu1_{mu1:.3f}_mu2_{mu2:.4f}.pdf"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    except FileNotFoundError as e:
        print(f"Missing file for mu1 = {mu1:.3f}, mu2 = {mu2:.4f}: {e}")


