import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX rendering with serif font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Domain and discretization settings
a = 0
b = 100
m = int(256 * 2)
X = np.linspace(a, b, m + 1)

# Time discretization
At = 0.05  # time step used during simulation
times_of_interest = [5, 10, 15, 20, 25]
time_indices = [int(t / At) for t in times_of_interest]

# Parameters
mu1 = 4.875
mu2 = 0.0225

# Load the simulation data
file_path = f"simulation_results/fem_simulation_mu1_{mu1:.3f}_mu2_{mu2:.4f}.npy"
U_FOM = np.load(file_path)

# Plot
plt.figure(figsize=(8, 5))
for idx in time_indices:
    plt.plot(X, U_FOM[:, idx], color='black')

# Axes and formatting
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$u(x,t)$', fontsize=14)
plt.title(r'$\boldsymbol{\mu} = [4.875,\; 0.0225]$', fontsize=14)
plt.xlim([0, 100])
plt.ylim([0, 8])
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("overlayed_fom_mu1_4.875_mu2_0.0225.pdf", format="pdf")
plt.show()
