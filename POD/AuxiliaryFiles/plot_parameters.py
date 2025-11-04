import numpy as np
import matplotlib.pyplot as plt

# LaTeX-style settings
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.autolayout': False
})

# === Training grid (3x3 uniform) ===
mu1_vals = np.linspace(4.25, 5.50, 3)
mu2_vals = np.linspace(0.015, 0.030, 3)
mu1_grid, mu2_grid = np.meshgrid(mu1_vals, mu2_vals)
mu1_train = mu1_grid.flatten()
mu2_train = mu2_grid.flatten()

# === Explicit testing points ===
mu1_test = np.array([4.75, 4.56, 5.19])
mu2_test = np.array([0.020, 0.019, 0.026])

# === Plotting ===
plt.figure(figsize=(6, 5))
plt.scatter(mu1_train, mu2_train, color='black', label=r'\textbf{Training points}')
# plt.scatter(mu1_test, mu2_test, color='blue', marker='x', s=80, label=r'\textbf{Testing points}')

plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.title(r'\textbf{Parameter sampling in the domain} $\mathcal{D}$')
plt.grid(True)
plt.legend(loc='upper left')
plt.xlim(4,5.75)
plt.ylim(0.01,0.035)
plt.tight_layout()

# === Save ===
plt.savefig("training_parametric_domain.png", format="png", bbox_inches='tight')
plt.show()
