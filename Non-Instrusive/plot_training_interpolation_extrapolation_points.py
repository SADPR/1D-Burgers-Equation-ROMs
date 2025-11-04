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

# selected evaluation points
interp_mu  = np.array([4.750, 0.0200])
extrap_mu  = np.array([6.20, 0.0400])#([5.650, 0.0330])



# =========================
# PLOT 1 — interpolation
# =========================
plt.figure(figsize=(6,5))
plt.scatter(mu1_train, mu2_train, color='black', label=r'\textbf{Training points}')
plt.scatter(interp_mu[0], interp_mu[1], color='blue', marker='x', s=150, label=r'\textbf{Interpolation point}')

plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.title(r'\textbf{Interpolation Case in Parameter Domain}')
plt.grid(True)
plt.legend(loc='upper left')
plt.xlim(4.0,5.80)
plt.ylim(0.010,0.035)
plt.tight_layout()
plt.savefig("interpolation_case_parametric_domain.png", format="png", bbox_inches='tight')
plt.close()


# =========================
# PLOT 2 — extrapolation
# =========================
plt.figure(figsize=(6,5))
plt.scatter(mu1_train, mu2_train, color='black', label=r'\textbf{Training points}')
plt.scatter(extrap_mu[0], extrap_mu[1], color='green', marker='*', s=200, label=r'\textbf{Extrapolation point}')

plt.xlabel(r'$\mu_1$')
plt.ylabel(r'$\mu_2$')
plt.title(r'\textbf{Extrapolation Case Beyond Parameter Domain}')
plt.grid(True)
plt.legend(loc='upper left')
plt.xlim(4.0,6.95)
plt.ylim(0.010,0.05)
plt.tight_layout()
plt.savefig("extrapolation_case_parametric_domain.png", format="png", bbox_inches='tight')
plt.close()
