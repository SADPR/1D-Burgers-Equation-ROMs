import numpy as np
import os

# -------- helper: # modes for a tolerance --------
def get_num_modes(tol):
    modes_file = f"../modes/U_modes_tol_{tol:.0e}.npy"
    return np.load(modes_file).shape[1]

# -------- helper: integrated (root-sum-squared) rel error --------
def integrated_rel_error(U_FOM, U_ROM):
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)   # Frobenius norms

# -------- inputs --------
test_points = [
    (4.75, 0.0200),
    (4.56, 0.0190),
    (5.19, 0.0260)
]
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# -------- compute --------
records = []   # (tol, n_modes, max_gal%, max_lspg%)

for tol in tolerances:
    errs_gal, errs_lspg = [], []

    for mu1, mu2 in test_points:
        t1, t2 = f"{mu1:.3f}", f"{mu2:.4f}"

        try:
            U_FOM   = np.load(f"../../FEM/fem_testing_data/fem_simulation_mu1_{t1}_mu2_{t2}.npy")
            U_GAL   = np.load(f"rom_solutions/U_PROM_tol_{tol:.0e}_mu1_{t1}_mu2_{t2}_galerkin.npy")
            U_LSPG  = np.load(f"rom_solutions/U_PROM_tol_{tol:.0e}_mu1_{t1}_mu2_{t2}_lspg.npy")

            errs_gal.append(integrated_rel_error(U_FOM, U_GAL)   * 100)  # in %
            errs_lspg.append(integrated_rel_error(U_FOM, U_LSPG) * 100)

        except FileNotFoundError as e:
            print("Missing file:", e)
            errs_gal.append(np.nan)
            errs_lspg.append(np.nan)

    records.append((
        tol,
        get_num_modes(tol),
        np.nanmax(errs_gal),
        np.nanmax(errs_lspg)
    ))

# -------- LaTeX table --------
print("\n% --- LaTeX table: integrated (root-sum-squared) relative errors ---")
print("\\begin{table}[!htbp]")
print("    \\centering")
print("    \\begin{tabular}{c c c c}")
print("        \\toprule")
print("        \\begin{tabular}[c]{@{}c@{}}Tolerance\\\\ $\\epsilon^2$\\end{tabular} &")
print("        \\begin{tabular}[c]{@{}c@{}}Modes\\\\ $n$\\end{tabular} &")
print("        \\begin{tabular}[c]{@{}c@{}}Galerkin\\\\ $\\mathbb{RE}_{2, \\mathbf{u}}$ (\\%)\\end{tabular} &")
print("        \\begin{tabular}[c]{@{}c@{}}LSPG\\\\ $\\mathbb{RE}_{2, \\mathbf{u}}$ (\\%)\\end{tabular} \\\\")
print("        \\midrule")

for tol, n_modes, e_gal, e_lspg in records:
    exp = int(-np.log10(tol))
    print(f"        $10^{{-{exp}}}$  & {n_modes:<3d} & {e_gal:6.2f} & {e_lspg:6.2f} \\\\")

print("        \\bottomrule")
print("    \\end{tabular}")
print("    \\caption{Integrated (root--sum--squared) relative $L_2$ errors across three test configurations for Galerkin and LSPG PROMs.}")
print("    \\label{tab:prom_errors_summary}")
print("\\end{table}")

