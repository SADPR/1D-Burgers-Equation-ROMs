import os
import subprocess

# List of setup files to run with the 'build_ext --inplace' command
setup_files = [
    "setup_files/setup_boundary_condition.py",
    "setup_files/setup_convection_matrix_supg.py",
    "setup_files/setup_diffusion_matrix.py",
    "setup_files/setup_forcing_vector.py",
    "setup_files/setup_mass_matrix_parallel.py",
    "setup_files/setup_sparse_solver_parallel.py",
    "setup_files/setup_svd_solver.py",
    "setup_files/setup_eigen_sparse_dense_operations.py",
    "setup_files/setup_mkl_sparse_dense_operations.py"
]

# Function to execute each setup file with the 'build_ext --inplace' command
def run_setup_file(setup_file):
    try:
        print(f"Building {setup_file}...")
        subprocess.run(["python3", setup_file, "build_ext", "--inplace"], check=True)
        print(f"{setup_file} built successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error building {setup_file}: {e}\n")

# Run all setup files
for setup_file in setup_files:
    run_setup_file(setup_file)

