import numpy as np
import matplotlib.pyplot as plt
import os

# Enable LaTeX text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_single_Q(modes, q):
    ''' Populates Q row by row '''
    k = int(modes*(modes+1)/2)
    Q = np.empty(k)
    index = 0
    for i in range(modes):
        for j in range(i, modes):
            Q[index] = q[i]*q[j]
            index += 1
    return Q

def compute_l2_norm_error(U_FOM, U_ROM):
    """Compute the L2 norm error between the FOM and ROM over the entire matrix."""
    return np.linalg.norm(U_FOM - U_ROM) / np.linalg.norm(U_FOM)

def plot_reconstruction_comparison():
    # Load domain
    a, b = 0, 100
    m = 511
    X = np.linspace(a, b, m + 1)

    # Time discretization
    At = 0.05
    times_of_interest = [5, 10, 15, 20, 25]  # seconds
    time_indices = [int(t / At) for t in times_of_interest]

    # Load the precomputed basis and H matrix
    Phi_p = np.load("U_truncated.npy")
    H = np.load("H_quadratic.npy")

    # Load a specific snapshot for reconstruction
    snapshot_file = '../FEM/fem_testing_data/fem_simulation_mu1_4.750_mu2_0.0200.npy'
    snapshot = np.load(snapshot_file)
    num_modes = Phi_p.shape[1]

    # Initialize arrays to store the reconstructed snapshots
    reconstructed_snapshots_pod = np.zeros_like(snapshot)
    reconstructed_snapshots_quadratic = np.zeros_like(snapshot)

    # Calculate reconstructions for each snapshot
    for i in range(snapshot.shape[1]):
        original_snapshot = snapshot[:, i]

        # Project the snapshot onto the reduced basis
        q_snapshot = Phi_p.T @ original_snapshot

        # Reconstruct using POD
        reconstructed_pod = Phi_p @ q_snapshot
        reconstructed_snapshots_pod[:, i] = reconstructed_pod

        # Reconstruct using Quadratic Approximation
        Q_single = get_single_Q(num_modes, q_snapshot)
        reconstructed_quadratic = Phi_p @ q_snapshot + H @ Q_single
        reconstructed_snapshots_quadratic[:, i] = reconstructed_quadratic

    # Calculate and print L2 norm errors over the entire matrices
    l2_error_pod = compute_l2_norm_error(snapshot, reconstructed_snapshots_pod)
    l2_error_quadratic = compute_l2_norm_error(snapshot, reconstructed_snapshots_quadratic)
    print(f"Total POD L2 Error = {l2_error_pod:.6f}")
    print(f"Total Quadratic L2 Error = {l2_error_quadratic:.6f}")

    # Plotting
    plt.figure(figsize=(7, 6))

    # Plot original snapshots (FOM) for the selected times
    plt.plot(X, snapshot[:, time_indices[0]], color='black', linestyle='-', linewidth=2, label='FOM')
    for t_index in time_indices[1:]:
        plt.plot(X, snapshot[:, t_index], color='black', linestyle='-', linewidth=2)

    # Plot POD reconstructed snapshots
    plt.plot(X, reconstructed_snapshots_pod[:, time_indices[0]], color='blue', linestyle='--', linewidth=2, label='POD')
    for t_index in time_indices[1:]:
        plt.plot(X, reconstructed_snapshots_pod[:, t_index], color='blue', linestyle='--', linewidth=2)

    # Plot Quadratic reconstructed snapshots
    plt.plot(X, reconstructed_snapshots_quadratic[:, time_indices[0]], color='red', linestyle='-.', linewidth=2, label='Quadratic')
    for t_index in time_indices[1:]:
        plt.plot(X, reconstructed_snapshots_quadratic[:, t_index], color='red', linestyle='-.', linewidth=2)

    # Annotate the plot with the reconstruction methods
    plt.title(f'FOM vs POD vs Quadratic Reconstruction at Different Times')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')
    plt.ylim(0, 7)
    plt.xlim(0, 100)
    plt.legend(loc='upper right')
    plt.grid(True)

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the figure as a PDF
    plt.savefig("fom_pod_quadratic_comparison.pdf", format='pdf')

    plt.show()

# Call the plotting function
plot_reconstruction_comparison()


