import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pickle

# Function to compute Euclidean distances between two sets of points
def compute_distances(X1, X2):
    """Compute pairwise Euclidean distances between two sets of points."""
    return np.sqrt(np.sum((X1 - X2) ** 2, axis=1))

# Function to compute the Gaussian RBF kernel
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

# Function to interpolate dynamically and compute the Jacobian on the fly
def interpolate_and_compute_jacobian(kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors):
    """Interpolate at new points and compute the Jacobian on the fly using nearest neighbors."""
    dist, idx = kdtree.query(q_p_sample, k=neighbors)  # Find nearest neighbors

    # Extract the neighbor points
    X_neighbors = q_p_train[idx].reshape(neighbors, -1)
    Y_neighbors = q_s_train[idx, :].reshape(neighbors, -1)

    # Compute the pairwise distances between neighbors (to form the Phi matrix)
    dists_neighbors = np.linalg.norm(X_neighbors[:, None, :] - X_neighbors[None, :, :], axis=-1)

    # Compute the RBF matrix for the neighbors (Phi matrix)
    Phi_neighbors = gaussian_rbf(dists_neighbors, epsilon)
    Phi_neighbors += np.eye(neighbors) * 1e-8  # Regularization for numerical stability

    # Solve for the weights using Y_neighbors
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)

    # Compute RBF values between q_p_sample and its neighbors
    rbf_values = gaussian_rbf(dist, epsilon)

    # Compute interpolated value
    f_new = rbf_values @ W_neighbors

    # Compute the Jacobian dynamically
    input_dim = q_p_train.shape[1]
    output_dim = q_s_train.shape[1]
    jacobian = np.zeros((output_dim, input_dim))

    # For each neighbor, compute the contribution to the Jacobian
    for i in range(neighbors):
        q_p_i = X_neighbors[i]
        r_i = compute_distances(q_p_sample, q_p_i.reshape(1, -1))  # Distance between q_p_sample and q_p_i

        # RBF kernel value
        phi_r_i = gaussian_rbf(r_i, epsilon)

        if np.abs(phi_r_i) > 1e-6:
            # Derivative of the RBF kernel with respect to q_p_sample
            dphi_dq_p = -2 * epsilon**2 * (q_p_sample - q_p_i) * phi_r_i

            # Outer product to compute the contribution to the Jacobian
            jacobian += np.outer(W_neighbors[i], dphi_dq_p)

    return f_new, jacobian

# Function to check the gradient of the POD-RBF model dynamically
def gradient_check_pod_rbf(U_p, snapshot_column, q_p_train, q_s_train, epsilon_values, epsilon, kdtree, neighbors):
    # Project the snapshot_column onto the primary POD basis to get q_p
    q_p = U_p.T @ snapshot_column

    # Interpolate and compute the Jacobian dynamically
    f_new, jacobian = interpolate_and_compute_jacobian(kdtree, q_p_train, q_s_train, q_p.reshape(1, -1), epsilon, neighbors)

    # Generate a random unit vector v of the same size as q_p
    v = np.random.randn(*q_p.shape)
    v /= np.linalg.norm(v) + 1e-10  # Normalize and add small value to prevent division by zero

    # Initialize list to store errors
    errors = []

    for eps in epsilon_values:
        # Perturb q_p and compute the RBF output for the perturbed q_p
        q_p_perturbed = q_p + eps * v
        q_s_perturbed, _ = interpolate_and_compute_jacobian(kdtree, q_p_train, q_s_train, q_p_perturbed.reshape(1, -1), epsilon, neighbors)

        # Calculate the error term
        error = np.linalg.norm(q_s_perturbed - f_new - eps * (jacobian @ v))
        errors.append(error)

    # Plot the errors against epsilon
    plt.figure(figsize=(8, 6))
    plt.loglog(epsilon_values, errors, marker='o', label='Computed Error')

    # Add reference lines for linear (O(epsilon)) and quadratic (O(epsilon^2)) behavior
    plt.loglog(epsilon_values, epsilon_values * errors[0] / epsilon_values[0], 'r--', label=r'O($\epsilon$) Reference')
    plt.loglog(epsilon_values, epsilon_values**2 * errors[0] / epsilon_values[0]**2, 'g--', label=r'O($\epsilon^2$) Reference')

    plt.xlabel('epsilon')
    plt.ylabel('Error')
    plt.title('Gradient Check Error vs. Epsilon')
    plt.grid(True)
    plt.legend()

    # Compute and print the slope of log(err) vs log(epsilon)
    slopes = np.diff(np.log(errors)) / np.diff(np.log(epsilon_values))
    print("Slopes between consecutive points on the log-log plot:", slopes)

    plt.show()

if __name__ == '__main__':
    # Load the precomputed KDTree, q_p_train, and q_s_train
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)
        kdtree = data['KDTree']
        q_p_train = data['q_p']
        q_s_train = data['q_s']

    # Load the snapshot data and select a specific column, e.g., column 100
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)  # Assuming the snapshot is in the correct shape
    snapshot_column = snapshot[:, 10]

    # Load U_p (primary POD basis)
    U_p = np.load('U_p.npy')

    # Define epsilon values for the gradient check
    epsilon_values = np.logspace(np.log10(1e-12), np.log10(10), 20)

    # Set RBF parameters
    epsilon = 1.0
    neighbors = 100

    # Perform gradient check for the POD-RBF model
    gradient_check_pod_rbf(U_p, snapshot_column, q_p_train, q_s_train, epsilon_values, epsilon, kdtree, neighbors)







