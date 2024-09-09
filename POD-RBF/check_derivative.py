import numpy as np
import matplotlib.pyplot as plt
import pickle
import jax.numpy as jnp

# Function to compute Euclidean distances between two sets of points
def compute_distances(X1, X2):
    """Compute pairwise Euclidean distances between two sets of points."""
    dists = jnp.sqrt(jnp.sum((X1 - X2) ** 2, axis=-1))
    return dists

# Function to compute the Gaussian RBF kernel
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return jnp.exp(-(epsilon * r) ** 2)

# Function to interpolate using precomputed weights
def interpolate_with_weights(X_train, W, x_new, epsilon):
    """Interpolate at new points using precomputed weights."""
    dists = compute_distances(x_new[None, :], X_train)  # Compute distances between the new point and training points
    rbf_values = gaussian_rbf(dists, epsilon)  # Apply RBF kernel
    f_new = rbf_values @ W  # Use precomputed weights for interpolation
    return f_new

# Function to compute the Jacobian manually
def manual_jacobian(q_p_train, W, q_p_sample, epsilon):
    """Compute the Jacobian manually for the RBF interpolation."""
    N = q_p_train.shape[0]  # Number of training points
    input_dim = q_p_train.shape[1]  # Dimension of the input (q_p)
    output_dim = W.shape[1]  # Dimension of the output (q_s)

    # Initialize the Jacobian matrix
    jacobian = jnp.zeros((output_dim, input_dim))  # Shape: (273, 28)

    # Compute distances from the sample point to the training points
    for i in range(N):
        q_p_i = q_p_train[i]  # i-th training point
        r_i = compute_distances(q_p_sample, q_p_i)  # Distance between q_p_sample and q_p_i

        # RBF kernel value
        phi_r_i = gaussian_rbf(r_i, epsilon)

        # Derivative of the RBF kernel with respect to q_p_sample
        dphi_dq_p = -2 * epsilon**2 * (q_p_sample - q_p_i) * phi_r_i  # Shape: (1, 28)

        # Outer product to compute the contribution to the Jacobian
        jacobian += jnp.outer(W[i], dphi_dq_p.reshape(-1))  # Outer product to match (273, 28)

    return jacobian

# Function to check the gradient of the POD-RBF model
def gradient_check_pod_rbf(U_p, snapshot_column, q_p_train, W, epsilon_values, epsilon):
    # Project the snapshot_column onto the primary POD basis to get q_p
    q_p = U_p.T @ snapshot_column

    # Compute the Jacobian manually
    jacobian = manual_jacobian(q_p_train, W, q_p, epsilon)

    # Generate a random unit vector v of the same size as q_p
    v = np.random.randn(*q_p.shape)
    v /= np.linalg.norm(v) + 1e-10  # Add small value to prevent division by zero

    # Initialize list to store errors
    errors = []

    for eps in epsilon_values:
        # Perturb q_p and compute the RBF output for the perturbed q_p
        q_p_perturbed = q_p + eps * v
        q_s_perturbed = interpolate_with_weights(q_p_train, W, q_p_perturbed, epsilon)

        # Calculate the error term
        q_s_original = interpolate_with_weights(q_p_train, W, q_p, epsilon)
        error = np.linalg.norm(q_s_perturbed - q_s_original - eps * (jacobian @ v))
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
    # Load the RBF model data (q_p_train and precomputed weights W)
    with open('rbf_weights.pkl', 'rb') as f:
        q_p_train, W = pickle.load(f)

    # Load the snapshot data and select a specific column, e.g., column 100
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)  # Assuming the snapshot is in the correct shape
    snapshot_column = snapshot[:, 0]

    # Load U_p (primary POD basis)
    U_p = np.load('U_p.npy')

    # Define epsilon values for the gradient check
    epsilon_values = np.logspace(np.log10(1e-12), np.log10(10), 20)

    # Perform gradient check for the POD-RBF model
    gradient_check_pod_rbf(U_p, snapshot_column, q_p_train, W, epsilon_values, epsilon=1.0)





