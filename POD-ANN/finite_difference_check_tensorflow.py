import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to perform Jacobian check using finite difference method
def finite_difference_jacobian_check(U_p, snapshot_column, model, epsilon_values, epsilon_fd=1e-5):
    # Project the snapshot_column onto the primary POD basis to get q_p
    q_p = np.dot(U_p.T, snapshot_column)

    # Convert q_p to a TensorFlow tensor
    q_p_tensor = tf.convert_to_tensor(q_p, dtype=tf.float32)
    q_p_tensor = tf.expand_dims(q_p_tensor, axis=0)  # Add batch dimension

    # Compute the analytical Jacobian using TensorFlow
    with tf.GradientTape() as tape:
        tape.watch(q_p_tensor)
        q_s = model(q_p_tensor)

    # Compute the Jacobian of q_s with respect to q_p_tensor
    jacobian_analytical = tape.jacobian(q_s, q_p_tensor).numpy().squeeze()

    # Initialize Jacobian from finite difference approximation
    jacobian_fd = np.zeros_like(jacobian_analytical)

    # Compute the finite difference approximation of the Jacobian
    for i in range(q_p.shape[0]):  # Loop over input dimensions
        # Perturb the input in the i-th dimension
        perturb = np.zeros_like(q_p)
        perturb[i] = epsilon_fd

        # Forward and backward perturbations
        q_p_plus = tf.convert_to_tensor(q_p + perturb, dtype=tf.float32)
        q_p_minus = tf.convert_to_tensor(q_p - perturb, dtype=tf.float32)
        
        q_p_plus = tf.expand_dims(q_p_plus, axis=0)
        q_p_minus = tf.expand_dims(q_p_minus, axis=0)

        # Model evaluations at perturbed points
        q_s_plus = model(q_p_plus).numpy().squeeze()
        q_s_minus = model(q_p_minus).numpy().squeeze()

        # Finite difference approximation
        jacobian_fd[:, i] = (q_s_plus - q_s_minus) / (2 * epsilon_fd)

    # Compare the analytical Jacobian with the finite difference Jacobian
    diff = np.linalg.norm(jacobian_analytical - jacobian_fd)/np.linalg.norm(jacobian_fd)
    print(f'Norm of difference between analytical and finite difference Jacobian: {diff}')

    return jacobian_analytical, jacobian_fd

# Example usage
snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
snapshot = np.load(snapshot_file)  # Assuming the snapshot is in the correct shape

# Select a specific column, e.g., column 100
snapshot_column = snapshot[:, 100]

U_p = np.load('U_p.npy')  # Load your primary POD basis
epsilon_values = np.logspace(-12, -1, 10)  # Values of epsilon to test

# Load the saved model (SavedModel format)
model = tf.keras.models.load_model('pod_ann_model')

# Perform finite difference Jacobian check
jacobian_analytical, jacobian_fd = finite_difference_jacobian_check(U_p, snapshot_column, model, epsilon_values)
