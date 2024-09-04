import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to perform Jacobian check
def jacobian_check_pod_ann(U_p, snapshot_column, model, epsilon_values):
    # Project the snapshot_column onto the primary POD basis to get q_p
    q_p = np.dot(U_p.T, snapshot_column)

    # Convert q_p to a TensorFlow tensor
    q_p_tensor = tf.convert_to_tensor(q_p, dtype=tf.float32)
    q_p_tensor = tf.expand_dims(q_p_tensor, axis=0)  # Add batch dimension

    # Record the Jacobian using GradientTape
    with tf.GradientTape() as tape:
        tape.watch(q_p_tensor)
        q_s = model(q_p_tensor)

    # Compute the Jacobian of q_s with respect to q_p_tensor
    jacobian = tape.jacobian(q_s, q_p_tensor).numpy().squeeze()

    # Generate a random unit vector v of the same size as q_p
    v = np.random.randn(*q_p.shape)
    v /= np.linalg.norm(v) + 1e-10  # Add small value to prevent division by zero

    # Initialize list to store errors
    errors = []

    for epsilon in epsilon_values:
        # Perturb q_p and compute the ANN output for the perturbed q_p
        q_p_perturbed = q_p + epsilon * v
        q_p_perturbed_tensor = tf.convert_to_tensor(q_p_perturbed, dtype=tf.float32)
        q_p_perturbed_tensor = tf.expand_dims(q_p_perturbed_tensor, axis=0)
        q_s_perturbed = model(q_p_perturbed_tensor).numpy().squeeze()

        # Calculate the error term
        q_s_np = q_s.numpy().squeeze()
        error = np.linalg.norm(q_s_perturbed - q_s_np - epsilon * (jacobian @ v))
        errors.append(error)

    # Plot the errors against epsilon
    plt.figure(figsize=(8, 6))
    plt.loglog(epsilon_values, errors, marker='o', label='Computed Error')

    # Add reference lines for linear (O(epsilon)) and quadratic (O(epsilon^2)) behavior
    plt.loglog(epsilon_values, epsilon_values * errors[0] / epsilon_values[0], 'r--', label=r'O($\epsilon$) Reference')
    plt.loglog(epsilon_values, epsilon_values**2 * errors[0] / epsilon_values[0]**2, 'g--', label=r'O($\epsilon^2$) Reference')

    plt.xlabel('epsilon')
    plt.ylabel('Error')
    plt.title('Jacobian Check Error vs. Epsilon')
    plt.grid(True)
    plt.legend()

    # Compute and print the slope of log(err) vs log(epsilon)
    slopes = np.diff(np.log(errors)) / np.diff(np.log(epsilon_values))
    print("Slopes between consecutive points on the log-log plot:", slopes)

    plt.show()

# Example usage
snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
snapshot = np.load(snapshot_file)  # Assuming the snapshot is in the correct shape

# Select a specific column, e.g., column 100
snapshot_column = snapshot[:, 0]

U_p = np.load('U_p.npy')  # Load your primary POD basis
epsilon_values = np.logspace(np.log10(1e-12), np.log10(10), 20)

# Load the saved model (SavedModel format)
model = tf.keras.models.load_model('pod_ann_model')

# Perform Jacobian check
jacobian_check_pod_ann(U_p, snapshot_column, model, epsilon_values)


