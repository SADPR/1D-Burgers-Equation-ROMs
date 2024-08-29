import tensorflow as tf
import numpy as np

# Define the DenseAutoencoder model
class DenseAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(513, activation='elu'),
            tf.keras.layers.Dense(256, activation='elu'),
            tf.keras.layers.Dense(128, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(latent_dim),
        ], name="encoder")
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(128, activation='elu'),
            tf.keras.layers.Dense(256, activation='elu'),
            tf.keras.layers.Dense(513, activation='elu'),
            tf.keras.layers.Dense(input_dim),
        ], name="decoder")

    def call(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def compute_jacobian(self, q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            decoded = self.decoder(tf.expand_dims(q, axis=0))
        jacobian = tape.jacobian(decoded, q)
        return tf.squeeze(jacobian, axis=0)

# Function to compute the finite difference Jacobian
def finite_difference_jacobian(model, q, h=1e-5):
    q = tf.identity(q)
    num_outputs = model.decoder(tf.expand_dims(q, axis=0)).shape[1]
    num_inputs = q.shape[0]
    jacobian = np.zeros((num_outputs, num_inputs))

    for i in range(num_inputs):
        q_plus = tf.identity(q)
        q_minus = tf.identity(q)

        q_plus = q_plus.numpy()
        q_minus = q_minus.numpy()

        q_plus[i] += h
        q_minus[i] -= h

        output_plus = model.decoder(tf.expand_dims(q_plus, axis=0)).numpy().squeeze()
        output_minus = model.decoder(tf.expand_dims(q_minus, axis=0)).numpy().squeeze()

        jacobian[:, i] = (output_plus - output_minus) / (2 * h)

    return jacobian

# Function to compare the analytical and numerical Jacobians
def compare_jacobians(analytical_jacobian, numerical_jacobian):
    difference = np.linalg.norm(analytical_jacobian - numerical_jacobian)
    print(f"Difference between analytical and finite difference Jacobians: {difference}")
    return difference

# Example usage
if __name__ == "__main__":
    # Load your trained autoencoder model
    input_dim = 513
    latent_dim = 28

    # Instantiate the model
    autoencoder = DenseAutoencoder(input_dim, latent_dim)

    # Build the model by calling it on dummy data
    dummy_input = tf.random.normal([1, input_dim])
    autoencoder(dummy_input)

    # Load the model weights into the instantiated model
    autoencoder.load_weights(f'dense_autoencoder_complete_latent_{latent_dim}_tensorflow.h5')

    # Load the snapshot
    snapshot_file = '../FEM/training_data/simulation_mu1_4.76_mu2_0.0182.npy'
    snapshot = np.load(snapshot_file)  # Assuming the snapshot is transposed to match the input dimension

    # Select a specific column, e.g., column 100
    snapshot_column = snapshot[:, 100]

    # Convert the selected column to a TensorFlow tensor
    snapshot_tensor = tf.convert_to_tensor(snapshot_column, dtype=tf.float32)

    # Encode the selected snapshot column to obtain q
    q = autoencoder.encoder(tf.expand_dims(snapshot_tensor, axis=0))
    q = tf.squeeze(q, axis=0)

    # Compute the analytical Jacobian
    analytical_jacobian = autoencoder.compute_jacobian(q).numpy()

    # Compute the numerical Jacobian using finite differences
    numerical_jacobian = finite_difference_jacobian(autoencoder, q)

    # Compare the two Jacobians
    difference = compare_jacobians(analytical_jacobian, numerical_jacobian)

    # Define a tolerance to check if the difference is acceptable
    tolerance = 1e-6
    if difference < tolerance:
        print("The Jacobian seems to be correctly implemented.")
    else:
        print("There might be an issue with the Jacobian implementation.")




