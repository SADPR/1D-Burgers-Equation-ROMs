# pinn_burgers_training_tensorflow.py
import os
import numpy as np
import tensorflow as tf
import scipy.optimize

# Ensure TensorFlow uses float64 precision
tf.keras.backend.set_floatx("float64")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Custom Layer for Computing Gradients
class GradientLayer(tf.keras.layers.Layer):
    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(**kwargs)

    def call(self, x):
        with tf.GradientTape() as g:
            g.watch(x)
            with tf.GradientTape() as gg:
                gg.watch(x)
                u = self.model(x)
            du_dtx = gg.batch_jacobian(u, x)
            du_dt = du_dtx[..., 0]
            du_dx = du_dtx[..., 1]
        d2u_dx2 = g.batch_jacobian(du_dx, x)[..., 1]
        return u, du_dt, du_dx, d2u_dx2

# Neural Network Architecture
class Network:
    @classmethod
    def build(cls, num_inputs=2, layers=[32, 16, 32], activation='tanh', num_outputs=1):
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation, kernel_initializer='he_normal')(x)
        outputs = tf.keras.layers.Dense(num_outputs, kernel_initializer='he_normal')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Optimizer using L-BFGS-B
class L_BFGS_B:
    def __init__(self, model, x_train, y_train, factr=1e7, m=50, maxls=50, maxiter=4500):
        self.model = model
        self.x_train = [tf.constant(x, dtype=tf.float32) for x in x_train]
        self.y_train = [tf.constant(y, dtype=tf.float32) for y in y_train]
        self.factr = factr
        self.m = m
        self.maxls = maxls
        self.maxiter = maxiter
        self.iteration = 0  # Initialize iteration counter

    def set_weights(self, flat_weights):
        shapes = [w.shape for w in self.model.get_weights()]
        split_ids = np.cumsum([np.prod(shape) for shape in [0] + shapes])
        weights = [flat_weights[from_id:to_id].reshape(shape)
                   for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes)]
        self.model.set_weights(weights)

    @tf.function
    def tf_evaluate(self, x, y):
        with tf.GradientTape() as g:
            loss = tf.reduce_mean(tf.keras.losses.mse(self.model(x), y))
        grads = g.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def evaluate(self, weights):
        self.set_weights(weights)
        loss, grads = self.tf_evaluate(self.x_train, self.y_train)
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([g.numpy().flatten() for g in grads]).astype('float64')
        return loss, grads

    def callback(self, weights):
        # Increment iteration counter
        self.iteration += 1
        # Evaluate current loss
        loss, _ = self.evaluate(weights)
        # Print epoch (iteration) number and corresponding loss
        print(f"Epoch (Iteration): {self.iteration} | Loss: {loss:.6f}")

    def fit(self):
        initial_weights = np.concatenate([w.flatten() for w in self.model.get_weights()])
        print(f'Optimizer: L-BFGS-B (maxiter={self.maxiter})')
        # Perform the L-BFGS-B optimization and track loss after each iteration
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights,
                                     factr=self.factr, m=self.m, maxls=self.maxls,
                                     maxiter=self.maxiter, callback=self.callback)

# Physics-Informed Neural Network
class PINN:
    def __init__(self, network, nu):
        self.network = network
        self.nu = nu
        self.grads = GradientLayer(self.network)

    def build(self):
        tx_eqn = tf.keras.layers.Input(shape=(2,))
        tx_ini = tf.keras.layers.Input(shape=(2,))
        tx_bnd = tf.keras.layers.Input(shape=(2,))

        u, du_dt, du_dx, d2u_dx2 = self.grads(tx_eqn)

        u_eqn = du_dt + u * du_dx - self.nu * d2u_dx2
        u_ini = self.network(tx_ini)
        u_bnd = self.network(tx_bnd)

        return tf.keras.models.Model(inputs=[tx_eqn, tx_ini, tx_bnd], outputs=[u_eqn, u_ini, u_bnd])

# Main Function for Training and Saving Model
if __name__ == '__main__':
    num_train_samples = 10000
    nu = 0.01 / np.pi

    # Build the network model
    network = Network.build()
    network.summary()

    # Build the PINN model
    pinn = PINN(network, nu).build()

    # Create training input
    tx_eqn = np.random.rand(num_train_samples, 2)
    tx_eqn[..., 1] = 2 * tx_eqn[..., 1] - 1
    tx_ini = 2 * np.random.rand(num_train_samples, 2) - 1
    tx_ini[..., 0] = 0
    tx_bnd = np.random.rand(num_train_samples, 2)
    tx_bnd[..., 1] = 2 * np.round(tx_bnd[..., 1]) - 1

    u_eqn = np.zeros((num_train_samples, 1))
    u_ini = np.sin(-np.pi * tx_ini[..., 1, np.newaxis])
    u_bnd = np.zeros((num_train_samples, 1))

    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [u_eqn, u_ini, u_bnd]

    # Train the model using L-BFGS-B algorithm
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # Save the trained model
    network.save('burgers_model.h5')
    print("Model saved as burgers_model.h5")

