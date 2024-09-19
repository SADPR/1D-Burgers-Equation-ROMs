# pinn_burgers_testing_tensorflow.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

# Ensure TensorFlow uses float64 precision
tf.keras.backend.set_floatx("float64")

# Load the pre-trained model
model_path = "burgers_model.h5"  # Ensure this path matches the saved model path
burgers_net = tf.keras.models.load_model(model_path)

# Define a function to make it easier to use the model
u = lambda t, x: burgers_net(tf.concat([t, x], axis=1))

### Plot 1: Solution at different time points across space
plt.figure(figsize=(5, 5), dpi=300)
n_temporal, n_spatial = 10, 2000
v = np.zeros([n_spatial, 2])  # Combine (x, t) as a vector
v[:, 1] = np.linspace(-1, +1, n_spatial)  # Spatial points

# Loop over different time points and plot the solution
for i in range(n_temporal):
    v[:, 0] = i / n_temporal  # Set the time
    plt.plot(v[:, 1], burgers_net.predict(v), label=f"t = {i/n_temporal:.2f}", lw=0.75)

# Configure the plot
plt.legend(loc="upper right", fontsize=5)
plt.xlim(-1, +1)
plt.ylim(-1, +1)
plt.ylabel("u(t, x)")
plt.xlabel("x")
plt.tight_layout()
plt.savefig("u-constant-time.png")
plt.show()

### Plot 2: Color mesh plot of the solution u(x, t)

# Generate a grid of points
n, m = 100, 200
X = np.linspace(-1., +1., m)
T = np.linspace(0., 1., n)
X0, T0 = np.meshgrid(X, T)
X = X0.reshape([n * m, 1])
T = T0.reshape([n * m, 1])

# Convert to TensorFlow tensors
T, X = map(tf.convert_to_tensor, [T, X])

# Plot color mesh
plt.figure(figsize=(5, 2), dpi=600)
U = burgers_net(tf.concat([T, X], axis=1)).numpy().reshape(n, m)  # Get predictions and reshape
plt.pcolormesh(T0, X0, U, cmap=cm.rainbow)
plt.colorbar()
plt.xlim(0., 1.)
plt.ylim(-1., 1.)
plt.clim(-1, 1)
plt.title("u(x, t)")
plt.ylabel("x")
plt.xlabel("t")
plt.tight_layout()
plt.savefig("u-profile.png")
plt.show()

### Plot 3: Solution profiles at different time points

# Generate spatial points
x = np.expand_dims(np.linspace(-1, +1, 200), axis=1)

# Plot for fixed time points
plt.figure(figsize=(9, 3), dpi=300)
for i, time_point in enumerate([0.25, 0.50, 0.75], start=1):
    t = np.ones_like(x) * time_point  # Set time to a constant value
    plt.subplot(1, 3, i)
    plt.title(f"t = {time_point}")
    plt.plot(x, u(t, x))  # Plot the solution for this time point
    plt.ylabel("u(t, x)")
    plt.xlabel("x")
    plt.xlim(-1, +1)
    plt.ylim(-1, +1)

# Configure and save the plot
plt.tight_layout()
plt.savefig("u-vs-x.png")
plt.show()
