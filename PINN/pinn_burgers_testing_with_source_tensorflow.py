# pinn_burgers_testing_with_source_custom_domain.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

# Ensure TensorFlow uses float64 precision
tf.keras.backend.set_floatx("float64")

# Load the pre-trained model
model_path = "burgers_model_with_source_custom_domain.h5"  # Ensure this path matches the saved model path
burgers_net = tf.keras.models.load_model(model_path)

# Define a function to make it easier to use the model
def u(t, x):
    return burgers_net(tf.concat([t, x], axis=1))

# Define the domain ranges
L = 10.0  # Spatial domain [0, 10]
T = 35.0  # Temporal domain [0, 35]

### Plot 1: Solution at different time points across space
plt.figure(figsize=(7, 5), dpi=300)  # Increase figure size for clarity
n_temporal, n_spatial = 10, 2000
v = np.zeros([n_spatial, 2])  # Combine (x, t) as a vector
v[:, 1] = np.linspace(0, L, n_spatial)  # Spatial points in [0, 10]

# Loop over different time points and plot the solution
for i in range(n_temporal):
    v[:, 0] = (i / (n_temporal - 1)) * T  # Set the time in [0, 35]
    plt.plot(v[:, 1], burgers_net.predict(v), label=f"t = {v[0, 0]:.1f}s", lw=0.75)

# Configure the plot
plt.legend(loc="upper right", fontsize=8)
plt.xlim(0, L)
plt.ylim(-1, 6)
plt.ylabel("u(t, x)")
plt.xlabel("x")
plt.title("Solution at Different Time Points Across Space")
plt.tight_layout()
plt.savefig("u-constant-time-custom.png")
plt.show()

### Plot 2: Color mesh plot of the solution u(x, t)

# Generate a grid of points
n, m = 100, 200
X = np.linspace(0., L, m)  # Spatial points in [0, 10]
T_vals = np.linspace(0., T, n)  # Time points in [0, 35]
X0, T0 = np.meshgrid(X, T_vals)
X = X0.reshape([n * m, 1])
T_vals = T0.reshape([n * m, 1])

# Convert to TensorFlow tensors
T_vals, X = map(tf.convert_to_tensor, [T_vals, X])

# Plot color mesh
plt.figure(figsize=(7, 4), dpi=600)
U = burgers_net(tf.concat([T_vals, X], axis=1)).numpy().reshape(n, m)  # Get predictions and reshape
plt.pcolormesh(T0, X0, U, cmap=cm.rainbow)
plt.colorbar()
plt.xlim(0., T)
plt.ylim(0., L)
plt.clim(-1, 6)
plt.title("u(x, t) over the Domain")
plt.ylabel("x")
plt.xlabel("t")
plt.tight_layout()
plt.savefig("u-profile-custom.png")
plt.show()

### Plot 3: Solution profiles at different time points

# Generate spatial points
x = np.expand_dims(np.linspace(0, L, 200), axis=1)

# Define specific time points to visualize
time_points = [5, 15, 25]  # Chosen time points in seconds

# Plot for fixed time points
plt.figure(figsize=(12, 4), dpi=300)
for i, time_point in enumerate(time_points, start=1):
    t = np.ones_like(x) * time_point  # Set time to a constant value
    plt.subplot(1, 3, i)
    plt.title(f"t = {time_point}s")
    plt.plot(x, u(t, x))  # Plot the solution for this time point
    plt.ylabel("u(t, x)")
    plt.xlabel("x")
    plt.xlim(0, L)
    plt.ylim(-1, 6)

# Configure and save the plot
plt.tight_layout()
plt.savefig("u-vs-x-custom.png")
plt.show()


