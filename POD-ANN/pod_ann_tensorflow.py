import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# Load snapshot data
data_path = '../FEM/fem_training_data/'
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)

# Perform SVD on the snapshots without normalization
U, S, VT = np.linalg.svd(all_snapshots, full_matrices=False)

# Set the number of modes for principal and secondary modes
r = 17  # Number of principal modes
R = 96  # Total number of modes

U_p = U[:, :r]
U_s = U[:, r:R]
U_combined = np.hstack((U_p, U_s))  # Combine U_p and U_s into a single matrix

# Save U_p and U_s
np.save('U_p.npy', U_p)
np.save('U_s.npy', U_s)

# Define the ANN model in TensorFlow
class POD_ANN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(POD_ANN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(32, activation='elu')
        self.fc2 = tf.keras.layers.Dense(64, activation='elu')
        self.fc3 = tf.keras.layers.Dense(128, activation='elu')
        self.fc4 = tf.keras.layers.Dense(256, activation='elu')
        self.fc5 = tf.keras.layers.Dense(256, activation='elu')
        self.fc6 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return self.fc6(x)

# Prepare training data
q = np.dot(U_combined.T, all_snapshots)  # Project snapshots onto the combined POD basis
q_p = q[:r, :]  # Principal modes
q_s = q[r:R, :]  # Secondary modes

# Convert to TensorFlow tensors
q_p_tensor = tf.convert_to_tensor(q_p.T, dtype=tf.float32)  # Transpose to (n_samples, r)
q_s_tensor = tf.convert_to_tensor(q_s.T, dtype=tf.float32)  # Transpose to (n_samples, R-r)

# Split data into training and testing sets
q_p_train, q_p_test, q_s_train, q_s_test = train_test_split(q_p_tensor.numpy(), q_s_tensor.numpy(), test_size=0.2, random_state=42)

# Initialize the model, loss function, and optimizer
input_dim = r
output_dim = R - r
model = POD_ANN(input_dim, output_dim)

# Loss function and optimizer
criterion = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# Compile the model
model.compile(optimizer=optimizer, loss=criterion)

# Prepare the dataset for TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((q_p_train, q_s_train)).batch(64).shuffle(buffer_size=1024)
test_dataset = tf.data.Dataset.from_tensor_slices((q_p_test, q_s_test)).batch(64)

# Train the model
model.fit(train_dataset, validation_data=test_dataset, epochs=100, callbacks=[scheduler])

# Save the entire model using the SavedModel format
model.save('pod_ann_model', save_format='tf')

