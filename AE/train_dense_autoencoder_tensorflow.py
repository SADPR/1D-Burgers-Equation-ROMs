import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import numpy as np
import os

# Define the Autoencoder model with dense layers and ELU activations
def create_autoencoder(input_dim, latent_dim):
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(513, activation='elu')(encoder_input)
    x = layers.Dense(256, activation='elu')(x)
    x = layers.Dense(128, activation='elu')(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.Dense(32, activation='elu')(x)
    encoder_output = layers.Dense(latent_dim)(x)
    
    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='elu')(decoder_input)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.Dense(128, activation='elu')(x)
    x = layers.Dense(256, activation='elu')(x)
    x = layers.Dense(513, activation='elu')(x)
    decoder_output = layers.Dense(input_dim)(x)
    
    # Autoencoder model
    encoder = models.Model(encoder_input, encoder_output, name='encoder')
    decoder = models.Model(decoder_input, decoder_output, name='decoder')
    autoencoder = models.Model(encoder_input, decoder(encoder(encoder_input)), name='autoencoder')
    
    return autoencoder, encoder, decoder

# Prepare the data
data_path = '../FEM/training_data/'  # Replace with your data folder
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)

# Normalize data
all_snapshots = all_snapshots.T.astype(np.float32)  # Transpose to (248000, 513)

# Create dataset and split
dataset = tf.data.Dataset.from_tensor_slices((all_snapshots, all_snapshots))
train_size = int(0.8 * len(all_snapshots))
val_size = len(all_snapshots) - train_size
train_dataset = dataset.take(train_size).shuffle(buffer_size=1024).batch(64)
val_dataset = dataset.skip(train_size).batch(64)

# Initialize the model, loss function, and optimizer
input_dim = all_snapshots.shape[1]
latent_dim = 28  # Set the latent dimension here
autoencoder, encoder, decoder = create_autoencoder(input_dim, latent_dim)
autoencoder.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=losses.MeanSquaredError())

# Learning rate scheduler
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# Early stopping
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Training the model
history = autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=[lr_scheduler, early_stopping]
)

# Save the training and validation losses
train_losses = history.history['loss']
val_losses = history.history['val_loss']
np.save(f'train_losses_latent_{latent_dim}_tensorflow.npy', np.array(train_losses))
np.save(f'val_losses_latent_{latent_dim}_tensorflow.npy', np.array(val_losses))

# Save the complete trained model
autoencoder.save(f'dense_autoencoder_complete_latent_{latent_dim}_tensorflow.h5')
