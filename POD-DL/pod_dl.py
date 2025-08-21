import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

all_snapshots = np.hstack(all_snapshots)  # Shape: (N, n_snapshots)

# Compute POD
U, S, VT = np.linalg.svd(all_snapshots, full_matrices=False)

n = 96  # Total number of POD modes
V = U[:, :n]  # POD basis
q = V.T @ all_snapshots  # Generalized coordinates (n, n_snapshots)

# Normalize q
q_mean = np.mean(q, axis=1, keepdims=True)
q_std = np.std(q, axis=1, keepdims=True)
q_normalized = (q - q_mean) / q_std

# Save V, mean and std for later use
np.save('V.npy', V)
np.save('q_mean.npy', q_mean)
np.save('q_std.npy', q_std)

# Transpose to shape (n_snapshots, n)
q_tensor = torch.tensor(q_normalized.T, dtype=torch.float32)

# Split data
q_train, q_test = train_test_split(q_tensor, test_size=0.1, random_state=42)
train_dataset = torch.utils.data.TensorDataset(q_train, q_train)
test_dataset = torch.utils.data.TensorDataset(q_test, q_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers=[128]):
        super(Autoencoder, self).__init__()

        # Encoder: input_dim → ... → latent_dim
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ELU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))  # Final to latent
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: latent_dim → reversed hidden layers → input_dim
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ELU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))  # Final to input_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# Initialize model
input_dim = n
latent_dim = 5  # You can vary this
model = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)

# Training loop with early stopping
def train_autoencoder(model, train_loader, test_loader, criterion, optimizer, scheduler,
                      num_epochs=1000, clip_value=1.0, patience=20, save_path='autoencoder_model.pth'):

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, _ in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                output = model(data)
                val_loss += criterion(output, data).item() * data.size(0)
        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping logic
        if val_loss < best_val_loss - 1e-6:  # Small improvement threshold
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
                break

    # Restore best model and save it
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved to {save_path}")


# Train
train_autoencoder(model, train_loader, test_loader, criterion, optimizer, scheduler)

# Save model
torch.save(model.state_dict(), 'autoencoder_model.pth')
