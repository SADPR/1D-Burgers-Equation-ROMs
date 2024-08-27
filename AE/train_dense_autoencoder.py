import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

# Define the Autoencoder model with dense layers and ELU activations
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DenseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 513),
            nn.ELU(),
            nn.Linear(513, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 513),
            nn.ELU(),
            nn.Linear(513, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Prepare the data
data_path = '../FEM/training_data/'  # Replace with your data folder
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)

# Compute the mean and standard deviation for normalization
mean = np.mean(all_snapshots)
std = np.std(all_snapshots)

# Save the mean and standard deviation for future use
np.save('data_mean.npy', mean)
np.save('data_std.npy', std)

# Convert to PyTorch tensors and normalize using PyTorch's operations
all_snapshots = torch.tensor(all_snapshots.T, dtype=torch.float32)  # Transpose to (248000, 513)
all_snapshots = (all_snapshots - mean) / std

# Create dataset and dataloaders
dataset = TensorDataset(all_snapshots, all_snapshots)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
input_dim = all_snapshots.shape[1]
latent_dim = 3  # Set the latent dimension here
model = DenseAutoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize lists to track losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                outputs = model(data)
                loss = criterion(outputs, data)
                val_loss += loss.item() * data.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    # Save the train and validation losses
    np.save(f'train_losses_latent_{latent_dim}.npy', np.array(train_losses))
    np.save(f'val_losses_latent_{latent_dim}.npy', np.array(val_losses))

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=10)

# Save the complete trained model
torch.save(model, f'dense_autoencoder_complete_latent_{latent_dim}.pth')



