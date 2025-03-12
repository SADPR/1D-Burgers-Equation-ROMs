import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

# Define the Convolutional Autoencoder model for 1D data
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),  # (batch_size, 16, 512)
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),  # Downsample by 2 (batch_size, 16, 256)
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # (batch_size, 32, 256)
            nn.ELU(),
            nn.MaxPool1d(2, stride=2),  # Downsample by 2 (batch_size, 32, 128)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # (batch_size, 64, 128)
            nn.ELU(),
            nn.MaxPool1d(2, stride=2)  # Downsample by 2 (batch_size, 64, 64)
        )

        # Fully connected layers for latent space representation
        self.fc1 = nn.Linear(64 * 64, latent_dim)  # Compress to latent space
        self.fc2 = nn.Linear(latent_dim, 64 * 64)  # Expand back

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # Upsample by 2 (batch_size, 32, 128)
            nn.ELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # Upsample by 2 (batch_size, 16, 256)
            nn.ELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),  # Upsample by 2 (batch_size, 1, 512)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc1(x)  # Compress to latent space
        x = self.fc2(x)  # Expand back
        x = x.view(x.size(0), 64, 64)  # Reshape to match the last Conv layer output shape
        x = self.decoder(x)
        return x

# Prepare the data
data_path = '../training_data/'  # Replace with your data folder
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (513, 500)
print("Original all_snapshots shape:", all_snapshots.shape)

# Reduce the input size to 512 by removing the last node
all_snapshots = all_snapshots[:-1, :]
print("Reduced all_snapshots shape (512 nodes):", all_snapshots.shape)

# Normalize the data
mean = np.mean(all_snapshots)
std = np.std(all_snapshots)
all_snapshots = (all_snapshots - mean) / std

# Convert to PyTorch tensors and reshape for Conv1D (batch_size, channels, width)
all_snapshots = torch.tensor(all_snapshots.T, dtype=torch.float32).unsqueeze(1)  # Shape (500, 1, 512)
print("Reshaped all_snapshots for Conv1D:", all_snapshots.shape)

# Create dataset and dataloaders
dataset = TensorDataset(all_snapshots, all_snapshots)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))

# Initialize the model, loss function, and optimizer
model = ConvAutoencoder(latent_dim=16)  # Specify the latent dimension
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0

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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                outputs = model(data)
                loss = criterion(outputs, data)
                val_loss += loss.item() * data.size(0)

        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_conv_autoencoder.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10)

# Save the trained model
torch.save(model, 'conv_autoencoder_complete.pth')