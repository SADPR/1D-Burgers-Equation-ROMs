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
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Prepare the data
data_path = 'training_data/'  # Replace with your data folder
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
all_snapshots = []

for file in files:
    snapshots = np.load(file)
    all_snapshots.append(snapshots)

all_snapshots = np.hstack(all_snapshots)  # Ensure shape is (248000, 513)

# Convert to PyTorch tensors
all_snapshots = torch.tensor(all_snapshots.T, dtype=torch.float32)  # Transpose to (248000, 513)

# Create dataset and dataloaders
dataset = TensorDataset(all_snapshots, all_snapshots)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Adjust the batch size based on the number of time steps
time_steps = 500  # Number of time steps per simulation
batch_size = 500  # Choose a batch size that matches the time steps or is a multiple
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
input_dim = all_snapshots.shape[1]
latent_dim = 16
model = DenseAutoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
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

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

# Save the trained model
torch.save(model.state_dict(), 'dense_autoencoder_elu.pth')

# Encode the data
def encode_data(model, data):
    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(data).numpy()
    return encoded_data

encoded_snapshots = encode_data(model, all_snapshots)
np.save('encoded_snapshots.npy', encoded_snapshots)

# Decode the data
def decode_data(model, encoded_data):
    model.eval()
    with torch.no_grad():
        decoded_data = model.decoder(torch.tensor(encoded_data, dtype=torch.float32)).numpy()
    return decoded_data

# Example of decoding
encoded_snapshots = np.load('encoded_snapshots.npy')
decoded_snapshots = decode_data(model, encoded_snapshots)

# Save decoded snapshots for comparison
np.save('decoded_snapshots.npy', decoded_snapshots)
