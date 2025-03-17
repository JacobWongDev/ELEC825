import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

data_path = "../UCI HAR Dataset/"

# List of signal files for training and testing (order matters)
signal_names = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]

def load_signals(p, split):
    signals = []
    for name in signal_names:
        filename = os.path.join(data_path + p, f"{name}_{split}.txt")
        data = np.loadtxt(filename)
        # data shape: (num_samples, 128)
        signals.append(data)
    # Stack along the last axis: result shape (num_samples, 128, 9)
    return np.stack(signals, axis=-1)

# Load training and test signals
X_train = load_signals(p="train/Inertial Signals/", split="train")
X_test  = load_signals(p="test/Inertial Signals/", split="test")

# Load labels (files have one label per sample)
y_train = np.loadtxt(os.path.join(data_path, "train/y_train.txt")).astype(int) - 1  # make 0-indexed
y_test  = np.loadtxt(os.path.join(data_path, "test/y_test.txt")).astype(int) - 1

print("X_train shape:", X_train.shape)  # expected: (7352, 128, 9)
print("X_test shape:", X_test.shape)    # expected: (2947, 128, 9)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

# Create datasets and dataloaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a simple GRU model for HAR
class HAR_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(HAR_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state: shape (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        # Use the last time step's output for classification
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 9     # 9 signals per time step
hidden_size = 64
num_classes = 6    # 6 activity classes
num_layers = 1
num_epochs = 20
learning_rate = 0.001

# Instantiate model, loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HAR_GRU(input_size, hidden_size, num_classes, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Model evaluation on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
