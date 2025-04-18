import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Select device to use for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define file path (Path must be set for each unique environment)
dataset_path = r"C:\Users\18ah11\Documents\QueensU\Masters\ELEC825\MHEALTHDATASET"

# Function to check if a file has integers
def contains_integer(f):
    return bool(re.search(r'\d', f))

# Create a list of only subject data files
files = sorted([f for f in os.listdir(dataset_path) if contains_integer(f)])

# Split subjects into training and testing sets
test_subjects = files[:2]
train_subjects = files[2:]


def load(file_list):
    data = []
    for file in file_list:
        # Define path for each subject
        path = os.path.join(dataset_path, file)
        # Read the .txt files as .csv
        dataframe = pd.read_csv(path, sep='\s+', header=None)
        # Add the entry to the data list
        data.append(dataframe)
    data = pd.concat(data, ignore_index=True)
    return data


# Function to create sliding windows for the dataset
def create_sliding_window(X, y, window_size=128, step_size=64):
    X_windows = []
    y_windows = []
    for start in range(0, len(X) - window_size, step_size):
        end = start + window_size
        window_X = X[start:end]
        window_y = y[start:end]

        label = np.bincount(window_y).argmax()
        X_windows.append(window_X)
        y_windows.append(label)
    return np.array(X_windows), np.array(y_windows)


# Load the training data
train_set = load(train_subjects)
test_set = load(test_subjects)

# Exclude class 0 from training and testing sets
train_set = train_set[train_set.iloc[:, -1] != 0]
test_set = test_set[test_set.iloc[:, -1] != 0]

# Define features and labels in dataframes
X_train = train_set.iloc[:, :-1]
X_test = test_set.iloc[:, :-1]
y_train = train_set.iloc[:, -1]
y_test = test_set.iloc[:, -1]

# Shift labels to be 0-indexed
y_train -= 1
y_test -= 1

# Normalize data to have mean of 0 and variance of 1
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Create sliding windows for data
X_train_seq, y_train_seq = create_sliding_window(X_train_norm, y_train.to_numpy(), window_size=128, step_size=64)
X_test_seq, y_test_seq = create_sliding_window(X_test_norm, y_test.to_numpy(), window_size=128, step_size=64)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).permute(0, 2, 1)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).permute(0, 2, 1)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.long)

# Create DataLoaders
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print("Done loading datasets!")



# ================================================== CNN Architecture ==================================================

class CNN(nn.Module):
    def __init__(self, input_height, num_classes):
        super(CNN, self).__init__()
        
        # Use 1d convolutions for sequential data
        self.conv1 = nn.Conv1d(23, 32, kernel_size=3, padding=1)
        # Apply batch normalization at every nonlinearity
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(64)

        self.input_size = 64
        self.hidden_size = 128

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=False)

        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

        # Apply dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.bn6(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Define hyperparameters
input_height = 23
num_classes = 12
epochs = 50
learningRate = 0.0003

model = CNN(input_height, num_classes).to(device)


def initialize_weights(m):
    # Check to make sure weight initialization is done only on linear layers
    if isinstance(m, nn.Linear):
        # Initialize parameters using He initialization
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# Apply the weight initialization to the model
model.apply(initialize_weights)
# Set optimizer to Adam
optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-4)
# Set learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# Set loss function to Cross Entropy
criterion = nn.CrossEntropyLoss()


# Training neural network
def train(model, train_loader, criterion, optimizer, device):
    loss_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Calculate loss function
            loss = criterion(outputs, labels)
            # Back propagation
            loss.backward()
            # Update weights
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        loss_list.append(avg_loss)

        scheduler.step(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return loss_list


# Testing neural network
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    # Calculate accuracy
    accuracy = 100*correct/total
    # Calculate F1 Score
    f1 = 100*f1_score(all_labels, all_predictions, average='weighted')
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'F1 Score on test set: {f1:.2f}%')


# Train and test the model
loss_list = train(model, train_loader, criterion, optimizer, device)
test(model, test_loader, device)

# Plot training loss over epochs
plt.plot(range(1, epochs + 1), loss_list, label="Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.legend()
plt.show()
