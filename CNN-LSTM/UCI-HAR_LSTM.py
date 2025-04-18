import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
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


# Function to create sliding windows for the dataset
def create_sliding_window(data, labels, window_size, step_size):
    X = []
    y = []

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data[start:end]
        label = np.bincount(labels[start:end]).argmax()
        X.append(window)
        y.append(label)
    return np.array(X), np.array(y)


# Define file paths (Paths must be set for each unique environment)
#  r'C:\Your\file\path\UCI HAR Dataset\UCI HAR Dataset\train\\'
train_path = r'C:\Users\18ah11\Documents\QueensU\Masters\ELEC825\UCI HAR Dataset\train\\'
test_path = r'C:\Users\18ah11\Documents\QueensU\Masters\ELEC825\UCI HAR Dataset\test\\'

# Load the features (Path must be set for each unique environment)
features = pd.read_csv('C:/Users/18ah11/Documents/QueensU/Masters/ELEC825/UCI HAR Dataset/features.txt', sep=r'\s+', header=None)
feature_names = features[1].values

# Load the activity labels (Path must be set for each unique environment)
activity_labels = pd.read_csv('C:/Users/18ah11/Documents/QueensU/Masters/ELEC825/UCI HAR Dataset/activity_labels.txt', sep=r'\s+', header=None)
activity_labels.columns = ['activity_id', 'activity_name']

# Create a dictionary to count occurrences of feature names
name_counts = {}
unique_feature_names = []

# Loop through feature names to create unique names
for name in feature_names:
    if name in name_counts:
        name_counts[name] += 1
        unique_name = f"{name}_{name_counts[name]}"
    else:
        name_counts[name] = 1
        unique_name = name
    unique_feature_names.append(unique_name)

feature_names = np.array(unique_feature_names)

# Load the training data
X_train = pd.read_csv(train_path + 'X_train.txt', sep=r'\s+', header=None, names=feature_names)
y_train = pd.read_csv(train_path + 'y_train.txt', sep=r'\s+', header=None, names=['activity_id'])
subject_train = pd.read_csv(train_path + 'subject_train.txt', sep=r'\s+', header=None, names=['subject'])

# Load the testing data
X_test = pd.read_csv(test_path + 'X_test.txt', sep=r'\s+', header=None, names=feature_names)
y_test = pd.read_csv(test_path + 'y_test.txt', sep=r'\s+', header=None, names=['activity_id'])
subject_test = pd.read_csv(test_path + 'subject_test.txt', sep=r'\s+', header=None, names=['subject'])


# Calculate mean and standard deviation of training data
features_to_scale = X_train.columns
train_mean = X_train[features_to_scale].mean()
train_std = X_train[features_to_scale].std()

# Standardize training set
X_train[features_to_scale] = (X_train[features_to_scale] - train_mean) / train_std
# Standardize test set based on the mean and std_dev from training set
X_test[features_to_scale] = (X_test[features_to_scale] - train_mean) / train_std

X_train_windowed, y_train_windowed = create_sliding_window(X_train.values, y_train.values.flatten(), window_size=16, step_size=8)
X_test_windowed, y_test_windowed = create_sliding_window(X_test.values, y_test.values.flatten(), window_size=16, step_size=8)

# Define training and testing tensors for PyTorch
X_train_tensor = torch.FloatTensor(X_train_windowed)
y_train_tensor = torch.LongTensor(y_train_windowed)

X_test_tensor = torch.FloatTensor(X_test_windowed)
y_test_tensor = torch.LongTensor(y_test_windowed)

# Zero-index output tensors
y_train_tensor -= 1
y_test_tensor -= 1

# Reshape input data for CNN
X_train_tensor = X_train_tensor.permute(0, 2, 1)
X_test_tensor = X_test_tensor.permute(0, 2, 1)

# Define training and testing datasets
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

# Define training and testing dataloaders
trainLoader = DataLoader(train_data, batch_size=32, shuffle=True)
testLoader = DataLoader(test_data, batch_size=32, shuffle=False)


# ================================================== CNN Architecture ==================================================

class CNN(nn.Module):
    def __init__(self, input_height, num_classes):
        super(CNN, self).__init__()
        
        # Use 1d convolutions for sequential data
        self.conv1 = nn.Conv1d(561, 32, kernel_size=3, padding=1)
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

        x = x.view(x.size(0), -1, self.input_size)

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.bn6(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Define hyperparameters
input_height = 561
num_classes = len(activity_labels['activity_id'].unique())
epochs = 60
learningRate = 0.0005

model = CNN(input_height, num_classes).to(device)


# Initialize the weights of each layer
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
optimizer = optim.Adam(model.parameters(), lr=learningRate)
# Set learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# Set loss function to Cross Entropy
criterion = nn.CrossEntropyLoss()


# Training neural network
def train(model, trainLoader, criterion, optimizer, device):
    loss_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainLoader):
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

        avg_loss = running_loss / len(trainLoader)
        loss_list.append(avg_loss)

        scheduler.step(running_loss / len(trainLoader))
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainLoader):.4f}")
    return loss_list


# Testing neural network
def test(model, testLoader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in testLoader:
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
loss_list = train(model, trainLoader, criterion, optimizer, device)
test(model, testLoader, device)

# Plot training loss over epochs
plt.plot(range(1, epochs + 1), loss_list, label="Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.legend()
plt.show()
