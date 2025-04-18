import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
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
dataset_path = r"C:\Users\18ah11\Documents\QueensU\Masters\ELEC825\WISDM_ar_v1.1"


# Separates dataframes into sliding windows
def create_sliding_windows(df, window_size, stride):
        X, y = [], []
        for user in df['user_id'].unique():
            user_df = df[df['user_id'] == user]
            for i in range(0, len(user_df) - window_size, stride):
                segment = user_df.iloc[i : i + window_size]
                X.append(segment[['x', 'y', 'z']].values)
                y.append(segment['activity_id'].mode()[0])
        return np.array(X), np.array(y)


def load_dataset(dataset_path, batch_size):

    # Define columns
    columns = ['user_id', 'activity', 'timestamp', 'x', 'y', 'z']
    data_path = os.path.join(dataset_path, 'WISDM_ar_v1.1_raw.txt')
    
    # Define dataframe
    df = pd.read_csv(data_path, header=None, names=columns, comment=';', on_bad_lines="skip")

    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce')
    
    # Ignore NaN elements
    df = df.dropna()

    # Map activity labels to numerical values
    activity_map = {
        'Walking': 0,
        'Jogging': 1,
        'Upstairs': 2,
        'Downstairs': 3,
        'Sitting': 4,
        'Standing': 5
    }
    df['activity_id'] = df['activity'].map(activity_map)

    # Split data into training and testing sets by users
    unique_users = df['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=5)

    # Define train and test dataframes
    train_df = df[df['user_id'].isin(train_users)]
    test_df = df[df['user_id'].isin(test_users)]

    # Extract features and labels
    X_train, y_train = train_df[['x', 'y', 'z']].values, train_df['activity_id'].values
    X_test, y_test = test_df[['x', 'y', 'z']].values, test_df['activity_id'].values

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 3)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, 3)).reshape(X_test.shape)

    # Apply sliding windows to dataframes
    X_train, y_train = create_sliding_windows(train_df, window_size, stride)
    X_test, y_test = create_sliding_windows(test_df, window_size, stride)

    # Reshape for CNN+LSTM
    X_train = X_train.reshape(X_train.shape[0], window_size, 3)  
    X_test = X_test.reshape(X_test.shape[0], window_size, 3)

    # Create PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Create training and testing datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create training and testing PyTorch dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Done loading datasets!")

    return train_loader, test_loader, df, activity_map


# Plot class distribution for unbalanced dataset
def plot_class_distribution(df, activity_map):
    class_counts = df['activity_id'].value_counts().sort_index()
    class_labels = [k for k, v in sorted(activity_map.items(), key=lambda item: item[1])]
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, class_counts)
    plt.title("Class Distribution in WISDM Dataset")
    plt.xlabel("Activity")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    

def initialize_weights(m):
    # Check to make sure weight initialization is done only on linear layers
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        # Initialize parameters using He initialization
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)



# ================================================== CNN Architecture ==================================================

class CNN(nn.Module):
    def __init__(self, input_height, num_classes):
        super(CNN, self).__init__()
        
        # Use 1d convolutions for sequential data
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
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
        self.dropout = nn.Dropout(0.2)


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
batch_size = 128
input_height = 46
num_classes = 6
epochs = 100
learningRate = 0.0005

model = CNN(input_height, num_classes).to(device)

# Sliding window size and stride
window_size = 64
stride = 32

# Apply the weight initialization to the model
model.apply(initialize_weights)
# Set optimizer to Adam
optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-4)
# Set learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# Set loss function to Cross Entropy
criterion = nn.CrossEntropyLoss()

# Load dataset
train_loader, test_loader, full_df, activity_map = load_dataset(dataset_path, batch_size)

# Plot class distribution
plot_class_distribution(full_df, activity_map)


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
            probabilities = nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, dim=1)
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
