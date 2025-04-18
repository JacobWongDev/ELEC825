import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Global variables
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "../UCI HAR Dataset/"
SENSOR_TYPES = [
    "body_ax",
    "body_ay",
    "body_az",
    "body_gx",
    "body_gy",
    "body_gz",
    "total_ax",
    "total_ay",
    "total_az",
]
DATA_COLUMNS = SENSOR_TYPES + ["activity"]
## Training
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.004
CRITERION = nn.CrossEntropyLoss()


class HAR_BiGRU(nn.Module):
    """
    Defines a PyTorch Bidirectional Gated Recurrent Unit (BiGRU) network.

    This network has 3 layers, each of size 32.
    """

    def __init__(self, input_size=9, hidden_size=32, num_layers=3, num_classes=6):
        super(HAR_BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.lnr = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x) -> int:
        out, _ = self.gru(x)
        return self.lnr(out[:, -1, :])


def load_UCI_HAR(path_to_folder: str) -> pd.DataFrame:
    """
    Process each of the Intertial Signals files

    NOTE: UCI-HAR is already pre-processed with a sliding window size 128.
    """
    signals = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_",
    ]
    signals_data = []
    for signal in signals:
        signals_data.append(
            np.loadtxt(
                path_to_folder + "train/Inertial Signals/" + signal + "train.txt",
                dtype=float,
            )
        )
    x_train = np.stack(signals_data, axis=-1)
    signals_data.clear()
    for signal in signals:
        signals_data.append(
            np.loadtxt(
                path_to_folder + "test/Inertial Signals/" + signal + "test.txt",
                dtype=float,
            )
        )
    x_test = np.stack(signals_data, axis=-1)
    y_train = np.loadtxt(path_to_folder + "train/y_train.txt").astype(int) - 1
    y_test = np.loadtxt(path_to_folder + "test/y_test.txt").astype(int) - 1
    return x_train, x_test, y_train, y_test


def train(train_loader, model):
    """
    Train model using ADAM optimizer. Uses the Cross-Entropy loss function.
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    model.train()
    for epoch in range(EPOCHS):
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.title("UCI-HAR bidirectional GRU training loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("UCI_HAR_GRU_LOSS.png")


def train_with_scheduler(train_loader, validation_loader, model):
    """
    Train model using ADAM optimizer and scheduler to decrease learning rate.
    Uses the Cross-Entropy loss function.
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=5, cooldown=3
    )
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        scheduler.step(val_acc)


def create_cm(true_values, predictions, class_names):
    """
    Create confusion matrix.
    """
    cm = confusion_matrix(true_values, predictions)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("")
    plt.savefig("cm.png")


def evaluate(data_loader, model):
    """
    Evaluate model performance using test dataset.
    """
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            p = torch.argmax(out, dim=1)
            true_values.extend(y.cpu().numpy())
            predictions.extend(p.cpu().numpy())
    print("Accuracy is: ", accuracy_score(true_values, predictions))
    print(
        "Classification report:\n",
        classification_report(true_values, predictions, zero_division=0),
    )
    create_cm(
        true_values,
        predictions,
        ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"],
    )


def dlr():
    """
    Train and test model with scheduler during optimization.
    """
    # Load UCI-HAR Data
    x_train, x_test, y_train, y_test = load_UCI_HAR(DATASET_PATH)
    # Split data into training and validation sets for scheduler
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state=SEED
    )
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor), batch_size=BATCH_SIZE
    )
    validation_loader = DataLoader(
        TensorDataset(x_valid_tensor, y_valid_tensor), batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor), batch_size=BATCH_SIZE
    )
    model = HAR_BiGRU()
    model.to(DEVICE)
    train_with_scheduler(train_loader, validation_loader, model)
    evaluate(test_loader, model)


def raw():
    """
    Train and test model without scheduler during optimization.
    """
    # Load UCI-HAR Data
    x_train, x_test, y_train, y_test = load_UCI_HAR(DATASET_PATH)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor), batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor), batch_size=BATCH_SIZE
    )
    model = HAR_BiGRU()
    model.to(DEVICE)
    train(train_loader, model)
    evaluate(test_loader, model)


if __name__ == "__main__":
    # Seed for deterministic behaviour
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dlr", action="store_true", help="Train model with scheduler on learning rate"
    )
    parser.add_argument("--raw", action="store_true", help="Train model on raw data")
    args = parser.parse_args()
    if args.raw:
        raw()
    elif args.dlr:
        dlr()
    else:
        print("Error: No mode specified!")
        parser.print_help()
