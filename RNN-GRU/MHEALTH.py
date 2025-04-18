import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Global variables
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "../MHEALTHDATASET"
SENSOR_TYPES = [
    "chest_ax",
    "chest_ay",
    "chest_az",
    "ecg_1",
    "ecg_2",
    "l_ankle_ax",
    "l_ankle_ay",
    "l_ankle_az",
    "l_ankle_gx",
    "l_ankle_gy",
    "l_ankle_gz",
    "l_ankle_mmx",
    "l_ankle_mmy",
    "l_ankle_mmz",
    "r_arm_ax",
    "r_arm_ay",
    "r_arm_az",
    "r_arm_gx",
    "r_arm_gy",
    "r_arm_gz",
    "r_arm_mmx",
    "r_arm_mmy",
    "r_arm_mmz",
]
DATA_COLUMNS = SENSOR_TYPES + ["activity"]
ACTIVITIES = [
    "Standing still",
    "Sitting and relaxing",
    "Supine",
    "Walking",
    "Climbing Stairs",
    "Waist bends forward",
    "Frontal elevation of arms",
    "Crouching",
    "Cycling",
    "Jogging",
    "Running",
    "Jump front and back",
]
## Training
BATCH_SIZE = 64
EPOCHS = 100
WINDOW_SIZE = 90
STRIDE = 45
LEARNING_RATE = 0.003
CRITERION = nn.CrossEntropyLoss()


class HAR_GRU(nn.Module):
    """
    Defines a PyTorch Gated Recurrent Unit (GRU) network.

    This network has 3 layers, each of size 32.
    """

    def __init__(self, input_size=23, hidden_size=32, num_layers=3, num_classes=12):
        super(HAR_GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lnr = nn.Linear(hidden_size, num_classes)

    def forward(self, x) -> int:
        """
        Computes the model output, given input vector x.
        """
        out, _ = self.gru(x)
        return self.lnr(out[:, -1, :])


def sliding_window(features, labels) -> tuple[np.array, np.array]:
    """
    Preprocessing dataset with sliding window.
    The label for each window is selected using majority voting.
    """
    x = []
    y = []
    for i in range(0, len(features) - WINDOW_SIZE, STRIDE):
        majority = Counter(labels[i : i + WINDOW_SIZE]).most_common(1)[0][0]
        x.append(features[i : i + WINDOW_SIZE])
        y.append(majority)
    return np.array(x), np.array(y)


def load_MHEALTH(files: list[str], path_to_folder: str) -> pd.DataFrame:
    """Process each of the mHealth_subject*.log files

    - Ignore any entries with the null class (0).
    - Shift all classes down by 1 (0 - 11) for a total of 12 classes.
    -
    """
    frames = []
    for file in files:
        df = pd.read_csv(path_to_folder + "/" + file, sep=r"\s+", header=None)
        df.columns = DATA_COLUMNS
        df["subject"] = file
        # Shift all classes down by 1 (1 - 12 to 0 - 11)
        df["activity"] = df["activity"] - 1
        frames.append(df)
    total_df = pd.concat(frames, ignore_index=True)
    # ignore entries with class 0 (now class -1 due to shift)
    return total_df[total_df["activity"] != -1]


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
    plt.title("MHEALTH GRU training loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("MHEALTH_GRU_LOSS.png")


def create_cm(true_values, predictions, class_names):
    """
    Create confusion matrix.
    """
    cm = confusion_matrix(true_values, predictions)
    plt.figure(figsize=(10, 8))
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
    plt.subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
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
    create_cm(true_values, predictions, ACTIVITIES)


if __name__ == "__main__":
    # Seed for deterministic behaviour
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # Load MHEALTH data from all mHealth_subject*.log files
    data_files = sorted(
        [f for f in os.listdir(DATASET_PATH) if f.startswith("mHealth_subject")]
    )
    df = load_MHEALTH(data_files, DATASET_PATH)
    # Split data into training and testing sets by users
    unique_users = df["subject"].unique()
    train_users, test_users = train_test_split(
        unique_users, test_size=0.2, random_state=SEED
    )
    # Define train and test dataframes
    train_df = df[df["subject"].isin(train_users)]
    test_df = df[df["subject"].isin(test_users)]
    x_train, y_train = train_df[SENSOR_TYPES].values, train_df[
        "activity"
    ].values.astype(int)
    x_test, y_test = test_df[SENSOR_TYPES].values, test_df["activity"].values.astype(
        int
    )
    # Apply sliding window transformation
    x_train, y_train = sliding_window(x_train, y_train)
    x_test, y_test = sliding_window(x_test, y_test)
    # Normalize the data feature-wise
    s = StandardScaler()
    x_train = x_train.reshape(-1, 1, 23)
    x_test = x_test.reshape(-1, 1, 23)
    x_train = s.fit_transform(x_train.reshape(-1, 23)).reshape(-1, WINDOW_SIZE, 23)
    x_test = s.transform(x_test.reshape(-1, 23)).reshape(-1, WINDOW_SIZE, 23)

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
    model = HAR_GRU()
    model.to(DEVICE)
    train(train_loader, model)
    evaluate(test_loader, model)
