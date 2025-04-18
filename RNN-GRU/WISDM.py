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
DATASET_PATH = "../WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
ACTIVITIES = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
ACTIVITIES_MAP = {
    "Downstairs": 0,
    "Jogging": 1,
    "Sitting": 2,
    "Standing": 3,
    "Upstairs": 4,
    "Walking": 5,
}
## Training
BATCH_SIZE = 64
EPOCHS = 100
WINDOW_SIZE = 60
STRIDE = 30
LEARNING_RATE = 0.0003
CRITERION = nn.CrossEntropyLoss()


class HAR_GRU(nn.Module):
    """
    Defines a PyTorch Gated Recurrent Unit (GRU) network.

    This network has 3 layers, each of size 32.
    """

    def __init__(self, input_size=3, hidden_size=32, num_layers=3, num_classes=6):
        super(HAR_GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lnr = nn.Linear(hidden_size, num_classes)

    def forward(self, x) -> int:
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


def load_WIDSM(path_to_data: str) -> pd.DataFrame:
    """Process WIDSM_ar_v1.1_raw.txt

    Data is not well formatted, so we skip lines where:
    - Timestamp is 0
    - Any of the other fields in an entry are invalid (empty or don't make sense)
    """
    data = []
    with open(path_to_data, "r") as file:
        for line in file:
            # First, split on ';' since lines can have multiple entries
            entry = line.strip().split(";")
            for e in entry:
                # Now, split line by ','
                split = e.strip().split(",")
                if (
                    split[0] == ""
                    or split[1] == ""
                    or split[2] == ""
                    or split[3] == ""
                    or split[4] == ""
                    or split[5] == ""
                ):
                    # Skip any lines where a field within an entry (activity, timestamp, user, ax, ay, az) is missing
                    continue
                if split[2] == "0":
                    # if timestamp is 0 (invalid), skip
                    continue
                user = int(split[0])
                activity = ACTIVITIES_MAP[split[1]]
                ax = float(split[3])
                ay = float(split[4])
                az = float(split[5])
                data.append([user, activity, ax, ay, az])
    return pd.DataFrame(data, columns=["user", "activity", "ax", "ay", "az"])


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
    plt.title("WISDM GRU training loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("WISDM_GRU_LOSS.png")


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
    create_cm(true_values, predictions, ACTIVITIES)


if __name__ == "__main__":
    # Seed for deterministic behaviour
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # Load WIDSM data from WIDSM_ar_v1.1/WIDSM_ar_v1.1_raw.txt
    df = load_WIDSM(DATASET_PATH)
    # Split data into training and testing sets by users
    unique_users = df["user"].unique()
    train_users, test_users = train_test_split(
        unique_users, test_size=0.2, random_state=SEED
    )
    # Define train and test dataframes
    train_df = df[df["user"].isin(train_users)]
    test_df = df[df["user"].isin(test_users)]
    x_train, y_train = train_df[["ax", "ay", "az"]].values, train_df[
        "activity"
    ].values.astype(int)
    x_test, y_test = test_df[["ax", "ay", "az"]].values, test_df[
        "activity"
    ].values.astype(int)
    # Apply sliding window transformation
    x_train, y_train = sliding_window(x_train, y_train)
    x_test, y_test = sliding_window(x_test, y_test)
    # Normalize the data feature-wise
    s = StandardScaler()
    x_train = x_train.reshape(-1, 1, 3)
    x_test = x_test.reshape(-1, 1, 3)
    x_train = s.fit_transform(x_train.reshape(-1, 3)).reshape(-1, WINDOW_SIZE, 3)
    x_test = s.transform(x_test.reshape(-1, 3)).reshape(-1, WINDOW_SIZE, 3)

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
