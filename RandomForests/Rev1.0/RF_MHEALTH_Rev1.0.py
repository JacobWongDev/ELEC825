import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
dataset_path = r'C:\Users\rando\OneDrive\Desktop\Queens Coursework\ELEC 825 - MACHINE LEARNING AND DEEP LEARNING\Project\Datasets\MHEALTH_Dataset\MHEALTHDATASET'

WINDOW_SIZE = 200  
OVERLAP = 0.0      
ORIGINAL_SAMPLING_RATE = 50  
TARGET_SAMPLING_RATE = 20    
TEST_SIZE = 0.1

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load data
def load_mhealth_data(dataset_path):
    all_data = []
    all_labels = []
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    patterns = [
        os.path.join(dataset_path, f"mHealth_subject*.log"),
        os.path.join(dataset_path, f"MHEALTH_subject*.txt"),
        os.path.join(dataset_path, f"subject*.csv"),
        os.path.join(dataset_path, "*/subject*.log"),
        os.path.join(dataset_path, "*/mHealth_subject*.txt"),
        os.path.join(dataset_path, "*.csv"),
        os.path.join(dataset_path, "*/*.csv"),
        os.path.join(dataset_path, "*/*/*.csv"),
        os.path.join(dataset_path, "*.txt"),
        os.path.join(dataset_path, "*/*.txt"),
        os.path.join(dataset_path, "*/*/*.txt"),
    ]
    
    files_found = False
    
    for pattern in patterns:
        files = glob.glob(pattern)
        
        if files:
            files_found = True
            for filename in files:
                try:
                    if filename.endswith('.csv'):
                        data = pd.read_csv(filename, header=None)
                    else:
                        try:
                            data = pd.read_csv(filename, header=None, sep='\t')
                        except:
                            data = pd.read_csv(filename, header=None, sep=',')
                    
                    if len(data) == 0:
                        continue
                    
                    X = data.iloc[:, :-1].values
                    y = data.iloc[:, -1].values
                    
                    all_data.append(X)
                    all_labels.append(y)
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            
            if all_data:
                break
    
    if not files_found:
        raise FileNotFoundError(f"No files found in {dataset_path}. Check the path and file naming convention.")
    
    if not all_data:
        raise ValueError("No data could be loaded from the files found. Check file format.")
        
    X = np.vstack(all_data)
    y = np.hstack(all_labels)
    
    return X, y


# # Plot the activity distribution
# def plot_activity_distribution(y):
#     # Count the occurrences of each activity label
#     unique_activities, counts = np.unique(y, return_counts=True)
    
#     # Create a bar plot for activity distribution
#     plt.figure(figsize=(10, 6))
#     plt.bar(unique_activities, counts, color='skyblue')
#     plt.xlabel('Activity Labels')
#     plt.ylabel('Number of Instances')
#     plt.title('Activity Distribution')
#     plt.xticks(unique_activities)
#     plt.show()

# Preprocess data
def preprocess_data(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    mask = y != 0
    X = X[mask]
    y = y[mask]
    
    y = y - 1
    
    if ORIGINAL_SAMPLING_RATE > TARGET_SAMPLING_RATE:
        ds_factor = ORIGINAL_SAMPLING_RATE // TARGET_SAMPLING_RATE
        X_ds = X[::ds_factor]
        y_ds = y[::ds_factor]
        
        return X_ds, y_ds
    else:
        return X, y
    

# Create sliding windows
def create_windows(X, y, window_size=WINDOW_SIZE, overlap=OVERLAP):
    step = int(window_size * (1 - overlap))
    n_features = X.shape[1]
    
    windows = []
    window_labels = []
    
    for i in range(0, len(X) - window_size + 1, step):
        window = X[i:i+window_size]
        
        window_label = stats.mode(y[i:i+window_size], keepdims=True)[0][0]
        
        windows.append(window)
        window_labels.append(window_label)
    
    windows = np.array(windows)
    window_labels = np.array(window_labels)
    
    return windows, window_labels

# Train and evaluate Random Forest
def train_and_evaluate_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    
    # Create and train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Flatten the data for Random Forest
    
    # Predictions
    y_pred = rf_model.predict(X_test.reshape(X_test.shape[0], -1))  # Flatten the test data
    
    # Evaluate performance
    evaluate_with_metrics(y_test, y_pred)
    
    return rf_model

# Evaluate with metrics
def evaluate_with_metrics(y_true, y_pred):
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_true, y_pred, average='macro'))

if __name__ == "__main__":
    X, y = load_mhealth_data(dataset_path)
    X, y = preprocess_data(X, y)
    X_windows, y_windows = create_windows(X, y)
    train_and_evaluate_rf(X_windows, y_windows)
    plot_activity_distribution(y)
