"""
MHEALTH Dataset Processing with TCN Model for Human Activity Recognition

This script processes the MHEALTH dataset and trains a Temporal Convolutional Network (TCN)
model for human activity recognition. It includes data loading, preprocessing, feature extraction,
data augmentation, model training with cross-validation, and comprehensive evaluation.

Usage:
    python mhealth_tcn.py [path_to_dataset]
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
import sys
import glob

# Data processing imports
import numpy as np
import pandas as pd
from scipy import stats

# Machine learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, ReLU, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONSTANTS
# =============================================================================

# Data processing parameters
WINDOW_SIZE = 200  # 5 seconds of data at 20Hz sampling rate (after downsampling)
OVERLAP = 0.0      # No overlap between windows (to prevent data leakage)
NUM_CLASSES = 12   # Number of activities in MHEALTH dataset
ORIGINAL_SAMPLING_RATE = 50  # Hz (original data collection rate)
TARGET_SAMPLING_RATE = 20    # Hz (rate after downsampling)

# Training parameters
BATCH_SIZE = 32
MAX_EPOCHS = 100
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
CROSS_VAL_FOLDS = 5

# Model parameters
TCN_FILTERS = 16   # Number of filters in TCN layers
L2_FACTOR = 1e-3   # L2 regularization factor
LEARNING_RATE = 0.0005  # Learning rate for optimizer

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Activity names for reporting
ACTIVITY_NAMES = [
    "Walking", "Jogging", "Stairs", "Sitting", "Standing",
    "Lying Down", "Brushing Teeth", "Combing Hair", "Writing",
    "Eating Soup", "Eating Chips", "Drinking"
]

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_mhealth_data(dataset_path):
    """
    Load the MHEALTH dataset from the specified path
    
    Args:
        dataset_path (str): Path to the MHEALTH dataset
        
    Returns:
        Tuple containing X and y data
    """
    all_data = []
    all_labels = []
    
    # Check if directory exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Try different file patterns and extensions common in MHEALTH dataset
    patterns = [
        # Standard MHEALTH naming convention
        os.path.join(dataset_path, f"mHealth_subject*.log"),
        os.path.join(dataset_path, f"MHEALTH_subject*.txt"),
        os.path.join(dataset_path, f"subject*.csv"),
        # Try looking in numbered directories
        os.path.join(dataset_path, "*/subject*.log"),
        os.path.join(dataset_path, "*/mHealth_subject*.txt"),
        # Look for all CSV files as a fallback
        os.path.join(dataset_path, "*.csv"),
        os.path.join(dataset_path, "*/*.csv"),
        os.path.join(dataset_path, "*/*/*.csv"),
        # Try txt files
        os.path.join(dataset_path, "*.txt"),
        os.path.join(dataset_path, "*/*.txt"),
        os.path.join(dataset_path, "*/*/*.txt"),
    ]
    
    files_found = False
    
    # Try each pattern until files are found
    for pattern in patterns:
        print(f"Searching with pattern: {pattern}")
        files = glob.glob(pattern)
        
        if files:
            files_found = True
            print(f"Found {len(files)} files with pattern {pattern}")
            
            for filename in files:
                try:
                    # Try different delimiters
                    if filename.endswith('.csv'):
                        data = pd.read_csv(filename, header=None)
                    else:
                        try:
                            # Try tab-separated first
                            data = pd.read_csv(filename, header=None, sep='\t')
                        except:
                            # Then try comma-separated
                            data = pd.read_csv(filename, header=None, sep=',')
                    
                    # Check if file contains data
                    if len(data) == 0:
                        print(f"Skipping empty file: {filename}")
                        continue
                    
                    # MHEALTH dataset format: the last column contains activity labels
                    X = data.iloc[:, :-1].values
                    y = data.iloc[:, -1].values
                    
                    all_data.append(X)
                    all_labels.append(y)
                    print(f"Successfully loaded {filename} with shape {X.shape}")
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            
            # If we found data, break out of the pattern loop
            if all_data:
                break
    
    if not files_found:
        raise FileNotFoundError(f"No files found in {dataset_path}. Check the path and file naming convention.")
    
    if not all_data:
        raise ValueError("No data could be loaded from the files found. Check file format.")
        
    # Concatenate all data
    X = np.vstack(all_data)
    y = np.hstack(all_labels)
    
    return X, y

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_data(X, y):
    """
    Preprocess the raw sensor data including downsampling
    
    Args:
        X (numpy.ndarray): Raw sensor data
        y (numpy.ndarray): Activity labels
        
    Returns:
        Preprocessed X and y data
    """
    print(f"Original data shape: {X.shape}, Original sampling rate: {ORIGINAL_SAMPLING_RATE}Hz")
    
    # Normalize the sensor data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Remove samples with label 0 (null activity)
    mask = y != 0
    X = X[mask]
    y = y[mask]
    
    # Adjust labels to start from 0 (subtract 1)
    y = y - 1
    
    # Downsample the data
    if ORIGINAL_SAMPLING_RATE > TARGET_SAMPLING_RATE:
        # Calculate downsampling factor
        ds_factor = ORIGINAL_SAMPLING_RATE // TARGET_SAMPLING_RATE
        print(f"Downsampling from {ORIGINAL_SAMPLING_RATE}Hz to {TARGET_SAMPLING_RATE}Hz (factor: {ds_factor})")
        
        # Create new downsampled arrays
        X_ds = X[::ds_factor]
        y_ds = y[::ds_factor]
        
        print(f"After downsampling - Data shape: {X_ds.shape}")
        return X_ds, y_ds
    else:
        print("No downsampling applied")
        return X, y

def create_windows(X, y, window_size=WINDOW_SIZE, overlap=OVERLAP):
    """
    Create fixed-size windows from sensor data with no overlap to prevent data leakage
    
    Args:
        X (numpy.ndarray): Sensor data
        y (numpy.ndarray): Activity labels
        window_size (int): Size of the window
        overlap (float): Overlap between consecutive windows (should be 0.0)
        
    Returns:
        Windowed X and corresponding y
    """
    # Calculate step size (with 0 overlap, step = window_size)
    step = int(window_size * (1 - overlap))
    n_features = X.shape[1]
    
    windows = []
    window_labels = []
    
    # Print information about windowing
    print(f"Creating windows with size {window_size} and step {step} (no overlap)")
    print(f"Original data shape: {X.shape}")
    
    # Loop through data with non-overlapping windows
    for i in range(0, len(X) - window_size + 1, step):
        window = X[i:i+window_size]
        
        # Use the most frequent label in the window as the window label
        window_label = stats.mode(y[i:i+window_size], keepdims=True)[0][0]
        
        windows.append(window)
        window_labels.append(window_label)
    
    # Convert to numpy arrays
    windows = np.array(windows)
    window_labels = np.array(window_labels)
    
    print(f"Created {len(windows)} windows with shape {windows.shape}")
    
    return windows, window_labels

def augment_data(X, y, noise_level=0.05):
    """
    Apply time series specific data augmentation to reduce overfitting
    
    Args:
        X: Input time series data, shape (n_samples, time_steps, n_features)
        y: Labels
        noise_level: Amount of Gaussian noise to add
        
    Returns:
        Augmented X and corresponding y
    """
    aug_X = []
    aug_y = []
    
    # Copy original data
    aug_X.append(X)
    aug_y.append(y)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, X.shape)
    noisy_X = X + noise
    aug_X.append(noisy_X)
    aug_y.append(y)
    
    # Time warping (stretching/compressing)
    n_samples, time_steps, n_features = X.shape
    warped_X = np.zeros_like(X)
    for i in range(n_samples):
        for j in range(n_features):
            # More aggressive time warping (based on sampling rate knowledge)
            orig_signal = X[i, :, j]
            # Random stretching factor between 0.8 and 1.2
            stretch_factor = 0.8 + 0.4 * np.random.random()
            warped_signal = np.interp(
                np.linspace(0, 1, time_steps),
                np.linspace(0, 1, time_steps) * stretch_factor,
                orig_signal
            )
            warped_X[i, :, j] = warped_signal
    
    aug_X.append(warped_X)
    aug_y.append(y)
    
    # Magnitude scaling (simulates sensor sensitivity variations)
    scaled_X = X.copy()
    # Apply random scaling per feature (between 0.8 and 1.2)
    for j in range(n_features):
        scale_factor = 0.8 + 0.4 * np.random.random()
        scaled_X[:, :, j] = scaled_X[:, :, j] * scale_factor
    
    aug_X.append(scaled_X)
    aug_y.append(y)
    
    # Time shifting (small shifts)
    shifted_X = np.zeros_like(X)
    for i in range(n_samples):
        for j in range(n_features):
            # Random shift between -10% and 10% of window
            shift = int(0.1 * time_steps * (2 * np.random.random() - 1))
            orig_signal = X[i, :, j]
            # Apply shift with zero padding
            if shift > 0:
                shifted_X[i, shift:, j] = orig_signal[:-shift]
                shifted_X[i, :shift, j] = 0
            elif shift < 0:
                shifted_X[i, :shift, j] = orig_signal[-shift:]
                shifted_X[i, shift:, j] = 0
            else:
                shifted_X[i, :, j] = orig_signal
    
    aug_X.append(shifted_X)
    aug_y.append(y)
    
    # Concatenate all augmented data
    print(f"Augmented data: {len(aug_X)} versions (original + {len(aug_X)-1} augmented)")
    return np.vstack(aug_X), np.concatenate(aug_y)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_tcn_model(input_shape, num_classes, tcn_filters=16, l2_factor=1e-3, learning_rate=0.0005):
    """
    Build a TCN (Temporal Convolutional Network) model for activity recognition
    with increased regularization to prevent overfitting and receptive field matched to sampling rate
    
    Args:
        input_shape (tuple): Shape of input data
        num_classes (int): Number of activity classes
        tcn_filters (int): Number of filters in TCN layers
        l2_factor (float): L2 regularization factor
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    # Use Input layer explicitly
    inputs = tf.keras.Input(shape=input_shape)
    
    # Input dropout for robustness (new addition)
    x = Dropout(0.1)(inputs)
    
    # First TCN layer with filter size matched to target sampling rate 
    # (e.g., for 20Hz, kernel_size=5 covers 0.25 seconds)
    tcn_layer_1 = Conv1D(
        filters=tcn_filters, 
        kernel_size=5,  # Increased from 3 to better match sampling rate
        padding='causal', 
        activation=None,
        kernel_regularizer=l2(l2_factor), 
        kernel_initializer='he_normal',
        name='tcn_conv1'
    )(x)
    tcn_layer_1 = BatchNormalization()(tcn_layer_1)
    tcn_layer_1 = ReLU()(tcn_layer_1)
    tcn_layer_1 = MaxPooling1D(pool_size=2)(tcn_layer_1)
    tcn_layer_1 = Dropout(0.5)(tcn_layer_1)
    
    # Second TCN layer with larger receptive field
    tcn_layer_2 = Conv1D(
        filters=tcn_filters, 
        kernel_size=5,  # Increased for better receptive field 
        padding='causal', 
        activation=None,
        kernel_regularizer=l2(l2_factor), 
        kernel_initializer='he_normal',
        name='tcn_conv2'
    )(tcn_layer_1)
    tcn_layer_2 = BatchNormalization()(tcn_layer_2)
    tcn_layer_2 = ReLU()(tcn_layer_2)
    tcn_layer_2 = MaxPooling1D(pool_size=2)(tcn_layer_2)
    tcn_layer_2 = Dropout(0.5)(tcn_layer_2)
    
    # Add a third layer to increase model capacity and receptive field
    tcn_layer_3 = Conv1D(
        filters=tcn_filters*2,  # Double filters
        kernel_size=5,  
        padding='causal', 
        activation=None,
        kernel_regularizer=l2(l2_factor), 
        kernel_initializer='he_normal',
        name='tcn_conv3'
    )(tcn_layer_2)
    tcn_layer_3 = BatchNormalization()(tcn_layer_3)
    tcn_layer_3 = ReLU()(tcn_layer_3)
    tcn_layer_3 = MaxPooling1D(pool_size=2)(tcn_layer_3)
    tcn_layer_3 = Dropout(0.5)(tcn_layer_3)
    
    # Flatten and dense layer 
    x = Flatten()(tcn_layer_3)
    x = Dense(32, activation=None, kernel_regularizer=l2(l2_factor))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.6)(x)  # Increased dropout
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use Adam with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Specify metrics including recall and f1 score
    metrics = [
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    # Print model summary
    model.summary()
    
    return model

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss
    
    Args:
        history: Keras training history
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training loss')
    ax2.plot(history.history['val_loss'], label='Validation loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_with_metrics(model, X_test, y_test, class_names=None):
    """
    Evaluate model with multiple metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        class_names: Optional list of class names
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate recall for each class and their average (macro)
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_per_class = recall_score(y_true, y_pred, average=None)
    
    # Calculate F1 score for each class and their average (macro)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Print final results
    print('Accuracy: {:.2f}%'.format(acc * 100))
    print('Recall: {:.2f}%'.format(recall_macro * 100))
    print('F1-Score: {:.2f}%'.format(f1_macro * 100))
    
    return {
        'accuracy': acc,
        'recall_macro': recall_macro,
        'recall_per_class': recall_per_class,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'y_true': y_true,
        'y_pred': y_pred
    }

def cross_validate_model(X_windows, y_windows, input_shape, num_classes, n_splits=CROSS_VAL_FOLDS):
    """
    Perform k-fold cross-validation and report metrics with standard deviation
    
    Args:
        X_windows: Windowed feature data
        y_windows: Window labels (not one-hot encoded)
        input_shape: Shape for CNN input
        num_classes: Number of activity classes
        n_splits: Number of folds for cross-validation
        
    Returns:
        Dictionary with mean and std of metrics
    """
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # Lists to store metrics from each fold
    avg_acc = []
    avg_recall = []
    avg_f1 = []
    
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    # Iterate through folds
    for i, (train_idx, test_idx) in enumerate(skf.split(X_windows, y_windows)):
        # Split data
        X_train, X_test = X_windows[train_idx], X_windows[test_idx]
        y_train, y_test = y_windows[train_idx], y_windows[test_idx]
        
        # Apply data augmentation to training data
        X_train, y_train = augment_data(X_train, y_train, noise_level=0.05)
        
        # Convert to one-hot encoding
        y_train_onehot = to_categorical(y_train, num_classes=num_classes)
        y_test_onehot = to_categorical(y_test, num_classes=num_classes)
        
        # Build and compile model
        model = build_tcn_model(input_shape, num_classes, TCN_FILTERS, L2_FACTOR, LEARNING_RATE)
        
        # Define callbacks
        callbacks = [
            # Early stopping with increased patience
            EarlyStopping(
                monitor='val_loss', 
                patience=15,
                restore_best_weights=True,
                min_delta=0.001  # Minimum change to qualify as improvement
            ),
            # Learning rate reduction on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train model
        model.fit(
            X_train, y_train_onehot,
            epochs=100,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1,
            shuffle=True  # Explicitly shuffle data
        )
        
        # Evaluate model
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = y_test
        
        # Calculate metrics
        acc_fold = accuracy_score(y_true, y_pred)
        recall_fold = recall_score(y_true, y_pred, average='macro')
        f1_fold = f1_score(y_true, y_pred, average='macro')
        
        # Store metrics
        avg_acc.append(acc_fold)
        avg_recall.append(recall_fold)
        avg_f1.append(f1_fold)
        
        # Print results for this fold
        print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(
            acc_fold, recall_fold, f1_fold, i+1))
        print('______________________________________________________')
        
        # Clear session to free memory
        K.clear_session()
    
    # Calculate standard deviations
    std_acc = np.std(avg_acc, ddof=1)
    std_recall = np.std(avg_recall, ddof=1)
    std_f1 = np.std(avg_f1, ddof=1)
    
    # Calculate means
    mean_acc = np.mean(avg_acc)
    mean_recall = np.mean(avg_recall)
    mean_f1 = np.mean(avg_f1)
    
    # Print cross-validation results
    print("\nCross-Validation Results:")
    print('Accuracy: {:.2f}% ± {:.2f}'.format(mean_acc * 100, std_acc * 100))
    print('Recall: {:.2f}% ± {:.2f}'.format(mean_recall * 100, std_recall * 100))
    print('F1-Score: {:.2f}% ± {:.2f}'.format(mean_f1 * 100, std_f1 * 100))
    
    return {
        'accuracy': {'mean': mean_acc, 'std': std_acc},
        'recall': {'mean': mean_recall, 'std': std_recall},
        'f1_score': {'mean': mean_f1, 'std': std_f1},
        'accuracies': avg_acc,
        'recalls': avg_recall,
        'f1_scores': avg_f1
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    # Parse command line arguments for dataset path
    dataset_path = "dataset/MHEALTHDATASET"
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    print("\n" + "=" * 80)
    print(f"MHEALTH DATASET PROCESSING WITH TCN MODEL")
    print("=" * 80)
    
    print(f"\nUsing dataset path: {dataset_path}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Available files and directories: {os.listdir()}")
    
    # 1. Data Loading
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    X, y = load_mhealth_data(dataset_path)
    X, y = preprocess_data(X, y)
    
    # 2. Feature Extraction
    print("\n" + "=" * 80)
    print("CREATING NON-OVERLAPPING WINDOWS")
    print("=" * 80)
    print("Using non-overlapping windows to prevent data leakage between train/validation/test sets")
    X_windows, y_windows = create_windows(X, y, overlap=0.0)  # Explicitly set overlap to 0
    
    # Input shape for model
    input_shape = (X_windows.shape[1], X_windows.shape[2])
    print(f"Window shape: {X_windows.shape}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Class distribution check
    unique, counts = np.unique(y_windows, return_counts=True)
    print("\nClass distribution in windowed data:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        if i < len(ACTIVITY_NAMES):
            class_name = ACTIVITY_NAMES[int(cls)]
        else:
            class_name = f"Class {int(cls)}"
        print(f"  {class_name}: {count} windows ({count/len(y_windows)*100:.1f}%)")
    
    # 3. Cross-validation
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION")
    print("=" * 80)
    cv_results = cross_validate_model(
        X_windows, y_windows, 
        input_shape=input_shape, 
        num_classes=NUM_CLASSES
    )
    
    # 4. Train Final Model
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    # Convert labels to one-hot encoding for final model
    y_onehot = to_categorical(y_windows, num_classes=NUM_CLASSES)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_windows, y_onehot, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y_windows
    )
    
    # Apply data augmentation
    X_train_aug, y_train_aug_indices = augment_data(
        X_train, np.argmax(y_train, axis=1), noise_level=0.05
    )
    y_train_aug = to_categorical(y_train_aug_indices, num_classes=NUM_CLASSES)
    
    # Build and train model
    model = build_tcn_model(input_shape, NUM_CLASSES, TCN_FILTERS, L2_FACTOR, LEARNING_RATE)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            'best_model.keras', 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # 5. Evaluation
    print("\n" + "=" * 80)
    print("EVALUATING FINAL MODEL")
    print("=" * 80)
    metrics = evaluate_with_metrics(model, X_test, y_test, class_names=ACTIVITY_NAMES)
    
    # 6. Visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    plot_training_history(history)
    
    # 7. Save Results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save metrics to CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Recall', 'F1 Score'],
        'Final Model (%)': [metrics['accuracy'] * 100, metrics['recall_macro'] * 100, metrics['f1_macro'] * 100],
        'CV Mean (%)': [cv_results['accuracy']['mean'] * 100, cv_results['recall']['mean'] * 100, cv_results['f1_score']['mean'] * 100],
        'CV Std (%)': [cv_results['accuracy']['std'] * 100, cv_results['recall']['std'] * 100, cv_results['f1_score']['std'] * 100]
    })
    results_df.to_csv('model_metrics.csv', index=False)
    print("Metrics saved to 'model_metrics.csv'")
    
    # Save model
    model.save('mhealth_tcn_model.keras')
    print("Model saved as 'mhealth_tcn_model.keras'")
    
    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)

# Execute main function when script is run directly
if __name__ == "__main__":
    main()