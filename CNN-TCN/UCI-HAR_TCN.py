"""
UCI HAR Dataset Processing with CNN-TCN Model for Human Activity Recognition

This script processes the UCI HAR dataset and trains a hybrid Convolutional Neural Network (CNN)
and Temporal Convolutional Network (TCN) model for human activity recognition with simplified
processing and training techniques.

Dataset: UCI Human Activity Recognition Using Smartphones Dataset
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Usage:
    python uci_har_cnn_tcn.py [path_to_dataset]
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
import sys
import re

# Data processing imports
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Machine learning imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

# Visualization imports
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS
# =============================================================================

# Dataset parameters
WINDOW_SIZE = 128  # UCI HAR uses 128 samples per window
NUM_CLASSES = 6    # UCI HAR has 6 activity classes
SAMPLING_RATE = 50 # UCI HAR was sampled at 50Hz

# Training parameters
BATCH_SIZE = 32
MAX_EPOCHS = 50
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2

# Model parameters
CNN_FILTERS = 16   # Number of filters in CNN layers
TCN_FILTERS = 32   # Number of filters in TCN layers
L2_FACTOR = 1e-3   # L2 regularization factor
LEARNING_RATE = 0.0005  # Learning rate for optimizer
DROPOUT_RATE = 0.4  # Fixed dropout rate

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Activity names for UCI HAR dataset
ACTIVITY_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
    "SITTING", "STANDING", "LAYING"
]

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_uci_har_data(dataset_path):
    """
    Load the UCI HAR dataset from the specified path
    
    Args:
        dataset_path (str): Path to the UCI HAR dataset
        
    Returns:
        Tuple containing (X_train, y_train, X_test, y_test)
    """
    # Define file paths
    train_x_path = os.path.join(dataset_path, "train", "X_train.txt")
    train_y_path = os.path.join(dataset_path, "train", "y_train.txt")
    test_x_path = os.path.join(dataset_path, "test", "X_test.txt")
    test_y_path = os.path.join(dataset_path, "test", "y_test.txt")
    
    # Load features
    print("Loading UCI HAR dataset features...")
    X_train = pd.read_csv(train_x_path, delim_whitespace=True, header=None).values
    X_test = pd.read_csv(test_x_path, delim_whitespace=True, header=None).values
    
    # Load labels
    print("Loading UCI HAR dataset labels...")
    y_train = pd.read_csv(train_y_path, header=None).values.ravel()
    y_test = pd.read_csv(test_y_path, header=None).values.ravel()
    
    # Load feature names for better understanding
    features_path = os.path.join(dataset_path, "features.txt")
    feature_names = pd.read_csv(features_path, delim_whitespace=True, header=None)[1].values
    
    print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Print class distribution
    print("\nClass distribution:")
    for i, activity in enumerate(ACTIVITY_NAMES, 1):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"  {activity}: {train_count} train, {test_count} test")
    
    return X_train, y_train, X_test, y_test, feature_names

def reshape_uci_har_data(X_train, X_test, feature_names):
    """
    Reshape the UCI HAR dataset for CNN processing
    
    Args:
        X_train (numpy.ndarray): Training features
        X_test (numpy.ndarray): Test features
        feature_names (numpy.ndarray): Names of features
        
    Returns:
        Tuple containing reshaped (X_train, X_test)
    """
    # Original window size is 128 with 50% overlap
    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    
    # Create a pattern to identify the feature type
    acc_pattern = re.compile(r'(acc|gyro)', re.IGNORECASE)
    
    # Identify accelerometer and gyroscope features
    acc_features = [i for i, name in enumerate(feature_names) if acc_pattern.search(name)]
    
    # Select only accelerometer/gyroscope features (not computed features like frequency domain)
    X_train_selected = X_train[:, acc_features]
    X_test_selected = X_test[:, acc_features]
    
    # Get the number of features after selection
    n_features = X_train_selected.shape[1]
    print(f"Selected {n_features} features for accelerometer/gyroscope signals")
    
    # Create a dummy timestep dimension
    timesteps = WINDOW_SIZE 
    features_per_channel = n_features // 6  # 3 axes each for acc and gyro
    
    # Reshape to [samples, timesteps, features]
    X_train_reshaped = np.zeros((n_samples_train, timesteps, features_per_channel))
    X_test_reshaped = np.zeros((n_samples_test, timesteps, features_per_channel))
    
    # Fill with repeated values to simulate the raw signal window
    for i in range(n_samples_train):
        for j in range(features_per_channel):
            X_train_reshaped[i, :, j] = np.linspace(
                X_train_selected[i, j], 
                X_train_selected[i, j + features_per_channel - 1],
                timesteps
            )
    
    for i in range(n_samples_test):
        for j in range(features_per_channel):
            X_test_reshaped[i, :, j] = np.linspace(
                X_test_selected[i, j], 
                X_test_selected[i, j + features_per_channel - 1],
                timesteps
            )
    
    print(f"Reshaped training data: {X_train_reshaped.shape}")
    print(f"Reshaped test data: {X_test_reshaped.shape}")
    
    return X_train_reshaped, X_test_reshaped

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_data(X_train, y_train, X_test, y_test, feature_names):
    """
    Preprocess the UCI HAR dataset with direct Gaussian smoothing like PyTorch example
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        feature_names (numpy.ndarray): Feature names
        
    Returns:
        Tuple of preprocessed data
    """
    # Apply Gaussian smoothing directly to the whole dataset (similar to PyTorch example)
    print("Applying Gaussian smoothing...")
    X_train_smoothed = gaussian_filter1d(X_train, sigma=2, axis=0)
    X_test_smoothed = gaussian_filter1d(X_test, sigma=2, axis=0)
    
    # Convert back to DataFrame with feature names (similar to PyTorch approach)
    X_train_df = pd.DataFrame(X_train_smoothed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_smoothed, columns=feature_names)
    
    # Extract values back to numpy arrays for further processing
    X_train_processed = X_train_df.values
    X_test_processed = X_test_df.values
            
    print("Normalizing data...")
    # Normalize the data
    scaler = StandardScaler()
    
    # Fit on training data
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    # Adjust labels to start from 0 (subtract 1 since UCI HAR starts from 1)
    y_train_adj = y_train - 1
    y_test_adj = y_test - 1
    
    return X_train_scaled, y_train_adj, X_test_scaled, y_test_adj

# Note: We're no longer using this function since noise is added directly in main()
def add_noise(X, noise_level=0.05):
    """
    Add minimal Gaussian noise data augmentation
    
    Args:
        X: Input data
        noise_level: Amount of Gaussian noise to add
        
    Returns:
        Augmented X
    """
    print("Adding noise augmentation...")
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, X.shape)
    noisy_X = X + noise
    
    # Combine original and noisy data
    X_combined = np.vstack([X, noisy_X])
    
    return X_combined

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_cnn_tcn_model(input_shape, num_classes, cnn_filters=16, tcn_filters=32, l2_factor=1e-3, learning_rate=0.0005, use_2d_conv=True):
    """
    Build a CNN-TCN hybrid model for activity recognition
    Combines CNN layers with TCN layers for temporal modeling
    
    Args:
        input_shape (tuple): Shape of input data
        num_classes (int): Number of activity classes
        cnn_filters (int): Number of filters in CNN layers
        tcn_filters (int): Number of filters in TCN layers
        l2_factor (float): L2 regularization factor
        learning_rate (float): Learning rate for optimizer
        use_2d_conv (bool): Whether to use 2D convolutions for CNN part
        
    Returns:
        Compiled Keras model
    """
    if use_2d_conv:
        # 2D CNN approach
        print(f"Using 2D CNN + TCN with input shape: {input_shape}")
        
        # Input shape should be (height, width, channels) for 'channels_last'
        inputs = tf.keras.Input(shape=input_shape)
        
        # First 2D CNN layer
        x = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3, 1),
            padding='same', 
            activation='relu',
            data_format='channels_last',
            kernel_regularizer=l2(l2_factor),
            kernel_initializer='he_normal',
            name='conv1'
        )(inputs)
        
        # Only pool along the height dimension if it's large enough
        current_height = input_shape[0]
        if current_height >= 2:
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 1), data_format='channels_last')(x)
            current_height = current_height // 2
        
        x = Dropout(DROPOUT_RATE)(x)
        
        # Second 2D CNN layer
        x = tf.keras.layers.Conv2D(
            filters=64, 
            kernel_size=(3, 1),
            padding='same', 
            activation='relu',
            data_format='channels_last',
            kernel_regularizer=l2(l2_factor),
            kernel_initializer='he_normal',
            name='conv2'
        )(x)
        
        # For MaxPooling, we'll make decisions based on the input shape
        if current_height >= 2:
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 1), data_format='channels_last')(x)
            current_height = current_height // 2
            
        x = Dropout(DROPOUT_RATE)(x)
        
        # Third 2D CNN layer
        x = tf.keras.layers.Conv2D(
            filters=128,  
            kernel_size=(3, 1),
            padding='same', 
            activation='relu',
            data_format='channels_last',
            kernel_regularizer=l2(l2_factor),
            kernel_initializer='he_normal',
            name='conv3'
        )(x)
        
        # Final MaxPool only if we have enough height dimension left
        if current_height >= 2:
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 1), data_format='channels_last')(x)
        
        x = Dropout(DROPOUT_RATE)(x)
        
        # Reshape for TCN (need 3D input - batch, time steps, features)
        # Assuming our current shape is (batch, height, width, channels)
        # We need to reshape to (batch, time steps, features)
        # In this case, we flatten the height and width dimensions to create a time sequence
        x = tf.keras.layers.Reshape((-1, 128))(x)  # 128 from last CNN filters
        
    else:
        # 1D CNN approach
        print(f"Using 1D CNN + TCN with input shape: {input_shape}")
        inputs = tf.keras.Input(shape=input_shape)
        
        # First CNN layer
        x = Conv1D(
            filters=cnn_filters, 
            kernel_size=5,
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(l2_factor),
            kernel_initializer='he_normal',
            name='conv1'
        )(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(DROPOUT_RATE)(x)
        
        # Second CNN layer
        x = Conv1D(
            filters=cnn_filters*2, 
            kernel_size=5,
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(l2_factor),
            kernel_initializer='he_normal',
            name='conv2'
        )(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(DROPOUT_RATE)(x)
        
        # Third CNN layer
        x = Conv1D(
            filters=cnn_filters*4,  
            kernel_size=5,  
            padding='same', 
            activation='relu',
            kernel_regularizer=l2(l2_factor),
            kernel_initializer='he_normal',
            name='conv3'
        )(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(DROPOUT_RATE)(x)
    
    # Add TCN layers (2 TCN layers with filter size 32)
    # First TCN layer
    x = tf.keras.layers.Conv1D(
        filters=tcn_filters,
        kernel_size=3,
        padding='causal',  # Causal padding for TCN
        activation='relu',
        dilation_rate=1,   # First layer with dilation rate 1
        kernel_regularizer=l2(l2_factor),
        kernel_initializer='he_normal',
        name='tcn1'
    )(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Second TCN layer with increased dilation
    x = tf.keras.layers.Conv1D(
        filters=tcn_filters,
        kernel_size=3,
        padding='causal',  # Causal padding for TCN
        activation='relu',
        dilation_rate=2,   # Second layer with increased dilation
        kernel_regularizer=l2(l2_factor),
        kernel_initializer='he_normal',
        name='tcn2'
    )(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Global pooling to reduce sequence length
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers for classification
    x = Dense(
        64, 
        activation='relu', 
        kernel_regularizer=l2(l2_factor),
        kernel_initializer='he_normal'
    )(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    outputs = Dense(
        num_classes, 
        activation='softmax',
        kernel_initializer='he_normal'
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use Adam with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
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
    plt.savefig('uci_har_training_history.png')
    plt.show()

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate model with basic metrics (accuracy, recall, F1)
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        class_names: Optional list of class names
    """
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Print final results
    print('Accuracy: {:.2f}%'.format(acc * 100))
    print('Recall: {:.2f}%'.format(recall * 100))
    print('F1-Score: {:.2f}%'.format(f1 * 100))
    
    return acc, recall, f1

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    # Parse command line arguments for dataset path
    dataset_path = "Dataset/UCIHAR/UCI HAR Dataset"
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    # Whether to use 2D convolutions like PyTorch example
    use_2d_conv = True  # Set to True to use the PyTorch-like approach
    
    print("\n" + "=" * 80)
    print(f"UCI HAR DATASET PROCESSING WITH CNN-TCN MODEL")
    print("=" * 80)
    
    print(f"\nUsing dataset path: {dataset_path}")
    
    # 1. Data Loading
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, feature_names = load_uci_har_data(dataset_path)
    
    if use_2d_conv:
        # Process data for 2D convolutions (similar to PyTorch example)
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA FOR 2D CNN-TCN")
        print("=" * 80)
        
        # Apply Gaussian smoothing directly
        print("Applying Gaussian smoothing...")
        X_train_smoothed = gaussian_filter1d(X_train_raw, sigma=2, axis=0)
        X_test_smoothed = gaussian_filter1d(X_test_raw, sigma=2, axis=0)
        
        # Normalize data
        print("Normalizing data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smoothed)
        X_test_scaled = scaler.transform(X_test_smoothed)
        
        # Adjust labels
        y_train = y_train_raw - 1
        y_test = y_test_raw - 1
        
        # Reshape for 2D CNN with enough height for pooling operations
        # Shape: (samples, height, width, channels) where height is feature count
        # This uses the features as the height dimension which allows pooling
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1, 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1, 1)
        
        # Define input shape for the model (height, width, channels)
        input_shape = (X_train_scaled.shape[1], 1, 1)
        
        print(f"2D CNN-TCN input shape: {input_shape}")
    else:
        # Process data for 1D CNN approach
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA FOR 1D CNN-TCN")
        print("=" * 80)
        
        # Reshape data for 1D CNN
        X_train_reshaped, X_test_reshaped = reshape_uci_har_data(X_train_raw, X_test_raw, feature_names)
        
        # Apply Gaussian smoothing and normalize
        print("Applying Gaussian smoothing...")
        n_samples_train, n_timesteps, n_features = X_train_reshaped.shape
        n_samples_test, _, _ = X_test_reshaped.shape
        
        for i in range(n_samples_train):
            for j in range(n_features):
                X_train_reshaped[i, :, j] = gaussian_filter1d(X_train_reshaped[i, :, j], sigma=2)
        
        for i in range(n_samples_test):
            for j in range(n_features):
                X_test_reshaped[i, :, j] = gaussian_filter1d(X_test_reshaped[i, :, j], sigma=2)
        
        # Normalize the data
        print("Normalizing data...")
        # Flatten for scaling
        X_train_flat = X_train_reshaped.reshape(n_samples_train, -1)
        X_test_flat = X_test_reshaped.reshape(n_samples_test, -1)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)
        
        # Reshape back to 3D
        X_train_reshaped = X_train_scaled.reshape(X_train_reshaped.shape)
        X_test_reshaped = X_test_scaled.reshape(X_test_reshaped.shape)
        
        # Adjust labels
        y_train = y_train_raw - 1
        y_test = y_test_raw - 1
        
        # Define input shape for the model
        input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
    
    # Simple data augmentation (add noise)
    print("Performing minimal data augmentation...")
    
    # Add noise to the shaped data
    noise = np.random.normal(0, 0.05, X_train_reshaped.shape)
    X_train_noisy = X_train_reshaped + noise
    X_train_aug = np.vstack([X_train_reshaped, X_train_noisy])
    
    # Duplicate labels for augmented data
    y_train_aug = np.concatenate([y_train, y_train])
    
    # Convert labels to one-hot encoding
    y_train_onehot = to_categorical(y_train_aug, num_classes=NUM_CLASSES)
    y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)
    
    print(f"Final data shapes - X_train: {X_train_reshaped.shape}, X_test: {X_test_reshaped.shape}")
    print(f"Augmented training data shape: {X_train_aug.shape}")
    print(f"Input shape for model: {input_shape}")
    
    # 4. Model Building and Training
    print("\n" + "=" * 80)
    print(f"TRAINING {'2D' if use_2d_conv else '1D'} CNN-TCN MODEL")
    print("=" * 80)
    
    # Build and train model (now using build_cnn_tcn_model)
    model = build_cnn_tcn_model(
        input_shape, 
        NUM_CLASSES, 
        CNN_FILTERS, 
        tcn_filters=TCN_FILTERS,  # Using 32 filters for TCN layers as requested
        l2_factor=L2_FACTOR, 
        learning_rate=LEARNING_RATE,
        use_2d_conv=use_2d_conv
    )
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            'uci_har_best_model.keras', 
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
        X_train_aug, y_train_onehot,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # 5. Evaluation
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    acc, recall, f1 = evaluate_model(model, X_test_reshaped, y_test_onehot, class_names=ACTIVITY_NAMES)
    
    # 6. Visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    plot_training_history(history)
    
    # 7. Save Model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    # Save model
    model_type = "2d" if use_2d_conv else "1d"
    model.save(f'uci_har_cnn_tcn_{model_type}_model.keras')
    print(f"Model saved as 'uci_har_cnn_tcn_{model_type}_model.keras'")
    
    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)

# Execute main function when script is run directly
if __name__ == "__main__":
    main()