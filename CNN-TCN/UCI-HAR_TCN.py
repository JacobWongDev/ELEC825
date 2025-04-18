import os
import sys
import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt

WINDOW_SIZE = 128
NUM_CLASSES = 6
SAMPLING_RATE = 50

BATCH_SIZE = 32
MAX_EPOCHS = 100
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2

CNN_FILTERS = 16
TCN_FILTERS = 32
L2_FACTOR = 1e-3
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.4

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def load_uci_har_data(dataset_path):
    train_x_path = os.path.join(dataset_path, "train", "X_train.txt")
    train_y_path = os.path.join(dataset_path, "train", "y_train.txt")
    test_x_path = os.path.join(dataset_path, "test", "X_test.txt")
    test_y_path = os.path.join(dataset_path, "test", "y_test.txt")

    X_train = pd.read_csv(train_x_path, delim_whitespace=True, header=None).values
    X_test = pd.read_csv(test_x_path, delim_whitespace=True, header=None).values

    y_train = pd.read_csv(train_y_path, header=None).values.ravel()
    y_test = pd.read_csv(test_y_path, header=None).values.ravel()

    features_path = os.path.join(dataset_path, "features.txt")
    feature_names = pd.read_csv(features_path, delim_whitespace=True, header=None)[
        1
    ].values

    print(
        f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples"
    )
    print(f"Number of features: {X_train.shape[1]}")

    print("\nClass distribution:")
    for i, activity in enumerate(ACTIVITY_NAMES, 1):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"  {activity}: {train_count} train, {test_count} test")

    return X_train, y_train, X_test, y_test, feature_names


def reshape_uci_har_data(X_train, X_test, feature_names):
    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]

    acc_pattern = re.compile(r"(acc|gyro)", re.IGNORECASE)

    acc_features = [
        i for i, name in enumerate(feature_names) if acc_pattern.search(name)
    ]

    X_train_selected = X_train[:, acc_features]
    X_test_selected = X_test[:, acc_features]

    n_features = X_train_selected.shape[1]
    print(f"Selected {n_features} features for accelerometer/gyroscope signals")

    timesteps = WINDOW_SIZE
    features_per_channel = n_features // 6

    X_train_reshaped = np.zeros((n_samples_train, timesteps, features_per_channel))
    X_test_reshaped = np.zeros((n_samples_test, timesteps, features_per_channel))

    for i in range(n_samples_train):
        for j in range(features_per_channel):
            X_train_reshaped[i, :, j] = np.linspace(
                X_train_selected[i, j],
                X_train_selected[i, j + features_per_channel - 1],
                timesteps,
            )

    for i in range(n_samples_test):
        for j in range(features_per_channel):
            X_test_reshaped[i, :, j] = np.linspace(
                X_test_selected[i, j],
                X_test_selected[i, j + features_per_channel - 1],
                timesteps,
            )

    print(f"Reshaped training data: {X_train_reshaped.shape}")
    print(f"Reshaped test data: {X_test_reshaped.shape}")

    return X_train_reshaped, X_test_reshaped


def preprocess_data(X_train, y_train, X_test, y_test, feature_names):
    print("Applying Gaussian smoothing...")
    X_train_smoothed = gaussian_filter1d(X_train, sigma=2, axis=0)
    X_test_smoothed = gaussian_filter1d(X_test, sigma=2, axis=0)

    X_train_df = pd.DataFrame(X_train_smoothed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_smoothed, columns=feature_names)

    X_train_processed = X_train_df.values
    X_test_processed = X_test_df.values

    print("Normalizing data...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    y_train_adj = y_train - 1
    y_test_adj = y_test - 1

    return X_train_scaled, y_train_adj, X_test_scaled, y_test_adj


def add_noise(X, noise_level=0.05):
    print("Adding noise augmentation...")
    noise = np.random.normal(0, noise_level, X.shape)
    noisy_X = X + noise
    X_combined = np.vstack([X, noisy_X])
    return X_combined


def build_cnn_tcn_model(
    input_shape,
    num_classes,
    cnn_filters=16,
    tcn_filters=32,
    l2_factor=1e-3,
    learning_rate=0.0005,
    use_2d_conv=True,
):
    if use_2d_conv:
        print(f"Using 2D CNN + TCN with input shape: {input_shape}")

        inputs = tf.keras.Input(shape=input_shape)

        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 1),
            padding="same",
            activation="relu",
            data_format="channels_last",
            kernel_regularizer=l2(l2_factor),
            kernel_initializer="he_normal",
            name="conv1",
        )(inputs)

        current_height = input_shape[0]
        if current_height >= 2:
            x = tf.keras.layers.MaxPool2D(
                pool_size=(2, 1), data_format="channels_last"
            )(x)
            current_height = current_height // 2

        x = Dropout(DROPOUT_RATE)(x)

        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 1),
            padding="same",
            activation="relu",
            data_format="channels_last",
            kernel_regularizer=l2(l2_factor),
            kernel_initializer="he_normal",
            name="conv2",
        )(x)

        if current_height >= 2:
            x = tf.keras.layers.MaxPool2D(
                pool_size=(2, 1), data_format="channels_last"
            )(x)
            current_height = current_height // 2

        x = Dropout(DROPOUT_RATE)(x)

        x = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 1),
            padding="same",
            activation="relu",
            data_format="channels_last",
            kernel_regularizer=l2(l2_factor),
            kernel_initializer="he_normal",
            name="conv3",
        )(x)

        if current_height >= 2:
            x = tf.keras.layers.MaxPool2D(
                pool_size=(2, 1), data_format="channels_last"
            )(x)

        x = Dropout(DROPOUT_RATE)(x)

        x = tf.keras.layers.Reshape((-1, 128))(x)

    else:
        print(f"Using 1D CNN + TCN with input shape: {input_shape}")
        inputs = tf.keras.Input(shape=input_shape)

        x = Conv1D(
            filters=cnn_filters,
            kernel_size=5,
            padding="same",
            activation="relu",
            kernel_regularizer=l2(l2_factor),
            kernel_initializer="he_normal",
            name="conv1",
        )(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(DROPOUT_RATE)(x)

        x = Conv1D(
            filters=cnn_filters * 2,
            kernel_size=5,
            padding="same",
            activation="relu",
            kernel_regularizer=l2(l2_factor),
            kernel_initializer="he_normal",
            name="conv2",
        )(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(DROPOUT_RATE)(x)

        x = Conv1D(
            filters=cnn_filters * 4,
            kernel_size=5,
            padding="same",
            activation="relu",
            kernel_regularizer=l2(l2_factor),
            kernel_initializer="he_normal",
            name="conv3",
        )(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(DROPOUT_RATE)(x)

    x = tf.keras.layers.Conv1D(
        filters=tcn_filters,
        kernel_size=3,
        padding="causal",
        activation="relu",
        dilation_rate=1,
        kernel_regularizer=l2(l2_factor),
        kernel_initializer="he_normal",
        name="tcn1",
    )(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = tf.keras.layers.Conv1D(
        filters=tcn_filters,
        kernel_size=3,
        padding="causal",
        activation="relu",
        dilation_rate=2,
        kernel_regularizer=l2(l2_factor),
        kernel_initializer="he_normal",
        name="tcn2",
    )(x)
    x = Dropout(DROPOUT_RATE)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(l2_factor),
        kernel_initializer="he_normal",
    )(x)
    x = Dropout(DROPOUT_RATE)(x)

    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(
        x
    )

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    return model


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(history.history["accuracy"], label="Training accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training and Validation Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Training loss")
    ax2.plot(history.history["val_loss"], label="Validation loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Training and Validation Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("uci_har_training_history.png")
    plt.show()


def evaluate_model(model, X_test, y_test, class_names=None):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1-Score: {:.2f}%".format(f1 * 100))

    return acc, recall, f1


def main():
    dataset_path = "Dataset/UCIHAR/UCI HAR Dataset"

    use_2d_conv = True

    print("\n" + "=" * 80)
    print(f"UCI HAR DATASET PROCESSING WITH CNN-TCN MODEL")
    print("=" * 80)

    print(f"\nUsing dataset path: {dataset_path}")

    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, feature_names = load_uci_har_data(
        dataset_path
    )

    if use_2d_conv:
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA FOR 2D CNN-TCN")
        print("=" * 80)

        print("Applying Gaussian smoothing...")
        X_train_smoothed = gaussian_filter1d(X_train_raw, sigma=2, axis=0)
        X_test_smoothed = gaussian_filter1d(X_test_raw, sigma=2, axis=0)

        print("Normalizing data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smoothed)
        X_test_scaled = scaler.transform(X_test_smoothed)

        y_train = y_train_raw - 1
        y_test = y_test_raw - 1

        X_train_reshaped = X_train_scaled.reshape(
            X_train_scaled.shape[0], X_train_scaled.shape[1], 1, 1
        )
        X_test_reshaped = X_test_scaled.reshape(
            X_test_scaled.shape[0], X_test_scaled.shape[1], 1, 1
        )

        input_shape = (X_train_scaled.shape[1], 1, 1)

        print(f"2D CNN-TCN input shape: {input_shape}")
    else:
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA FOR 1D CNN-TCN")
        print("=" * 80)

        X_train_reshaped, X_test_reshaped = reshape_uci_har_data(
            X_train_raw, X_test_raw, feature_names
        )

        print("Applying Gaussian smoothing...")
        n_samples_train, n_timesteps, n_features = X_train_reshaped.shape
        n_samples_test, _, _ = X_test_reshaped.shape

        for i in range(n_samples_train):
            for j in range(n_features):
                X_train_reshaped[i, :, j] = gaussian_filter1d(
                    X_train_reshaped[i, :, j], sigma=2
                )

        for i in range(n_samples_test):
            for j in range(n_features):
                X_test_reshaped[i, :, j] = gaussian_filter1d(
                    X_test_reshaped[i, :, j], sigma=2
                )

        print("Normalizing data...")
        X_train_flat = X_train_reshaped.reshape(n_samples_train, -1)
        X_test_flat = X_test_reshaped.reshape(n_samples_test, -1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)

        X_train_reshaped = X_train_scaled.reshape(X_train_reshaped.shape)
        X_test_reshaped = X_test_scaled.reshape(X_test_reshaped.shape)

        y_train = y_train_raw - 1
        y_test = y_test_raw - 1

        input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])

    print("Performing minimal data augmentation...")

    noise = np.random.normal(0, 0.05, X_train_reshaped.shape)
    X_train_noisy = X_train_reshaped + noise
    X_train_aug = np.vstack([X_train_reshaped, X_train_noisy])

    y_train_aug = np.concatenate([y_train, y_train])

    y_train_onehot = to_categorical(y_train_aug, num_classes=NUM_CLASSES)
    y_test_onehot = to_categorical(y_test, num_classes=NUM_CLASSES)

    print(
        f"Final data shapes - X_train: {X_train_reshaped.shape}, X_test: {X_test_reshaped.shape}"
    )
    print(f"Augmented training data shape: {X_train_aug.shape}")
    print(f"Input shape for model: {input_shape}")

    print("\n" + "=" * 80)
    print(f"TRAINING {'2D' if use_2d_conv else '1D'} CNN-TCN MODEL")
    print("=" * 80)

    model = build_cnn_tcn_model(
        input_shape,
        NUM_CLASSES,
        CNN_FILTERS,
        tcn_filters=TCN_FILTERS,
        l2_factor=L2_FACTOR,
        learning_rate=LEARNING_RATE,
        use_2d_conv=use_2d_conv,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, min_delta=0.001
        ),
        ModelCheckpoint(
            "uci_har_best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
        ),
    ]

    history = model.fit(
        X_train_aug,
        y_train_onehot,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    acc, recall, f1 = evaluate_model(
        model, X_test_reshaped, y_test_onehot, class_names=ACTIVITY_NAMES
    )

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    plot_training_history(history)

    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_type = "2d" if use_2d_conv else "1d"
    model.save(f"uci_har_cnn_tcn_{model_type}_model.keras")
    print(f"Model saved as 'uci_har_cnn_tcn_{model_type}_model.keras'")

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
