import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
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
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

WINDOW_SIZE = 200
OVERLAP = 5.0
NUM_CLASSES = 12
ORIGINAL_SAMPLING_RATE = 50
TARGET_SAMPLING_RATE = 20

BATCH_SIZE = 32
MAX_EPOCHS = 100
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
CROSS_VAL_FOLDS = 5

TCN_FILTERS = 16
L2_FACTOR = 1e-3
LEARNING_RATE = 0.0005

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

ACTIVITY_NAMES = [
    "Walking",
    "Jogging",
    "Stairs",
    "Sitting",
    "Standing",
    "Lying Down",
    "Brushing Teeth",
    "Combing Hair",
    "Writing",
    "Eating Soup",
    "Eating Chips",
    "Drinking",
]


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
        print(f"Searching with pattern: {pattern}")
        files = glob.glob(pattern)

        if files:
            files_found = True
            print(f"Found {len(files)} files with pattern {pattern}")

            for filename in files:
                try:
                    if filename.endswith(".csv"):
                        data = pd.read_csv(filename, header=None)
                    else:
                        try:
                            data = pd.read_csv(filename, header=None, sep="\t")
                        except:
                            data = pd.read_csv(filename, header=None, sep=",")

                    if len(data) == 0:
                        print(f"Skipping empty file: {filename}")
                        continue

                    X = data.iloc[:, :-1].values
                    y = data.iloc[:, -1].values

                    all_data.append(X)
                    all_labels.append(y)
                    print(f"Successfully loaded {filename} with shape {X.shape}")

                except Exception as e:
                    print(f"Error loading {filename}: {e}")

            if all_data:
                break

    if not files_found:
        raise FileNotFoundError(
            f"No files found in {dataset_path}. Check the path and file naming convention."
        )

    if not all_data:
        raise ValueError(
            "No data could be loaded from the files found. Check file format."
        )

    X = np.vstack(all_data)
    y = np.hstack(all_labels)

    return X, y


def preprocess_data(X, y):
    print(
        f"Original data shape: {X.shape}, Original sampling rate: {ORIGINAL_SAMPLING_RATE}Hz"
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    mask = y != 0
    X = X[mask]
    y = y[mask]

    y = y - 1

    if ORIGINAL_SAMPLING_RATE > TARGET_SAMPLING_RATE:
        ds_factor = ORIGINAL_SAMPLING_RATE // TARGET_SAMPLING_RATE
        print(
            f"Downsampling from {ORIGINAL_SAMPLING_RATE}Hz to {TARGET_SAMPLING_RATE}Hz (factor: {ds_factor})"
        )

        X_ds = X[::ds_factor]
        y_ds = y[::ds_factor]

        print(f"After downsampling - Data shape: {X_ds.shape}")
        return X_ds, y_ds
    else:
        print("No downsampling applied")
        return X, y


def create_windows(X, y, window_size=WINDOW_SIZE, overlap=OVERLAP):
    step = int(window_size * (1 - overlap))
    n_features = X.shape[1]

    windows = []
    window_labels = []

    print(f"Creating windows with size {window_size} and step {step} (no overlap)")
    print(f"Original data shape: {X.shape}")

    for i in range(0, len(X) - window_size + 1, step):
        window = X[i : i + window_size]

        window_label = stats.mode(y[i : i + window_size], keepdims=True)[0][0]

        windows.append(window)
        window_labels.append(window_label)

    windows = np.array(windows)
    window_labels = np.array(window_labels)

    print(f"Created {len(windows)} windows with shape {windows.shape}")

    return windows, window_labels


def augment_data(X, y, noise_level=0.05):
    aug_X = []
    aug_y = []

    aug_X.append(X)
    aug_y.append(y)

    noise = np.random.normal(0, noise_level, X.shape)
    noisy_X = X + noise
    aug_X.append(noisy_X)
    aug_y.append(y)

    n_samples, time_steps, n_features = X.shape
    warped_X = np.zeros_like(X)
    for i in range(n_samples):
        for j in range(n_features):
            orig_signal = X[i, :, j]
            stretch_factor = 0.8 + 0.4 * np.random.random()
            warped_signal = np.interp(
                np.linspace(0, 1, time_steps),
                np.linspace(0, 1, time_steps) * stretch_factor,
                orig_signal,
            )
            warped_X[i, :, j] = warped_signal

    aug_X.append(warped_X)
    aug_y.append(y)

    scaled_X = X.copy()
    for j in range(n_features):
        scale_factor = 0.8 + 0.4 * np.random.random()
        scaled_X[:, :, j] = scaled_X[:, :, j] * scale_factor

    aug_X.append(scaled_X)
    aug_y.append(y)

    shifted_X = np.zeros_like(X)
    for i in range(n_samples):
        for j in range(n_features):
            shift = int(0.1 * time_steps * (2 * np.random.random() - 1))
            orig_signal = X[i, :, j]
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

    print(
        f"Augmented data: {len(aug_X)} versions (original + {len(aug_X)-1} augmented)"
    )
    return np.vstack(aug_X), np.concatenate(aug_y)


def build_tcn_model(
    input_shape, num_classes, tcn_filters=16, l2_factor=1e-3, learning_rate=0.0005
):
    inputs = tf.keras.Input(shape=input_shape)

    x = Dropout(0.1)(inputs)

    tcn_layer_1 = Conv1D(
        filters=tcn_filters,
        kernel_size=5,
        padding="causal",
        activation=None,
        kernel_regularizer=l2(l2_factor),
        kernel_initializer="he_normal",
        name="tcn_conv1",
    )(x)
    tcn_layer_1 = BatchNormalization()(tcn_layer_1)
    tcn_layer_1 = ReLU()(tcn_layer_1)
    tcn_layer_1 = MaxPooling1D(pool_size=2)(tcn_layer_1)
    tcn_layer_1 = Dropout(0.5)(tcn_layer_1)

    tcn_layer_2 = Conv1D(
        filters=tcn_filters,
        kernel_size=5,
        padding="causal",
        activation=None,
        kernel_regularizer=l2(l2_factor),
        kernel_initializer="he_normal",
        name="tcn_conv2",
    )(tcn_layer_1)
    tcn_layer_2 = BatchNormalization()(tcn_layer_2)
    tcn_layer_2 = ReLU()(tcn_layer_2)
    tcn_layer_2 = MaxPooling1D(pool_size=2)(tcn_layer_2)
    tcn_layer_2 = Dropout(0.5)(tcn_layer_2)

    tcn_layer_3 = Conv1D(
        filters=tcn_filters * 2,
        kernel_size=5,
        padding="causal",
        activation=None,
        kernel_regularizer=l2(l2_factor),
        kernel_initializer="he_normal",
        name="tcn_conv3",
    )(tcn_layer_2)
    tcn_layer_3 = BatchNormalization()(tcn_layer_3)
    tcn_layer_3 = ReLU()(tcn_layer_3)
    tcn_layer_3 = MaxPooling1D(pool_size=2)(tcn_layer_3)
    tcn_layer_3 = Dropout(0.5)(tcn_layer_3)

    x = Flatten()(tcn_layer_3)
    x = Dense(32, activation=None, kernel_regularizer=l2(l2_factor))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.6)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    metrics = [
        "accuracy",
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
    ]

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics)

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
    plt.savefig("training_history.png")
    plt.show()


def evaluate_with_metrics(model, X_test, y_test, class_names=None):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)

    recall_macro = recall_score(y_true, y_pred, average="macro")
    recall_per_class = recall_score(y_true, y_pred, average=None)

    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_per_class = f1_score(y_true, y_pred, average=None)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Recall: {:.2f}%".format(recall_macro * 100))
    print("F1-Score: {:.2f}%".format(f1_macro * 100))

    return {
        "accuracy": acc,
        "recall_macro": recall_macro,
        "recall_per_class": recall_per_class,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def cross_validate_model(
    X_windows, y_windows, input_shape, num_classes, n_splits=CROSS_VAL_FOLDS
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    avg_acc = []
    avg_recall = []
    avg_f1 = []

    print(f"\nPerforming {n_splits}-fold cross-validation...")

    for i, (train_idx, test_idx) in enumerate(skf.split(X_windows, y_windows)):
        X_train, X_test = X_windows[train_idx], X_windows[test_idx]
        y_train, y_test = y_windows[train_idx], y_windows[test_idx]

        X_train, y_train = augment_data(X_train, y_train, noise_level=0.05)

        y_train_onehot = to_categorical(y_train, num_classes=num_classes)
        y_test_onehot = to_categorical(y_test, num_classes=num_classes)

        model = build_tcn_model(
            input_shape, num_classes, TCN_FILTERS, L2_FACTOR, LEARNING_RATE
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                min_delta=0.001,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
            ),
        ]

        model.fit(
            X_train,
            y_train_onehot,
            epochs=100,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = y_test

        acc_fold = accuracy_score(y_true, y_pred)
        recall_fold = recall_score(y_true, y_pred, average="macro")
        f1_fold = f1_score(y_true, y_pred, average="macro")

        avg_acc.append(acc_fold)
        avg_recall.append(recall_fold)
        avg_f1.append(f1_fold)

        print(
            "Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]".format(
                acc_fold, recall_fold, f1_fold, i + 1
            )
        )
        print("______________________________________________________")

        K.clear_session()

    std_acc = np.std(avg_acc, ddof=1)
    std_recall = np.std(avg_recall, ddof=1)
    std_f1 = np.std(avg_f1, ddof=1)

    mean_acc = np.mean(avg_acc)
    mean_recall = np.mean(avg_recall)
    mean_f1 = np.mean(avg_f1)

    print("\nCross-Validation Results:")
    print("Accuracy: {:.2f}% ± {:.2f}".format(mean_acc * 100, std_acc * 100))
    print("Recall: {:.2f}% ± {:.2f}".format(mean_recall * 100, std_recall * 100))
    print("F1-Score: {:.2f}% ± {:.2f}".format(mean_f1 * 100, std_f1 * 100))

    return {
        "accuracy": {"mean": mean_acc, "std": std_acc},
        "recall": {"mean": mean_recall, "std": std_recall},
        "f1_score": {"mean": mean_f1, "std": std_f1},
        "accuracies": avg_acc,
        "recalls": avg_recall,
        "f1_scores": avg_f1,
    }


def main():
    dataset_path = "Dataset/MHEALTHDATASET"

    print("\n" + "=" * 80)
    print(f"MHEALTH DATASET PROCESSING WITH TCN MODEL")
    print("=" * 80)

    print(f"\nUsing dataset path: {dataset_path}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Available files and directories: {os.listdir()}")

    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    X, y = load_mhealth_data(dataset_path)
    X, y = preprocess_data(X, y)

    print("\n" + "=" * 80)
    print("CREATING NON-OVERLAPPING WINDOWS")
    print("=" * 80)
    print(
        "Using non-overlapping windows to prevent data leakage between train/validation/test sets"
    )
    X_windows, y_windows = create_windows(X, y, overlap=0.0)

    input_shape = (X_windows.shape[1], X_windows.shape[2])
    print(f"Window shape: {X_windows.shape}")
    print(f"Number of classes: {NUM_CLASSES}")

    unique, counts = np.unique(y_windows, return_counts=True)
    print("\nClass distribution in windowed data:")
    for i, (cls, count) in enumerate(zip(unique, counts)):
        if i < len(ACTIVITY_NAMES):
            class_name = ACTIVITY_NAMES[int(cls)]
        else:
            class_name = f"Class {int(cls)}"
        print(f"  {class_name}: {count} windows ({count/len(y_windows)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION")
    print("=" * 80)
    cv_results = cross_validate_model(
        X_windows, y_windows, input_shape=input_shape, num_classes=NUM_CLASSES
    )

    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)

    y_onehot = to_categorical(y_windows, num_classes=NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(
        X_windows,
        y_onehot,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_windows,
    )

    X_train_aug, y_train_aug_indices = augment_data(
        X_train, np.argmax(y_train, axis=1), noise_level=0.05
    )
    y_train_aug = to_categorical(y_train_aug_indices, num_classes=NUM_CLASSES)

    model = build_tcn_model(
        input_shape, NUM_CLASSES, TCN_FILTERS, L2_FACTOR, LEARNING_RATE
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, min_delta=0.001
        ),
        ModelCheckpoint(
            "best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
        ),
    ]

    history = model.fit(
        X_train_aug,
        y_train_aug,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    print("\n" + "=" * 80)
    print("EVALUATING FINAL MODEL")
    print("=" * 80)
    metrics = evaluate_with_metrics(model, X_test, y_test, class_names=ACTIVITY_NAMES)

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    plot_training_history(history)

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Recall", "F1 Score"],
            "Final Model (%)": [
                metrics["accuracy"] * 100,
                metrics["recall_macro"] * 100,
                metrics["f1_macro"] * 100,
            ],
            "CV Mean (%)": [
                cv_results["accuracy"]["mean"] * 100,
                cv_results["recall"]["mean"] * 100,
                cv_results["f1_score"]["mean"] * 100,
            ],
            "CV Std (%)": [
                cv_results["accuracy"]["std"] * 100,
                cv_results["recall"]["std"] * 100,
                cv_results["f1_score"]["std"] * 100,
            ],
        }
    )
    results_df.to_csv("model_metrics.csv", index=False)
    print("Metrics saved to 'model_metrics.csv'")

    model.save("mhealth_tcn_model.keras")
    print("Model saved as 'mhealth_tcn_model.keras'")

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
