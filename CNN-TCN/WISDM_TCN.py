import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats._stats_py")
DATA_PATH = "Dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_transformed.arff"
N_FEATURES = 43
N_CLASSES = 6
WINDOW_SIZE = 100
STEP_SIZE = 5
TCN_FILTER = 32
L2_FACTOR = 1e-4
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
MAX_EPOCHS = 100
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
CROSS_VAL_FOLDS = 5
MIN_SAMPLES_PER_CLASS_SPLIT = 10
MIN_SAMPLES_PER_CLASS_CV = 10
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
ACTIVITY_NAMES = []


def load_wisdm_arff_data_manual(file_path):
    attributes = []
    data_lines = []
    reading_data = False
    identifiers_to_exclude = ["UNIQUE_ID", "user"]
    try:
        print(f"Attempting to load ARFF file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Dataset file not found at the specified path: {file_path}"
            )
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                if reading_data:
                    data_lines.append(line)
                elif line.lower().startswith("@attribute"):
                    parts = line.split(maxsplit=2)
                    if len(parts) < 2:
                        print(f"Warning: Skipping malformed @attribute line: {line}")
                        continue
                    attr_name = parts[1].strip("'\"")
                    attributes.append(attr_name)
                elif line.lower().startswith("@data"):
                    reading_data = True
        if not reading_data:
            raise ValueError("@data section not found.")
        if not attributes:
            raise ValueError("No @attribute lines found.")
        if len(attributes) < len(identifiers_to_exclude) + 2:
            raise ValueError(f"Insufficient attributes found ({len(attributes)}).")
        class_name = attributes[-1]
        feature_names = [
            attr
            for attr in attributes
            if attr != class_name and attr not in identifiers_to_exclude
        ]
        print(f"Total attributes found in header: {len(attributes)}")
        print(f"Assumed Class column: '{class_name}'")
        print(f"Excluding identifiers: {identifiers_to_exclude}")
        print(f"Identified {len(feature_names)} feature columns.")
        global N_FEATURES
        N_FEATURES = len(feature_names)
        print(f"Set N_FEATURES = {N_FEATURES}")
        data_io = io.StringIO("\n".join(data_lines))
        df = pd.read_csv(
            data_io, header=None, names=attributes, na_values="?", comment=None
        )
        print(f"Loaded dataframe shape before cleaning: {df.shape}")
        if df.empty:
            print("Warning: DataFrame empty after loading. Check @data.")
            return pd.DataFrame(), feature_names, class_name
        print("Attempting numeric conversion for identified feature columns...")
        for col in feature_names:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"Attempted numeric conversion for {len(feature_names)} feature columns.")
        if class_name in df.columns:
            df[class_name] = df[class_name].astype(str).str.strip()
            print(f"Class column '{class_name}' type set to string.")
        else:
            raise ValueError(f"Assumed class column '{class_name}' not found.")
        columns_to_keep = feature_names + [class_name]
        df_cleaned = df[columns_to_keep].copy()
        rows_before_drop = df_cleaned.shape[0]
        df_cleaned.dropna(inplace=True)
        rows_after_drop = df_cleaned.shape[0]
        if rows_before_drop > rows_after_drop:
            print(
                f"Dropped {rows_before_drop - rows_after_drop} rows containing NaN values."
            )
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        print(
            "Please check the DATA_PATH variable and ensure the file exists at that location relative to your script."
        )
        return pd.DataFrame(), [], ""
    except Exception as e:
        print(f"An error occurred during ARFF data loading: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame(), [], ""
    print(f"Loaded dataframe shape after cleaning: {df_cleaned.shape}")
    if df_cleaned.empty:
        print("Error: DataFrame empty after cleaning.")
        return pd.DataFrame(), feature_names, class_name
    print(
        f"Activity distribution ('{class_name}'):\n",
        df_cleaned[class_name].value_counts(),
    )
    return df_cleaned, feature_names, class_name


def create_segments(X, y, window_size, step_size, n_features):
    segments = []
    labels = []
    num_samples = X.shape[0]
    if step_size < window_size:
        overlap = window_size - step_size
        print(
            f"Creating segments with window_size={window_size}, step_size={step_size} ({overlap} steps overlap)..."
        )
    else:
        print(
            f"Creating segments with window_size={window_size}, step_size={step_size} (NO OVERLAP)..."
        )
    for i in range(0, num_samples - window_size + 1, step_size):
        segment_features = X[i : i + window_size]
        segment_labels = y[i : i + window_size]
        if segment_features.shape[0] == window_size:
            if segment_labels.size > 0:
                label = stats.mode(segment_labels, keepdims=False)[0]
                segments.append(segment_features)
                labels.append(label)
    if not segments:
        print(
            "Warning: No segments were created. Check window/step size relative to data length. Returning empty arrays."
        )
        return np.empty((0, window_size, n_features)), np.empty((0,))
    X_segmented = np.array(segments)
    y_segmented = np.array(labels)
    print(f"Created {X_segmented.shape[0]} segments.")
    print(f"Segmented data shapes: X={X_segmented.shape}, y={y_segmented.shape}")
    unique_labels, counts = np.unique(y_segmented, return_counts=True)
    print("Segmented label distribution:")
    global ACTIVITY_NAMES
    label_map = {i: name for i, name in enumerate(ACTIVITY_NAMES)}
    for label_val, count in zip(unique_labels, counts):
        label_name = label_map.get(label_val, f"Unknown Label {label_val}")
        print(f"  Label {label_val} ('{label_name}'): {count} segments")
        if count < MIN_SAMPLES_PER_CLASS_SPLIT:
            print(
                f"  !!! WARNING: Label {label_val} ('{label_name}') has only {count} segment(s). Stratified train/test split requires at least {MIN_SAMPLES_PER_CLASS_SPLIT}. Might be excluded. !!!"
            )
        if count < MIN_SAMPLES_PER_CLASS_CV:
            print(
                f"  !!! WARNING: Label {label_val} ('{label_name}') has only {count} segment(s). Stratified {CROSS_VAL_FOLDS}-Fold CV requires at least {MIN_SAMPLES_PER_CLASS_CV}. Might be excluded. !!!"
            )
    return X_segmented, y_segmented


def augment_data(X, y, noise_level=0.05):
    print(f"Applying data augmentation (noise, warp, scale, shift)...")
    aug_X = []
    aug_y = []
    aug_X.append(X)
    aug_y.append(y)
    print(f" - Original shape: X={X.shape}, y={y.shape}")
    noise = np.random.normal(0, noise_level, X.shape)
    noisy_X = X + noise
    aug_X.append(noisy_X)
    aug_y.append(y)
    print(f" - Added Noise shape: X={noisy_X.shape}, y={y.shape}")
    if len(X.shape) != 3:
        print(
            f"Warning: Expected X with shape (samples, timesteps, features), but got {X.shape}. Skipping time-based augmentations."
        )
        combined_X = np.vstack(aug_X)
        combined_y = np.concatenate(aug_y)
        print(
            f"Augmented data shape (Noise only): X={combined_X.shape}, y={combined_y.shape}"
        )
        return combined_X, combined_y
    n_samples, time_steps, n_features = X.shape
    print(" - Applying Time Warping.")
    warped_X = np.zeros_like(X)
    for i in range(n_samples):
        for j in range(n_features):
            orig_signal = X[i, :, j]
            stretch_factor = 0.8 + 0.4 * np.random.random()
            orig_time_axis = np.linspace(0, 1, time_steps)
            target_time_axis = np.linspace(0, 1, time_steps)
            source_time_points = target_time_axis / stretch_factor
            warped_signal = np.interp(source_time_points, orig_time_axis, orig_signal)
            warped_X[i, :, j] = warped_signal
    aug_X.append(warped_X)
    aug_y.append(y)
    print(f" - Added Warping shape: X={warped_X.shape}, y={y.shape}")
    scaled_X = X.copy()
    for j in range(n_features):
        scale_factor = 0.8 + 0.4 * np.random.random()
        scaled_X[:, :, j] = scaled_X[:, :, j] * scale_factor
    aug_X.append(scaled_X)
    aug_y.append(y)
    print(f" - Added Scaling shape: X={scaled_X.shape}, y={y.shape}")
    print(" - Applying Time Shifting.")
    shifted_X = np.zeros_like(X)
    for i in range(n_samples):
        shift = int(0.1 * time_steps * (2 * np.random.random() - 1))
        if shift != 0:
            shifted_X[i, :, :] = np.roll(X[i, :, :], shift, axis=0)
            if shift > 0:
                shifted_X[i, :shift, :] = 0
            else:
                shifted_X[i, shift:, :] = 0
        else:
            shifted_X[i, :, :] = X[i, :, :]
    aug_X.append(shifted_X)
    aug_y.append(y)
    print(f" - Added Shifting shape: X={shifted_X.shape}, y={y.shape}")
    combined_X = np.vstack(aug_X)
    combined_y = np.concatenate(aug_y)
    print(f"Total Augmented data shape: X={combined_X.shape}, y={combined_y.shape}")
    print(
        f"Augmentation resulted in {len(aug_X)} versions (original + {len(aug_X)-1} augmented)"
    )
    return combined_X, combined_y


def build_model(
    input_segment_shape,
    num_classes,
    tcn_filters=TCN_FILTER,
    l2_factor=L2_FACTOR,
    learning_rate=LEARNING_RATE,
):
    inputs = keras.Input(shape=input_segment_shape)
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
    tcn_layer_1 = Dropout(0.3)(tcn_layer_1)
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
    tcn_layer_2 = Dropout(0.3)(tcn_layer_2)
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
    tcn_layer_3 = Dropout(0.3)(tcn_layer_3)
    x = Flatten()(tcn_layer_3)
    x = Dense(32, activation=None, kernel_regularizer=l2(l2_factor))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.6)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    metrics = [
        "accuracy",
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
    ]
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics)
    model.summary()
    return model


def plot_training_history(history, filename="training_history.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    if "accuracy" in history.history and "val_accuracy" in history.history:
        ax1.plot(history.history["accuracy"], label="Training Accuracy")
        ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Training and Validation Accuracy")
        ax1.legend()
        ax1.grid(True)
    else:
        ax1.set_title("Accuracy Plot Unavailable (Check metric names)")
        ax1.grid(True)
    if "loss" in history.history and "val_loss" in history.history:
        ax2.plot(history.history["loss"], label="Training Loss")
        ax2.plot(history.history["val_loss"], label="Validation Loss")
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_title("Training and Validation Loss")
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.set_title("Loss Plot Unavailable (Check metric names)")
        ax2.set_xlabel("Epoch")
        ax2.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Training history plot saved to {filename}")
    try:
        plt.show()
    except Exception as e:
        print(
            f"Note: Could not display plot directly ({e}). It has been saved to {filename}."
        )


def evaluate_with_metrics(model, X_test, y_test_onehot, class_names=None):
    print("\n--- Model Evaluation ---")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)
    unique_true_labels = np.unique(y_true)
    if not unique_true_labels.size:
        print(
            "Warning: No true labels found in the test set after filtering. Cannot evaluate."
        )
        return {
            "accuracy": 0,
            "recall_macro": 0,
            "f1_macro": 0,
            "y_true": y_true,
            "y_pred": y_pred,
            "recall_per_class": {},
            "f1_per_class": {},
        }
    if class_names is not None:
        target_names_report = [
            class_names[i] for i in unique_true_labels if i < len(class_names)
        ]
        labels_report = unique_true_labels
        if len(target_names_report) != len(labels_report):
            print(
                f"Warning: Mismatch between unique true labels ({len(labels_report)}) and available names ({len(target_names_report)}). Report might be incomplete."
            )
            if len(class_names) >= max(unique_true_labels) + 1:
                target_names_report = [class_names[i] for i in labels_report]
            else:
                target_names_report = [f"Class {i}" for i in labels_report]
    else:
        target_names_report = [f"Class {i}" for i in unique_true_labels]
        labels_report = unique_true_labels
    acc = accuracy_score(y_true, y_pred)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recall_per_class = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels_report
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels_report
    )
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names_report,
            labels=labels_report,
            zero_division=0,
        )
    )
    print("Overall Accuracy: {:.2f}%".format(acc * 100))
    print("Macro Avg Recall: {:.2f}%".format(recall_macro * 100))
    print("Macro Avg F1-Score: {:.2f}%".format(f1_macro * 100))
    results = {
        "accuracy": acc,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "y_true": y_true,
        "y_pred": y_pred,
        "recall_per_class": {},
        "f1_per_class": {},
    }
    print("\nPer-Class Metrics:")
    for i, label_index in enumerate(labels_report):
        name = target_names_report[i]
        recall_val = recall_per_class[i] if i < len(recall_per_class) else 0
        f1_val = f1_per_class[i] if i < len(f1_per_class) else 0
        results["recall_per_class"][name] = recall_val
        results["f1_per_class"][name] = f1_val
        print(f"  - {name}: Recall={recall_val:.4f}, F1-Score={f1_val:.4f}")
    return results


def cross_validate_model(
    X_segmented_data,
    y_segmented_data_int,
    input_segment_shape,
    num_classes,
    n_splits=CROSS_VAL_FOLDS,
):
    unique_labels_cv, counts_cv = np.unique(y_segmented_data_int, return_counts=True)
    valid_indices_cv = np.where(counts_cv >= MIN_SAMPLES_PER_CLASS_CV)[0]
    labels_to_keep_cv = unique_labels_cv[valid_indices_cv]
    global ACTIVITY_NAMES
    label_map_cv = {i: name for i, name in enumerate(ACTIVITY_NAMES)}
    if len(labels_to_keep_cv) < len(unique_labels_cv):
        print(f"\n!!! WARNING for Cross-Validation !!!")
        print(
            f"The following labels have fewer than {MIN_SAMPLES_PER_CLASS_CV} samples and will be EXCLUDED from cross-validation:"
        )
        labels_to_exclude_cv = unique_labels_cv[
            np.where(counts_cv < MIN_SAMPLES_PER_CLASS_CV)[0]
        ]
        for label in labels_to_exclude_cv:
            name = label_map_cv.get(label, f"Unknown {label}")
            count = counts_cv[np.where(unique_labels_cv == label)[0][0]]
            print(f"  - Label {label} ('{name}', Count: {count})")
        mask_cv = np.isin(y_segmented_data_int, labels_to_keep_cv)
        X_cv_filtered = X_segmented_data[mask_cv]
        y_cv_filtered_int = y_segmented_data_int[mask_cv]
        print(
            f"Using {len(y_cv_filtered_int)} samples for cross-validation after filtering."
        )
        if len(labels_to_keep_cv) < 2:
            print(
                "Error: Not enough classes with sufficient samples for cross-validation. Skipping CV."
            )
            return {
                "accuracy": {"mean": 0, "std": 0},
                "recall": {"mean": 0, "std": 0},
                "f1_score": {"mean": 0, "std": 0},
                "accuracies": [],
                "recalls": [],
                "f1_scores": [],
            }
    else:
        X_cv_filtered = X_segmented_data
        y_cv_filtered_int = y_segmented_data_int
        print("\nAll classes have sufficient samples for cross-validation.")
    if len(y_cv_filtered_int) == 0:
        print("Error: No data left after filtering for cross-validation. Skipping CV.")
        return {
            "accuracy": {"mean": 0, "std": 0},
            "recall": {"mean": 0, "std": 0},
            "f1_score": {"mean": 0, "std": 0},
            "accuracies": [],
            "recalls": [],
            "f1_scores": [],
        }
    min_samples_in_fold_classes = min(
        np.unique(y_cv_filtered_int, return_counts=True)[1]
    )
    actual_n_splits = min(n_splits, min_samples_in_fold_classes)
    if actual_n_splits < n_splits:
        print(
            f"Warning: Reducing n_splits for CV from {n_splits} to {actual_n_splits} due to smallest class having {min_samples_in_fold_classes} samples."
        )
    if actual_n_splits < 2:
        print(
            f"Error: Cannot perform cross-validation with less than 2 splits (smallest class has {min_samples_in_fold_classes} samples). Skipping CV."
        )
        return {
            "accuracy": {"mean": 0, "std": 0},
            "recall": {"mean": 0, "std": 0},
            "f1_score": {"mean": 0, "std": 0},
            "accuracies": [],
            "recalls": [],
            "f1_scores": [],
        }
    skf = StratifiedKFold(
        n_splits=actual_n_splits, shuffle=True, random_state=RANDOM_SEED
    )
    fold_accuracies = []
    fold_recalls_macro = []
    fold_f1s_macro = []
    print(
        f"\n--- Starting {actual_n_splits}-Fold Cross-Validation on (potentially filtered) Segmented Data ---"
    )
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_cv_filtered, y_cv_filtered_int)
    ):
        print(f"\n--- Fold {fold + 1}/{actual_n_splits} ---")
        X_train_fold_seg, X_val_fold_seg = (
            X_cv_filtered[train_idx],
            X_cv_filtered[val_idx],
        )
        y_train_fold_int, y_val_fold_int = (
            y_cv_filtered_int[train_idx],
            y_cv_filtered_int[val_idx],
        )
        print(
            f"Fold Train segments: {X_train_fold_seg.shape[0]}, Fold Validation segments: {X_val_fold_seg.shape[0]}"
        )
        X_train_fold_aug, y_train_fold_int_aug = augment_data(
            X_train_fold_seg, y_train_fold_int, noise_level=0.05
        )
        y_train_fold_aug_onehot = to_categorical(
            y_train_fold_int_aug, num_classes=num_classes
        )
        y_val_fold_onehot = to_categorical(y_val_fold_int, num_classes=num_classes)
        model = build_model(
            input_segment_shape, num_classes, TCN_FILTER, L2_FACTOR, LEARNING_RATE
        )
        callbacks_cv = [
            EarlyStopping(
                monitor="val_loss",
                patience=30,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6, verbose=1
            ),
        ]
        fold_unique_classes = np.unique(y_train_fold_int)
        fold_class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=fold_unique_classes, y=y_train_fold_int
        )
        fold_class_weights_dict = dict(zip(fold_unique_classes, fold_class_weights))
        print(f"Fold {fold+1} Class Weights: {fold_class_weights_dict}")
        model.fit(
            X_train_fold_aug,
            y_train_fold_aug_onehot,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val_fold_seg, y_val_fold_onehot),
            callbacks=callbacks_cv,
            class_weight=fold_class_weights_dict,
            verbose=1,
            shuffle=True,
        )
        y_pred_prob_fold = model.predict(X_val_fold_seg)
        y_pred_fold = np.argmax(y_pred_prob_fold, axis=1)
        y_true_fold = y_val_fold_int
        acc_fold = accuracy_score(y_true_fold, y_pred_fold)
        recall_fold_macro = recall_score(
            y_true_fold, y_pred_fold, average="macro", zero_division=0
        )
        f1_fold_macro = f1_score(
            y_true_fold, y_pred_fold, average="macro", zero_division=0
        )
        fold_accuracies.append(acc_fold)
        fold_recalls_macro.append(recall_fold_macro)
        fold_f1s_macro.append(f1_fold_macro)
        print(
            "Fold {} -> Accuracy: {:.4f}, Recall (Macro): {:.4f}, F1 (Macro): {:.4f}".format(
                fold + 1, acc_fold, recall_fold_macro, f1_fold_macro
            )
        )
        print("______________________________________________________")
        K.clear_session()
        del model
    mean_acc = np.mean(fold_accuracies) if fold_accuracies else 0
    std_acc = np.std(fold_accuracies) if fold_accuracies else 0
    mean_recall_macro = np.mean(fold_recalls_macro) if fold_recalls_macro else 0
    std_recall_macro = np.std(fold_recalls_macro) if fold_recalls_macro else 0
    mean_f1_macro = np.mean(fold_f1s_macro) if fold_f1s_macro else 0
    std_f1_macro = np.std(fold_f1s_macro) if fold_f1s_macro else 0
    print("\n--- Cross-Validation Summary ---")
    print("Mean Accuracy: {:.2f}% (± {:.2f}%)".format(mean_acc * 100, std_acc * 100))
    print(
        "Mean Recall (Macro): {:.2f}% (± {:.2f}%)".format(
            mean_recall_macro * 100, std_recall_macro * 100
        )
    )
    print(
        "Mean F1-Score (Macro): {:.2f}% (± {:.2f}%)".format(
            mean_f1_macro * 100, std_f1_macro * 100
        )
    )
    return {
        "accuracy": {"mean": mean_acc, "std": std_acc},
        "recall": {"mean": mean_recall_macro, "std": std_recall_macro},
        "f1_score": {"mean": mean_f1_macro, "std": std_f1_macro},
        "accuracies": fold_accuracies,
        "recalls": fold_recalls_macro,
        "f1_scores": fold_f1s_macro,
    }


def main():
    print("\n" + "=" * 80)
    print(f"WISDM ARFF DATASET PROCESSING WITH TCN MODEL")
    print(f"(Reverted Window=100, Restored Full Augmentation)")
    print(f"Window Size: {WINDOW_SIZE}, Step Size: {STEP_SIZE}")
    print(
        f"LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}, L2: {L2_FACTOR}, Min Samples Split: {MIN_SAMPLES_PER_CLASS_SPLIT}, Min Samples CV: {MIN_SAMPLES_PER_CLASS_CV}"
    )
    print("=" * 80)
    print("\n" + "=" * 30 + " 1. LOADING DATA " + "=" * 31)
    df_cleaned, identified_feature_names, class_name = load_wisdm_arff_data_manual(
        DATA_PATH
    )
    if df_cleaned.empty:
        print("Failed to load data. Exiting.")
        return
    print("\n" + "=" * 30 + " 2. PREPARING DATA " + "=" * 30)
    label_encoder = LabelEncoder()
    df_cleaned["activity_encoded"] = label_encoder.fit_transform(df_cleaned[class_name])
    global N_CLASSES, ACTIVITY_NAMES
    original_classes = list(label_encoder.classes_)
    N_CLASSES = len(original_classes)
    ACTIVITY_NAMES = original_classes
    print(f"Encoded {N_CLASSES} original activities: {ACTIVITY_NAMES}")
    valid_feature_names = [
        f for f in identified_feature_names if f in df_cleaned.columns
    ]
    if len(valid_feature_names) != N_FEATURES:
        print(
            f"Error: Feature mismatch after loading. Expected {N_FEATURES}, got {len(valid_feature_names)}. Exiting."
        )
        return
    if N_FEATURES == 0:
        print("Error: No valid features identified. Exiting.")
        return
    X = df_cleaned[valid_feature_names].values.astype(np.float32)
    y_int = df_cleaned["activity_encoded"].values
    print(f"Original data shapes before segmentation: X={X.shape}, y={y_int.shape}")
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled.")
    print("\n" + "=" * 28 + " 3. CREATING SEGMENTS " + "=" * 28)
    X_segmented, y_segmented_int = create_segments(
        X_scaled, y_int, WINDOW_SIZE, STEP_SIZE, N_FEATURES
    )
    if X_segmented.shape[0] == 0:
        print(
            "Error: No segments were created. Check window/step size relative to data length. Exiting."
        )
        return
    input_segment_shape = (WINDOW_SIZE, N_FEATURES)
    print(f"Input shape for model set to: {input_segment_shape}")
    print("\n" + "=" * 28 + " 4. CROSS-VALIDATION " + "=" * 28)
    cv_results = cross_validate_model(
        X_segmented_data=X_segmented,
        y_segmented_data_int=y_segmented_int,
        input_segment_shape=input_segment_shape,
        num_classes=N_CLASSES,
        n_splits=CROSS_VAL_FOLDS,
    )
    print("\n" + "=" * 27 + " 5. PREPARING FINAL SPLIT DATA " + "=" * 27)
    unique_labels_final, counts_final = np.unique(y_segmented_int, return_counts=True)
    valid_indices_final = np.where(counts_final >= MIN_SAMPLES_PER_CLASS_SPLIT)[0]
    labels_to_keep_final = unique_labels_final[valid_indices_final]
    label_map_final = {i: name for i, name in enumerate(ACTIVITY_NAMES)}
    X_final_filtered = X_segmented
    y_final_filtered_int = y_segmented_int
    if len(labels_to_keep_final) < len(unique_labels_final):
        print(f"!!! WARNING for Final Train/Test Split !!!")
        print(
            f"The following labels have fewer than {MIN_SAMPLES_PER_CLASS_SPLIT} samples and will be EXCLUDED from the final train/test split:"
        )
        labels_to_exclude_final = unique_labels_final[
            np.where(counts_final < MIN_SAMPLES_PER_CLASS_SPLIT)[0]
        ]
        for label in labels_to_exclude_final:
            name = label_map_final.get(label, f"Unknown {label}")
            count_idx = np.where(unique_labels_final == label)[0]
            count = counts_final[count_idx[0]] if count_idx.size > 0 else "N/A"
            print(f"  - Label {label} ('{name}', Count: {count})")
        mask_final = np.isin(y_segmented_int, labels_to_keep_final)
        X_final_filtered = X_segmented[mask_final]
        y_final_filtered_int = y_segmented_int[mask_final]
        print(
            f"Using {len(y_final_filtered_int)} samples for final train/test split after filtering."
        )
    else:
        print("All classes have sufficient samples for stratified train/test split.")
    if len(y_final_filtered_int) == 0:
        print("Error: No data remaining after filtering for final split. Exiting.")
        return
    num_unique_classes_after_filter = len(np.unique(y_final_filtered_int))
    if num_unique_classes_after_filter < 2:
        print(
            f"Error: Only {num_unique_classes_after_filter} class(es) remaining after filtering. Cannot perform train/test split. Exiting."
        )
        return
    print("\n" + "=" * 27 + " 6. TRAINING FINAL MODEL " + "=" * 27)
    X_train_seg, X_test_seg, y_train_int, y_test_int = train_test_split(
        X_final_filtered,
        y_final_filtered_int,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_final_filtered_int,
    )
    print(
        f"Final Train segments: {X_train_seg.shape[0]}, Final Test segments: {X_test_seg.shape[0]}"
    )
    X_train_aug, y_train_int_aug = augment_data(
        X_train_seg, y_train_int, noise_level=0.05
    )
    y_train_aug_onehot = to_categorical(y_train_int_aug, num_classes=N_CLASSES)
    y_test_onehot = to_categorical(y_test_int, num_classes=N_CLASSES)
    final_model = build_model(
        input_segment_shape, N_CLASSES, TCN_FILTER, L2_FACTOR, LEARNING_RATE
    )
    final_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            min_delta=0.0005,
            verbose=1,
        ),
        ModelCheckpoint(
            "best_wisdm_tcn_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
        ),
    ]
    print("\nCalculating class weights for final training...")
    unique_classes = np.unique(y_train_int)
    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=unique_classes, y=y_train_int
    )
    class_weights_dict = dict(zip(unique_classes, weights))
    print(f"Class Weights for final model: {class_weights_dict}")
    history = final_model.fit(
        X_train_aug,
        y_train_aug_onehot,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=final_callbacks,
        class_weight=class_weights_dict,
        verbose=1,
        shuffle=True,
    )
    print("\n" + "=" * 25 + " 7. EVALUATING FINAL MODEL " + "=" * 26)
    print("Loading best weights from training...")
    try:
        final_model.load_weights("best_wisdm_tcn_model.keras")
    except Exception as e:
        print(
            f"Warning: Could not load best weights ('best_wisdm_tcn_model.keras'). Evaluating with final weights from training. Error: {e}"
        )
    final_metrics = evaluate_with_metrics(
        final_model, X_test_seg, y_test_onehot, class_names=ACTIVITY_NAMES
    )
    print("\n" + "=" * 24 + " 8. GENERATING VISUALIZATIONS " + "=" * 24)
    plot_training_history(history, filename="training_history_wisdm_tcn_updated.png")
    print("\n" + "=" * 30 + " 9. SAVING RESULTS " + "=" * 31)
    results_summary_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Recall (Macro)", "F1 Score (Macro)"],
            "Final Model Test (%)": [
                final_metrics["accuracy"] * 100,
                final_metrics["recall_macro"] * 100,
                final_metrics["f1_macro"] * 100,
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
    results_filename = "model_metrics_wisdm_tcn_updated.csv"
    results_summary_df.to_csv(results_filename, index=False)
    print(f"Metrics summary saved to '{results_filename}'")
    recall_keys = final_metrics.get("recall_per_class", {}).keys()
    f1_keys = final_metrics.get("f1_per_class", {}).keys()
    if recall_keys and f1_keys:
        all_keys = sorted(list(set(recall_keys) | set(f1_keys)))
        recall_values = [
            final_metrics["recall_per_class"].get(k, 0.0) for k in all_keys
        ]
        f1_values = [final_metrics["f1_per_class"].get(k, 0.0) for k in all_keys]
        per_class_metrics_df = pd.DataFrame(
            {"Activity": all_keys, "Recall": recall_values, "F1-Score": f1_values}
        )
        per_class_filename = "per_class_metrics_wisdm_tcn_updated.csv"
        per_class_metrics_df.to_csv(per_class_filename, index=False)
        print(f"Per-class metrics saved to '{per_class_filename}'")
    else:
        print(
            "Could not save per-class metrics (keys not found in results dictionary)."
        )
    final_model_filename = "final_wisdm_tcn_model_updated.keras"
    final_model.save(final_model_filename)
    print(f"Final model saved as '{final_model_filename}'")
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
