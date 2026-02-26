"""
Training Script (RESEARCH-GRADE LOGGING — Option 2 Style)
UPDATED WITH TARGET SCALING (CRITICAL FIX)
"""

import os
import random
import time
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ================= PATH SETUP =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "new_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "new_models")

# ================= REPRODUCIBILITY =================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

EPOCHS = 500
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 50


# =========================================================
# DATA
# =========================================================
def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Sanity checks
    assert df["Temperature_C"].between(-10, 85).all(), "Temperature out of range"
    assert df["Salinity_ppt"].between(0, 50).all(), "Salinity out of range"

    X = df[["Temperature_C", "Salinity_ppt"]].values
    y = df[["Optimal_Frequency_kHz", "Optimal_Power_W"]].values

    return X, y


def create_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def create_target_scaler(y_train):
    scaler = StandardScaler()
    scaler.fit(y_train)
    return scaler


# =========================================================
# MODEL
# =========================================================
def build_model(input_dim=2, output_dim=2):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(output_dim, activation="linear"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    return model


# =========================================================
# ⭐ METRIC REPORT
# =========================================================
def print_detailed_metrics(stage_name, y_true, y_pred):
    print(f"\n===== {stage_name} METRICS =====")

    outputs = ["Frequency", "Power"]
    r2_list = []

    for i, name in enumerate(outputs):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2_list.append(r2)

        print(
            f"{name}: "
            f"R2={r2:.4f}  "
            f"MAE={mae:.4f}  "
            f"RMSE={rmse:.4f}"
        )

    overall_r2 = np.mean(r2_list)
    print(f"Overall R2: {overall_r2:.4f}")
    print("=" * 40)


# =========================================================
# ⭐ TRAINING CURVE
# =========================================================
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["mae"], label="Train")
    axes[1].plot(history.history["val_mae"], label="Validation")
    axes[1].set_title("MAE Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✓ Training curves saved → {save_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n=== TRAINING PIPELINE ===")

    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_and_preprocess_data()

    # ================= SPLIT =================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )

    # ================= SCALING =================
    scaler_X = create_scaler(X_train)
    scaler_y = create_target_scaler(y_train)

    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # ================= MODEL =================
    model = build_model()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.h5"),
        monitor="val_loss",
        save_best_only=True,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=1e-7
    )

    # ================= TRAIN =================
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=1,
    )

    # ================= TRAIN METRICS =================
    y_train_pred_scaled = model.predict(X_train_scaled, verbose=0)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    print_detailed_metrics("TRAIN", y_train, y_train_pred)

    # ================= VALIDATION METRICS =================
    y_val_pred_scaled = model.predict(X_val_scaled, verbose=0)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
    print_detailed_metrics("VALIDATION", y_val, y_val_pred)

    # ================= TEST METRICS =================
    start = time.time()
    y_test_pred_scaled = model.predict(X_test_scaled, verbose=0)
    print(f"Inference time: {time.time() - start:.4f} sec")

    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    print_detailed_metrics("TEST", y_test, y_test_pred)

    # ================= SAVE SCALERS =================
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler_X, f)

    with open(os.path.join(MODEL_DIR, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)

    # ================= PLOTS =================
    plot_training_history(history)

    print("\n✓ Training completed successfully")


if __name__ == "__main__":
    main()