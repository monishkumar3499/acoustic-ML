"""
Evaluation Script (RESEARCH-GRADE VISUALS)
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "new_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "new_models", "best_model.h5")  # ⭐ important
SCALER_PATH = os.path.join(BASE_DIR, "new_models", "scaler.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "new_models")

EPSILON = 1e-8


# =========================================================
# LOADERS
# =========================================================
def load_model_and_scaler():
    model = keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler_X = pickle.load(f)
    return model, scaler_X


def load_test_data():
    df = pd.read_csv(DATA_PATH)

    X = df[["Temperature_C", "Salinity_ppt"]].values
    y = df[["Optimal_Frequency_kHz", "Optimal_Power_W"]].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_test, y_test


# =========================================================
# CORE EVALUATION
# =========================================================
def evaluate_performance(model, scaler_X, X_test, y_test):
    X_test_scaled = scaler_X.transform(X_test)
    y_pred = model.predict(X_test_scaled, verbose=0)

    for i, name in enumerate(["Frequency", "Power"]):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))

        mape = (
            np.mean(
                np.abs(
                    (y_test[:, i] - y_pred[:, i])
                    / (y_test[:, i] + EPSILON)
                )
            )
            * 100
        )

        print(
            f"\n{name} -> "
            f"R2:{r2:.4f} "
            f"MAE:{mae:.4f} "
            f"RMSE:{rmse:.4f} "
            f"MAPE:{mape:.2f}%"
        )

    return y_pred


# =========================================================
# ⭐ GRAPH 1 — MODEL PERFORMANCE (MOST IMPORTANT)
# =========================================================
def plot_actual_vs_predicted(y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Frequency (kHz)", "Power (W)"]

    for i in range(2):
        axes[i].scatter(
            y_test[:, i],
            y_pred[:, i],
            alpha=0.6,
            edgecolors="k",
        )

        axes[i].plot(
            [y_test[:, i].min(), y_test[:, i].max()],
            [y_test[:, i].min(), y_test[:, i].max()],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
        )

        r2 = r2_score(y_test[:, i], y_pred[:, i])

        axes[i].set_title(
            f"{titles[i]} — Model Performance (R² = {r2:.4f})"
        )
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_performance.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"✓ Saved: {path}")


# =========================================================
# ⭐ GRAPH 2 — RESIDUAL PLOT (BIAS CHECK)
# =========================================================
def plot_residuals(y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Frequency Residuals", "Power Residuals"]

    for i in range(2):
        residuals = y_test[:, i] - y_pred[:, i]

        axes[i].scatter(
            y_pred[:, i],
            residuals,
            alpha=0.6,
            edgecolors="k",
        )
        axes[i].axhline(0, linestyle="--", linewidth=2)

        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Residual (Actual - Predicted)")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "residual_plot.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"✓ Saved: {path}")


# =========================================================
# ⭐ GRAPH 3 — RESIDUAL DISTRIBUTION (ERROR QUALITY)
# =========================================================
def plot_residual_distribution(y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Frequency Error Distribution", "Power Error Distribution"]

    for i in range(2):
        residuals = y_test[:, i] - y_pred[:, i]

        axes[i].hist(residuals, bins=40, edgecolor="black", alpha=0.7)
        axes[i].axvline(0, linestyle="--", linewidth=2)

        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Residual")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "residual_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"✓ Saved: {path}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n=== RESEARCH EVALUATION ===")

    model, scaler_X = load_model_and_scaler()
    X_test, y_test = load_test_data()

    y_pred = evaluate_performance(model, scaler_X, X_test, y_test)

    # ⭐ Generate research graphs
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_residual_distribution(y_test, y_pred)

    print("\n✓ All research graphs generated successfully")


if __name__ == "__main__":
    main()