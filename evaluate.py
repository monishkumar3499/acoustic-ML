"""
evaluate.py
===========
Fixed Research Evaluation Script — Acoustic AI Multi-Output Regression

Bug Fixed: Now correctly loads scaler_y and applies inverse_transform
           before computing metrics. Previous version compared scaled
           predictions against original-scale ground truth — invalid.

Usage:
    python evaluate.py
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras

from utils import print_metrics, plot_actual_vs_predicted, plot_residuals, plot_residual_distribution

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "new_models")
DATA_PATH  = os.path.join(BASE_DIR, "new_dataset.csv")

NN_MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.h5")
POLY_PATH       = os.path.join(MODEL_DIR, "poly.pkl")
SCALER_X_PATH   = os.path.join(MODEL_DIR, "scaler_X.pkl")
SCALER_Y_PATH   = os.path.join(MODEL_DIR, "scaler_y.pkl")
XGB_PATH        = os.path.join(MODEL_DIR, "xgb_model.pkl")

SEED = 42


def load_artefacts():
    model    = keras.models.load_model(NN_MODEL_PATH)
    with open(POLY_PATH,     "rb") as f: poly     = pickle.load(f)
    with open(SCALER_X_PATH, "rb") as f: scaler_X = pickle.load(f)
    with open(SCALER_Y_PATH, "rb") as f: scaler_y = pickle.load(f)

    xgb_model = None
    if os.path.exists(XGB_PATH):
        with open(XGB_PATH, "rb") as f: xgb_model = pickle.load(f)

    return model, poly, scaler_X, scaler_y, xgb_model


def prepare_test_data(poly, scaler_X):
    """
    Recreate the EXACT same test split used in train.py.
    Uses same random_state=42 and split fractions.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures

    df   = pd.read_csv(DATA_PATH)
    X_raw = df[["Temperature_C", "Salinity_ppt"]].values
    y     = df[["Optimal_Frequency_kHz", "Optimal_Power_W"]].values

    X = poly.transform(X_raw)   # use fitted poly (do NOT refit)

    _, X_tmp, _, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    _, X_te, _, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)

    X_te_s = scaler_X.transform(X_te)
    return X_te_s, y_te


def main():
    print("\n" + "="*60)
    print("  ACOUSTIC AI — FIXED EVALUATION PIPELINE")
    print("="*60)

    model, poly, scaler_X, scaler_y, xgb_model = load_artefacts()
    X_te_s, y_te = prepare_test_data(poly, scaler_X)

    print(f"✓ Test samples: {len(X_te_s)}")

    # ── Neural Network prediction ─────────────────────────────────────────
    import time
    t0 = time.time()
    y_pred_nn_scaled = model.predict(X_te_s, verbose=0)
    elapsed = time.time() - t0

    # ⭐ CRITICAL FIX: inverse transform targets before metric computation
    y_pred_nn = scaler_y.inverse_transform(y_pred_nn_scaled)

    print(f"✓ NN inference time: {elapsed:.4f}s  "
          f"({elapsed/len(X_te_s)*1000:.3f} ms/sample)")

    print_metrics("NEURAL NETWORK TEST", y_te, y_pred_nn)

    # ── XGBoost ───────────────────────────────────────────────────────────
    if xgb_model is not None:
        y_pred_xgb = xgb_model.predict(X_te_s)
        print_metrics("XGBOOST TEST", y_te, y_pred_xgb)

        # ── Ensemble ──────────────────────────────────────────────────────
        y_pred_ens = 0.5 * y_pred_nn + 0.5 * y_pred_xgb
        print_metrics("ENSEMBLE TEST (NN+XGB)", y_te, y_pred_ens)

        y_final = y_pred_ens
    else:
        y_final = y_pred_nn

    # ── Diagnostic plots ──────────────────────────────────────────────────
    plot_actual_vs_predicted(y_te, y_final, MODEL_DIR)
    plot_residuals(y_te, y_final, MODEL_DIR)
    plot_residual_distribution(y_te, y_final, MODEL_DIR)

    print("\n✓ Evaluation complete. Plots saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()