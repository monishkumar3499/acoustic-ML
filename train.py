"""
train.py
========
Fixed & Enhanced Training Pipeline — Acoustic AI Multi-Output Regression

Key Improvements over Previous Version:
  1. Polynomial feature engineering (degree=2): captures T², S², T×S interactions
  2. Right-sized neural network (no over-parameterization for 2 inputs)
  3. Dropout reduced — smooth synthetic data doesn't need heavy regularization
  4. Cosine decay LR schedule instead of ReduceLROnPlateau heuristic
  5. XGBoost ensemble for tabular data strength
  6. Proper scaler_y saved and used everywhere (evaluate.py bug fixed)
  7. No hard-clamped Power ceiling — new dataset goes to 4.3 W naturally
  8. All scalers & poly transformer saved for consistent inference
"""

import os, random, time, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠ xgboost not installed — skipping ensemble. "
          "Install with: pip install xgboost")

import tensorflow as tf
from tensorflow import keras

from utils import (FREQ_MIN, FREQ_MAX, POWER_MIN, POWER_MAX,
                   print_metrics, plot_actual_vs_predicted,
                   plot_residuals, plot_residual_distribution)

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "new_dataset.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "new_models")

EPOCHS       = 500
BATCH_SIZE   = 64
PATIENCE     = 60
POLY_DEGREE  = 2          # adds T², S², T×S  → 5 features total

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    assert df["Temperature_C"].between(-15, 90).all(), "Temperature out of range"
    assert df["Salinity_ppt"].between(-1, 55).all(),   "Salinity out of range"
    X = df[["Temperature_C", "Salinity_ppt"]].values
    y = df[["Optimal_Frequency_kHz", "Optimal_Power_W"]].values
    print(f"✓ Loaded {len(df)} samples | "
          f"Freq: {y[:,0].min():.2f}–{y[:,0].max():.2f} kHz | "
          f"Power: {y[:,1].min():.2f}–{y[:,1].max():.2f} W")
    return X, y


# ── Neural Network ────────────────────────────────────────────────────────────
def build_nn(input_dim):
    """
    Right-sized network for 5-feature polynomial input.
    Smaller than before → less overfitting, faster convergence.
    """
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(128, activation="relu",
                            kernel_initializer="he_normal",
                            kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.10)(x)

    x = keras.layers.Dense(64, activation="relu",
                            kernel_initializer="he_normal",
                            kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.10)(x)

    x = keras.layers.Dense(32, activation="relu",
                            kernel_initializer="he_normal")(x)
    x = keras.layers.Dropout(0.05)(x)

    out = keras.layers.Dense(2, activation="linear")(x)

    # Cosine decay schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-3,
        decay_steps=EPOCHS * 100,   # generous decay span
        alpha=1e-5
    )

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss="huber",          # more robust to occasional noise than MSE
        metrics=["mae"]
    )
    return model


# ── Training curve ────────────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("Loss Curve (Huber)"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["mae"],     label="Train")
    axes[1].plot(history.history["val_mae"], label="Validation")
    axes[1].set_title("MAE Curve"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"✓ Training curves saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  ACOUSTIC AI — FIXED TRAINING PIPELINE")
    print("="*60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load
    X_raw, y = load_data()

    # 2. Polynomial features: [T, S] → [T, S, T², T·S, S²]
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
    X = poly.fit_transform(X_raw)
    print(f"✓ Feature expansion: {X_raw.shape[1]} → {X.shape[1]} features "
          f"{poly.get_feature_names_out(['T','S']).tolist()}")

    # 3. Split: 70 / 15 / 15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                 random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                  random_state=SEED)
    print(f"✓ Split → Train:{len(X_tr)}  Val:{len(X_val)}  Test:{len(X_te)}")

    # 4. Scale inputs
    scaler_X = StandardScaler().fit(X_tr)
    X_tr_s   = scaler_X.transform(X_tr)
    X_val_s  = scaler_X.transform(X_val)
    X_te_s   = scaler_X.transform(X_te)

    # 5. Scale targets — CRITICAL (fixes Power R² lag)
    scaler_y = StandardScaler().fit(y_tr)
    y_tr_s   = scaler_y.transform(y_tr)
    y_val_s  = scaler_y.transform(y_val)

    # ── Neural Network ────────────────────────────────────────────────────
    print("\n── Training Neural Network ──")
    model = build_nn(input_dim=X_tr_s.shape[1])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE,
        restore_best_weights=True, verbose=1
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.h5"),
        monitor="val_loss", save_best_only=True,
        save_weights_only=False
    )

    t0 = time.time()
    history = model.fit(
        X_tr_s, y_tr_s,
        validation_data=(X_val_s, y_val_s),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    print(f"✓ Neural network trained in {time.time()-t0:.1f}s  "
          f"({len(history.history['loss'])} epochs)")

    # NN evaluation
    def nn_predict(X_scaled):
        y_s = model.predict(X_scaled, verbose=0)
        return scaler_y.inverse_transform(y_s)

    print_metrics("NN TRAIN", y_tr,  nn_predict(X_tr_s))
    print_metrics("NN VAL",   y_val, nn_predict(X_val_s))
    print_metrics("NN TEST",  y_te,  nn_predict(X_te_s))

    plot_history(history)

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_model = None
    if XGB_AVAILABLE:
        print("\n── Training XGBoost ──")
        xgb_model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            ),
            n_jobs=-1
        )
        xgb_model.fit(X_tr_s, y_tr)   # XGBoost uses original-scale y
        print_metrics("XGB TRAIN", y_tr,  xgb_model.predict(X_tr_s))
        print_metrics("XGB VAL",   y_val, xgb_model.predict(X_val_s))
        print_metrics("XGB TEST",  y_te,  xgb_model.predict(X_te_s))

    # ── Ensemble (average NN + XGB) ───────────────────────────────────────
    if xgb_model is not None:
        print("\n── Ensemble (NN + XGBoost average) ──")
        def ensemble_predict(X_scaled):
            nn_pred  = nn_predict(X_scaled)
            xgb_pred = xgb_model.predict(X_scaled)
            return 0.5 * nn_pred + 0.5 * xgb_pred

        print_metrics("ENSEMBLE TRAIN", y_tr,  ensemble_predict(X_tr_s))
        print_metrics("ENSEMBLE VAL",   y_val, ensemble_predict(X_val_s))
        print_metrics("ENSEMBLE TEST",  y_te,  ensemble_predict(X_te_s))

        # Plots use ensemble predictions
        y_te_pred = ensemble_predict(X_te_s)
    else:
        y_te_pred = nn_predict(X_te_s)

    # ── Diagnostic plots ──────────────────────────────────────────────────
    plot_actual_vs_predicted(y_te, y_te_pred, MODEL_DIR)
    plot_residuals(y_te, y_te_pred, MODEL_DIR)
    plot_residual_distribution(y_te, y_te_pred, MODEL_DIR)

    # ── Save all artefacts ────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "poly.pkl"),     "wb") as f: pickle.dump(poly,     f)
    with open(os.path.join(MODEL_DIR, "scaler_X.pkl"), "wb") as f: pickle.dump(scaler_X, f)
    with open(os.path.join(MODEL_DIR, "scaler_y.pkl"), "wb") as f: pickle.dump(scaler_y, f)
    if xgb_model:
        with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)

    # Save legacy scaler name for backward compat
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f: pickle.dump(scaler_X, f)

    print("\n✓ All artefacts saved to", MODEL_DIR)
    print("✓ Training pipeline complete.\n")


if __name__ == "__main__":
    main()