"""
predict.py
==========
Inference Script — Acoustic AI Multi-Output Regression

Runtime Pipeline:
  User Input → Range Validation → Polynomial Expansion
             → Input Scaling → NN Prediction → Inverse Target Scaling
             → (Optional) XGBoost Prediction → Ensemble Average
             → Safety Clipping → Output Validation → Final Result

Usage:
    python predict.py                        # interactive
    python predict.py --temp 25 --sal 35     # direct args
"""

import os, pickle, argparse
import numpy as np

import tensorflow as tf
from tensorflow import keras

FREQ_MIN,  FREQ_MAX  =  48.0, 82.0
POWER_MIN, POWER_MAX =   0.35,  4.25

from utils import validate_input_ranges, validate_output_ranges

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "new_models")

NN_PATH    = os.path.join(MODEL_DIR, "best_model.h5")
POLY_PATH  = os.path.join(MODEL_DIR, "poly.pkl")
SX_PATH    = os.path.join(MODEL_DIR, "scaler_X.pkl")
SY_PATH    = os.path.join(MODEL_DIR, "scaler_y.pkl")
XGB_PATH   = os.path.join(MODEL_DIR, "xgb_model.pkl")

# Lazy-loaded singletons
_model    = None
_poly     = None
_scaler_X = None
_scaler_y = None
_xgb      = None


def _load():
    global _model, _poly, _scaler_X, _scaler_y, _xgb
    if _model is None:
        _model    = keras.models.load_model(NN_PATH)
        with open(POLY_PATH, "rb") as f: _poly     = pickle.load(f)
        with open(SX_PATH,   "rb") as f: _scaler_X = pickle.load(f)
        with open(SY_PATH,   "rb") as f: _scaler_y = pickle.load(f)
        if os.path.exists(XGB_PATH):
            with open(XGB_PATH, "rb") as f: _xgb  = pickle.load(f)


def predict_optimal_parameters(temperature: float, salinity: float) -> dict:
    """
    Predict optimal acoustic frequency and power from environmental conditions.

    Parameters
    ----------
    temperature : float   Temperature in °C  (valid: −10 to 85)
    salinity    : float   Salinity in ppt     (valid:   0 to 50)

    Returns
    -------
    dict with keys: temperature_C, salinity_ppt,
                    optimal_frequency_kHz, optimal_power_W
    """
    # Input validation
    is_valid, warnings = validate_input_ranges(temperature, salinity)
    for w in warnings:
        print(f"  ⚠  {w}")
    if not is_valid:
        raise ValueError("Input outside physically valid bounds — prediction aborted.")

    # Load models (lazy)
    _load()

    # Feature engineering
    X_raw   = np.array([[temperature, salinity]])
    X_poly  = _poly.transform(X_raw)
    X_scaled = _scaler_X.transform(X_poly)

    # NN prediction
    y_nn_scaled = _model.predict(X_scaled, verbose=0)
    y_nn        = _scaler_y.inverse_transform(y_nn_scaled)

    # XGBoost prediction (if available)
    if _xgb is not None:
        y_xgb = _xgb.predict(X_scaled)
        y_pred = 0.5 * y_nn + 0.5 * y_xgb
    else:
        y_pred = y_nn

    freq  = float(y_pred[0, 0])
    power = float(y_pred[0, 1])

    # Safety clipping
    freq  = np.clip(freq,  FREQ_MIN,  FREQ_MAX)
    power = np.clip(power, POWER_MIN, POWER_MAX)

    # Output validation
    _, out_warn = validate_output_ranges(freq, power)
    for w in out_warn:
        print(f"  ⚠  Output: {w}")

    return {
        "temperature_C":          temperature,
        "salinity_ppt":           salinity,
        "optimal_frequency_kHz":  round(freq,  4),
        "optimal_power_W":        round(power, 4),
    }


def _print_result(result: dict):
    print("\n" + "─"*42)
    print("  PREDICTION RESULT")
    print("─"*42)
    print(f"  Temperature  : {result['temperature_C']:.2f} °C")
    print(f"  Salinity     : {result['salinity_ppt']:.2f} ppt")
    print("─"*42)
    print(f"  Frequency    : {result['optimal_frequency_kHz']:.4f} kHz")
    print(f"  Power        : {result['optimal_power_W']:.4f} W")
    print("─"*42 + "\n")


def _get_user_input():
    while True:
        try:
            temp = float(input("  Enter Temperature (°C) : ").strip())
            sal  = float(input("  Enter Salinity   (ppt) : ").strip())
            return temp, sal
        except ValueError:
            print("  ✗ Invalid input. Please enter numeric values.")


def main():
    parser = argparse.ArgumentParser(description="Acoustic AI Predictor")
    parser.add_argument("--temp", type=float, default=None, help="Temperature °C")
    parser.add_argument("--sal",  type=float, default=None, help="Salinity ppt")
    args = parser.parse_args()

    print("\n" + "="*42)
    print("  ACOUSTIC AI — INFERENCE")
    print("="*42)

    if args.temp is not None and args.sal is not None:
        # Direct argument mode
        result = predict_optimal_parameters(args.temp, args.sal)
        _print_result(result)
    else:
        # Interactive loop
        print("  (Press Ctrl+C to exit)\n")
        while True:
            try:
                temp, sal = _get_user_input()
                result = predict_optimal_parameters(temp, sal)
                _print_result(result)
            except ValueError as e:
                print(f"  ✗ {e}")
            except KeyboardInterrupt:
                print("\n  Goodbye.")
                break


if __name__ == "__main__":
    main()