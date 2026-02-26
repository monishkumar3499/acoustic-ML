import os
import numpy as np
import pickle
from tensorflow import keras

from utils import (
    validate_input_ranges,
    validate_output_ranges,
    FREQ_MIN,
    FREQ_MAX,
    POWER_MIN,
    POWER_MAX,
)

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "new_models", "best_model.h5")
SCALER_X_PATH = os.path.join(BASE_DIR, "new_models", "scaler.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "new_models", "scaler_y.pkl")


# =========================================================
# DRIVE VOLTAGE CONVERSION
# =========================================================
def predict_drive_voltage(power_acoustic_w,
                          impedance_ohm=8.0,
                          efficiency=0.4):
    """
    Convert predicted acoustic power to required drive voltage.
    Returns RMS and peak voltage for sine drive.
    """

    if efficiency <= 0 or efficiency > 1:
        raise ValueError("Efficiency must be between 0 and 1")

    if impedance_ohm <= 0:
        raise ValueError("Impedance must be positive")

    # Electrical power required
    P_elec = power_acoustic_w / efficiency

    # RMS voltage
    V_rms = np.sqrt(P_elec * impedance_ohm)

    # Peak voltage for sine wave
    V_peak = np.sqrt(2) * V_rms

    return V_rms, V_peak


# =========================================================
# LOADERS
# =========================================================
def load_model_and_scalers():
    model = keras.models.load_model(MODEL_PATH)

    with open(SCALER_X_PATH, "rb") as f:
        scaler_X = pickle.load(f)

    with open(SCALER_Y_PATH, "rb") as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y


# =========================================================
# PREDICTION CORE
# =========================================================
def predict_optimal_parameters(temperature, salinity,
                               model=None,
                               scaler_X=None,
                               scaler_y=None):

    # ---------- input validation ----------
    is_valid, warnings = validate_input_ranges(temperature, salinity)

    for w in warnings:
        print(f"⚠ Warning: {w}")

    if not is_valid:
        raise ValueError("❌ Input outside physically valid bounds")

    # ---------- lazy loading ----------
    if model is None or scaler_X is None or scaler_y is None:
        model, scaler_X, scaler_y = load_model_and_scalers()

    # ---------- prepare input ----------
    X = np.array([[temperature, salinity]])
    X_scaled = scaler_X.transform(X)

    # ---------- model prediction (scaled space) ----------
    y_scaled = model.predict(X_scaled, verbose=0)

    # ⭐ CRITICAL: inverse transform
    y_pred = scaler_y.inverse_transform(y_scaled)

    freq = float(y_pred[0, 0])
    power = float(y_pred[0, 1])

    # ---------- safety clipping (deployment guard) ----------
    freq = float(np.clip(freq, FREQ_MIN, FREQ_MAX))
    power = float(np.clip(power, POWER_MIN, POWER_MAX))

    # ---------- output validation ----------
    _, out_warnings = validate_output_ranges(freq, power)
    for w in out_warnings:
        print(f"⚠ Output Warning: {w}")

    return {
        "temperature_C": temperature,
        "salinity_ppt": salinity,
        "optimal_frequency_kHz": freq,
        "optimal_power_W": power,
    }


# =========================================================
# USER INPUT
# =========================================================
def get_user_input():
    while True:
        try:
            temp = float(input("Enter Temperature (°C): ").strip())
            sal = float(input("Enter Salinity (ppt): ").strip())
            return temp, sal
        except ValueError:
            print("❌ Invalid input. Please enter numeric values.\n")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n=== ACOUSTIC PARAMETER PREDICTOR ===\n")

    temperature, salinity = get_user_input()

    model, scaler_X, scaler_y = load_model_and_scalers()

    result = predict_optimal_parameters(
        temperature,
        salinity,
        model,
        scaler_X,
        scaler_y,
    )

    # ---- compute driving voltage ----
    V_rms, V_peak = predict_drive_voltage(
        result["optimal_power_W"],
        impedance_ohm=8.0,
        efficiency=0.4,
    )

    print("\n Optimal Parameters")
    print("-" * 40)
    print(f"Temperature (°C):        {result['temperature_C']}")
    print(f"Salinity (ppt):          {result['salinity_ppt']}")
    print(f"Optimal Frequency (kHz): {result['optimal_frequency_kHz']:.4f}")
    print(f"Optimal Power (W):       {result['optimal_power_W']:.4f}")
    print(f"Drive Voltage (RMS):     {V_rms:.4f} V")
    print(f"Drive Voltage (Peak):    {V_peak:.4f} V")
    print("-" * 40)


if __name__ == "__main__":
    main()