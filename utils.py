"""
Utility Functions for Thermo-Acoustic Ultrasonic Optimization System
FINAL — WIDE RANGE + TARGET-SCALING AWARE

Temperature: −10 → 85 °C
Salinity: 0 → 50 ppt
Frequency: 35 → 110 kHz (safe operating envelope)
Power: 0.2 → 3.5 W

Enhancements:
- Physics-aware validation
- Safer MAPE computation
- Wide-range guards aligned with new dataset
- Output validation added
- Surface plot supports target-scaled models
- Improved grid coverage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# 🔍 CONSTANTS (aligned with NEW wide-range dataset)
# ============================================================

TEMP_MIN, TEMP_MAX = -10.0, 85.0
SAL_MIN, SAL_MAX = 0.0, 50.0

# Keep slightly wider than dataset for safety/future retraining
FREQ_MIN, FREQ_MAX = 35.0, 110.0
POWER_MIN, POWER_MAX = 0.2, 3.5


# ============================================================
# ✅ INPUT VALIDATION
# ============================================================

def validate_input_ranges(temperature, salinity):
    """
    Validate input parameters against training and physical limits.
    """
    warnings = []
    is_valid = True

    # Temperature checks
    if temperature < TEMP_MIN or temperature > TEMP_MAX:
        warnings.append(
            f"Temperature {temperature}°C outside training range ({TEMP_MIN}–{TEMP_MAX} °C)"
        )

        if temperature < -20 or temperature > 110:
            is_valid = False
            warnings.append("Temperature outside physically reasonable bounds")

    # Salinity checks
    if salinity < SAL_MIN or salinity > SAL_MAX:
        warnings.append(
            f"Salinity {salinity} ppt outside training range ({SAL_MIN}–{SAL_MAX} ppt)"
        )

        if salinity < 0:
            is_valid = False
            warnings.append("Salinity cannot be negative")

    return is_valid, warnings


# ============================================================
# ✅ OUTPUT VALIDATION
# ============================================================

def validate_output_ranges(freq_khz, power_w):
    """
    Validate predicted acoustic parameters.
    Useful for catching model drift in deployment.
    """
    warnings = []
    is_valid = True

    if not (FREQ_MIN <= freq_khz <= FREQ_MAX):
        warnings.append(
            f"Frequency {freq_khz:.3f} kHz outside expected range ({FREQ_MIN}–{FREQ_MAX})"
        )

    if not (POWER_MIN <= power_w <= POWER_MAX):
        warnings.append(
            f"Power {power_w:.3f} W outside expected range ({POWER_MIN}–{POWER_MAX})"
        )

    return is_valid, warnings


# ============================================================
# 📊 METRICS
# ============================================================

def _safe_mape(y_true, y_pred, eps=1e-8):
    """Numerically stable MAPE."""
    y_true_safe = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def calculate_comprehensive_metrics(y_true, y_pred):
    """Research-grade regression metrics."""
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error(y_true, y_pred),
        "mape": _safe_mape(y_true, y_pred),
    }


# ============================================================
# 📉 RESIDUAL ANALYSIS
# ============================================================

def create_residual_plot(y_true, y_pred, output_name, save_path=None):
    """Create residual diagnostics."""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.4)
    ax1.axhline(y=0, linestyle="--", linewidth=2)
    ax1.set_xlabel(f"Predicted {output_name}")
    ax1.set_ylabel("Residuals")
    ax1.set_title(f"{output_name}: Residual Plot", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, linestyle="--", linewidth=2)
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"{output_name}: Residual Distribution", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ============================================================
# 📈 PREDICTION COMPARISON
# ============================================================

def create_prediction_comparison(y_true, y_pred, output_name, save_path=None):
    """Actual vs predicted scatter."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.6, s=25, edgecolors="k", linewidths=0.4)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", linewidth=2, label="Perfect Prediction")

    ax.set_xlabel(f"Actual {output_name}")
    ax.set_ylabel(f"Predicted {output_name}")
    ax.set_title(f"{output_name}: Actual vs Predicted", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    r2 = r2_score(y_true, y_pred)
    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ============================================================
# 🔲 PARAMETER GRID
# ============================================================

def generate_parameter_grid(
    temp_range=(TEMP_MIN, TEMP_MAX),
    sal_range=(SAL_MIN, SAL_MAX),
    n_points=30,
):
    """Generate dense grid covering full training domain."""
    temp_vals = np.linspace(temp_range[0], temp_range[1], n_points)
    sal_vals = np.linspace(sal_range[0], sal_range[1], n_points)
    return np.meshgrid(temp_vals, sal_vals)


# ============================================================
# 🌐 3D SURFACE PLOT (TARGET-SCALING AWARE)
# ============================================================

def create_surface_plot(
    model,
    scaler_X,
    scaler_y,  # ⭐ NEW (important)
    output_idx,
    output_name,
    save_path=None,
):
    """
    Surface visualization that correctly handles target-scaled models.
    """

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    temp_grid, sal_grid = generate_parameter_grid()

    X = np.column_stack([temp_grid.ravel(), sal_grid.ravel()])
    X_scaled = scaler_X.transform(X)

    preds_scaled = model.predict(X_scaled, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled)  # ⭐ critical fix

    Z = preds[:, output_idx].reshape(temp_grid.shape)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(temp_grid, sal_grid, Z, cmap="viridis", alpha=0.85)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Salinity (ppt)")
    ax.set_zlabel(output_name)
    ax.set_title(f"{output_name} vs Environmental Conditions", fontweight="bold")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ============================================================
# 📤 EXPORT
# ============================================================

def export_predictions_to_csv(temperatures, salinities, predictions, filepath):
    """Export inference results."""
    df = pd.DataFrame(
        {
            "Temperature_C": temperatures,
            "Salinity_ppt": salinities,
            "Predicted_Frequency_kHz": predictions[:, 0],
            "Predicted_Power_W": predictions[:, 1],
        }
    )
    df.to_csv(filepath, index=False)
    return df


# ============================================================
# 🧾 REPORTING
# ============================================================

def print_performance_summary(metrics, output_name="Overall"):
    print(f"\n{output_name} Performance Metrics:")
    print("-" * 50)
    print(f"  R² Score:                {metrics['r2']:.6f}")
    print(f"  Mean Absolute Error:     {metrics['mae']:.6f}")
    print(f"  Root Mean Squared Error: {metrics['rmse']:.6f}")
    print(f"  Mean Squared Error:      {metrics['mse']:.6f}")
    print(f"  Mean Absolute % Error:   {metrics['mape']:.4f}%")


def check_model_requirements(r2_score_value, target_r2=0.95):
    meets = r2_score_value >= target_r2

    if meets:
        status = "PASS"
        message = f"Model achieves target R² ≥ {target_r2}"
    else:
        status = "FAIL"
        message = f"Model R² ({r2_score_value:.4f}) below target ({target_r2})"

    return meets, status, message


# ============================================================
# 💾 METADATA
# ============================================================

def save_model_metadata(filepath, model_info):
    import json
    with open(filepath, "w") as f:
        json.dump(model_info, f, indent=4)


def load_model_metadata(filepath):
    import json
    with open(filepath, "r") as f:
        return json.load(f)