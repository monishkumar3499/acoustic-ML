# ============================================================
# utils.py
# Utility Functions — Acoustic AI Multi-Output Regression
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Physical constants ───────────────────────────────────────────────────────
TEMP_MIN,  TEMP_MAX  = -10.0, 85.0
SAL_MIN,   SAL_MAX   =   0.0, 50.0
FREQ_MIN,  FREQ_MAX  =  48.0, 82.0
POWER_MIN, POWER_MAX =   0.35,  4.25


# ── Input validation ─────────────────────────────────────────────────────────
def validate_input_ranges(temperature, salinity):
    warnings, is_valid = [], True

    if not (TEMP_MIN <= temperature <= TEMP_MAX):
        warnings.append(f"Temperature {temperature}°C outside training range "
                        f"({TEMP_MIN}–{TEMP_MAX} °C)")
        if temperature < -25 or temperature > 120:
            is_valid = False
            warnings.append("Temperature outside physically reasonable bounds")

    if not (SAL_MIN <= salinity <= SAL_MAX):
        warnings.append(f"Salinity {salinity} ppt outside training range "
                        f"({SAL_MIN}–{SAL_MAX} ppt)")
        if salinity < 0:
            is_valid = False
            warnings.append("Salinity cannot be negative")

    return is_valid, warnings


# ── Output validation ────────────────────────────────────────────────────────
def validate_output_ranges(freq_khz, power_w):
    warnings, is_valid = [], True

    if not (FREQ_MIN <= freq_khz <= FREQ_MAX):
        warnings.append(f"Frequency {freq_khz:.3f} kHz outside expected range "
                        f"({FREQ_MIN}–{FREQ_MAX})")
    if not (POWER_MIN <= power_w <= POWER_MAX):
        warnings.append(f"Power {power_w:.3f} W outside expected range "
                        f"({POWER_MIN}–{POWER_MAX})")

    return is_valid, warnings


# ── Metrics ──────────────────────────────────────────────────────────────────
def _safe_mape(y_true, y_pred, eps=1e-8):
    y_safe = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_safe)) * 100


def calculate_metrics(y_true, y_pred):
    return {
        "r2":   r2_score(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": _safe_mape(y_true, y_pred),
    }


def print_metrics(stage, y_true, y_pred, output_names=("Frequency (kHz)", "Power (W)")):
    print(f"\n{'='*52}")
    print(f"  {stage} METRICS")
    print(f"{'='*52}")
    r2_list = []
    for i, name in enumerate(output_names):
        m = calculate_metrics(y_true[:, i], y_pred[:, i])
        r2_list.append(m["r2"])
        print(f"  {name:<18}  R²={m['r2']:.4f}  MAE={m['mae']:.4f}  "
              f"RMSE={m['rmse']:.4f}  MAPE={m['mape']:.2f}%")
    print(f"  {'Overall':<18}  R²={np.mean(r2_list):.4f}")
    print(f"{'='*52}")


# ── Plotting helpers ─────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_true, y_pred, output_dir="new_models"):
    import os
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Frequency (kHz): Actual vs Predicted",
              "Power (W): Actual vs Predicted"]
    for i in range(2):
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        axes[i].scatter(y_true[:, i], y_pred[:, i],
                        alpha=0.5, s=18, edgecolors="k", linewidths=0.3)
        axes[i].plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect")
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        axes[i].set_title(f"{titles[i]}  (R²={r2:.4f})", fontweight="bold")
        axes[i].set_xlabel("Actual"); axes[i].set_ylabel("Predicted")
        axes[i].legend(); axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"✓ Saved: {path}")


def plot_residuals(y_true, y_pred, output_dir="new_models"):
    import os
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Frequency Residuals", "Power Residuals"]
    for i in range(2):
        res = y_true[:, i] - y_pred[:, i]
        axes[i].scatter(y_pred[:, i], res, alpha=0.5, s=18,
                        edgecolors="k", linewidths=0.3)
        axes[i].axhline(0, color="r", linestyle="--", linewidth=2)
        axes[i].set_title(titles[i], fontweight="bold")
        axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Residual")
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "residual_plot.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"✓ Saved: {path}")


def plot_residual_distribution(y_true, y_pred, output_dir="new_models"):
    import os
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Frequency Error Distribution", "Power Error Distribution"]
    for i in range(2):
        res = y_true[:, i] - y_pred[:, i]
        axes[i].hist(res, bins=50, edgecolor="black", alpha=0.75)
        axes[i].axvline(0, color="r", linestyle="--", linewidth=2)
        axes[i].set_title(titles[i], fontweight="bold")
        axes[i].set_xlabel("Residual"); axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "residual_distribution.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"✓ Saved: {path}")