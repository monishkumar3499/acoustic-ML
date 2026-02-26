# Acoustic AI — Thermo-Acoustic Parameter Prediction

> A physics-aware multi-output regression system that predicts optimal underwater acoustic operating parameters from environmental conditions using a Neural Network + XGBoost ensemble.

---

## Overview

This project predicts the **optimal acoustic frequency** and **transmission power** required for efficient underwater sound communication, given the water's temperature and salinity. It is designed for use in oceanographic sensor networks, underwater modems, sonar calibration, and acoustic communication research.

### What It Does

| Input | Unit | Range |
|---|---|---|
| Temperature | °C | −10 → 85 |
| Salinity | ppt | 0 → 50 |

| Output | Unit | Range |
|---|---|---|
| Optimal Frequency | kHz | ~49 → 71 |
| Optimal Power | W | ~0.35 → 4.2 |

### Why It Matters

In underwater acoustics, the optimal transmission frequency and power are not fixed — they depend on environmental conditions. Water temperature directly affects the **speed of sound** (and therefore the optimal frequency), while salinity increases **acoustic attenuation** (and therefore the required power). Getting these parameters wrong causes signal loss, wasted energy, or failed communication. This model automates that optimization in real time.

---

## Project Structure

```
acoustic_ai/
│
├── new_dataset.csv          # 5000-sample enhanced training dataset
├── train.py                 # Full training pipeline (NN + XGBoost ensemble)
├── evaluate.py              # Evaluation on held-out test set
├── predict.py               # Interactive / CLI inference
├── utils.py                 # Shared constants, validation, metrics, plots
├── test_acoustic_ai.py      # 40-case test suite
│
└── new_models/              # Created after running train.py
    ├── best_model.h5        # Trained neural network weights
    ├── poly.pkl             # Polynomial feature transformer
    ├── scaler_X.pkl         # Input StandardScaler
    ├── scaler_y.pkl         # Target StandardScaler
    ├── xgb_model.pkl        # XGBoost model (if xgboost installed)
    └── training_history.png # Loss and MAE curves
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib xgboost
```

> XGBoost is optional but recommended — it enables the ensemble and boosts accuracy to ~99% R².

### 2. Train

```bash
python train.py
```

Trains the neural network and XGBoost model, saves all artefacts to `new_models/`, prints metrics for train / validation / test splits, and saves diagnostic plots.

### 3. Evaluate

```bash
python evaluate.py
```

Reconstructs the exact test split and reports R², MAE, RMSE, and MAPE. Saves three diagnostic plots.

### 4. Predict

```bash
# Interactive mode
python predict.py

# Direct argument mode
python predict.py --temp 25 --sal 35
```

### 5. Run Tests

```bash
python test_acoustic_ai.py           # Full 40-case suite
python test_acoustic_ai.py --no-model  # Dataset + validation only (no model needed)
```

---

## Dataset

`new_dataset.csv` contains **5000 samples** covering the full operating domain of the model. It was constructed using a physics-derived second-order polynomial model with small Gaussian noise to simulate real measurement variance.

**Physics Formulas the data is based on:**

```
Frequency (kHz) = 52.0 + 0.22·T − 0.0008·T² + 0.08·S + 0.0004·T·S

Power (W) = 0.50 + 0.050·S − 0.0004·S² + 0.018·T − 0.00006·T² + 0.00025·T·S
```

where T = Temperature (°C), S = Salinity (ppt).

**Physical justification:**
- Temperature raises the speed of sound in water (~1420 m/s at 5°C → ~1555 m/s at 80°C), requiring a higher optimal transmission frequency.
- Salinity increases water density and acoustic attenuation, requiring more power to maintain signal quality at the receiver.
- The T×S cross-term captures the nonlinear coupling between the two effects.

### Dataset Composition

| Type | Count | Method |
|---|---|---|
| Grid samples | 2500 | Uniform 50×50 grid over full domain |
| Random samples | 2500 | Random uniform draws |
| **Total** | **5000** | Shuffled, no duplicates |

The structured grid ensures uniform coverage across the entire domain. The random samples add density in realistic operating regions and prevent the model from only learning at grid points.

### Key Improvement Over Previous Dataset

The original dataset hard-clamped Power at **3.5 W**, creating a flat plateau where hundreds of different (T, S) combinations all mapped to the same output value. This made Power R² fundamentally limited. The new dataset has a natural maximum of **~4.15 W** at the hottest, saltiest corner — no artificial ceiling, no discontinuity.

---

## Model Architecture

### Pipeline

```
Raw Input [T, S]
       ↓
Polynomial Expansion → [T, S, T², T·S, S²]  (5 features)
       ↓
StandardScaler (input)
       ↓
┌─────────────────────────────────┐
│   Neural Network                │
│   Dense(128) + BN + Dropout(0.1)│
│   Dense(64)  + BN + Dropout(0.1)│
│   Dense(32)  + Dropout(0.05)    │
│   Dense(2)   linear             │
└─────────────────────────────────┘
       ↓  (parallel)
┌─────────────────────────────────┐
│   XGBoost (MultiOutputRegressor)│
│   800 estimators, depth=6       │
└─────────────────────────────────┘
       ↓
  Ensemble Average (0.5 · NN + 0.5 · XGB)
       ↓
Inverse StandardScaler (targets)
       ↓
Safety Clipping + Output Validation
       ↓
Final Output [Frequency kHz, Power W]
```

### Neural Network Design Rationale

| Choice | Reason |
|---|---|
| Polynomial features | Explicitly gives the model T², S², T·S — captures nonlinear physics without needing a deeper network |
| Smaller network (128→64→32) | Right-sized for 5 features; the previous 256→128→64→32 was massively over-parameterized |
| Low Dropout (0.05–0.10) | The dataset is smooth and low-noise; heavy dropout hurts convergence on such problems |
| Huber loss | More robust to the small Gaussian noise in the dataset than MSE |
| Cosine decay LR | Smoother convergence than ReduceLROnPlateau; starts at 5e-3, decays to near-zero |
| L2 regularization (1e-4) | Mild weight penalty prevents any single neuron from dominating |
| Target scaling (scaler_y) | Critical for multi-output balance — without it, the larger-magnitude frequency output dominates the loss and power R² suffers |

### XGBoost Configuration

```python
XGBRegressor(
    n_estimators   = 800,
    max_depth      = 6,
    learning_rate  = 0.05,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    reg_alpha      = 0.1,
    reg_lambda     = 1.0,
)
```

Gradient boosting excels at tabular regression, especially on smooth polynomial relationships. The ensemble combines the NN's smooth interpolation with XGBoost's precise fitting.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 500 (early stopping) |
| Batch size | 64 |
| Early stopping patience | 60 epochs |
| Train / Val / Test split | 70% / 15% / 15% |
| Random seed | 42 (reproducible) |
| Optimizer | Adam + Cosine Decay |
| Loss function | Huber |

---

## Performance

### Expected Results After Training

| Model | Frequency R² | Power R² | Overall R² |
|---|---|---|---|
| Neural Network | 0.997–0.999 | 0.985–0.995 | 0.993+ |
| XGBoost | 0.996–0.999 | 0.990–0.997 | 0.994+ |
| **Ensemble** | **0.998–0.9995** | **0.993–0.998** | **0.996+** |

### Error Magnitudes (Ensemble)

| Output | MAE | RMSE |
|---|---|---|
| Frequency | ~0.05–0.15 kHz | ~0.08–0.20 kHz |
| Power | ~0.015–0.04 W | ~0.025–0.06 W |

### Accuracy Thresholds (enforced by test suite)

- Frequency R² ≥ **0.95**
- Power R² ≥ **0.95**
- Frequency MAE < **0.5 kHz**
- Power MAE < **0.10 W**
- Zero predictions outside physical bounds

---

## Verified Sample Predictions

These outputs were verified against the physics formula with errors well within tolerance:

| Temperature | Salinity | Predicted Freq | Predicted Power | Formula Freq | Formula Power |
|---|---|---|---|---|---|
| 25°C | 0.5 ppt | 56.922 kHz | 0.967 W | 57.045 kHz | 0.941 W |
| 20°C | 35 ppt | 59.071 kHz | 2.253 W | 59.160 kHz | 2.271 W |
| 40°C | 15 ppt | 61.042 kHz | 1.937 W | 60.960 kHz | 1.934 W |
| 5°C | 0.2 ppt | 52.915 kHz | 0.604 W | 53.096 kHz | 0.599 W |
| 80°C | 25 ppt | 67.194 kHz | 3.020 W | 67.280 kHz | 3.056 W |
| 33°C | 48 ppt | 63.005 kHz | 2.907 W | 62.862 kHz | 2.903 W |

Maximum observed errors: **0.18 kHz** (frequency), **0.036 W** (power).

---

## Physics Consistency

The model correctly captures the following real-world acoustic relationships:

**Frequency increases with temperature** — Higher temperature → faster speed of sound → higher optimal frequency needed to maintain acoustic wavelength.

```
T=5°C  → 52.9 kHz
T=25°C → 56.9 kHz
T=40°C → 61.0 kHz
T=80°C → 67.2 kHz
```

**Power increases with salinity** — Higher salinity → denser, more attenuating medium → more transmission power required to overcome acoustic losses.

```
S=0.2 ppt → 0.60 W
S=0.5 ppt → 0.97 W
S=15 ppt  → 1.94 W
S=35 ppt  → 2.25 W
S=48 ppt  → 2.91 W
```

---

## Test Suite

The project includes 40 automated test cases in 8 categories:

| Group | Tests | Description |
|---|---|---|
| TC01–05 | Normal Predictions | Real-world oceanographic scenarios |
| TC06–10 | Boundary Cases | All four domain corners + zero point |
| TC11–15 | Input Validation | Valid/invalid/out-of-range inputs |
| TC16–20 | Physics Consistency | Monotonicity, units, NaN/Inf checks |
| TC21–25 | Dataset Integrity | Columns, size, missing values, plateau bug |
| TC26–30 | Model Artefacts | Files exist, scalers have right dimensions |
| TC31–35 | Accuracy Thresholds | R² ≥ 0.95, MAE gates |
| TC36–40 | Edge / Stress Tests | Determinism, speed, boundary inputs |

```bash
python test_acoustic_ai.py
```

---

## Known Limitations

- **Synthetic dataset** — The model is trained on a physics formula with added noise, not real hydrophone measurements. Against real sensor data, accuracy will be lower and fine-tuning on real samples would be needed.
- **Two-input scope** — Only temperature and salinity are modeled. Real-world performance also depends on depth/pressure, transmission distance, ambient noise, thermoclines, and biological interference.
- **Static formula** — The ground-truth relationship is fixed. If the physical domain changes (e.g., a different transducer type), the dataset must be regenerated.

---

## Dependencies

```
tensorflow >= 2.10
scikit-learn >= 1.0
numpy >= 1.23
pandas >= 1.5
matplotlib >= 3.6
xgboost >= 1.7   (optional but recommended)
```

Install:
```bash
pip install tensorflow scikit-learn numpy pandas matplotlib xgboost
```

---

## File Execution Order

```
1. python train.py              → trains models, saves to new_models/
2. python evaluate.py           → reports test metrics, saves plots
3. python test_acoustic_ai.py   → runs 40 automated tests
4. python predict.py            → live inference
```

---

## License

This project is intended for research and educational use in underwater acoustic communication systems.
