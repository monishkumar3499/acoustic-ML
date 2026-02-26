Acoustic AI — Multi-Output Regression (Latest Model)
🔷 Project Overview

This project implements a physics-aware multi-output neural network that predicts optimal acoustic operating parameters from environmental conditions.

🎯 Objective

Given environmental inputs:

Temperature (°C)

Salinity (ppt)

Predict:

Optimal Frequency (kHz)

Optimal Power (W)

The system is designed to be:

✅ Numerically stable

✅ Physically consistent

✅ Research-grade

✅ Deployment-ready

🧠 End-to-End Pipeline
User Input → Input Validation → Feature Scaling → Neural Network
           → Inverse Scaling → Safety Guards → Final Output
📂 Project Structure (Latest)
acoustic_ai/
│
├── new_dataset.csv
├── train.py
├── evaluate.py
├── predict.py
├── utils.py
├── new_models/
│   ├── best_model.h5
│   ├── scaler.pkl
│   ├── scaler_y.pkl
│   └── training_history.png
🔬 Dataset Summary (Updated)
Input Features
Feature	Unit	Range
Temperature_C	°C	−10 → 85
Salinity_ppt	ppt	0 → 50
Target Variables
Target	Unit	Typical Range
Optimal_Frequency_kHz	kHz	~50–75
Optimal_Power_W	W	~0.5–3.2
⚙️ Data Processing Pipeline
Train / Validation / Test Split

Train: 70%

Validation: 15%

Test: 15%

Purpose

Prevent overfitting

Enable unbiased evaluation

Support hyperparameter tuning

Feature Scaling
Input scaling

StandardScaler is applied:

𝑋
𝑠
𝑐
𝑎
𝑙
𝑒
𝑑
=
(
𝑋
−
𝜇
)
/
𝜎
X
scaled
	​

=(X−μ)/σ
⭐ Target scaling (critical improvement)

Targets are also standardized:

𝑦
𝑠
𝑐
𝑎
𝑙
𝑒
𝑑
=
(
𝑦
−
𝜇
𝑦
)
/
𝜎
𝑦
y
scaled
	​

=(y−μ
y
	​

)/σ
y
	​


Why this matters

stabilizes multi-output training

balances frequency vs power loss

prevents output explosion

improves power R²

🧩 Neural Network Architecture
Model Type

Multi-Output Feedforward Neural Network

The model simultaneously predicts frequency and power from shared environmental features.

🏗 Architecture
Input (2)
 ↓
Dense(256) + ReLU
BatchNorm + Dropout(0.30)
 ↓
Dense(128) + ReLU
BatchNorm + Dropout(0.25)
 ↓
Dense(64) + ReLU
BatchNorm + Dropout(0.20)
 ↓
Dense(32) + ReLU
BatchNorm + Dropout(0.15)
 ↓
Dense(2) → [Frequency, Power]
🔍 Design Rationale

Dense layers

capture nonlinear thermo-acoustic mapping

enable shared feature learning

ReLU

avoids vanishing gradients

computationally efficient

Batch Normalization

stabilizes training

improves convergence

improves generalization

Dropout

reduces overfitting

improves robustness

Linear output

required for regression

⚡ Training Strategy
Optimizer

Adam (lr = 0.001)

Chosen for:

fast convergence

adaptive learning

strong tabular performance

Loss Function

Mean Squared Error (MSE)

Used because:

smooth gradients

penalizes large deviations

standard for regression

Regularization
Early Stopping

patience = 50

monitor = val_loss

Prevents overfitting.

ReduceLROnPlateau

factor = 0.5

patience = 20

Improves late-stage convergence.

Model Checkpoint

Best weights saved to:

new_models/best_model.h5
📊 Evaluation Metrics

The system reports research-grade metrics:

R²

MAE

RMSE

MAPE

Expected Performance (Latest Model)

Based on current runs:

Frequency R² ≈ 0.98–0.995

Power R² ≈ 0.92–0.97

Overall R² ≈ 0.95+

Interpretation

Frequency prediction: excellent

Power prediction: strong and stable

Model generalization: high

📈 Generated Visual Diagnostics

The evaluation pipeline produces publication-ready plots:

1️⃣ Actual vs Predicted

Checks overall accuracy.

2️⃣ Residual Plot

Detects bias and heteroscedasticity.

3️⃣ Residual Distribution

Evaluates error normality.

🧪 Inference Pipeline
Runtime Flow
User Input
 ↓
Range Validation
 ↓
Input Scaling
 ↓
Model Prediction (scaled space)
 ↓
Inverse Target Scaling
 ↓
Safety Clipping
 ↓
Final Output
Safety Mechanisms

The system includes:

input range validation

physical sanity checks

output range guards

clipping to safe envelope

This makes the model deployment-safe.

🔬 Learned Physical Behavior

The trained model captures expected thermo-acoustic trends:

Temperature → dominant driver of frequency

Salinity → dominant driver of power

Smooth monotonic behavior

Mild nonlinear coupling

This indicates physics-consistent learning.

🚀 How to Run
Train
python train.py
Evaluate
python evaluate.py
Predict
python predict.py
🔮 Next Research Upgrades

To evolve into full acoustic optimization:

add distance input

physics-based SNR model

ΔSNR comparison vs baseline

multi-head architecture

uncertainty estimation

🏁 Conclusion

The latest Acoustic AI model is a stable, physics-aware multi-output regression system that demonstrates high predictive accuracy and strong generalization across a wide environmental range. With proper scaling, validation, and safety guards, the pipeline is now suitable for research publication and further extension toward full acoustic communication optimization.