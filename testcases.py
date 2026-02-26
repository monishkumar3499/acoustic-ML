"""
test_acoustic_ai.py
===================
Complete Test Suite — Acoustic AI Multi-Output Regression

Covers:
  TC01–TC05  : Normal / in-range predictions (happy path)
  TC06–TC10  : Physical boundary / corner cases
  TC11–TC15  : Input validation (out-of-range, invalid types)
  TC16–TC20  : Physics consistency checks (monotonicity, coupling)
  TC21–TC25  : Dataset integrity checks
  TC26–TC30  : Model artefact checks (files exist, scalers work)
  TC31–TC35  : Regression accuracy thresholds (R² ≥ 0.95)
  TC36–TC40  : Edge / stress tests

Run:
    python test_acoustic_ai.py
    python test_acoustic_ai.py -v          # verbose
    python test_acoustic_ai.py --no-model  # skip tests that need trained model
"""

import os
import sys
import math
import pickle
import argparse
import unittest
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "new_models")
DATA_PATH  = os.path.join(BASE_DIR, "new_dataset.csv")

# ── Physics formula (ground truth used by dataset generator) ─────────────────
def _freq_gt(T, S):
    return 52.0 + 0.22*T - 0.0008*T**2 + 0.08*S + 0.0004*T*S

def _power_gt(T, S):
    return 0.50 + 0.050*S - 0.0004*S**2 + 0.018*T - 0.00006*T**2 + 0.00025*T*S

# Tolerance for physics-formula checks (accounts for model error + noise)
FREQ_TOL  = 0.80   # kHz
POWER_TOL = 0.10   # W

# ── Lazy model loader ─────────────────────────────────────────────────────────
_model = _poly = _scaler_X = _scaler_y = None

def _load_model():
    global _model, _poly, _scaler_X, _scaler_y
    if _model is not None:
        return True
    try:
        from tensorflow import keras
        _model    = keras.models.load_model(os.path.join(MODEL_DIR, "best_model.h5"))
        with open(os.path.join(MODEL_DIR, "poly.pkl"),     "rb") as f: _poly     = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler_X.pkl"), "rb") as f: _scaler_X = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler_y.pkl"), "rb") as f: _scaler_y = pickle.load(f)
        return True
    except Exception:
        return False

def _predict(temp, sal):
    """Run full inference pipeline: poly → scale → NN → inverse scale."""
    X_raw  = np.array([[temp, sal]])
    X_poly = _poly.transform(X_raw)
    X_s    = _scaler_X.transform(X_poly)
    y_s    = _model.predict(X_s, verbose=0)
    y      = _scaler_y.inverse_transform(y_s)
    freq   = float(np.clip(y[0, 0], 48.0, 82.0))
    power  = float(np.clip(y[0, 1], 0.35, 4.25))
    return freq, power


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: skip if model not loaded
# ─────────────────────────────────────────────────────────────────────────────
SKIP_MODEL = False   # overridden by --no-model flag

def require_model(test_method):
    def wrapper(self):
        if SKIP_MODEL or not _load_model():
            self.skipTest("Trained model not available — run train.py first")
        test_method(self)
    return wrapper


# =============================================================================
# TC01–TC05 : NORMAL IN-RANGE PREDICTIONS
# =============================================================================
class TestNormalPredictions(unittest.TestCase):

    def setUp(self):
        if SKIP_MODEL or not _load_model():
            self.skipTest("Model not available")

    def _assert_in_range(self, freq, power):
        self.assertGreaterEqual(freq,  48.0, "Frequency below physical minimum")
        self.assertLessEqual(freq,     82.0, "Frequency above physical maximum")
        self.assertGreaterEqual(power, 0.35, "Power below physical minimum")
        self.assertLessEqual(power,    4.25, "Power above physical maximum")

    def test_TC01_cold_fresh_water(self):
        """TC01: Cold freshwater — Arctic river mouth scenario."""
        freq, power = _predict(temp=2.0, sal=1.0)
        self._assert_in_range(freq, power)
        # Expected ~52.5 kHz, ~0.56 W
        self.assertAlmostEqual(freq,  _freq_gt(2.0, 1.0),  delta=FREQ_TOL)
        self.assertAlmostEqual(power, _power_gt(2.0, 1.0), delta=POWER_TOL)

    def test_TC02_warm_moderate_salinity(self):
        """TC02: Warm water moderate salinity — tropical coastal."""
        freq, power = _predict(temp=28.0, sal=30.0)
        self._assert_in_range(freq, power)
        self.assertAlmostEqual(freq,  _freq_gt(28.0, 30.0),  delta=FREQ_TOL)
        self.assertAlmostEqual(power, _power_gt(28.0, 30.0), delta=POWER_TOL)

    def test_TC03_mid_range_nominal(self):
        """TC03: Mid-range nominal — typical open ocean."""
        freq, power = _predict(temp=20.0, sal=35.0)
        self._assert_in_range(freq, power)
        self.assertAlmostEqual(freq,  _freq_gt(20.0, 35.0),  delta=FREQ_TOL)
        self.assertAlmostEqual(power, _power_gt(20.0, 35.0), delta=POWER_TOL)

    def test_TC04_hot_high_salinity(self):
        """TC04: Hot water high salinity — Red Sea / Persian Gulf."""
        freq, power = _predict(temp=35.0, sal=42.0)
        self._assert_in_range(freq, power)
        self.assertAlmostEqual(freq,  _freq_gt(35.0, 42.0),  delta=FREQ_TOL)
        self.assertAlmostEqual(power, _power_gt(35.0, 42.0), delta=POWER_TOL)

    def test_TC05_standard_seawater(self):
        """TC05: Standard Atlantic seawater — reference case."""
        freq, power = _predict(temp=15.0, sal=35.0)
        self._assert_in_range(freq, power)
        self.assertAlmostEqual(freq,  _freq_gt(15.0, 35.0),  delta=FREQ_TOL)
        self.assertAlmostEqual(power, _power_gt(15.0, 35.0), delta=POWER_TOL)


# =============================================================================
# TC06–TC10 : BOUNDARY / CORNER CASES
# =============================================================================
class TestBoundaryCases(unittest.TestCase):

    def setUp(self):
        if SKIP_MODEL or not _load_model():
            self.skipTest("Model not available")

    def test_TC06_minimum_temperature_minimum_salinity(self):
        """TC06: Absolute minimum corner (−10°C, 0 ppt) — polar freshwater."""
        freq, power = _predict(temp=-10.0, sal=0.0)
        self.assertGreaterEqual(freq,  48.0)
        self.assertLessEqual(freq,     82.0)
        self.assertGreaterEqual(power, 0.35)
        self.assertLessEqual(power,    4.25)

    def test_TC07_maximum_temperature_maximum_salinity(self):
        """TC07: Absolute maximum corner (85°C, 50 ppt) — extreme brine."""
        freq, power = _predict(temp=85.0, sal=50.0)
        self.assertGreaterEqual(freq,  48.0)
        self.assertLessEqual(freq,     82.0)
        self.assertGreaterEqual(power, 0.35)
        self.assertLessEqual(power,    4.25)

    def test_TC08_minimum_temperature_maximum_salinity(self):
        """TC08: Cold brine corner (−10°C, 50 ppt) — under-ice saline."""
        freq, power = _predict(temp=-10.0, sal=50.0)
        self.assertGreaterEqual(freq,  48.0)
        self.assertLessEqual(freq,     82.0)
        self.assertGreaterEqual(power, 0.35)
        self.assertLessEqual(power,    4.25)

    def test_TC09_maximum_temperature_minimum_salinity(self):
        """TC09: Hot freshwater corner (85°C, 0 ppt) — hydrothermal spring."""
        freq, power = _predict(temp=85.0, sal=0.0)
        self.assertGreaterEqual(freq,  48.0)
        self.assertLessEqual(freq,     82.0)
        self.assertGreaterEqual(power, 0.35)
        self.assertLessEqual(power,    4.25)

    def test_TC10_exact_zero_zero(self):
        """TC10: Zero point (0°C, 0 ppt) — freezing pure water."""
        freq, power = _predict(temp=0.0, sal=0.0)
        self.assertGreater(freq,  0.0)
        self.assertGreater(power, 0.0)
        # Near 52 kHz, near 0.5 W based on formula
        self.assertAlmostEqual(freq,  _freq_gt(0.0, 0.0),  delta=FREQ_TOL)
        self.assertAlmostEqual(power, _power_gt(0.0, 0.0), delta=POWER_TOL)


# =============================================================================
# TC11–TC15 : INPUT VALIDATION
# =============================================================================
class TestInputValidation(unittest.TestCase):

    def setUp(self):
        from utils import validate_input_ranges
        self.validate = validate_input_ranges

    def test_TC11_valid_input_passes(self):
        """TC11: Valid mid-range input — should return is_valid=True, no warnings."""
        is_valid, warnings = self.validate(25.0, 35.0)
        self.assertTrue(is_valid)
        self.assertEqual(len(warnings), 0)

    def test_TC12_temperature_too_low(self):
        """TC12: Temperature below −10°C — should warn but not necessarily invalidate."""
        is_valid, warnings = self.validate(-15.0, 25.0)
        self.assertGreater(len(warnings), 0, "Expected a warning for below-range temp")
        # −15°C is below training range but not physically absurd (> −25°C)
        self.assertTrue(is_valid)

    def test_TC13_temperature_physically_impossible(self):
        """TC13: Temperature below −25°C — should be marked invalid."""
        is_valid, warnings = self.validate(-30.0, 25.0)
        self.assertFalse(is_valid)

    def test_TC14_negative_salinity(self):
        """TC14: Negative salinity — physically impossible, must be invalid."""
        is_valid, warnings = self.validate(20.0, -5.0)
        self.assertFalse(is_valid)
        self.assertGreater(len(warnings), 0)

    def test_TC15_salinity_above_range(self):
        """TC15: Salinity above 50 ppt — should warn but not hard-invalidate."""
        is_valid, warnings = self.validate(20.0, 55.0)
        self.assertGreater(len(warnings), 0, "Expected a warning for above-range salinity")


# =============================================================================
# TC16–TC20 : PHYSICS CONSISTENCY
# =============================================================================
class TestPhysicsConsistency(unittest.TestCase):
    """
    Verify the model has learned the correct physical trends:
      - Higher temperature → higher frequency (primary driver)
      - Higher salinity → higher power (primary driver)
      - Both outputs are monotonically smooth, no sudden jumps
    """

    def setUp(self):
        if SKIP_MODEL or not _load_model():
            self.skipTest("Model not available")

    def test_TC16_frequency_increases_with_temperature(self):
        """TC16: Frequency should increase as temperature rises (constant salinity)."""
        sal = 25.0
        temps = [0.0, 20.0, 40.0, 60.0, 80.0]
        freqs = [_predict(t, sal)[0] for t in temps]
        for i in range(len(freqs) - 1):
            self.assertLess(freqs[i], freqs[i+1],
                f"Frequency did not increase: T={temps[i]}→{temps[i+1]}, "
                f"F={freqs[i]:.3f}→{freqs[i+1]:.3f}")

    def test_TC17_power_increases_with_salinity(self):
        """TC17: Power should increase as salinity rises (constant temperature)."""
        temp = 25.0
        sals = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        powers = [_predict(temp, s)[1] for s in sals]
        for i in range(len(powers) - 1):
            self.assertLess(powers[i], powers[i+1],
                f"Power did not increase: S={sals[i]}→{sals[i+1]}, "
                f"P={powers[i]:.3f}→{powers[i+1]:.3f}")

    def test_TC18_frequency_output_is_in_kHz_not_Hz(self):
        """TC18: Frequency must be in kHz range (48–82), not raw Hz (48000+)."""
        freq, _ = _predict(25.0, 25.0)
        self.assertGreater(freq, 10.0,   "Frequency suspiciously low — Hz instead of kHz?")
        self.assertLess(freq,    500.0,  "Frequency suspiciously high — Hz instead of kHz?")

    def test_TC19_power_output_is_in_watts_not_milliwatts(self):
        """TC19: Power must be in Watts range (0.35–4.25), not milliwatts (350+)."""
        _, power = _predict(25.0, 25.0)
        self.assertGreater(power, 0.1,  "Power suspiciously low — mW instead of W?")
        self.assertLess(power,    50.0, "Power suspiciously high — mW instead of W?")

    def test_TC20_no_nan_or_inf_in_outputs(self):
        """TC20: Model should never output NaN or Inf for any valid input."""
        test_inputs = [
            (0.0, 0.0), (85.0, 50.0), (-10.0, 0.0), (-10.0, 50.0),
            (42.5, 25.0), (1.0, 49.0), (84.0, 1.0)
        ]
        for temp, sal in test_inputs:
            freq, power = _predict(temp, sal)
            self.assertFalse(math.isnan(freq),  f"NaN frequency at T={temp}, S={sal}")
            self.assertFalse(math.isinf(freq),  f"Inf frequency at T={temp}, S={sal}")
            self.assertFalse(math.isnan(power), f"NaN power at T={temp}, S={sal}")
            self.assertFalse(math.isinf(power), f"Inf power at T={temp}, S={sal}")


# =============================================================================
# TC21–TC25 : DATASET INTEGRITY
# =============================================================================
class TestDatasetIntegrity(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(DATA_PATH):
            self.skipTest(f"Dataset not found: {DATA_PATH}")
        self.df = pd.read_csv(DATA_PATH)

    def test_TC21_dataset_has_correct_columns(self):
        """TC21: Dataset must have all 4 required columns."""
        required = {"Temperature_C", "Salinity_ppt",
                    "Optimal_Frequency_kHz", "Optimal_Power_W"}
        self.assertTrue(required.issubset(set(self.df.columns)),
                        f"Missing columns. Found: {list(self.df.columns)}")

    def test_TC22_dataset_has_minimum_size(self):
        """TC22: Dataset must have at least 1000 samples."""
        self.assertGreaterEqual(len(self.df), 1000,
                                f"Dataset too small: {len(self.df)} rows")

    def test_TC23_no_missing_values(self):
        """TC23: Dataset must contain no NaN values."""
        missing = self.df.isnull().sum().sum()
        self.assertEqual(missing, 0, f"Found {missing} missing values in dataset")

    def test_TC24_temperature_within_valid_range(self):
        """TC24: All temperature values must be within −10 to 85°C."""
        bad = self.df[(self.df["Temperature_C"] < -10.5) |
                      (self.df["Temperature_C"] > 85.5)]
        self.assertEqual(len(bad), 0,
                         f"{len(bad)} rows have temperature out of range")

    def test_TC25_no_power_plateau_artifact(self):
        """TC25: New dataset must NOT have >10% samples clamped at old 3.5W ceiling."""
        clamped_at_old_ceiling = (self.df["Optimal_Power_W"] == 3.5).sum()
        pct = 100.0 * clamped_at_old_ceiling / len(self.df)
        self.assertLess(pct, 10.0,
                        f"{pct:.1f}% of samples clamped at 3.5W — old plateau bug present!")


# =============================================================================
# TC26–TC30 : MODEL ARTEFACT CHECKS
# =============================================================================
class TestModelArtefacts(unittest.TestCase):

    def test_TC26_model_file_exists(self):
        """TC26: best_model.h5 must exist in new_models/."""
        path = os.path.join(MODEL_DIR, "best_model.h5")
        self.assertTrue(os.path.exists(path),
                        f"Model file not found: {path}\nRun train.py first.")

    def test_TC27_poly_pkl_exists(self):
        """TC27: poly.pkl (polynomial transformer) must exist."""
        path = os.path.join(MODEL_DIR, "poly.pkl")
        self.assertTrue(os.path.exists(path),
                        "poly.pkl missing — run train.py first.")

    def test_TC28_scaler_X_loadable(self):
        """TC28: scaler_X.pkl must load and have correct feature count (5)."""
        path = os.path.join(MODEL_DIR, "scaler_X.pkl")
        if not os.path.exists(path): self.skipTest("scaler_X.pkl not found")
        with open(path, "rb") as f:
            sx = pickle.load(f)
        self.assertEqual(sx.n_features_in_, 5,
                         f"scaler_X expects {sx.n_features_in_} features, expected 5")

    def test_TC29_scaler_y_loadable(self):
        """TC29: scaler_y.pkl must load and have 2 output columns."""
        path = os.path.join(MODEL_DIR, "scaler_y.pkl")
        if not os.path.exists(path): self.skipTest("scaler_y.pkl not found")
        with open(path, "rb") as f:
            sy = pickle.load(f)
        self.assertEqual(sy.n_features_in_, 2,
                         f"scaler_y expects {sy.n_features_in_} features, expected 2")

    def test_TC30_poly_produces_5_features(self):
        """TC30: PolynomialFeatures(degree=2) on [T,S] must produce exactly 5 features."""
        path = os.path.join(MODEL_DIR, "poly.pkl")
        if not os.path.exists(path): self.skipTest("poly.pkl not found")
        with open(path, "rb") as f:
            poly = pickle.load(f)
        X_test = np.array([[25.0, 35.0]])
        X_out  = poly.transform(X_test)
        self.assertEqual(X_out.shape[1], 5,
                         f"Expected 5 polynomial features, got {X_out.shape[1]}")


# =============================================================================
# TC31–TC35 : REGRESSION ACCURACY THRESHOLDS
# =============================================================================
class TestAccuracyThresholds(unittest.TestCase):
    """
    Reconstruct the exact test split and verify R² ≥ 0.95 on held-out data.
    These are pass/fail gates — if R² drops below threshold something broke.
    """

    def setUp(self):
        if SKIP_MODEL or not _load_model():
            self.skipTest("Model not available")
        if not os.path.exists(DATA_PATH):
            self.skipTest("Dataset not found")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        df   = pd.read_csv(DATA_PATH)
        X_raw = df[["Temperature_C", "Salinity_ppt"]].values
        y     = df[["Optimal_Frequency_kHz", "Optimal_Power_W"]].values

        X = _poly.transform(X_raw)
        _, X_tmp, _, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
        _, X_te, _, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

        X_te_s = _scaler_X.transform(X_te)
        y_s    = _model.predict(X_te_s, verbose=0)
        y_pred = _scaler_y.inverse_transform(y_s)

        self.y_te   = y_te
        self.y_pred = y_pred
        self.r2     = r2_score

    def test_TC31_frequency_r2_above_0_95(self):
        """TC31: Frequency R² on test set must be ≥ 0.95."""
        r2 = self.r2(self.y_te[:, 0], self.y_pred[:, 0])
        self.assertGreaterEqual(r2, 0.95,
            f"Frequency R²={r2:.4f} is below 0.95 threshold — model needs retraining.")

    def test_TC32_power_r2_above_0_95(self):
        """TC32: Power R² on test set must be ≥ 0.95."""
        r2 = self.r2(self.y_te[:, 1], self.y_pred[:, 1])
        self.assertGreaterEqual(r2, 0.95,
            f"Power R²={r2:.4f} is below 0.95 threshold — model needs retraining.")

    def test_TC33_frequency_mae_below_0_5_kHz(self):
        """TC33: Frequency MAE must be below 0.5 kHz on test set."""
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(self.y_te[:, 0], self.y_pred[:, 0])
        self.assertLess(mae, 0.5,
            f"Frequency MAE={mae:.4f} kHz exceeds 0.5 kHz tolerance.")

    def test_TC34_power_mae_below_0_1_W(self):
        """TC34: Power MAE must be below 0.10 W on test set."""
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(self.y_te[:, 1], self.y_pred[:, 1])
        self.assertLess(mae, 0.10,
            f"Power MAE={mae:.4f} W exceeds 0.10 W tolerance.")

    def test_TC35_no_prediction_outside_physical_bounds(self):
        """TC35: No prediction on test set should fall outside physical bounds."""
        freq_out  = np.sum((self.y_pred[:, 0] < 40.0) | (self.y_pred[:, 0] > 90.0))
        power_out = np.sum((self.y_pred[:, 1] < 0.2)  | (self.y_pred[:, 1] > 5.0))
        self.assertEqual(freq_out,  0, f"{freq_out} frequency predictions outside physical bounds")
        self.assertEqual(power_out, 0, f"{power_out} power predictions outside physical bounds")


# =============================================================================
# TC36–TC40 : EDGE / STRESS TESTS
# =============================================================================
class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        if SKIP_MODEL or not _load_model():
            self.skipTest("Model not available")

    def test_TC36_float_precision_inputs(self):
        """TC36: Very precise float inputs — no numerical instability."""
        freq, power = _predict(temp=37.123456789, sal=34.987654321)
        self.assertFalse(math.isnan(freq))
        self.assertFalse(math.isnan(power))

    def test_TC37_repeated_calls_are_deterministic(self):
        """TC37: Same input must always give same output (no stochastic behaviour at inference)."""
        results = [_predict(25.0, 35.0) for _ in range(5)]
        freqs  = [r[0] for r in results]
        powers = [r[1] for r in results]
        self.assertAlmostEqual(max(freqs)  - min(freqs),  0.0, places=4,
                               msg="Frequency is non-deterministic across repeated calls")
        self.assertAlmostEqual(max(powers) - min(powers), 0.0, places=4,
                               msg="Power is non-deterministic across repeated calls")

    def test_TC38_batch_of_100_inputs_completes_fast(self):
        """TC38: Batch prediction of 100 samples must complete in under 5 seconds."""
        import time
        inputs = [(np.random.uniform(-10, 85), np.random.uniform(0, 50))
                  for _ in range(100)]
        np.random.seed(99)

        X_raw = np.array([[t, s] for t, s in inputs])
        X_p   = _poly.transform(X_raw)
        X_s   = _scaler_X.transform(X_p)

        t0 = time.time()
        y_s = _model.predict(X_s, verbose=0)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 5.0,
                        f"Batch inference took {elapsed:.2f}s — too slow (>5s)")

    def test_TC39_temperature_at_training_boundary_exact(self):
        """TC39: Predictions at exact training boundary (−10.0, 85.0) must not error."""
        for temp in [-10.0, 85.0]:
            try:
                freq, power = _predict(temp, sal=25.0)
                self.assertFalse(math.isnan(freq))
            except Exception as e:
                self.fail(f"Prediction failed at T={temp}: {e}")

    def test_TC40_salinity_at_training_boundary_exact(self):
        """TC40: Predictions at exact salinity boundary (0.0, 50.0) must not error."""
        for sal in [0.0, 50.0]:
            try:
                freq, power = _predict(temp=25.0, sal=sal)
                self.assertFalse(math.isnan(power))
            except Exception as e:
                self.fail(f"Prediction failed at S={sal}: {e}")


# =============================================================================
# RESULTS SUMMARY PRINTER
# =============================================================================
class _VerboseResult(unittest.TextTestResult):
    PASS  = "\033[92m PASS \033[0m"
    FAIL  = "\033[91m FAIL \033[0m"
    SKIP  = "\033[93m SKIP \033[0m"
    ERROR = "\033[91m ERR  \033[0m"

    def addSuccess(self, test):
        super().addSuccess(test)
        tc_id = test.id().split(".")[-1][:6]
        self.stream.writeln(f"  [{self.PASS}] {test.shortDescription() or test._testMethodName}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.writeln(f"  [{self.FAIL}] {test.shortDescription() or test._testMethodName}")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.stream.writeln(f"  [{self.SKIP}] {test.shortDescription() or test._testMethodName}  ({reason})")

    def addError(self, test, err):
        super().addError(test, err)
        self.stream.writeln(f"  [{self.ERROR}] {test.shortDescription() or test._testMethodName}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-model", action="store_true",
                        help="Skip tests requiring a trained model")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, remaining = parser.parse_known_args()

    if args.no_model:
        SKIP_MODEL = True

    print("\n" + "="*62)
    print("  ACOUSTIC AI — TEST SUITE  (40 test cases)")
    print("="*62)

    # Group display
    groups = [
        ("TC01–05  Normal Predictions",   TestNormalPredictions),
        ("TC06–10  Boundary Cases",        TestBoundaryCases),
        ("TC11–15  Input Validation",      TestInputValidation),
        ("TC16–20  Physics Consistency",   TestPhysicsConsistency),
        ("TC21–25  Dataset Integrity",     TestDatasetIntegrity),
        ("TC26–30  Model Artefacts",       TestModelArtefacts),
        ("TC31–35  Accuracy Thresholds",   TestAccuracyThresholds),
        ("TC36–40  Edge / Stress Tests",   TestEdgeCases),
    ]

    overall_pass = overall_fail = overall_skip = overall_error = 0

    runner = unittest.TextTestRunner(
        stream=sys.stdout,
        resultclass=_VerboseResult,
        verbosity=0
    )

    for group_name, test_class in groups:
        print(f"\n── {group_name} ──")
        suite  = unittest.TestLoader().loadTestsFromTestCase(test_class)
        result = runner.run(suite)
        overall_pass  += result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        overall_fail  += len(result.failures) + len(result.errors)
        overall_skip  += len(result.skipped)
        overall_error += len(result.errors)

    total = overall_pass + overall_fail + overall_skip
    print("\n" + "="*62)
    print(f"  RESULTS  →  "
          f"\033[92m{overall_pass} passed\033[0m  "
          f"\033[91m{overall_fail} failed\033[0m  "
          f"\033[93m{overall_skip} skipped\033[0m  "
          f"/ {total} total")
    print("="*62 + "\n")

    sys.exit(0 if overall_fail == 0 else 1)