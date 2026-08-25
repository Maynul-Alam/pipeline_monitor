"""
main.py
End-to-end pipeline: data generation (if needed), preprocessing, training, detection.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import our modules
from src.data_preprocessing import get_preprocessed_data
from src.train import train_model, save_model
from src.detect import run_detection_pipeline

# -------------------- CONFIGURATION --------------------
DATA_RAW_PATH = "data/raw/pipeline_data.csv"
MODEL_PATH = "models/pipeline_model.h5"
SCALER_PATH = "models/scaler.save"
PLOT_PATH = "logs/anomaly_plot.png"

FEATURES = ['pressure', 'temperature', 'flow_rate', 'volume_rate']
TIMESTEPS = 24          # window size (e.g., 24 hours)
EPOCHS = 50
BATCH_SIZE = 64
LSTM_UNITS = 64
THRESHOLD_MULTIPLIER = 3
TEST_SPLIT = 0.2        # fraction for validation

# -------------------- DUMMY DATA GENERATION (if file not found) --------------------
if not os.path.exists(DATA_RAW_PATH):
    print("Real data not found. Generating synthetic pipeline data with anomalies...")
    os.makedirs(os.path.dirname(DATA_RAW_PATH), exist_ok=True)
    
    np.random.seed(42)
    n_hours = 5000
    timestamps = pd.date_range(start='2020-01-01', periods=n_hours, freq='H')
    
    # Normal behaviour: sinusoidal with noise
    t = np.linspace(0, 20 * np.pi, n_hours)
    pressure = 50 + 10 * np.sin(t) + np.random.normal(0, 1, n_hours)
    temperature = 25 + 5 * np.cos(t * 0.8) + np.random.normal(0, 0.5, n_hours)
    flow_rate = 100 + 20 * np.sin(t * 1.2) + np.random.normal(0, 2, n_hours)
    volume_rate = 80 + 15 * np.cos(t * 0.9) + np.random.normal(0, 1.5, n_hours)
    
    # Inject some anomalies (leak-like events)
    anomaly_indices = np.random.choice(n_hours, size=30, replace=False)
    for idx in anomaly_indices:
        pressure[idx] += np.random.uniform(15, 25)   # sudden pressure drop
        flow_rate[idx] -= np.random.uniform(20, 40)  # flow drop
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'pressure': pressure,
        'temperature': temperature,
        'flow_rate': flow_rate,
        'volume_rate': volume_rate
    })
    df.to_csv(DATA_RAW_PATH, index=False)
    print(f"Synthetic data saved to {DATA_RAW_PATH}")

# -------------------- 1. PREPROCESS --------------------
print("Loading and preprocessing data...")
X, scaler = get_preprocessed_data(
    filepath=DATA_RAW_PATH,
    feature_columns=FEATURES,
    timesteps=TIMESTEPS,
    scaler_path=SCALER_PATH,
    fit_scaler=True
)
print(f"Sequences shape: {X.shape}")

# Split into train / validation (sequential split, no shuffle)
split_idx = int((1 - TEST_SPLIT) * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# -------------------- 2. TRAIN --------------------
print("Training the LSTM Autoencoder...")
model, history = train_model(
    X_train, X_val,
    timesteps=TIMESTEPS,
    n_features=len(FEATURES),
    lstm_units=LSTM_UNITS,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# -------------------- 3. SAVE MODEL & SCALER --------------------
save_model(model, MODEL_PATH, scaler, SCALER_PATH)

# -------------------- 4. DETECT ANOMALIES (on full dataset) --------------------
print("Running anomaly detection on full dataset...")
errors, threshold, anomalies = run_detection_pipeline(
    model, X,
    threshold_multiplier=THRESHOLD_MULTIPLIER,
    plot_path=PLOT_PATH
)

# Print summary
n_anomalies = np.sum(anomalies)
total = len(anomalies)
print(f"\n--- Summary ---")
print(f"Total sequences evaluated: {total}")
print(f"Anomalies detected: {n_anomalies} ({100 * n_anomalies / total:.2f}%)")
print(f"Anomaly threshold: {threshold:.6f}")
print(f"Plot saved to: {PLOT_PATH}")
print("Pipeline completed successfully!")