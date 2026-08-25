"""
app/app.py
Streamlit web application for pipeline anomaly detection.
"""
import sys
import os

# Ensure the project root is in sys.path so we can import 'src' modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Correct local import (utils.py is in the same folder as app.py)
from utils import load_model_and_scaler, preprocess_new_data, get_reconstruction_error
from src.detect import get_threshold, detect_anomalies

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Pipeline Leak Detector", layout="wide")
st.title("🔧 Natural Gas Pipeline Anomaly Detection")
st.markdown("Upload your sensor data or use the default model to detect leaks/degradation.")

# -------------------- LOAD DEFAULT MODEL --------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pipeline_model.h5")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.save")
FEATURES = ['pressure', 'temperature', 'flow_rate', 'volume_rate']
TIMESTEPS = 24
THRESHOLD_MULTIPLIER = 3

@st.cache_resource
def load_default_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    else:
        return None, None

model, scaler = load_default_artifacts()

if model is None:
    st.warning("Default model not found. Please run `main.py` first to train and save the model.")
    st.stop()

# -------------------- SIDEBAR: UPLOAD OR USE DEFAULT DATA --------------------
st.sidebar.header("Data Input")
option = st.sidebar.radio("Choose data source:", ("Use default CSV (data/raw/pipeline_data.csv)", "Upload your own CSV"))

if option == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with timestamp and feature columns", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
        st.sidebar.success("File uploaded successfully!")
    else:
        st.sidebar.info("Please upload a CSV file.")
        st.stop()
else:
    default_path = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_data.csv")
    if os.path.exists(default_path):
        df = pd.read_csv(default_path, parse_dates=['timestamp'])
        st.sidebar.success(f"Loaded default data from {default_path}")
    else:
        st.sidebar.error("Default data not found. Run `main.py` to generate it.")
        st.stop()

# -------------------- MAIN AREA: DATA PREVIEW --------------------
st.subheader("📊 Data Preview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head(10))

# -------------------- DETECTION BUTTON --------------------
if st.button("🚨 Run Anomaly Detection"):
    with st.spinner("Processing data..."):
        # Preprocess
        X = preprocess_new_data(df, FEATURES, TIMESTEPS, scaler)
        if len(X) == 0:
            st.error(f"Not enough data. Need at least {TIMESTEPS} rows.")
            st.stop()
        
        # Compute errors
        errors = get_reconstruction_error(model, X)
        threshold = get_threshold(errors, multiplier=THRESHOLD_MULTIPLIER)
        anomalies = detect_anomalies(errors, threshold)
        
        # Prepare results
        n_anomalies = np.sum(anomalies)
        total = len(errors)
        anomaly_percent = 100 * n_anomalies / total if total > 0 else 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sequences", total)
        col2.metric("Anomalies Detected", f"{n_anomalies} ({anomaly_percent:.1f}%)")
        col3.metric("Threshold", f"{threshold:.6f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(errors, label='Reconstruction Error', color='blue', alpha=0.7)
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold')
        anomaly_indices = np.where(anomalies)[0]
        if len(anomaly_indices) > 0:
            ax.scatter(anomaly_indices, errors[anomalies], color='red', s=30, label='Anomalies')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('MSE')
        ax.set_title('Reconstruction Error & Anomalies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Show anomaly timestamps if available
        if len(anomaly_indices) > 0:
            st.subheader("⚠️ Anomaly Events")
            # Map sequence indices back to original timestamps (add TIMESTEPS offset)
            anomaly_times = df['timestamp'].iloc[anomaly_indices + TIMESTEPS]
            anomaly_df = pd.DataFrame({
                'Timestamp': anomaly_times,
                'Error': errors[anomalies]
            })
            st.dataframe(anomaly_df)
        else:
            st.success("✅ No anomalies detected in the current dataset.")
else:
    st.info("Click the button above to run anomaly detection on the loaded data.")