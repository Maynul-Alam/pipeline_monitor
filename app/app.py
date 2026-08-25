"""
app/app.py
Streamlit web application for pipeline anomaly detection – final cloud-ready version.
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- PATH SETUP --------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import load_model_and_scaler, preprocess_new_data, get_reconstruction_error
from src.detect import get_threshold, detect_anomalies

# -------------------- CONFIGURATION --------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pipeline_model.h5")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.save")
FEATURES = ['pressure', 'temperature', 'flow_rate', 'volume_rate']
TIMESTEPS = 24
THRESHOLD_MULTIPLIER = 3

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Pipeline Leak Detector", layout="wide")
st.title("🔧 Natural Gas Pipeline Anomaly Detection")
st.markdown("Upload your sensor data or use the default model to detect leaks/degradation.")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_default_artifacts():
    """Load model and scaler; raise exceptions if anything fails."""
    try:
        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Failed to load model or scaler: {e}")

# -------------------- GENERATE SYNTHETIC DATA (in memory) --------------------
def generate_synthetic_data():
    """Generate synthetic pipeline data with anomalies, returns a DataFrame."""
    np.random.seed(42)
    n_hours = 5000
    t = np.linspace(0, 20 * np.pi, n_hours)
    pressure = 50 + 10 * np.sin(t) + np.random.normal(0, 1, n_hours)
    temperature = 25 + 5 * np.cos(t * 0.8) + np.random.normal(0, 0.5, n_hours)
    flow_rate = 100 + 20 * np.sin(t * 1.2) + np.random.normal(0, 2, n_hours)
    volume_rate = 80 + 15 * np.cos(t * 0.9) + np.random.normal(0, 1.5, n_hours)
    # Inject anomalies
    anomaly_indices = np.random.choice(n_hours, size=30, replace=False)
    for idx in anomaly_indices:
        pressure[idx] += np.random.uniform(15, 25)
        flow_rate[idx] -= np.random.uniform(20, 40)
    timestamps = pd.date_range(start='2020-01-01', periods=n_hours, freq='h')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'pressure': pressure,
        'temperature': temperature,
        'flow_rate': flow_rate,
        'volume_rate': volume_rate
    })
    return df

def load_data():
    """Load data from CSV if exists, otherwise generate synthetic data."""
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "pipeline_data.csv")
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path, parse_dates=['timestamp'])
            return df
        except Exception as e:
            st.warning(f"Could not read CSV: {e}. Generating synthetic data instead.")
            return generate_synthetic_data()
    else:
        st.info("No default data found. Generating synthetic dataset for demo...")
        return generate_synthetic_data()

# -------------------- MAIN APP FLOW --------------------
def main():
    # 1. Check if model exists first
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at: {MODEL_PATH}")
        st.info("Please run `python main.py` locally to train the model, then commit the `models/` folder to GitHub.")
        st.stop()

    # 2. Load model
    try:
        model, scaler = load_default_artifacts()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

    # 3. Data input
    st.sidebar.header("Data Input")
    option = st.sidebar.radio("Choose data source:", ("Use default data", "Upload your own CSV"))

    if option == "Upload your own CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV with timestamp and feature columns", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
                st.sidebar.success("File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                st.stop()
        else:
            st.sidebar.info("Please upload a CSV file.")
            st.stop()
    else:
        df = load_data()
        st.sidebar.success(f"Data loaded: {df.shape[0]} rows")

    # 4. Data preview
    st.subheader("📊 Data Preview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head(10))

    # 5. Detection button
    if st.button("🚨 Run Anomaly Detection"):
        with st.spinner("Processing data and detecting anomalies..."):
            try:
                X = preprocess_new_data(df, FEATURES, TIMESTEPS, scaler)
                if len(X) == 0:
                    st.error(f"Not enough data. Need at least {TIMESTEPS} rows.")
                    st.stop()
                
                errors = get_reconstruction_error(model, X)
                threshold = get_threshold(errors, multiplier=THRESHOLD_MULTIPLIER)
                anomalies = detect_anomalies(errors, threshold)
                
                n_anomalies = np.sum(anomalies)
                total = len(errors)
                anomaly_percent = 100 * n_anomalies / total if total > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Sequences", total)
                col2.metric("Anomalies Detected", f"{n_anomalies} ({anomaly_percent:.1f}%)")
                col3.metric("Threshold", f"{threshold:.6f}")
                
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
                
                if len(anomaly_indices) > 0:
                    st.subheader("⚠️ Anomaly Events")
                    anomaly_times = df['timestamp'].iloc[anomaly_indices + TIMESTEPS]
                    anomaly_df = pd.DataFrame({
                        'Timestamp': anomaly_times,
                        'Error': errors[anomalies]
                    })
                    st.dataframe(anomaly_df)
                else:
                    st.success("✅ No anomalies detected in the current dataset.")
            except Exception as e:
                st.exception(e)

if __name__ == "__main__":
    main()