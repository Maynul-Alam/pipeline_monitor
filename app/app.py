"""
app/app.py
Interactive Pipeline Simulation with anomaly injection.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Pipeline Simulator", layout="wide")
st.title("🔧 Interactive Pipeline Simulator")
st.markdown("Adjust the parameters below to simulate pipeline behaviour and see anomalies in real time.")

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("⚙️ Control Parameters")

# Base values
pressure_base = st.sidebar.slider("Base Pressure (psi)", 30, 80, 50, 1)
temperature_base = st.sidebar.slider("Base Temperature (°C)", 10, 40, 25, 1)
flow_base = st.sidebar.slider("Base Flow Rate (m³/h)", 50, 150, 100, 5)
volume_base = st.sidebar.slider("Base Volume Rate (m³)", 40, 120, 80, 5)

# Noise level
noise = st.sidebar.slider("Noise Level", 0.0, 2.0, 1.0, 0.1)

# Anomaly injection
st.sidebar.subheader("💥 Inject Anomaly")
if st.sidebar.button("Drop Pressure (Leak)"):
    st.session_state.anomaly_pressure_drop = 20
    st.session_state.anomaly_flow_drop = 30
    st.session_state.anomaly_time = datetime.now()
else:
    # gradually reduce anomaly effect
    if 'anomaly_pressure_drop' in st.session_state:
        st.session_state.anomaly_pressure_drop *= 0.98
        if st.session_state.anomaly_pressure_drop < 0.5:
            st.session_state.anomaly_pressure_drop = 0
            st.session_state.anomaly_flow_drop = 0

# -------------------- GENERATE SIMULATION DATA --------------------
def generate_data():
    # Time settings
    duration_seconds = 60  # simulate last 60 seconds
    n_points = 300
    t = np.linspace(0, duration_seconds, n_points)
    
    # Base signals with sinusoidal variations
    pressure = pressure_base + 5 * np.sin(2 * np.pi * t / 30) + noise * np.random.randn(n_points)
    temperature = temperature_base + 3 * np.cos(2 * np.pi * t / 25) + noise * 0.5 * np.random.randn(n_points)
    flow = flow_base + 10 * np.sin(2 * np.pi * t / 20) + noise * 2 * np.random.randn(n_points)
    volume = volume_base + 8 * np.cos(2 * np.pi * t / 35) + noise * 1.5 * np.random.randn(n_points)
    
    # Apply anomalies if injected
    if 'anomaly_pressure_drop' in st.session_state and st.session_state.anomaly_pressure_drop > 0.5:
        # Find the index where anomaly starts (near the end)
        anomaly_idx = int(0.7 * n_points)
        pressure[anomaly_idx:] -= st.session_state.anomaly_pressure_drop * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 5))
        flow[anomaly_idx:] -= st.session_state.anomaly_flow_drop * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 5))
    
    return t, pressure, temperature, flow, volume

# -------------------- PLOT --------------------
t, pressure, temperature, flow, volume = generate_data()

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(t, pressure, color='blue', linewidth=2)
axes[0].set_ylabel('Pressure (psi)')
axes[0].grid(True, alpha=0.3)
if 'anomaly_pressure_drop' in st.session_state and st.session_state.anomaly_pressure_drop > 0.5:
    axes[0].axvspan(t[int(0.7*len(t))], t[-1], alpha=0.2, color='red', label='Anomaly')
    axes[0].legend()

axes[1].plot(t, temperature, color='orange', linewidth=2)
axes[1].set_ylabel('Temperature (°C)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, flow, color='green', linewidth=2)
axes[2].set_ylabel('Flow Rate (m³/h)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, volume, color='purple', linewidth=2)
axes[3].set_ylabel('Volume Rate (m³)')
axes[3].set_xlabel('Time (seconds)')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# -------------------- ANOMALY DETECTION (Simple Threshold) --------------------
st.subheader("⚠️ Anomaly Detection (Threshold Based)")
# Simple rule: if pressure drops below (base - 10) and flow drops below (base - 20)
last_pressure = pressure[-1]
last_flow = flow[-1]
pressure_drop = pressure_base - last_pressure
flow_drop = flow_base - last_flow

if pressure_drop > 10 and flow_drop > 20:
    st.error("🚨 **ANOMALY DETECTED: Possible leak!** (Pressure and flow have dropped significantly)")
elif pressure_drop > 8:
    st.warning("⚠️ **Caution:** Pressure is low. Monitor the situation.")
else:
    st.success("✅ System operating within normal parameters.")

# -------------------- LIVE DATA TABLE --------------------
with st.expander("📊 View Raw Data"):
    df = pd.DataFrame({
        'Time (s)': t.round(2),
        'Pressure (psi)': pressure.round(2),
        'Temperature (°C)': temperature.round(2),
        'Flow Rate (m³/h)': flow.round(2),
        'Volume Rate (m³)': volume.round(2)
    })
    st.dataframe(df)

# -------------------- AUTO-REFRESH --------------------
st.sidebar.markdown("---")
st.sidebar.info("💡 Click 'Drop Pressure (Leak)' to simulate a leak. The app updates in real time.")