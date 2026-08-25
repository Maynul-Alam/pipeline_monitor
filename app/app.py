"""
app/app.py
Advanced Pipeline Simulator with Automatic Random Anomalies
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import random

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Pipeline Simulator", layout="wide")
st.title("🛢️ Advanced Pipeline Simulator")
st.markdown("Simulate real pipeline behavior with automatic random anomalies to test detection.")

# -------------------- SESSION STATE INIT --------------------
if 'anomaly_active' not in st.session_state:
    st.session_state.anomaly_active = False
    st.session_state.anomaly_type = None
    st.session_state.anomaly_start_time = None
    st.session_state.history_pressure = []
    st.session_state.history_flow = []
    st.session_state.history_temperature = []
    st.session_state.history_volume = []
    st.session_state.history_time = []
    st.session_state.anomaly_detected = False
    st.session_state.detection_log = []
    st.session_state.last_random_anomaly_time = datetime.now()
    st.session_state.random_anomaly_enabled = False

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("⚙️ Pipeline Settings")

# Operating conditions
col1, col2 = st.sidebar.columns(2)
with col1:
    pressure_setpoint = st.slider("Pressure Setpoint (psi)", 40, 80, 60, 1)
    temperature_setpoint = st.slider("Temperature Setpoint (°C)", 15, 40, 25, 1)
with col2:
    flow_setpoint = st.slider("Flow Setpoint (m³/h)", 80, 150, 110, 5)
    volume_setpoint = st.slider("Volume Setpoint (m³)", 50, 120, 80, 5)

# Pipeline dynamics
st.sidebar.subheader("🌊 System Dynamics")
valve_position = st.slider("Valve Opening (%)", 0, 100, 80, 5)
pump_speed = st.slider("Pump Speed (%)", 50, 100, 85, 5)
ambient_temp = st.slider("Ambient Temperature (°C)", -5, 35, 20, 1)
noise_level = st.slider("Sensor Noise Level", 0.0, 3.0, 0.8, 0.1)

# -------------------- RANDOM ANOMALY GENERATOR --------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Automatic Anomaly Generator")
random_anomaly_enabled = st.sidebar.checkbox("Enable random anomalies", value=False)
st.session_state.random_anomaly_enabled = random_anomaly_enabled

if random_anomaly_enabled:
    anomaly_interval = st.sidebar.slider("Average interval (seconds)", 10, 60, 25, 5)
    st.sidebar.info(f"Random anomaly will be injected every ~{anomaly_interval}s")
else:
    st.sidebar.info("Toggle on to automatically test detection")

# -------------------- MANUAL ANOMALY INJECTION --------------------
st.sidebar.markdown("---")
st.sidebar.subheader("💥 Manual Anomaly Injection")

anomaly_type = st.sidebar.selectbox(
    "Select Anomaly Type",
    ["None", "Leak (Pressure Drop)", "Blockage (Flow Drop)", "Pump Failure (Pressure/Flow Drop)", 
     "Sensor Drift", "Pressure Surge", "Temperature Spike"]
)

if st.sidebar.button("Inject Anomaly", type="primary"):
    if anomaly_type != "None":
        st.session_state.anomaly_active = True
        st.session_state.anomaly_type = anomaly_type
        st.session_state.anomaly_start_time = datetime.now()
        st.session_state.anomaly_detected = False
        st.sidebar.success(f"✅ {anomaly_type} injected!")

if st.sidebar.button("Reset System"):
    st.session_state.anomaly_active = False
    st.session_state.anomaly_type = None
    st.session_state.anomaly_start_time = None
    st.session_state.history_pressure = []
    st.session_state.history_flow = []
    st.session_state.history_temperature = []
    st.session_state.history_volume = []
    st.session_state.history_time = []
    st.session_state.anomaly_detected = False
    st.session_state.detection_log = []
    st.session_state.last_random_anomaly_time = datetime.now()
    st.sidebar.success("🔄 System reset!")

# -------------------- RANDOM ANOMALY LOGIC --------------------
def check_and_inject_random():
    """Inject random anomaly if enabled and time interval passed."""
    if not st.session_state.random_anomaly_enabled:
        return
    
    now = datetime.now()
    elapsed = (now - st.session_state.last_random_anomaly_time).total_seconds()
    interval = st.session_state.get('anomaly_interval', 25)  # default 25s
    
    if elapsed > interval:
        # Choose random anomaly type
        anomaly_types = ["Leak (Pressure Drop)", "Blockage (Flow Drop)", 
                        "Pump Failure (Pressure/Flow Drop)", "Sensor Drift", 
                        "Pressure Surge", "Temperature Spike"]
        # Weight towards leak and blockage for realism
        weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.1]
        chosen = random.choices(anomaly_types, weights=weights)[0]
        
        # Inject
        st.session_state.anomaly_active = True
        st.session_state.anomaly_type = chosen
        st.session_state.anomaly_start_time = now
        st.session_state.anomaly_detected = False
        st.session_state.last_random_anomaly_time = now
        
        # Log it
        st.session_state.detection_log.append(f"{now.strftime('%H:%M:%S')} - 🤖 Random anomaly injected: {chosen}")

# -------------------- SIMULATION ENGINE --------------------
def simulate_pipeline(anomaly_active, anomaly_type, anomaly_start_time):
    """Generate realistic pipeline data with dynamics and anomalies."""
    
    duration_seconds = 120
    n_points = 600
    t = np.linspace(0, duration_seconds, n_points)
    
    # Base signals
    pressure_base = pressure_setpoint * (pump_speed / 100) * (valve_position / 100) * 0.8
    flow_base = flow_setpoint * (valve_position / 100) * (pump_speed / 100) * 0.7
    temp_base = temperature_setpoint + (ambient_temp - 20) * 0.3 + (pressure_setpoint - 60) * 0.05
    volume_base = volume_setpoint + (flow_setpoint - 110) * 0.1
    
    # Generate signals
    pressure = pressure_base + 5 * np.sin(2 * np.pi * t / 25) + 2 * np.sin(2 * np.pi * t / 60)
    flow = flow_base + 8 * np.sin(2 * np.pi * t / 20) + 3 * np.cos(2 * np.pi * t / 45)
    temperature = temp_base + 2 * np.sin(2 * np.pi * t / 30) + 1.5 * np.cos(2 * np.pi * t / 50)
    volume = volume_base + 6 * np.sin(2 * np.pi * t / 35) + 2 * np.cos(2 * np.pi * t / 55)
    
    # Add noise
    pressure += np.random.normal(0, noise_level, n_points)
    flow += np.random.normal(0, noise_level * 1.5, n_points)
    temperature += np.random.normal(0, noise_level * 0.3, n_points)
    volume += np.random.normal(0, noise_level * 2, n_points)
    
    # ========== APPLY ANOMALIES ==========
    if anomaly_active and anomaly_start_time is not None:
        anomaly_idx = int(0.5 * n_points)
        
        if anomaly_type == "Leak (Pressure Drop)":
            pressure[anomaly_idx:] -= 15 + 5 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 10))
            flow[anomaly_idx:] -= 20 + 10 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 15))
            temperature[anomaly_idx:] += 3 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 20))
            
        elif anomaly_type == "Blockage (Flow Drop)":
            flow[anomaly_idx:] -= 40 + 20 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 5))
            pressure[anomaly_idx:] += 10 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 8))
            temperature[anomaly_idx:] += 5 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 10))
            
        elif anomaly_type == "Pump Failure (Pressure/Flow Drop)":
            pressure[anomaly_idx:] -= 30 + 20 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 3))
            flow[anomaly_idx:] -= 50 + 30 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 3))
            temperature[anomaly_idx:] += 8 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 5))
            
        elif anomaly_type == "Sensor Drift":
            pressure[anomaly_idx:] += 5 * (t[anomaly_idx:] - t[anomaly_idx]) / 20
            flow[anomaly_idx:] += 8 * (t[anomaly_idx:] - t[anomaly_idx]) / 25
            
        elif anomaly_type == "Pressure Surge":
            surge_location = int(0.6 * n_points)
            pressure[surge_location:surge_location+50] += 25 * np.sin(2 * np.pi * (t[surge_location:surge_location+50] - t[surge_location]) / 3)
            flow[surge_location:surge_location+50] += 15 * np.sin(2 * np.pi * (t[surge_location:surge_location+50] - t[surge_location]) / 2)
            
        elif anomaly_type == "Temperature Spike":
            spike_location = int(0.65 * n_points)
            temperature[spike_location:spike_location+30] += 15 * np.exp(-((t[spike_location:spike_location+30] - t[spike_location]) / 5))
    
    return t, pressure, flow, temperature, volume

# -------------------- ANOMALY DETECTION ENGINE --------------------
def detect_anomalies(pressure, flow, temperature, volume, baseline_pressure, baseline_flow):
    anomalies = []
    detection_time = None
    
    if len(pressure) < 10:
        return anomalies, False, detection_time
    
    current_pressure = pressure[-1]
    current_flow = flow[-1]
    current_temp = temperature[-1]
    
    # Rule-based
    pressure_drop = baseline_pressure - current_pressure
    flow_drop = baseline_flow - current_flow
    pressure_roc = abs(pressure[-1] - pressure[-5]) if len(pressure) >= 5 else 0
    
    # Z-score
    pressure_mean = np.mean(pressure[-50:]) if len(pressure) >= 50 else np.mean(pressure)
    pressure_std = np.std(pressure[-50:]) if len(pressure) >= 50 else np.std(pressure)
    pressure_zscore = abs((current_pressure - pressure_mean) / (pressure_std + 0.001))
    
    flow_mean = np.mean(flow[-50:]) if len(flow) >= 50 else np.mean(flow)
    flow_std = np.std(flow[-50:]) if len(flow) >= 50 else np.std(flow)
    flow_zscore = abs((current_flow - flow_mean) / (flow_std + 0.001))
    
    pressure_change = abs(pressure[-1] - pressure[-2]) if len(pressure) >= 2 else 0
    flow_change = abs(flow[-1] - flow[-2]) if len(flow) >= 2 else 0
    
    anomaly_detected = False
    anomaly_reasons = []
    
    if pressure_drop > 12 and flow_drop > 20:
        anomaly_detected = True
        anomaly_reasons.append("🔴 Pressure and flow dropped significantly (leak/blockage)")
    elif pressure_drop > 8:
        anomaly_detected = True
        anomaly_reasons.append("🟡 Pressure is decreasing (possible leak)")
    elif flow_drop > 15:
        anomaly_detected = True
        anomaly_reasons.append("🟡 Flow is decreasing (possible blockage)")
    
    if pressure_zscore > 4.0:
        anomaly_detected = True
        anomaly_reasons.append(f"📊 Pressure statistically abnormal (Z={pressure_zscore:.2f})")
    if flow_zscore > 3.5:
        anomaly_detected = True
        anomaly_reasons.append(f"📊 Flow statistically abnormal (Z={flow_zscore:.2f})")
    
    if pressure_change > 10:
        anomaly_detected = True
        anomaly_reasons.append(f"⚡ Rapid pressure change ({pressure_change:.1f} psi)")
    if flow_change > 15:
        anomaly_detected = True
        anomaly_reasons.append(f"⚡ Rapid flow change ({flow_change:.1f} m³/h)")
    
    if current_temp > temperature_setpoint + 8:
        anomaly_detected = True
        anomaly_reasons.append("🌡️ Temperature is too high")
    
    if anomaly_detected:
        detection_time = datetime.now()
    
    return anomaly_reasons, anomaly_detected, detection_time

# -------------------- MAIN APP --------------------
def main():
    # Check for random anomaly injection
    check_and_inject_random()
    
    # Generate simulation data
    t, pressure, flow, temperature, volume = simulate_pipeline(
        st.session_state.anomaly_active,
        st.session_state.anomaly_type,
        st.session_state.anomaly_start_time
    )
    
    # Update history (only store last 50 points for smooth plot)
    st.session_state.history_pressure.extend(pressure[-50:])
    st.session_state.history_flow.extend(flow[-50:])
    st.session_state.history_temperature.extend(temperature[-50:])
    st.session_state.history_volume.extend(volume[-50:])
    
    max_history = 600
    if len(st.session_state.history_pressure) > max_history:
        st.session_state.history_pressure = st.session_state.history_pressure[-max_history:]
        st.session_state.history_flow = st.session_state.history_flow[-max_history:]
        st.session_state.history_temperature = st.session_state.history_temperature[-max_history:]
        st.session_state.history_volume = st.session_state.history_volume[-max_history:]
    
    # -------------------- PLOT --------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(t, pressure, color='blue', linewidth=2, label='Current')
    axes[0].axhline(y=pressure_setpoint, color='blue', linestyle='--', alpha=0.5, label='Setpoint')
    axes[0].set_ylabel('Pressure (psi)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    axes[1].plot(t, flow, color='green', linewidth=2, label='Current')
    axes[1].axhline(y=flow_setpoint, color='green', linestyle='--', alpha=0.5, label='Setpoint')
    axes[1].set_ylabel('Flow Rate (m³/h)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    axes[2].plot(t, temperature, color='orange', linewidth=2, label='Current')
    axes[2].axhline(y=temperature_setpoint, color='orange', linestyle='--', alpha=0.5, label='Setpoint')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    axes[3].plot(t, volume, color='purple', linewidth=2, label='Current')
    axes[3].axhline(y=volume_setpoint, color='purple', linestyle='--', alpha=0.5, label='Setpoint')
    axes[3].set_ylabel('Volume Rate (m³)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')
    
    # Highlight anomaly zone
    if st.session_state.anomaly_active:
        anomaly_idx = int(0.5 * len(t))
        for ax in axes:
            ax.axvspan(t[anomaly_idx], t[-1], alpha=0.15, color='red', label='Anomaly Zone')
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # -------------------- ANOMALY DETECTION --------------------
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("🔍 Real-time Monitoring")
        
        anomaly_reasons, anomaly_detected, detection_time = detect_anomalies(
            pressure, flow, temperature, volume,
            pressure_setpoint, flow_setpoint
        )
        
        if anomaly_detected:
            st.error("🚨 **ANOMALY DETECTED!**")
            for reason in anomaly_reasons:
                st.warning(reason)
            
            if detection_time:
                log_entry = f"{detection_time.strftime('%H:%M:%S')} - {'; '.join(anomaly_reasons)}"
                if log_entry not in st.session_state.detection_log[-10:]:
                    st.session_state.detection_log.append(log_entry)
            
            st.markdown("""
            <div style="border: 3px solid red; padding: 20px; border-radius: 10px; background-color: #FFEBEE;">
                <h3 style="color: red; margin: 0;">⚠️ SYSTEM ALERT</h3>
                <p style="margin: 5px 0;">Anomaly detected! Check system parameters immediately.</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.success("✅ System operating within normal parameters.")
            if st.session_state.random_anomaly_enabled:
                st.info("🔄 Random anomalies enabled – waiting for next injection...")
            else:
                st.info("💡 Enable random anomalies or inject manually to test detection.")
    
    with col2:
        st.subheader("📊 Live Metrics")
        st.metric("Current Pressure", f"{pressure[-1]:.1f} psi", 
                  f"{pressure[-1] - pressure_setpoint:.1f}" if len(pressure) > 0 else "0")
        st.metric("Current Flow", f"{flow[-1]:.1f} m³/h",
                  f"{flow[-1] - flow_setpoint:.1f}" if len(flow) > 0 else "0")
    
    with col3:
        st.subheader("📈 Status")
        if st.session_state.anomaly_active:
            st.warning(f"⚠️ Anomaly active: {st.session_state.anomaly_type}")
            if st.session_state.anomaly_detected:
                st.success("✅ System detected it!")
            else:
                st.info("🔄 Detection system monitoring...")
        else:
            st.success("✅ Normal operation")
    
    # -------------------- DETECTION LOG --------------------
    with st.expander("📋 Detection Log"):
        if st.session_state.detection_log:
            for log in st.session_state.detection_log[-15:]:
                st.write(f"• {log}")
        else:
            st.write("No anomalies detected yet.")
    
    # -------------------- RAW DATA --------------------
    with st.expander("📊 View Raw Data"):
        df = pd.DataFrame({
            'Time (s)': t.round(2),
            'Pressure (psi)': pressure.round(2),
            'Flow (m³/h)': flow.round(2),
            'Temperature (°C)': temperature.round(2),
            'Volume (m³)': volume.round(2)
        })
        st.dataframe(df)

    # Auto-refresh
    if st.session_state.anomaly_active:
        st.sidebar.info("🔄 Anomaly active – monitoring in real time.")

if __name__ == "__main__":
    main()