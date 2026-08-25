"""
app/app.py
Pipeline Simulator – Periodic Leak Detection with Severity Scoring & Export
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import random

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Pipeline Leak Detector",
    layout="wide",
    initial_sidebar_state="auto"
)

# -------------------- CUSTOM CSS FOR MOBILE --------------------
st.markdown("""
<style>
    @media (max-width: 768px) {
        .stApp { padding: 0.5rem !important; }
        h1 { font-size: 1.8rem !important; }
        h2, h3 { font-size: 1.2rem !important; }
        .stMetric label { font-size: 0.8rem !important; }
        .stMetric .stMetricValue { font-size: 1.2rem !important; }
        .stButton button { width: 100% !important; padding: 0.5rem !important; }
        .stSlider { padding: 0.2rem 0 !important; }
        .stColumns { gap: 0.5rem !important; }
    }
    .css-1d391kg { padding-top: 0 !important; }
    div[data-testid="stAlert"] { padding: 0.5rem !important; font-size: 0.9rem !important; }
    .streamlit-expanderHeader { font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

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
    st.session_state.detection_log = []  # list of dicts for export
    st.session_state.periodic_leak_enabled = False

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("⚙️ Pipeline Settings")

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        pressure_setpoint = st.slider("Pressure (psi)", 40, 80, 60, 1)
        temperature_setpoint = st.slider("Temp (°C)", 15, 40, 25, 1)
    with col2:
        flow_setpoint = st.slider("Flow (m³/h)", 80, 150, 110, 5)
        volume_setpoint = st.slider("Volume (m³)", 50, 120, 80, 5)

    st.subheader("🌊 System Dynamics")
    valve_position = st.slider("Valve (%)", 0, 100, 80, 5)
    pump_speed = st.slider("Pump (%)", 50, 100, 85, 5)
    ambient_temp = st.slider("Ambient (°C)", -5, 35, 20, 1)
    noise_level = st.slider("Noise", 0.0, 3.0, 0.8, 0.1)

    st.markdown("---")
    st.subheader("💧 Periodic Leak Simulation")
    periodic_leak_enabled = st.checkbox("Enable periodic leaks", value=False)
    st.session_state.periodic_leak_enabled = periodic_leak_enabled
    if periodic_leak_enabled:
        leak_interval = st.slider("Leak interval (seconds)", 3, 15, 5, 1)
        leak_duration = st.slider("Leak duration (seconds)", 1, 8, 3, 1)
    else:
        leak_interval = 5
        leak_duration = 3

    st.markdown("---")
    st.subheader("💥 Manual Leak Injection")
    if st.button("Inject Leak", type="primary"):
        st.session_state.anomaly_active = True
        st.session_state.anomaly_type = "Leak (Pressure Drop)"
        st.session_state.anomaly_start_time = datetime.now()
        st.session_state.anomaly_detected = False
        st.success("✅ Leak injected!")

    if st.button("Reset System"):
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
        st.success("🔄 System reset!")

# -------------------- PERIODIC LEAK LOGIC --------------------
def apply_periodic_leaks(t, pressure, flow, temperature, interval, duration):
    """
    Apply pressure/flow drops periodically at specified interval and duration.
    """
    for i, time_val in enumerate(t):
        # Check if time_val is within a leak window
        if (time_val % interval) < duration:
            # Apply leak effect: pressure drop and flow drop
            strength = 1.0 - (time_val % interval) / duration  # linear fade in/out
            pressure[i] -= 15 * strength + np.random.normal(0, 0.5)
            flow[i] -= 20 * strength + np.random.normal(0, 0.5)
            temperature[i] += 2 * strength + np.random.normal(0, 0.1)
    return pressure, flow, temperature

# -------------------- SIMULATION ENGINE --------------------
def simulate_pipeline(anomaly_active, anomaly_type, anomaly_start_time, periodic=False, interval=5, duration=3):
    duration_seconds = 120
    n_points = 600
    t = np.linspace(0, duration_seconds, n_points)
    
    pressure_base = pressure_setpoint * (pump_speed / 100) * (valve_position / 100) * 0.8
    flow_base = flow_setpoint * (valve_position / 100) * (pump_speed / 100) * 0.7
    temp_base = temperature_setpoint + (ambient_temp - 20) * 0.3 + (pressure_setpoint - 60) * 0.05
    volume_base = volume_setpoint + (flow_setpoint - 110) * 0.1
    
    pressure = pressure_base + 5 * np.sin(2 * np.pi * t / 25) + 2 * np.sin(2 * np.pi * t / 60)
    flow = flow_base + 8 * np.sin(2 * np.pi * t / 20) + 3 * np.cos(2 * np.pi * t / 45)
    temperature = temp_base + 2 * np.sin(2 * np.pi * t / 30) + 1.5 * np.cos(2 * np.pi * t / 50)
    volume = volume_base + 6 * np.sin(2 * np.pi * t / 35) + 2 * np.cos(2 * np.pi * t / 55)
    
    pressure += np.random.normal(0, noise_level, n_points)
    flow += np.random.normal(0, noise_level * 1.5, n_points)
    temperature += np.random.normal(0, noise_level * 0.3, n_points)
    volume += np.random.normal(0, noise_level * 2, n_points)
    
    # Apply periodic leaks if enabled
    if periodic:
        pressure, flow, temperature = apply_periodic_leaks(t, pressure, flow, temperature, interval, duration)
    
    # Also handle manual anomaly injection (single leak)
    if anomaly_active and anomaly_type == "Leak (Pressure Drop)" and anomaly_start_time is not None and not periodic:
        anomaly_idx = int(0.5 * n_points)
        pressure[anomaly_idx:] -= 15 + 5 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 10))
        flow[anomaly_idx:] -= 20 + 10 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 15))
        temperature[anomaly_idx:] += 3 * (1 - np.exp(-(t[anomaly_idx:] - t[anomaly_idx]) / 20))
    
    return t, pressure, flow, temperature, volume

# -------------------- ANOMALY DETECTION & SEVERITY --------------------
def compute_severity(pressure_drop, flow_drop, temp_rise, zscore_p, zscore_f):
    """Compute severity score 0-100 based on multiple factors."""
    score = 0
    # Pressure drop contribution (max 40)
    if pressure_drop > 0:
        score += min(40, (pressure_drop / 20) * 40)
    # Flow drop contribution (max 35)
    if flow_drop > 0:
        score += min(35, (flow_drop / 30) * 35)
    # Z-score contributions (max 25)
    if zscore_p > 3:
        score += min(15, (zscore_p / 6) * 15)
    if zscore_f > 3:
        score += min(10, (zscore_f / 6) * 10)
    # Bonus for combined effects
    if pressure_drop > 8 and flow_drop > 15:
        score += 10
    return min(100, int(score))

def detect_anomalies(pressure, flow, temperature, volume, baseline_pressure, baseline_flow):
    if len(pressure) < 10:
        return [], False, None, 0
    current_pressure = pressure[-1]
    current_flow = flow[-1]
    current_temp = temperature[-1]
    pressure_drop = baseline_pressure - current_pressure
    flow_drop = baseline_flow - current_flow
    
    pressure_mean = np.mean(pressure[-50:]) if len(pressure) >= 50 else np.mean(pressure)
    pressure_std = np.std(pressure[-50:]) if len(pressure) >= 50 else np.std(pressure)
    pressure_zscore = abs((current_pressure - pressure_mean) / (pressure_std + 0.001))
    flow_mean = np.mean(flow[-50:]) if len(flow) >= 50 else np.mean(flow)
    flow_std = np.std(flow[-50:]) if len(flow) >= 50 else np.std(flow)
    flow_zscore = abs((current_flow - flow_mean) / (flow_std + 0.001))
    
    anomaly_detected = False
    reasons = []
    
    if pressure_drop > 12 and flow_drop > 20:
        anomaly_detected = True
        reasons.append("🔴 Pressure & flow dropped (leak/blockage)")
    elif pressure_drop > 8:
        anomaly_detected = True
        reasons.append("🟡 Pressure decreasing (possible leak)")
    elif flow_drop > 15:
        anomaly_detected = True
        reasons.append("🟡 Flow decreasing (possible blockage)")
    if pressure_zscore > 4.0:
        anomaly_detected = True
        reasons.append(f"📊 Pressure statistical (Z={pressure_zscore:.2f})")
    if flow_zscore > 3.5:
        anomaly_detected = True
        reasons.append(f"📊 Flow statistical (Z={flow_zscore:.2f})")
    if current_temp > temperature_setpoint + 8:
        anomaly_detected = True
        reasons.append("🌡️ Temperature too high")
    
    severity = 0
    if anomaly_detected:
        severity = compute_severity(pressure_drop, flow_drop, current_temp - temperature_setpoint,
                                    pressure_zscore, flow_zscore)
        detection_time = datetime.now()
        # Log the event with severity
        log_entry = {
            'timestamp': detection_time.strftime('%Y-%m-%d %H:%M:%S'),
            'pressure_drop': f"{pressure_drop:.2f}",
            'flow_drop': f"{flow_drop:.2f}",
            'severity': severity,
            'reasons': '; '.join(reasons)
        }
        # Store in session state for export
        if log_entry not in st.session_state.detection_log[-10:]:
            st.session_state.detection_log.append(log_entry)
        return reasons, True, detection_time, severity
    else:
        return reasons, False, None, 0

# -------------------- MAIN APP --------------------
def main():
    # Determine if periodic leaks are active
    periodic = st.session_state.periodic_leak_enabled
    interval = st.session_state.get('leak_interval', 5)
    duration = st.session_state.get('leak_duration', 3)
    
    # Generate simulation data
    t, pressure, flow, temperature, volume = simulate_pipeline(
        st.session_state.anomaly_active,
        st.session_state.anomaly_type,
        st.session_state.anomaly_start_time,
        periodic=periodic,
        interval=interval,
        duration=duration
    )
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, pressure, color='blue', linewidth=2, label='Pressure')
    axes[0].axhline(y=pressure_setpoint, color='blue', linestyle='--', alpha=0.5, label='Setpoint')
    axes[0].set_ylabel('Pressure (psi)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize='x-small')
    
    axes[1].plot(t, flow, color='green', linewidth=2, label='Flow')
    axes[1].axhline(y=flow_setpoint, color='green', linestyle='--', alpha=0.5, label='Setpoint')
    axes[1].set_ylabel('Flow (m³/h)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize='x-small')
    
    axes[2].plot(t, temperature, color='orange', linewidth=2, label='Temperature')
    axes[2].axhline(y=temperature_setpoint, color='orange', linestyle='--', alpha=0.5, label='Setpoint')
    axes[2].set_ylabel('Temp (°C)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', fontsize='x-small')
    
    axes[3].plot(t, volume, color='purple', linewidth=2, label='Volume')
    axes[3].axhline(y=volume_setpoint, color='purple', linestyle='--', alpha=0.5, label='Setpoint')
    axes[3].set_ylabel('Volume (m³)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right', fontsize='x-small')
    
    # Highlight leak periods if periodic
    if periodic:
        for i in range(0, int(t[-1]), interval):
            if i + duration <= t[-1]:
                for ax in axes:
                    ax.axvspan(i, i+duration, alpha=0.1, color='red')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Monitoring and Metrics
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("🔍 Real-time Monitoring")
        reasons, detected, detection_time, severity = detect_anomalies(
            pressure, flow, temperature, volume,
            pressure_setpoint, flow_setpoint
        )
        if detected:
            st.error("🚨 **ANOMALY DETECTED!**")
            for reason in reasons:
                st.warning(reason)
            st.metric("Severity Score", f"{severity} / 100", delta="High" if severity > 70 else "Medium" if severity > 40 else "Low")
            st.markdown(f"""
            <div style="border: 2px solid red; padding: 10px; border-radius: 8px; background-color: #FFEBEE; margin: 5px 0;">
                <h4 style="color: red; margin: 0;">⚠️ ALERT</h4>
                <p style="margin: 5px 0;">Leak detected! Severity: {severity}/100</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ System normal.")
            if periodic:
                st.info(f"🔄 Periodic leak simulation active (every {interval}s)")
            else:
                st.info("💡 Enable periodic leaks or inject manually.")
    with col2:
        st.subheader("📊 Live Metrics")
        st.metric("Pressure", f"{pressure[-1]:.1f} psi", f"{pressure[-1]-pressure_setpoint:.1f}")
        st.metric("Flow", f"{flow[-1]:.1f} m³/h", f"{flow[-1]-flow_setpoint:.1f}")
    with col3:
        st.subheader("📈 Status")
        if st.session_state.anomaly_active or periodic:
            st.warning("⚠️ Leak condition")
        else:
            st.success("✅ Normal")
    
    # -------------------- EXPORT SECTION --------------------
    st.subheader("📤 Export Anomaly Log")
    if st.button("Export Detection Log (CSV)"):
        if st.session_state.detection_log:
            df_export = pd.DataFrame(st.session_state.detection_log)
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"leak_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
        else:
            st.info("No anomalies detected yet. Nothing to export.")
    
    # Detection Log (collapsible)
    with st.expander("📋 Detection Log (with Severity)"):
        if st.session_state.detection_log:
            # Display as table
            df_log = pd.DataFrame(st.session_state.detection_log)
            st.dataframe(df_log, use_container_width=True)
        else:
            st.write("No anomalies detected yet.")
    
    with st.expander("📊 Raw Data"):
        df = pd.DataFrame({
            'Time (s)': t.round(2),
            'Pressure': pressure.round(2),
            'Flow': flow.round(2),
            'Temp': temperature.round(2),
            'Volume': volume.round(2)
        })
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()