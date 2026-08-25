"""
app/app.py
Pipeline Leak Simulator – Single continuous leak after delay
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Pipeline Leak Detector",
    layout="wide",
    initial_sidebar_state="auto"
)

# -------------------- CUSTOM CSS FOR VISIBILITY --------------------
st.markdown("""
<style>
    /* Mobile responsive */
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
    /* Make alert boxes very visible */
    div[data-testid="stAlert"] {
        padding: 1rem !important;
        font-size: 1.1rem !important;
        border-radius: 8px !important;
        border: 2px solid #b71c1c !important;
    }
    .stAlert svg {
        width: 24px !important;
        height: 24px !important;
    }
    .stAlert .stAlertContent {
        color: #000 !important;
        font-weight: bold !important;
    }
    /* Custom alert override for error */
    .custom-alert {
        background-color: #b71c1c !important;
        color: white !important;
        padding: 15px !important;
        border-radius: 10px !important;
        margin: 10px 0 !important;
        border: 3px solid #ff1744 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        text-align: center !important;
    }
    .custom-alert h4 {
        color: white !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE INIT --------------------
if 'anomaly_active' not in st.session_state:
    st.session_state.anomaly_active = False
    st.session_state.anomaly_start_time = None
    st.session_state.detection_log = []
    st.session_state.leak_enabled = False

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
    st.subheader("💧 Leak Simulation")
    leak_enabled = st.checkbox("Enable leak", value=False)
    st.session_state.leak_enabled = leak_enabled
    if leak_enabled:
        leak_start_time = st.slider("Leak starts after (seconds)", 1, 30, 5, 1)
        leak_duration = st.slider("Leak duration (seconds)", 5, 100, 60, 5)
    else:
        leak_start_time = 5
        leak_duration = 60

    st.markdown("---")
    st.subheader("💥 Manual Injection")
    if st.button("Inject Leak Now", type="primary"):
        st.session_state.anomaly_active = True
        st.session_state.anomaly_start_time = datetime.now()
        st.success("✅ Leak injected instantly!")

    if st.button("Reset System"):
        st.session_state.anomaly_active = False
        st.session_state.anomaly_start_time = None
        st.session_state.detection_log = []
        st.success("🔄 System reset!")

# -------------------- SIMULATION ENGINE --------------------
def simulate_pipeline(leak_enabled, start_delay, duration):
    """Generates a 120-second simulation with a single leak."""
    total_time = 120
    n_points = 600
    t = np.linspace(0, total_time, n_points)
    
    # Base signals
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
    
    # Apply leak if enabled
    if leak_enabled:
        leak_start_idx = int((start_delay / total_time) * n_points)
        leak_end_idx = int(((start_delay + duration) / total_time) * n_points)
        leak_end_idx = min(leak_end_idx, n_points)
        for i in range(leak_start_idx, leak_end_idx):
            # Gradual pressure and flow drop with recovery after leak ends
            if i < leak_end_idx:
                # Leak active: pressure and flow drop, temp rises
                progress = (i - leak_start_idx) / (leak_end_idx - leak_start_idx)
                pressure[i] -= 15 + 5 * progress
                flow[i] -= 20 + 10 * progress
                temperature[i] += 3 + 2 * progress
            else:
                # Recovery (if we wanted to model recovery, but not needed here)
                pass
    
    return t, pressure, flow, temperature, volume

# -------------------- ANOMALY DETECTION & SEVERITY --------------------
def compute_severity(pressure_drop, flow_drop, temp_rise, zscore_p, zscore_f):
    score = 0
    if pressure_drop > 0:
        score += min(40, (pressure_drop / 20) * 40)
    if flow_drop > 0:
        score += min(35, (flow_drop / 30) * 35)
    if zscore_p > 3:
        score += min(15, (zscore_p / 6) * 15)
    if zscore_f > 3:
        score += min(10, (zscore_f / 6) * 10)
    if pressure_drop > 8 and flow_drop > 15:
        score += 10
    return min(100, int(score))

def detect_anomalies(pressure, flow, temperature, baseline_pressure, baseline_flow):
    if len(pressure) < 10:
        return [], False, None, 0
    current_pressure = pressure[-1]
    current_flow = flow[-1]
    current_temp = temperature[-1]
    pressure_drop = baseline_pressure - current_pressure
    flow_drop = baseline_flow - current_flow
    
    # Z-scores based on recent history (last 50 points)
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
        reasons.append("🔴 Large pressure & flow drop (leak/blockage)")
    elif pressure_drop > 8:
        anomaly_detected = True
        reasons.append("🟡 Pressure decreasing (possible leak)")
    elif flow_drop > 15:
        anomaly_detected = True
        reasons.append("🟡 Flow decreasing (possible blockage)")
    if pressure_zscore > 4.0:
        anomaly_detected = True
        reasons.append(f"📊 Pressure statistical anomaly (Z={pressure_zscore:.2f})")
    if flow_zscore > 3.5:
        anomaly_detected = True
        reasons.append(f"📊 Flow statistical anomaly (Z={flow_zscore:.2f})")
    if current_temp > temperature_setpoint + 8:
        anomaly_detected = True
        reasons.append("🌡️ Temperature too high")
    
    severity = 0
    detection_time = None
    if anomaly_detected:
        severity = compute_severity(pressure_drop, flow_drop, current_temp - temperature_setpoint,
                                    pressure_zscore, flow_zscore)
        detection_time = datetime.now()
        # Log
        log_entry = {
            'timestamp': detection_time.strftime('%Y-%m-%d %H:%M:%S'),
            'pressure_drop': f"{pressure_drop:.2f}",
            'flow_drop': f"{flow_drop:.2f}",
            'severity': severity,
            'reasons': '; '.join(reasons)
        }
        if log_entry not in st.session_state.detection_log[-10:]:
            st.session_state.detection_log.append(log_entry)
    return reasons, anomaly_detected, detection_time, severity

# -------------------- MAIN APP --------------------
def main():
    # Get leak parameters from session state
    leak_enabled = st.session_state.leak_enabled
    start_delay = st.session_state.get('leak_start_time', 5)
    duration = st.session_state.get('leak_duration', 60)
    
    # Generate simulation data
    t, pressure, flow, temperature, volume = simulate_pipeline(
        leak_enabled, start_delay, duration
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
    
    # Highlight leak period if enabled
    if leak_enabled:
        leak_start_idx = int((start_delay / t[-1]) * len(t))
        leak_end_idx = int(((start_delay + duration) / t[-1]) * len(t))
        leak_end_idx = min(leak_end_idx, len(t))
        if leak_end_idx > leak_start_idx:
            for ax in axes:
                ax.axvspan(t[leak_start_idx], t[leak_end_idx-1], alpha=0.15, color='red', label='Leak')
                ax.legend(loc='upper right', fontsize='x-small')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Monitoring & Metrics
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("🔍 Real-time Monitoring")
        reasons, detected, detection_time, severity = detect_anomalies(
            pressure, flow, temperature, pressure_setpoint, flow_setpoint
        )
        if detected:
            # Custom high-visibility alert
            st.markdown(f"""
            <div class="custom-alert" style="background-color:#b71c1c;color:white;padding:15px;border-radius:10px;border:3px solid #ff1744;">
                <h4 style="color:white;margin:0;">🚨 ANOMALY DETECTED!</h4>
                <p style="margin:5px 0;">Severity: <strong>{severity}/100</strong></p>
                <p style="margin:5px 0;">Reasons: {'; '.join(reasons)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ System operating normally.")
            if leak_enabled:
                st.info(f"⏳ Leak will start at t={start_delay}s and last for {duration}s")
            else:
                st.info("💡 Enable a leak in the sidebar to test detection.")
    
    with col2:
        st.subheader("📊 Live Metrics")
        st.metric("Pressure", f"{pressure[-1]:.1f} psi", f"{pressure[-1]-pressure_setpoint:.1f}")
        st.metric("Flow", f"{flow[-1]:.1f} m³/h", f"{flow[-1]-flow_setpoint:.1f}")
    
    with col3:
        st.subheader("📈 Status")
        if detected:
            st.error("🚨 LEAK DETECTED")
        elif leak_enabled and t[-1] > start_delay:
            st.warning("⚠️ Leak active")
        else:
            st.success("✅ Normal")
    
    # Export
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
            st.info("No anomalies detected yet.")
    
    with st.expander("📋 Detection Log (with Severity)"):
        if st.session_state.detection_log:
            st.dataframe(pd.DataFrame(st.session_state.detection_log), use_container_width=True)
        else:
            st.write("No anomalies yet.")
    
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