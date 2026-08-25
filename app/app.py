"""
app/app.py
Pipeline Leak Simulator – Fully working with random anomalies.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Pipeline Leak Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
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
    div[data-testid="stAlert"] {
        padding: 1rem !important;
        font-size: 1.1rem !important;
        border-radius: 8px !important;
        border: 2px solid #b71c1c !important;
    }
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
    .custom-alert h4 { color: white !important; margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'anomaly_active' not in st.session_state:
    st.session_state.anomaly_active = False
if 'anomaly_type' not in st.session_state:
    st.session_state.anomaly_type = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

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
    st.subheader("🎲 Random Anomalies")

    random_enabled = st.checkbox("Enable random anomalies", value=False)
    if random_enabled:
        anomaly_interval = st.slider("Interval (s)", 2, 15, 5, 1)
    else:
        anomaly_interval = 5

    st.markdown("---")
    st.subheader("💥 Manual Injection")

    anomaly_type = st.selectbox(
        "Type",
        ["None", "Leak (Pressure Drop)", "Blockage (Flow Drop)",
         "Pump Failure", "Sensor Drift", "Pressure Surge", "Temperature Spike"]
    )

    if st.button("Inject Anomaly", type="primary"):
        if anomaly_type != "None":
            st.session_state.anomaly_active = True
            st.session_state.anomaly_type = anomaly_type
            st.success(f"✅ {anomaly_type} injected!")

    if st.button("🔄 Reset System"):
        st.session_state.anomaly_active = False
        st.session_state.anomaly_type = None
        st.session_state.detection_log = []
        st.success("🔄 System reset!")

# -------------------- RANDOM ANOMALY LOGIC --------------------
def check_random_anomaly(random_enabled, interval):
    if not random_enabled:
        return
    now = datetime.now()
    elapsed = (now - st.session_state.last_update).total_seconds()
    if elapsed > interval:
        anomaly_types = ["Leak (Pressure Drop)", "Blockage (Flow Drop)",
                        "Pump Failure", "Sensor Drift", "Pressure Surge", "Temperature Spike"]
        weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.1]
        chosen = random.choices(anomaly_types, weights=weights)[0]
        st.session_state.anomaly_active = True
        st.session_state.anomaly_type = chosen
        st.session_state.last_update = now

# -------------------- SIMULATION --------------------
def generate_data(p_set, f_set, t_set, v_set, valve, pump, ambient, noise):
    TOTAL_TIME = 120
    N_POINTS = 600
    t = np.linspace(0, TOTAL_TIME, N_POINTS)

    pressure_base = p_set * (pump / 100) * (valve / 100) * 0.8
    flow_base = f_set * (valve / 100) * (pump / 100) * 0.7
    temp_base = t_set + (ambient - 20) * 0.3 + (p_set - 60) * 0.05
    volume_base = v_set + (f_set - 110) * 0.1

    pressure = pressure_base + 5 * np.sin(2 * np.pi * t / 25) + 2 * np.sin(2 * np.pi * t / 60)
    flow = flow_base + 8 * np.sin(2 * np.pi * t / 20) + 3 * np.cos(2 * np.pi * t / 45)
    temp = temp_base + 2 * np.sin(2 * np.pi * t / 30) + 1.5 * np.cos(2 * np.pi * t / 50)
    volume = volume_base + 6 * np.sin(2 * np.pi * t / 35) + 2 * np.cos(2 * np.pi * t / 55)

    pressure += np.random.normal(0, noise, N_POINTS)
    flow += np.random.normal(0, noise * 1.5, N_POINTS)
    temp += np.random.normal(0, noise * 0.3, N_POINTS)
    volume += np.random.normal(0, noise * 2, N_POINTS)

    return t, pressure, flow, temp, volume

# -------------------- DETECTION --------------------
def compute_severity(p_drop, f_drop, t_rise, z_p, z_f):
    score = 0
    if p_drop > 0: score += min(40, (p_drop / 20) * 40)
    if f_drop > 0: score += min(35, (f_drop / 30) * 35)
    if z_p > 3: score += min(15, (z_p / 6) * 15)
    if z_f > 3: score += min(10, (z_f / 6) * 10)
    if p_drop > 8 and f_drop > 15: score += 10
    return min(100, int(score))

def detect_anomalies(pressure, flow, temp, baseline_p, baseline_f, temp_setpoint):
    if len(pressure) < 10:
        return [], False, 0

    idx = -1
    curr_p = pressure[idx]
    curr_f = flow[idx]
    curr_t = temp[idx]
    p_drop = baseline_p - curr_p
    f_drop = baseline_f - curr_f

    # Statistics
    recent_p = pressure[-50:] if len(pressure) >= 50 else pressure
    recent_f = flow[-50:] if len(flow) >= 50 else flow
    p_mean, p_std = np.mean(recent_p), np.std(recent_p)
    f_mean, f_std = np.mean(recent_f), np.std(recent_f)
    z_p = abs((curr_p - p_mean) / (p_std + 0.001))
    z_f = abs((curr_f - f_mean) / (f_std + 0.001))

    reasons = []
    detected = False

    if p_drop > 12 and f_drop > 20:
        detected = True
        reasons.append("🔴 Large pressure & flow drop (leak)")
    elif p_drop > 8:
        detected = True
        reasons.append("🟡 Pressure decreasing (possible leak)")
    elif f_drop > 15:
        detected = True
        reasons.append("🟡 Flow decreasing (possible blockage)")
    if z_p > 4.0:
        detected = True
        reasons.append(f"📊 Pressure statistical (Z={z_p:.2f})")
    if z_f > 3.5:
        detected = True
        reasons.append(f"📊 Flow statistical (Z={z_f:.2f})")
    if curr_t > temp_setpoint + 8:
        detected = True
        reasons.append("🌡️ Temperature too high")

    severity = compute_severity(p_drop, f_drop, curr_t - temp_setpoint, z_p, z_f) if detected else 0
    return reasons, detected, severity

# -------------------- MAIN --------------------
st.title("🛢️ Pipeline Leak Simulator")
st.markdown("Adjust settings in the sidebar. Enable **random anomalies** or inject manually to test detection.")

# Check for random anomaly
check_random_anomaly(random_enabled, anomaly_interval)

# Generate data
t, pressure, flow, temp, volume = generate_data(
    pressure_setpoint, flow_setpoint, temperature_setpoint, volume_setpoint,
    valve_position, pump_speed, ambient_temp, noise_level
)

# Apply anomaly if active
if st.session_state.anomaly_active and st.session_state.anomaly_type is not None:
    anomaly_idx = int(0.5 * len(t))
    anomaly_type = st.session_state.anomaly_type
    if "Leak" in anomaly_type:
        pressure[anomaly_idx:] -= 15
        flow[anomaly_idx:] -= 20
        temp[anomaly_idx:] += 3
    elif "Blockage" in anomaly_type:
        flow[anomaly_idx:] -= 40
        pressure[anomaly_idx:] += 10
        temp[anomaly_idx:] += 5
    elif "Pump Failure" in anomaly_type:
        pressure[anomaly_idx:] -= 30
        flow[anomaly_idx:] -= 50
        temp[anomaly_idx:] += 8
    elif "Sensor Drift" in anomaly_type:
        drift = np.linspace(0, 10, len(t) - anomaly_idx)
        pressure[anomaly_idx:] += drift
        flow[anomaly_idx:] += drift * 0.8
    elif "Pressure Surge" in anomaly_type:
        surge = 25 * np.sin(2 * np.pi * t[:100] / 5)
        pressure[anomaly_idx:anomaly_idx+100] += surge
        flow[anomaly_idx:anomaly_idx+100] += 15
    elif "Temperature Spike" in anomaly_type:
        spike = 15 * np.exp(-(t[:30] / 5))
        temp[anomaly_idx:anomaly_idx+30] += spike

# Detection
reasons, detected, severity = detect_anomalies(
    pressure, flow, temp, pressure_setpoint, flow_setpoint, temperature_setpoint
)

if detected and st.session_state.anomaly_active:
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'anomaly_type': st.session_state.anomaly_type,
        'severity': severity,
        'reasons': '; '.join(reasons)
    }
    if not st.session_state.detection_log or st.session_state.detection_log[-1] != log_entry:
        st.session_state.detection_log.append(log_entry)
    # Reset anomaly after logging
    st.session_state.anomaly_active = False

# ---- Plot ----
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axes[0].plot(t, pressure, color='blue', linewidth=2)
axes[0].axhline(y=pressure_setpoint, color='blue', linestyle='--', alpha=0.5)
axes[0].set_ylabel('Pressure (psi)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, flow, color='green', linewidth=2)
axes[1].axhline(y=flow_setpoint, color='green', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Flow (m³/h)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, temp, color='orange', linewidth=2)
axes[2].axhline(y=temperature_setpoint, color='orange', linestyle='--', alpha=0.5)
axes[2].set_ylabel('Temp (°C)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, volume, color='purple', linewidth=2)
axes[3].axhline(y=volume_setpoint, color='purple', linestyle='--', alpha=0.5)
axes[3].set_xlabel('Time (seconds)')
axes[3].set_ylabel('Volume (m³)')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---- Metrics and Alert ----
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("🔍 Real-time Monitoring")
    if detected:
        st.markdown(f"""
        <div class="custom-alert" style="background-color:#b71c1c;color:white;padding:15px;border-radius:10px;border:3px solid #ff1744;">
            <h4 style="color:white;margin:0;">🚨 ANOMALY DETECTED!</h4>
            <p style="margin:5px 0;">Severity: <strong>{severity}/100</strong></p>
            <p style="margin:5px 0;">Type: {st.session_state.anomaly_type}</p>
            <p style="margin:5px 0;">Reasons: {'; '.join(reasons)}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if random_enabled:
            st.success("✅ System normal – random anomalies active")
        else:
            st.success("✅ System normal")

with col2:
    st.subheader("📊 Live Metrics")
    st.metric("Pressure", f"{pressure[-1]:.1f} psi", f"{pressure[-1] - pressure_setpoint:.1f}")
    st.metric("Flow", f"{flow[-1]:.1f} m³/h", f"{flow[-1] - flow_setpoint:.1f}")

with col3:
    st.subheader("📈 Status")
    if detected:
        st.error("🚨 Anomaly")
    elif st.session_state.anomaly_active:
        st.warning("⚠️ Anomaly active")
    else:
        st.success("✅ Normal")

# ---- Detection Log ----
with st.expander("📋 Detection Log"):
    if st.session_state.detection_log:
        st.dataframe(pd.DataFrame(st.session_state.detection_log), use_container_width=True)
    else:
        st.write("No anomalies detected yet.")

# ---- Export ----
if st.session_state.detection_log:
    if st.button("📤 Export Log (CSV)"):
        df_exp = pd.DataFrame(st.session_state.detection_log)
        csv = df_exp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"leak_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

# ---- Raw Data ----
with st.expander("📊 Raw Data"):
    df = pd.DataFrame({
        'Time (s)': t.round(2),
        'Pressure': pressure.round(2),
        'Flow': flow.round(2),
        'Temp': temp.round(2),
        'Volume': volume.round(2)
    })
    st.dataframe(df, use_container_width=True)