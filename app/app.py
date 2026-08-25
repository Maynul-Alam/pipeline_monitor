"""
app/app.py
Pipeline Leak Simulator – Full features + auto‑leak after delay.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

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
    .css-1d391kg { padding-top: 0 !important; }
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
    .stProgress > div > div {
        background-color: #1e88e5 !important;
        height: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE INIT --------------------
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False
if 'sim_time' not in st.session_state:
    st.session_state.sim_time = 0.0
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'leak_started' not in st.session_state:
    st.session_state.leak_started = False
if 'leak_start_time' not in st.session_state:
    st.session_state.leak_start_time = 5
if 'leak_duration' not in st.session_state:
    st.session_state.leak_duration = 60

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
    st.subheader("💧 Leak Parameters")
    leak_start_time = st.slider("Leak starts after (s)", 1, 30, 5, 1)
    leak_duration = st.slider("Leak duration (s)", 5, 100, 60, 5)

    # Store leak params in session state for use in detection loop
    st.session_state.leak_start_time = leak_start_time
    st.session_state.leak_duration = leak_duration

    st.markdown("---")
    if st.button("▶️ Start Simulation", type="primary"):
        st.session_state.sim_running = True
        st.session_state.sim_time = 0.0
        st.session_state.detection_log = []
        st.session_state.leak_started = False
        # Generate fresh data using current sidebar values
        t, p, f, temp, v = generate_simulation_data(
            pressure_setpoint, flow_setpoint, temperature_setpoint, volume_setpoint,
            valve_position, pump_speed, ambient_temp, noise_level,
            leak_start_time, leak_duration
        )
        st.session_state.sim_data = (t, p, f, temp, v)
        st.session_state.current_idx = 0
        st.rerun()

    if st.button("🔄 Reset System"):
        st.session_state.sim_running = False
        st.session_state.sim_time = 0.0
        st.session_state.sim_data = None
        st.session_state.current_idx = 0
        st.session_state.detection_log = []
        st.session_state.leak_started = False
        st.rerun()

# -------------------- SIMULATION ENGINE --------------------
def generate_simulation_data(p_set, f_set, t_set, v_set, valve, pump, ambient, noise,
                             leak_start, leak_duration):
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

    # Apply leak
    leak_start_idx = int((leak_start / TOTAL_TIME) * N_POINTS)
    leak_end_idx = int(((leak_start + leak_duration) / TOTAL_TIME) * N_POINTS)
    leak_end_idx = min(leak_end_idx, N_POINTS)
    for i in range(leak_start_idx, leak_end_idx):
        progress = (i - leak_start_idx) / (leak_end_idx - leak_start_idx + 1)
        pressure[i] -= 15 + 5 * progress
        flow[i] -= 20 + 10 * progress
        temp[i] += 3 + 2 * progress

    return t, pressure, flow, temp, volume

# -------------------- DETECTION FUNCTIONS --------------------
def compute_severity(p_drop, f_drop, t_rise, z_p, z_f):
    score = 0
    if p_drop > 0: score += min(40, (p_drop / 20) * 40)
    if f_drop > 0: score += min(35, (f_drop / 30) * 35)
    if z_p > 3: score += min(15, (z_p / 6) * 15)
    if z_f > 3: score += min(10, (z_f / 6) * 10)
    if p_drop > 8 and f_drop > 15: score += 10
    return min(100, int(score))

def detect_at_index(pressure, flow, temp, idx, baseline_p, baseline_f, temp_setpoint):
    if idx < 10:
        return [], False, 0
    start = max(0, idx - 50)
    recent_p = pressure[start:idx+1]
    recent_f = flow[start:idx+1]
    curr_p = pressure[idx]
    curr_f = flow[idx]
    curr_t = temp[idx]
    p_drop = baseline_p - curr_p
    f_drop = baseline_f - curr_f

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

# -------------------- MAIN APP --------------------
st.title("🛢️ Pipeline Leak Simulator")
st.markdown("Adjust settings in the sidebar, then click **Start Simulation** – a leak will occur automatically after the set delay.")

# If simulation is running, update in a loop
if st.session_state.sim_running and st.session_state.sim_data is not None:
    t, p, f, temp, v = st.session_state.sim_data
    TOTAL_TIME = t[-1]
    total_steps = 100
    step_time = TOTAL_TIME / total_steps

    current_time = st.session_state.sim_time
    if current_time >= TOTAL_TIME:
        st.session_state.sim_running = False
        st.success("✅ Simulation finished.")
        st.rerun()
        st.stop()

    idx = np.argmin(np.abs(t - current_time))
    st.session_state.current_idx = idx
    curr_p = p[idx]
    curr_f = f[idx]
    curr_t = temp[idx]
    curr_v = v[idx]

    # Retrieve leak start from session state
    leak_start = st.session_state.leak_start_time
    if current_time >= leak_start:
        st.session_state.leak_started = True
        reasons, detected, severity = detect_at_index(p, f, temp, idx,
                                                      pressure_setpoint, flow_setpoint,
                                                      temperature_setpoint)
        if detected:
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'time_sec': f"{current_time:.1f}",
                'pressure_drop': f"{pressure_setpoint - curr_p:.2f}",
                'flow_drop': f"{flow_setpoint - curr_f:.2f}",
                'severity': severity,
                'reasons': '; '.join(reasons)
            }
            if not st.session_state.detection_log or st.session_state.detection_log[-1] != log_entry:
                st.session_state.detection_log.append(log_entry)
    else:
        detected = False
        reasons = []
        severity = 0

    # ---- Plot ----
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, p, color='blue', linewidth=1, alpha=0.6)
    axes[0].axhline(y=pressure_setpoint, color='blue', linestyle='--', alpha=0.4)
    axes[0].scatter(current_time, curr_p, color='red', s=80, zorder=5)
    axes[0].set_ylabel('Pressure (psi)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, f, color='green', linewidth=1, alpha=0.6)
    axes[1].axhline(y=flow_setpoint, color='green', linestyle='--', alpha=0.4)
    axes[1].scatter(current_time, curr_f, color='red', s=80, zorder=5)
    axes[1].set_ylabel('Flow (m³/h)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, temp, color='orange', linewidth=1, alpha=0.6)
    axes[2].axhline(y=temperature_setpoint, color='orange', linestyle='--', alpha=0.4)
    axes[2].scatter(current_time, curr_t, color='red', s=80, zorder=5)
    axes[2].set_ylabel('Temp (°C)')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, v, color='purple', linewidth=1, alpha=0.6)
    axes[3].axhline(y=volume_setpoint, color='purple', linestyle='--', alpha=0.4)
    axes[3].scatter(current_time, curr_v, color='red', s=80, zorder=5)
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_ylabel('Volume (m³)')
    axes[3].grid(True, alpha=0.3)

    # Highlight leak period
    if st.session_state.leak_started:
        leak_start_idx = np.argmin(np.abs(t - leak_start))
        for ax in axes:
            ax.axvspan(t[leak_start_idx], t[-1], alpha=0.15, color='red')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---- Metrics ----
    col1, col2, col3 = st.columns(3)
    col1.metric("Pressure", f"{curr_p:.1f} psi", f"{curr_p - pressure_setpoint:.1f}")
    col2.metric("Flow", f"{curr_f:.1f} m³/h", f"{curr_f - flow_setpoint:.1f}")
    col3.metric("Temperature", f"{curr_t:.1f} °C", f"{curr_t - temperature_setpoint:.1f}")

    # ---- Alert ----
    if detected:
        st.markdown(f"""
        <div class="custom-alert" style="background-color:#b71c1c;color:white;padding:15px;border-radius:10px;border:3px solid #ff1744;">
            <h4 style="color:white;margin:0;">🚨 ANOMALY DETECTED!</h4>
            <p style="margin:5px 0;">Severity: <strong>{severity}/100</strong></p>
            <p style="margin:5px 0;">Reasons: {'; '.join(reasons)}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.session_state.leak_started:
            st.warning("⚠️ Leak active – monitoring...")
        else:
            st.success("✅ System normal.")

    # ---- Progress ----
    progress = current_time / TOTAL_TIME
    st.progress(min(1.0, progress))

    # ---- Detection Log ----
    if st.session_state.detection_log:
        st.subheader("📋 Detection Log")
        st.dataframe(pd.DataFrame(st.session_state.detection_log), use_container_width=True)

    # ---- Advance time ----
    st.session_state.sim_time += step_time
    time.sleep(0.3)
    st.rerun()

else:
    # Not running – show placeholder or previous log
    if st.session_state.detection_log:
        st.subheader("📋 Detection Log")
        st.dataframe(pd.DataFrame(st.session_state.detection_log), use_container_width=True)
    else:
        st.info("Configure settings in the sidebar and click 'Start Simulation'.")

# Export button (always visible)
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