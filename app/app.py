"""
app/app.py
Pipeline Leak Simulator – Auto‑play with leak after 5 seconds.
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
    initial_sidebar_state="collapsed"  # hide sidebar for cleaner demo
)

# -------------------- CUSTOM CSS (high visibility) --------------------
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
        .stColumns { gap: 0.5rem !important; }
    }
    /* Alert styling */
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
    /* Progress bar */
    .stProgress > div > div {
        background-color: #1e88e5 !important;
        height: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if 'sim_running' not in st.session_state:
    st.session_state.sim_running = False
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None

# -------------------- SIMULATION PARAMETERS (fixed) --------------------
TOTAL_TIME = 120          # seconds
N_POINTS = 600
LEAK_START = 5            # seconds
LEAK_DURATION = 20        # seconds (or until end)

# Pipeline settings (fixed for demo)
PRESSURE_SETPOINT = 60
FLOW_SETPOINT = 110
TEMP_SETPOINT = 25
VOLUME_SETPOINT = 80
VALVE = 80
PUMP = 85
AMBIENT = 20
NOISE = 0.8

def generate_simulation_data():
    """Generate full 120s dataset with a leak starting at LEAK_START."""
    t = np.linspace(0, TOTAL_TIME, N_POINTS)
    # Base signals
    pressure_base = PRESSURE_SETPOINT * (PUMP / 100) * (VALVE / 100) * 0.8
    flow_base = FLOW_SETPOINT * (VALVE / 100) * (PUMP / 100) * 0.7
    temp_base = TEMP_SETPOINT + (AMBIENT - 20) * 0.3 + (PRESSURE_SETPOINT - 60) * 0.05
    volume_base = VOLUME_SETPOINT + (FLOW_SETPOINT - 110) * 0.1

    pressure = pressure_base + 5 * np.sin(2 * np.pi * t / 25) + 2 * np.sin(2 * np.pi * t / 60)
    flow = flow_base + 8 * np.sin(2 * np.pi * t / 20) + 3 * np.cos(2 * np.pi * t / 45)
    temp = temp_base + 2 * np.sin(2 * np.pi * t / 30) + 1.5 * np.cos(2 * np.pi * t / 50)
    volume = volume_base + 6 * np.sin(2 * np.pi * t / 35) + 2 * np.cos(2 * np.pi * t / 55)

    pressure += np.random.normal(0, NOISE, N_POINTS)
    flow += np.random.normal(0, NOISE * 1.5, N_POINTS)
    temp += np.random.normal(0, NOISE * 0.3, N_POINTS)
    volume += np.random.normal(0, NOISE * 2, N_POINTS)

    # Apply leak: continuous after LEAK_START
    leak_start_idx = int((LEAK_START / TOTAL_TIME) * N_POINTS)
    # leak continues to end (or until LEAK_START+LEAK_DURATION)
    leak_end_idx = min(N_POINTS, int(((LEAK_START + LEAK_DURATION) / TOTAL_TIME) * N_POINTS))
    for i in range(leak_start_idx, leak_end_idx):
        progress = (i - leak_start_idx) / (leak_end_idx - leak_start_idx + 1)
        pressure[i] -= 15 + 5 * progress
        flow[i] -= 20 + 10 * progress
        temp[i] += 3 + 2 * progress

    return t, pressure, flow, temp, volume

# -------------------- ANOMALY DETECTION (time‑aware) --------------------
def compute_severity(p_drop, f_drop, t_rise, z_p, z_f):
    score = 0
    if p_drop > 0: score += min(40, (p_drop / 20) * 40)
    if f_drop > 0: score += min(35, (f_drop / 30) * 35)
    if z_p > 3: score += min(15, (z_p / 6) * 15)
    if z_f > 3: score += min(10, (z_f / 6) * 10)
    if p_drop > 8 and f_drop > 15: score += 10
    return min(100, int(score))

def detect_at_index(pressure, flow, temp, idx, baseline_p, baseline_f):
    if idx < 10:
        return [], False, 0
    # Use last 50 points for statistics
    start = max(0, idx - 50)
    recent_p = pressure[start:idx+1]
    recent_f = flow[start:idx+1]
    curr_p = pressure[idx]
    curr_f = flow[idx]
    curr_t = temp[idx]
    p_drop = baseline_p - curr_p
    f_drop = baseline_f - curr_f

    if len(recent_p) > 1:
        p_mean, p_std = np.mean(recent_p), np.std(recent_p)
        f_mean, f_std = np.mean(recent_f), np.std(recent_f)
    else:
        p_mean, p_std = baseline_p, 1.0
        f_mean, f_std = baseline_f, 1.0

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
    if curr_t > TEMP_SETPOINT + 8:
        detected = True
        reasons.append("🌡️ Temperature too high")

    severity = compute_severity(p_drop, f_drop, curr_t - TEMP_SETPOINT, z_p, z_f) if detected else 0
    return reasons, detected, severity

# -------------------- MAIN APP --------------------
st.title("🛢️ Pipeline Leak Simulator")
st.markdown("Click **Start Simulation** to run a 120‑second scenario. A leak will appear **after 5 seconds** – watch the detection system respond!")

# Button to start
if st.button("▶️ Start Simulation", type="primary"):
    st.session_state.sim_running = True
    st.session_state.detection_log = []
    st.session_state.sim_data = None

# Placeholders for dynamic updates
plot_placeholder = st.empty()
metrics_placeholder = st.empty()
alert_placeholder = st.empty()
log_placeholder = st.empty()
progress_bar = st.progress(0)

# If simulation is running or data exists, update
if st.session_state.sim_running:
    # Generate data once
    if st.session_state.sim_data is None:
        t, p, f, temp, v = generate_simulation_data()
        st.session_state.sim_data = (t, p, f, temp, v)
    else:
        t, p, f, temp, v = st.session_state.sim_data

    # Run the time loop from 0 to TOTAL_TIME
    total_steps = 100  # update every ~1.2s
    step_time = TOTAL_TIME / total_steps

    for step in range(total_steps + 1):
        current_time = step * step_time
        if current_time > TOTAL_TIME:
            current_time = TOTAL_TIME

        idx = np.argmin(np.abs(t - current_time))
        curr_p = p[idx]
        curr_f = f[idx]
        curr_t = temp[idx]
        curr_v = v[idx]

        # Determine if we are in leak window
        in_leak = (LEAK_START <= current_time <= LEAK_START + LEAK_DURATION)
        if in_leak:
            reasons, detected, severity = detect_at_index(p, f, temp, idx, PRESSURE_SETPOINT, FLOW_SETPOINT)
            # Log if detected and not already logged
            if detected:
                log_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'time_sec': f"{current_time:.1f}",
                    'pressure_drop': f"{PRESSURE_SETPOINT - curr_p:.2f}",
                    'flow_drop': f"{FLOW_SETPOINT - curr_f:.2f}",
                    'severity': severity,
                    'reasons': '; '.join(reasons)
                }
                # avoid duplicates
                if not st.session_state.detection_log or st.session_state.detection_log[-1] != log_entry:
                    st.session_state.detection_log.append(log_entry)
        else:
            detected = False
            reasons = []
            severity = 0

        # ---- Update Plot ----
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        # Full traces
        axes[0].plot(t, p, color='blue', linewidth=1, alpha=0.6)
        axes[0].axhline(y=PRESSURE_SETPOINT, color='blue', linestyle='--', alpha=0.4)
        axes[0].scatter(current_time, curr_p, color='red', s=80, zorder=5)
        axes[0].set_ylabel('Pressure (psi)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, f, color='green', linewidth=1, alpha=0.6)
        axes[1].axhline(y=FLOW_SETPOINT, color='green', linestyle='--', alpha=0.4)
        axes[1].scatter(current_time, curr_f, color='red', s=80, zorder=5)
        axes[1].set_ylabel('Flow (m³/h)')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, temp, color='orange', linewidth=1, alpha=0.6)
        axes[2].axhline(y=TEMP_SETPOINT, color='orange', linestyle='--', alpha=0.4)
        axes[2].scatter(current_time, curr_t, color='red', s=80, zorder=5)
        axes[2].set_ylabel('Temp (°C)')
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(t, v, color='purple', linewidth=1, alpha=0.6)
        axes[3].axhline(y=VOLUME_SETPOINT, color='purple', linestyle='--', alpha=0.4)
        axes[3].scatter(current_time, curr_v, color='red', s=80, zorder=5)
        axes[3].set_xlabel('Time (seconds)')
        axes[3].set_ylabel('Volume (m³)')
        axes[3].grid(True, alpha=0.3)

        # Highlight leak period
        start_idx = int((LEAK_START / TOTAL_TIME) * len(t))
        end_idx = int(((LEAK_START + LEAK_DURATION) / TOTAL_TIME) * len(t))
        if end_idx > start_idx:
            for ax in axes:
                ax.axvspan(t[start_idx], t[end_idx-1], alpha=0.15, color='red')
        plt.tight_layout()
        plot_placeholder.pyplot(fig)
        plt.close(fig)

        # ---- Update Metrics ----
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Pressure", f"{curr_p:.1f} psi", f"{curr_p - PRESSURE_SETPOINT:.1f}")
            col2.metric("Flow", f"{curr_f:.1f} m³/h", f"{curr_f - FLOW_SETPOINT:.1f}")
            col3.metric("Temperature", f"{curr_t:.1f} °C", f"{curr_t - TEMP_SETPOINT:.1f}")

        # ---- Update Alert ----
        with alert_placeholder.container():
            if detected:
                st.markdown(f"""
                <div class="custom-alert" style="background-color:#b71c1c;color:white;padding:15px;border-radius:10px;border:3px solid #ff1744;">
                    <h4 style="color:white;margin:0;">🚨 ANOMALY DETECTED!</h4>
                    <p style="margin:5px 0;">Severity: <strong>{severity}/100</strong></p>
                    <p style="margin:5px 0;">Reasons: {'; '.join(reasons)}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ System operating normally.")

        # ---- Progress Bar ----
        progress_bar.progress(min(1.0, current_time / TOTAL_TIME))

        # Sleep for smooth animation
        time.sleep(0.4)  # adjust for speed

        # If we reached the end, stop
        if current_time >= TOTAL_TIME:
            st.session_state.sim_running = False
            st.success("✅ Simulation finished.")
            break

    # After loop, show final log
    with log_placeholder.container():
        st.subheader("📋 Detection Log")
        if st.session_state.detection_log:
            st.dataframe(pd.DataFrame(st.session_state.detection_log), use_container_width=True)
        else:
            st.write("No anomalies detected.")
else:
    # Not running – show placeholder or previous log
    if st.session_state.detection_log:
        st.subheader("📋 Detection Log")
        st.dataframe(pd.DataFrame(st.session_state.detection_log), use_container_width=True)
    else:
        st.info("Click 'Start Simulation' to begin.")

# Add export button if log exists
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