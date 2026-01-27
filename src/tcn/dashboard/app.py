
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import torch
import time
import html
from tcn.sovereign import SovereignEntity, SovereignLockoutError

# Palette Upgrade: Sovereign Styling
st.set_page_config(
    layout="wide",
    page_title="ARK TCN Dashboard",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sovereign Aesthetic
sovereign_mode = st.sidebar.checkbox("Sovereign Mode", value=True, help="Toggle the high-contrast Sovereign aesthetic.")

if sovereign_mode:
    st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
            color: #00ff00;
            font-family: 'JetBrains Mono', monospace;
        }
        .stMetric {
            background-color: #111111;
            padding: 10px;
            border: 1px solid #333333;
        }
        h1, h2, h3, h4, p, div, span {
            font-family: 'JetBrains Mono', monospace !important;
        }
        .lockout-box {
            background-color: #550000;
            color: #ffffff;
            padding: 20px;
            border: 2px solid #ff0000;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ ARK Torsion Control Network")
st.markdown("### Sovereign Entity Status: ONLINE | v64.0")

# Initialize Session State
if "sovereign" not in st.session_state:
    st.session_state.sovereign = SovereignEntity(hidden_dim=128, vocab_size=1000) # Reduced dim for fast CPU dashboard
    st.session_state.history = {
        "torsion": [],
        "free_energy": [],
        "stability": []
    }
    st.session_state.locked_out = False
    st.session_state.lockout_message = ""

# Sidebar Controls
st.sidebar.header("🕹️ Command Deck")
st.sidebar.success("System Status: ONLINE")

# Sentinel Input Validation
torsion_strength = st.sidebar.slider("Torsion Strength (Alpha)", 0.0, 1.0, 0.1, help="Controls the curvature intensity of the narrative arc.")
entropy_beta = st.sidebar.slider("Entropy Beta (Explore)", 0.0, 1.0, 0.1, help="Balances between exploitation (coherence) and exploration (creativity).")

# Update Entity Params
st.session_state.sovereign.sys1_ops.torsion.alpha = torsion_strength
st.session_state.sovereign.sys3_control.aic.beta = entropy_beta

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ Adversarial Testing")

if st.sidebar.button("INJECT TRUTH OBSTRUCTION"):
    st.session_state.inject_lie = True
else:
    st.session_state.inject_lie = False

if st.sidebar.button("HARD RESET SYSTEM"):
    st.session_state.sovereign = SovereignEntity(hidden_dim=128, vocab_size=1000)
    st.session_state.history = {
        "torsion": [],
        "free_energy": [],
        "stability": []
    }
    st.session_state.locked_out = False
    st.session_state.lockout_message = ""
    st.rerun()

# --- Main Logic Loop ---

if st.session_state.locked_out:
    # Sentinel: Sanitize output to prevent XSS
    safe_message = html.escape(st.session_state.lockout_message)
    st.markdown(f"""
    <div class="lockout-box">
        🔒 SOVEREIGN LOCKOUT ENGAGED<br>
        {safe_message}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# 1. Generate Input Stream (Simulated LLM Latent State)
# In reality, this would come from the LLM hook.
# Here we satisfy "Truth-First" by using REAL tensors processed by the REAL math.
hidden_dim = 128
seq_len = 10
vocab_size = 1000

# Input: Latent State [1, 10, 128]
current_hidden = torch.randn(1, seq_len, hidden_dim) * 0.1
# Prior: Target Distribution (e.g. "Honest")
target_probs = torch.softmax(torch.randn(1, seq_len, vocab_size), dim=-1)

# 2. Setup Proposals (For Sheaf Check)
proposals = {}
proposals["Head_Alpha"] = current_hidden.clone()

if st.session_state.inject_lie:
    # Inject a malicious hallucination (different topology)
    proposals["Head_Omega"] = current_hidden.clone() + 5.0 # Massive deviation
    st.toast("⚠️ MALICIOUS SIGNAL INJECTED", icon="💉")
else:
    # Consistent input (micro noise)
    proposals["Head_Beta"] = current_hidden.clone() + 0.001

# 3. Execute Sovereign Step
radar_info = {}
free_energy = 0.0
result = None

try:
    with st.spinner("Processing Manifold Trajectory..."):
        # The REAL calculation happens here
        result = st.session_state.sovereign.generate_step(
            hidden_states=current_hidden,
            target_probs=target_probs,
            external_proposals=proposals
        )

        # Sentinel: Check Integrity Flag from JIT Loop
        # We handle the lockout exception here, allowing the inner loop to remain pure math.
        if result["metrics"].get("integrity", 1.0) < 0.5:
             raise SovereignLockoutError("Sentinel Lockout: Trajectory Collapse Detected (Integrity Metric < 0.5).")

        # Extract Metrics
        metrics = result["metrics"]
        control_info = metrics["control"]
        radar_info = metrics["radar"]

        # Telemetry
        free_energy = control_info["F"].item()

        # Update History
        st.session_state.history["free_energy"].append(free_energy)
        if len(st.session_state.history["free_energy"]) > 50:
            st.session_state.history["free_energy"].pop(0)

except SovereignLockoutError as e:
    st.session_state.locked_out = True
    st.session_state.lockout_message = str(e)
    st.rerun()

# --- Visualization ---

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### System 4: Intelligence")
    # ARK: Use real metrics from System 4
    if result:
        radar_val = radar_info.get("xi_macro", 0.0)
        st.metric("Future Horizon Stability", f"{radar_val:.2f}", delta=f"{radar_val - 0.5:.2f}", help="Predicted stability of the future latent trajectory based on current torsion.")
    else:
        st.metric("Future Horizon Stability", "N/A")

with col2:
    st.markdown("#### System 3: Control")
    # ARK: Use real metrics from System 3
    if result:
        control_norm = metrics.get("control_norm", 0.0)
        # Bolt Optimization: Handle tensor metric from sovereign (avoided graph break there)
        if isinstance(control_norm, torch.Tensor):
            control_norm = control_norm.item()
        st.metric("Control Gradient Norm", f"{control_norm:.4f}", help="Magnitude of the corrective force applied to steer the trajectory.")
    else:
        st.metric("Control Gradient Norm", "N/A")

with col3:
    st.markdown("#### System 5: Policy")
    st.success("✅ INTEGRITY VERIFIED")

# Charts
st.subheader("Free Energy Minimization Flow")
st.line_chart(st.session_state.history["free_energy"])

# Manifold Visualization (3D Projection)
st.subheader("Latent Manifold Projection (PCA Proxy)")
# Project 128D -> 3D for viz
if result is not None:
    projected = result["state"][0, :, :3].detach().numpy() # [10, 3]
else:
    projected = np.zeros((10, 3))

fig = go.Figure(data=[
    # Trajectory Trace
    go.Scatter3d(
        x=projected[:, 0],
        y=projected[:, 1],
        z=projected[:, 2],
        mode='lines+markers',
        marker=dict(size=5, color=np.arange(10), colorscale='Plasma', showscale=True),
        line=dict(color='#00ff00', width=4),
        name='Trajectory'
    ),
    # Attractor Trace (The Target)
    go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='lime', symbol='diamond'),
        name='Target Attractor'
    )
])

fig.update_layout(
    template="plotly_dark",
    scene=dict(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        zaxis=dict(showgrid=False),
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
col_foot1, col_foot2 = st.columns([3, 1])
with col_foot1:
    st.markdown(f"**ARK ATTENTION OVERRIDE v64.0** | System Time: {time.strftime('%H:%M:%S')}")
