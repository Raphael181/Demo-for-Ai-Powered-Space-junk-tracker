
import streamlit as st
import numpy as np
import pickle
import json
import plotly.graph_objects as go

# -----------------------------
# Function: load_model
# -----------------------------
# Loads the RandomForestClassifier from file.
@st.cache_resource
def load_model():
    with open("rf_collision_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# -----------------------------
# Function: visualize_orbits
# -----------------------------
# Plots two simple orbits as circles for visualization.
# Not physically accurate, but useful for hackathon demo.
def visualize_orbits(orbit1, orbit2):
    theta = np.linspace(0, 2*np.pi, 200)

    # Orbit 1 (assume circular for simplicity)
    x1 = orbit1["a"] * np.cos(theta)
    y1 = orbit1["a"] * np.sin(theta)
    z1 = np.zeros_like(theta)

    # Orbit 2 (with inclination tilt)
    x2 = orbit2["a"] * np.cos(theta)
    y2 = orbit2["a"] * np.sin(theta) * np.cos(np.radians(orbit2["i"]))
    z2 = orbit2["a"] * np.sin(theta) * np.sin(np.radians(orbit2["i"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode="lines", name="Orbit 1"))
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode="lines", name="Orbit 2"))

    fig.update_layout(
        title="Simplified Orbit Visualization",
        scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"),
        width=700, height=500
    )
    return fig

# -----------------------------
# Main App
# -----------------------------
st.title("🛰️ AI-Powered Space Junk Tracker")
st.markdown("Predict potential orbital collisions with a simple ML model.")

model = load_model()

# Sidebar inputs
st.sidebar.header("Input Orbital Parameters")
a1 = st.sidebar.slider("Semi-major axis Orbit 1 (km)", 6500, 8000, 7000)
e1 = st.sidebar.slider("Eccentricity Orbit 1", 0.0, 0.1, 0.01)
i1 = st.sidebar.slider("Inclination Orbit 1 (deg)", 0, 180, 45)
a2 = st.sidebar.slider("Semi-major axis Orbit 2 (km)", 6500, 8000, 7100)
e2 = st.sidebar.slider("Eccentricity Orbit 2", 0.0, 0.1, 0.02)
i2 = st.sidebar.slider("Inclination Orbit 2 (deg)", 0, 180, 60)
v_rel = st.sidebar.slider("Relative velocity (km/s)", 0.0, 15.0, 7.5)
d_min = st.sidebar.slider("Minimum approach distance (km)", 0.0, 10.0, 5.0)

# Build input vector
features = np.array([[a1, e1, i1, v_rel, d_min]])

# Prediction
prediction = model.predict(features)[0]
prob = model.predict_proba(features)[0][1]

st.subheader("🔮 Collision Prediction")
if prediction == 1:
    st.error(f"⚠️ Potential Collision Risk! (Probability: {prob:.2f})")
else:
    st.success(f"✅ Safe (Probability of collision: {prob:.2f})")

# Orbit visualization
orbit1 = {"a": a1, "e": e1, "i": i1}
orbit2 = {"a": a2, "e": e2, "i": i2}
st.plotly_chart(visualize_orbits(orbit1, orbit2))
