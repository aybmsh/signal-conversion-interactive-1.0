
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")
st.title("Analog to Digital Signal Conversion – Interactive Sample")

@st.cache_data
def load_original():
    return pd.read_csv("signal_original_20ms.csv")

df_full = load_original()

# Sidebar controls
st.sidebar.header("Controls")
sampling_rate = st.sidebar.slider("Sampling Rate (Hz)", 400, 44100, 8000, step=100)
bit_depth = st.sidebar.slider("Bit Depth", 2, 8, 8)

show_original = st.sidebar.checkbox("Show Original Signal", value=True)
show_sampled = st.sidebar.checkbox("Show Sampled Signal", value=True)
show_quantized = st.sidebar.checkbox("Show Quantized Signal", value=True)
show_reconstructed = st.sidebar.checkbox("Show Reconstructed Signal", value=True)
show_smoothed = st.sidebar.checkbox("Show Smoothed Reconstruction", value=False)

# Data and interpolation
time = df_full["time"].to_numpy()
ampl = df_full["amplitude"].to_numpy()
duration = time[-1] - time[0]
num_samples = int(duration * sampling_rate)
sample_times = np.linspace(time[0], time[-1], num_samples)
interpolator = interp1d(time, ampl, kind='linear')
sampled_ampl = interpolator(sample_times)

# Quantization
levels = np.linspace(-1, 1, 2 ** bit_depth)
quantized_ampl = [min(levels, key=lambda l: abs(val - l)) for val in sampled_ampl]
binary_codes = [format(levels.tolist().index(q), f'0{bit_depth}b') for q in quantized_ampl]

# Reconstructed stair-step signal
stair_x = []
stair_y = []
for i in range(len(sample_times)):
    stair_x.append(sample_times[i])
    stair_y.append(quantized_ampl[i])
    if i < len(sample_times) - 1:
        stair_x.append(sample_times[i+1])
        stair_y.append(quantized_ampl[i])

# Smoothed reconstruction (interpolated)
smooth_interp = interp1d(sample_times, quantized_ampl, kind='cubic')
smooth_time = np.linspace(sample_times[0], sample_times[-1], 1000)
smooth_ampl = smooth_interp(smooth_time)

# Plotting
fig = go.Figure()
if show_original:
    fig.add_trace(go.Scatter(x=time, y=ampl, name="Original Signal", line=dict(color="lightgray")))
if show_sampled:
    fig.add_trace(go.Scatter(x=sample_times, y=sampled_ampl, mode="markers",
                             name="Sampled Signal", marker=dict(color="blue")))
if show_quantized:
    fig.add_trace(go.Scatter(x=sample_times, y=quantized_ampl, mode="markers+lines",
                             name=f"Quantized ({bit_depth} bits)",
                             line=dict(shape="hv", color="green"), marker=dict(color="green")))
if show_reconstructed:
    fig.add_trace(go.Scatter(x=stair_x, y=stair_y, mode="lines",
                             name="Reconstructed Signal (ZOH)", line=dict(color="red")))
if show_smoothed:
    fig.add_trace(go.Scatter(x=smooth_time, y=smooth_ampl, mode="lines",
                             name="Smoothed Reconstruction", line=dict(color="orange", dash="dot")))

fig.update_layout(title="Signal Digitization Process",
                  xaxis_title="Time (s)", yaxis_title="Amplitude",
                  yaxis=dict(range=[-1.1, 1.1]), height=500)
st.plotly_chart(fig, use_container_width=True)
