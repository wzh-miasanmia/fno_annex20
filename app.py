"""Annex 20 browser demo backed by the two trained CPU inference models."""
import base64
import hashlib
import io
import json
import logging
from pathlib import Path
import threading

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np
import streamlit as st
import torch

from inference import load_concentration, load_velocity, make_xy, predict_hybrid

ROOT = Path(__file__).resolve().parent
DEFAULT_SCENARIO = (0.30, 4.50, 1.50)


def checkpoint(manifest, name):
    record = manifest[name]
    direct = ROOT / record["file"]
    if direct.is_file():
        data = direct.read_bytes()
    else:
        encrypted = (ROOT / record["encrypted_file"]).read_bytes()
        key = base64.b64decode(st.secrets["ANNEX20_MODEL_KEY"], validate=True)
        data = AESGCM(key).decrypt(encrypted[:12], encrypted[12:], name.encode("ascii"))
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"{name} checkpoint does not match the deployment manifest.")
    return io.BytesIO(data)


@st.cache_resource
def load_models():
    torch.set_num_threads(2)
    manifest = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    velocity = load_velocity(checkpoint(manifest, "velocity"))
    concentration = load_concentration(checkpoint(manifest, "concentration"))
    return velocity, concentration, threading.Lock()


@st.cache_data(max_entries=24, show_spinner=False)
def prediction(inlet, x, y):
    velocity, concentration, lock = load_models()
    with lock:
        fields = predict_hybrid(velocity, concentration, inlet, x, y)
    if not all(np.isfinite(values).all() for values in fields):
        raise ValueError("The model returned a nonfinite field.")
    return fields, concentration[4]


def field_plot(values, label, cmap, x, y, meta, vectors=None):
    lx, ly = float(meta["Lx"]), float(meta["Ly"])
    fig = Figure(figsize=(9, 3.3), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    image = ax.imshow(values, origin="lower", extent=(0, lx, 0, ly),
                      aspect="equal", cmap=cmap, interpolation="nearest")
    half_size = float(meta["src_size"]) / 2
    ax.add_patch(Rectangle((x - half_size, y - half_size), 2 * half_size, 2 * half_size,
                           fill=False, edgecolor="white", linewidth=1.5))
    if vectors is not None:
        grid_x, grid_y = make_xy(lx, ly, *values.shape)
        sparse = np.s_[::10, ::16]
        ax.quiver(grid_x[sparse], grid_y[sparse], vectors[0][sparse], vectors[1][sparse],
                  color="white", alpha=0.75, angles="xy", width=0.002)
    ax.set(xlim=(0, lx), ylim=(0, ly), xlabel="x (m)", ylabel="y (m)")
    fig.colorbar(image, ax=ax, label=label, shrink=0.83, pad=0.02)
    return fig


st.set_page_config(page_title="Annex 20 | Flow & concentration", page_icon="🌬️", layout="wide")
st.title("Annex 20 · Flow and concentration")
st.write("Explore how inlet velocity and source position change the predicted fields.")

with st.sidebar:
    st.header("Scenario")
    with st.form("scenario"):
        inlet = st.slider("Inlet velocity (m/s)", 0.05, 1.70, DEFAULT_SCENARIO[0], step=0.01)
        source_x = st.slider("Source x (m)", 0.50, 8.50, DEFAULT_SCENARIO[1], step=0.05)
        source_y = st.slider("Source y (m)", 0.50, 2.50, DEFAULT_SCENARIO[2], step=0.05)
        run = st.form_submit_button("Run prediction", width="stretch", type="primary")
    st.header("Display")
    vectors = st.toggle("Show velocity vectors", value=False)
    clip_concentration = st.toggle("Hide negative concentrations in the plot", value=True)

if "applied_scenario" not in st.session_state:
    st.session_state.applied_scenario = DEFAULT_SCENARIO
if run:
    st.session_state.applied_scenario = (inlet, source_x, source_y)

inlet, source_x, source_y = st.session_state.applied_scenario
with st.spinner("Computing the velocity and concentration fields…"):
    try:
        (ux, uy, speed, concentration), geometry = prediction(inlet, source_x, source_y)
    except (OSError, RuntimeError, ValueError):
        logging.exception("Annex 20 prediction failed")
        st.error("Prediction could not be completed. Please try again or contact the demo owner.")
        st.stop()

st.caption(f"Prediction at inlet {inlet:.2f} m/s · source ({source_x:.2f}, {source_y:.2f}) m")
velocity_tab, concentration_tab = st.tabs(("Velocity", "Concentration"), default="Concentration")
with velocity_tab:
    field = st.radio("Velocity field", ("Speed", "Horizontal velocity", "Vertical velocity"), horizontal=True)
    values = {"Speed": speed, "Horizontal velocity": ux, "Vertical velocity": uy}[field]
    colour = "viridis" if field == "Speed" else "RdBu_r"
    st.pyplot(field_plot(values, f"{field} (m/s)", colour, source_x, source_y, geometry,
                         (ux, uy) if vectors else None))
with concentration_tab:
    values = np.maximum(concentration, 0) if clip_concentration else concentration
    st.pyplot(field_plot(values, "Concentration (CFD training units)", "magma",
                         source_x, source_y, geometry))
st.caption("Research demo. The FNO predicts velocity first, then the FNO–CNN predicts concentration.")
