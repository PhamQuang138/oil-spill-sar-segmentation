"""Streamlit app for SAR oil-spill segmentation.

Run:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.base import BasePredictor
from src.inference.ensemble import EnsemblePredictor
from src.inference.hung_adapter import HungPredictor
from src.inference.khoa_adapter import KhoaPredictor
from src.inference.postprocess import postprocess_mask
from src.inference.quang_adapter import QuangPredictor


MODEL_CONFIGS = [
    {
        "name": "Quang (DeepLabV3+)",
        "path": PROJECT_ROOT / "weights" / "quang_best_deeplabv3plus_checkpoint.pth",
        "factory": QuangPredictor,
        "threshold": 0.50,
    },
    {
        "name": "Hung (SegFormer)",
        "path": PROJECT_ROOT / "weights" / "Hung_best_segformer_checkpoint.pth",
        "factory": HungPredictor,
        "threshold": 0.60,
    },
    {
        "name": "Khoa (UNet++ EfficientNet-B4)",
        "path": PROJECT_ROOT / "weights" / "Khoa_best_Unetplusplus_checkpoint.pth",
        "factory": KhoaPredictor,
        "threshold": 0.65,
    },
]

ENSEMBLE_NAME = "Ensemble (Trung binh xac suat)"
DEFAULT_THRESHOLDS = {
    "Quang (DeepLabV3+)": 0.50,
    "Hung (SegFormer)": 0.60,
    "Khoa (UNet++ EfficientNet-B4)": 0.65,
    ENSEMBLE_NAME: 0.50,
}


st.set_page_config(page_title="SAR Oil Spill Detection", layout="wide")


@st.cache_resource(show_spinner=False)
def load_models(device: str):
    """Load available single models, then create ensemble from loaded models."""

    models = {}
    load_messages = []

    for config in MODEL_CONFIGS:
        weight_path = config["path"]
        model_name = config["name"]

        if not weight_path.exists():
            load_messages.append((model_name, "missing", f"Khong thay file: {weight_path.name}"))
            continue

        try:
            models[model_name] = config["factory"](str(weight_path), device)
            load_messages.append((model_name, "ok", "Da load"))
        except Exception as exc:  # pragma: no cover - shown in Streamlit UI
            load_messages.append((model_name, "error", str(exc)))

    single_models = list(models.values())
    if len(single_models) >= 2:
        models[ENSEMBLE_NAME] = EnsemblePredictor(single_models)
        load_messages.append((ENSEMBLE_NAME, "ok", f"Ket hop {len(single_models)} model"))

    return models, load_messages


st.sidebar.title("Cau hinh suy luan")

if st.sidebar.button("Reload models / clear cache"):
    st.cache_resource.clear()
    st.rerun()

device_option = st.sidebar.selectbox("Thiet bi tinh toan", ["Auto", "CPU", "CUDA"])
if device_option == "Auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = device_option.lower()

with st.spinner("Dang load model..."):
    available_models, load_messages = load_models(device)

with st.sidebar.expander("Trang thai model", expanded=False):
    for model_name, status, message in load_messages:
        if status == "ok":
            st.success(f"{model_name}: {message}")
        elif status == "missing":
            st.warning(f"{model_name}: {message}")
        else:
            st.error(f"{model_name}: {message}")

if not available_models:
    st.error("Khong tim thay model nao load thanh cong. Hay kiem tra thu muc weights/ va dependency.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Chon mo hinh", list(available_models.keys()))
default_threshold = DEFAULT_THRESHOLDS.get(selected_model_name, 0.50)
threshold = st.sidebar.slider(
    "Nguong xac suat (threshold)",
    0.0,
    1.0,
    default_threshold,
    0.05,
    key=f"threshold_{selected_model_name}",
)
use_postprocess = st.sidebar.checkbox("Hau xu ly mask", value=True)
min_area = st.sidebar.number_input("Min area connected component", 0, 10000, 64, 16)
open_kernel = st.sidebar.selectbox("Opening kernel", [0, 3, 5, 7], index=1)
close_kernel = st.sidebar.selectbox("Closing kernel", [0, 3, 5, 7], index=2)
alpha = st.sidebar.slider("Do dam overlay", 0.0, 1.0, 0.4, 0.1)


st.title("SAR Oil Spill Detection App")
st.markdown(f"Mo hinh: **{selected_model_name}** | Device: **{device.upper()}**")

uploaded_file = st.file_uploader("Tai len anh SAR (png, jpg, tif)", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    if st.button("Chay nhan dien", use_container_width=True, type="primary"):
        with st.spinner("Dang phan tich anh ve tinh..."):
            predictor = available_models[selected_model_name]
            prob_map, raw_mask = predictor.predict(image_np, threshold=threshold)

            if use_postprocess:
                mask = postprocess_mask(
                    raw_mask,
                    min_area=int(min_area),
                    open_kernel=int(open_kernel),
                    close_kernel=int(close_kernel),
                )
            else:
                mask = raw_mask.astype(np.uint8)

            overlay = BasePredictor.generate_overlay(image_np, mask, alpha=alpha)
            oil_percent = mask.mean() * 100

        st.success(
            f"Hoan tat. Vung nghi dau chiem khoang {oil_percent:.2f}% dien tich khung hinh "
            f"(threshold={threshold:.2f}, postprocess={'on' if use_postprocess else 'off'})."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image_np, caption=f"1. Anh goc ({image_np.shape[1]}x{image_np.shape[0]})", use_container_width=True)
        with col2:
            st.image(mask * 255, caption=f"2. Mask ({oil_percent:.2f}% oil)", use_container_width=True)
        with col3:
            st.image(overlay, caption="3. Overlay", use_container_width=True)

        with st.expander("Xem probability map va raw mask"):
            prob_vis = np.clip(prob_map * 255, 0, 255).astype(np.uint8)
            raw_oil_percent = raw_mask.mean() * 100
            c1, c2 = st.columns(2)
            with c1:
                st.image(prob_vis, caption="Probability map", use_container_width=True)
            with c2:
                st.image(raw_mask * 255, caption=f"Raw mask truoc hau xu ly ({raw_oil_percent:.2f}% oil)", use_container_width=True)
