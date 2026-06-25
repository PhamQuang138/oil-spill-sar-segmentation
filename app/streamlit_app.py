"""
streamlit_app.py — Giao diện chính của ứng dụng
Chạy lệnh: streamlit run app/streamlit_app.py
"""
import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import sys
from pathlib import Path

# Thêm root dự án vào sys.path để import được src
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.base import BasePredictor
from src.inference.quang_adapter import QuangPredictor
from src.inference.hung_adapter import HungPredictor
from src.inference.ensemble import EnsemblePredictor

st.set_page_config(page_title="SAR Oil Spill Detection", layout="wide")

# ── Caching Models (Chỉ load 1 lần vào RAM/VRAM) ──
@st.cache_resource
def load_models(device):
    models = {}
    
    # 1. Load model Quang
    quang_path = PROJECT_ROOT / "weights" / "quang_best_deeplabv3plus_checkpoint.pth"
    if quang_path.exists():
        try:
            models["Quang (DeepLabV3+)"] = QuangPredictor(str(quang_path), device)
        except Exception as e:
            st.sidebar.error(f"Lỗi load model Quang: {e}")

    # 2. Load model Hung
    hung_path = PROJECT_ROOT / "weights" / "Hung_best_segformer_checkpoint.pth"
    if hung_path.exists():
        try:
            models["Hung (SegFormer)"] = HungPredictor(str(hung_path), device)
        except Exception as e:
            st.sidebar.error(f"Lỗi load model Hung: {e}")
            
    # Model Khoa (Placeholder)
    # khoa_path = PROJECT_ROOT / "weights" / "khoa_best_model.pth"
    
    # 3. Ensemble (Chỉ tạo nếu có ít nhất 2 model)
    if len(models) >= 2:
        models["Ensemble (Trung bình cộng)"] = EnsemblePredictor(list(models.values()))
        
    return models

# ── Giao diện Sidebar ──
st.sidebar.title("⚙️ Cấu hình suy luận")

device_option = st.sidebar.selectbox("Thiết bị tính toán (Device)", ["Auto", "CPU", "CUDA"])
if device_option == "Auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = device_option.lower()

available_models = load_models(device)

if not available_models:
    st.error("⚠️ Không tìm thấy file trọng số (weights) nào! Vui lòng kiểm tra thư mục `weights/`.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Chọn Mô hình", list(available_models.keys()))
threshold = st.sidebar.slider("Ngưỡng xác suất (Threshold)", 0.0, 1.0, 0.5, 0.05)
alpha = st.sidebar.slider("Độ đậm Overlay (Alpha)", 0.0, 1.0, 0.4, 0.1)

# ── Main UI ──
st.title("🌊 SAR Oil Spill Detection App")
st.markdown(f"Đang sử dụng mô hình: **{selected_model_name}** | Chạy trên: **{device.upper()}**")

uploaded_file = st.file_uploader("Tải lên ảnh SAR (png, jpg, tif)", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    # Đọc ảnh gốc bằng PIL -> Numpy RGB
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    
    if st.button("🚀 Chạy Nhận Diện", use_container_width=True, type="primary"):
        with st.spinner('Đang phân tích ảnh vệ tinh...'):
            predictor = available_models[selected_model_name]
            
            # Predict
            prob_map, mask = predictor.predict(image_np, threshold=threshold)
            
            # Tạo overlay dùng hàm chung của BasePredictor
            overlay = BasePredictor.generate_overlay(image_np, mask, alpha=alpha)
            
            # Tính toán %
            oil_percent = (mask.mean() * 100)
            
            # Hiển thị
            st.success(f"Hoàn tất! Dự đoán vết dầu loang chiếm khoảng **{oil_percent:.2f}%** diện tích khung hình.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image_np, caption=f"1. Ảnh gốc ({image_np.shape[1]}x{image_np.shape[0]})", use_container_width=True)
            with col2:
                st.image(mask * 255, caption=f"2. Mask (Dầu chiếm {oil_percent:.2f}%)", use_container_width=True)
            with col3:
                st.image(overlay, caption="3. Overlay", use_container_width=True)
