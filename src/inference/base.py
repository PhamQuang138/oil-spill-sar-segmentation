"""
base.py — Interface chuẩn cho tất cả các mô hình trong dự án.
Quy định rõ Input và Output để Streamlit gọi chung một cách dễ dàng.
"""
from abc import ABC, abstractmethod
import numpy as np
import cv2

class BasePredictor(ABC):
    def __init__(self, weight_path: str, device: str = "cpu"):
        self.weight_path = weight_path
        self.device = device
        self.model = None
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Khởi tạo kiến trúc và nạp trọng số (.pth)"""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> object:
        """Nhận ảnh RGB (numpy) -> Trả về Tensor chuẩn bị đưa vào model"""
        pass

    @abstractmethod
    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """
        Input: Ảnh RGB numpy array (HxWx3)
        Output: Ma trận xác suất (Probability Map) dạng numpy array (HxW), giá trị [0, 1]
        """
        pass

    def predict(self, image: np.ndarray, threshold: float = 0.5):
        """Hàm chuẩn được gọi bởi Streamlit"""
        prob_map = self.predict_proba(image)
        mask = (prob_map >= threshold).astype(np.uint8)
        return prob_map, mask

    @staticmethod
    def generate_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Hàm dùng chung để tạo ảnh đè (Overlay) màu đỏ lên vùng dầu loang"""
        # Đảm bảo ảnh gốc và mask cùng kích thước
        h, w = mask.shape
        img_resized = cv2.resize(image, (w, h))
        
        overlay = img_resized.copy()
        overlay[mask == 1] = [255, 50, 50] # Tô đỏ vùng mask
        
        blended = cv2.addWeighted(img_resized, 1 - alpha, overlay, alpha, 0)
        return blended