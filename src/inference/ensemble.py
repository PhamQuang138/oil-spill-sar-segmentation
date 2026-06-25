"""
ensemble.py — Adapter kết hợp các mô hình (Averaging Probability Maps)
"""
import numpy as np
from .base import BasePredictor

class EnsemblePredictor:
    def __init__(self, predictors_list):
        """Nhận vào danh sách các object Predictor đã được khởi tạo"""
        self.predictors = predictors_list

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        prob_maps = []
        for predictor in self.predictors:
            prob = predictor.predict_proba(image)
            prob_maps.append(prob)
        
        # Tính trung bình cộng của tất cả probability maps
        avg_prob = np.mean(prob_maps, axis=0)
        return avg_prob

    def predict(self, image: np.ndarray, threshold: float = 0.5):
        avg_prob = self.predict_proba(image)
        mask = (avg_prob >= threshold).astype(np.uint8)
        return avg_prob, mask
