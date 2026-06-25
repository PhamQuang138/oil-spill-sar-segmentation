"""
quang_adapter.py — Adapter cho mô hình DeepLabV3+ của Quang
Yêu cầu: pip install segmentation-models-pytorch
"""
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from .base import BasePredictor

class QuangPredictor(BasePredictor):
    def load_model(self):
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        ).to(self.device)
        
        checkpoint = torch.load(self.weight_path, map_location=self.device)
        # Hỗ trợ nạp cả khi dict lưu dạng {"model_state_dict": ...} hoặc lưu trực tiếp
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        # Resize về 256x256
        img_resized = cv2.resize(image, (256, 256))
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        
        # (H, W, C) -> (C, H, W) -> Thêm batch size (1, C, H, W)
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs
