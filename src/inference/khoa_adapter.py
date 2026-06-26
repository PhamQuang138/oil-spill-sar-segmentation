"""
khoa_adapter.py - Adapter for Khoa's UNet++ EfficientNet-B4 model.
"""
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from .base import BasePredictor


class KhoaPredictor(BasePredictor):
    def load_model(self):
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",
            activation=None,
        ).to(self.device)

        checkpoint = torch.load(self.weight_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std

        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs
