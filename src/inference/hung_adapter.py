"""
hung_adapter.py - Adapter for Hung's SegFormer model.
"""
import cv2
import numpy as np
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from .base import BasePredictor


class HungPredictor(BasePredictor):
    def load_model(self):
        config = SegformerConfig(num_labels=2)
        self.model = SegformerForSemanticSegmentation(config).to(self.device)

        checkpoint = torch.load(self.weight_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if state_dict and all(key.startswith("model.") for key in state_dict):
            state_dict = {
                key.removeprefix("model."): value
                for key, value in state_dict.items()
            }

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        img_resized = cv2.resize(image, (256, 256))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std

        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(image)
        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)
            logits = outputs.logits

            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )

            probs = torch.softmax(upsampled_logits, dim=1)[0, 1].cpu().numpy()
        return probs
