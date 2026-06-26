"""Post-processing utilities for binary oil-spill masks."""

from __future__ import annotations

import cv2
import numpy as np


def postprocess_mask(
    mask: np.ndarray,
    min_area: int = 64,
    open_kernel: int = 3,
    close_kernel: int = 5,
) -> np.ndarray:
    """Clean a binary mask with morphology and small-component removal.

    Args:
        mask: Binary mask with values 0/1 or 0/255.
        min_area: Connected components smaller than this are removed.
        open_kernel: Kernel size for opening. Set 0 to disable.
        close_kernel: Kernel size for closing. Set 0 to disable.

    Returns:
        Binary uint8 mask with values 0/1.
    """

    binary = (mask > 0).astype(np.uint8)

    if open_kernel and open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if close_kernel and close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 1

    return cleaned
