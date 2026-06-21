"""
predict.py — Inference với API chuẩn hóa cho SegFormer oil spill segmentation.

Public API:
    predict(image_path, img_size)           → np.ndarray (H, W) mask
    predict_batch(image_dir, img_size)      → List[np.ndarray]
    predict_with_overlay(image_path, img_size) → np.ndarray (H, W, 3) RGB

Command line:
    python predict.py --input path/to/image.png
    python predict.py --input path/to/folder/ --output predictions/
    python predict.py --input image.png --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.data.preprocessing import (
    preprocess_for_inference,
    preprocess_batch,
    postprocess_mask,
    load_sar_image,
)
from src.model.segformer import build_model, OilSpillSegFormer
from src.utils.checkpoint import CheckpointManager
from src.utils.visualization import overlay_prediction


# ── Module-level singleton (lazy init) ───────────────────────────────────────
_model: Optional[OilSpillSegFormer] = None
_device: Optional[torch.device] = None


def _get_model(
    checkpoint_path: Union[str, Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[OilSpillSegFormer, torch.device]:
    """
    Lazy-load model vào bộ nhớ (tránh reload nhiều lần khi gọi predict() liên tiếp).

    Args:
        checkpoint_path: Đường dẫn tới file .pth. Mặc định: best_model.pth.
        device:          torch.device. Mặc định: CUDA nếu có, else CPU.

    Returns:
        (model, device) tuple.
    """
    global _model, _device

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"

    checkpoint_path = Path(checkpoint_path)

    if _model is None:
        _model = build_model(pretrained=False)
        ckpt_mgr = CheckpointManager(config.CHECKPOINT_DIR)
        ckpt_mgr.load(_model, checkpoint_path, device=device)
        _model.to(device).eval()
        _device = device

    return _model, _device


# ── Public API ────────────────────────────────────────────────────────────────

def predict(
    image_path: Union[str, Path],
    img_size: Tuple[int, int] = config.IMG_SIZE,
    checkpoint_path: Union[str, Path, None] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Dự đoán binary mask vết dầu loang từ một ảnh SAR.

    Args:
        image_path:      Đường dẫn tới file PNG/JPG của ảnh SAR.
        img_size:        (H, W) kích thước resize trước khi đưa vào model.
                         Mặc định: config.IMG_SIZE = (512, 512).
        checkpoint_path: Đường dẫn tới checkpoint. Mặc định: best_model.pth.
        device:          torch.device. Mặc định: tự động chọn CUDA/CPU.

    Returns:
        np.ndarray (H, W) uint8: binary mask {0=background, 1=oil spill}.

    Example:
        >>> from predict import predict
        >>> mask = predict("sar_image.png", img_size=(512, 512))
        >>> print(mask.shape)   # (512, 512)
        >>> print(mask.max())   # 1
    """
    model, dev = _get_model(checkpoint_path, device)

    tensor = preprocess_for_inference(image_path, img_size=img_size)
    tensor = tensor.to(dev)

    with torch.no_grad():
        logits = model(tensor)  # (1, 2, H, W)

    return postprocess_mask(logits)  # (H, W) uint8


def predict_batch(
    image_dir: Union[str, Path],
    img_size: Tuple[int, int] = config.IMG_SIZE,
    checkpoint_path: Union[str, Path, None] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> List[np.ndarray]:
    """
    Dự đoán binary mask cho toàn bộ ảnh trong một thư mục.

    Args:
        image_dir:       Thư mục chứa ảnh SAR.
        img_size:        (H, W) kích thước resize.
        checkpoint_path: Đường dẫn checkpoint. Mặc định: best_model.pth.
        device:          torch.device.
        batch_size:      Số ảnh xử lý mỗi batch (tránh OOM).
        extensions:      Các định dạng ảnh hợp lệ.

    Returns:
        List[np.ndarray]: Danh sách mask (H, W) uint8 theo thứ tự file.

    Example:
        >>> from predict import predict_batch
        >>> masks = predict_batch("val/images/", img_size=(512, 512))
        >>> print(len(masks))         # số lượng ảnh
        >>> print(masks[0].shape)     # (512, 512)
    """
    model, dev = _get_model(checkpoint_path, device)

    image_dir  = Path(image_dir)
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    ])

    if not image_paths:
        print(f"Không tìm thấy ảnh trong: {image_dir}")
        return []

    all_masks: List[np.ndarray] = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch predict"):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensor = preprocess_batch(batch_paths, img_size=img_size).to(dev)

        with torch.no_grad():
            logits = model(batch_tensor)   # (B, 2, H, W)
            preds  = logits.argmax(dim=1)  # (B, H, W)

        for j in range(preds.size(0)):
            all_masks.append(preds[j].cpu().numpy().astype(np.uint8))

    return all_masks


def predict_with_overlay(
    image_path: Union[str, Path],
    img_size: Tuple[int, int] = config.IMG_SIZE,
    checkpoint_path: Union[str, Path, None] = None,
    device: Optional[torch.device] = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Dự đoán và trả về ảnh SAR gốc được overlay bởi vùng dầu loang dự đoán.

    Args:
        image_path:      Đường dẫn tới ảnh SAR.
        img_size:        (H, W) kích thước resize.
        checkpoint_path: Đường dẫn checkpoint.
        device:          torch.device.
        alpha:           Độ trong suốt của overlay [0, 1].

    Returns:
        np.ndarray (H, W, 3) uint8 RGB — ảnh SAR có overlay màu đỏ vùng dầu.

    Example:
        >>> from predict import predict_with_overlay
        >>> import matplotlib.pyplot as plt
        >>> img_overlay = predict_with_overlay("sar_image.png")
        >>> plt.imshow(img_overlay); plt.show()
    """
    mask   = predict(image_path, img_size, checkpoint_path, device)
    sar    = load_sar_image(image_path, img_size=img_size)
    return overlay_prediction(sar, mask, alpha=alpha)


# ── Command-line interface ────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SegFormer oil spill inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Đường dẫn ảnh đơn hoặc thư mục",
    )
    parser.add_argument(
        "--output", type=Path, default=config.PREDICTION_DIR,
        help="Thư mục lưu kết quả",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=config.CHECKPOINT_DIR / "best_model.pth",
        help="Đường dẫn tới checkpoint .pth",
    )
    parser.add_argument(
        "--img-size", type=int, nargs=2, default=list(config.IMG_SIZE),
        metavar=("H", "W"),
        help="Kích thước ảnh resize",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Số ảnh xử lý mỗi batch (chỉ dùng với --input folder)",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="Lưu ảnh overlay thay vì mask thuần",
    )
    return parser.parse_args()


def main() -> None:
    args      = _parse_args()
    img_size  = tuple(args.img_size)
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device} | Input: {args.input}")

    # ── Batch mode (folder input) ──────────────────────────────────────────
    if args.input.is_dir():
        image_paths = sorted(
            list(args.input.glob("*.png")) +
            list(args.input.glob("*.jpg")) +
            list(args.input.glob("*.jpeg"))
        )
        print(f"  Tìm thấy {len(image_paths)} ảnh trong {args.input}")

        model, dev = _get_model(args.checkpoint, device)

        for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Predicting"):
            batch_paths  = image_paths[i : i + args.batch_size]
            batch_tensor = preprocess_batch(batch_paths, img_size=img_size).to(dev)

            with torch.no_grad():
                logits = model(batch_tensor)
                preds  = logits.argmax(dim=1)

            for j, img_path in enumerate(batch_paths):
                mask = preds[j].cpu().numpy().astype(np.uint8)

                if args.overlay:
                    sar  = load_sar_image(img_path, img_size=img_size)
                    out  = overlay_prediction(sar, mask)
                    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(args.output / f"{img_path.stem}_overlay.png"), out_bgr)
                else:
                    # Lưu mask binary: oil = 255 để dễ nhìn
                    cv2.imwrite(
                        str(args.output / f"{img_path.stem}_mask.png"),
                        (mask * 255).astype(np.uint8),
                    )

    # ── Single image mode ──────────────────────────────────────────────────
    else:
        if args.overlay:
            result   = predict_with_overlay(
                args.input, img_size=img_size, checkpoint_path=args.checkpoint, device=device
            )
            out_path = args.output / f"{args.input.stem}_overlay.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        else:
            mask     = predict(
                args.input, img_size=img_size, checkpoint_path=args.checkpoint, device=device
            )
            out_path = args.output / f"{args.input.stem}_mask.png"
            cv2.imwrite(str(out_path), (mask * 255).astype(np.uint8))

        print(f"  Kết quả lưu tại: {out_path}")


if __name__ == "__main__":
    main()
