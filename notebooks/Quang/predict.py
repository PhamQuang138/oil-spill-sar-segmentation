"""
predict.py — Dự đoán và trực quan hóa kết quả bằng DeepLabV3+
Chạy: python predict.py --img_path path/to/sar_image.png
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Import cấu hình
import config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_trained_model(device):
    """Nạp kiến trúc mô hình và load file trọng số (weights) tốt nhất."""
    model = smp.DeepLabV3Plus(
        encoder_name=config.BACKBONE,
        encoder_weights=None,  # Khi dự đoán không cần load lại ImageNet, chỉ lấy bộ khung
        in_channels=config.C_IN,
        classes=config.NUM_CLASSES,
    ).to(device)

    ckpt_path = config.CHECKPOINT_DIR / "best_deeplabv3plus.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Chưa tìm thấy file mô hình tại {ckpt_path}. Bạn đã train xong chưa?")

    print(f"  [INFO] Đang nạp trọng số từ: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load state dict (các mảng số liệu) vào kiến trúc mạng
    model.load_state_dict(checkpoint["model_state_dict"])

    # Bật chế độ đánh giá (tắt Dropout và BatchNorm cập nhật)
    model.eval()
    return model


def preprocess_for_inference(img_path: Path, device):
    """Đọc ảnh từ ổ cứng và chuyển đổi thành Tensor phù hợp cho model."""
    img_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        raise IOError(f"Không thể đọc ảnh: {img_path}")

    img_resized = cv2.resize(img_raw, (config.IMG_SIZE[1], config.IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    nrm = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

    # (H, W, C) -> (C, H, W) và thêm chiều Batch size = 1
    tensor = torch.from_numpy(nrm.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_resized, tensor.to(device)


def predict_and_visualize(img_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [INFO] Bắt đầu Inference. Device: {device}")

    # 1. Load Mô hình
    try:
        model = load_trained_model(device)
    except Exception as e:
        print(f"  [LỖI] {e}")
        return

    # 2. Tiền xử lý
    img_gray, img_tensor = preprocess_for_inference(img_path, device)

    # 3. Chạy Dự đoán qua mạng Deep Learning
    with torch.no_grad():
        # Dùng AMP (autocast) giống như lúc train để dự đoán cho nhanh
        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)

            # Threshold > 0.5 để quyết định pixel đó có phải là dầu hay không
            mask_pred = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # 4. Trực quan hóa kết quả (Overlay ảnh)
    rgb_background = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay = rgb_background.copy()

    # Tô màu đỏ [R, G, B] cho những nơi mask = 1
    overlay[mask_pred == 1] = [230, 60, 60]

    # Chồng lớp (Alpha blending)
    result_img = cv2.addWeighted(rgb_background, 0.6, overlay, 0.4, 0)

    # 5. Vẽ và Lưu hình
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title(f"1. Ảnh SAR Gốc\n({img_gray.shape[1]}x{img_gray.shape[0]})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_pred, cmap='jet')

    # Tính diện tích (tỷ lệ %)
    oil_pct = np.mean(mask_pred) * 100
    plt.title(f"2. Mask Dự đoán\n(Dầu chiếm {oil_pct:.2f}%)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.title("3. Ảnh Overlay")
    plt.axis('off')

    # Định tuyến thư mục lưu
    save_path = config.PREDICTION_DIR / f"pred_{img_path.name}"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

    print(f"  [OK] Dự đoán thành công! Kết quả đã lưu tại:\n  -> {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLabV3+ Inference cho 1 ảnh")

    # Dùng default=... thay vì help=...
    parser.add_argument(
        "--img",
        type=Path,
        default=r"C:\Users\duyqu\Downloads\2aOboQXg461SxxAafIkNS1sXXehaos9rm44BZYtE.jpg",
        help="Đường dẫn đến file ảnh SAR"
    )

    args = parser.parse_args()
    predict_and_visualize(args.img)