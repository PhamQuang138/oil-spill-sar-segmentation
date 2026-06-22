"""
evaluate.py — Đánh giá toàn diện các chỉ số của mô hình ĐÃ TRAIN (Standalone)
Chạy: python evaluate.py
"""
import sys
from pathlib import Path
from typing import Tuple
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Import config
HERE = Path(__file__).parent
QUANG_DIR = HERE.parent / "Quang"
sys.path.insert(0, str(QUANG_DIR))
import config

# ── Preprocessing utilities ──────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MASK_THRESHOLD = 127

def load_sar_image(image_path: Path, img_size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None: raise IOError(f"Can't read image: {image_path}")
    H, W = img_size
    if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img

def load_mask(mask_path: Path, img_size: Tuple[int, int]) -> np.ndarray:
    msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if msk is None: raise IOError(f"Can't read mask: {mask_path}")
    H, W = img_size
    if msk.shape[:2] != (H, W): msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
    return (msk > MASK_THRESHOLD).astype(np.uint8)

# ── Dataset ────────────────────────────────────────────────────────────────
class OilSpillDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, img_size: Tuple[int, int]):
        self.img_paths = sorted(list(Path(img_dir).glob("*.png")))
        self.mask_paths = sorted(list(Path(mask_dir).glob("*.png")))
        self.n = min(len(self.img_paths), len(self.mask_paths))
        self.img_paths = self.img_paths[: self.n]
        self.mask_paths = self.mask_paths[: self.n]
        self.img_size = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        img = load_sar_image(self.img_paths[idx], self.img_size)
        msk = load_mask(self.mask_paths[idx], self.img_size)

        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        nrm = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        img_t = torch.from_numpy(nrm.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(msk[None].astype(np.float32)).float()
        return img_t, mask_t

# ── Metrics ────────────────────────────────────────────────────────────────
def calculate_metrics(logits: torch.Tensor, masks: torch.Tensor, smooth: float = 1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()

    # Dice
    inter = (preds * masks).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = ((2 * inter + smooth) / (denom + smooth)).mean().item()

    # Classification Metrics
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    TP = (preds_flat * masks_flat).sum().item()
    FP = (preds_flat * (1 - masks_flat)).sum().item()
    FN = ((1 - preds_flat) * masks_flat).sum().item()
    TN = ((1 - preds_flat) * (1 - masks_flat)).sum().item()

    eps = 1e-7
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    pre = TP / (TP + FP + eps)
    rec = TP / (TP + FN + eps)

    return dice, acc, pre, rec

# ── Evaluation ─────────────────────────────────────────────────────────────
def load_trained_model(device):
    print("  [INFO] Khởi tạo kiến trúc DeepLabV3+...")
    model = smp.DeepLabV3Plus(
        encoder_name=config.BACKBONE,
        encoder_weights=None,
        in_channels=config.C_IN,
        classes=config.NUM_CLASSES,
    ).to(device)

    ckpt_path = config.CHECKPOINT_DIR / "best_deeplabv3plus.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {ckpt_path}!")

    print(f"  [INFO] Đang nạp trọng số từ: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_epoch = checkpoint.get("epoch", "N/A")
    return model, best_epoch


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 50}")
    print("  ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION")
    print(f"{'=' * 50}")
    print(f"  Thiết bị : {device}")

    val_loader = DataLoader(
        OilSpillDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    try:
        model, best_epoch = load_trained_model(device)
    except Exception as e:
        print(f"  [LỖI] {e}")
        return

    dices, accs, pres, recs = [], [], [], []

    print("  [INFO] Đang quét qua toàn bộ ảnh Validation...")
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(imgs)

            d, a, p, r = calculate_metrics(logits, masks)
            dices.append(d); accs.append(a); pres.append(p); recs.append(r)

    print(f"\n{'=' * 50}")
    print(f"  KẾT QUẢ ĐÁNH GIÁ (Tại Epoch tốt nhất: {best_epoch})")
    print(f"{'=' * 50}")
    print(f"  - F1-Score (Dice) : {np.mean(dices):.4f}")
    print(f"  - Accuracy        : {np.mean(accs):.4f}")
    print(f"  - Precision       : {np.mean(pres):.4f}")
    print(f"  - Recall          : {np.mean(recs):.4f}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    evaluate()