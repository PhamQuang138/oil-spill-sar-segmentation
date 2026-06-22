"""
train_deeplabv3plus.py — Train DeepLabV3+ (Bản Tối ưu hóa Toàn diện)
- Bổ sung RAM Caching, Early Stopping, cuDNN Benchmark, và Full Metrics.
"""
from pathlib import Path
import sys
import random
import csv
from typing import Tuple

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    if img is None:
        raise IOError(f"Can't read image: {image_path}")
    H, W = img_size
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img

def load_mask(mask_path: Path, img_size: Tuple[int, int]) -> np.ndarray:
    msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if msk is None:
        raise IOError(f"Can't read mask: {mask_path}")
    H, W = img_size
    if msk.shape[:2] != (H, W):
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
    return (msk > MASK_THRESHOLD).astype(np.uint8)

def to_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def normalize_imagenet(image_rgb: np.ndarray) -> np.ndarray:
    return (image_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

# ── Dataset (Tối ưu RAM Caching) ───────────────────────────────────────────
class OilSpillDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, img_size: Tuple[int, int], augment: bool = False, cache_ram: bool = True):
        self.img_paths = sorted(list(Path(img_dir).glob("*.png")))
        self.mask_paths = sorted(list(Path(mask_dir).glob("*.png")))
        self.n = min(len(self.img_paths), len(self.mask_paths))
        self.img_paths = self.img_paths[: self.n]
        self.mask_paths = self.mask_paths[: self.n]
        self.img_size = img_size
        self.augment = augment
        self.cache_ram = cache_ram

        # Đưa toàn bộ ảnh vào RAM để tăng tốc đọc dữ liệu trên Windows
        self.images = []
        self.masks = []
        if self.cache_ram:
            print(f"  [INFO] Đang nạp dataset vào RAM từ: {img_dir.name}...")
            for p in self.img_paths:
                self.images.append(load_sar_image(p, self.img_size))
            for p in self.mask_paths:
                self.masks.append(load_mask(p, self.img_size))

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # Lấy thẳng từ RAM nếu bật cache, ngược lại đọc từ ổ cứng
        if self.cache_ram:
            img = self.images[idx]
            msk = self.masks[idx]
        else:
            img = load_sar_image(self.img_paths[idx], self.img_size)
            msk = load_mask(self.mask_paths[idx], self.img_size)

        if self.augment:
            if random.random() < config.AUG_FLIP_P:
                if random.random() < 0.5:
                    img, msk = np.fliplr(img).copy(), np.fliplr(msk).copy()
                else:
                    img, msk = np.flipud(img).copy(), np.flipud(msk).copy()
            if random.random() < config.AUG_ROTATE_90_P:
                k = random.choice([1, 2, 3])
                img, msk = np.rot90(img, k).copy(), np.rot90(msk, k).copy()

        rgb = to_rgb(img)
        nrm = normalize_imagenet(rgb)

        img_t = torch.from_numpy(nrm.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(msk[None].astype(np.float32)).float()

        return img_t, mask_t

# ── Model builder ─────────────────────────────────────────────────────────
def build_deeplab(num_classes: int = 1, pretrained: bool = True):
    try:
        import segmentation_models_pytorch as smp
        print("  [INFO] Using segmentation_models_pytorch.DeepLabV3Plus (ResNet34)")
        model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )
        return model
    except ImportError:
        raise RuntimeError("Please install SMP: pip install segmentation-models-pytorch")

# ── Metrics & Losses ───────────────────────────────────────────────────────
def dice_loss_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersect = (probs_flat * targets_flat).sum(1)
    denom = probs_flat.sum(1) + targets_flat.sum(1)
    dice = (2 * intersect + smooth) / (denom + smooth)
    return 1.0 - dice.mean()

def calculate_metrics(logits: torch.Tensor, masks: torch.Tensor, smooth: float = 1e-6):
    """Tính toán cùng lúc Dice, Accuracy, Precision, Recall"""
    preds = (torch.sigmoid(logits) > 0.5).float()

    # Tính Dice
    inter = (preds * masks).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = ((2 * inter + smooth) / (denom + smooth)).mean().item()

    # Tính ma trận nhầm lẫn
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

# ── Training Loop ─────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [INFO] Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")

    # Tối ưu phần cứng GPU
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  [INFO] Kích hoạt cuDNN Benchmark để tăng tốc Conv2D")

    img_size = tuple(args.img_size)

    # Đã bật cache_ram=True để tăng tốc độ nạp dữ liệu
    train_ds = OilSpillDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, img_size, augment=True, cache_ram=True)
    val_ds = OilSpillDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, img_size, augment=False, cache_ram=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_deeplab(num_classes=1, pretrained=not args.no_pretrained).to(device)

    bce_pos_weight = torch.tensor(config.BCE_POS_WEIGHT, dtype=torch.float32).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.LR_T_MAX, eta_min=config.LR_ETA_MIN)

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")
    best_val_dice = 0.0

    # Early Stopping variables
    patience = 15
    epochs_no_improve = 0

    # Khởi tạo file CSV
    log_file = open(config.LOG_PATH, mode='w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_Dice", "Val_Acc", "Val_Pre", "Val_Rec"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0

        for i, (imgs, masks) in enumerate(train_loader, 1):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(imgs)
                loss_bce = bce_loss_fn(logits, masks)
                loss_dice = dice_loss_logits(logits, masks, smooth=config.DICE_SMOOTH)
                loss = config.DICE_WEIGHT * loss_dice + (1.0 - config.DICE_WEIGHT) * loss_bce

            scaler.scale(loss).backward()

            if config.GRAD_CLIP_MAX_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if i % config.LOG_EVERY_BATCH == 0:
                print(f"  Epoch {epoch:03d} | Batch {i:03d}/{len(train_loader)} | Train Loss: {running_loss / config.LOG_EVERY_BATCH:.4f}")
                running_loss = 0.0

        scheduler.step()
        train_loss_avg = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_losses, dices, accs, pres, recs = [], [], [], [], []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                    logits = model(imgs)
                    loss_bce = bce_loss_fn(logits, masks)
                    loss_dice = dice_loss_logits(logits, masks, smooth=config.DICE_SMOOTH)
                    loss = config.DICE_WEIGHT * loss_dice + (1.0 - config.DICE_WEIGHT) * loss_bce
                val_losses.append(loss.item())

                d, a, p, r = calculate_metrics(logits, masks, config.DICE_SMOOTH)
                dices.append(d); accs.append(a); pres.append(p); recs.append(r)

        val_loss_avg = np.mean(val_losses)
        val_dice_avg = np.mean(dices)

        print(f"  Epoch {epoch:03d} End — Val Loss: {val_loss_avg:.4f} | Val Dice: {val_dice_avg:.4f} | Acc: {np.mean(accs):.4f}")

        # Ghi Log vào CSV
        log_writer.writerow([epoch, f"{train_loss_avg:.4f}", f"{val_loss_avg:.4f}", f"{val_dice_avg:.4f}", f"{np.mean(accs):.4f}", f"{np.mean(pres):.4f}", f"{np.mean(recs):.4f}"])
        log_file.flush()

        # Save Best Model & Early Stopping Logic
        if val_dice_avg > best_val_dice:
            best_val_dice = val_dice_avg
            epochs_no_improve = 0
            config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            ckpt_path = config.CHECKPOINT_DIR / "best_deeplabv3plus.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice_avg,
            }, ckpt_path)
            print(f"  [+] Saved new best checkpoint -> {ckpt_path.name}")
        else:
            epochs_no_improve += 1
            print(f"  [-] Không cải thiện ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"\n  [CẢNH BÁO] Early Stopping kích hoạt tại Epoch {epoch}. Mô hình đã ngừng cải thiện.")
                break

    log_file.close()
    print(f"\n  [OK] Training finished! Best Val Dice: {best_val_dice:.4f}")

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--img-size", type=int, nargs=2, default=list(config.IMG_SIZE))
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    return p.parse_args()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train(parse_args())