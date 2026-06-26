"""
train_deeplabv3plus.py — Train DeepLabV3+ (Bản Tối ưu hóa Toàn diện)
- Bổ sung RAM Caching, Early Stopping, cuDNN Benchmark, và Full Metrics.
"""
from pathlib import Path
import sys
import random
import csv
import warnings
from dataclasses import dataclass
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
@dataclass
class Counts:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def update(self, logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> None:
        preds = (torch.sigmoid(logits) > threshold).float()
        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)
        self.tp += (preds_flat * masks_flat).sum().item()
        self.fp += (preds_flat * (1.0 - masks_flat)).sum().item()
        self.fn += ((1.0 - preds_flat) * masks_flat).sum().item()
        self.tn += ((1.0 - preds_flat) * (1.0 - masks_flat)).sum().item()

    def metrics(self):
        eps = 1e-7
        dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        acc = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        pre = self.tp / (self.tp + self.fp + eps)
        rec = self.tp / (self.tp + self.fn + eps)
        return {
            "dice": float(dice),
            "iou": float(iou),
            "accuracy": float(acc),
            "precision": float(pre),
            "recall": float(rec),
        }


def dice_loss_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersect = (probs_flat * targets_flat).sum(1)
    denom = probs_flat.sum(1) + targets_flat.sum(1)
    dice = (2 * intersect + smooth) / (denom + smooth)
    return 1.0 - dice.mean()

def tversky_loss_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    tp = (probs_flat * targets_flat).sum(dim=1)
    fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
    fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()

def focal_tversky_loss_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
) -> torch.Tensor:
    base = tversky_loss_logits(logits, targets, alpha=alpha, beta=beta)
    return torch.pow(base, gamma)

def build_loss(args, device: torch.device):
    bce_pos_weight = torch.tensor(config.BCE_POS_WEIGHT, dtype=torch.float32).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)

    def loss_fn(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if args.loss == "tversky":
            return tversky_loss_logits(logits, masks, alpha=args.alpha, beta=args.beta)
        if args.loss == "focal_tversky":
            return focal_tversky_loss_logits(logits, masks, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        if args.loss == "bce_tversky":
            loss_bce = bce_loss_fn(logits, masks)
            loss_tversky = tversky_loss_logits(logits, masks, alpha=args.alpha, beta=args.beta)
            return args.tversky_weight * loss_tversky + (1.0 - args.tversky_weight) * loss_bce

        loss_bce = bce_loss_fn(logits, masks)
        loss_dice = dice_loss_logits(logits, masks, smooth=config.DICE_SMOOTH)
        return config.DICE_WEIGHT * loss_dice + (1.0 - config.DICE_WEIGHT) * loss_bce

    return loss_fn

def parse_thresholds(raw: str):
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value.")
    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}.")
    return thresholds

def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        print("  [INFO] Checkpoint is not compatible with weights_only=True; using trusted local load.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for param in encoder.parameters():
        param.requires_grad = trainable

def build_optimizer(model: torch.nn.Module, args) -> optim.Optimizer:
    encoder = getattr(model, "encoder", None)
    encoder_param_ids = {id(param) for param in encoder.parameters()} if encoder is not None else set()
    encoder_params = []
    decoder_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in encoder_param_ids:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr * args.encoder_lr_multiplier})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": args.lr * args.decoder_lr_multiplier})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")
    return optim.AdamW(param_groups, lr=args.lr, weight_decay=config.WEIGHT_DECAY)

def evaluate(model, val_loader, loss_fn, device: torch.device, threshold: float):
    model.eval()
    val_losses = []
    counts = Counts()
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            val_losses.append(loss.item())
            counts.update(logits.float(), masks.float(), threshold)
    metrics = counts.metrics()
    metrics["loss"] = float(np.mean(val_losses))
    metrics["threshold"] = threshold
    return metrics

def evaluate_threshold_sweep(model, val_loader, loss_fn, device: torch.device, thresholds):
    model.eval()
    val_losses = []
    counts_by_threshold = {threshold: Counts() for threshold in thresholds}
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            val_losses.append(loss.item())
            logits_float = logits.float()
            masks_float = masks.float()
            for threshold, counts in counts_by_threshold.items():
                counts.update(logits_float, masks_float, threshold)

    best_threshold = thresholds[0]
    best_metrics = counts_by_threshold[best_threshold].metrics()
    for threshold in thresholds[1:]:
        metrics = counts_by_threshold[threshold].metrics()
        if metrics["dice"] > best_metrics["dice"]:
            best_threshold = threshold
            best_metrics = metrics
    best_metrics["loss"] = float(np.mean(val_losses))
    best_metrics["threshold"] = float(best_threshold)
    return best_metrics

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
def train_legacy(args):
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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [INFO] Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"  [INFO] Loss: {args.loss}")
    print(
        f"  [INFO] LR: {args.lr:g} | encoder x{args.encoder_lr_multiplier:g} | "
        f"decoder x{args.decoder_lr_multiplier:g} | freeze encoder epochs: {args.freeze_encoder_epochs}"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  [INFO] cuDNN Benchmark enabled")

    img_size = tuple(args.img_size)
    train_ds = OilSpillDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, img_size, augment=True, cache_ram=True)
    val_ds = OilSpillDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, img_size, augment=False, cache_ram=True)
    print(f"  [INFO] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_deeplab(num_classes=1, pretrained=(not args.no_pretrained and args.checkpoint is None)).to(device)
    if args.checkpoint is not None:
        print(f"  [INFO] Loading checkpoint for integrated fine-tune: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, device)

    loss_fn = build_loss(args, device)
    thresholds = parse_thresholds(args.thresholds)
    current_encoder_trainable = args.freeze_encoder_epochs == 0
    set_encoder_trainable(model, current_encoder_trainable)
    optimizer = build_optimizer(model, args)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    best_val_dice = 0.0
    best_sweep_dice = 0.0
    best_threshold = args.threshold
    epochs_no_improve = 0

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.log_path, mode='w', newline='', encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "Epoch",
        "Train_Loss",
        "Val_Loss",
        "Val_Dice",
        "Val_IoU",
        "Val_Acc",
        "Val_Pre",
        "Val_Rec",
        "Sweep_Val_Dice",
        "Sweep_Threshold",
        "LR",
        "Encoder_Trainable",
    ])

    if args.checkpoint is not None or args.eval_baseline:
        baseline_metrics = evaluate(model, val_loader, loss_fn, device, args.threshold)
        baseline_sweep = evaluate_threshold_sweep(model, val_loader, loss_fn, device, thresholds)
        best_val_dice = baseline_metrics["dice"]
        best_sweep_dice = baseline_sweep["dice"]
        best_threshold = baseline_sweep["threshold"]
        log_writer.writerow([
            0,
            "",
            f"{baseline_metrics['loss']:.6f}",
            f"{baseline_metrics['dice']:.6f}",
            f"{baseline_metrics['iou']:.6f}",
            f"{baseline_metrics['accuracy']:.6f}",
            f"{baseline_metrics['precision']:.6f}",
            f"{baseline_metrics['recall']:.6f}",
            f"{baseline_sweep['dice']:.6f}",
            f"{baseline_sweep['threshold']:.2f}",
            ",".join(f"{group['lr']:.8f}" for group in optimizer.param_groups),
            int(current_encoder_trainable),
        ])
        log_file.flush()
        print(
            f"  Baseline epoch 000 | Val Dice: {baseline_metrics['dice']:.4f} | "
            f"Sweep Dice: {baseline_sweep['dice']:.4f} @thr={baseline_sweep['threshold']:.2f}"
        )
        if args.checkpoint is not None:
            torch.save({
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "val_dice": best_val_dice,
                "sweep_val_dice": best_sweep_dice,
                "sweep_threshold": best_threshold,
                "loss": args.loss,
                "note": "Loaded checkpoint before integrated fine-tune.",
            }, args.save_path)

    for epoch in range(1, args.epochs + 1):
        encoder_trainable = epoch > args.freeze_encoder_epochs
        if encoder_trainable != current_encoder_trainable:
            set_encoder_trainable(model, encoder_trainable)
            optimizer = build_optimizer(model, args)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - epoch + 1),
                eta_min=args.min_lr,
            )
            current_encoder_trainable = encoder_trainable

        model.train()
        running_loss = 0.0
        epoch_loss = 0.0

        for i, (imgs, masks) in enumerate(train_loader, 1):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            if config.GRAD_CLIP_MAX_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % config.LOG_EVERY_BATCH == 0:
                print(f"  Epoch {epoch:03d} | Batch {i:04d}/{len(train_loader)} | Train Loss: {running_loss / config.LOG_EVERY_BATCH:.4f}")
                running_loss = 0.0

        scheduler.step()
        train_loss_avg = epoch_loss / len(train_loader)
        metrics = evaluate(model, val_loader, loss_fn, device, args.threshold)
        sweep_metrics = evaluate_threshold_sweep(model, val_loader, loss_fn, device, thresholds)
        lr_text = ",".join(f"{group['lr']:.8f}" for group in optimizer.param_groups)

        print(
            f"  Epoch {epoch:03d} End | Val Loss: {metrics['loss']:.4f} | "
            f"Val Dice: {metrics['dice']:.4f} | Sweep Dice: {sweep_metrics['dice']:.4f} "
            f"@thr={sweep_metrics['threshold']:.2f}"
        )

        log_writer.writerow([
            epoch,
            f"{train_loss_avg:.6f}",
            f"{metrics['loss']:.6f}",
            f"{metrics['dice']:.6f}",
            f"{metrics['iou']:.6f}",
            f"{metrics['accuracy']:.6f}",
            f"{metrics['precision']:.6f}",
            f"{metrics['recall']:.6f}",
            f"{sweep_metrics['dice']:.6f}",
            f"{sweep_metrics['threshold']:.2f}",
            lr_text,
            int(encoder_trainable),
        ])
        log_file.flush()

        if sweep_metrics["dice"] > best_sweep_dice:
            best_val_dice = metrics["dice"]
            best_sweep_dice = sweep_metrics["dice"]
            best_threshold = sweep_metrics["threshold"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": best_val_dice,
                "sweep_val_dice": best_sweep_dice,
                "sweep_threshold": best_threshold,
                "loss": args.loss,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "tversky_weight": args.tversky_weight,
            }, args.save_path)
            print(f"  [+] Saved new best checkpoint -> {args.save_path}")
        else:
            epochs_no_improve += 1
            print(f"  [-] No improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"  [INFO] Early stopping at epoch {epoch}.")
                break

    log_file.close()
    print(f"\n  [OK] Training finished! Best Val Dice @ {args.threshold:.2f}: {best_val_dice:.4f}")
    print(f"  [OK] Best Sweep Dice: {best_sweep_dice:.4f} @thr={best_threshold:.2f}")
    print(f"  [OK] Log: {args.log_path}")
    print(f"  [OK] Best checkpoint: {args.save_path}")

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--img-size", type=int, nargs=2, default=list(config.IMG_SIZE))
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--checkpoint", type=Path, default=None, help="Load a checkpoint and fine-tune in this train script.")
    p.add_argument("--save-path", type=Path, default=config.CHECKPOINT_DIR / "best_deeplabv3plus.pth")
    p.add_argument("--log-path", type=Path, default=config.LOG_PATH)
    p.add_argument("--loss", choices=["bce_dice", "tversky", "focal_tversky", "bce_tversky"], default="bce_dice")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--tversky-weight", type=float, default=0.6)
    p.add_argument("--encoder-lr-multiplier", type=float, default=1.0)
    p.add_argument("--decoder-lr-multiplier", type=float, default=1.0)
    p.add_argument("--freeze-encoder-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--min-lr", type=float, default=config.LR_ETA_MIN)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds", type=str, default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--eval-baseline", action="store_true", help="Evaluate epoch 0 before training from scratch.")
    return p.parse_args()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train(parse_args())
