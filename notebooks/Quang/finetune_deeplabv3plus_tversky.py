"""Fine-tune DeepLabV3+ ResNet34 with Tversky/Focal Tversky loss.

This script is prepared for follow-up training. It starts from an existing
DeepLabV3+ checkpoint and fine-tunes on the same Kaggle SAR oil-spill dataset.

Examples:
    python finetune_deeplabv3plus_tversky.py --epochs 0
    python finetune_deeplabv3plus_tversky.py
    python finetune_deeplabv3plus_tversky.py --loss bce_dice --lr 2e-6
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from train_deeplabv3plus import OilSpillDataset, build_deeplab, dice_loss_logits


ROOT = HERE.parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    def metrics(self) -> dict[str, float]:
        eps = 1e-7
        dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "dice": float(dice),
            "iou": float(iou),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
        }


def tversky_loss_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Tversky loss for imbalanced binary segmentation.

    beta > alpha penalizes false negatives more strongly, which is useful when
    missing oil spill pixels is more costly than predicting a few extra pixels.
    """

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
    smooth: float = 1e-6,
) -> torch.Tensor:
    base = tversky_loss_logits(logits, targets, alpha=alpha, beta=beta, smooth=smooth)
    return torch.pow(base, gamma)


def build_loss(args, device: torch.device):
    bce_pos_weight = torch.tensor(config.BCE_POS_WEIGHT, dtype=torch.float32, device=device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if args.loss == "tversky":
            return tversky_loss_logits(logits, targets, alpha=args.alpha, beta=args.beta)
        if args.loss == "focal_tversky":
            return focal_tversky_loss_logits(
                logits,
                targets,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )
        if args.loss == "bce_tversky":
            tv = tversky_loss_logits(logits, targets, alpha=args.alpha, beta=args.beta)
            bce = bce_loss_fn(logits, targets)
            return args.tversky_weight * tv + (1.0 - args.tversky_weight) * bce

        bce = bce_loss_fn(logits, targets)
        dice = dice_loss_logits(logits, targets, smooth=config.DICE_SMOOTH)
        return config.DICE_WEIGHT * dice + (1.0 - config.DICE_WEIGHT) * bce

    return loss_fn


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        print("Checkpoint is not compatible with weights_only=True; falling back to trusted local load.")
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


def evaluate(model, loader, loss_fn, device: torch.device, threshold: float):
    model.eval()
    losses = []
    counts = Counts()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            losses.append(loss.item())
            counts.update(logits.float(), masks.float(), threshold)

    metrics = counts.metrics()
    metrics["loss"] = float(np.mean(losses))
    return metrics


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value.")
    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}.")
    return thresholds


def evaluate_threshold_sweep(model, loader, loss_fn, device: torch.device, thresholds: list[float]):
    model.eval()
    losses = []
    counts_by_threshold = {threshold: Counts() for threshold in thresholds}

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
            losses.append(loss.item())
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

    best_metrics["loss"] = float(np.mean(losses))
    best_metrics["threshold"] = float(best_threshold)
    return best_metrics


def dataset_dirs(dataset_root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        dataset_root / "images" / "images" / "train",
        dataset_root / "masks" / "masks" / "train",
        dataset_root / "images" / "images" / "val",
        dataset_root / "masks" / "masks" / "val",
    )


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    encoder = getattr(model, "encoder", None)
    encoder_param_ids = set()
    if encoder is not None:
        encoder_param_ids = {id(param) for param in encoder.parameters()}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints" / "quang_best_deeplabv3plus_checkpoint.pth")
    parser.add_argument("--output-dir", type=Path, default=HERE / "finetune_outputs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.2)
    parser.add_argument("--decoder-lr-multiplier", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--loss", choices=["bce_dice", "tversky", "focal_tversky", "bce_tversky"], default="bce_tversky")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--tversky-weight", type=float, default=0.6)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", type=str, default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--seed", type=int, default=-1, help="Use -1 to keep random training order/augmentation.")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.seed >= 0:
        set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Loss: {args.loss}")
    print(
        f"Fine-tune setup: epochs={args.epochs} lr={args.lr:g} "
        f"decoder_lr_multiplier={args.decoder_lr_multiplier:g} "
        f"freeze_encoder_epochs={args.freeze_encoder_epochs} patience={args.patience} "
        f"seed={args.seed if args.seed >= 0 else 'random'}"
    )
    print(
        f"Loss setup: alpha={args.alpha:g} beta={args.beta:g} "
        f"tversky_weight={args.tversky_weight:g}"
    )

    train_img_dir, train_mask_dir, val_img_dir, val_mask_dir = dataset_dirs(args.dataset_root)
    train_ds = OilSpillDataset(train_img_dir, train_mask_dir, config.IMG_SIZE, augment=True, cache_ram=True)
    val_ds = OilSpillDataset(val_img_dir, val_mask_dir, config.IMG_SIZE, augment=False, cache_ram=True)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

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

    model = build_deeplab(num_classes=1, pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)

    loss_fn = build_loss(args, device)
    thresholds = parse_thresholds(args.thresholds)
    current_encoder_trainable = args.freeze_encoder_epochs == 0
    set_encoder_trainable(model, current_encoder_trainable)
    optimizer = build_optimizer(model, args)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    baseline_metrics = evaluate(model, val_loader, loss_fn, device, args.threshold)
    baseline_sweep_metrics = evaluate_threshold_sweep(model, val_loader, loss_fn, device, thresholds)
    best_dice = baseline_metrics["dice"]
    best_sweep_dice = baseline_sweep_metrics["dice"]
    no_improve = 0
    log_path = args.output_dir / "deeplabv3plus_finetune_log.csv"
    best_path = args.output_dir / "deeplabv3plus_finetuned_tversky_best.pth"

    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "val_dice": best_dice,
            "loss": args.loss,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "threshold": args.threshold,
            "seed": args.seed,
            "sweep_threshold": baseline_sweep_metrics["threshold"],
            "sweep_val_dice": best_sweep_dice,
            "note": "Initial loaded checkpoint before fine-tuning.",
        },
        best_path,
    )
    print(
        f"Baseline epoch 000: val_dice={baseline_metrics['dice']:.4f} "
        f"val_iou={baseline_metrics['iou']:.4f} P={baseline_metrics['precision']:.4f} "
        f"R={baseline_metrics['recall']:.4f} | sweep_dice={baseline_sweep_metrics['dice']:.4f} "
        f"@thr={baseline_sweep_metrics['threshold']:.2f}"
    )

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_dice",
            "val_iou",
            "val_acc",
            "val_precision",
            "val_recall",
            "sweep_val_dice",
            "sweep_threshold",
            "lr",
            "encoder_trainable",
        ])
        writer.writerow([
            0,
            "",
            f"{baseline_metrics['loss']:.6f}",
            f"{baseline_metrics['dice']:.6f}",
            f"{baseline_metrics['iou']:.6f}",
            f"{baseline_metrics['accuracy']:.6f}",
            f"{baseline_metrics['precision']:.6f}",
            f"{baseline_metrics['recall']:.6f}",
            f"{baseline_sweep_metrics['dice']:.6f}",
            f"{baseline_sweep_metrics['threshold']:.2f}",
            ",".join(f"{group['lr']:.8f}" for group in optimizer.param_groups),
            int(args.freeze_encoder_epochs == 0),
        ])

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
            epoch_losses = []

            for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}"):
                imgs = imgs.to(device)
                masks = masks.to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(imgs)
                    loss = loss_fn(logits, masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())

            scheduler.step()
            metrics = evaluate(model, val_loader, loss_fn, device, args.threshold)
            sweep_metrics = evaluate_threshold_sweep(model, val_loader, loss_fn, device, thresholds)
            train_loss = float(np.mean(epoch_losses))
            lr = ",".join(f"{group['lr']:.8f}" for group in optimizer.param_groups)

            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{metrics['loss']:.6f}",
                f"{metrics['dice']:.6f}",
                f"{metrics['iou']:.6f}",
                f"{metrics['accuracy']:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['recall']:.6f}",
                f"{sweep_metrics['dice']:.6f}",
                f"{sweep_metrics['threshold']:.2f}",
                lr,
                int(encoder_trainable),
            ])
            f.flush()

            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f} "
                f"val_dice={metrics['dice']:.4f} val_iou={metrics['iou']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} | "
                f"sweep_dice={sweep_metrics['dice']:.4f} @thr={sweep_metrics['threshold']:.2f}"
            )

            if sweep_metrics["dice"] > best_sweep_dice:
                best_dice = metrics["dice"]
                best_sweep_dice = sweep_metrics["dice"]
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_dice": best_dice,
                        "sweep_val_dice": best_sweep_dice,
                        "sweep_threshold": sweep_metrics["threshold"],
                        "loss": args.loss,
                        "alpha": args.alpha,
                        "beta": args.beta,
                        "gamma": args.gamma,
                        "tversky_weight": args.tversky_weight,
                        "threshold": args.threshold,
                        "seed": args.seed,
                    },
                    best_path,
                )
                print(f"  Saved best fine-tuned checkpoint: {best_path}")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    print(f"Done. Best validation Dice @ fixed threshold {args.threshold:.2f}: {best_dice:.4f}")
    print(f"Done. Best validation Dice after threshold sweep: {best_sweep_dice:.4f}")
    print(f"Log: {log_path}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
