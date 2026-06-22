"""
train.py — Entry point for training SegFormer on SAR oil spill dataset.

Usage:
    python train.py
    python train.py --backbone nvidia/mit-b2 --epochs 150 --batch-size 8
    python train.py --resume checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.data.dataset import OilSpillDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.model.segformer import build_model
from src.training.trainer import Trainer
from src.utils.checkpoint import CheckpointManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SegFormer for SAR oil spill segmentation"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=config.BACKBONE,
        help=f"HuggingFace backbone (default: {config.BACKBONE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Number of training epochs (default: {config.EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=list(config.IMG_SIZE),
        metavar=("H", "W"),
        help=f"Image size H W (default: {config.IMG_SIZE})",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch (no ImageNet weights)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=config.NUM_WORKERS,
        help=f"DataLoader workers (default: {config.NUM_WORKERS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_size = tuple(args.img_size)  # (H, W)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  deep-sar-segformer — SegFormer Oil Spill Segmentation")
    print(f"  Device   : {device}")
    print(f"  Backbone : {args.backbone}")
    print(f"  Img size : {img_size}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"{'='*60}\n")

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    train_ds = OilSpillDataset(
        img_dir   = config.TRAIN_IMG_DIR,
        mask_dir  = config.TRAIN_MASK_DIR,
        transform = get_train_transforms(img_size),
        img_size  = img_size,
    )
    val_ds = OilSpillDataset(
        img_dir   = config.VAL_IMG_DIR,
        mask_dir  = config.VAL_MASK_DIR,
        transform = get_val_transforms(img_size),
        img_size  = img_size,
    )

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}\n")

    # Windows fix: num_workers > 0 requires if __name__ == '__main__' guard
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        backbone    = args.backbone,
        num_classes = config.NUM_CLASSES,
        pretrained  = not args.no_pretrained,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume is not None:
        ckpt_mgr = CheckpointManager(config.CHECKPOINT_DIR)
        ckpt_data = ckpt_mgr.load(model, args.resume, device=device)
        start_epoch = ckpt_data.get("epoch", 1) + 1
        print(f"  Resuming from epoch {start_epoch}\n")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        log_path       = config.LOG_PATH,
        checkpoint_dir = config.CHECKPOINT_DIR,
    )

    remaining_epochs = args.epochs - (start_epoch - 1)
    trainer.fit(epochs=remaining_epochs)
    print("\n  Training complete. Best checkpoint saved to:", config.CHECKPOINT_DIR)


if __name__ == "__main__":
    # Required on Windows for multiprocessing DataLoader workers
    import multiprocessing
    multiprocessing.freeze_support()
    main()
