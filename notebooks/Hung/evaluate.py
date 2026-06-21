"""
evaluate.py — Evaluate a trained SegFormer checkpoint on the validation set.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --save-visuals
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.data.dataset import OilSpillDataset
from src.data.transforms import get_val_transforms
from src.model.segformer import build_model
from src.training.metrics import MetricAccumulator
from src.utils.checkpoint import CheckpointManager
from src.utils.visualization import save_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SegFormer checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.CHECKPOINT_DIR / "best_model.pth",
        help="Path to .pth checkpoint",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save side-by-side comparison images to predictions/",
    )
    parser.add_argument(
        "--num-visuals",
        type=int,
        default=10,
        help="Number of sample visualizations to save (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}\n")

    # ── Dataset ───────────────────────────────────────────────────────────────
    val_ds = OilSpillDataset(
        img_dir   = config.VAL_IMG_DIR,
        mask_dir  = config.VAL_MASK_DIR,
        transform = get_val_transforms(config.IMG_SIZE),
        img_size  = config.IMG_SIZE,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,  # safe for single-process evaluation
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(pretrained=False)
    ckpt_mgr = CheckpointManager(config.CHECKPOINT_DIR)
    ckpt_mgr.load(model, args.checkpoint, device=device)
    model.to(device).eval()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    acc = MetricAccumulator(num_classes=config.NUM_CLASSES)
    n_saved = 0
    import cv2

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            gts    = masks.numpy()
            acc.update(preds, gts)

            # Optionally save comparison images
            if args.save_visuals and n_saved < args.num_visuals:
                for i in range(images.size(0)):
                    if n_saved >= args.num_visuals:
                        break
                    # De-normalize for visualization
                    img_np = images[i].cpu().numpy()
                    img_np = (img_np.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                    sar_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    save_path = config.PREDICTION_DIR / f"eval_{batch_idx:04d}_{i}.png"
                    save_comparison(sar_gray, gts[i], preds[i], save_path)
                    n_saved += 1

    metrics = acc.compute()
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("="*50)

    if args.save_visuals:
        print(f"\n  Saved {n_saved} comparison images to: {config.PREDICTION_DIR}")


if __name__ == "__main__":
    main()
