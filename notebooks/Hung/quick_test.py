"""
quick_test.py — Test nhanh với ảnh random từ val set.

Local (không cần transformers): Chỉ xem ảnh và mask gốc.
Trên Kaggle: Chạy inference với model và vẽ overlay.

Chạy local:
    python quick_test.py              # random 1 ảnh
    python quick_test.py --n 4        # random 4 ảnh
    python quick_test.py --seed 42    # cố định random seed
"""

import sys
import argparse
import random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",         type=int, default=1,  help="So anh test")
    p.add_argument("--seed",      type=int, default=None)
    p.add_argument("--split",     type=str, default="val", choices=["train", "val"])
    p.add_argument("--with-model",action="store_true",   help="Chay inference (can transformers)")
    return p.parse_args()


def load_image_mask(img_path: Path, mask_path: Path, img_size=config.IMG_SIZE):
    """Load 1 cap (anh, mask) va resize."""
    H, W = img_size
    img  = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE)
    msk  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    img  = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    msk  = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
    msk_bin = (msk > 127).astype(np.uint8)
    return img, msk_bin


def overlay(sar_gray, mask, alpha=0.45):
    """Overlay mask do len anh SAR grayscale."""
    rgb = cv2.cvtColor(sar_gray, cv2.COLOR_GRAY2RGB)
    ovr = rgb.copy()
    ovr[mask == 1] = (230, 60, 60)
    return cv2.addWeighted(rgb, 1 - alpha, ovr, alpha, 0)


def visualize_samples(samples, out_path: Path, with_pred=False):
    """Ve grid: [SAR | Ground Truth | (Prediction)]"""
    n_cols = 3 if with_pred else 2
    fig, axes = plt.subplots(len(samples), n_cols,
                             figsize=(n_cols * 4, len(samples) * 4))

    if len(samples) == 1:
        axes = [axes]

    col_titles = ["SAR Image", "Ground Truth"] + (["Model Prediction"] if with_pred else [])

    for row, item in enumerate(samples):
        img, msk, name = item["img"], item["mask"], item["name"]
        oil_pct = msk.mean() * 100

        axes[row][0].imshow(img, cmap="gray")
        axes[row][0].set_title(f"{name}\n({img.shape[1]}x{img.shape[0]})", fontsize=9)

        axes[row][1].imshow(overlay(img, msk), vmin=0, vmax=255)
        axes[row][1].set_title(f"GT overlay\nOil: {oil_pct:.1f}%", fontsize=9)

        if with_pred and "pred" in item:
            pred_pct = item["pred"].mean() * 100
            axes[row][2].imshow(overlay(img, item["pred"]), vmin=0, vmax=255)
            axes[row][2].set_title(f"Pred overlay\nOil: {pred_pct:.1f}%", fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    for col, title in enumerate(col_titles):
        axes[0][col].set_title(f"[{title}]\n" + axes[0][col].get_title(), fontsize=9)

    plt.suptitle(f"SegFormer Oil Spill — {len(samples)} random val samples", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    return out_path


def run_inference(samples):
    """Chay model inference (can transformers)."""
    import torch
    from src.model.segformer import build_model
    from src.data.preprocessing import preprocess_for_inference, postprocess_mask
    from src.utils.checkpoint import CheckpointManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = build_model(pretrained=False)
    ckpt_mgr = CheckpointManager(config.CHECKPOINT_DIR)
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    ckpt_mgr.load(model, ckpt_path, device=device)
    model.to(device).eval()
    print(f"[OK]  Model loaded: {ckpt_path.name}")

    import torch
    with torch.no_grad():
        for item in samples:
            tensor = preprocess_for_inference(item["img_path"], img_size=config.IMG_SIZE)
            logits = model(tensor.to(device))
            item["pred"] = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    return samples


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # Chon split
    if args.split == "val":
        img_dir, mask_dir = config.VAL_IMG_DIR, config.VAL_MASK_DIR
    else:
        img_dir, mask_dir = config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR

    img_paths  = sorted(img_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))

    if not img_paths:
        print(f"[FAIL] Khong tim thay anh trong: {img_dir}")
        sys.exit(1)

    # Random chon n anh
    n = min(args.n, len(img_paths))
    indices = random.sample(range(len(img_paths)), n)
    print(f"[INFO] Split: {args.split} | Total: {len(img_paths)} anh | Chon: {n}")

    samples = []
    for idx in indices:
        img, msk = load_image_mask(img_paths[idx], mask_paths[idx])
        samples.append({
            "img":      img,
            "mask":     msk,
            "name":     img_paths[idx].stem,
            "img_path": img_paths[idx],
        })
        oil_pct = msk.mean() * 100
        print(f"  [{idx:4d}] {img_paths[idx].name:<30s}  oil={oil_pct:.1f}%")

    # Inference (tuy chon)
    if args.with_model:
        try:
            samples = run_inference(samples)
            print("[OK]  Inference done")
        except ImportError as e:
            print(f"[WARN] Can't load model ({e}) — hien thi ground truth only")
            args.with_model = False

    # Ve va luu
    config.PREDICTION_DIR.mkdir(exist_ok=True)
    out_name = f"quick_test_{args.split}_n{n}.png"
    out_path = config.PREDICTION_DIR / out_name
    visualize_samples(samples, out_path, with_pred=args.with_model)

    print(f"\nDone! Mo file de xem:")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()
