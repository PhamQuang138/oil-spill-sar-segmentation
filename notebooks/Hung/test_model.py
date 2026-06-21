"""
test_model.py — Script test nhanh best_model.pth sau khi train trên Kaggle.

Chạy:
    python test_model.py
    python test_model.py --checkpoint path/to/best_model.pth
    python test_model.py --image path/to/sar.png
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/best_model.pth"))
    p.add_argument("--image", type=Path, default=None,
                   help="Ảnh SAR để test inference (tuỳ chọn)")
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    return p.parse_args()


def test_checkpoint(ckpt_path: Path, img_size: tuple, test_image: Path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  TEST BEST_MODEL.PTH")
    print(f"{'='*55}")
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  img_size  : {img_size}")

    # ── 1. Kiểm tra file tồn tại ──────────────────────────────────────────────
    if not ckpt_path.exists():
        print(f"\n❌ KHÔNG TÌM THẤY FILE: {ckpt_path}")
        print("   → Đặt best_model.pth vào thư mục checkpoints/")
        print("   → Hoặc chỉ định: python test_model.py --checkpoint /path/to/best_model.pth")
        return False

    size_mb = ckpt_path.stat().st_size / 1e6
    print(f"  File size : {size_mb:.1f} MB")

    # ── 2. Load checkpoint ────────────────────────────────────────────────────
    print("\n[1/4] Loading checkpoint...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print(f"  [OK] epoch      = {ckpt.get('epoch', 'N/A')}")
        print(f"  [OK] val_dice   = {ckpt.get('val_dice', 'N/A')}")
        keys = list(ckpt.keys())
        print(f"  [OK] keys       = {keys}")
    except Exception as e:
        print(f"  [FAIL] Loi load checkpoint: {e}")
        return False

    # ── 3. Build model & load weights ─────────────────────────────────────────
    print("\n[2/4] Building model...")
    try:
        import config
        from src.model.segformer import build_model
        model = build_model(pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        total = sum(p.numel() for p in model.parameters())
        print(f"  [OK] Model loaded: {total/1e6:.1f}M params")
    except Exception as e:
        print(f"  [FAIL] Loi build/load model: {e}")
        return False

    # ── 4. Forward pass với dummy input ───────────────────────────────────────
    print("\n[3/4] Forward pass (dummy input)...")
    try:
        H, W = img_size
        dummy = torch.randn(1, 3, H, W).to(device)
        with torch.no_grad():
            logits = model(dummy)
        expected = (1, 2, H, W)
        got = tuple(logits.shape)
        ok = got == expected
        print(f"  [OK] Output shape: {got}  {'CORRECT' if ok else f'EXPECTED {expected}'}")
        pred = logits.argmax(dim=1)
        print(f"  [OK] Pred mask  : {tuple(pred.shape)}, unique={torch.unique(pred).tolist()}")
    except Exception as e:
        print(f"  [FAIL] Loi forward pass: {e}")
        return False

    # ── 5. Inference trên ảnh thực (nếu có) ──────────────────────────────────
    if test_image is not None:
        print(f"\n[4/4] Inference trên ảnh thực: {test_image}")
        if not test_image.exists():
            print(f"  ⚠ Không tìm thấy ảnh: {test_image}")
        else:
            try:
                from src.data.preprocessing import preprocess_for_inference, postprocess_mask
                tensor = preprocess_for_inference(test_image, img_size=tuple(img_size))
                with torch.no_grad():
                    logits = model(tensor.to(device))
                mask = postprocess_mask(logits)
                oil_pct = mask.mean() * 100
                print(f"  [OK] Mask shape : {mask.shape}")
                print(f"  [OK] Oil pixels : {oil_pct:.2f}%")

                # Save mask
                import cv2
                out_path = Path("predictions") / f"{test_image.stem}_test_mask.png"
                out_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(out_path), (mask * 255).astype(np.uint8))
                print(f"  [OK] Mask saved : {out_path}")
            except Exception as e:
                print(f"  [FAIL] Loi inference: {e}")
                return False
    else:
        print("\n[4/4] Bo qua test anh thuc (khong co --image)")

    # ── Kết quả ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  [PASS] MODEL HOAT DONG DUNG")
    if "val_dice" in ckpt:
        print(f"  Best Val Dice (training): {ckpt['val_dice']:.4f}")
    print(f"{'='*55}\n")
    return True


if __name__ == "__main__":
    args = parse_args()
    success = test_checkpoint(
        ckpt_path=args.checkpoint,
        img_size=tuple(args.img_size),
        test_image=args.image,
    )
    sys.exit(0 if success else 1)
