"""Evaluate threshold search, probability ensemble, and post-processing.

This script uses existing checkpoints and validation data. It does not train.

Example:
    .venv\\Scripts\\python.exe src\\evaluation\\development_eval.py
    .venv\\Scripts\\python.exe src\\evaluation\\development_eval.py --limit 32
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.hung_adapter import HungPredictor
from src.inference.khoa_adapter import KhoaPredictor
from src.inference.postprocess import postprocess_mask
from src.inference.quang_adapter import QuangPredictor


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Counts:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred_b = pred.astype(bool)
        target_b = target.astype(bool)
        self.tp += np.logical_and(pred_b, target_b).sum()
        self.fp += np.logical_and(pred_b, ~target_b).sum()
        self.fn += np.logical_and(~pred_b, target_b).sum()
        self.tn += np.logical_and(~pred_b, ~target_b).sum()

    def metrics(self) -> dict[str, float]:
        eps = 1e-7
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }


class OilSpillValDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, image_size: int = 256, limit: int | None = None):
        image_paths = sorted(image_dir.glob("*.png"))
        mask_by_name = {path.name: path for path in mask_dir.glob("*.png")}
        pairs = [(img, mask_by_name[img.name]) for img in image_paths if img.name in mask_by_name]
        if limit:
            pairs = pairs[:limit]
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"Cannot read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"Cannot read mask: {mask_path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        image_norm = (image_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
        return tensor, torch.from_numpy(mask), image_rgb, image_path.name


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def predict_batch(model_name: str, predictor, images: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        if model_name == "SegFormer":
            outputs = predictor.model(pixel_values=images.to(device))
            logits = F.interpolate(outputs.logits, size=(256, 256), mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:
            logits = predictor.model(images.to(device))
            probs = torch.sigmoid(logits).squeeze(1)
    return probs.detach().cpu().numpy()


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_threshold_curves(rows: list[dict[str, object]], output_path: Path) -> None:
    models = sorted({str(row["model"]) for row in rows if row["variant"] == "raw"})
    plt.figure(figsize=(10, 6))
    for model in models:
        model_rows = [row for row in rows if row["model"] == model and row["variant"] == "raw"]
        xs = [float(row["threshold"]) for row in model_rows]
        ys = [float(row["dice"]) for row in model_rows]
        plt.plot(xs, ys, marker="o", linewidth=2, label=model)
    plt.xlabel("Threshold")
    plt.ylabel("Dice")
    plt.title("Threshold search on validation set")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_best_metrics(best_rows: list[dict[str, object]], output_path: Path) -> None:
    labels = [f"{row['model']}\n{row['variant']}" for row in best_rows]
    dice = [float(row["dice"]) for row in best_rows]
    iou = [float(row["iou"]) for row in best_rows]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, dice, width, label="Dice")
    plt.bar(x + width / 2, iou, width, label="IoU")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Best validation metrics")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = image_rgb.copy()
    mark = base.copy()
    mark[mask.astype(bool)] = color
    return cv2.addWeighted(base, 0.62, mark, 0.38, 0)


def save_visual_grid(
    image_rgb: np.ndarray,
    target: np.ndarray,
    raw_mask: np.ndarray,
    post_mask: np.ndarray,
    ensemble_mask: np.ndarray,
    output_path: Path,
) -> None:
    panels = [
        ("SAR", image_rgb),
        ("Ground truth", overlay(image_rgb, target, (30, 220, 30))),
        ("Best raw", overlay(image_rgb, raw_mask, (255, 60, 60))),
        ("Post-process", overlay(image_rgb, post_mask, (255, 180, 40))),
        ("Ensemble", overlay(image_rgb, ensemble_mask, (80, 160, 255))),
    ]
    plt.figure(figsize=(15, 4))
    for idx, (title, panel) in enumerate(panels, 1):
        plt.subplot(1, len(panels), idx)
        plt.imshow(panel)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def best_by_dice(rows: list[dict[str, object]], model: str, variant: str = "raw") -> dict[str, object]:
    candidates = [row for row in rows if row["model"] == model and row["variant"] == variant]
    return max(candidates, key=lambda row: float(row["dice"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "development_experiments")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--num-visuals", type=int, default=6)
    parser.add_argument("--threshold-start", type=float, default=0.25)
    parser.add_argument("--threshold-end", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)
    visual_dir = ensure_dir(output_dir / "visuals")

    thresholds = np.round(
        np.arange(args.threshold_start, args.threshold_end + 1e-9, args.threshold_step),
        2,
    ).tolist()

    dataset = OilSpillValDataset(
        image_dir=args.dataset_root / "images" / "images" / "val",
        mask_dir=args.dataset_root / "masks" / "masks" / "val",
        image_size=256,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    predictors = {
        "DeepLabV3+": QuangPredictor(str(args.weights_dir / "quang_best_deeplabv3plus_checkpoint.pth"), str(device)),
        "SegFormer": HungPredictor(str(args.weights_dir / "Hung_best_segformer_checkpoint.pth"), str(device)),
        "UNet++": KhoaPredictor(str(args.weights_dir / "Khoa_best_Unetplusplus_checkpoint.pth"), str(device)),
    }

    for predictor in predictors.values():
        predictor.model.eval()

    counts: dict[tuple[str, str, float], Counts] = {}
    for model_name in [*predictors.keys(), "Ensemble"]:
        for threshold in thresholds:
            counts[(model_name, "raw", threshold)] = Counts()

    first_visuals: list[dict[str, object]] = []

    print(f"Device: {device}")
    print(f"Validation images: {len(dataset)}")
    print(f"Thresholds: {thresholds}")

    for images, masks, image_rgbs, names in tqdm(loader, desc="Validation inference"):
        masks_np = masks.numpy().astype(np.uint8)
        probs_by_model = {}
        for model_name, predictor in predictors.items():
            probs_by_model[model_name] = predict_batch(model_name, predictor, images, device)

        probs_by_model["Ensemble"] = np.mean(
            [probs_by_model["DeepLabV3+"], probs_by_model["SegFormer"], probs_by_model["UNet++"]],
            axis=0,
        )

        for model_name, probs in probs_by_model.items():
            for threshold in thresholds:
                preds = probs >= threshold
                for pred, target in zip(preds, masks_np):
                    counts[(model_name, "raw", threshold)].update(pred, target)

        if len(first_visuals) < args.num_visuals:
            batch_size = masks_np.shape[0]
            for i in range(batch_size):
                if len(first_visuals) >= args.num_visuals:
                    break
                first_visuals.append(
                    {
                        "name": names[i],
                        "image": image_rgbs[i].numpy().astype(np.uint8),
                        "target": masks_np[i],
                        "probs": {key: value[i] for key, value in probs_by_model.items()},
                    }
                )

    rows: list[dict[str, object]] = []
    for (model_name, variant, threshold), count in counts.items():
        metric = count.metrics()
        rows.append(
            {
                "model": model_name,
                "variant": variant,
                "threshold": threshold,
                **{key: round(value, 6) for key, value in metric.items()},
                "tp": int(count.tp),
                "fp": int(count.fp),
                "fn": int(count.fn),
                "tn": int(count.tn),
            }
        )
    rows.sort(key=lambda row: (str(row["model"]), str(row["variant"]), float(row["threshold"])))

    raw_best_rows = [best_by_dice(rows, model) for model in ["DeepLabV3+", "SegFormer", "UNet++", "Ensemble"]]

    # Post-processing is evaluated only at each model's best raw threshold.
    post_counts: dict[str, Counts] = {str(row["model"]): Counts() for row in raw_best_rows}
    best_thresholds = {str(row["model"]): float(row["threshold"]) for row in raw_best_rows}

    for images, masks, _, _ in tqdm(loader, desc="Post-process evaluation"):
        masks_np = masks.numpy().astype(np.uint8)
        probs_by_model = {}
        for model_name, predictor in predictors.items():
            probs_by_model[model_name] = predict_batch(model_name, predictor, images, device)
        probs_by_model["Ensemble"] = np.mean(
            [probs_by_model["DeepLabV3+"], probs_by_model["SegFormer"], probs_by_model["UNet++"]],
            axis=0,
        )

        for model_name, probs in probs_by_model.items():
            threshold = best_thresholds[model_name]
            for prob, target in zip(probs, masks_np):
                raw_mask = (prob >= threshold).astype(np.uint8)
                cleaned = postprocess_mask(raw_mask, min_area=args.min_area)
                post_counts[model_name].update(cleaned, target)

    post_rows = []
    for model_name, count in post_counts.items():
        metric = count.metrics()
        post_rows.append(
            {
                "model": model_name,
                "variant": "postprocess",
                "threshold": best_thresholds[model_name],
                **{key: round(value, 6) for key, value in metric.items()},
                "tp": int(count.tp),
                "fp": int(count.fp),
                "fn": int(count.fn),
                "tn": int(count.tn),
            }
        )

    all_rows = rows + post_rows
    fieldnames = ["model", "variant", "threshold", "iou", "dice", "precision", "recall", "accuracy", "tp", "fp", "fn", "tn"]
    save_csv(output_dir / "threshold_search_results.csv", all_rows, fieldnames)
    save_csv(output_dir / "best_results.csv", raw_best_rows + post_rows, fieldnames)

    plot_threshold_curves(rows, output_dir / "threshold_dice_curves.png")
    plot_best_metrics(raw_best_rows + post_rows, output_dir / "best_metrics_comparison.png")

    best_raw_model = max(raw_best_rows[:3], key=lambda row: float(row["dice"]))
    best_raw_model_name = str(best_raw_model["model"])
    best_raw_threshold = float(best_raw_model["threshold"])
    ensemble_threshold = best_thresholds["Ensemble"]

    for idx, item in enumerate(first_visuals):
        image = item["image"]
        target = item["target"]
        probs = item["probs"]
        raw_mask = (probs[best_raw_model_name] >= best_raw_threshold).astype(np.uint8)
        post_mask = postprocess_mask(raw_mask, min_area=args.min_area)
        ensemble_mask = (probs["Ensemble"] >= ensemble_threshold).astype(np.uint8)
        save_visual_grid(
            image,
            target,
            raw_mask,
            post_mask,
            ensemble_mask,
            visual_dir / f"development_visual_{idx + 1:02d}_{item['name']}.png",
        )

    print("\nBest raw results by Dice:")
    for row in raw_best_rows:
        print(
            f"  {row['model']:10s} threshold={float(row['threshold']):.2f} "
            f"Dice={float(row['dice']):.4f} IoU={float(row['iou']):.4f} "
            f"P={float(row['precision']):.4f} R={float(row['recall']):.4f}"
        )

    print("\nPost-processing at best raw threshold:")
    for row in post_rows:
        print(
            f"  {row['model']:10s} threshold={float(row['threshold']):.2f} "
            f"Dice={float(row['dice']):.4f} IoU={float(row['iou']):.4f} "
            f"P={float(row['precision']):.4f} R={float(row['recall']):.4f}"
        )

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
