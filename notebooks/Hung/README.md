# deep-sar-segformer

> SegFormer-based semantic segmentation of **oil spill regions** in SAR satellite imagery.

## Architecture

| Component | Detail |
|-----------|--------|
| Backbone | SegFormer MiT-B0 → MiT-B5 (HuggingFace `transformers`) |
| Decode Head | All-MLP lightweight decoder (no convolutions) |
| Input | 3-channel (SAR grayscale → RGB duplicated), 512×512 |
| Output | Binary mask: `0` = background, `1` = oil spill |
| Loss | BCEDiceLoss (Dice + weighted BCE for class imbalance) |
| Optimizer | AdamW + CosineAnnealingLR |
| Training | Mixed precision (AMP) + gradient clipping |

## Project Structure

```
deep-sar-segformer/
├── config.py          ← single source of truth for all hyperparameters
├── train.py           ← training entry point
├── evaluate.py        ← validation metrics + visual output
├── predict.py         ← inference on images / folders
├── requirements.txt
├── SKILL.md           ← AI agent context & quick commands
├── .cursorrules       ← coding rules for AI-assisted development
├── src/
│   ├── data/
│   │   ├── dataset.py      ← OilSpillDataset (PyTorch Dataset)
│   │   └── transforms.py   ← Albumentations augmentation pipelines
│   ├── model/
│   │   ├── segformer.py    ← OilSpillSegFormer wrapper
│   │   └── losses.py       ← DiceLoss, BCEDiceLoss
│   ├── training/
│   │   ├── trainer.py      ← Trainer (AMP, grad clip, checkpointing)
│   │   └── metrics.py      ← IoU, Dice, Precision, Recall
│   └── utils/
│       ├── checkpoint.py   ← save/load best checkpoint
│       └── visualization.py← overlay, comparison, training curves
└── tests/
    ├── test_dataset.py
    ├── test_model.py
    └── test_metrics.py
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run unit tests (sanity check, no GPU needed for metrics/model-shape tests)
```bash
pytest tests/ -v
```

### 3. Train
```bash
# Default: MiT-B0, 100 epochs, 512×512
python train.py

# MiT-B2 variant, 150 epochs
python train.py --backbone nvidia/mit-b2 --epochs 150 --batch-size 4
```

### 4. Evaluate
```bash
python evaluate.py --save-visuals
```

### 5. Predict
```bash
python predict.py --input path/to/sar_image.png
python predict.py --input path/to/folder/
```

## Dataset

`../datasets/deep-sar-oil-spill-segmentation-refined/`
```
images/images/train/   SAR PNG images (training)
images/images/val/     SAR PNG images (validation)
masks/masks/train/     binary masks  (training)
masks/masks/val/       binary masks  (validation)
```

Masks: `0` = background, `255` = oil spill (binarised to {0,1} in Dataset).

## Backbone Options

| ID | Params | Speed | Accuracy |
|----|--------|-------|----------|
| `nvidia/mit-b0` | 3.7M | ⚡⚡⚡ | Baseline |
| `nvidia/mit-b1` | 13.7M | ⚡⚡ | +2 mIoU |
| `nvidia/mit-b2` | 24.7M | ⚡⚡ | +3 mIoU |
| `nvidia/mit-b3` | 44.1M | ⚡ | +4 mIoU |
| `nvidia/mit-b5` | 84.6M | 🐢 | Best |

Change in `config.py`:
```python
BACKBONE = "nvidia/mit-b2"
```

## Key Hyperparameters (`config.py`)

```python
IMG_SIZE        = (512, 512)    # resize target
BACKBONE        = "nvidia/mit-b0"
BATCH_SIZE      = 4
EPOCHS          = 100
LR              = 6e-5          # AdamW
BCE_POS_WEIGHT  = 3.0           # oil class weight (handles imbalance)
DICE_WEIGHT     = 0.5           # blend: 0.5*Dice + 0.5*BCE
MIXED_PRECISION = True          # AMP (disable if CPU-only)
```
