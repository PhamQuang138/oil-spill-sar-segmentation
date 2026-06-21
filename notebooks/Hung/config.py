"""
config.py — Single source of truth for all hyperparameters.
deep-sar-segformer: SegFormer Oil Spill Segmentation

Modify values HERE only. All other modules import from this file.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.resolve()
DATASET_ROOT   = PROJECT_ROOT.parent / "datasets" / "deep-sar-oil-spill-segmentation-refined"

TRAIN_IMG_DIR  = DATASET_ROOT / "images" / "images" / "train"
TRAIN_MASK_DIR = DATASET_ROOT / "masks"  / "masks"  / "train"
VAL_IMG_DIR    = DATASET_ROOT / "images" / "images" / "val"
VAL_MASK_DIR   = DATASET_ROOT / "masks"  / "masks"  / "val"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_PATH       = PROJECT_ROOT / "training_log.csv"
PREDICTION_DIR = PROJECT_ROOT / "predictions"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# ── Image ─────────────────────────────────────────────────────────────────────
# SegFormer expects images divisible by 32.
# Phase 1 dev: 256×256.  Final training: 512×512.
IMG_SIZE = (256, 256)   # (H, W)
C_IN     = 1            # SAR grayscale (replicated to 3 channels for ViT patch embed)

# ── SegFormer Model ───────────────────────────────────────────────────────────
# Backbone options: "nvidia/mit-b0" | "mit-b1" | "mit-b2" | "mit-b3" | "mit-b4" | "mit-b5"
# b0 = fastest (3.7M params), b5 = most accurate (84.6M params)
BACKBONE        = "nvidia/mit-b0"
NUM_CLASSES     = 2          # 0: background, 1: oil spill
PRETRAINED      = True       # load ImageNet weights from HuggingFace Hub

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 4          # reduce if OOM; SegFormer needs less than CNN equiv
NUM_WORKERS     = 4          # DataLoader workers (set 0 on Windows if issues arise)
EPOCHS          = 100
MIXED_PRECISION = True       # torch.cuda.amp AMP — disable if GPU doesn't support

# Gradient clipping
GRAD_CLIP_MAX_NORM = 1.0

# ── Optimizer (AdamW) ─────────────────────────────────────────────────────────
LR              = 6e-5       # HF-recommended for SegFormer fine-tuning
WEIGHT_DECAY    = 0.01
BETAS           = (0.9, 0.999)
EPSILON         = 1e-8

# ── LR Scheduler (CosineAnnealingLR) ─────────────────────────────────────────
LR_T_MAX        = EPOCHS     # period of cosine
LR_ETA_MIN      = 1e-7       # minimum LR at end of cycle

# ── Loss ──────────────────────────────────────────────────────────────────────
DICE_SMOOTH     = 1e-6
BCE_POS_WEIGHT  = 3.0        # weight for positive (oil) class — handles imbalance
DICE_WEIGHT     = 0.5        # combined loss = DICE_WEIGHT * Dice + (1-DICE_WEIGHT) * BCE

# ── Data Augmentation ─────────────────────────────────────────────────────────
# Applied to TRAIN split only
AUG_FLIP_P      = 0.5        # horizontal + vertical flip prob
AUG_ROTATE_90_P = 0.5        # random 90° rotation
AUG_ELASTIC_P   = 0.3        # elastic deformation
AUG_GAUSS_NOISE_P = 0.3      # Gaussian noise (SAR speckle simulation)
AUG_BRIGHTNESS_P  = 0.2      # random brightness/contrast

# ── Logging & Checkpointing ───────────────────────────────────────────────────
SAVE_BEST       = True       # save checkpoint when val Dice improves
LOG_EVERY_BATCH = 10         # print batch loss every N batches
EVAL_EVERY_EPOCH = 1         # run validation every N epochs
