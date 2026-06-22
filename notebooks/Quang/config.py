"""
config.py — Single source of truth for all hyperparameters.
DeepLabV3+ Oil Spill Segmentation (Balanced Version)
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.resolve()
DATASET_ROOT   = Path(r"D:\HocSAu\dataset")

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
IMG_SIZE = (256, 256)
C_IN     = 3

# ── Model (DeepLabV3+) ────────────────────────────────────────────────────────
BACKBONE        = "resnet34"
NUM_CLASSES     = 1
PRETRAINED      = True

# ── Training (Windows Specifics) ──────────────────────────────────────────────
BATCH_SIZE      = 4          # Batch = 4 cần LR nhỏ hơn
NUM_WORKERS     = 0
EPOCHS          = 100
PATIENCE        = 15
MIXED_PRECISION = True

GRAD_CLIP_MAX_NORM = 1.0

# ── Optimizer (AdamW) ─────────────────────────────────────────────────────────
LR              = 6e-5       # Đã giảm từ 1e-4 về 6e-5 để phù hợp Batch 4
WEIGHT_DECAY    = 0.01
BETAS           = (0.9, 0.999)
EPSILON         = 1e-8

# ── LR Scheduler (CosineAnnealingLR) ─────────────────────────────────────────
LR_T_MAX        = EPOCHS
LR_ETA_MIN      = 1e-7

# ── Loss ──────────────────────────────────────────────────────────────────────
DICE_SMOOTH     = 1e-6
BCE_POS_WEIGHT  = 2.0        # Tăng lên 2.0 để mô hình tập trung khoanh vùng dầu
DICE_WEIGHT     = 0.5

# ── Data Augmentation (Đã nới lỏng) ──────────────────────────────────────────
AUG_FLIP_P        = 0.5      # Giữ nguyên lật/xoay (Augmentation cơ bản, tốt cho không gian)
AUG_ROTATE_90_P   = 0.5
AUG_ELASTIC_P     = 0.0      # Tắt hẳn bóp méo hình ảnh (Giúp AI học đặc trưng tự nhiên)
AUG_GAUSS_NOISE_P = 0.1      # Giảm nhiễu mạnh (Chỉ giữ lại 10% để chống học vẹt nhẹ)
AUG_BRIGHTNESS_P  = 0.2

# ── Logging & Checkpointing ───────────────────────────────────────────────────
SAVE_BEST        = True
LOG_EVERY_BATCH  = 10
EVAL_EVERY_EPOCH = 1