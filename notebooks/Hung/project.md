# Project Documentation — deep-sar-segformer
## SegFormer-based Oil Spill Detection in SAR Satellite Imagery

> **Mục đích file này**: Tài liệu kỹ thuật đầy đủ để hiểu dự án và viết báo cáo.
> Cập nhật cùng code khi có thay đổi lớn.

---

## 1. Tổng quan (Overview)

### 1.1 Bài toán

Phát hiện và phân vùng **vết dầu loang** (oil spill) trên ảnh **SAR (Synthetic Aperture Radar)**
vệ tinh. Đây là bài toán **binary semantic segmentation**:

| Nhãn | Giá trị mask | Ý nghĩa |
|------|-------------|---------|
| 0 | 0 | Nền (biển, đất) |
| 1 | 255 (→ 1) | Vùng dầu loang |

### 1.2 Tại sao dùng SAR?

- SAR chụp được trong mọi điều kiện thời tiết (mây, đêm) — ảnh quang học (optical)
  không thể làm điều này.
- Dầu làm mặt biển nhẵn → tín hiệu radar thấp (dark spot trên ảnh SAR).
- Đây là tiêu chuẩn quốc tế trong giám sát ô nhiễm biển.

### 1.3 Tại sao dùng SegFormer?

- **CNN** (U-Net, DeepLab): context cục bộ, bỏ qua phụ thuộc tầm xa.
- **SegFormer** (Vision Transformer + MLP Decoder): Hierarchical Transformer encoder
  giúp học được cả đặc trưng cục bộ *và* global context trong một lần forward.
- Lightweight MLP decode head → nhẹ hơn ASPP hay FPN.
- Pretrained trên ImageNet → transfer learning hiệu quả ngay cả với dataset nhỏ.

---

## 2. Dataset

### 2.1 Nguồn gốc

**Kaggle**: [Deep SAR Oil Spill Segmentation (Refined)](https://www.kaggle.com/datasets/your-dataset-slug)

Tên thư mục: `deep-sar-oil-spill-segmentation-refined`

### 2.2 Cấu trúc

```
datasets/
└── deep-sar-oil-spill-segmentation-refined/
    ├── images/images/
    │   ├── train/    ← ảnh SAR PNG (training)
    │   └── val/      ← ảnh SAR PNG (validation)
    └── masks/masks/
        ├── train/    ← binary mask PNG (training)
        └── val/      ← binary mask PNG (validation)
```

### 2.3 Đặc điểm ảnh

| Thuộc tính | Giá trị |
|-----------|---------|
| Định dạng | PNG 8-bit |
| Kênh màu gốc | RGB (R=G=B → thực chất grayscale) |
| Đọc thực tế | Grayscale → `cv2.IMREAD_GRAYSCALE` |
| Kênh đầu vào model | **3** (grayscale nhân đôi sang RGB để dùng pretrained weights) |
| Kích thước gốc | Thay đổi (resize về 512×512 khi training) |

### 2.4 Phân bố nhãn (Class Imbalance)

Vùng dầu loang chiếm tỷ lệ nhỏ trong hầu hết ảnh → **class imbalance** là thách thức chính.

Giải pháp: BCE với `pos_weight = 3.0` và Dice loss (không nhạy cảm với imbalance).

---

## 3. Kiến trúc mô hình (Architecture)

### 3.1 SegFormer tổng quan

```
Input SAR (B, 3, H, W)
        │
        ▼
┌─────────────────────────┐
│  Mix Transformer (MiT)  │  ← Hierarchical Transformer Encoder
│  Encoder backbone        │     Patch sizes: 4, 8, 16, 32
│  Stages: 4 stages        │     Output: C1, C2, C3, C4 feature maps
└──────────┬──────────────┘
           │  (B, C1, H/4, W/4)
           │  (B, C2, H/8, W/8)
           │  (B, C3, H/16, W/16)
           │  (B, C4, H/32, W/32)
           ▼
┌─────────────────────────┐
│  All-MLP Decode Head    │  ← Lightweight: chỉ dùng Linear layers
│  (no convolutions)      │     Unifies multi-scale features
│                         │     Output: (B, num_classes, H/4, W/4)
└──────────┬──────────────┘
           │
           ▼  bilinear upsample ×4
    (B, num_classes, H, W)
           │
           ▼  argmax(dim=1)
      Binary mask (B, H, W)
```

### 3.2 Backbone variants

| Backbone | Params | GFLOPs | Ghi chú |
|----------|--------|--------|---------|
| `nvidia/mit-b0` | 3.7M | 8.4 | Phát triển & thử nghiệm nhanh |
| `nvidia/mit-b1` | 13.7M | 15.9 | Cân bằng tốc độ/độ chính xác |
| `nvidia/mit-b2` | 24.7M | 62.4 | **Khuyến nghị** cho training đầy đủ |
| `nvidia/mit-b3` | 44.1M | 79.0 | Accuracy cao, cần GPU >8GB |
| `nvidia/mit-b5` | 84.6M | 182.0 | State-of-the-art, cần GPU lớn |

### 3.3 Input/Output specs

```python
# Input
pixel_values: torch.Tensor  # shape (B, 3, H, W), float32
                             # normalized: mean=(0.485,0.456,0.406)
                             #             std=(0.229,0.224,0.225)

# Output (raw logits, chưa softmax)
logits: torch.Tensor        # shape (B, 2, H, W), float32

# Predicted mask
pred_mask = logits.argmax(dim=1)  # shape (B, H, W), int64, values {0, 1}
```

---

## 4. Tiền xử lý & Tăng cường dữ liệu (Preprocessing & Augmentation)

### 4.1 Pipeline tiền xử lý (`src/data/preprocessing.py`)

Mọi ảnh đều qua các bước sau theo thứ tự:

```
1. Đọc ảnh grayscale        cv2.IMREAD_GRAYSCALE
2. Resize                   (512, 512) — INTER_LINEAR cho ảnh, INTER_NEAREST cho mask
3. Binarize mask            pixel > 127 → 1, else → 0
4. Gray → RGB               cv2.COLOR_GRAY2RGB   (để dùng ImageNet pretrained)
5. Normalize                mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)
6. ToTensor                 (H,W,C) → (C,H,W), float32
```

### 4.2 Augmentation (chỉ training)

| Augmentation | Xác suất | Lý do |
|-------------|---------|-------|
| Horizontal Flip | 0.5 | SAR không phân biệt trái/phải |
| Vertical Flip | 0.5 | Không có "đầu" hay "cuối" trên ảnh vệ tinh |
| RandomRotate90 | 0.5 | Góc nhìn vệ tinh có thể khác nhau |
| ElasticTransform | 0.3 | Biên dạng của vết dầu thay đổi theo sóng biển |
| GaussNoise | 0.3 | Mô phỏng speckle noise đặc trưng của SAR |
| RandomBrightnessContrast | 0.2 | Thay đổi điều kiện ánh sáng |

---

## 5. Hàm Loss

### 5.1 Dice Loss

$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum p_i \cdot g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}$$

Trong đó:
- $p_i$ = xác suất dự đoán (sau softmax)
- $g_i$ = nhãn ground truth (one-hot)
- $\epsilon$ = `1e-6` (Laplace smoothing)

**Ưu điểm**: Không nhạy cảm với class imbalance, tối ưu trực tiếp Dice coefficient.

### 5.2 Binary Cross-Entropy (với pos_weight)

$$\mathcal{L}_{BCE} = -[w_{pos} \cdot g \cdot \log(p) + (1-g) \cdot \log(1-p)]$$

`pos_weight = 3.0`: phạt nặng hơn khi bỏ sót vùng dầu (false negative).

### 5.3 Combined Loss

$$\mathcal{L}_{combined} = \lambda \cdot \mathcal{L}_{Dice} + (1-\lambda) \cdot \mathcal{L}_{BCE}$$

Mặc định $\lambda = 0.5$.

---

## 6. Huấn luyện (Training)

### 6.1 Hyperparameters mặc định

| Hyperparameter | Giá trị | Lý do |
|----------------|--------|-------|
| Optimizer | AdamW | Chuẩn cho Transformer |
| Learning Rate | 6e-5 | Khuyến nghị của HuggingFace SegFormer |
| Weight Decay | 0.01 | Regularization cho Transformer |
| Scheduler | CosineAnnealingLR | Giảm LR mượt mà, không plateau |
| Epochs | 100 | Đủ để hội tụ với pretrained weights |
| Batch Size | 4 | Giới hạn bởi VRAM GPU |
| Mixed Precision | True | Tăng tốc 1.5-2× trên GPU NVIDIA |
| Grad Clip | 1.0 | Tránh exploding gradients ở Transformer |

### 6.2 Luồng training một epoch

```
for batch in train_loader:
    1. images, masks → GPU
    2. optimizer.zero_grad()
    3. [AMP autocast]
       logits = model(images)           → (B, 2, H, W)
       loss = BCEDiceLoss(logits, masks)
    4. scaler.scale(loss).backward()
    5. clip_grad_norm_(max_norm=1.0)
    6. optimizer.step() + scaler.update()
```

### 6.3 Checkpoint strategy

- Lưu `best_model.pth` khi `mean_dice` trên validation cải thiện.
- Log mọi epoch vào `training_log.csv`.

---

## 7. Metrics đánh giá

### 7.1 Định nghĩa

| Metric | Công thức | Ý nghĩa |
|--------|----------|---------|
| IoU (Jaccard) | TP / (TP + FP + FN) | Độ chồng lấp giữa dự đoán và ground truth |
| Dice (F1) | 2·TP / (2·TP + FP + FN) | Tương tự IoU nhưng ít phạt FP/FN hơn |
| Precision | TP / (TP + FP) | Bao nhiêu % dự đoán oil là đúng |
| Recall | TP / (TP + FN) | Bao nhiêu % oil thật được tìm thấy |

**Metric chính**: `iou_oil` và `dice_oil` — chỉ tính trên lớp dầu (class 1).

### 7.2 Mục tiêu (target)

| Metric | Mục tiêu tối thiểu | Mục tiêu tốt |
|--------|-------------------|-------------|
| IoU oil | ≥ 0.50 | ≥ 0.70 |
| Dice oil | ≥ 0.60 | ≥ 0.80 |
| mIoU | ≥ 0.60 | ≥ 0.75 |

---

## 8. Cấu trúc code

```
deep-sar-segformer/
├── config.py              ← Mọi hyperparameter — chỉ thay đổi ở đây
├── train.py               ← Entry point huấn luyện
├── evaluate.py            ← Đánh giá trên val set
├── predict.py             ← Inference với API chuẩn: predict(image_path, img_size)
├── project.md             ← File này
├── SKILL.md               ← Hướng dẫn nhanh cho AI agents
├── README.md              ← Tổng quan + Quick Start
├── requirements.txt       ← Dependencies
├── .cursorrules           ← Quy tắc code cho AI-assisted development
│
├── src/
│   ├── data/
│   │   ├── dataset.py         ← OilSpillDataset (PyTorch Dataset class)
│   │   ├── transforms.py      ← Albumentations augmentation pipelines
│   │   └── preprocessing.py   ← Hàm tiền xử lý thuần túy (không phụ thuộc PyTorch)
│   │
│   ├── model/
│   │   ├── segformer.py       ← OilSpillSegFormer (wrapper của HuggingFace)
│   │   └── losses.py          ← DiceLoss, BCEDiceLoss
│   │
│   ├── training/
│   │   ├── trainer.py         ← Trainer class (AMP + grad clip + logging)
│   │   └── metrics.py         ← IoU, Dice, Precision, Recall + Accumulator
│   │
│   └── utils/
│       ├── checkpoint.py      ← save/load best_model.pth
│       └── visualization.py   ← overlay, comparison panels, training curves
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_metrics.py
│
└── notebooks/
    └── kaggle_segformer_oilspill.ipynb  ← Notebook chạy trên Kaggle
```

### 8.1 Luồng dữ liệu (Data Flow)

```
PNG files on disk
      │
      ▼ OilSpillDataset.__getitem__()
      │  └── preprocessing.load_sar_image()    ← đọc + resize
      │  └── preprocessing.binarize_mask()     ← threshold mask
      │  └── preprocessing.to_rgb()            ← gray → 3ch
      │
      ▼ Albumentations transform
      │  └── Augmentation (train only)
      │  └── Normalize (ImageNet stats)
      │  └── ToTensorV2
      │
      ▼ DataLoader (batch)
      │
      ▼ OilSpillSegFormer.forward()
      │  └── HuggingFace MiT encoder
      │  └── MLP decode head
      │  └── F.interpolate (upsample ×4)
      │
      ▼ BCEDiceLoss
      │
      ▼ AdamW.step()
```

---

## 9. Hướng dẫn sử dụng (Usage Guide)

### 9.1 Cài đặt

```bash
pip install -r requirements.txt
```

### 9.2 Training

```bash
# Default (MiT-B0, 100 epochs, 512×512)
python train.py

# Custom backbone và epochs
python train.py --backbone nvidia/mit-b2 --epochs 150 --batch-size 4

# Resume từ checkpoint
python train.py --resume checkpoints/best_model.pth
```

### 9.3 Đánh giá

```bash
python evaluate.py
python evaluate.py --checkpoint checkpoints/best_model.pth --save-visuals
```

### 9.4 Inference

```python
# API chuẩn
from predict import predict, predict_batch

# Dự đoán 1 ảnh
mask = predict("path/to/sar.png", img_size=(512, 512))

# Dự đoán cả thư mục
masks = predict_batch("path/to/folder/", img_size=(512, 512))
```

```bash
# Command line
python predict.py --input path/to/image.png --output predictions/
```

---

## 10. Môi trường chạy (Environment)

### 10.1 Local (Windows)

```
Python    : 3.10+
CUDA      : 11.8+ (nếu có GPU)
GPU đề xuất: RTX 3060 (6GB) trở lên
NUM_WORKERS: 0 (Windows multiprocessing limit)
```

### 10.2 Kaggle

```
GPU       : Tesla P100 (16GB) hoặc T4 (15GB)
Dataset   : Thêm từ Kaggle Datasets (không cần tải về)
Notebook  : notebooks/kaggle_segformer_oilspill.ipynb
NUM_WORKERS: 2
```

---

## 11. Kết quả thực nghiệm (Experimental Results)

> *Cập nhật bảng này sau khi training xong.*

| Model | Backbone | Epochs | img_size | IoU oil | Dice oil | mIoU | Notes |
|-------|----------|--------|----------|---------|---------|------|-------|
| — | mit-b0 | — | 512×512 | — | — | — | baseline |
| — | mit-b2 | — | 512×512 | — | — | — | |

---

## 12. Tài liệu tham khảo (References)

1. **SegFormer**: Xie, E., Wang, W., Yu, Z., et al. (2021).
   *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.*
   NeurIPS 2021. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)

2. **Mix Transformer (MiT)**: Backbone của SegFormer — Hierarchical Vision Transformer
   với overlapping patch embedding và efficient self-attention.

3. **Dice Loss**: Milletari, F., Navab, N., Ahmadi, S. A. (2016).
   *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.*
   3DV 2016.

4. **Dataset**: Kaggle — *Deep SAR Oil Spill Segmentation (Refined)*

5. **SAR Oil Spill Detection**: Fingas, M., Brown, C. (2018).
   *A Review of Oil Spill Remote Sensing.* Sensors, 18(1), 91.

6. **AdamW**: Loshchilov, I., Hutter, F. (2019).
   *Decoupled Weight Decay Regularization.* ICLR 2019.

7. **Albumentations**: Buslaev, A., et al. (2020).
   *Albumentations: Fast and Flexible Image Augmentations.* Information, 11(2), 125.

---

## 13. Ghi chú phát triển (Dev Notes)

### Known Issues / Gotchas

- **Windows + DataLoader**: `NUM_WORKERS > 0` cần `if __name__ == '__main__':` guard.
  Trên Windows nên đặt `NUM_WORKERS = 0` nếu gặp lỗi.
- **`ignore_mismatched_sizes=True`**: Cần thiết vì HuggingFace load pretrained decoder
  head cho 150 classes (ADE20K), nhưng ta chỉ dùng 2 classes.
- **Mask binarization threshold**: 127 (không phải 128) để handle JPEG artifact nếu có.
- **GradScaler on CPU**: `MIXED_PRECISION=True` tự động disable khi không có CUDA.

### Changelog

| Phiên bản | Ngày | Thay đổi |
|-----------|------|---------|
| v1.0 | 2026-06-19 | Setup khung dự án: SegFormer + BCEDiceLoss + Trainer |
| v1.1 | 2026-06-19 | Tách preprocessing, chuẩn hóa predict API, Kaggle notebook |
