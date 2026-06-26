# Cấu Trúc Notebook: UNet++ Phát Hiện Vết Dầu Tràn Trên Ảnh SAR

## 1. Mục đích tài liệu

Tài liệu này mô tả cấu trúc chuẩn của notebook `.ipynb` dùng để huấn luyện, đánh giá và suy luận mô hình **UNet++ với encoder EfficientNet-B4** cho bài toán phân vùng vết dầu tràn trên ảnh SAR.

Notebook được thiết kế để chạy trên **Kaggle GPU P100**, sử dụng PyTorch và `segmentation_models_pytorch`. Ảnh SAR gốc có kích thước `416 x 416`, 3 kênh; toàn bộ pipeline chuẩn hóa đầu vào về `256 x 256` cho cả training, evaluation và inference.

| Thuộc tính | Giá trị |
|---|---|
| Bài toán | Binary semantic segmentation |
| Kiến trúc | UNet++ |
| Encoder | EfficientNet-B4 |
| Framework | PyTorch + segmentation_models_pytorch |
| Kích thước input thống nhất | `256 x 256` |
| Ngưỡng phân lớp | `0.5`, khai báo duy nhất trong `CONFIG` |
| Thiết bị thực thi | Kaggle GPU P100 |

---

## 2. Tổng quan thứ tự cell

| Cell | Nội dung | Kết quả chính |
|---:|---|---|
| 1 | Cài đặt thư viện và import | Môi trường sẵn sàng |
| 2 | `CONFIG` tập trung | Một nguồn cấu hình duy nhất |
| 3 | Fix seed | Kết quả có thể tái lập |
| 4 | Load và EDA dataset | Thống kê, ảnh mẫu, kiểm tra dữ liệu |
| 5 | Pipeline tiền xử lý chung | Albumentations dùng chung train/inference |
| 6 | Dataset class và DataLoader | Batch ảnh-mask chuẩn hóa |
| 7 | Định nghĩa UNet++ | Model sẵn sàng train |
| 8 | Training loop | Checkpoint, best model, training history |
| 9 | Visualize quá trình train | Loss curve và metric curve |
| 10 | Evaluation trên tập test | Bảng IoU, Dice, Precision, Recall |
| 11 | Inference | API duy nhất `predict(image_path)` |

---

## 3. Chi tiết từng cell

### Cell 1. Cài đặt thư viện và import

**Mục đích:** Cài đặt các package còn thiếu trên Kaggle, import toàn bộ thư viện cần cho đọc ảnh, tiền xử lý, huấn luyện, đánh giá, trực quan hóa và quản lý đường dẫn.

**Nội dung chính:**

```python
!pip install -q segmentation-models-pytorch albumentations opencv-python-headless

import os
import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
```

| Thành phần | Mô tả |
|---|---|
| Input | Kaggle notebook, dataset đã được gắn vào phiên chạy |
| Output | Các module Python sẵn sàng trong bộ nhớ |
| File được lưu | Không có |

---

### Cell 2. CONFIG tập trung

**Mục đích:** Khai báo toàn bộ tham số dự án tại một vị trí duy nhất. Các cell sau chỉ được phép đọc giá trị từ `CONFIG`, không hardcode lại learning rate, batch size, epoch, threshold, image size hoặc đường dẫn.

**Nội dung chính:**

```python
CONFIG = {
    "seed": 42,
    "model_name": "unetpp_efficientnetb4",
    "encoder_name": "efficientnet-b4",
    "encoder_weights": "imagenet",
    "image_size": (256, 256),
    "in_channels": 3,
    "classes": 1,
    "threshold": 0.5,
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "train_image_dir": "/kaggle/input/<dataset>/images/train",
    "train_mask_dir": "/kaggle/input/<dataset>/masks/train",
    "val_image_dir": "/kaggle/input/<dataset>/images/val",
    "val_mask_dir": "/kaggle/input/<dataset>/masks/val",
    "test_image_dir": "/kaggle/input/<dataset>/images/test",
    "test_mask_dir": "/kaggle/input/<dataset>/masks/test",
    "checkpoint_dir": "/kaggle/working/checkpoints",
    "output_dir": "/kaggle/working/outputs",
}
```

**Quy ước bắt buộc:**

- `CONFIG["image_size"]` luôn là `(256, 256)` trong mọi pha xử lý.
- `CONFIG["threshold"] = 0.5` được định nghĩa duy nhất tại cell này và dùng chung cho metric, evaluation, visualization, và inference.
- Tạo các thư mục lưu output bằng `Path(...).mkdir(parents=True, exist_ok=True)`.

| Thành phần | Mô tả |
|---|---|
| Input | Đường dẫn Kaggle dataset và các hyperparameter thí nghiệm |
| Output | Dict `CONFIG`, các đường dẫn output hợp lệ |
| File được lưu | Chưa có; các thư mục `/kaggle/working/checkpoints/` và `/kaggle/working/outputs/` được tạo |

---

### Cell 3. Fix seed

**Mục đích:** Cố định nguồn ngẫu nhiên của Python, NumPy và PyTorch để kết quả có thể tái lập trong điều kiện môi trường tương đương.

**Nội dung chính:**

```python
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG["seed"])
```

| Thành phần | Mô tả |
|---|---|
| Input | `CONFIG["seed"]` |
| Output | Trạng thái random được cố định |
| File được lưu | Không có |

---

### Cell 4. Load và EDA dataset

**Mục đích:** Kiểm tra cấu trúc thư mục, số lượng ảnh-mask, kích thước gốc, giá trị mask và mức độ mất cân bằng lớp trước khi huấn luyện.

**Nội dung chính:**

1. Đọc danh sách ảnh PNG và mask PNG theo thứ tự cố định bằng `sorted()`.
2. Kiểm tra ảnh và mask cùng tên file.
3. Thống kê số lượng mẫu cho train, validation và test.
4. Đo kích thước gốc kỳ vọng `416 x 416`, số kênh ảnh là `3`.
5. Tính tỷ lệ pixel oil spill trong mask.
6. Hiển thị các cặp ảnh SAR - mask - overlay mẫu.

| Thành phần | Mô tả |
|---|---|
| Input | `CONFIG["*_image_dir"]`, `CONFIG["*_mask_dir"]` và file PNG gốc |
| Output | Danh sách path ảnh/mask, thống kê dataset, figure EDA |
| File được lưu | Tùy chọn: `outputs/eda_samples.png`, `outputs/class_distribution.png` |

**Kiểm tra cần đạt:**

```text
SAR image shape: (416, 416, 3)
Mask shape:      (416, 416)
Mask values:     {0, 1} hoặc {0, 255}
```

---

### Cell 5. Pipeline tiền xử lý chung

**Mục đích:** Định nghĩa một pipeline Albumentations duy nhất, bảo đảm train, validation, test và `predict(image_path)` xử lý ảnh theo cùng chuẩn resize-normalize-tensor.

**Nội dung chính:**

```python
def build_transforms(is_train: bool = False) -> A.Compose:
    transforms = [
        A.Resize(
            height=CONFIG["image_size"][0],
            width=CONFIG["image_size"][1],
        )
    ]

    if is_train:
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

    transforms.extend([
        A.Normalize(),
        ToTensorV2(),
    ])
    return A.Compose(transforms)

TRAIN_TRANSFORM = build_transforms(is_train=True)
INFERENCE_TRANSFORM = build_transforms(is_train=False)
```

**Quy ước bắt buộc:**

- Augmentation chỉ được áp dụng khi `is_train=True`.
- `A.Resize`, `A.Normalize` và `ToTensorV2` giống nhau ở train, validation, test và inference.
- Mask được resize bằng nội suy gần nhất do Albumentations tự xử lý khi truyền vào tham số `mask`.
- Không viết một preprocessing riêng cho hàm inference.

| Thành phần | Mô tả |
|---|---|
| Input | Ảnh SAR RGB, mask binary tùy chọn |
| Output | Tensor ảnh `(3, 256, 256)` và tensor mask `(256, 256)` |
| File được lưu | Không có |

---

### Cell 6. Dataset class và DataLoader

**Mục đích:** Xây dựng lớp dataset để đọc từng cặp ảnh-mask, áp dụng pipeline dùng chung và tạo DataLoader cho train/validation/test.

**Nội dung chính:**

```python
class OilSpillDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(str(self.image_paths[index])), cv2.COLOR_BGR2RGB)
        mask = None
        if self.mask_paths is not None:
            mask = cv2.imread(str(self.mask_paths[index]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)

        transformed = self.transform(image=image, mask=mask) if mask is not None else self.transform(image=image)
        if mask is None:
            return transformed["image"]
        return transformed["image"], transformed["mask"].unsqueeze(0).float()
```

| Thành phần | Mô tả |
|---|---|
| Input | Danh sách path ảnh/mask và `TRAIN_TRANSFORM` hoặc `INFERENCE_TRANSFORM` |
| Output | DataLoader trả batch ảnh `(B, 3, 256, 256)` và mask `(B, 1, 256, 256)` |
| File được lưu | Không có |

**Kiểm tra batch:**

```python
images, masks = next(iter(train_loader))
assert images.shape[1:] == (3, 256, 256)
assert masks.shape[1:] == (1, 256, 256)
```

---

### Cell 7. Định nghĩa model UNet++

**Mục đích:** Khởi tạo UNet++ với encoder EfficientNet-B4 pretrained ImageNet, một kênh đầu ra cho bài toán binary segmentation, loss function và optimizer.

**Nội dung chính:**

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.UnetPlusPlus(
    encoder_name=CONFIG["encoder_name"],
    encoder_weights=CONFIG["encoder_weights"],
    in_channels=CONFIG["in_channels"],
    classes=CONFIG["classes"],
    activation=None,
).to(DEVICE)

criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)
```

| Thành phần | Mô tả |
|---|---|
| Input | `CONFIG`, batch tensor `(B, 3, 256, 256)` |
| Output | Logits `(B, 1, 256, 256)` |
| File được lưu | Không có |

---

### Cell 8. Training loop

**Mục đích:** Huấn luyện model theo epoch, tính metric validation, lưu checkpoint, lưu best model và ghi lại toàn bộ lịch sử train.

**Các bước trong mỗi epoch:**

1. Đặt `model.train()`, duyệt `train_loader`, forward logits và cập nhật trọng số.
2. Đặt `model.eval()`, duyệt validation/test loader với `torch.no_grad()`.
3. Dùng `torch.sigmoid(logits)` và đúng `CONFIG["threshold"]` để sinh mask dự đoán binary.
4. Tính loss, IoU, Dice, Precision, Recall.
5. Ghi một dòng metric vào history.
6. Lưu checkpoint theo validation IoU của epoch.
7. Cập nhật best model nếu IoU validation cao hơn giá trị tốt nhất trước đó.

**Format checkpoint bắt buộc:**

```python
checkpoint_name = f"unetpp_epoch{epoch}_iou{val_iou:.4f}.pth"
checkpoint_path = Path(CONFIG["checkpoint_dir"]) / checkpoint_name
torch.save(checkpoint, checkpoint_path)
```

**Best model riêng:**

```python
best_model_path = Path(CONFIG["checkpoint_dir"]) / "unetpp_best_model.pth"
torch.save(checkpoint, best_model_path)
```

**Nội dung checkpoint tối thiểu:**

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_iou": val_iou,
    "config": CONFIG,
}
```

| Thành phần | Mô tả |
|---|---|
| Input | `train_loader`, `val_loader`, model, optimizer, loss function, `CONFIG` |
| Output | Model đã cập nhật, dictionary `history` theo epoch |
| File được lưu | `/kaggle/working/checkpoints/unetpp_epoch{n}_iou{score}.pth`, `/kaggle/working/checkpoints/unetpp_best_model.pth`, `outputs/training_history.csv` |

**Cấu trúc training history:**

| epoch | train_loss | val_loss | iou | dice | precision | recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | ... | ... | ... | ... | ... | ... |

---

### Cell 9. Visualize quá trình train

**Mục đích:** Trực quan hóa lịch sử loss và metric để theo dõi mức độ hội tụ, overfitting và sự đánh đổi Precision-Recall.

**Nội dung chính:**

1. Đọc `history` trong bộ nhớ hoặc `outputs/training_history.csv`.
2. Vẽ `train_loss` và `val_loss` theo epoch.
3. Vẽ IoU, Dice, Precision, Recall theo epoch.
4. Lưu figure với độ phân giải phù hợp cho báo cáo.

| Thành phần | Mô tả |
|---|---|
| Input | `history` hoặc `outputs/training_history.csv` |
| Output | Hai biểu đồ quá trình training |
| File được lưu | `outputs/loss_curve.png`, `outputs/metric_curve.png` |

---

### Cell 10. Evaluation trên tập test

**Mục đích:** Đánh giá khách quan checkpoint tốt nhất trên tập test có ground truth, báo cáo bốn metric chính và trực quan hóa dự đoán mẫu.

**Nội dung chính:**

```python
checkpoint = torch.load(
    Path(CONFIG["checkpoint_dir"]) / "unetpp_best_model.pth",
    map_location=DEVICE,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

1. Tạo test dataset bằng `INFERENCE_TRANSFORM`.
2. Sinh probability bằng `sigmoid` và phân ngưỡng bằng `CONFIG["threshold"]`.
3. Tích lũy `TP`, `FP`, `FN` trên toàn bộ tập test.
4. Tính IoU, Dice, Precision, Recall.
5. Hiển thị ảnh SAR, ground truth, prediction và overlay.

| Thành phần | Mô tả |
|---|---|
| Input | `unetpp_best_model.pth`, test DataLoader, `CONFIG["threshold"]` |
| Output | Bảng metric tổng hợp và figure prediction mẫu |
| File được lưu | `outputs/test_metrics.json` và tùy chọn `outputs/test_predictions.png` |

**Bảng kết quả báo cáo:**

| Model | Encoder | IoU | Dice | Precision | Recall |
|---|---|---:|---:|---:|---:|
| UNet++ | EfficientNet-B4 | ... | ... | ... | ... |

---

### Cell 11. Inference với API thống nhất

**Mục đích:** Cung cấp một API inference duy nhất cho mọi model, không cho phép tham số phụ gây khác biệt preprocessing, image size hoặc threshold khi gọi.

**API bắt buộc:**

```python
predict(image_path)
```

**Nội dung tham khảo:**

```python
def predict(image_path):
    """Trả về mask uint8 (256, 256), với 0=background và 1=oil spill."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFERENCE_TRANSFORM(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))

    return (probability.squeeze().cpu().numpy() >= CONFIG["threshold"]).astype(np.uint8)
```

| Thành phần | Mô tả |
|---|---|
| Input | Một `image_path` duy nhất trỏ đến ảnh PNG/JPG SAR |
| Output | Binary mask `uint8` kích thước `(256, 256)` |
| File được lưu | Tùy chọn: `outputs/predictions/{image_stem}_mask.png` và overlay tương ứng |

**Quy ước bắt buộc:**

- Không thêm bất kỳ tham số nào ngoài `image_path` vào `predict`.
- Hàm phải dùng lại `INFERENCE_TRANSFORM` của Cell 5.
- Hàm phải dùng `CONFIG["threshold"]`, không viết giá trị ngưỡng trực tiếp trong hàm.
- Checkpoint tốt nhất phải được load trước khi gọi `predict(image_path)`.

---

## 4. Danh sách artifact cần nộp

| Artifact | Đường dẫn đề xuất | Ý nghĩa |
|---|---|---|
| Notebook hoàn chỉnh | `notebooks/Khoa/unetpp_efficientnetb4.ipynb` | Mã nguồn tái lập thí nghiệm |
| Best model | `/kaggle/working/checkpoints/unetpp_best_model.pth` | Inference/evaluate không cần train lại |
| Checkpoint theo epoch | `/kaggle/working/checkpoints/unetpp_epoch{n}_iou{score}.pth` | Lưu trạng thái model theo từng epoch |
| Training history | `/kaggle/working/outputs/training_history.csv` | Giá trị loss và metric từng epoch |
| Loss curve | `/kaggle/working/outputs/loss_curve.png` | Theo dõi hội tụ |
| Metric curve | `/kaggle/working/outputs/metric_curve.png` | Theo dõi IoU, Dice, Precision, Recall |
| Kết quả test | `/kaggle/working/outputs/test_metrics.json` | Kết quả định lượng cuối cùng |
| Prediction samples | `/kaggle/working/outputs/predictions/` | Đánh giá trực quan segmentation |

## 5. Checklist trước khi chạy

- [ ] GPU P100 đã được bật trên Kaggle.
- [ ] Đường dẫn dataset trong `CONFIG` chính xác.
- [ ] `CONFIG["image_size"]` là `(256, 256)`.
- [ ] `CONFIG["threshold"]` là `0.5` và chỉ được khai báo một lần.
- [ ] `seed_everything(CONFIG["seed"])` được gọi trước khi tạo dataset/model.
- [ ] Train, validation, test và inference dùng chung pipeline resize-normalize-tensor.
- [ ] Checkpoint được đặt tên theo `unetpp_epoch{n}_iou{score}.pth`.
- [ ] Best model, history CSV, loss curve và metric curve được lưu.
- [ ] Inference chỉ gọi qua `predict(image_path)`.
