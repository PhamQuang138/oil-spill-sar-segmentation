# Phat Hien Vet Dau Tran Tren Anh SAR Bang UNet++

## 1. Mo ta bai toan

Du an giai bai toan **binary semantic segmentation** nham phat hien va phan vung vet dau tran tren bien tu anh radar khau do tong hop (SAR - Synthetic Aperture Radar).

Moi pixel cua anh duoc du doan thuoc mot trong hai lop:

| Lop | Gia tri mask | Y nghia |
|---|---:|---|
| Background | `0` | Mat bien, dat lien hoac khu vuc khong co dau |
| Oil spill | `1` | Vung nghi ngo/co vet dau tran |

Mo hinh su dung la **UNet++** voi encoder **EfficientNet-B4**, duoc cai dat bang thu vien `segmentation_models_pytorch` tren nen tang PyTorch. Qua trinh huan luyen duoc thiet ke de chay tren Kaggle GPU P100.

Muc tieu cua du an la xay dung pipeline co the tai lap, trong do preprocessing, threshold, cau hinh va API inference duoc thong nhat de tranh sai khac giua training, evaluation va suy luan.

---

## 2. Cau truc thu muc project

```text
notebooks/Khoa/
├── README.md                         # Tai lieu du an
├── unetpp_efficientnetb4.ipynb       # Notebook chinh chay tren Kaggle
├── checkpoints/
│   └── unetpp_efficientnetb4_epoch{n}_iou{score}.pth
├── outputs/
│   ├── loss_curve.png                # Duong cong train/validation loss
│   ├── metric_curve.png              # Duong cong IoU, Dice, Precision, Recall
│   ├── training_history.csv          # Lich su train tung epoch
│   └── predictions/                  # Mask va anh overlay suy luan
└── requirements.txt                  # Danh sach thu vien

data/
└── oil-spill-sar/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/                     # Tuy chon, co the khong co mask
    └── masks/
        ├── train/
        └── val/
```

Ten checkpoint phai tuan theo dung format:

```text
unetpp_efficientnetb4_epoch{n}_iou{score}.pth
```

Vi du:

```text
unetpp_efficientnetb4_epoch24_iou0.8173.pth
```

---

## 3. Mo ta dataset

Dataset gom anh SAR dinh dang PNG, anh goc co kich thuoc `416 x 416` pixel va `3` kenh mau. Mask tuong ung la anh phan vung nhi phan, trong do pixel vet dau co gia tri `1` (hoac `255` truoc khi nhi phan hoa), nen co gia tri `0`.

| Thanh phan | Dinh dang | Kich thuoc goc | Ghi chu |
|---|---|---:|---|
| Anh SAR | PNG, 3 kenh | `416 x 416` | Dau vao cua mo hinh |
| Mask | PNG, 1 kenh | `416 x 416` | Nhan binary cho tung pixel |
| Anh train | `images/train/` | Theo dataset | Co mask tuong ung |
| Anh validation | `images/val/` | Theo dataset | Dung de chon checkpoint tot nhat |
| Anh test/inference | `images/test/` | Theo dataset | Co the khong co mask |

Mac du anh goc la `416 x 416`, **toan bo training, validation va inference phai dung duy nhat kich thuoc `256 x 256`**. Anh duoc resize bang `cv2.INTER_LINEAR`; mask duoc resize bang `cv2.INTER_NEAREST` de khong lam thay doi nhan.

Can dam bao ten file anh va mask khop nhau, vi du:

```text
images/train/sample_001.png
masks/train/sample_001.png
```

---

## 4. Pipeline xu ly

### 4.1 CONFIG duy nhat

Tat ca tham so phai nam trong **mot dict `CONFIG` duy nhat o dau notebook**. Khong hardcode learning rate, batch size, epoch, image size, threshold hoac duong dan o cac cell/ham khac.

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
    "train_image_dir": "/kaggle/input/.../images/train",
    "train_mask_dir": "/kaggle/input/.../masks/train",
    "val_image_dir": "/kaggle/input/.../images/val",
    "val_mask_dir": "/kaggle/input/.../masks/val",
    "checkpoint_dir": "/kaggle/working/checkpoints",
    "output_dir": "/kaggle/working/outputs",
}
```

Gia tri `CONFIG["threshold"]` la noi **duy nhat** dinh nghia nguong chuyen xac suat sang mask binary. Tat ca metric, evaluation, visualization va `predict(image_path)` phai dung dung gia tri nay.

### 4.2 Preprocessing va augmentation

Xay dung **mot pipeline Albumentations dung chung** cho dataset va inference. Pipeline co mot ham khoi tao duy nhat, trong do augmentation chi bat khi `is_train=True`; cac buoc resize, normalize va chuyen tensor la giong nhau cho train, validation va inference.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(is_train: bool = False):
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

`OilSpillDataset` dung `TRAIN_TRANSFORM` cho train va `INFERENCE_TRANSFORM` cho validation. Ham `predict(image_path)` bat buoc dung `INFERENCE_TRANSFORM`; khong tao mot preprocessing rieng cho inference.

### 4.3 Train

1. Co dinh seed cho `random`, `numpy`, `torch`, CUDA va cuDNN.
2. Doc cap anh-mask, resize ve `256 x 256`, nhi phan hoa mask.
3. Khoi tao `smp.UnetPlusPlus` voi encoder `efficientnet-b4` va weights ImageNet.
4. Huan luyen bang PyTorch tren GPU P100.
5. Sau moi epoch, tinh loss va IoU, Dice, Precision, Recall tren validation set.
6. Luu checkpoint tot nhat theo validation IoU va cap nhat `training_history.csv`.
7. Ket thuc training, ve va luu `loss_curve.png` va `metric_curve.png`.

Ham fix seed tham khao:

```python
def seed_everything(seed: int) -> None:
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG["seed"])
```

### 4.4 Evaluate

1. Load checkpoint co IoU validation tot nhat.
2. Dung `INFERENCE_TRANSFORM` de xu ly validation images o dung `256 x 256`.
3. Sinh xac suat bang `sigmoid(logits)` va nhi phan hoa theo `CONFIG["threshold"]`.
4. Tinh IoU, Dice, Precision va Recall tren toan bo validation set.
5. Luu anh so sanh SAR - ground truth - prediction/overlay de kiem tra dinh tinh.

### 4.5 Inference

API inference phai co **mot chu ky duy nhat** cho moi model:

```python
predict(image_path)
```

Ham nay tu dong load checkpoint da duoc chi dinh trong `CONFIG`, doc anh, ap dung `INFERENCE_TRANSFORM`, du doan va tra ve mask `uint8` kich thuoc `256 x 256` co gia tri `{0, 1}`. Khong them tham so `threshold`, `image_size`, `checkpoint_path` hay tham so tuy chinh khac vao API nay.

```python
def predict(image_path):
    """Tra ve binary mask uint8 (256, 256) voi 0=background, 1=oil spill."""
    image = read_image(image_path)
    tensor = INFERENCE_TRANSFORM(image=image)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probability = torch.sigmoid(MODEL(tensor))

    return (probability.squeeze().cpu().numpy() >= CONFIG["threshold"]).astype("uint8")
```

---

## 5. Cac metric su dung

Metric duoc tinh tren mask binary sau khi phan nguong bang `CONFIG["threshold"]`.

| Metric | Cong thuc | Y nghia |
|---|---|---|
| IoU (Jaccard) | `TP / (TP + FP + FN)` | Do chong lap giua vung du doan va vung dau that |
| Dice (F1) | `2TP / (2TP + FP + FN)` | Do tuong dong giua prediction va ground truth |
| Precision | `TP / (TP + FP)` | Ty le pixel du doan la dau va dung |
| Recall | `TP / (TP + FN)` | Ty le pixel dau that duoc phat hien |

Trong do `TP`, `FP`, `FN` lan luot la true positive, false positive va false negative. Nho `epsilon` nho trong phep chia de tranh chia cho 0.

---

## 6. Huong dan chay tung buoc

### Buoc 1: Tao Kaggle Notebook va gan dataset

1. Tao notebook moi tren Kaggle va bat GPU P100 trong `Notebook options`.
2. Gan dataset SAR vao notebook.
3. Cap nhat cac duong dan dataset trong `CONFIG`.
4. Tao cac thu muc `checkpoint_dir` va `output_dir` trong `/kaggle/working/`.

### Buoc 2: Cai dat dependencies

```bash
pip install -q segmentation-models-pytorch albumentations opencv-python-headless
```

### Buoc 3: Khai bao CONFIG va fix seed

1. Chay cell chua dict `CONFIG`.
2. Chay `seed_everything(CONFIG["seed"])` truoc khi tao dataset, dataloader va model.
3. Kiem tra `CONFIG["image_size"] == (256, 256)` va `CONFIG["threshold"]` chi duoc khai bao mot lan.

### Buoc 4: Tao dataset va dataloader

1. Kiem tra anh-mask co ten file tuong ung.
2. Train dataset dung `build_transforms(is_train=True)`.
3. Validation dataset dung `build_transforms(is_train=False)`.
4. Kiem tra shape batch la `(B, 3, 256, 256)` va mask la `(B, 1, 256, 256)`.

### Buoc 5: Train va luu artifacts

1. Khoi tao UNet++:

```python
model = smp.UnetPlusPlus(
    encoder_name=CONFIG["encoder_name"],
    encoder_weights=CONFIG["encoder_weights"],
    in_channels=CONFIG["in_channels"],
    classes=CONFIG["classes"],
).to(DEVICE)
```

2. Train theo so epoch trong `CONFIG["epochs"]`.
3. Khi validation IoU cai thien, luu checkpoint theo ten:

```python
checkpoint_name = (
    f'{CONFIG["model_name"]}_epoch{epoch}_iou{val_iou:.4f}.pth'
)
```

4. Luu lich su theo epoch vao `training_history.csv`.
5. Luu `loss_curve.png` va `metric_curve.png` sau khi training.

Checkpoint can chua it nhat:

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_iou": val_iou,
    "config": CONFIG,
}
```

### Buoc 6: Evaluate

1. Load checkpoint tot nhat ma khong train lai.
2. Chay evaluation tren validation set.
3. Bao cao IoU, Dice, Precision, Recall.
4. Luu visualizations vao `outputs/predictions/`.

### Buoc 7: Inference

1. Load checkpoint tot nhat mot lan.
2. Goi API duy nhat:

```python
mask = predict("/kaggle/input/.../sample.png")
```

3. Luu mask hoac ve overlay de quan sat vung vet dau.

---

## 7. Ket qua mong doi

Sau khi chay hoan chinh, project can co cac dau ra sau:

| Dau ra | Vi tri | Muc dich |
|---|---|---|
| Best checkpoint | `checkpoints/*.pth` | Load lai model de evaluate/inference, khong can train lai |
| Training history | `outputs/training_history.csv` | Theo doi loss va metric tung epoch |
| Loss curve | `outputs/loss_curve.png` | Kiem tra qua trinh hoi tu va overfitting |
| Metric curve | `outputs/metric_curve.png` | Theo doi IoU, Dice, Precision, Recall |
| Prediction masks/overlays | `outputs/predictions/` | Danh gia dinh tinh ket qua segmentation |

Bao cao ket qua tren validation set bang bang sau:

| Model | Encoder | Image size | Threshold | IoU | Dice | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| UNet++ | EfficientNet-B4 | `256 x 256` | `CONFIG["threshold"]` | ... | ... | ... | ... |

Can doi chieu dong thoi metric dinh luong va cac anh overlay. IoU/Dice cao nhung prediction phat hien qua nhieu vung toi khong phai dau can duoc kiem tra bang Precision va visualizations.

---

## 8. Dependencies / requirements

Tao file `requirements.txt` voi cac phu thuoc sau:

```text
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.1
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.66.0
Pillow>=10.0.0
```

Kiem tra moi truong truoc khi train:

```python
import torch
import segmentation_models_pytorch as smp

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("SMP:", smp.__version__)
```

Muc tieu la GPU P100 duoc nhan dien va toan bo pipeline chay voi input `(B, 3, 256, 256)` o ca training, evaluation va inference.
