# Dataset

Khong commit dataset truc tiep vao repo.

## Link dataset

Them link dataset vao day:

```text
Dataset source:
Google Drive/Kaggle/Hugging Face:
```

## Cau truc dataset de xuat

```text
data/
├── raw/
│   ├── images/
│   └── masks/
└── processed/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

## Chuan tien xu ly

Can thong nhat:

- Image size: `256x256` hoac `512x512`.
- Cach normalize anh SAR.
- Cach doc mask ground truth.
- Ti le chia train/val/test.
- Augmentation co dung hay khong.

