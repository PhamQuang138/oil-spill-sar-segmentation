# Dataset

Không commit dataset trực tiếp vào repo.

## Link dataset

Thêm link dataset vào đây:

```text
Nguồn dataset:
Google Drive/Kaggle/Hugging Face:
```

## Cấu trúc dataset đề xuất

```text
data/
|-- raw/
|   |-- images/
|   `-- masks/
`-- processed/
    |-- train/
    |   |-- images/
    |   `-- masks/
    |-- val/
    |   |-- images/
    |   `-- masks/
    `-- test/
        |-- images/
        `-- masks/
```

## Chuẩn tiền xử lý

Cả nhóm cần thống nhất:

- Kích thước ảnh: `256x256` hoặc `512x512`.
- Cách normalize ảnh SAR.
- Cách đọc mask ground truth.
- Tỉ lệ chia train/val/test.
- Có dùng augmentation hay không.

