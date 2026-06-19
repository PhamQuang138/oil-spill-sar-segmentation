# Ghi chú cho nhóm

Đọc file này trước khi bắt đầu code.

## Nguyên tắc chung

- Không sửa code của thành viên khác nếu chưa trao đổi trước.
- Không commit dataset lớn, file weight lớn, file cache hoặc thư mục môi trường ảo.
- Mỗi model phải có config, cách train, cách inference và kết quả metric riêng.
- Mọi thay đổi ảnh hưởng đến preprocessing, input size hoặc output format phải báo cả nhóm.

## Branch

```text
main     : bản nộp/demo ổn định
develop  : bản tích hợp chung
app      : app demo và inference tổng hợp
Quang    : phần việc của Quang
Hung     : phần việc của Hưng
Khoa     : phần việc của Khoa
```

Lệnh làm việc cơ bản:

```bash
git checkout develop
git pull
git checkout -b Quang
```

Nếu branch đã tồn tại:

```bash
git checkout Quang
git pull
```

Commit và push:

```bash
git add .
git commit -m "train unet baseline"
git push origin Quang
```

## Phân công file

Quang:

- `notebooks/Quang/`
- `configs/quang.yaml`
- `src/models/quang_model.py`
- `src/training/train_quang.py`

Hưng:

- `notebooks/Hung/`
- `configs/hung.yaml`
- `src/models/hung_model.py`
- `src/training/train_hung.py`

Khoa:

- `notebooks/Khoa/`
- `configs/khoa.yaml`
- `src/models/khoa_model.py`
- `src/training/train_khoa.py`

App và inference chung:

- `app/`
- `src/inference/`
- `src/utils/visualization.py`

## Chuẩn metric

Mỗi model cần báo cáo tối thiểu:

| Model | Dice | IoU | Precision | Recall | Ghi chú |
|---|---:|---:|---:|---:|---|
| Quang | | | | | |
| Hưng | | | | | |
| Khoa | | | | | |
| Ensemble | | | | | |

## Checklist trước khi merge

- Code train chạy được.
- Code inference chạy được với một ảnh test.
- Config đã được commit.
- README hoặc note có ghi cách chạy.
- Không commit data/weight lớn.
- Kết quả metric được ghi vào `reports/results.md`.

## Quy ước commit message

Nên viết ngắn gọn theo dạng:

```text
add unet model
fix dataset loading
train deeplabv3 baseline
add ensemble inference
update streamlit overlay
```

