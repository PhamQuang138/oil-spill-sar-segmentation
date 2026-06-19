# Team Notes

Doc file nay truoc khi bat dau code.

## Nguyen tac chung

- Khong sua code cua thanh vien khac neu chua trao doi.
- Khong commit dataset lon, weight lon, file cache hoac thu muc moi truong ao.
- Moi model phai co config, cach train, cach inference va ket qua metric rieng.
- Moi thay doi anh huong den preprocessing, input size, output format phai bao ca nhom.

## Branch

```text
main     : ban nop/demo on dinh
develop  : ban tich hop chung
app      : app demo va inference tong hop
Quang    : phan viec cua Quang
Hung     : phan viec cua Hung
Khoa     : phan viec cua Khoa
```

Lenh lam viec co ban:

```bash
git checkout develop
git pull
git checkout -b Quang
```

Neu branch da ton tai:

```bash
git checkout Quang
git pull
```

Commit:

```bash
git add .
git commit -m "train unet baseline"
git push origin Quang
```

## Phan cong file

Quang:

- `notebooks/Quang/`
- `configs/quang.yaml`
- `src/models/quang_model.py`
- `src/training/train_quang.py`

Hung:

- `notebooks/Hung/`
- `configs/hung.yaml`
- `src/models/hung_model.py`
- `src/training/train_hung.py`

Khoa:

- `notebooks/Khoa/`
- `configs/khoa.yaml`
- `src/models/khoa_model.py`
- `src/training/train_khoa.py`

App / inference chung:

- `app/`
- `src/inference/`
- `src/utils/visualization.py`

## Chuan metric

Moi model can bao cao toi thieu:

| Model | Dice | IoU | Precision | Recall | Ghi chu |
|---|---:|---:|---:|---:|---|
| Quang | | | | | |
| Hung | | | | | |
| Khoa | | | | | |
| Ensemble | | | | | |

## Checklist truoc khi merge

- Code train chay duoc.
- Code inference chay duoc voi mot anh test.
- Config da duoc commit.
- README hoac note co ghi cach chay.
- Khong commit data/weight lon.
- Ket qua metric duoc ghi vao `reports/results.md`.

## Quy uoc commit message

Nen viet ngan gon theo dang:

```text
add unet model
fix dataset loading
train deeplabv3 baseline
add ensemble inference
update streamlit overlay
```

