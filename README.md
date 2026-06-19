# Oil Spill SAR Segmentation

Project cuoi ki mon Hoc Sau: segmentation vung loang dau tren bien tu anh SAR, so sanh 3 model rieng le va tich hop vao app demo.

## Muc tieu

- Train 3 model segmentation doc lap cho bai toan phat hien vung loang dau tren anh SAR.
- Danh gia tung model bang cac metric thong nhat: Dice, IoU, Precision, Recall.
- Tong hop ket qua bang ensemble.
- Xay dung app cho phep upload anh SAR, chon model, hien thi mask va overlay ket qua.

## Thanh vien va nhanh lam viec

| Thanh vien | Branch | Pham vi chinh |
|---|---|---|
| Quang | `Quang` | Model 1, training script, config va report ket qua |
| Hung | `Hung` | Model 2, training script, config va report ket qua |
| Khoa | `Khoa` | Model 3, training script, config va report ket qua |

Nhanh tich hop:

- `main`: ban on dinh de nop/demo.
- `develop`: nhanh tong hop code tu cac thanh vien.
- `app`: phat trien app demo va inference tich hop.

Khong merge truc tiep code dang thu nghiem vao `main`.

## Cau truc repo

```text
.
├── app/                  # Streamlit app
├── configs/              # Config train/inference cho tung model
├── data/                 # Huong dan dataset, khong commit data lon
├── docs/                 # Note, ke hoach, quy uoc lam viec
├── notebooks/            # Notebook rieng cua tung thanh vien
├── reports/              # Bang ket qua, hinh anh, noi dung bao cao
├── src/
│   ├── datasets/         # Dataset va preprocessing
│   ├── inference/        # Predict, ensemble, postprocess
│   ├── models/           # Kien truc model
│   ├── training/         # Script train
│   └── utils/            # Metric, visualization, helper
├── weights/              # Khong commit weight lon len GitHub
├── requirements.txt
└── README.md
```

## Chuan dau vao/dau ra bat buoc

Tat ca model can thong nhat interface de app co the tich hop:

- Input: anh SAR da preprocess ve cung kich thuoc, vi du `256x256` hoac `512x512`.
- Output: probability map hoac binary mask cung kich thuoc voi input.
- Binary mask:
  - `0`: background / bien khong co dau.
  - `1`: vung nghi loang dau.
- Model inference nen co ham:

```python
def load_model(weight_path: str, device: str):
    ...

def predict(image, model, device: str, threshold: float = 0.5):
    ...
```

## Ensemble

Co the dung 1 trong 2 cach:

- Majority voting: pixel nao co it nhat 2/3 model du doan la dau thi lay la dau.
- Average probability: lay trung binh probability cua 3 model roi threshold.

Voi demo cuoi ki, uu tien average probability neu ca 3 model tra ve probability map.

## App demo

App du kien dung Streamlit:

- Upload anh SAR.
- Chon model: Quang / Hung / Khoa / Ensemble.
- Dieu chinh threshold.
- Hien thi anh goc, mask du doan, overlay.
- Xuat/tai mask ket qua.

Chay app:

```bash
streamlit run app/streamlit_app.py
```

## Quy trinh lam viec

1. Moi thanh vien checkout branch rieng.
2. Chi sua file trong pham vi duoc phan cong neu khong co trao doi truoc.
3. Commit thuong xuyen voi message ro rang.
4. Khi code chay duoc, tao Pull Request vao `develop`.
5. Test lai inference truoc khi merge vao `develop`.
6. Chi merge `develop` vao `main` khi da san sang demo/nop.

## Du lieu va weight

Khong commit cac file lon:

- Dataset goc.
- Dataset processed.
- File weight `.pt`, `.pth`, `.ckpt`.

Luu dataset/weight tren Google Drive, Kaggle, Roboflow hoac Hugging Face, sau do them link vao `docs/DATASET.md`.

