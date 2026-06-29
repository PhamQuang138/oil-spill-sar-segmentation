# Phân đoạn vùng loang dầu trên ảnh SAR

Dự án xây dựng pipeline phát hiện và phân đoạn vùng nghi loang dầu trên ảnh vệ tinh SAR bằng deep learning. Hệ thống hỗ trợ chạy từng mô hình riêng lẻ hoặc ensemble, cho phép upload ảnh SAR, điều chỉnh threshold, hậu xử lý mask và hiển thị kết quả trực quan bằng Streamlit.

## Mục tiêu

- Phân đoạn vùng nghi loang dầu ở mức pixel trên ảnh SAR.
- So sánh ba mô hình học sâu: DeepLabV3+, SegFormer và UNet++.
- Chuẩn hóa đầu ra của các mô hình thành probability map để dễ tích hợp.
- Tối ưu kết quả suy luận bằng threshold search, ensemble và hậu xử lý mask.
- Cung cấp app demo để xem ảnh gốc, mask, probability map và overlay.

## Cấu trúc chính

```text
.
├── app/
│   └── streamlit_app.py              # Giao diện demo Streamlit
├── configs/
│   ├── quang.yaml                    # Cấu hình tham khảo cho DeepLabV3+
│   ├── hung.yaml                     # Cấu hình tham khảo cho SegFormer
│   └── khoa.yaml                     # Cấu hình tham khảo cho UNet++
├── notebooks/
│   ├── Quang/                        # Script train/evaluate/fine-tune DeepLabV3+
│   ├── Hung/                         # Script train/evaluate/predict SegFormer
│   └── Khoa/                         # Checkpoint và tài nguyên UNet++
├── src/
│   ├── evaluation/
│   │   └── development_eval.py       # Đánh giá threshold, ensemble, hậu xử lý
│   └── inference/
│       ├── base.py                   # Interface chung cho predictor
│       ├── quang_adapter.py          # Adapter DeepLabV3+ ResNet34
│       ├── hung_adapter.py           # Adapter SegFormer MiT-B0
│       ├── khoa_adapter.py           # Adapter UNet++ EfficientNet-B4
│       ├── ensemble.py               # Ensemble bằng trung bình probability map
│       └── postprocess.py            # Hậu xử lý mask nhị phân
├── weights/                          # Checkpoint mô hình, không commit lên Git
├── requirements.txt                  # Thư viện cần cài
└── README.md
```

## Mô hình

| Thành viên | Mô hình | File weight app sử dụng | Threshold mặc định |
|---|---|---|---|
| Quang | DeepLabV3+ ResNet34 | `weights/quang_best_deeplabv3plus_checkpoint.pth` | `0.50` |
| Hưng | SegFormer MiT-B0 | `weights/Hung_best_segformer_checkpoint.pth` | `0.60` |
| Khoa | UNet++ EfficientNet-B4 | `weights/Khoa_best_Unetplusplus_checkpoint.pth` | `0.65` |
| Chung | Ensemble trung bình xác suất | Tạo từ các model load thành công | `0.50` |

App sẽ tự tạo lựa chọn Ensemble nếu load được ít nhất hai mô hình.

## Cài đặt

Khuyến nghị dùng môi trường ảo Python.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu dùng CUDA, hãy cài bản PyTorch phù hợp với GPU trước hoặc kiểm tra lại phiên bản `torch` sau khi cài.

## Chuẩn bị checkpoint

Đặt các file checkpoint vào thư mục `weights/` với đúng tên:

```text
weights/
├── quang_best_deeplabv3plus_checkpoint.pth
├── Hung_best_segformer_checkpoint.pth
└── Khoa_best_Unetplusplus_checkpoint.pth
```

Thư mục `weights/` không nên commit lên Git vì checkpoint thường có dung lượng lớn.

## Chạy app demo

```bash
streamlit run app/streamlit_app.py
```

Luồng sử dụng app:

1. Chọn thiết bị chạy: Auto, CPU hoặc CUDA.
2. Chọn mô hình: DeepLabV3+, SegFormer, UNet++ hoặc Ensemble.
3. Điều chỉnh threshold nếu cần.
4. Bật/tắt hậu xử lý mask.
5. Upload ảnh SAR định dạng `png`, `jpg`, `jpeg`, `tif` hoặc `tiff`.
6. Bấm `Chạy nhận diện`.
7. Xem ảnh gốc, mask dự đoán, overlay, probability map và raw mask.

## Đầu vào và đầu ra

Đầu vào app là ảnh SAR được đọc bằng PIL và chuyển về RGB.

Mỗi predictor cần trả về:

- `prob_map`: ma trận xác suất lớp oil spill, giá trị trong khoảng `[0, 1]`.
- `mask`: mask nhị phân sau threshold, trong đó `1` là vùng nghi loang dầu và `0` là nền.

Overlay được tạo bằng cách tô đỏ các pixel có `mask = 1` lên ảnh gốc.

## Hậu xử lý và ensemble

Pipeline suy luận có các bước tối ưu sau:

- **Threshold search**: chọn ngưỡng tốt nhất trên validation thay vì cố định mọi mô hình ở `0.50`.
- **Ensemble**: lấy trung bình probability map của các mô hình đã load thành công.
- **Post-process**: dùng morphology opening/closing và loại connected components nhỏ để giảm nhiễu mask.

Các tham số hậu xử lý trong app:

- `min_area`: diện tích nhỏ nhất của connected component được giữ lại.
- `open_kernel`: kernel cho morphology opening.
- `close_kernel`: kernel cho morphology closing.
- `alpha`: độ đậm của overlay.

## Đánh giá trên validation

Script đánh giá nằm ở:

```bash
python src/evaluation/development_eval.py
```

Script này dùng checkpoint hiện có để:

- Quét threshold trên validation set.
- Tính IoU, Dice, Precision, Recall.
- Đánh giá ensemble trung bình probability map.
- Đánh giá hậu xử lý mask.
- Xuất bảng kết quả và hình trực quan.

Mặc định script kỳ vọng dữ liệu validation có cấu trúc:

```text
dataset/
├── images/images/val/
└── masks/masks/val/
```

Có thể chạy thử nhanh bằng giới hạn số ảnh:

```bash
python src/evaluation/development_eval.py --limit 32
```

## Train và fine-tune

Các script huấn luyện chính nằm trong thư mục notebook của từng thành viên:

- DeepLabV3+: `notebooks/Quang/train_deeplabv3plus.py`
- Fine-tune DeepLabV3+: `notebooks/Quang/finetune_deeplabv3plus_tversky.py`
- SegFormer: `notebooks/Hung/train.py`

Nhánh fine-tune DeepLabV3+ hỗ trợ các biến thể loss như BCE+Dice, Tversky, Focal Tversky và BCE+Tversky. Mục tiêu là cải thiện cân bằng giữa false positive và false negative cho lớp oil spill vốn chiếm tỉ lệ nhỏ.

## Quy ước dữ liệu và Git

Không commit các dữ liệu hoặc artifact lớn:

- Dataset gốc hoặc dataset đã xử lý.
- Checkpoint `.pth`, `.pt`, `.ckpt`.
- Log, output tạm, cache Python.

Các thư mục như `data/`, `dataset/`, `weights/` được giữ local để chạy thử và demo.

## Ghi chú phát triển

- `src/inference/base.py` định nghĩa interface chung để app gọi mọi mô hình giống nhau.
- Adapter của từng mô hình chịu trách nhiệm load checkpoint, preprocess ảnh và sinh probability map.
- Ensemble chỉ phụ thuộc vào `predict_proba`, nên có thể thêm mô hình mới nếu mô hình đó tuân theo interface chung.
