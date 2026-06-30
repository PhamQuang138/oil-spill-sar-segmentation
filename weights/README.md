# Weights



Tải checkpoint đã train từ Google Drive:

```text
Google Drive: TODO: https://drive.google.com/drive/folders/1gGlfn9oDPecxgXpSQkkypruZ1C5bKBmS?usp=sharing
```

Sau khi tải, đặt các file trong thư mục này với đúng tên:

```text
weights/
├── quang_best_deeplabv3plus_checkpoint.pth
├── Hung_best_segformer_checkpoint.pth
└── Khoa_best_Unetplusplus_checkpoint.pth
```

App Streamlit sẽ tự load các file trên khi chạy:

```bash
streamlit run app/streamlit_app.py
```
