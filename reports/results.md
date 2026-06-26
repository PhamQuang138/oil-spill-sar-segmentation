# Results

Cap nhat ket qua sau moi lan train dang tin cay.

| Model | Branch | Input Size | Dice | IoU | Precision | Recall | Weight/Run | Note |
|---|---|---:|---:|---:|---:|---:|---|---|
| DeepLabV3+ ResNet34 | `Quang` | 256 | 0.8832 | 0.7909 | 0.8663 | 0.9009 | `weights/quang_best_deeplabv3plus_checkpoint.pth` | Threshold search best at 0.50 |
| SegFormer MiT-B0 | `Hung` | 256 | 0.8992 | 0.8168 | 0.8940 | 0.9044 | `weights/Hung_best_segformer_checkpoint.pth` | Threshold search best at 0.60 |
| UNet++ EfficientNet-B4 | `Khoa` | 256 | 0.8960 | 0.8115 | 0.8892 | 0.9028 | `weights/Khoa_best_Unetplusplus_checkpoint.pth` | Threshold search best at 0.65 |
| Ensemble average | `src/evaluation/development_eval.py` | 256 | 0.9026 | 0.8225 | 0.8904 | 0.9151 | 3 checkpoints above | Best raw ensemble at threshold 0.50 |
| Ensemble + post-process | `src/evaluation/development_eval.py` | 256 | 0.9027 | 0.8226 | 0.8903 | 0.9154 | 3 checkpoints above | Morphology + remove components `<64` px |
