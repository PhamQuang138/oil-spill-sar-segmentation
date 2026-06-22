"""
visualize_logs.py — Trực quan hóa toàn diện Dữ liệu (Dataset) và Kết quả (Logs)
Chạy: python visualize_logs.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from pathlib import Path
import config

def plot_training_curves():
    """1. Vẽ biểu đồ quá trình học (Loss, Dice, Accuracy, Precision, Recall)"""
    print("  [1/3] Đang phân tích file training_log.csv...")
    if not config.LOG_PATH.exists():
        print(f"  [LỖI] Không tìm thấy file log: {config.LOG_PATH}")
        return

    # Đọc dữ liệu từ CSV
    df = pd.read_csv(config.LOG_PATH)
    df.columns = df.columns.str.strip()

    if 'Epoch' not in df.columns:
        col_names = list(df.columns)
        col_names[0] = 'Epoch'
        df.columns = col_names

    cols = df.columns.tolist()

    # Tìm Epoch tốt nhất dựa trên Val_Dice
    best_epoch = df['Epoch'][df['Val_Dice'].idxmax()]
    best_dice = df['Val_Dice'].max()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # ── Biểu đồ 1: Loss (Train vs Val) ──
    ax1.plot(df['Epoch'], df['Train_Loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss', color='#d62728', linewidth=2)
    ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
    ax1.set_title('1. Mức độ hội tụ của Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (BCE + Dice)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # ── Biểu đồ 2: Dice & Accuracy ──
    ax2.plot(df['Epoch'], df['Val_Dice'], label='Val Dice (F1)', color='#2ca02c', linewidth=2)
    if 'Val_Acc' in cols:
        ax2.plot(df['Epoch'], df['Val_Acc'], label='Val Accuracy', color='#ff7f0e', linewidth=2, linestyle='-.')

    # Đánh dấu Epoch tốt nhất
    ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
    ax2.scatter(best_epoch, best_dice, color='red', s=100, zorder=5, label=f'Best Dice: {best_dice:.4f}')

    ax2.set_title(f'2. Độ chính xác (Best Epoch: {best_epoch})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score (0 -> 1)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # ── Biểu đồ 3: Precision & Recall ──
    has_pre_rec = False
    if 'Val_Pre' in cols and 'Val_Rec' in cols:
        ax3.plot(df['Epoch'], df['Val_Pre'], label='Val Precision', color='#9467bd', linewidth=2)
        ax3.plot(df['Epoch'], df['Val_Rec'], label='Val Recall', color='#8c564b', linewidth=2)
        has_pre_rec = True
    elif 'Val_Precision' in cols and 'Val_Recall' in cols:
        ax3.plot(df['Epoch'], df['Val_Precision'], label='Val Precision', color='#9467bd', linewidth=2)
        ax3.plot(df['Epoch'], df['Val_Recall'], label='Val Recall', color='#8c564b', linewidth=2)
        has_pre_rec = True

    ax3.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
    if not has_pre_rec:
        ax3.text(0.5, 0.5, 'Chưa có dữ liệu\nPrecision & Recall', ha='center', va='center', fontsize=12)

    ax3.set_title('3. Chi tiết: Precision vs Recall', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Score (0 -> 1)')
    ax3.grid(True, linestyle='--', alpha=0.6)
    if has_pre_rec:
        ax3.legend()

    save_path = config.PREDICTION_DIR / '1_training_learning_curves.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"  -> [OK] Đã lưu: {save_path}")


def visualize_dataset_samples(num_samples=3):
    """2. Trực quan hóa một vài mẫu ảnh đầu vào và Ground Truth"""
    print(f"  [2/3] Đang bốc ngẫu nhiên {num_samples} ảnh từ tập Train để trực quan hóa...")
    img_paths = list(config.TRAIN_IMG_DIR.glob("*.png")) + list(config.TRAIN_IMG_DIR.glob("*.jpg"))
    if not img_paths:
        print("  [LỖI] Không tìm thấy ảnh trong thư mục train!")
        return

    samples = random.sample(img_paths, min(num_samples, len(img_paths)))
    fig, axes = plt.subplots(num_samples, 3, figsize=(14, 4 * num_samples))
    if num_samples == 1: axes = [axes]

    for i, img_path in enumerate(samples):
        mask_path = config.TRAIN_MASK_DIR / img_path.name
        if not mask_path.exists(): continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Tạo ảnh Overlay
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlay = rgb_img.copy()
        overlay[mask > 127] = [255, 50, 50]  # Đổ màu đỏ vào vùng dầu
        blended = cv2.addWeighted(rgb_img, 0.7, overlay, 0.3, 0)

        # Plot
        axes[i][0].imshow(img, cmap='gray')
        axes[i][0].set_title(f"1. SAR Image: {img_path.name[:15]}...", fontsize=12)
        axes[i][0].axis('off')

        axes[i][1].imshow(mask, cmap='jet')
        axes[i][1].set_title("2. Ground Truth Mask", fontsize=12)
        axes[i][1].axis('off')

        axes[i][2].imshow(blended)
        axes[i][2].set_title("3. Overlay Visualization", fontsize=12)
        axes[i][2].axis('off')

    save_path = config.PREDICTION_DIR / '2_dataset_samples.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"  -> [OK] Đã lưu: {save_path}")


def analyze_class_imbalance(sample_size=100):
    """3. Vẽ biểu đồ tròn phân tích sự mất cân bằng dữ liệu (Class Imbalance)"""
    print(f"  [3/3] Đang phân tích tỷ lệ pixel trên {sample_size} mask ngẫu nhiên...")
    mask_paths = list(config.TRAIN_MASK_DIR.glob("*.png")) + list(config.TRAIN_MASK_DIR.glob("*.jpg"))
    if not mask_paths: return

    samples = random.sample(mask_paths, min(sample_size, len(mask_paths)))
    total_pixels = 0
    oil_pixels = 0

    for mask_path in samples:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            total_pixels += mask.size
            oil_pixels += np.sum(mask > 127)

    sea_pixels = total_pixels - oil_pixels

    # Vẽ biểu đồ Pie Chart
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = ['Biển (Background/Sea)', 'Vết dầu (Oil Spill)']
    sizes = [sea_pixels, oil_pixels]
    colors = ['#1f77b4', '#ff7f0e']
    explode = (0, 0.2)  # Cắt miếng bánh "Vết dầu" ra cho nổi bật

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
           shadow=True, startangle=100, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('Phân bố Dữ liệu Pixel (Class Imbalance Analysis)', fontsize=11, fontweight='bold')

    save_path = config.PREDICTION_DIR / 'piechart.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"  -> [OK] Đã lưu: {save_path}")


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print("  BỘ CÔNG CỤ TRỰC QUAN HÓA DỮ LIỆU & KẾT QUẢ")
    print(f"{'='*50}")

    # Đảm bảo thư mục lưu ảnh tồn tại
    config.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    plot_training_curves()
    visualize_dataset_samples(num_samples=3)
    analyze_class_imbalance(sample_size=100)

    print(f"\n[HOÀN TẤT] Toàn bộ biểu đồ đã được lưu tại thư mục: {config.PREDICTION_DIR.name}/")
    plt.show()  # Hiển thị tất cả cửa sổ hình ảnh cùng lúc