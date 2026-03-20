import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Thư mục chứa kết quả đảo ngược
TRIGGER_DIR = "trigger_inv"


def get_latest_trigger_images(model_dir):
    """Tìm bộ ảnh (mask, patch, fus) ở Epoch cuối cùng của một mô hình"""
    masks = glob.glob(os.path.join(model_dir, "mask_*.png"))
    if not masks:
        return None

    # Hàm con để lấy số Epoch từ tên file (ví dụ: mask_e96_reg... -> 96)
    def extract_epoch(filepath):
        filename = os.path.basename(filepath)
        try:
            return int(filename.split("_")[1][1:])  # Cắt chữ 'e' và lấy số
        except:
            return -1

    # Tìm file mask có Epoch lớn nhất (chạy xong xuôi nhất)
    best_mask = max(masks, key=extract_epoch)

    # Lấy chuỗi định danh (ví dụ: 'e96_reg9152.76')
    identifier = best_mask.split("mask_")[1]

    # Tìm các file patch và fus tương ứng
    best_patch = os.path.join(model_dir, f"patch_{identifier}")
    best_fus = os.path.join(model_dir, f"fus_{identifier}")

    if os.path.exists(best_patch) and os.path.exists(best_fus):
        return best_mask, best_patch, best_fus
    return None


def plot_trigger_gallery():
    """Vẽ bộ sưu tập ảnh Trigger cho GTSRB và STL-10"""
    # Tìm ngẫu nhiên 1 thư mục GTSRB và 1 thư mục STL-10
    gtsrb_dir = None
    stl10_dir = None

    for folder in os.listdir(TRIGGER_DIR):
        if "gtsrb" in folder.lower() and os.path.isdir(
            os.path.join(TRIGGER_DIR, folder)
        ):
            gtsrb_dir = os.path.join(TRIGGER_DIR, folder)
        elif "stl10" in folder.lower() and os.path.isdir(
            os.path.join(TRIGGER_DIR, folder)
        ):
            stl10_dir = os.path.join(TRIGGER_DIR, folder)

    if not gtsrb_dir or not stl10_dir:
        print("Lỗi: Không tìm thấy đủ thư mục GTSRB và STL-10 trong trigger_inv.")
        return

    # Lấy ảnh
    gtsrb_imgs = get_latest_trigger_images(gtsrb_dir)
    stl10_imgs = get_latest_trigger_images(stl10_dir)

    if not gtsrb_imgs or not stl10_imgs:
        print("Lỗi: Không tìm thấy ảnh png bên trong các thư mục.")
        return

    # --- TIẾN HÀNH VẼ LÊN LƯỚI (2 HÀNG x 3 CỘT) ---
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    fig.suptitle(
        "Visualized Inverted Triggers by DECREE", fontsize=16, fontweight="bold", y=0.98
    )

    datasets = [("GTSRB", gtsrb_imgs), ("STL-10", stl10_imgs)]
    col_titles = [
        "Mask (Vị trí đốm sáng)",
        "Patch (Màu sắc/Kết cấu)",
        "Fused (Áp dụng lên ảnh thật)",
    ]

    for row, (dataset_name, imgs) in enumerate(datasets):
        for col, img_path in enumerate(imgs):
            ax = axes[row, col]
            img = mpimg.imread(img_path)
            ax.imshow(img)

            # Tắt trục tọa độ cho ảnh đẹp hơn
            ax.set_xticks([])
            ax.set_yticks([])

            # Cài đặt tiêu đề cột (chỉ ở hàng trên cùng)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, pad=10)

            # Cài đặt tên Dataset (chỉ ở cột ngoài cùng bên trái)
            if col == 0:
                ax.set_ylabel(dataset_name, fontsize=14, fontweight="bold", labelpad=15)

    plt.tight_layout()
    # Chừa chút không gian cho tiêu đề tổng
    plt.subplots_adjust(top=0.9)
    plt.savefig("Figure_Triggers_Visualization.jpg", dpi=300, bbox_inches="tight")
    print("-> Đã xuất ảnh cực nét: Figure_Triggers_Visualization.jpg")
    plt.close()


if __name__ == "__main__":
    print("Đang quét thư mục trigger_inv...")
    plot_trigger_gallery()
